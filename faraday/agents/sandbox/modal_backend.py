"""
Centralized sandbox management with improved efficiency, clarity, and error handling.
"""

import modal
import time
import threading
from typing import Optional, Dict, Any, Callable
import os

from faraday.agents.sandbox.config import SandboxConfig, SandboxState, tprint
from faraday.agents.sandbox.packages import EXTENDED_SANDBOX

class SandboxImageCache:
    """Global cache for sandbox images to avoid rebuilding"""
    _instance = None
    _images: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_image(self, config: SandboxConfig) -> Optional[Any]:
        """Get cached image or None if not available"""
        cache_key = self._get_cache_key(config)
        with self._lock:
            return self._images.get(cache_key)
    
    def set_image(self, config: SandboxConfig, image: Any) -> None:
        """Cache the built image"""
        cache_key = self._get_cache_key(config)
        with self._lock:
            self._images[cache_key] = image
    
    def _get_cache_key(self, config: SandboxConfig) -> str:
        """Generate cache key for image configuration"""
        return f"debian-{config.python_version}"
def build_optimized_sandbox_image(config: SandboxConfig) -> Any:
    """
    Build sandbox image with caching and optimization.
    Returns cached image if available, otherwise builds new one.
    """
    cache = SandboxImageCache()
    
    # Try to get cached image first
    cached_image = cache.get_image(config)
    if cached_image is not None:
        tprint("Using cached sandbox image", config.verbose)
        return cached_image
    
    tprint("Building new sandbox image...", config.verbose)
    start_time = time.time()
    
    # Build the image with enabled output (required for image building outside modal run)
    with modal.enable_output():
        base_image = modal.Image.debian_slim(python_version=config.python_version)        
        faraday_image = (
            base_image
            .apt_install([
                "git",
                "unzip",
                "wget",
                "libxrender1",
                "pandoc",
                "latexmk",
                "texlive-latex-base",
                "texlive-latex-recommended",
                "texlive-fonts-recommended",
                # Build deps for plip/openbabel wheels on Debian slim
                "build-essential",
                "cmake",
                "swig",
                "pkg-config",
                "libopenbabel-dev",
                "libxml2-dev",
                "zlib1g-dev",
            ])
            .pip_install(EXTENDED_SANDBOX)
            # Install plip without pulling openbabel from PyPI
            .run_commands("pip install --no-cache-dir --no-deps plip")
        )
    
    # Cache the built image
    cache.set_image(config, faraday_image)
    
    build_time = time.time() - start_time
    tprint(f"Sandbox image built in {build_time:.2f} seconds", config.verbose)
    
    return faraday_image


class SandboxManager:
    """
    Centralized sandbox management with clear state tracking and efficient initialization.
    """
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.state = SandboxState.UNINITIALIZED
        self.sandbox_id: Optional[str] = None
        self.error_message: Optional[str] = None
        self.initialization_time: Optional[float] = None
        
        # Threading support for background initialization
        self._init_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        
        # Progress callback for user feedback
        self.progress_callback: Optional[Callable[[str], None]] = None
        
        # Cached filesystem/mount state
        self._cloud_storage_available: Optional[bool] = None
        self._directories_initialized: bool = False
        
        # Start initialization if not lazy
        if self.config.enable_sandbox and not self.config.lazy_initialization:
            self.start_initialization()
    
    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for progress updates during initialization"""
        self.progress_callback = callback
    
    def _update_progress(self, message: str) -> None:
        """Update progress via callback or print"""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            tprint(message, self.config.verbose)
    
    def start_initialization(self) -> None:
        """Start sandbox initialization in background thread"""
        with self._state_lock:
            if self.state == SandboxState.UNINITIALIZED:
                self.state = SandboxState.INITIALIZING
                self._init_thread = threading.Thread(
                    target=self._initialize_sandbox_safely,
                    daemon=True
                )
                self._init_thread.start()
                self._update_progress("Starting sandbox initialization...")
    
    def _initialize_sandbox_safely(self) -> None:
        """Safe wrapper for sandbox initialization with error handling"""
        try:
            self._initialize_sandbox()
            with self._state_lock:
                self.state = SandboxState.READY
                self._update_progress(f"Sandbox ready (ID: {self.sandbox_id})")
        except Exception as e:
            with self._state_lock:
                self.state = SandboxState.ERROR
                self.error_message = str(e)
                self._update_progress(f"Sandbox initialization failed: {e}")
    
    def _initialize_sandbox(self) -> None:
        """Core sandbox initialization logic"""
        start_time = time.time()
        
        # Get or create Modal app
        self._update_progress("Setting up Modal app...")
        app = modal.App.lookup(self.config.app_name, create_if_missing=True)
        
        # Build or get cached image
        self._update_progress("Preparing sandbox image...")
        sandbox_image = build_optimized_sandbox_image(self.config)
        
        # Create sandbox with appropriate configuration
        self._update_progress("Creating sandbox instance...")
        
        try:
            sandbox = self._create_sandbox(app, sandbox_image)
            self.sandbox_id = sandbox.object_id
            self.initialization_time = time.time() - start_time
        except Exception:
            raise

    def _create_sandbox(self, app: Any, image: Any) -> Any:
        """Create a sandbox, optionally with cloud storage mounts."""
        if not self.config.needs_bucket:
            return self._create_sandbox_basic(app, image)
        try:
            return self._create_sandbox_with_bucket(app, image)
        except Exception as exc:
            if self.config.cloud_storage_mode == "required":
                raise RuntimeError(
                    f"Cloud storage is required but mount setup failed: {exc}"
                ) from exc
            self._update_progress(
                f"Optional cloud storage mount unavailable, using basic sandbox: {exc}"
            )
            return self._create_sandbox_basic(app, image)

    def _build_aws_secret(self) -> Any:
        access_key = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
        secret_key = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
        session_token = (os.getenv("AWS_SESSION_TOKEN") or "").strip()
        if not access_key or not secret_key:
            raise RuntimeError("missing AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
        secret_payload = {
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
        }
        if session_token:
            secret_payload["AWS_SESSION_TOKEN"] = session_token
        return modal.Secret.from_dict(secret_payload)
    
    def _create_sandbox_with_bucket(self, app: Any, image: Any) -> Any:
        """Create sandbox with S3 bucket mounting.
        
        Mount structure:
        - /cloud-storage/        → Current chat's files ({user_id}/{chat_id}/)
        """
        self._update_progress(f"Creating sandbox with bucket for user {self.config.user_id}")
        
        if not self.config.bucket_name:
            raise RuntimeError("cloud bucket mounts require bucket_name")

        # Get AWS credentials secret
        aws_secret = self._build_aws_secret()
        
        # Build volumes dictionary with chat storage
        volumes = {
            '/cloud-storage': modal.CloudBucketMount(
                bucket_name=self.config.bucket_name,
                key_prefix=self.config.bucket_path,
                secret=aws_secret
            )
        }
        
        with modal.enable_output():
            return modal.Sandbox.create(
                image=image,
                app=app,
                volumes=volumes
            )
    
    def _create_sandbox_basic(self, app: Any, image: Any) -> Any:
        """Create basic sandbox without bucket mounting"""
        self._update_progress("Creating basic sandbox")
        with modal.enable_output():
            return modal.Sandbox.create(image=image, app=app)
    
    def wait_for_ready(self, timeout: float = 120.0) -> bool:
        """
        Wait for sandbox to be ready.
        Returns True if ready, False if timeout or error.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._state_lock:
                if self.state == SandboxState.READY:
                    return True
                elif self.state == SandboxState.ERROR:
                    return False
            
            time.sleep(0.5)
        
        return False
    
    def get_sandbox(self) -> Optional[Any]:
        """
        Get the sandbox instance if ready.
        For lazy initialization, starts initialization if needed.
        """
        # Start lazy initialization if needed
        if (self.state == SandboxState.UNINITIALIZED and 
            self.config.lazy_initialization and 
            self.config.enable_sandbox):
            self.start_initialization()
        
        # Wait for sandbox to be ready
        if self.state == SandboxState.INITIALIZING:
            self._update_progress("Waiting for sandbox to be ready...")
            if not self.wait_for_ready():
                return None
        
        # Return sandbox if ready
        if self.state == SandboxState.READY and self.sandbox_id:
            try:
                return modal.Sandbox.from_id(self.sandbox_id)
            except Exception as e:
                self._update_progress(f"Failed to reconnect to sandbox: {e}")
                # Try to reinitialize
                self.state = SandboxState.UNINITIALIZED
                self.sandbox_id = None
                return None
        
        return None

    def _detect_cloud_storage(self, sandbox: Any) -> bool:
        """Detect if /cloud-storage mount is available; cache the result."""
        if self._cloud_storage_available is not None:
            return self._cloud_storage_available
        try:
            proc = sandbox.exec("test", "-d", "/cloud-storage")
            proc.wait()
            self._cloud_storage_available = (proc.returncode == 0)
        except Exception:
            self._cloud_storage_available = False
        return self._cloud_storage_available

    def ensure_directories(self, sandbox: Any, directories) -> bool:
        """
        Ensure required directories exist inside the sandbox exactly once.
        Returns True if cloud storage mount is available and used; False if using fallback.
        """
        cloud_available = self._detect_cloud_storage(sandbox)
        
        if not self._directories_initialized:
            try:
                # Compute target directories with appropriate fallbacks
                targets = []
                for d in directories:
                    if "/cloud-storage/" in d:
                        if cloud_available:
                            targets.append(d)
                        else:
                            targets.append(d.replace("/cloud-storage/", "/tmp/"))
                    else:
                        targets.append(d)
                
                # Batch create with a single shell invocation
                if targets:
                    mkdir_cmd = "mkdir -p " + " ".join(f'"{d}"' for d in targets)
                    p = sandbox.exec("bash", "-lc", mkdir_cmd)
                    p.wait()
                self._directories_initialized = True
            except Exception as e:
                # Do not retry per run; mark initialized to avoid repeated slow attempts
                self._directories_initialized = True
                self._update_progress(f"Directory initialization encountered an error: {e}")
        return cloud_available
    
    def is_healthy(self) -> bool:
        """Check if sandbox is healthy and responsive"""
        if self.state != SandboxState.READY or not self.sandbox_id:
            return False
        
        try:
            sandbox = modal.Sandbox.from_id(self.sandbox_id)
            # Quick health check
            if hasattr(sandbox, 'poll'):
                return sandbox.poll() is None
            return True
        except Exception:
            return False
    
    def terminate(self) -> None:
        """Terminate the sandbox and clean up resources"""
        if self.sandbox_id:
            try:
                sandbox = modal.Sandbox.from_id(self.sandbox_id)
                sandbox.terminate()
                self._update_progress(f"Sandbox {self.sandbox_id} terminated")
            except Exception as e:
                self._update_progress(f"Error terminating sandbox: {e}")
            finally:
                with self._state_lock:
                    self.state = SandboxState.TERMINATED
                    self.sandbox_id = None
    
    def reinitialize(self) -> None:
        """Reinitialize the sandbox with current config."""
        # Wait for any in-progress initialization to complete before terminating
        if self.state == SandboxState.INITIALIZING:
            self._update_progress("Waiting for current initialization to complete before reinitializing...")
            self.wait_for_ready(timeout=120)
        
        # Terminate existing sandbox
        if self.state in (SandboxState.READY, SandboxState.ERROR, SandboxState.TERMINATED):
            if self.sandbox_id:
                self.terminate()
        
        # Reset state and cached values
        with self._state_lock:
            self.state = SandboxState.UNINITIALIZED
            self.sandbox_id = None
            self.error_message = None
            self.initialization_time = None
            self._cloud_storage_available = None
            self._directories_initialized = False
        
        # Start new initialization
        self._update_progress("Reinitializing sandbox with updated config...")
        self.start_initialization()
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive status information"""
        return {
            "state": self.state.value,
            "sandbox_id": self.sandbox_id,
            "error_message": self.error_message,
            "initialization_time": self.initialization_time,
            "config": {
                "user_id": self.config.user_id,
                "chat_id": self.config.chat_id,
                "lazy_initialization": self.config.lazy_initialization,
                "needs_bucket": self.config.needs_bucket,
                "cloud_storage_mode": self.config.cloud_storage_mode,
            }
        }
