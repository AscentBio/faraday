from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


_DEFAULT_CONTAINER_CONFIG_PATHS = (
    Path("/app/config/faraday.yaml"),
    Path("/app/config/faraday.yml"),
    Path("/etc/faraday/config.yaml"),
    Path("/etc/faraday/config.yml"),
)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        normalized = str(path.expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(Path(normalized))
    return unique


def _paths_from_config_dir(config_dir: str) -> list[Path]:
    base = Path(config_dir).expanduser()
    return [
        base / "faraday.yaml",
        base / "faraday.yml",
        base / "config.yaml",
        base / "config.yml",
    ]


def _candidate_config_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.getenv("FARADAY_CONFIG")
    if env_path:
        paths.append(Path(env_path).expanduser())

    paths.extend(_DEFAULT_CONTAINER_CONFIG_PATHS)

    cwd = Path.cwd()
    paths.extend(
        [
            cwd / "faraday.yaml",
            cwd / "faraday.yml",
            Path.home() / ".faraday" / "config.yaml",
            Path.home() / ".faraday" / "config.yml",
        ]
    )
    return _dedupe_paths(paths)


def _read_config_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if not raw.strip():
        return {}
    try:
        parsed = yaml.safe_load(raw)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def get_runtime_config() -> dict[str, Any]:
    config_path = get_runtime_config_path()
    if config_path is not None:
        return _read_config_file(config_path)
    return {}


def get_runtime_config_path() -> Optional[Path]:
    for candidate in _candidate_config_paths():
        if candidate.exists():
            return candidate
    return None


def get_config_value(*path: str, default: Any = None) -> Any:
    cfg = get_runtime_config()
    cursor: Any = cfg
    for key in path:
        if not isinstance(cursor, dict):
            return default
        if key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _first_set(*key_paths: tuple[str, ...], default: Any = None) -> Any:
    """Return the value from the first key path that is explicitly set in the config.

    Used to support the standardized keys (``app.*``, ``sandbox.*``) while
    preserving backward compatibility with legacy keys (``runtime.*``,
    ``execution.*``).
    """
    for path in key_paths:
        val = get_config_value(*path, default=None)
        if val is not None:
            return val
    return default


def get_backend_value(name: str, default: str) -> str:
    value = get_config_value("backends", name, default=None)
    if value is None:
        value = get_config_value("backend", name, default=default)
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    return normalized or default


def get_string_value(*path: str, default: Optional[str] = None) -> Optional[str]:
    value = get_config_value(*path, default=default)
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    normalized = value.strip()
    return normalized or default


def get_path_value(*path: str, default: Optional[str] = None) -> Optional[str]:
    value = get_string_value(*path, default=default)
    if value is None:
        return default
    # Path("") expands to '.'; treat unset/empty defaults as absent so callers
    # do not silently use the process working directory.
    if not value:
        return default
    return str(Path(value).expanduser())


def get_bool_value(*path: str, default: Optional[bool] = None) -> Optional[bool]:
    value = get_config_value(*path, default=default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def normalize_runtime_app(raw: Optional[str]) -> str:
    normalized = (raw or "").strip().lower().replace("_", "-")
    if normalized in {"docker", "container"}:
        return "docker"
    return "host"


def normalize_runtime_backend(raw: Optional[str]) -> str:
    """Normalize ``runtime.backend``: host (Faraday on host) vs docker (app in Docker)."""
    normalized = (raw or "").strip().lower().replace("_", "-")
    if normalized in {"docker", "container"}:
        return "docker"
    return "host"


def _runtime_backend_key_set() -> bool:
    raw = get_config_value("runtime", "backend", default=None)
    return raw is not None and str(raw).strip() != ""


def _resolve_runtime_launch_and_app(
    default_launch: str = "host",
    default_app: str = "host",
) -> tuple[str, str]:
    """Resolve effective launch + app mode.

    Resolution order for app mode:
    1. ``app.mode`` (standardized)
    2. ``runtime.backend`` (sets both launch + app)
    3. ``runtime.app`` / ``runtime.launch`` (legacy)
    """
    # Standardized key takes priority.
    app_mode_raw = get_string_value("app", "mode", default=None)
    if app_mode_raw is not None and str(app_mode_raw).strip():
        app_eff = normalize_runtime_app(app_mode_raw)
        launch_eff = app_eff
        return launch_eff, app_eff

    if _runtime_backend_key_set():
        base = normalize_runtime_backend(get_string_value("runtime", "backend", default="host"))
        if base == "docker":
            launch_eff, app_eff = "docker", "docker"
        else:
            launch_eff, app_eff = "host", "host"
        override_l = get_string_value("runtime", "override_launch", default=None)
        if override_l is not None and str(override_l).strip():
            launch_eff = normalize_runtime_launch(override_l)
        override_a = get_string_value("runtime", "override_app", default=None)
        if override_a is not None and str(override_a).strip():
            app_eff = normalize_runtime_app(override_a)
        return launch_eff, app_eff

    launch_raw = get_string_value("runtime", "launch", default=None)
    app_raw = get_string_value("runtime", "app", default=None)
    launch_eff = normalize_runtime_launch(launch_raw if launch_raw is not None else default_launch)
    app_eff = normalize_runtime_app(app_raw if app_raw is not None else default_app)
    return launch_eff, app_eff


def get_runtime_app(default: str = "host") -> str:
    return _resolve_runtime_launch_and_app(default_launch="host", default_app=default)[1]


def normalize_runtime_launch(raw: Optional[str]) -> str:
    normalized = (raw or "").strip().lower().replace("_", "-")
    if normalized in {"docker", "container"}:
        return "docker"
    return "host"


def get_runtime_launch(default: str = "host") -> str:
    return _resolve_runtime_launch_and_app(default_launch=default, default_app="host")[0]


def get_runtime_app_docker_image(default: str = "faraday-oss") -> str:
    """Docker image for the Faraday app when using ``faraday --use-docker``.

    Reads ``app.app_image`` (standardized) or ``runtime.app_image`` (legacy).
    This is distinct from ``sandbox.docker_image``, which selects the
    code-execution sandbox image.
    """
    value = _first_set(("app", "app_image"), ("runtime", "app_image"))
    if value is None:
        return default
    s = str(value).strip()
    return s or default


def get_run_output_dir(default: str = "./run_outputs") -> str:
    """Return the configured run output root directory.

    Reads ``outputs.root`` from the config YAML. Falls back to
    ``./run_outputs`` relative to the current working directory.
    """
    value = get_path_value("outputs", "root", default=None)
    if value:
        return value
    # In app-container mode, default outputs should land on the bind-mounted
    # host workspace so run artifacts are visible on the local machine.
    if os.getenv("FARADAY_RUNNING_IN_APP_DOCKER", "").strip().lower() in {"1", "true", "yes", "on"}:
        mount_path = get_workspace_mount_path(default="/workspace")
        mount_path = (mount_path or "/workspace").rstrip("/") or "/"
        return f"{mount_path}/run_outputs" if mount_path != "/" else "/run_outputs"
    return str((Path.cwd() / default).resolve())


def get_workspace_source_root(default: Optional[str] = None) -> Optional[str]:
    val = _first_set(
        ("app", "workspace", "source_root"),
        ("runtime", "workspace", "source_root"),
    )
    if val is not None:
        s = str(val).strip()
        if s:
            return str(Path(s).expanduser())
    return default


def get_workspace_mount_path(default: str = "/workspace") -> str:
    value = _first_set(
        ("sandbox", "workspace", "container_path"),
        ("execution", "workspace", "mount_path"),
    )
    if value is None:
        value = default
    else:
        value = str(value).strip() or default
    normalized = (value or default).strip() or default
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    normalized = normalized.rstrip("/") or "/"
    return normalized


def normalize_execution_backend(raw: str) -> str:
    """Map config/CLI ``execution.backend`` to a concrete runtime backend."""
    normalized = (raw or "").strip().lower().replace("_", "-")
    if not normalized:
        return "docker"
    aliases = {
        # "local" means run on this machine (same as host). Use `docker` for the container sandbox.
        "local": "host",
        "harbor": "docker",
        "none": "disabled",
        "off": "disabled",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"docker", "modal", "host", "disabled"}:
        return "docker"
    return normalized


def get_execution_backend(default: str = "docker") -> str:
    raw_value = _first_set(
        ("sandbox", "backend"),
        ("execution", "backend"),
        default=default,
    )
    if raw_value is None:
        return default
    return normalize_execution_backend(str(raw_value))


def get_runtime_dependency_packages(default_backend: str = "docker") -> list[str]:
    """Return runtime pip dependencies required by current config.

    Dependencies are selected from config + environment so lean Harbor/local runs
    can avoid optional heavy integrations.
    """

    core = [
        "openai",
        "openai-agents",
        "pydantic",
        "pyyaml",
        "tenacity",
    ]

    backend = get_execution_backend(default=default_backend)

    # Feature flags (defaults intentionally conservative for lean runtime setups)
    enable_modal = get_bool_value("features", "enable_modal", default=(backend == "modal"))
    enable_exa = get_bool_value("features", "enable_exa", default=False)
    # Keep common Python execution dependencies available in local/Harbor runs.
    enable_python_science_stack = get_bool_value(
        "features",
        "enable_python_science_stack",
        default=(backend in {"docker", "host"}),
    )
    # Cheminformatics stack is heavy and sometimes platform-constrained; opt in.
    enable_cheminformatics_stack = get_bool_value(
        "features",
        "enable_cheminformatics_stack",
        default=False,
    )
    optional: list[str] = []
    if enable_modal:
        optional.append("modal")
    if enable_exa:
        optional.append("exa-py")
    if enable_python_science_stack:
        optional.extend(["requests", "matplotlib", "seaborn", "mygene"])
    if enable_cheminformatics_stack:
        optional.extend(["rdkit", "datamol"])
    # Preserve order while deduplicating.
    seen: set[str] = set()
    combined: list[str] = []
    for package in [*core, *optional]:
        normalized = package.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        combined.append(normalized)
    return combined
