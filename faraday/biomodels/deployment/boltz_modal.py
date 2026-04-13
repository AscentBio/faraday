

import modal
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

TOOL_NAME = "Boltz-2"
RESULTS_BUCKET = "faraday-app-user-data"
MIN_PROTEIN_LENGTH = 10
MAX_PROTEIN_LENGTH = 2000
MAX_LIGANDS_PER_REQUEST = 50
PER_LIGAND_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 7200
MAX_SMILES_LENGTH = 500
MAX_MOLECULAR_WEIGHT = 2000
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

ERROR_CATEGORY_KEYWORDS = {
    "boltz_computation": ["boltz", "prediction", "inference", "cofold"],
    "network": ["network", "connection", "timeout", "dns"],
    "gpu_resources": ["memory", "cuda", "gpu", "out of memory", "device"],
    "ligand_validation": ["smiles", "molecule", "rdkit", "invalid"],
    "protein_validation": ["protein", "sequence", "fasta", "amino"],
    "cloud_storage": ["cloud", "s3", "storage", "bucket", "aws"],
    "configuration": ["yaml", "config", "file not found"],
    "file_system": ["volume", "mount", "directory", "path"],
    "database": ["database", "job"],
}

ERROR_GUIDANCE = {
    "boltz_computation": "This appears to be a Boltz model computation error. Check input formats and try with simpler molecules.",
    "network": "Network connectivity issue. Check internet connection and try again.",
    "gpu_resources": "GPU resource issue. The system may be overloaded. Try reducing batch size or retry later.",
    "ligand_validation": "Invalid ligand structure. Check SMILES format and ensure molecules are valid.",
    "protein_validation": "Invalid protein sequence. Check FASTA format and amino acid codes.",
    "cloud_storage": "Cloud storage error. Results may not be saved properly. Contact support if issue persists.",
    "configuration": "Configuration file error. This is likely a system issue. Contact support.",
    "file_system": "File system error. This is likely a system issue. Contact support.",
    "database": "Database connection error. Job status may not be updated properly.",
    "unknown": "Unexpected error. Please contact support with the experiment ID.",
}

# Constants
PYPACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "boto3",
    "rdkit",
    "pyyaml",
    "torch",

    "pydantic>=2.0.0",
]


# Docker image setup
base_image = modal.Image.debian_slim(python_version="3.10")


def tprint(msg: str, **kwargs):
    """Enhanced logging with structured data"""
    timestamp = datetime.now().astimezone(timezone(timedelta(hours=-8))).strftime("%Y-%m-%d %H:%M:%S")
    if kwargs:
        context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f'[{timestamp}] {msg} | {context}')
    else:
        print(f'[{timestamp}] {msg}')

def categorize_error(error_str: str) -> str:
    """Categorize errors for better debugging and user feedback"""
    error_str = error_str.lower()

    for error_type, keywords in ERROR_CATEGORY_KEYWORDS.items():
        if any(keyword in error_str for keyword in keywords):
            return error_type
    return "unknown"

def calculate_dynamic_timeout(num_ligands: int, base_timeout: int = 1800) -> int:
    """Calculate dynamic timeout based on number of ligands"""
    calculated_timeout = base_timeout + (num_ligands * PER_LIGAND_TIMEOUT_SECONDS)
    return min(calculated_timeout, MAX_TIMEOUT_SECONDS)

def extract_protein_sequence(sequence: str) -> str:
    """Normalize raw protein input to a contiguous uppercase amino acid sequence."""
    if sequence.startswith(">"):
        lines = sequence.strip().splitlines()
        if len(lines) < 2:
            raise ValueError("Invalid FASTA format: missing sequence after header")
        sequence = "".join(lines[1:])
    return "".join(sequence.split()).upper()

def validate_protein_sequence(sequence: str) -> tuple[bool, str, str]:
    """Enhanced protein sequence validation"""
    if not sequence or not sequence.strip():
        return False, "Protein sequence cannot be empty", ""

    try:
        sequence = extract_protein_sequence(sequence)
    except ValueError as exc:
        return False, str(exc), ""

    if len(sequence) < MIN_PROTEIN_LENGTH:
        return False, f"Protein sequence too short: {len(sequence)} residues (minimum: {MIN_PROTEIN_LENGTH})", ""

    if len(sequence) > MAX_PROTEIN_LENGTH:
        return False, f"Protein sequence too long: {len(sequence)} residues (maximum: {MAX_PROTEIN_LENGTH})", ""

    invalid_chars = set(sequence) - STANDARD_AMINO_ACIDS
    if invalid_chars:
        return False, f"Invalid amino acids found: {sorted(invalid_chars)}. Only standard 20 amino acids allowed.", ""

    return True, "Valid protein sequence", sequence

def create_error_context(error: Exception, experiment_id: str = "", ligand_index: int = -1) -> dict:
    """Create detailed error context for debugging"""
    error_type = categorize_error(str(error))
    
    context = {
        "error_message": str(error),
        "error_type": error_type,
        "timestamp": datetime.now().isoformat(),
    }
    
    if experiment_id:
        context["experiment_id"] = experiment_id
    
    if ligand_index >= 0:
        context["ligand_index"] = ligand_index
    
    context["guidance"] = ERROR_GUIDANCE.get(error_type, ERROR_GUIDANCE["unknown"])
    return context

def create_experiment_status_response(experiment_id: str, status: str, message: str, **metadata) -> dict:
    """Create a consistent API payload for experiment creation responses."""
    return {
        "experiment_id": experiment_id,
        "job_id": experiment_id,
        "metadata": {
            "status": status,
            "message": message,
            **metadata,
        },
    }

def create_cofold_error_response(
    error_msg: str,
    error_context: dict[str, Any],
    input_ligand: str,
    ligand_index: int,
) -> dict:
    """Create a consistent error payload for single-ligand cofold failures."""
    return {
        "results": [],
        "error": error_msg,
        "error_context": error_context,
        "ligand_smiles": input_ligand,
        "ligand_index": ligand_index,
        "affinity_data": None,
        "cloud_storage": None,
        "metadata": {
            "status": "error",
            "message": error_context["guidance"],
            "error_type": error_context["error_type"],
            "cloud_saved": False,
        },
    }

def build_boltz_config(protein_sequence: str, ligand_smiles: str) -> dict[str, Any]:
    """Build the Boltz YAML payload for a protein-ligand affinity prediction."""
    return {
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein_sequence,
                }
            },
            {
                "ligand": {
                    "id": "B",
                    "smiles": ligand_smiles,
                }
            },
        ],
        "properties": [
            {
                "affinity": {
                    "binder": "B",
                }
            }
        ],
    }

def build_experiment_storage_prefix(
    user_id: str,
    chat_id: str,
    experiment_id: str,
    subdir: str = "",
) -> str:
    """Create a normalized storage prefix for experiment artifacts."""
    path_parts = [user_id, chat_id, "tool_outputs", "boltz2", experiment_id]
    if subdir:
        path_parts.append(subdir.strip("/"))
    return "/".join(path_parts) + "/"

def build_storage_location(
    user_id: str,
    chat_id: str,
    experiment_id: str,
    subdir: str = "",
) -> dict[str, str]:
    """Return S3 and agent-accessible paths for experiment artifacts."""
    base_path = build_experiment_storage_prefix(user_id, chat_id, experiment_id, subdir)
    s3_path = f"s3://{RESULTS_BUCKET}/{base_path}"
    return {
        "bucket": RESULTS_BUCKET,
        "base_path": base_path,
        "s3_path": s3_path,
        "agent_path": convert_s3_to_agent_path(s3_path),
    }

def create_s3_client():
    """Create an authenticated S3 client from Modal secrets."""
    import boto3
    import os

    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

boltz_image = (
    base_image
    .apt_install([  
        "git", 
        "unzip", 
        "wget",
        "libxrender1",
        "libxext6",
        "libglib2.0-0"
    ])
    .pip_install(*PYPACKAGES)
    .run_commands(
        "git clone https://github.com/jwohlwend/boltz.git",
        "cd boltz && pip install -e .",
    )
)

web_image = modal.Image.debian_slim(python_version="3.10")
web_image = web_image.pip_install("boto3")

volume = modal.Volume.from_name("boltz-tmp-volume", create_if_missing=True)

# Setup Modal app
app = modal.App("boltz-modal-v2", 
                image=boltz_image,
                secrets=[modal.Secret.from_dotenv(filename='.env.prod')],
                )

# Data models
class InputQuery(BaseModel):
    """Input query model for protein ligand co-folding"""
    input_protein: str = Field(..., description="The protein sequence in FASTA format")
    list_of_ligands: list[str] = Field(..., description="The ligand in SMILES format")
    chat_id: str = Field(..., description="The chat ID to use for generation")
    user_id: str = Field(..., description="The user ID to use for generation")


class ToolStatusEntry(BaseModel):
    id: int
    created_at: str
    updated_at: str
    user_id: str
    chat_id: str
    tool_name: str
    status: str  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_results: Optional[Dict[str, Any]] = None
    tool_run_id: Optional[str] = None
    error_message: Optional[str] = None



@app.cls(volumes={"/boltz_tmp": volume})
class BoltzJobManager:


    @modal.enter()
    def setup(self):
        import os
        from faraday.storage import get_storage_backend
        self._storage = get_storage_backend()
        
    @modal.method()
    def create_job(self, tool_run_id: str, user_id: str, chat_id: str,
                  tool_args: dict) -> dict:
        """
        Create a new job entry in the prod_tools_table.
        
        Args:
            tool_run_id: Unique tool run ID (experiment ID)
            user_id: User ID for the job
            chat_id: Chat ID for the job
            protein_sequence: Protein sequence for input_data
            ligands: List of ligand SMILES for input_data
            
        Returns:
            Dictionary with the created job data
        """
        now = datetime.now().isoformat()
        
        # Create job entry without ID (let database auto-generate)
        job_data = {
            "created_at": now,
            "updated_at": now,
            "user_id": user_id,
            "chat_id": chat_id,
            "tool_run_id": tool_run_id,
            "tool_name": TOOL_NAME,
            "status": "pending",
            "tool_args": tool_args
        }
        
        tprint(f'saving job: {job_data}')
        created_job_data = self._storage.create_tool_job(job_data)
        
        if not created_job_data:
            raise Exception("Failed to create job in database")
        
        tprint(f'job created successfully with id: {created_job_data.get("id")}')
        return created_job_data
    
    @modal.method()
    def update_job_status(self, tool_run_id: str, status: str, **updates) -> dict:
        """
        Update job status and other fields in the prod_tools_table.
        
        Args:
            tool_run_id: Tool run ID (experiment ID) of the job to update
            status: New status (pending, running, completed, failed)
            **updates: Additional fields to update (started_at, completed_at, duration_seconds, output_data, error_message)
            
        Returns:
            Updated job data dictionary
        """
        tprint('\n\n')
        now = datetime.now().isoformat()
        
        # Prepare update data
        update_data = {
            "status": status,
            "updated_at": now,
            **updates
        }
        
        # Auto-set started_at when status changes to running
        if status == "running" and "started_at" not in updates:
            update_data["started_at"] = now
            
        # Auto-set completed_at when status changes to completed or failed
        if status in ["completed", "failed"] and "completed_at" not in updates:
            update_data["completed_at"] = now
            
        # Calculate duration if we have both started_at and completed_at
        if status in ["completed", "failed"] and "duration_seconds" not in updates:
            # Get current job to check started_at
            current_job = self._get_job_from_db(tool_run_id)
            if current_job and current_job.get("started_at"):
                try:
                    started = datetime.fromisoformat(current_job["started_at"].replace('Z', '+00:00'))
                    completed = datetime.fromisoformat(now)
                    duration = (completed - started).total_seconds()
                    update_data["duration_seconds"] = duration
                except Exception as e:
                    tprint(f"Could not calculate duration: {e}")
        
        tprint(f"Attempting to update job {tool_run_id} with data: {update_data}")
        result = self._storage.update_tool_job(tool_run_id, update_data)
        
        if not result:
            existing_job = self._get_job_from_db(tool_run_id)
            raise Exception(f"Failed to update job {tool_run_id} in database. Job exists: {existing_job is not None}")
            
        return self._get_job_from_db(tool_run_id)
    
    @modal.method()
    def get_job(self, tool_run_id: str) -> Optional[dict]:
        """Get job by tool_run_id from prod_tools_table."""
        return self._get_job_from_db(tool_run_id)
    
    def _get_job_from_db(self, tool_run_id: str) -> Optional[dict]:
        """Internal method to get job from database."""
        tprint(f"\t\t-Looking up job with tool_run_id: {tool_run_id}")
        job_data = self._storage.get_tool_job(tool_run_id)
        if job_data:
            tprint(f"\t\t-Found job data: {job_data}")
        return job_data


SCALEDOWN_WINDOW = 20 * 60  # 20 minute

def create_affinity_markdown_table(affinity_results: list) -> str:
    """
    Create a markdown table from affinity results.
    
    Args:
        affinity_results: List of dictionaries containing affinity data
        
    Returns:
        Markdown formatted table string
    """
    if not affinity_results:
        return "No affinity data available - all ligand predictions failed or contained no affinity information."
    
    # Create markdown table
    table_lines = [
        "| Ligand Index | SMILES | Affinity Prediction Value | Affinity Probability Binary |",
        "|--------------|--------|---------------------------|----------------------------|"
    ]
    
    for result in affinity_results:
        ligand_index = result.get("ligand_index", "N/A")
        smiles = result.get("ligand_smiles", "N/A")
        affinity_pred = result.get("affinity_pred_value", "N/A")
        affinity_prob = result.get("affinity_probability_binary", "N/A")
        
        # Format numerical values
        if isinstance(affinity_pred, (int, float)):
            affinity_pred = f"{affinity_pred:.4f}"
        if isinstance(affinity_prob, (int, float)):
            affinity_prob = f"{affinity_prob:.4f}"
        
        table_lines.append(f"| {ligand_index} | {smiles} | {affinity_pred} | {affinity_prob} |")
    
    return "\n".join(table_lines)


@app.function(image=boltz_image, scaledown_window=SCALEDOWN_WINDOW, retries=2)
def process_input_molecule(input_smiles: str) -> str:
    """
    Check if the provided input molecule is valid.
    
    Args:
        input_smiles: Input molecule SMILES to validate
        
    Returns:
        Valid input molecule SMILES
    """
    from rdkit.Chem import MolFromSmiles as smi2mol
    from rdkit.Chem import AllChem as Chem

    try:
        # Basic input validation
        if not input_smiles or not input_smiles.strip():
            tprint("Empty SMILES string provided", error_type="ligand_validation")
            return ""
        
        input_smiles = input_smiles.strip()
        
        # Check for reasonable length
        if len(input_smiles) > MAX_SMILES_LENGTH:
            tprint(f"SMILES string too long: {len(input_smiles)} characters", error_type="ligand_validation")
            return ""
        
        seed_mol = smi2mol(input_smiles)
        
        if seed_mol is None:
            tprint(f"Invalid SMILES provided: {input_smiles}", error_type="ligand_validation")
            return ""
        
        # Check for reasonable molecular properties
        mol_weight = Chem.CalcExactMolWt(seed_mol)
        if mol_weight > MAX_MOLECULAR_WEIGHT:
            tprint(f"Molecule too large: {mol_weight:.2f} Da", error_type="ligand_validation")
            return ""
        
        # Preprocess input SMILES
        smiles = Chem.MolToSmiles(seed_mol)
        tprint(f"Successfully validated SMILES: {input_smiles} -> {smiles}")
        return smiles
            
    except Exception as e:
        error_context = create_error_context(e)
        tprint(f"Error validating input molecule: {e}", 
               error_type=error_context['error_type'],
               guidance=error_context['guidance'])
        return ""


@app.function(image=boltz_image, scaledown_window=SCALEDOWN_WINDOW, volumes={"/boltz_tmp": volume}, retries=2)
def build_boltz_yaml(input_protein: str, input_ligand: str, output_path: str) -> str:
    """
    Build and save Boltz YAML configuration file with full format support.
    
    Args:
        input_protein: Protein sequence in FASTA format
        input_ligand: Ligand SMILES string
        output_path: Path to save the YAML file (must be within /boltz_tmp)

    Returns:
        Path to the saved YAML file
    """
    import yaml
    from pathlib import Path
    import os
    
    # Reload volume to see any recent changes
    volume.reload()
    tprint(f'ls /boltz_tmp after reload: {os.listdir("/boltz_tmp")}')

    # Ensure output_path is within the volume mount
    if not output_path.startswith('/boltz_tmp/'):
        raise ValueError(f"Output path must be within volume mount /boltz_tmp/, got: {output_path}")

    protein_sequence = extract_protein_sequence(input_protein)
    tprint(
        "Building Boltz YAML config",
        protein_length=len(protein_sequence),
        ligand_smiles=input_ligand,
        output_path=output_path,
    )

    config = build_boltz_config(protein_sequence, input_ligand)
    tprint(f"YAML configuration: {config}")
    
    # Ensure directory exists
    yaml_path = Path(output_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save YAML file
    tprint(f"Saving YAML file to: {yaml_path}")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    tprint(f"YAML configuration saved to: {yaml_path}")
    tprint(f"YAML content:\n{yaml.dump(config, default_flow_style=False, indent=2)}")

    # Commit changes to make them visible to other containers
    volume.commit()
    tprint(f"Volume committed after saving YAML file")
    
    return str(yaml_path)


@app.function(image=boltz_image, scaledown_window=SCALEDOWN_WINDOW, gpu="A10G", volumes={"/boltz_tmp": volume}, retries=2, timeout=3600)
def cofold(input_protein: str, input_ligand: str, experiment_id: str = "", ligand_index: int = 0, 
           user_id: str = "", chat_id: str = "") -> dict:
    """
    Run boltz cofold using YAML configuration for protein-ligand cofolding.
    
    Args:
        input_protein: Protein sequence in FASTA format
        input_ligand: Single ligand SMILES string
        experiment_id: Experiment ID for organizing results
        ligand_index: Index of this ligand in the batch
        user_id: User ID for cloud storage
        chat_id: Chat ID for cloud storage
        
    Returns:
        Dictionary containing cofolding results, cloud storage info, or error information
    """
    import subprocess
    import os
    from pathlib import Path

    try:
        # Reload volume to see latest changes
        volume.reload()
        tprint(f'Volume reloaded in cofold function')

        top_dir = Path(f'/boltz_tmp/{experiment_id}')
        
        # Create unique output directory for this ligand WITHIN the volume
        volume_output_dir = top_dir.joinpath(f"boltz_output_ligand_{ligand_index}")

        # Use volume mount path for all operations
        # volume_output_dir = f"{top_dir}/{output_dir_name}"
        tprint(f"Creating output directory in volume: {volume_output_dir}")
        os.makedirs(volume_output_dir, exist_ok=True)

        # Build YAML configuration file WITHIN the volume
        yaml_file_path = volume_output_dir.joinpath("config.yaml")
        tprint(f"Building YAML file at: {yaml_file_path}")
        
        # Call the build function with the volume path
        build_boltz_yaml.remote(input_protein, input_ligand, str(yaml_file_path))
        
        # Reload to see the newly created YAML file
        volume.reload()
        tprint(f'(in cofold) yaml_file_path: {yaml_file_path}')
        tprint(f'(in cofold) yaml_file_path exists: {os.path.exists(yaml_file_path)}')
        
        if not os.path.exists(yaml_file_path):
            raise FileNotFoundError(f"YAML file was not created at expected location: {yaml_file_path}")
        
        # Prepare Boltz command using YAML configuration
        cmd = [
            "boltz", "predict",
            str(yaml_file_path),  # Use the volume path
            "--out_dir", str(volume_output_dir),  # Output to volume directory
            "--use_msa_server",
            "--accelerator", "gpu",
        ]
        
        tprint(f"Running Boltz command for ligand {ligand_index}: {' '.join(cmd)}")
        
        # Calculate dynamic timeout based on complexity
        timeout_seconds = calculate_dynamic_timeout(1, base_timeout=1800)  # Single ligand
        tprint(f"Using timeout: {timeout_seconds} seconds for ligand {ligand_index}")
        
        # Run Boltz prediction
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        tprint(f'checking result.returncode: {result.returncode}', ligand_index=ligand_index)
        if result.returncode != 0:
            error_msg = f"Boltz command failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            error_context = create_error_context(Exception(error_msg), experiment_id, ligand_index)
            
            tprint(f"Boltz prediction failed for ligand {ligand_index}", 
                   error_type=error_context['error_type'],
                   return_code=result.returncode,
                   experiment_id=experiment_id)
            
            return create_cofold_error_response(error_msg, error_context, input_ligand, ligand_index)
        
        # Process output files from Boltz (now in volume)
        output_files = list(volume_output_dir.glob("**/*"))
     
        results = []    
        # Extract affinity values from predictions
        affinity_data = None
        cloud_storage_result = None
        try:
            predictions_dir = volume_output_dir.joinpath("boltz_results_config", "predictions")
            tprint(f'predictions_dir: {predictions_dir}')
            if predictions_dir.exists():
                # Find the randomly named subdirectory (should be only one)
                subdirs = [d for d in predictions_dir.iterdir() if d.is_dir()]
                if subdirs:
                    prediction_subdir = subdirs[0]  # Take the first (and should be only) subdirectory
                    subdir_name = prediction_subdir.name
                    # Save files in this subdir to cloud storage.
                    cloud_storage_result = save_to_cloud_function.spawn(
                        local_dir=str(prediction_subdir),
                        cloud_dir=f"results/ligand_{ligand_index}",
                        user_id=user_id,
                        chat_id=chat_id,
                        experiment_id=experiment_id
                    )
                    
                    # Look for affinity JSON file
                    affinity_file = prediction_subdir / f"affinity_{subdir_name}.json"
                    if affinity_file.exists():
                        with open(affinity_file, 'r') as f:
                            import json
                            affinity_json = json.load(f)
                            affinity_data = {
                                'affinity_pred_value': affinity_json.get('affinity_pred_value'),
                                'affinity_probability_binary': affinity_json.get('affinity_probability_binary'),
                                'prediction_dir': subdir_name
                            }
                            tprint(f"Extracted affinity data for ligand {ligand_index}: {affinity_data}")
                    else:
                        tprint(f"Affinity file not found: {affinity_file}")
                else:
                    tprint(f"No prediction subdirectories found in: {predictions_dir}")
            else:
                tprint(f"Predictions directory not found: {predictions_dir}")
        except Exception as e:
            tprint(f"Error extracting affinity data for ligand {ligand_index}: {str(e)}")
            affinity_data = None
        
        # Commit changes to volume
        volume.commit()
        tprint(f"Volume committed after processing ligand {ligand_index}")


        return {
            'results': results,
            'error': None,
            'ligand_smiles': input_ligand,
            'ligand_index': ligand_index,
            'affinity_data': affinity_data,
            'cloud_storage': cloud_storage_result,
            'metadata': {
                'status': 'success',
                'message': f'Successfully generated {len(results)} cofolded structures for ligand {ligand_index}',
                'total_structures': len(results),
                'output_files': [str(f) for f in output_files],
                'local_output_directory': str(volume_output_dir),
                'cloud_saved': bool(user_id and chat_id and experiment_id)
            }
        }

    except TimeoutError as e:
        timeout_used = calculate_dynamic_timeout(1, base_timeout=1800)
        error_msg = f"cofold: Boltz prediction timed out after {timeout_used//60} minutes for ligand {ligand_index}"
        error_context = create_error_context(e, experiment_id, ligand_index)
        
        tprint(error_msg, 
               error_type=error_context['error_type'],
               timeout_minutes=timeout_used//60,
               experiment_id=experiment_id)
        
        return create_cofold_error_response(error_msg, error_context, input_ligand, ligand_index)
    except Exception as e:
        error_msg = f"cofold: Error in Boltz cofolding for ligand {ligand_index}: {str(e)}"
        error_context = create_error_context(e, experiment_id, ligand_index)
        
        tprint(error_msg, 
               error_type=error_context['error_type'],
               experiment_id=experiment_id,
               ligand_index=ligand_index)
        
        return create_cofold_error_response(error_msg, error_context, input_ligand, ligand_index)

@app.function(image=boltz_image, scaledown_window=SCALEDOWN_WINDOW, volumes={"/boltz_tmp": volume}, retries=1, timeout=7200)
def process_cofold_batch_async(
    protein_sequence: str, 
    validated_ligands: list[str], 
    experiment_id: str,
    user_id: str, 
    chat_id: str
):
    """
    Process all cofold runs asynchronously and handle results in background.
    
    Args:
        protein_sequence: Protein sequence in FASTA format
        validated_ligands: List of validated ligand SMILES
        experiment_id: Experiment ID
        user_id: User ID for cloud storage
        chat_id: Chat ID for cloud storage
    """
    import time
    from datetime import datetime
    
    start_time = time.time()
    
    # Calculate dynamic timeout for the entire batch
    batch_timeout = calculate_dynamic_timeout(len(validated_ligands), base_timeout=3600)
    tprint(f"Starting async batch processing for experiment: {experiment_id}", 
           experiment_id=experiment_id,
           total_ligands=len(validated_ligands),
           estimated_timeout_minutes=batch_timeout//60)
    
    # Initialize job manager and update status to running
    try:
        job_manager = BoltzJobManager()
        update_result = job_manager.update_job_status.remote(experiment_id, "running")
        tprint(f"Job status update result: {update_result}")
    except Exception as e:
        tprint(f"Error updating job status to running: {e}")
        # Continue anyway, don't fail the entire batch
    
    try:
        # Create list of args for cofold (as tuples for positional arguments)
        args = []
        for i, ligand in enumerate(validated_ligands):
            args.append((
                protein_sequence,       # input_protein
                ligand,                 # input_ligand
                experiment_id,          # experiment_id
                i,                      # ligand_index
                user_id,                # user_id
                chat_id,                # chat_id
            ))

        # Generate structures using parallel cofold calls with progress tracking
        tprint(f"Starting {len(args)} parallel cofold runs", 
               experiment_id=experiment_id,
               parallel_jobs=len(args))
        
        responses = cofold.starmap(args)
        
        elapsed_time = time.time() - start_time
        tprint(f"All cofold runs completed in {elapsed_time:.2f} seconds", 
               experiment_id=experiment_id,
               processing_time_seconds=elapsed_time,
               average_time_per_ligand=elapsed_time/len(args) if args else 0)

        # Process responses and create experiment manifest
        successful_results = []
        failed_results = []
        affinity_results = []
        
        # Create ligand mapping for easy reference
        ligand_mapping = {str(i): smiles for i, smiles in enumerate(validated_ligands)}
        
        experiment_manifest = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "protein_sequence": protein_sequence,
            "total_ligands": len(validated_ligands),
            "processing_time_seconds": elapsed_time,
            "status": "completed",
            "ligand_mapping": ligand_mapping,
            "ligand_experiments": []
        }
        
        for i, response in enumerate(responses):
            affinity_data = response.get("affinity_data")
            
            ligand_info = {
                "ligand_index": i,
                "ligand_smiles": validated_ligands[i],
                "status": response.get("metadata", {}).get("status", "unknown"),
                "structures_generated": len(response.get("results", [])),
                "error": response.get("error"),
                "cloud_saved": response.get("metadata", {}).get("cloud_saved", False)
            }
            
            # Add affinity data if available
            if affinity_data:
                ligand_info["affinity_pred_value"] = affinity_data.get("affinity_pred_value")
                ligand_info["affinity_probability_binary"] = affinity_data.get("affinity_probability_binary")
                affinity_results.append({
                    "ligand_index": i,
                    "ligand_smiles": validated_ligands[i],
                    "affinity_pred_value": affinity_data.get("affinity_pred_value"),
                    "affinity_probability_binary": affinity_data.get("affinity_probability_binary")
                })
           
            experiment_manifest["ligand_experiments"].append(ligand_info)
            
            if response.get("metadata", {}).get("status") == "success":
                successful_results.append(response)
            else:
                failed_results.append(response)

        # Update manifest with final results
        experiment_manifest["successful_count"] = len(successful_results)
        experiment_manifest["failed_count"] = len(failed_results)
        
        # Create affinity markdown table
        affinity_markdown = create_affinity_markdown_table(affinity_results)
        experiment_manifest["affinity_table_markdown"] = affinity_markdown
        
        # Save final experiment manifest
        save_experiment_manifest.spawn(
            manifest=experiment_manifest,
            user_id=user_id,
            chat_id=chat_id,
            experiment_id=experiment_id
        )
        
        # Update job status to completed with enhanced results
        try:
            job_manager = BoltzJobManager()
            success_rate = len(successful_results) / len(validated_ligands) * 100 if validated_ligands else 0
            job_manager.update_job_status.remote(
                experiment_id, 
                "completed",
                tool_results={
                    "experiment_id": experiment_id,
                    "successful_count": len(successful_results),
                    "failed_count": len(failed_results),
                    "total_ligands": len(validated_ligands),
                    "success_rate": success_rate,
                    "processing_time_seconds": int(elapsed_time),
                    "average_time_per_ligand": elapsed_time / len(validated_ligands) if validated_ligands else 0,
                    "affinity_predictions_available": len(affinity_results) > 0
                }
            )
        except Exception as job_error:
            tprint(f"Error updating job status to completed: {job_error}", 
                   error_type="database",
                   experiment_id=experiment_id)
        
        success_rate = len(successful_results) / len(validated_ligands) * 100 if validated_ligands else 0
        tprint(f"Async batch processing completed for experiment: {experiment_id}", 
               experiment_id=experiment_id,
               successful_count=len(successful_results),
               failed_count=len(failed_results),
               success_rate=f"{success_rate:.1f}%",
               total_processing_time=f"{elapsed_time:.1f}s")
    

    except TimeoutError as e:
        batch_timeout = calculate_dynamic_timeout(len(validated_ligands), base_timeout=3600)
        error_msg = f"process_cofold_batch_async: Batch processing timed out after {batch_timeout//60} minutes for experiment {experiment_id}"
        error_context = create_error_context(e, experiment_id)
        
        tprint(error_msg, 
               error_type=error_context['error_type'],
               experiment_id=experiment_id,
               timeout_minutes=batch_timeout//60,
               total_ligands=len(validated_ligands))
        
        # Try to update job status even on timeout
        try:
            job_manager = BoltzJobManager()
            job_manager.update_job_status.remote(
                experiment_id,
                "failed",
                error_message=error_msg,
                tool_results={
                    "experiment_id": experiment_id,
                    "error": error_msg,
                    "error_type": error_context['error_type'],
                    "processing_time_seconds": time.time() - start_time,
                    "total_ligands": len(validated_ligands),
                    "timeout_minutes": batch_timeout//60
                }
            )
        except Exception as job_error:
            tprint(f"Error updating job status after timeout: {job_error}", 
                   error_type="database",
                   experiment_id=experiment_id)
        
        return {
            'results': [],
            'error': error_msg,
            'error_context': error_context,
            'ligand_smiles': validated_ligands,
            'ligand_index': list(range(len(validated_ligands))),
            'affinity_data': None,
            'cloud_storage': None,
            'metadata': {
                'status': 'error',
                'message': error_context['guidance'],
                'error_type': error_context['error_type'],
                'cloud_saved': False
            }
        }

    except Exception as e:
        error_message = f"Error in async batch processing for experiment {experiment_id}: {str(e)}"
        error_context = create_error_context(e, experiment_id)
        
        tprint(error_message, 
               error_type=error_context['error_type'],
               experiment_id=experiment_id,
               processing_time=time.time() - start_time)
        
        # Save error manifest with enhanced context
        ligand_mapping = {str(i): smiles for i, smiles in enumerate(validated_ligands)}
        
        error_manifest = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "protein_sequence": protein_sequence,
            "protein_length": len(protein_sequence),
            "total_ligands": len(validated_ligands),
            "status": "error",
            "error_message": error_message,
            "error_type": error_context['error_type'],
            "error_guidance": error_context['guidance'],
            "processing_time_seconds": time.time() - start_time,
            "ligand_mapping": ligand_mapping
        }
        
        save_experiment_manifest.spawn(
            manifest=error_manifest,
            user_id=user_id,
            chat_id=chat_id,
            experiment_id=experiment_id
        )
        
        # Update job status to failed with enhanced error context
        try:
            job_manager = BoltzJobManager()
            job_manager.update_job_status.remote(
                experiment_id,
                "failed",
                error_message=error_message,
                tool_results={
                    "experiment_id": experiment_id,
                    "error": error_message,
                    "error_type": error_context['error_type'],
                    "processing_time_seconds": time.time() - start_time,
                    "total_ligands": len(validated_ligands)
                }
            )
        except Exception as job_error:
            tprint(f"Error updating job status to failed: {job_error}", 
                   error_type="database",
                   experiment_id=experiment_id)

@app.function(image=web_image, volumes={"/boltz_tmp": volume})
@modal.fastapi_endpoint(method="POST", docs=True)
def run_boltz2(query: InputQuery) -> dict:
    """
    Main inference endpoint for molecule generation.
    Returns immediately with experiment_id and cloud path while processing continues in background.
    
    Args:
        query: Input query containing SMILES, scaffold, and generation parameters
        
    Returns:
        Dictionary containing experiment_id and cloud storage information
    """
    import time
    import uuid
    from datetime import datetime
    
    start_time = time.time()
    
    # Generate experiment ID (which will also be the job ID)
    experiment_id = f"boltz2_exp_{uuid.uuid4().hex[:6]}"
    tprint(f"Starting experiment: {experiment_id}")

    # Enhanced input validation
    tprint('Starting enhanced input validation', experiment_id=experiment_id)
    
    # Validate protein sequence first
    is_valid_protein, protein_message, clean_protein = validate_protein_sequence(query.input_protein)
    if not is_valid_protein:
        error_context = create_error_context(ValueError(protein_message), experiment_id)
        tprint(f"Protein validation failed: {protein_message}", 
               error_type=error_context['error_type'],
               experiment_id=experiment_id)
        return create_experiment_status_response(
            experiment_id,
            "error",
            protein_message,
            error_type=error_context["error_type"],
            guidance=error_context["guidance"],
        )
    
    # Validate ligand batch size
    if len(query.list_of_ligands) == 0:
        return create_experiment_status_response(
            experiment_id,
            "error",
            "No ligands provided. Please provide at least one valid SMILES string.",
            error_type="ligand_validation",
        )
    
    if len(query.list_of_ligands) > MAX_LIGANDS_PER_REQUEST:
        return create_experiment_status_response(
            experiment_id,
            "error",
            f"Too many ligands: {len(query.list_of_ligands)} (maximum: {MAX_LIGANDS_PER_REQUEST}). Please split into smaller batches.",
            error_type="ligand_validation",
        )
    
    # Validate ligand molecules
    tprint(f'Validating {len(query.list_of_ligands)} ligand molecules')
    invalid_ligands = []
    validated_ligands = []
    
    # Process ligands in parallel but with better error tracking
    validation_results = []
    for i, ligand in enumerate(query.list_of_ligands):
        try:
            input_smiles = process_input_molecule.remote(ligand)
            validation_results.append((i, ligand, input_smiles))
        except Exception as e:
            tprint(f"Error processing ligand {i}: {e}", ligand_index=i)
            validation_results.append((i, ligand, ""))
    
    for i, original_ligand, validated_smiles in validation_results:
        if validated_smiles == "":
            invalid_ligands.append(f"Index {i}: {original_ligand}")
        else:
            validated_ligands.append(validated_smiles)
    
    if invalid_ligands:
        error_msg = f"Invalid ligands found ({len(invalid_ligands)}/{len(query.list_of_ligands)}): {', '.join(invalid_ligands[:3])}{'...' if len(invalid_ligands) > 3 else ''}"
        return create_experiment_status_response(
            experiment_id,
            "error",
            error_msg,
            error_type="ligand_validation",
            invalid_count=len(invalid_ligands),
            valid_count=len(validated_ligands),
            guidance="Check SMILES format and ensure molecules are chemically valid. Common issues: invalid characters, unclosed rings, or impossible valences.",
        )
    
    tprint(f"Validation complete: {len(validated_ligands)} valid ligands, protein sequence: {len(clean_protein)} residues", 
           experiment_id=experiment_id,
           valid_ligands=len(validated_ligands),
           protein_length=len(clean_protein))

    try:
        # Create job in database
        tprint(f"Creating job in database for experiment: {experiment_id}")
        job_manager = BoltzJobManager()
        job_manager.create_job.remote(
            tool_run_id=experiment_id,
            user_id=query.user_id,
            chat_id=query.chat_id,
            tool_args={
                "protein_sequence": clean_protein,  # Use cleaned protein sequence
                "protein_length": len(clean_protein),
                "ligands": validated_ligands,
                "ligand_count": len(validated_ligands)
            }
        )
        tprint(f"Job created in database for experiment: {experiment_id}")

        # Create initial experiment manifest
        ligand_mapping = {str(i): smiles for i, smiles in enumerate(validated_ligands)}
        
        initial_manifest = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "protein_sequence": clean_protein,  # Use cleaned protein sequence
            "protein_length": len(clean_protein),
            "total_ligands": len(validated_ligands),
            "status": "processing",
            "ligand_mapping": ligand_mapping,
            "ligand_smiles": validated_ligands,
            "cloud_storage_path": f'tool_outputs/boltz2/{experiment_id}/'
        }
        
        # Save initial manifest
        save_experiment_manifest.spawn(
            manifest=initial_manifest,
            user_id=query.user_id,
            chat_id=query.chat_id,
            experiment_id=experiment_id
        )
        
        # Start async processing with clean protein sequence
        process_cofold_batch_async.spawn(
            protein_sequence=clean_protein,  # Use validated and cleaned protein sequence
            validated_ligands=validated_ligands,
            experiment_id=experiment_id,
            user_id=query.user_id,
            chat_id=query.chat_id
        )
        
        elapsed_time = time.time() - start_time
        tprint(f"Experiment {experiment_id} started, returning immediately after {elapsed_time:.2f} seconds")

        # Return immediately while processing continues in background
        # Calculate estimated completion time
        estimated_time_minutes = calculate_dynamic_timeout(len(validated_ligands), base_timeout=1800) // 60
        
        return {
            **create_experiment_status_response(
                experiment_id,
                "processing",
                f"Experiment started successfully with {len(validated_ligands)} ligands. Processing in background.",
                total_ligands=len(validated_ligands),
                protein_length=len(clean_protein),
                estimated_completion_minutes=estimated_time_minutes,
                validation_time_seconds=round(elapsed_time, 2),
            ),
            "ligand_info": {
                "validated_ligands": validated_ligands,
                "ligand_count": len(validated_ligands)
            },
            "processing_info": {
                "estimated_time_minutes": estimated_time_minutes,
                "parallel_processing": True,
                "gpu_accelerated": True
            }
        }

    except Exception as e:
        error_message = f"Error starting experiment {experiment_id}: {str(e)}"
        error_context = create_error_context(e, experiment_id)
        
        tprint(error_message, 
               error_type=error_context['error_type'],
               experiment_id=experiment_id)
        
        return create_experiment_status_response(
            experiment_id,
            "error",
            error_context["guidance"],
            error_type=error_context["error_type"],
            detailed_error=error_message,
        )

def create_readme_content(experiment_id: str, total_ligands: int, protein_length: int = 0) -> str:
    """Create user-friendly README content for experiment results"""
    from datetime import datetime
    
    readme = f"""BOLTZ-2 PROTEIN-LIGAND COFOLDING RESULTS
=========================================
Experiment ID: {experiment_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Protein length: {protein_length if protein_length > 0 else 'Not specified'} amino acids
Ligands tested: {total_ligands}


CONTENTS
--------
• experiment_manifest.json - Complete summary with affinity predictions
• results/ - {total_ligands} ligand directories with PDB structures and affinity data


QUICK START
-----------
1. Open experiment_manifest.json and check "affinity_table_markdown" for ranked results
2. Navigate to results/ligand_X/ for the best-scoring ligand
3. Open structure_*.pdb with PyMOL, ChimeraX, or VMD
4. Review affinity_*.json for binding predictions


KEY METRICS
-----------
• affinity_pred_value: Binding strength (higher = stronger)
• affinity_probability_binary: Confidence score (0-1, >0.7 = reliable)


RESOURCES
---------
• Boltz-2: https://github.com/jwohlwend/boltz
• PyMOL: https://pymol.org/
• ChimeraX: https://www.cgl.ucsf.edu/chimerax/
"""
    return readme.strip()

@app.function(image=boltz_image, scaledown_window=SCALEDOWN_WINDOW, volumes={"/boltz_tmp": volume})
def save_experiment_manifest(manifest: dict, user_id: str, chat_id: str, experiment_id: str) -> dict:
    """
    Save experiment manifest and README to cloud storage.
    
    Args:
        manifest: Experiment manifest dictionary
        user_id: User ID for bucket identification
        chat_id: Chat ID for file organization
        experiment_id: Experiment ID
        
    Returns:
        Dictionary with save status and file information
    """
    import json

    try:
        storage = build_storage_location(user_id, chat_id, experiment_id)
        s3_client = create_s3_client()
        
        # Convert manifest to JSON
        manifest_json = json.dumps(manifest, indent=2)
        
        # Upload manifest to S3
        s3_key = f"{storage['base_path']}experiment_manifest.json"
        s3_client.put_object(
            Bucket=storage["bucket"],
            Key=s3_key,
            Body=manifest_json.encode('utf-8'),
            ContentType='application/json'
        )
        
        tprint(f"Successfully saved experiment manifest to s3://{storage['bucket']}/{s3_key}")
        
        # Create and upload README file
        total_ligands = manifest.get("total_ligands", 0)
        protein_length = len(manifest.get("protein_sequence", ""))
        readme_content = create_readme_content(experiment_id, total_ligands, protein_length)
        
        readme_key = f"{storage['base_path']}README.txt"
        s3_client.put_object(
            Bucket=storage["bucket"],
            Key=readme_key,
            Body=readme_content.encode('utf-8'),
            ContentType='text/plain'
        )
        
        tprint(f"Successfully saved README to s3://{storage['bucket']}/{readme_key}")
        
        return {
            "status": "success",
            "message": f"Experiment manifest and README saved for {experiment_id}",
            "bucket": storage["bucket"],
            "manifest_s3_key": s3_key,
            "readme_s3_key": readme_key,
            "s3_path": storage["s3_path"],
            "agent_path": storage["agent_path"]
        }
        
    except Exception as e:
        error_message = f"Error saving experiment manifest: {str(e)}"
        tprint(error_message)
        return {
            "status": "error",
            "message": error_message
        }

@app.function(image=boltz_image, scaledown_window=SCALEDOWN_WINDOW, volumes={"/boltz_tmp": volume})
def save_to_cloud_function(local_dir: str, cloud_dir: str, user_id: str, chat_id: str, experiment_id: str) -> dict:
    """
    Save the output files to the user's cloud storage.
    
    Args:
        local_dir: Local directory containing the output files (should be volume path)
        cloud_dir: Subdirectory name within the experiment (e.g., "ligand_0")
        user_id: User ID for bucket identification
        chat_id: Chat ID for file organization
        experiment_id: Experiment ID for organizing results
        
    Returns:
        Dictionary with save status and file information
    """
    from pathlib import Path

    try:
        # Reload volume to ensure we have the latest data
        volume.reload()
        tprint(f"Volume reloaded in save_to_cloud_function")
        
        storage = build_storage_location(user_id, chat_id, experiment_id, cloud_dir)
        s3_client = create_s3_client()
        
        # Process local directory (should be volume mounted path)
        local_dir = Path(local_dir)
        tprint(f"Processing directory: {local_dir}")
        tprint(f"Directory exists: {local_dir.exists()}")
        
        if not local_dir.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")
            
        output_files = [f for f in local_dir.glob("**/*") if f.is_file()]
        tprint(f"Found {len(output_files)} files to upload")
        
        if not output_files:
            raise ValueError(f"No files found in directory: {local_dir}")
        
        # Upload all files in the output directory to S3
        uploaded_files = []
        for file_path in output_files:
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{storage['base_path']}{relative_path}"
            
            tprint(f"Uploading {file_path} to s3://{storage['bucket']}/{s3_key}")
            
            with open(file_path, 'rb') as f:
                s3_client.put_object(
                    Bucket=storage["bucket"],
                    Key=s3_key,
                    Body=f.read(),
                    ContentType='application/octet-stream'
                )
            
            uploaded_files.append({
                'local_path': str(file_path),
                's3_key': s3_key,
                'filename': file_path.name
            })
        
        tprint(f"Successfully saved {len(uploaded_files)} files to {storage['s3_path']}")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(uploaded_files)} Boltz output files to {cloud_dir}",
            "bucket": storage["bucket"],
            "base_path": storage["base_path"],
            "s3_base_path": storage["s3_path"],
            "agent_path": storage["agent_path"],
            "uploaded_files": uploaded_files,
            "file_count": len(uploaded_files)
        }
        
    except Exception as e:
        error_message = f"Error saving to cloud storage: {str(e)}"
        tprint(error_message)
        return {
            "status": "error",
            "message": error_message
        }

# RESULT RETRIEVAL ENDPOINTS
# ==========================

class ResultQuery(BaseModel):
    """Query model for retrieving experiment results"""
    experiment_id: str = Field(..., description="The experiment ID to retrieve results for")

@app.function(image=web_image)
@modal.fastapi_endpoint(method="GET", docs=True)
def get_results(experiment_id: str) -> dict:
    """
    Retrieve results for a completed Boltz experiment.
    
    Args:
        experiment_id: The experiment ID from the original inference call
        
    Returns:
        Dictionary containing experiment results, status, and cloud storage information
    """
    import time
    from datetime import datetime
    
    start_time = time.time()
    tprint(f"Retrieving results for experiment: {experiment_id}", experiment_id=experiment_id)
    
    try:
        # Get job status from database
        job_manager = BoltzJobManager()
        job_data = job_manager.get_job.remote(experiment_id)
        
        if not job_data:
            return {
                "experiment_id": experiment_id,
                "status": "not_found",
                "message": f"No experiment found with ID: {experiment_id}",
                "error_type": "experiment_not_found"
            }
        
        # Extract job information
        job_status = job_data.get("status", "unknown")
        tool_results = job_data.get("tool_results", {})
        tool_args = job_data.get("tool_args", {})
        error_message = job_data.get("error_message")
        created_at = job_data.get("created_at")
        completed_at = job_data.get("completed_at")
        duration_seconds = job_data.get("duration_seconds")
        
        # Base response structure
        response = {
            "experiment_id": experiment_id,
            "status": job_status,
            "created_at": created_at,
            "completed_at": completed_at,
            "duration_seconds": duration_seconds,
            "retrieval_time": datetime.now().isoformat(),
            "query_time_seconds": round(time.time() - start_time, 3)
        }
        
        # Add job arguments (input data)
        if tool_args:
            response["input_data"] = {
                "protein_sequence": tool_args.get("protein_sequence", "N/A"),
                "protein_length": tool_args.get("protein_length", 0),
                "ligands": tool_args.get("ligands", []),
                "ligand_count": tool_args.get("ligand_count", 0)
            }
        
        # Handle different job statuses
        if job_status == "pending":
            response["message"] = "Experiment is still pending. Processing has not started yet."
            
        elif job_status == "running":
            response["message"] = "Experiment is currently running. Results not yet available."
            if duration_seconds:
                response["elapsed_time_minutes"] = round(duration_seconds / 60, 1)
                
        elif job_status == "completed":
            response["message"] = "Experiment completed successfully."
            
            # Add detailed results
            if tool_results:
                response["results"] = {
                    "successful_count": tool_results.get("successful_count", 0),
                    "failed_count": tool_results.get("failed_count", 0),
                    "total_ligands": tool_results.get("total_ligands", 0),
                    "success_rate": tool_results.get("success_rate", 0),
                    "processing_time_seconds": tool_results.get("processing_time_seconds", 0),
                    "average_time_per_ligand": tool_results.get("average_time_per_ligand", 0),
                    "affinity_predictions_available": tool_results.get("affinity_predictions_available", False)
                }
                
                # Calculate processing time in human-readable format
                if "processing_time_seconds" in tool_results:
                    total_seconds = tool_results["processing_time_seconds"]
                    response["results"]["processing_time_formatted"] = format_duration(total_seconds)
            
            # Add cloud storage information
            # Extract user_id and chat_id from job data to construct cloud paths
            user_id = job_data.get("user_id")
            chat_id = job_data.get("chat_id")
            
            if user_id and chat_id:
                storage = build_storage_location(user_id, chat_id, experiment_id)
                
                response["cloud_storage"] = {
                    "bucket": storage["bucket"],
                    "base_path": storage["base_path"],
                    "s3_path": storage["s3_path"],
                    "agent_path": storage["agent_path"],
                    "manifest_file": f"{storage['s3_path']}experiment_manifest.json",
                    "access_instructions": "Use AWS CLI or SDK to access files: aws s3 ls s3://bucket/path/"
                }
                
                # Add expected file structure with clear organization
                ligand_count = tool_args.get("ligand_count", 0)
                
                response["cloud_storage"]["file_structure"] = {
                    "experiment_manifest.json": "Complete experiment summary with all results",
                    "README.txt": "Guide to understanding and using your results",
                    "results/": f"{ligand_count} ligand directories containing structures and affinity data"
                }
                
                response["cloud_storage"]["usage_instructions"] = [
                    "1. Start by reading README.txt for an overview",
                    "2. Check experiment_manifest.json for detailed results summary",
                    "3. Navigate to results/ligand_X/ folders for individual ligand data",
                    "4. Each ligand folder contains PDB structures and affinity predictions"
                ]
                
        elif job_status == "failed":
            response["message"] = "Experiment failed during processing."
            response["error"] = error_message or "Unknown error occurred"
            
            # Try to categorize the error
            if error_message:
                error_context = create_error_context(Exception(error_message), experiment_id)
                response["error_type"] = error_context["error_type"]
                response["guidance"] = error_context["guidance"]
            
            # Add partial results if available
            if tool_results:
                response["partial_results"] = tool_results
                
        else:
            response["message"] = f"Unknown status: {job_status}"
            
        tprint(f"Successfully retrieved results for experiment: {experiment_id}", 
               experiment_id=experiment_id,
               status=job_status,
               query_time=response["query_time_seconds"])
        
        return response
        
    except Exception as e:
        error_message = f"Error retrieving results for experiment {experiment_id}: {str(e)}"
        error_context = create_error_context(e, experiment_id)
        
        tprint(error_message, 
               error_type=error_context['error_type'],
               experiment_id=experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "status": "error",
            "message": error_context['guidance'],
            "error_type": error_context['error_type'],
            "detailed_error": error_message,
            "query_time_seconds": round(time.time() - start_time, 3)
        }

@app.function(image=web_image)
@modal.fastapi_endpoint(method="GET", docs=True)
def get_experiment_manifest(experiment_id: str) -> dict:
    """
    Retrieve the experiment manifest directly from cloud storage.
    
    Args:
        experiment_id: The experiment ID to retrieve manifest for
        
    Returns:
        Dictionary containing the experiment manifest or error information
    """
    import json
    import time
    from datetime import datetime
    
    start_time = time.time()
    tprint(f"Retrieving manifest for experiment: {experiment_id}", experiment_id=experiment_id)
    
    try:
        # First get job data to find user_id and chat_id
        job_manager = BoltzJobManager()
        job_data = job_manager.get_job.remote(experiment_id)
        
        if not job_data:
            return {
                "experiment_id": experiment_id,
                "status": "not_found",
                "message": f"No experiment found with ID: {experiment_id}"
            }
        
        user_id = job_data.get("user_id")
        chat_id = job_data.get("chat_id")
        
        if not user_id or not chat_id:
            return {
                "experiment_id": experiment_id,
                "status": "error",
                "message": "Missing user_id or chat_id in job data"
            }
        
        storage = build_storage_location(user_id, chat_id, experiment_id)
        s3_client = create_s3_client()
        manifest_key = f"{storage['base_path']}experiment_manifest.json"
        
        # Try to retrieve the manifest
        try:
            response = s3_client.get_object(Bucket=storage["bucket"], Key=manifest_key)
            manifest_content = response['Body'].read().decode('utf-8')
            manifest_data = json.loads(manifest_content)
            
            # Add retrieval metadata
            manifest_data["retrieval_info"] = {
                "retrieved_at": datetime.now().isoformat(),
                "s3_path": f"{storage['s3_path']}experiment_manifest.json",
                "agent_path": convert_s3_to_agent_path(f"{storage['s3_path']}experiment_manifest.json"),
                "query_time_seconds": round(time.time() - start_time, 3)
            }
            
            tprint(f"Successfully retrieved manifest for experiment: {experiment_id}", 
                   experiment_id=experiment_id,
                   manifest_status=manifest_data.get("status", "unknown"))
            
            return manifest_data
            
        except s3_client.exceptions.NoSuchKey:
            return {
                "experiment_id": experiment_id,
                "status": "manifest_not_found",
                "message": "Experiment manifest not found in cloud storage. The experiment may still be processing or may have failed before creating the manifest.",
                "s3_path": f"{storage['s3_path']}experiment_manifest.json",
                "agent_path": convert_s3_to_agent_path(f"{storage['s3_path']}experiment_manifest.json"),
                "query_time_seconds": round(time.time() - start_time, 3)
            }
            
    except Exception as e:
        error_message = f"Error retrieving manifest for experiment {experiment_id}: {str(e)}"
        error_context = create_error_context(e, experiment_id)
        
        tprint(error_message, 
               error_type=error_context['error_type'],
               experiment_id=experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "status": "error",
            "message": error_context['guidance'],
            "error_type": error_context['error_type'],
            "detailed_error": error_message,
            "query_time_seconds": round(time.time() - start_time, 3)
        }

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def convert_s3_to_agent_path(s3_path: str) -> str:
    """
    Convert S3 path to simplified agent path.
    
    Converts: s3://bucket/user_id/chat_id/tool_outputs/... 
    To: /cloud_storage/tool_outputs/...
    
    Args:
        s3_path: S3 path string
        
    Returns:
        Simplified agent path string
    """
    import re
    # Match pattern: s3://bucket_name/user_id/chat_id/... and replace with /cloud_storage/...
    # Pattern matches: s3://[bucket]/[user_id]/[chat_id]/ and replaces with /cloud_storage/
    pattern = r's3://[^/]+/[^/]+/[^/]+/'
    agent_path = re.sub(pattern, '/cloud_storage/', s3_path)
    return agent_path
    