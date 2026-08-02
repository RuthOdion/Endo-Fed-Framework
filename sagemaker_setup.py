"""
Amazon SageMaker environment setup and utilities.

Handles detection of SageMaker environment, dependency installation,
GPU verification, S3 data management, and IAM role configuration
for the endometriosis FL framework.
"""

import os
import subprocess
import sys
from pathlib import Path


def detect_sagemaker_environment() -> dict:
    """
    Detect if running on Amazon SageMaker and gather environment info.

    Returns:
        Dictionary with environment details.
    """
    is_sagemaker = os.path.exists("/opt/ml") or "SM_CHANNEL" in os.environ

    env_info = {
        "is_sagemaker": is_sagemaker,
        "instance_type": os.environ.get("SM_INSTANCE_TYPE", "unknown"),
        "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "python_version": sys.version,
        "notebook_instance": os.environ.get("NOTEBOOK_INSTANCE_NAME", ""),
    }

    if is_sagemaker:
        print("Running on Amazon SageMaker")
        print(f"  Instance type: {env_info['instance_type']}")
        print(f"  Region: {env_info['region']}")
    else:
        print("Not running on SageMaker (local environment)")

    return env_info


def install_dependencies() -> bool:
    """
    Install required Python dependencies from requirements.txt.

    Returns:
        True if installation succeeded.
    """
    requirements_path = Path(__file__).parent / "requirements.txt"

    if not requirements_path.exists():
        print(f"requirements.txt not found at: {requirements_path}")
        return False

    print("Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_path),
            "--quiet",
        ])
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Dependency installation failed: {e}")
        return False


def check_gpu() -> dict:
    """
    Check GPU availability and configuration.

    Returns:
        Dictionary with GPU information.
    """
    gpu_info = {
        "available": False,
        "device_name": "",
        "memory_gb": 0,
        "cuda_version": "",
    }

    # Check NVIDIA GPU via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            parts = output.split(",")
            gpu_info["available"] = True
            gpu_info["device_name"] = parts[0].strip() if len(parts) > 0 else ""
            if len(parts) > 1:
                mem_str = parts[1].strip().replace("MiB", "").strip()
                try:
                    gpu_info["memory_gb"] = round(float(mem_str) / 1024, 1)
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check TensorFlow GPU detection
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        gpu_info["tf_gpu_count"] = len(gpus)
        if gpus:
            gpu_info["available"] = True
    except ImportError:
        gpu_info["tf_gpu_count"] = 0

    if gpu_info["available"]:
        print(f"GPU detected: {gpu_info['device_name']} ({gpu_info['memory_gb']} GB)")
    else:
        print("No GPU detected. Running on CPU (training will be slow).")

    return gpu_info


def get_sagemaker_role() -> str:
    """
    Get the SageMaker execution IAM role.

    Returns:
        IAM role ARN string.
    """
    try:
        import sagemaker
        session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        print(f"SageMaker role: {role}")
        return role
    except Exception as e:
        print(f"Could not get SageMaker role: {e}")
        # Fallback: check environment variable
        role = os.environ.get("SAGEMAKER_ROLE", "")
        if role:
            return role
        return "arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole"


def setup_s3_bucket(bucket_name: str = None, region: str = None) -> str:
    """
    Set up S3 bucket for data and results storage.

    Args:
        bucket_name: Custom bucket name. Auto-generates if None.
        region: AWS region.

    Returns:
        Bucket name.
    """
    try:
        import boto3
    except ImportError:
        print("boto3 not available. Install with: pip install boto3")
        return ""

    if region is None:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if bucket_name is None:
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        bucket_name = f"endometriosis-fl-{account_id}-{region}"

    s3 = boto3.client("s3", region_name=region)

    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"S3 bucket created: s3://{bucket_name}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"S3 bucket already exists: s3://{bucket_name}")
    except Exception as e:
        print(f"Bucket creation error: {e}")

    return bucket_name


def upload_datasets_to_s3(
    local_data_path: str = "data/raw",
    bucket_name: str = None,
    s3_prefix: str = "datasets",
) -> None:
    """
    Upload local datasets to S3 for persistent storage.

    Args:
        local_data_path: Local path to data directory.
        bucket_name: Target S3 bucket.
        s3_prefix: S3 key prefix.
    """
    try:
        import boto3
    except ImportError:
        print("boto3 not available.")
        return

    if bucket_name is None:
        print("Bucket name required.")
        return

    local_path = Path(local_data_path)
    if not local_path.exists():
        print(f"Data path not found: {local_path}")
        return

    s3 = boto3.client("s3")
    uploaded = 0

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.relative_to(local_path)}"
            try:
                s3.upload_file(str(file_path), bucket_name, s3_key)
                uploaded += 1
            except Exception as e:
                print(f"Upload failed for {file_path}: {e}")

    print(f"Uploaded {uploaded} files to s3://{bucket_name}/{s3_prefix}")


def download_datasets_from_s3(
    bucket_name: str,
    s3_prefix: str = "datasets",
    local_data_path: str = "data/raw",
) -> None:
    """
    Download datasets from S3 to local storage.

    Args:
        bucket_name: Source S3 bucket.
        s3_prefix: S3 key prefix.
        local_data_path: Local destination path.
    """
    try:
        import boto3
    except ImportError:
        print("boto3 not available.")
        return

    s3 = boto3.client("s3")
    local_path = Path(local_data_path)
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        paginator = s3.get_paginator("list_objects_v2")
        downloaded = 0

        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative_path = s3_key[len(s3_prefix):].lstrip("/")
                local_file = local_path / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                s3.download_file(bucket_name, s3_key, str(local_file))
                downloaded += 1

        print(f"Downloaded {downloaded} files to {local_path}")
    except Exception as e:
        print(f"Download failed: {e}")


def upload_results_to_s3(
    local_results_path: str = "results",
    bucket_name: str = None,
    s3_prefix: str = "results",
) -> None:
    """
    Upload experiment results to S3.

    Args:
        local_results_path: Local results directory.
        bucket_name: Target S3 bucket.
        s3_prefix: S3 key prefix.
    """
    try:
        import boto3
    except ImportError:
        print("boto3 not available.")
        return

    if bucket_name is None:
        print("Bucket name required.")
        return

    local_path = Path(local_results_path)
    if not local_path.exists():
        print(f"Results path not found: {local_path}")
        return

    s3 = boto3.client("s3")
    uploaded = 0

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.relative_to(local_path)}"
            try:
                s3.upload_file(str(file_path), bucket_name, s3_key)
                uploaded += 1
            except Exception as e:
                print(f"Upload failed for {file_path}: {e}")

    print(f"Uploaded {uploaded} result files to s3://{bucket_name}/{s3_prefix}")


def setup_sagemaker() -> dict:
    """
    Complete SageMaker environment setup.

    Performs:
    1. Environment detection
    2. Dependency installation
    3. GPU verification
    4. S3 bucket setup

    Returns:
        Setup information dictionary.
    """
    print("\n" + "=" * 60)
    print("ENDOMETRIOSIS FL FRAMEWORK - SAGEMAKER SETUP")
    print("=" * 60)

    # Step 1: Detect environment
    print("\n[Step 1] Detecting environment...")
    env_info = detect_sagemaker_environment()

    # Step 2: Install dependencies
    print("\n[Step 2] Installing dependencies...")
    install_success = install_dependencies()

    # Step 3: Check GPU
    print("\n[Step 3] Checking GPU...")
    gpu_info = check_gpu()

    # Step 4: Setup S3 (only on SageMaker)
    bucket_name = ""
    if env_info["is_sagemaker"]:
        print("\n[Step 4] Setting up S3...")
        bucket_name = setup_s3_bucket()

    setup_info = {
        "environment": env_info,
        "dependencies_installed": install_success,
        "gpu": gpu_info,
        "s3_bucket": bucket_name,
    }

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print(f"  Environment: {'SageMaker' if env_info['is_sagemaker'] else 'Local'}")
    print(f"  GPU available: {gpu_info['available']}")
    print(f"  Dependencies: {'OK' if install_success else 'FAILED'}")
    print("=" * 60)

    return setup_info


if __name__ == "__main__":
    setup_sagemaker()
