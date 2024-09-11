import gc
import subprocess

import torch

from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def clean_gpu(device: torch.device):
    """Free up memory and reset stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def print_memory_stats(device: torch.device):
    """Print two different measures of GPU memory usage."""
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB"
    )
    # reserved (aka 'max memory cached') is the allocated memory plus pre-cached memory
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB")


def calculate_model_size(model) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def check_gpu_and_cuda():
    """Checks if a CUDA-capable GPU is available and if CUDA is installed."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        logger.info(f"GPU found: {gpu}")
        logger.info(
            "CUDA-capable GPU is available and PyTorch is built with CUDA support."
        )
    try:
        output_cuda = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        cuda_version = output_cuda[
            output_cuda.find("release") + len("release") + 1 : output_cuda.find(
                ",", output_cuda.find("release")
            )
        ]
        logger.info("CUDA %s is installed.", cuda_version)
    except subprocess.CalledProcessError:
        logger.warning(
            "Warning: CUDA-capable GPU is available, but CUDA is not installed. Please install CUDA."
        )
    except Exception as e:
        logger.error("%s", e)
        raise e

    # Check if a CUDA-capable GPU is available
    if torch.cuda.is_available():
        logger.info(
            "CUDA-capable GPU is available and PyTorch is built with CUDA support. You are all set!"
        )
    else:
        logger.warning(
            "PyTorch is built without CUDA support for CUDA version %s. Please visit 'https://pytorch.org/get-started/locally/' to install a compatible version.\nrun command 'pip uninstall torch torchvision torchaudio' and find run the right version of PyTorch for your CUDA version.\n",
            cuda_version,
        )


def log_reserved_memory():
    """Logs the reserved memory on the GPU and CPU."""
    if torch.cuda.is_available():
        gpu_reserved = torch.cuda.memory_reserved(0)
        gpu_free = torch.cuda.max_memory_allocated(0) - torch.cuda.memory_allocated(0)
        logger.info("GPU Memory - Reserved: %s, Free: %s", gpu_reserved, gpu_free)
    else:
        logger.info("CUDA not available. GPU memory stats cannot be logged.")

    # Safely retrieving CPU memory stats
    try:
        cpu_stats = torch.cuda.memory_stats()
        cpu_reserved = cpu_stats.get("reserved_host_bytes.all.current", "Not available")
    except AttributeError:
        cpu_reserved = "Not available due to PyTorch version or configuration."

    logger.info("CPU Memory - Reserved: %s", cpu_reserved)


def check_bfloat16_support():
    """
    Check if the current GPU supports bfloat16 data type using PyTorch.

    Returns:
        bool: True if bfloat16 is supported, False otherwise.
    """
    if not torch.cuda.is_available():
        print("No GPU found.")
        return False

    try:
        # Create a tensor with bfloat16 dtype
        tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, device="cuda")
        print("bfloat16 is supported on your GPU.")
        return True
    except Exception as e:
        print(f"bfloat16 is not supported on your GPU: {e}")
        return False
