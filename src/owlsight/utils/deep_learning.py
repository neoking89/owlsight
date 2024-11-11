import gc
import subprocess
import torch

from owlsight.utils.logger import logger




def free_memory():
    """Free up memory and reset stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def print_memory_stats(device: torch.device):
    """Print two different measures of GPU memory usage."""
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB"
    )
    print(
        f"Max memory reserved: { torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB"
    )


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

    cuda_version = None
    try:
        output_cuda = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        cuda_version = output_cuda[
            output_cuda.find("release")
            + len("release")
            + 1 : output_cuda.find(",", output_cuda.find("release"))
        ]
        logger.info("CUDA %s is installed.", cuda_version)
    except subprocess.CalledProcessError:
        logger.warning(
            "Warning: CUDA-capable GPU is available, but CUDA is not installed. Please install CUDA."
        )
    except Exception as e:
        logger.error("%s", e)

    if torch.cuda.is_available():
        logger.info("CUDA-capable GPU is available for PyTorch. You are all set!")
    else:
        logger.warning(
            "Cuda is currently unavailable. This could be expected if no GPU is available. If not, please visit 'https://pytorch.org/get-started/locally/' to install a compatible version.\nrun command 'pip uninstall torch torchvision torchaudio' and find run the right version of PyTorch for your CUDA version.",
        )


def log_reserved_memory():
    """Logs the reserved memory on the GPU and CPU."""
    if torch.cuda.is_available():
        gpu_reserved = torch.cuda.memory_reserved(0)
        gpu_free = torch.cuda.max_memory_allocated(0) - torch.cuda.memory_allocated(0)
        logger.info("GPU Memory - Reserved: %s, Free: %s", gpu_reserved, gpu_free)
    else:
        logger.info("CUDA not available. GPU memory stats cannot be logged.")

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
        _ = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, device="cuda")
        print("bfloat16 is supported on your GPU.")
        return True
    except Exception as e:
        print(f"bfloat16 is not supported on your GPU: {e}")
        return False


def calculate_memory_for_model(n_bilion_parameters: int, n_bit: int = 32) -> float:
    """
    Calculate the memory required for a model in GB.

    Parameters:
    n_bilion_parameters (int): The number of parameters in the model in billions.
    n_bit (int): The number of bits used to represent the model parameters. Default is 32. Quantized models use 16/8/4 bits.
    """
    return ((n_bilion_parameters * 4) / (32 / n_bit)) * 1.2


def calculate_available_vram() -> float:
    """
    Calculate the available VRAM on the GPU in GB.
    """
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
        available_memory_gb = available_memory / 1024**3
        return available_memory_gb
    else:
        logger.warning("CUDA not available. Cannot calculate available VRAM.")
        return 0.0


def calculate_max_parameters_per_dtype():
    """
    Calculate the maximum number of parameters that can be run on the GPU
    for different data types (32-bit, 16-bit, 8-bit, 4-bit).
    """
    available_vram = calculate_available_vram()
    if available_vram > 0:
        logger.info(f"Available VRAM: {available_vram:.2f} GB")

        for bits in [32, 16, 8, 4]:
            max_params = available_vram / calculate_memory_for_model(
                1, bits
            )  # for 1 billion parameters
            logger.info(
                f"Maximum number of billion parameters for {bits}-bit model: {max_params:.2f} billion"
            )
    else:
        logger.warning("No available VRAM to calculate parameters.")


def get_best_device() -> str:
    """
    Check for best device and return the device name.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
