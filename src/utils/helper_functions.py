from typing import List, Tuple
import re
import gc

import torch


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


def extract_markdown(md_string: str) -> List[Tuple[str, str]]:
    """
    Extract language and code blocks from a markdown string.
    """
    pattern = r"```(\w+)([\s\S]*?)```"
    return [
        (match[0].strip(), match[1].strip()) for match in re.findall(pattern, md_string)
    ]
