"""helper functions for huggingface models"""

import os
import numpy as np
from typing import Iterable, Optional, Union, Dict, List
import requests
import subprocess
from datetime import datetime, timezone
from functools import lru_cache

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from tqdm import tqdm

from owlsight.utils.helper_functions import validate_input_params
from owlsight.utils.logger import logger

MODELHUB_PREFIX = "https://huggingface.co/"


@lru_cache(maxsize=128)
def calculate_days_ago_created(time_created: datetime) -> int:
    """
    Calculate the number of days since a given time.
    """
    now_utc = datetime.now(timezone.utc)
    time_difference = now_utc - time_created
    days_difference = time_difference.days

    return days_difference


def engagement_score(likes: int, downloads: int, created_days_ago: int) -> float:
    """
    Calculate an engagement score based on likes, downloads and days since creation.

    This function computes an engagement score that takes into account
    the number of likes, downloads and days since creation for a model, and the number of days ago a model was created.
    It applies stronger penalties for very low engagement (low likes and downloads),
    and uses logarithmic scaling to balance disparities between likes and downloads.

    Parameters
    ----------
    likes : int
        The number of likes for the model.
    downloads : int
        The number of downloads for the model.
    created_days_ago : int
        The number of days since the model was created.

    Returns
    -------
    float
        The calculated engagement score.
    """
    # Constants
    C: int = 10  # Offset to avoid zero division
    download_threshold = 50  # Threshold below which downloads are heavily penalized

    # Base score calculation with penalty for very low downloads and likes
    if likes == 0 and downloads == 0:
        base_score = 0  # No engagement, so score should be zero
    else:
        # Logarithmic scaling for likes and downloads to reduce impact of low values
        base_score = (np.log1p(likes) / (np.log1p(downloads + C))) ** 1.2

    # Download factor with clamping and penalty for low downloads
    download_factor: float = np.log(downloads + C) / np.log(1000)
    download_factor = min(max(download_factor, 0.1), 1)  # Clamp between 0.1 and 1

    # Penalize very low downloads
    low_download_penalty = 0.2 if downloads < download_threshold else 1

    # Adjustment for zero likes
    zero_likes_penalty: float = 0.5 if likes == 0 else 1

    days_ago_score = calculate_days_ago_score(created_days_ago)

    # Final score calculation
    score: float = (base_score * download_factor * zero_likes_penalty * low_download_penalty) + days_ago_score

    return score


def _make_hashable(value):
    """Convert potentially unhashable types to hashable ones for caching."""
    if isinstance(value, (list, set)):
        return tuple(sorted(_make_hashable(x) for x in value))
    if isinstance(value, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in value.items()))
    return value


@lru_cache(maxsize=128)
def _cached_get_model_list(
    filter_by: Optional[tuple] = None, search: Optional[str] = None, include_metadata: bool = False, **kwargs
) -> tuple:
    """Cached inner function that returns all models without top_n slicing."""
    hf_api = HfApi()

    # Convert filter_by back to list if it's not None
    filter_by_list = list(filter_by) if filter_by is not None else None

    validate_input_params(hf_api.list_models, kwargs)
    model_gen = hf_api.list_models(
        filter=filter_by_list,
        search=search,
        cardData=include_metadata,
        **kwargs,
    )

    model_info_list = []

    for model_info in model_gen:
        days_ago_created = calculate_days_ago_created(model_info.created_at)
        score = engagement_score(model_info.likes, model_info.downloads, days_ago_created)
        model_info.engagement_score = score
        model_info_list.append(model_info)

    model_info_list.sort(key=lambda x: x.engagement_score, reverse=True)

    # Convert to tuple for caching
    return tuple(model_info_list)


def get_model_list(
    filter_by: Union[str, Iterable[str], None] = None,
    top_n: Optional[int] = None,
    include_metadata: bool = False,
    search: Optional[str] = None,
    **kwargs,
) -> List[ModelInfo]:
    """
    Get a list of models that match the given criteria, with caching support.

    Parameters
    ----------
    filter_by : Union[str, Iterable[str], None], optional
        A string or list of strings to filter the models by. Defaults to None.
    top_n : int, optional
        The number of models to return. Defaults to None.
    include_metadata : bool, optional
        Whether to add extra metadata to the model. Defaults to False.
    search : str, optional
        A string to filter the models by during searching. Defaults to None.
    hf_api : HfApi, optional
        An instance of HuggingFace API to use. If None, a new instance will be created. Defaults to None.

    Returns
    -------
    List[ModelInfo]
        A list of models that match the given criteria.
    """
    # Convert filter_by to hashable type (tuple) for caching
    hashable_filter_by = _make_hashable(filter_by) if filter_by is not None else None

    # Convert kwargs to hashable format
    hashable_kwargs = _make_hashable(kwargs)

    # Get full cached result
    cached_results = _cached_get_model_list(
        filter_by=hashable_filter_by, search=search, include_metadata=include_metadata, **dict(hashable_kwargs)
    )

    # Convert back to list and apply top_n if specified
    model_info_list = list(cached_results)
    if top_n:
        model_info_list = model_info_list[:top_n]

    return model_info_list


def download_huggingface_model(model_name: str, save_path: str, chunk_size: int = 1024) -> None:
    """
    Construct the URL to download the model

    Parameters
    ----------
    model_name : str
        The name of the model to download
    save_path : str
        The path where the model will be saved
    chunk_size : int
        The size of each chunk to download

    Returns
    -------
    None
    """
    base_url = f"https://huggingface.co/{model_name}/resolve/main/"
    file_names = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]

    # Create the folder where the model will be saved
    os.makedirs(save_path, exist_ok=True)

    for file_name in file_names:
        # Download each file
        url = base_url + file_name
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            # Show progress bar
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(os.path.join(save_path, file_name), "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        else:
            logger.error("Failed to download %s due to %s", url, response.status_code)


def show_model_memory(model_name: str) -> Optional[str]:
    """
    Executes a shell command to estimate the memory usage of a given model.

    Args:
        model_name (str): Name of the model to estimate memory for.

    Returns:
        Optional[str]: The output of the memory estimation command, or None if an error occurs.
    """
    command = [
        "accelerate",
        "estimate-memory",
        model_name,
        "--library_name",
        "transformers",
    ]

    try:
        # Execute the command and capture the output
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",  # Specify UTF-8 encoding
            errors="ignore",  # Ignore decoding errors
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while estimating memory: {e.stderr}")
        return None
    except UnicodeDecodeError as e:
        print(f"A Unicode decoding error occurred: {e}")
        return None


def _get_hf_model_data(model_info: ModelInfo) -> Dict[str, str]:
    days_difference = calculate_days_ago_created(model_info.created_at)
    model_data = {
        "engagement score": np.round(model_info.engagement_score, 4),
        "days ago created": days_difference,
        "downloads": model_info.downloads,
        "likes": model_info.likes,
        "url": os.path.join(MODELHUB_PREFIX, model_info.modelId),
    }

    return model_data


def calculate_days_ago_score(created_days_ago, weight=0.01, N=0.3):
    """
    Optimized version with better score distribution:
    Reduced weight to prevent scores from decaying too quickly

    Parameters
    ----------
    created_days_ago : int
        The number of days since the model was created.
    weight : float, optional
        The weight factor for the exponential decay. Defaults to 0.01.
    N : float, optional
        Increase this value to boost the score. Defaults to 0.25.
    """
    if not created_days_ago:
        return 0.0

    return np.exp(-(weight * (1 + created_days_ago))) * N
