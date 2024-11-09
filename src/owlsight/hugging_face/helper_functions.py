# src/models/hugging_face/helper_functions.py

"""helper functions for huggingface models"""

import os
from typing import Iterable, Optional, Union, Dict
import requests
import subprocess

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from tqdm import tqdm


from owlsight.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)

MODELHUB_PREFIX = "https://huggingface.co/"


def get_model_gen(
    filter_by: Union[str, Iterable[str], None] = None,
    sort_by: str = "downloads",
    top_n: int = 10,
    include_metadata: bool = False,
    search: Optional[str] = None,
    hf_api: Optional[HfApi] = None,
) -> Iterable[ModelInfo]:
    """
    Get a generator of models that match the given criteria.

    Parameters
    ----------
    filter_by : Union[str, Iterable[str], None], optional
        A string or list of strings to filter the models by. Defaults to None.
    sort_by : str, Literal["downlads, "likes"]
        The attribute to sort by. Defaults to "downloads".
    top_n : int, optional
        The number of models to return. Defaults to 10.
    include_metadata : bool, optional
        Whether to add extra metadata to the model. Defaults to False.
    search : str, optional
        A string to filter the models by during searching. Defaults to None.
    hf_api : HfApi, optional
        An instance of HuggingFace API to use. If None, a new instance will be created. Defaults to None.

    Returns
    -------
    Iterable[ModelInfo]
        A generator of models that match the given criteria.
    """
    if hf_api is None:
        hf_api = HfApi()
    model_gen = hf_api.list_models(
        filter=filter_by,
        search=search,
        limit=top_n,
        cardData=include_metadata,
        sort=sort_by,
        direction=-1,
    )

    return model_gen


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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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


def _get_hf_model_data(model_info: "ModelInfo") -> Dict[str, str]:
    if model_info.lastModified is None:
        last_modified = "N/A"
    else:
        last_modified = (
            model_info.lastModified.split("T")[0]
            if isinstance(model_info.lastModified, str)
            else str(model_info.lastModified.date())
        )
    model_data = {
        "last modified": last_modified,
        "downloads": model_info.downloads,
        "likes": model_info.likes,
        "url": os.path.join(MODELHUB_PREFIX, model_info.modelId),
    }
    return model_data
