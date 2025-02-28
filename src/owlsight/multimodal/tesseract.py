"""This module contains a function to check if Tesseract is installed on the system."""

import ctypes
import os
import platform
import subprocess
import traceback
from typing import Union
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from owlsight.utils.helper_functions import os_is_windows
from owlsight.utils.logger import logger

try:
    import pytesseract
except ImportError:
    logger.warning("pytesseract is not installed. Please install it using 'pip install pytesseract'.")


def setup_tesseract() -> str:
    """Initialize Tesseract. Return the path to the Tesseract executable."""
    if not os_is_windows():
        logger.error("Automatic installation is currently only supported on Windows.")
        return ""
    TESSERACT_LOCATION = find_tesseract_installation()
    if TESSERACT_LOCATION:
        logger.info("Tesseract is installed at %s.", TESSERACT_LOCATION)
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_LOCATION
    else:
        logger.info("Tesseract not found. Downloading and installing...")
        download_and_install_tesseract()
        TESSERACT_LOCATION = find_tesseract_installation()
        if TESSERACT_LOCATION:
            logger.info("Tesseract is now installed at %s.", TESSERACT_LOCATION)
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_LOCATION
        else:
            url = "https://github.com/UB-Mannheim/tesseract/wiki"
            logger.error("Failed to install Tesseract! You may download it manually from %s", url)
            raise RuntimeError(f"Failed to install Tesseract! You may download it manually from {url}")

    return TESSERACT_LOCATION


def download_and_install_tesseract():
    """Download and install Tesseract if not found."""
    output_dir = "."
    downloaded_file = download_tesseract(output_dir)
    if downloaded_file:
        exe_path = os.path.join(output_dir, downloaded_file)
        logger.info("Installing Tesseract from %s...", exe_path)
        install_tesseract(exe_path)
        os.remove(exe_path)
        logger.info("Removed the Tesseract installer %s", exe_path)


def find_tesseract_installation() -> str:
    """Find and return the path to the Tesseract executable."""
    tesseract_path = ""
    if os_is_windows():
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    else:
        common_paths = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]

    for path in common_paths:
        if os.path.exists(path):
            return path
    try:
        subprocess.run(
            ["tesseract", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return "tesseract"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return tesseract_path


def download_tesseract(output_dir=".", timeout=10) -> Union[str, None]:
    """
    Download Tesseract .exe from the specified GitHub page and show a progress bar during download.

    Parameters
    ----------
    output_dir : str
        The directory to save the downloaded file to.

    Returns
    -------
    Union[str, None]
        The name of the downloaded file, or None if the download failed.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(
            "BeautifulSoup is not installed. Please install it using 'pip install beautifulsoup4'."
        ) from e
    tesseract_url = "https://github.com/UB-Mannheim/tesseract/wiki"
    try:
        res = requests.get(tesseract_url, timeout=timeout)
        res.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
        soup = BeautifulSoup(res.text, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if "tesseract-ocr-w64-setup" in href and href.endswith(".exe"):
                logger.info("Found an exe file online called %s", href)
                file_name = os.path.basename(urlparse(href).path)
                file_path = os.path.join(output_dir, file_name)
                if not os.path.exists(file_path):
                    try:
                        logger.info("Downloading %s...", file_name)
                        with requests.get(href, stream=True, timeout=timeout) as exe_response:
                            exe_response.raise_for_status()
                            total_size_in_bytes = int(exe_response.headers.get("content-length", 0))
                            block_size = 1024  # 1 Kibibyte
                            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
                            with open(file_path, "wb") as file:
                                for data in exe_response.iter_content(block_size):
                                    progress_bar.update(len(data))
                                    file.write(data)
                            progress_bar.close()
                        logger.info("Successfully downloaded %s", file_name)
                        return file_name
                    except Exception as e:
                        logger.error("Download interrupted, removing partially downloaded file.")
                        os.remove(file_path)  # Remove the partially downloaded file
                        raise e
                else:
                    logger.info("File %s already exists.", file_path)
                    return os.path.basename(file_path)
        logger.error("Failed to find the download link.")
    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred: {err}")
    return None


def is_admin():
    """Check if the script is running with administrative privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception as e:
        logger.error(f"Error checking admin status: {e}")
        return False


def install_tesseract(exe_path: str):
    """Install Tesseract by running the downloaded .exe installer."""
    if platform.system() == "Windows":
        if is_admin():
            try:
                subprocess.run([exe_path, "/S"], check=True)
            except subprocess.CalledProcessError:
                logger.error(
                    "CalledProcessError while installing Tesseract:\n%s",
                    traceback.format_exc(),
                )
            except Exception:
                logger.error(
                    "An unexpected error occurred during Tesseract installation:\n%s",
                    traceback.format_exc(),
                )
        else:
            raise OSError("The script is not running with administrative privileges. Please run as administrator.")
    else:
        logger.error("Automatic installation is currently only supported on Windows.")
