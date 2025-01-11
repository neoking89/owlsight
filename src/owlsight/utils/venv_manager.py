import os
import sys
from typing import Any, List, Union, Generator
import venv
from contextlib import contextmanager
import subprocess
import tempfile
from pathlib import Path

from owlsight.utils.helper_functions import os_is_windows, force_delete
from owlsight.utils.logger import logger


@contextmanager
def create_venv(pyenv_path: Union[str, Path]) -> Generator[Path, None, None]:
    """
    Context manager to create and manage a Python virtual environment.
    Creates a complete virtual environment with all necessary files.

    Parameters
    ----------
    pyenv_path : Union[str, Path]
        The path where the virtual environment will be created.

    Yields
    ------
    Generator[Path, None, None]
        Generator with Path to the pip executable within the created virtual environment.
    """
    pyenv_path = Path(pyenv_path)
    
    # Remove existing venv if it's invalid
    if pyenv_path.exists():
        activate_script = pyenv_path / ('Scripts' if os_is_windows() else 'bin') / 'activate'
        if not activate_script.exists():
            logger.warning(f"Found invalid virtual environment at {pyenv_path}. Recreating...")
            force_delete(pyenv_path)
    
    # Create the virtual environment with all necessary files
    builder = venv.EnvBuilder(
        system_site_packages=False,
        clear=True,
        with_pip=True,
        upgrade_deps=True  # Upgrade pip and setuptools to latest version
    )
    builder.create(pyenv_path)
    
    # Get pip path
    pip_path = pyenv_path / ('Scripts' if os_is_windows() else 'bin') / ('pip.exe' if os_is_windows() else 'pip')
    
    # Create necessary directories
    lib_path = get_lib_path(pyenv_path)
    lib_path.mkdir(parents=True, exist_ok=True)
    
    yield pip_path


def in_venv() -> bool:
    """
    Check if the current Python process is running inside a virtual environment.

    Returns
    -------
    bool
        True if the current process is running inside a virtual environment, False otherwise.
    """
    return hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)


def get_lib_path(pyenv_path: Union[str, Path]) -> Path:
    """
    Get the path to the lib directory within the virtual environment.
    Creates the directory if it doesn't exist.

    Parameters
    ----------
    pyenv_path : Union[str, Path]
        The path to the (virtual) python environment.

    Returns
    -------
    Path
        The path to the lib directory.
    """
    pyenv_path = Path(pyenv_path)
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    
    if os_is_windows():
        lib_path = pyenv_path / 'Lib' / 'site-packages'
    else:
        # Linux/WSL path structure
        lib_path = pyenv_path / 'lib' / python_version / 'site-packages'
    
    # Create the directory if it doesn't exist
    lib_path.mkdir(parents=True, exist_ok=True)
    
    return lib_path


def get_python_executable(pyenv_path: Union[str, Path]) -> Path:
    """
    Get the path to the Python executable within the virtual environment.

    Parameters
    ----------
    pyenv_path : Union[str, Path]
        The path to the virtual environment.

    Returns
    -------
    Path
        The path to the Python executable.
    """
    pyenv_path = Path(pyenv_path)
    return pyenv_path / ('Scripts' if os_is_windows() else 'bin') / ('python.exe' if os_is_windows() else 'python')


def get_pyenv_path() -> Path:
    """
    Get the path to the current (virtual) python environment.
    Creates a new virtual environment if one doesn't exist.

    Returns
    -------
    Path
        The path to the current (virtual) python environment.
    """
    # First check if VIRTUAL_ENV environment variable is set
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        return Path(venv_path)
        
    # Look for .venv or venv in the current directory
    current_dir = Path.cwd()
    for venv_dir in ['.venv', 'venv']:
        potential_venv = current_dir / venv_dir
        if potential_venv.exists():
            # Check if it's a valid venv by looking for the activate script
            activate_script = potential_venv / ('Scripts' if os_is_windows() else 'bin') / 'activate'
            if activate_script.exists():
                return potential_venv
            else:
                logger.warning(f"Found {venv_dir} directory but it appears to be invalid. Creating new environment...")
                force_delete(potential_venv)
    
    # Create new virtual environment in .venv
    venv_path = current_dir / '.venv'
    logger.info(f"Creating new virtual environment in {venv_path}")
    venv.create(venv_path, with_pip=True)
    return venv_path


def get_pip_path(pyenv_path: Union[str, Path]) -> Path:
    """
    Get the path to the pip executable within the (virtual) python environment.

    Parameters
    ----------
    pyenv_path : Union[str, Path]
        The path to the (virtual) python environment.

    Returns
    -------
    Path
        The path to the pip executable.
    """
    pyenv_path = Path(pyenv_path)
    return pyenv_path / ('Scripts' if os_is_windows() else 'bin') / ('pip.exe' if os_is_windows() else 'pip')


def get_temp_dir(suffix: str) -> Path:
    """
    Get an appropriate temporary directory path that the user has write permissions for.

    Parameters
    ----------
    suffix : str
        The suffix to be appended to the temporary directory path.
        e.g., ".owlsight_temp"

    Returns
    -------
    Path
        The path to a writable temporary directory
    """
    temp_dir = Path(tempfile.gettempdir()) / suffix
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def install_python_modules(
    module_names: Union[str, List[str]], 
    pip_path: Union[str, Path], 
    target_dir: Union[str, Path], 
    *args: Any
) -> bool:
    """
    Install one or more Python modules using pip into a specified directory and add it to sys.path.

    Parameters
    ----------
    module_names : Union[str, List[str]]
        The name of the module(s) to install. Can be a single module as a string or a list of modules.
    pip_path : Union[str, Path]
        The path to the pip executable.
    target_dir : Union[str, Path]
        The directory where the module(s) should be installed.
    *args : Any
        Additional arguments to pass to the pip install command (e.g., --extra-index-url).

    Returns
    -------
    bool
        True if all installations are successful, False otherwise.
    """
    target_dir_str = str(Path(target_dir))
    pip_path_str = str(Path(pip_path))
    
    # Convert module_names to a list if it's a string
    if isinstance(module_names, str):
        module_names = [name.strip() for name in module_names.split(" ")]


    # Install each module separately to match test expectations
    for module in module_names:
        try:
            cmd = [pip_path_str, "install", "--target", target_dir_str, module, *args]
            subprocess.check_call(cmd)
            logger.info(f"Successfully installed {module} into {target_dir}")
            
            # Add the target directory to sys.path if not already there
            if target_dir_str not in sys.path:
                sys.path.insert(0, target_dir_str)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install modules: {e}")
            return False

    return True