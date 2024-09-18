from typing import List


class ModuleNotFoundInVenvError(Exception):
    """Exception raised when a module is not found in the virtual environment."""

    def __init__(self, module_name: str, venv_path: str, lib_contents: List[str]):
        self.module_name = module_name
        self.venv_path = venv_path
        self.lib_contents = lib_contents
        self.message = f"Module '{module_name}' not found in venv '{venv_path}'. Contents of lib directory: {lib_contents}"
        super().__init__(self.message)


class QuantizationNotSupportedError(Exception):
    """Exception raised when quantization is not supported for a given model."""
