import re
import inspect
import traceback
from typing import Dict, List, Optional
from types import MethodType

from owlsight.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class TransformersArgumentInferer:
    """
    Infer arguments of a method/function from its docstring or signature.
    This assumes the docstring in the style of the HuggingFace Transformers library.
    """

    def __init__(self):
        self._last_returned_value: Optional[Dict[str, dict | str]] = None

    def transform_inferred_arguments_to_text(self, inferred_arguments: Dict[str, dict | str]) -> Optional[str]:
        """
        Transform the dict acquired from self.__call__ to a string.\n
        For example, as placeholder text in an UI application."""
        if not isinstance(inferred_arguments, dict):
            raise TypeError(f"Expected dict, got {type(inferred_arguments)} instead.")

        if not inferred_arguments:
            raise ValueError("No arguments found.")

        if all(isinstance(value, str) for value in inferred_arguments.values()):
            return "\n".join(val for _, val in inferred_arguments.items())

        return "\n".join([f"{arg}: {arg_info['types']}" for arg, arg_info in inferred_arguments.items()])

    def __call__(self, func: MethodType, docstring: Optional[str] = None) -> Dict[str, dict | str]:
        """
        Infer the arguments of the specified function or method.

        Parameters
        ----------
        func : Callable
            The method or function
        docstring : str, optional
            The docstring of the method, by default None
            In some cases, the docstring is defined outside the func, so it can be passed here.

        Returns
        -------
        Dict[str, dict] | List[str]
            A dictionary with the argument names as keys and a dictionary with the argument details as values,
            or a list of obliged argument names.
        """
        if not callable(func):
            raise TypeError(f"Expected a callable (function or method), got {type(func)} instead.")
        if docstring is None:
            docstring = func.__doc__

        if not docstring:
            logger.error("Docstring not found for func %s  Using inspect to infer arguments.", func.__name__)
            signature = inspect.signature(func)
            arguments: List[str] = self._infer_arguments_from_signature(signature)
            return arguments

        try:
            arguments: dict = self._infer_arguments_from_doc_str(docstring)
        except Exception:
            logger.error(
                "Failed to infer arguments from docstring due to:\n%s",
                traceback.format_exc(),
            )
            logger.error("Using inspect to infer arguments.")
            signature = inspect.signature(func)
            arguments = self._infer_arguments_from_signature(signature)

        self._last_returned_value = arguments

        return arguments

    def _infer_arguments_from_doc_str(self, docstring: str) -> dict:
        arg_pattern = re.compile(r"\s*(\w+)\s+\(([^)]+)\):\s*(.*)")

        lines = docstring.strip().split("\n")
        arg_lines = []
        current_arg = None

        for line in lines:
            if re.match(arg_pattern, line):
                if current_arg:
                    arg_lines.append(current_arg)
                current_arg = line
            elif current_arg:
                current_arg += " " + line.strip()

        if current_arg:
            arg_lines.append(current_arg)

        arguments = {}
        for line in arg_lines:
            match = re.match(arg_pattern, line)
            if match:
                arg_name, arg_types, description = match.groups()
                if arg_name in ["args", "kwargs"]:
                    continue
                is_optional = "optional" in arg_types.lower()
                arguments[arg_name] = {
                    "types": arg_types,
                    "optional": is_optional,
                    "description": description,
                }

        return arguments

    def _infer_arguments_from_signature(self, signature: inspect.Signature) -> Dict[str, str]:
        return {name: str(param) for name, param in signature.parameters.items() if name not in "self"}
