import inspect
from typing import Any


def _get_signature(obj: Any) -> str:
    """
    Get the signature of a callable object.

    Parameters:
    -----------
    obj : Any
        Object to get signature for

    Returns:
    --------
    str
        String representation of the object's signature
    """
    try:
        if callable(obj):
            return str(inspect.signature(obj))
        return ""
    except (ValueError, TypeError):
        return "(Unable to retrieve signature)"