from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Union


class OptionType(Enum):
    """Enum class for the different types of options that can be used in the UI."""

    ACTION = auto()  # A static option that can be selected directly
    EDITABLE = auto()  # An option where the user can input custom text
    TOGGLE = auto()  # A toggle option that can switch between multiple values


@dataclass
class MenuItem:
    """Container for the menu items that are displayed in the UI."""

    type: OptionType
    description: str
    default: Any = None
    choices: Optional[Union[List, Any]] = None


@dataclass
class AppDTO:
    """
    Data transfer object for transferring data between the UI and the backend.

    Attributes
    return_value_only : bool
        If True, returns the raw result (string or chosen_value).
        If False, returns a dict {chosen_label: chosen_value}.
    start_index : int
        The index (or startposition) at which to start the selector. Default is 0.
    last_config_choice : str
        The key of the last selected config option.
        This option is added to prevent ambiguity, as some keys might be shared among config options.
        Eg: "search" might be present in both config:rag and config:huggingface.
    """

    return_value_only: bool = False
    start_index: int = 0
    last_config_choice: str = ""
