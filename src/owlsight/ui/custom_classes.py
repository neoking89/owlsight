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
