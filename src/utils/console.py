from enum import Enum, auto
from typing import List, Dict, Tuple, Union, Any
import sys

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, Window, HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings

sys.path.append(".")
from src.utils.constants import COLOR_CODES


# Enum for different option types
class OptionType(Enum):
    SINGLE = auto()  # A static option that can be selected
    EDITABLE = auto()  # An option where the user can input text
    TOGGLE = auto()  # A toggle option that can switch between multiple values


class Selector:
    def __init__(self, options_dict: Dict[str, Union[None, str, List[Any]]]) -> None:
        """
        Initializes the Selector with the given options.

        :param options_dict: A dictionary where the key is the option label and the value defines
                             the type of option. None is for static options, an empty string for editable input,
                             and a list (e.g., [True, False], [1, 2, 3]) for toggleable options.
        """
        self.options: List[Tuple[str, OptionType]] = []
        self.current_index: int = 0
        self.selected: bool = False
        self.user_inputs: Dict[str, str] = {}  # To store user input for editable fields
        self.toggle_values: Dict[str, Any] = (
            {}
        )  # To store current value for toggleable fields
        self.toggle_choices: Dict[str, List[Any]] = (
            {}
        )  # To store possible toggle values

        # Parse the options dictionary to categorize the options
        for key, value in options_dict.items():
            if value is None:
                self.options.append((key, OptionType.SINGLE))
            elif isinstance(value, list):  # Toggleable option
                self.options.append((key, OptionType.TOGGLE))
                self.toggle_choices[key] = value  # Store available toggle values
                self.toggle_values[key] = value[0]  # Set default to the first value
            elif value == "":
                self.options.append((key, OptionType.EDITABLE))
                self.user_inputs[key] = ""  # Initialize editable field as empty

    def get_formatted_options(self) -> List[Tuple[str, str]]:
        """
        Formats the options for display with an arrow pointing to the current selection.

        :return: A list of formatted text representing each option.
        """
        formatted = []
        for i, (label, opt_type) in enumerate(self.options):
            if opt_type == OptionType.SINGLE:
                formatted.append(
                    ("", f"{'>' if i == self.current_index else ' '} {label}\n")
                )
            elif opt_type == OptionType.TOGGLE:
                current_value = self.toggle_values[label]
                formatted.append(
                    (
                        "",
                        f"{'>' if i == self.current_index else ' '} {label}: {current_value}\n",
                    )
                )
            elif opt_type == OptionType.EDITABLE:
                formatted.append(
                    (
                        "",
                        f"{'>' if i == self.current_index else ' '} {label} {self.user_inputs[label]}\n",
                    )
                )
        return formatted

    def cycle_toggle_left(self, label: str) -> None:
        """Cycle to the previous value in the toggle list."""
        choices = self.toggle_choices[label]
        current_value = self.toggle_values[label]
        current_index = choices.index(current_value)
        self.toggle_values[label] = choices[
            (current_index - 1) % len(choices)
        ]  # Move left

    def cycle_toggle_right(self, label: str) -> None:
        """Cycle to the next value in the toggle list."""
        choices = self.toggle_choices[label]
        current_value = self.toggle_values[label]
        current_index = choices.index(current_value)
        self.toggle_values[label] = choices[
            (current_index + 1) % len(choices)
        ]  # Move right


def get_user_choice(
    options_dict: Dict[str, Union[None, str, List[Any]]],
    return_value_only: bool = True,
) -> str | Dict[str, Any]:
    """
    Runs the command-line interface that allows the user to select or input options.

    :param options_dict: The options to display to the user.
    :return: The selected or inputted option as a string.
    """
    selector = Selector(options_dict)

    def get_text_fragments() -> List[Tuple[str, str]]:
        return selector.get_formatted_options()

    text_area = Window(
        content=FormattedTextControl(get_text_fragments),
        always_hide_cursor=True,
    )

    kb = KeyBindings()

    @kb.add("up")
    def up_key(event) -> None:
        """Move the selection up."""
        selector.current_index = (selector.current_index - 1) % len(selector.options)

    @kb.add("down")
    def down_key(event) -> None:
        """Move the selection down."""
        selector.current_index = (selector.current_index + 1) % len(selector.options)

    @kb.add("enter")
    def enter_key(event) -> None:
        """Mark the option as selected and exit."""
        selector.selected = True
        event.app.exit()

    @kb.add("left")
    def left_key(event) -> None:
        """Cycle to the previous value if the current option is a toggleable field."""
        current_option, opt_type = selector.options[selector.current_index]
        if opt_type == OptionType.TOGGLE:
            selector.cycle_toggle_left(current_option)
        event.app.invalidate()

    @kb.add("right")
    def right_key(event) -> None:
        """Cycle to the next value if the current option is a toggleable field."""
        current_option, opt_type = selector.options[selector.current_index]
        if opt_type == OptionType.TOGGLE:
            selector.cycle_toggle_right(current_option)
        event.app.invalidate()

    @kb.add("c-c", eager=True)
    @kb.add("c-q")
    def exit_(event) -> None:
        """Exit the application gracefully on Ctrl+C or Ctrl+Q."""
        event.app.exit()

    @kb.add("<any>")
    def handle_input(event) -> None:
        """Handle character input for editable fields."""
        current_option, opt_type = selector.options[selector.current_index]
        if opt_type == OptionType.EDITABLE:
            key = event.key_sequence[0].key
            if key == " " and not selector.user_inputs[current_option]:
                return  # Prevent leading space
            if len(key) == 1 and key.isprintable():
                selector.user_inputs[current_option] += key
                event.app.invalidate()

    @kb.add("backspace")
    def handle_backspace(event) -> None:
        """Handle backspace for editable fields."""
        current_option, opt_type = selector.options[selector.current_index]
        if opt_type == OptionType.EDITABLE and selector.user_inputs[current_option]:
            selector.user_inputs[current_option] = selector.user_inputs[current_option][
                :-1
            ]
            event.app.invalidate()

    layout = Layout(HSplit([text_area]))

    application = Application(
        layout=layout,
        key_bindings=kb,
    )

    application.run()

    # Return the selected option as a string based on its type
    if selector.selected:
        selected_option, opt_type = selector.options[selector.current_index]
        if opt_type == OptionType.EDITABLE:
            # Return the user input for editable fields
            return (
                {selected_option: selector.user_inputs[selected_option]}
                if not return_value_only
                else selector.user_inputs[selected_option]
            )
        elif opt_type == OptionType.TOGGLE:
            # Return the current value for toggle fields
            return (
                {selected_option: selector.toggle_values[selected_option]}
                if not return_value_only
                else selector.toggle_values[selected_option]
            )
        else:
            # Return the label for static single-select fields
            return selected_option

    return ""


def print_colored(text: str, color: str) -> None:
    """
    Print text in the specified color.

    Parameters
    ----------
    text : str
        The text to print.
    color : str
        The color to print the text. Options are 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', and 'reset'.

    Raises
    ------
    ValueError
        If the provided color is not valid.
    """
    if color not in COLOR_CODES:
        raise ValueError(
            f"Invalid color '{color}'. Valid options are: {', '.join(COLOR_CODES.keys())}"
        )

    color_code = COLOR_CODES[color]
    reset_code = COLOR_CODES["reset"]

    # Print the text with the selected color and reset the color afterward
    print(f"{color_code}{text}{reset_code}")


if __name__ == "__main__":
    print_colored("make a choice:", "cyan")
    # Example options dictionary
    options = {
        "You are a:": "",  # Editable input
        "apple": None,  # Static option
        "pear": None,  # Static option
        "banana": None,  # Static option
        "Is it ripe?": [True, False],  # Toggleable option
        "Days in sun": [1, 2, 3],  # Toggleable option
    }

    result = get_user_choice(options)
    print(result)
