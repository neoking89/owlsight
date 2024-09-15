import sys
from typing import List

import keyboard

from src.utils.constants import COLOR_CODES


def choose_from_menu(choices: List[str]) -> str:
    """
    Allow the user to choose from a list of options using arrow keys.

    Parameters
    ----------
    choices : List[str]
        The list of options to choose from.

    Returns
    -------
    str
        The selected choice.
    """
    current_selection = 0

    # Display initial options list below current output
    print("\n" * len(choices), end="")  # Reserve space for the options to be displayed
    _display_choices(choices, current_selection)

    while True:
        event = keyboard.read_event(suppress=True)
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == "up" and current_selection > 0:
                current_selection -= 1
                _display_choices(choices, current_selection)
            elif event.name == "down" and current_selection < len(choices) - 1:
                current_selection += 1
                _display_choices(choices, current_selection)
            elif event.name == "enter":
                return choices[current_selection]


def choose_from_prompt_and_menu(
    prompt: str, initial_input: str, choices: List[str]
) -> str:
    """
    Allow the user to type input inline with the prompt and choose from a list of options using arrow keys.

    This function lets the user type inline text, and once the user presses Enter, it switches
    to selecting from the available menu options.

    Parameters
    ----------
    prompt : str
        The prompt message displayed before the editable string.
    initial_input : str
        The initial value to be pre-filled in the input area.
    choices : List[str]
        The list of options from which the user can select using arrow keys.

    Returns
    -------
    str
        The final input if no menu option is selected or the selected menu option.
    """
    user_input = initial_input
    current_selection = -1  # -1 represents the input line
    cursor_position = len(user_input)

    _display_prompt_and_choices(prompt, user_input, choices, current_selection)

    while True:
        event = keyboard.read_event(suppress=True)
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == "up" and current_selection > -1:
                current_selection -= 1
            elif event.name == "down" and current_selection < len(choices) - 1:
                current_selection += 1
            elif event.name == "left" and cursor_position > 0:
                cursor_position -= 1
            elif event.name == "right" and cursor_position < len(user_input):
                cursor_position += 1
            elif event.name == "enter":
                print("\n" * (len(choices) + 1))  # Add proper spacing after enter
                return (
                    user_input
                    if current_selection == -1
                    else choices[current_selection]
                )
            elif event.name == "space" and current_selection == -1:
                user_input = (
                    user_input[:cursor_position] + " " + user_input[cursor_position:]
                )
                cursor_position += 1
            elif (
                event.name == "backspace"
                and cursor_position > 0
                and current_selection == -1
            ):
                user_input = (
                    user_input[: cursor_position - 1] + user_input[cursor_position:]
                )
                cursor_position -= 1
            elif (
                len(event.name) == 1 and current_selection == -1
            ):  # Single character input
                user_input = (
                    user_input[:cursor_position]
                    + event.name
                    + user_input[cursor_position:]
                )
                cursor_position += 1

            _display_prompt_and_choices(prompt, user_input, choices, current_selection)

            # Move cursor to correct position if on input line
            if current_selection == -1:
                sys.stdout.write(f"\033[{len(prompt) + cursor_position + 3}G")
            sys.stdout.flush()


# Private functions for rendering UI
def _display_choices(choices: List[str], current_selection: int) -> None:
    """
    Display the list of choices, highlighting the current selection.

    Parameters
    ----------
    choices : List[str]
        The list of options to display.
    current_selection : int
        The index of the currently selected option.
    """
    # Move the cursor up to where the choices are displayed
    for _ in range(len(choices)):
        sys.stdout.write("\033[F")  # Move cursor up by one line

    # Redraw the choices
    for index, choice in enumerate(choices):
        if index == current_selection:
            sys.stdout.write(f"\r> {choice}\n")  # Highlight current selection
        else:
            sys.stdout.write(f"\r  {choice}\n")  # Display other choices


def _display_prompt_and_choices(
    prompt: str, user_input: str, choices: List[str], current_selection: int
) -> None:
    """
    Display the prompt with inline user input, and list of choices, highlighting the current selection.

    Parameters
    ----------
    prompt : str
        The prompt message shown before the editable string.
    user_input : str
        The current user input text.
    choices : List[str]
        The list of options to display.
    current_selection : int
        The index of the currently selected option or -1 for the input line.
    """
    # Clear the console
    sys.stdout.write("\033[H\033[J")

    # Display prompt with user input
    if current_selection == -1:
        sys.stdout.write(f"> {prompt} {user_input}")
    else:
        sys.stdout.write(f"  {prompt} {user_input}")
    sys.stdout.flush()

    # Display choices
    for index, choice in enumerate(choices):
        if index == current_selection:
            sys.stdout.write(f"\n> {choice}")
        else:
            sys.stdout.write(f"\n  {choice}")

    # Move cursor back to end of user input on the first line
    sys.stdout.write(f"\033[{len(choices)}A\033[{len(prompt) + len(user_input) + 3}G")
    sys.stdout.flush()


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
