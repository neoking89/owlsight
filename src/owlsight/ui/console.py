#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a console-based user interface for selecting and configuring
options using prompt_toolkit, with no highlight around the selected item.
"""

import sys
import traceback
from enum import Enum, auto
from typing import List, Dict, Tuple, Union, Any, Optional

from prompt_toolkit import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import TextArea, Frame
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application.current import get_app
from prompt_toolkit.styles import Style

from owlsight.utils.constants import COLOR_CODES, MENU_KEYS, MAIN_MENU, get_prompt_cache
from owlsight.utils.logger import logger

try:
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
except ImportError:

    class NoConsoleScreenBufferError(Exception):
        """Fallback exception when no console screen buffer is available."""

        pass


class HistoryCompleter(Completer):
    """
    A completer that provides suggestions based on previously entered history.
    """

    def __init__(self, history: FileHistory) -> None:
        self.chat_history = history

    def get_completions(self, document, complete_event):
        text_so_far = document.text_before_cursor
        unique_history_items = list(set(self.chat_history.get_strings()))
        for item in unique_history_items:
            if item.startswith(text_so_far):
                yield Completion(item, start_position=-len(text_so_far))


class OptionType(Enum):
    SINGLE = auto()  # A static option that can be selected directly
    EDITABLE = auto()  # An option where the user can input custom text
    TOGGLE = auto()  # A toggle option that can switch between multiple values


class Selector:
    """
    A selector that manages a list of options (single, toggle, or editable).
    """

    def __init__(self, options_dict: Dict[str, Union[None, str, List[Any]]], start_index: int = 0) -> None:
        self.current_index: int = start_index
        self.options: List[Tuple[str, OptionType]] = []
        self.selected: bool = False
        self.user_inputs: Dict[str, str] = {}
        self.toggle_values: Dict[str, Any] = {}
        self.toggle_choices: Dict[str, List[Any]] = {}

        # Parse the dictionary and set up internal structures
        for key, value in options_dict.items():
            if value is None:
                self.options.append((key, OptionType.SINGLE))
            elif isinstance(value, list):
                self.options.append((key, OptionType.TOGGLE))
                self.toggle_choices[key] = value
                self.toggle_values[key] = value[0]
            elif isinstance(value, str):
                self.options.append((key, OptionType.EDITABLE))
                self.user_inputs[key] = value


class OptionSelectorApp:
    """
    The main application class for displaying and handling user input with no highlight.
    """

    def __init__(self) -> None:
        self.selector: Optional[Selector] = None
        self.controls: List[Any] = []
        self.buffers: Dict[str, TextArea] = {}
        self.kb = KeyBindings()
        self.layout: Optional[Layout] = None
        self.application: Optional[Application] = None
        self.chat_history: Dict[str, FileHistory] = {}

        # Build key bindings first
        self.build_key_bindings()

        # Modern dark theme styling
        self.style = Style.from_dict(
            {
                # Base colors and removing white bar
                "": "bg:#1a1a1a fg:#ffffff",  # Global default
                "bottom-toolbar": "bg:#1a1a1a",
                "frame.border": "bg:#1a1a1a fg:#404040",  # Frame border color
                "frame.label": "bg:#1a1a1a fg:#3498db",  # Frame title color
                # Menu elements
                "arrow": "fg:#3498db bold",  # Modern blue arrow
                # "selected": "fg:#ffffff bg:#2c3e50",  # Selection highlight
                "title": "fg:#2ecc71 bold",  # Title text
                "option": "fg:#ecf0f1",  # Normal option text
                # Input area
                "text-area": "bg:#1a1a1a fg:#ffffff",
                "text-area.cursor-line": "bg:#1a1a1a",
                "cursor": "fg:#ffffff",
                # Completion menu
                "completion-menu": "bg:#2c3e50 fg:#ffffff",
                "completion-menu.completion": "bg:#2c3e50 fg:#ffffff",
                "completion-menu.completion.current": "bg:#34495e fg:#ffffff",
            }
        )

    def set_current_selection(self) -> List[Tuple[str, str]]:
        """Set the currently selected option index."""
        current_selection = self.selector.options[self.selector.current_index][0]
        return [("class:title", f"   Make a choice: {current_selection}")]

    def set_selector(self, selector: Selector) -> None:
        """
        Assign a Selector and rebuild UI components.
        """
        self.selector = selector
        self.controls.clear()
        self.buffers.clear()
        self.build_controls()

        # Create a title bar that shows current selection
        title_bar = Window(
            height=1,
            content=FormattedTextControl(
                lambda: self.set_current_selection(),
            ),
            style="bg:#1a1a1a",
        )

        # Frame around options with modern styling
        framed_controls = Frame(
            body=HSplit(self.controls),
            style="bg:#1a1a1a",
            width=None,
            height=None,
            title=" Use ↑/↓ to navigate, ←/→ to toggle/edit, Enter to select ",
        )

        self.layout = Layout(
            HSplit([
                title_bar,
                framed_controls
            ])
        )

        try:
            self._initialize_application()
        except NoConsoleScreenBufferError:
            logger.error("Error initializing the application:\n%s", traceback.format_exc())
            sys.exit(1)

    def build_controls(self) -> None:
        """
        Create a UI control (Window or VSplit) for each option in the Selector.
        """
        for i, (label, opt_type) in enumerate(self.selector.options):
            if opt_type == OptionType.SINGLE:
                control = self.create_single_option_control(i, label)
            elif opt_type == OptionType.TOGGLE:
                control = self.create_toggle_option_control(i, label)
            elif opt_type == OptionType.EDITABLE:
                control = self.create_editable_option_control(i, label)
            else:
                continue
            self.controls.append(control)

    def get_arrow(self, i: int) -> str:
        """
        Returns an arrow for the currently selected option; otherwise a space.
        This is our only indication of "selection".
        """
        return ">" if i == self.selector.current_index else " "

    def invalidate(self) -> None:
        """
        Force a redraw of the screen.
        """
        app = get_app(return_none=True)
        if app:
            app.invalidate()

    def create_single_option_control(self, i: int, label: str) -> Window:
        """
        A plain, single (static) option with no highlight except for an arrow.
        """

        def get_text():
            arrow = self.get_arrow(i)
            return [("", f"{arrow} {label}")]

        control = FormattedTextControl(get_text)
        return Window(content=control, height=1)

    def create_toggle_option_control(self, i: int, label: str) -> Window:
        """
        A toggle option. Only difference from SINGLE is we display the toggle value.
        """

        def get_text():
            arrow = self.get_arrow(i)
            current_value = self.selector.toggle_values[label]
            return [("", f"{arrow} {label}: {current_value}")]

        control = FormattedTextControl(get_text)
        return Window(content=control, height=1)

    def create_editable_option_control(self, i: int, label: str) -> VSplit:
        """
        A combined prompt (arrow + label) next to a TextArea for user input.
        """
        if label not in self.chat_history:
            self.chat_history[label] = FileHistory(get_prompt_cache())

        completer = HistoryCompleter(self.chat_history[label])

        # TODO: change to multiline output?
        text_area = TextArea(
            text=self.selector.user_inputs[label],
            multiline=False,
            wrap_lines=False,
            focus_on_click=True,
            height=1,
            history=self.chat_history[label],
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
        )
        self.buffers[label] = text_area

        def get_prompt():
            arrow = self.get_arrow(i)
            return [("", f"{arrow} {label} ")]

        prompt_control = FormattedTextControl(get_prompt)
        prompt_window = Window(content=prompt_control, dont_extend_width=True)

        return VSplit([prompt_window, text_area], height=1)

    def update_focus(self, app: Application) -> None:
        """
        Make sure we focus the correct control: if it's EDITABLE, focus the TextArea.
        """
        current_option, opt_type = self.selector.options[self.selector.current_index]
        if opt_type == OptionType.EDITABLE:
            app.layout.focus(self.buffers[current_option])
        else:
            app.layout.focus(self.controls[self.selector.current_index])

    def build_key_bindings(self) -> None:
        """
        Define how the user navigates with the keyboard and triggers selection.
        """

        @self.kb.add("up")
        def move_up(event):
            self.selector.current_index = (self.selector.current_index - 1) % len(self.selector.options)
            self.update_focus(event.app)
            self.invalidate()

        @self.kb.add("down")
        def move_down(event):
            self.selector.current_index = (self.selector.current_index + 1) % len(self.selector.options)
            self.update_focus(event.app)
            self.invalidate()

        @self.kb.add("left")
        def left(event):
            current_option, opt_type = self.selector.options[self.selector.current_index]
            if opt_type == OptionType.TOGGLE:
                choices = self.selector.toggle_choices[current_option]
                current_value = self.selector.toggle_values[current_option]
                current_index = choices.index(current_value)
                self.selector.toggle_values[current_option] = choices[(current_index - 1) % len(choices)]
            elif opt_type == OptionType.EDITABLE:
                self.buffers[current_option].buffer.cursor_left()
            self.invalidate()

        @self.kb.add("right")
        def right(event):
            current_option, opt_type = self.selector.options[self.selector.current_index]
            if opt_type == OptionType.TOGGLE:
                choices = self.selector.toggle_choices[current_option]
                current_value = self.selector.toggle_values[current_option]
                current_index = choices.index(current_value)
                self.selector.toggle_values[current_option] = choices[(current_index + 1) % len(choices)]
            elif opt_type == OptionType.EDITABLE:
                self.buffers[current_option].buffer.cursor_right()
            self.invalidate()

        @self.kb.add("enter")
        def enter(event):
            self.selector.selected = True
            current_option, opt_type = self.selector.options[self.selector.current_index]
            if opt_type == OptionType.EDITABLE:
                user_input = self.buffers[current_option].text
                self._handle_editable_input(current_option, user_input)
                self.selector.user_inputs[current_option] = user_input
            event.app.exit()

        @self.kb.add("c-c")
        @self.kb.add("c-q")
        def exit_(event):
            event.app.exit()

    def run(self) -> None:
        """
        Launch the application (blocking call).
        """

        def pre_run():
            self.update_focus(self.application)

        self.application.run(pre_run=pre_run)

    def _handle_editable_input(self, current_option: str, user_input: str) -> None:
        """
        Hook for any custom logic when an EDITABLE input is 'accepted'.
        """
        if current_option == MENU_KEYS["assistant"]:
            self.chat_history[current_option].append_string(user_input)

    def _initialize_application(self) -> None:
        """Initialize the prompt_toolkit Application with optimized settings."""
        self.application = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            mouse_support=False,  # Disable mouse for better performance
        )


# Instantiate a single global app
app = OptionSelectorApp()


def get_user_choice(
    options_dict: Dict[str, Union[None, str, List[Any]]],
    return_value_only: bool = True,
    start_index: int = 0,
) -> Union[str, Dict[str, Any]]:
    """
    Display a styled (yet highlight-free) menu of options. The user uses arrow keys
    to move up/down and optionally left/right to toggle or move cursor in an editable field.
    Pressing Enter finalizes the selection or input.

    Parameters
    ----------
    options_dict : Dict[str, Union[None, str, List[Any]]]
        Key-value pairs of label -> (None | str | list).
        None means a normal single option.
        str means an editable field (the default string).
        list means a toggle field with multiple values.
    return_value_only : bool
        If True, returns the raw result (string or final value).
        If False, returns a dict {chosen_label: chosen_value}.
    start_index : int
        The index at which to start the selector. Default is 0.

    Returns
    -------
    Union[str, Dict[str, Any]]
        If the user selects or inputs something, returns either a string or a dict.
        Returns "" if nothing was selected.
    """
    global app
    selector = Selector(options_dict, start_index)
    app.set_selector(selector)
    app.run()

    if selector.selected:
        selected_option, opt_type = selector.options[selector.current_index]
        # If editable, update from the text buffer
        if opt_type == OptionType.EDITABLE:
            selector.user_inputs[selected_option] = app.buffers[selected_option].text
            result = selector.user_inputs[selected_option]
            return {selected_option: result} if not return_value_only else result
        elif opt_type == OptionType.TOGGLE:
            result = selector.toggle_values[selected_option]
            return {selected_option: result} if not return_value_only else result
        else:
            # SINGLE
            return selected_option

    # If we never selected or pressed enter
    return ""


def get_user_input(
    menu: Optional[Dict[str, Union[None, str, List[Any]]]] = None,
    start_index: int = 0,
) -> Tuple[str, Union[str, None]]:
    """
    Helper function: get the user choice from a menu, returning both the chosen value and key.

    Returns (value, option_key) or (value, None).
    """
    if menu is None:
        menu = MAIN_MENU

    user_choice: Union[str, Dict[str, Any]] = get_user_choice(menu, return_value_only=False, start_index=start_index)
    if isinstance(user_choice, dict):
        option = list(user_choice.keys())[0]
        return user_choice[option], option
    return user_choice, None


def print_colored(text: str, color: str) -> None:
    """
    Print text in a specified ANSI color.

    Raises ValueError if the color is invalid.
    """
    if color not in COLOR_CODES:
        valid_colors = ", ".join(COLOR_CODES.keys())
        raise ValueError(f"Invalid color '{color}'. Valid options are: {valid_colors}")

    color_code = COLOR_CODES[color]
    reset_code = COLOR_CODES["reset"]
    print(f"{color_code}{text}{reset_code}")


if __name__ == "__main__":
    # Example usage showing no highlight at all:
    options = {
        "Option 1": None,
        "Custom Input": "Enter text...",
        "Theme": ["Light", "Dark", "System"],
        "Language": ["English", "Spanish", "French"],
    }
    result = get_user_choice(options)
    print("Selected:", result)
