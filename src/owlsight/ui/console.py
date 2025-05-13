#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a console-based user interface for selecting and configuring
options using prompt_toolkit.
"""

import sys
import traceback
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

from owlsight.configurations.constants import ASSISTENT_PROMPT, MAIN_MENU, CONFIG_DESCRIPTIONS
from owlsight.configurations.schema import Schema
from owlsight.ui.constants import BACKGROUND_STYLE, COLOR_CODES, GLOBAL_STYLE, INSTRUCTIONS
from owlsight.ui.custom_classes import OptionType, AppDTO
from owlsight.utils.constants import get_prompt_cache, KB_AUTOCOMPLETE
from owlsight.utils.custom_classes import GlobalPythonVarsDict, _AVAILBLE_DB_TAGS
from owlsight.utils.logger import logger

if sys.platform == "win32":
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
else:
    class NoConsoleScreenBufferError(Exception):
        """Dummy exception placeholder for non-Windows systems."""
        pass


class CombinedCompleter(Completer):
    """
    A completer that combines multiple completers and provides suggestions based on any of them.
    """

    def __init__(self, *completers: Completer):
        self.completers = completers

    def get_completions(self, document, complete_event):
        for c in self.completers:
            for completion in c.get_completions(document, complete_event):
                yield completion


class ItemCompleter(Completer):
    """
    A completer that provides a list of items for a specific triggerterm.

    Example:
        completer = ItemCompleter(["item1", "item2", "item3"], "item")
        session = PromptSession(completer=completer)
        session.prompt()
        session.prompt("Input: ")
        > "item"
        [dropdown should appear]
    """

    def __init__(self, items: List[str], trigger: str, start_position=0):
        self.items = items
        self.trigger = trigger
        self.start_position = start_position

    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor
        if text_before_cursor.strip().endswith(self.trigger):
            for item in self.items:
                yield Completion(item, start_position=self.start_position)


class HistoryCompleter(Completer):
    """
    A completer that provides suggestions based on previously entered history.
    Caches each FileHistory instance so repeated usage doesn't re-read from disk unnecessarily.
    """

    def __init__(self, history: FileHistory) -> None:
        super().__init__()
        self.chat_history = history

    def get_completions(self, document, complete_event):
        """
        Provide completion suggestions by matching the current input
        against previously entered lines in the FileHistory.
        """
        text_so_far = document.text_before_cursor
        unique_history_items = set(self.chat_history.get_strings())
        for item in unique_history_items:
            if item.startswith(text_so_far):
                yield Completion(item, start_position=-len(text_so_far))


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

        for key, value in options_dict.items():
            if value is None:
                self.options.append((key, OptionType.ACTION))
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

    We reuse this single global instance to avoid re-building UI artifacts each time.
    """

    def __init__(self) -> None:
        self.selector: Optional[Selector] = None
        self.controls: List[Any] = []
        self.buffers: Dict[str, TextArea] = {}
        self.kb = KeyBindings()
        self.layout: Optional[Layout] = None
        self.application: Optional[Application] = None
        self.chat_history: Dict[str, FileHistory] = {}
        self.style = GLOBAL_STYLE
        self.build_key_bindings()

        self._last_config_choice = ""
        self._global_dict = GlobalPythonVarsDict()
        self._db_tag_completer = ItemCompleter(
            items=[f"{tag}:" for tag in _AVAILBLE_DB_TAGS], trigger="[[", start_position=0
        )

    def set_current_description(self) -> None:
        """
        Set the currently selected option's description (for the title bar).
        """
        current_selection = self.selector.options[self.selector.current_index][0]
        is_main_menu = self.selector.options[-1][0] == list(MAIN_MENU.keys())[-1]
        description = ""
        current_config_section = ""

        if not is_main_menu:
            current_config_section = self._last_config_choice
            description = CONFIG_DESCRIPTIONS.get(current_config_section, {}).get(current_selection, "")
        else:
            if self.selector.current_index == 0:
                description = Schema.MENU["assistant"].description
            else:
                description = Schema.MENU.get(current_selection, {}).description
        return [("class:description", f" {description}")]

    def set_current_selection(self) -> List[List[Tuple[str, str]]]:
        """
        Show the currently selected option label (for the title bar).
        """
        current_selection = self.selector.options[self.selector.current_index][0]
        title = f" Current choice: {current_selection}"
        return [("class:title", title)]

    def set_selector(self, selector: Selector) -> None:
        """
        Assign a Selector and rebuild UI components.
        """
        self.selector = selector
        self.controls.clear()
        self.buffers.clear()
        self.build_controls()

        title_bar = HSplit(
            [
                Window(
                    content=FormattedTextControl(lambda: self.set_current_selection()),
                    style=BACKGROUND_STYLE,
                ),
                Window(
                    content=FormattedTextControl(lambda: self.set_current_description()),
                    style="grey",
                ),
            ]
        )
        framed_controls = Frame(
            body=HSplit(self.controls),
            style=BACKGROUND_STYLE,
            width=None,
            title=INSTRUCTIONS.MAIN_MENU,
        )
        self.layout = Layout(HSplit([title_bar, framed_controls]))
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
            if opt_type == OptionType.ACTION:
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
        This is our minimal 'highlight' indicator (actually just an arrow).
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
        A plain, single (static) option with a simple arrow indicator.
        """

        def get_text():
            arrow = self.get_arrow(i)
            return f"{arrow} {label}"

        return Window(content=FormattedTextControl(get_text))

    def create_toggle_option_control(self, i: int, label: str) -> Window:
        """
        A toggle option. Displays the toggle's current value after the label.
        """

        def get_text():
            arrow = self.get_arrow(i)
            current_value = self.selector.toggle_values[label]
            return [("class:arrow", f"{arrow} "), ("class:option", f"{label}: "), ("class:toggle", f"{current_value}")]

        return Window(content=FormattedTextControl(get_text))

    def create_editable_option_control(self, i: int, label: str) -> VSplit:
        """
        A combined label + TextArea for user input. Each label has its own FileHistory.
        """
        if label not in self.chat_history:
            self.chat_history[label] = FileHistory(get_prompt_cache())
        history = self.chat_history[label]
        history_completer = HistoryCompleter(history)
        python_code_completer = ItemCompleter(items=self._global_dict.get_public_keys(), trigger="{{", start_position=0)

        completer = CombinedCompleter(
            history_completer,
            python_code_completer,
            self._db_tag_completer,
        )

        text_area = TextArea(
            text=self.selector.user_inputs[label],
            multiline=True,
            wrap_lines=True,
            focus_on_click=True,
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
        )
        self.buffers[label] = text_area

        def get_prompt():
            arrow = self.get_arrow(i)
            return [("", f"{arrow} {label} ")]

        prompt_window = Window(content=FormattedTextControl(get_prompt), dont_extend_width=True)
        return VSplit([prompt_window, text_area])

    def update_focus(self, app: Application) -> None:
        """
        Focus the correct control depending on whether it's EDITABLE or not.
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
                idx = choices.index(current_value)
                self.selector.toggle_values[current_option] = choices[(idx - 1) % len(choices)]
            elif opt_type == OptionType.EDITABLE:
                self.buffers[current_option].buffer.cursor_left()
            self.invalidate()

        @self.kb.add("right")
        def right(event):
            current_option, opt_type = self.selector.options[self.selector.current_index]
            if opt_type == OptionType.TOGGLE:
                choices = self.selector.toggle_choices[current_option]
                current_value = self.selector.toggle_values[current_option]
                idx = choices.index(current_value)
                self.selector.toggle_values[current_option] = choices[(idx + 1) % len(choices)]
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

        @self.kb.add("c-c")  # Ctrl+C for copy
        def copy_text(event):
            """Copy selected text using prompt_toolkit's clipboard."""
            current_option, opt_type = self.selector.options[self.selector.current_index]
            if opt_type == OptionType.EDITABLE:
                buffer = self.buffers[current_option].buffer
                if buffer.selection_state:
                    # Store current selection state
                    start = buffer.selection_state.original_cursor_position
                    end = buffer.cursor_position
                    sel_type = buffer.selection_state.type

                    # Copy the selection
                    data = buffer.copy_selection()
                    event.app.clipboard.set_data(data)

                    # Restore selection
                    buffer.cursor_position = end
                    buffer.start_selection(selection_type=sel_type)
                    buffer.cursor_position = start
            self.invalidate()

        @self.kb.add("c-y")  # Ctrl+V for paste
        def paste_text(event):
            """Paste text using prompt_toolkit's clipboard."""
            current_option, opt_type = self.selector.options[self.selector.current_index]
            if opt_type == OptionType.EDITABLE:
                buffer = self.buffers[current_option].buffer
                buffer.paste_clipboard_data(event.app.clipboard.get_data())  # Get from prompt_toolkit's clipboard
            self.invalidate()

        @self.kb.add("c-q")
        def exit_(event):
            event.app.exit()

        @self.kb.add(*KB_AUTOCOMPLETE)
        def _(event):
            "Initialize autocompletion, or select the next completion."
            buff = event.app.current_buffer
            if buff.complete_state:
                buff.complete_next()
            else:
                buff.start_completion(select_first=False)

        @self.kb.add("c-a")  # Ctrl+A
        def select_all_text(event):
            """Select all text in the current editable field."""
            current_option, opt_type = self.selector.options[self.selector.current_index]
            if opt_type == OptionType.EDITABLE:
                buffer = self.buffers[current_option].buffer
                buffer.cursor_position = len(buffer.text)
                buffer.start_selection()
                buffer.cursor_position = 0
            self.invalidate()

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
        if current_option == ASSISTENT_PROMPT:
            self.chat_history[current_option].append_string(user_input)

    def _initialize_application(self) -> None:
        """
        Initialize the prompt_toolkit Application with performance-focused settings.
        """
        self.application = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            mouse_support=False,
        )


app = OptionSelectorApp()


def get_user_choice(
    options_dict: Dict[str, Union[None, str, List[Any]]],
    app_dto: Optional[AppDTO] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Display a styled (yet highlight-free) menu of options. The user uses arrow keys
    to move up/down and optionally left/right to toggle or move cursor in an editable field.
    Pressing Enter finalizes the selection or input.

    Parameters:
    ----------
    options_dict: Dict[str, Union[None, str, List[Any]]]
        A dictionary of options to display. The value can be None, a string, or a list.
    app_dto: Optional[AppDTO]
        An optional DTO to pass in configuration data.
    """
    global app
    if app_dto is None:
        app_dto = AppDTO()
    app._last_config_choice = app_dto.last_config_choice

    selector = Selector(options_dict, app_dto.start_index)
    app.set_selector(selector)
    app.run()

    if selector.selected:
        selected_option, opt_type = selector.options[selector.current_index]
        if opt_type == OptionType.EDITABLE:
            selector.user_inputs[selected_option] = app.buffers[selected_option].text
            result = selector.user_inputs[selected_option]
            return {selected_option: result} if not app_dto.return_value_only else result
        elif opt_type == OptionType.TOGGLE:
            result = selector.toggle_values[selected_option]
            return {selected_option: result} if not app_dto.return_value_only else result
        else:
            return selected_option
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
    app_dto = AppDTO(return_value_only=False, start_index=start_index)
    user_choice: Union[str, Dict[str, Any]] = get_user_choice(menu, app_dto)

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
    options = {
        "Option 1": None,
        "Custom Input": "Enter text...",
        "Theme": ["Light", "Dark", "System"],
        "Language": ["English", "Spanish", "French"],
    }
    result = get_user_choice(options)
    print("Selected:", result)
