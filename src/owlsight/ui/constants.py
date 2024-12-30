from prompt_toolkit.styles import Style


BACKGROUND_STYLE = "bg:#1a1a1a"
GLOBAL_STYLE = Style.from_dict(
    {
        # Base colors and removing white bar
        "": "bg:#1a1a1a fg:#ffffff bold",  # Global default
        "bottom-toolbar": BACKGROUND_STYLE,
        "frame.border": "bg:#1a1a1a fg:blue bold",
        "frame.label": "bg:#1a1a1a fg:#3498db bold",
        # Menu elements
        "arrow": "fg:#3498db bold",  # Modern blue arrow
        "title": "fg:#2ecc71 bold",  # Title text
        "option": "fg:#ecf0f1",  # Normal option text
        "toggle": "fg:#f39c12 bg:#1a1a1a bold",  # Bright orange text for toggles on dark background
        # Input area
        "text-area": "ansigreen",
        "text-area.cursor-line": BACKGROUND_STYLE,
        "cursor": "fg:#ffffff bg:#1a1a1a underline",
        # Completion menu
        "completion-menu": "bg:#2c3e50 fg:#ffffff",
        "completion-menu.completion": "bg:#2c3e50 fg:#ffffff",
        "completion-menu.completion.current": "bg:#34495e fg:#ffffff",
    }
)
EDIT_CODE_BLOCK_COLOR = "blue"

# ANSI color codes for terminal output
COLOR_CODES = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "reset": "\033[0m",  # Resets to default color
}

class INSTRUCTIONS:
    MAIN_MENU = " Use ↑/↓ to navigate, ←/→ to toggle/edit, Enter to select "