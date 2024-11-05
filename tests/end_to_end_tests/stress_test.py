import functools
import platform
import os
import subprocess
import random
from typing import Optional
import logging
import traceback
import time
import shutil

import pytest
import psutil
from pynput.keyboard import Key, Controller

from owlsight.utils.constants import get_cache_dir

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BUTTON_PRESS_TIME = 0.12

# Import pygetwindow only on Windows
if platform.system() == "Windows":
    try:
        import pygetwindow as gw
    except ImportError:
        subprocess.run(["pip", "install", "pygetwindow"], check=True)
        import pygetwindow as gw


def move_down_up(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.ensure_terminal_focused():
                raise RuntimeError("Could not focus terminal window")

            # Move down n times
            for _ in range(n):
                self.press_key(Key.down)

            # Execute the original function
            result = func(self, *args, **kwargs)

            # Move back up n times
            for _ in range(n):
                self.press_key(Key.up)

            return result

        return wrapper

    return decorator


class OwlsightStressTester:
    def __init__(self):
        self.keyboard = Controller()
        self.system = platform.system()
        self.owlsight_pid: Optional[int] = None
        self.max_startup_retries = 3
        self.window_wait_time = 2.0

        # Main menu options
        self.main_menu = ["how can I assist you?", "shell", "python", "config", "save", "load", "clear history", "quit"]
        self.in_main_menu = True
        self.main_menu_index = 0  # start at the top

        # Test commands for different modes
        self.python_commands = ["1+1", "print('test')", "owl_show()", "a=42"]
        self.shell_commands = ["pwd", "echo test", "ls", "dir"]
        self.ai_prompts = ["hi", "write a function", "help", "what is Python?"]

    def find_terminal_window(self) -> bool:
        """Find the terminal window with retries"""
        logger.debug("Looking for terminal window...")
        for attempt in range(3):
            if self.system == "Windows":
                windows = gw.getWindowsWithTitle("Owlsight-Terminal")
                if windows:
                    logger.debug(f"Found terminal window on attempt {attempt + 1}")
                    return True
            else:
                result = subprocess.run("xdotool search --name 'Owlsight-Terminal'", shell=True, capture_output=True)
                if result.returncode == 0:
                    logger.debug(f"Found terminal window on attempt {attempt + 1}")
                    return True

            logger.debug(f"Window not found, attempt {attempt + 1}, waiting...")
            time.sleep(1.0)

        return False

    def ensure_terminal_focused(self) -> bool:
        """Ensure the terminal window is focused before sending keyboard inputs"""
        for attempt in range(3):
            try:
                if self.system == "Windows":
                    windows = gw.getWindowsWithTitle("Owlsight-Terminal")
                    if not windows:
                        logger.error(f"Terminal window not found, attempt {attempt + 1}")
                        time.sleep(1.0)
                        continue

                    window = windows[0]
                    if not window.isActive:
                        window.activate()
                        time.sleep(0.5)  # Increased wait time after activation
                    return True
                else:
                    result = subprocess.run(
                        "xdotool search --name 'Owlsight-Terminal' windowactivate", shell=True, capture_output=True
                    )
                    if result.returncode == 0:
                        time.sleep(0.5)  # Increased wait time after activation
                        return True

                    logger.error(f"Failed to focus terminal window, attempt {attempt + 1}")
                    time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error focusing terminal window: {e}")
                time.sleep(1.0)

        return False

    def set_main_menu_index(self, key: Key):
        """Set the main menu index based on key press"""
        if self.in_main_menu:
            if key == Key.down:
                self.main_menu_index = (self.main_menu_index + 1) % len(self.main_menu)
            elif key == Key.up:
                self.main_menu_index = (self.main_menu_index - 1) % len(self.main_menu)

    def get_main_menu_option(self) -> str:
        """Get the current main menu option"""
        return self.main_menu[self.main_menu_index]

    def find_owlsight_process(self) -> Optional[int]:
        """Find the Owlsight process"""
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if "owlsight" in str(proc.info["cmdline"]).lower():
                    return proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def is_owlsight_running(self) -> bool:
        """Check if Owlsight process is still running"""
        if self.owlsight_pid:
            try:
                return psutil.pid_exists(self.owlsight_pid)
            except Exception as e:
                logger.error(f"Error checking if Owlsight is running: {e}")
        return False

    def type_fast(self, text: str):
        """Type text with minimal delay"""
        if not self.ensure_terminal_focused():
            raise RuntimeError("Could not focus terminal window")

        logger.debug(f"Typing text: {text}")
        for char in text:
            self.keyboard.press(char)
            self.keyboard.release(char)
        time.sleep(BUTTON_PRESS_TIME)

    def press_key(self, key: Key, times=1):
        """Press key with minimal delay"""
        if not self.ensure_terminal_focused():
            raise RuntimeError("Could not focus terminal window")

        logger.debug(f"Pressing key: {key} {times} times")
        for _ in range(times):
            self.keyboard.press(key)
            self.keyboard.release(key)
            self.set_main_menu_index(key)
        time.sleep(BUTTON_PRESS_TIME)

    @move_down_up(2)
    def test_python(self):
        """Test Python interpreter"""
        logger.debug("Starting test_python")
        if self.main_menu_index != 2:
            self._raise_wrong_mode_error("Python")
        self.press_key(Key.enter)
        time.sleep(BUTTON_PRESS_TIME)
        command = random.choice(self.python_commands)
        logger.debug(f"Executing Python command: {command}")
        self.type_fast(command)
        self.press_key(Key.enter)
        time.sleep(BUTTON_PRESS_TIME)
        self.type_fast("exit()")
        self.press_key(Key.enter)
        time.sleep(BUTTON_PRESS_TIME)

    @move_down_up(3)
    def test_rag(self):
        """Test RAG config"""
        logger.debug("Starting test_rag")
        if self.main_menu_index != 3:
            self._raise_wrong_mode_error("Config")
        self.press_key(Key.right, 3)
        self.press_key(Key.enter)
        # should be in rag menu now
        # activate rag
        self.press_key(Key.down)
        self.press_key(Key.right)
        self.press_key(Key.enter)
        self.press_key(Key.down, 2)
        self.type_fast("pandas")
        self.press_key(Key.enter)
        self.press_key(Key.down, 4)
        self.type_fast("How do I merge two dataframes?")
        # execute rag
        self.press_key(Key.enter)

        # deactivate rag
        self.press_key(Key.down)
        self.press_key(Key.right)
        self.press_key(Key.enter)

        # go back to main menu
        self.press_key(Key.enter)

    @move_down_up(1)
    def test_shell(self):
        """Test shell command execution"""
        logger.debug("Starting test_shell")
        if self.main_menu_index != 1:
            self._raise_wrong_mode_error("shell")
        self.press_key(Key.enter)
        time.sleep(BUTTON_PRESS_TIME)
        command = random.choice(self.shell_commands)
        logger.debug(f"Executing shell command: {command}")
        self.type_fast(command)
        self.press_key(Key.enter)
        time.sleep(BUTTON_PRESS_TIME)
        self.type_fast("exit")
        self.press_key(Key.enter)
        time.sleep(BUTTON_PRESS_TIME)

    def test_ai(self):
        """Test AI interaction"""
        logger.debug("Starting test_ai")
        if self.main_menu_index != 0:
            self._raise_wrong_mode_error("AI")
        prompt = random.choice(self.ai_prompts)
        logger.debug(f"Sending AI prompt: {prompt}")
        self.type_fast(prompt)
        self.press_key(Key.enter)
        time.sleep(0.2)

    def execute_random_action(self):
        """Execute a random action from available options"""
        if not self.ensure_terminal_focused():
            raise RuntimeError("Could not focus terminal window")

        actions = [
            (self.test_python, "python test"),
            (self.test_shell, "shell test"),
            (self.test_ai, "AI test"),
            (self.test_rag, "RAG test"),
        ]
        action, name = random.choice(actions)
        logger.debug(f"Selected action: {name}")

        try:
            action()
        except Exception as e:
            logger.error(f"Exception occurred during {name}: {traceback.format_exc()}")
            raise

        if not self.is_owlsight_running():
            logger.error(f"Owlsight process is not running after {name}")
            raise RuntimeError(f"Owlsight process died during {name}")

    def startup(self) -> bool:
        """Start Owlsight and verify it's running"""
        logger.debug("Starting up Owlsight")

        for attempt in range(self.max_startup_retries):
            logger.debug(f"Startup attempt {attempt + 1}/{self.max_startup_retries}")

            # Start terminal
            if self.system == "Windows":
                cmd = f"start powershell -NoExit -Command \"cd '{os.getcwd()}'; $host.UI.RawUI.WindowTitle = 'Owlsight-Terminal'\""
                subprocess.Popen(cmd, shell=True)
            else:
                subprocess.Popen(
                    ["gnome-terminal", "--working-directory", os.getcwd(), "--title=Owlsight-Terminal", "--"]
                )

            # Wait for terminal window to appear
            time.sleep(self.window_wait_time)

            if not self.find_terminal_window():
                logger.error(f"Terminal window not found on attempt {attempt + 1}")
                continue

            if not self.ensure_terminal_focused():
                logger.error(f"Failed to focus terminal window on attempt {attempt + 1}")
                continue

            logger.debug("Terminal window found and focused, starting Owlsight")

            # Start Owlsight
            self.type_fast("owlsight")
            self.press_key(Key.enter)
            time.sleep(5)  # Wait for Owlsight to start

            # Find process
            self.owlsight_pid = self.find_owlsight_process()
            if self.owlsight_pid:
                logger.debug(f"Owlsight started with PID: {self.owlsight_pid}")
                return True

            logger.error(f"Failed to find Owlsight process on attempt {attempt + 1}")

            # Cleanup before retry
            self.cleanup()
            time.sleep(1.0)

        logger.error("All startup attempts failed")
        return False

    def final_end_to_end_test(self) -> tuple[bool, str]:
        """Run final end-to-end verification"""
        logger.debug("Starting final_end_to_end_test")
        if not self.ensure_terminal_focused():
            raise RuntimeError("Could not focus terminal window")

        if self.main_menu_index != 0:
            self._raise_wrong_mode_error("AI")
        try:
            self.press_key(Key.up)
            time.sleep(0.05)

            # Execute quit
            self.press_key(Key.enter)
            time.sleep(1.0)

            # Verify process termination
            for _ in range(10):
                if not self.is_owlsight_running():
                    return True, "Clean exit confirmed"
                time.sleep(BUTTON_PRESS_TIME)

            return False, "Owlsight process still running after quit"

        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            raise

    def cleanup(self):
        """Force cleanup if necessary"""
        logger.debug("Cleaning up")
        cache_dir = get_cache_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        if self.is_owlsight_running():
            if self.system == "Windows":
                for window in gw.getWindowsWithTitle("Owlsight-Terminal"):
                    window.close()
            else:
                subprocess.run(["pkill", "-f", "Owlsight-Terminal"])

    def _raise_wrong_mode_error(self, expected: str):
        actual = self.get_main_menu_option()
        logger.error(f"Wrong menu selected. Expected: {expected}, Actual: {actual}")
        raise RuntimeError(f"Wrong menu selected. Expected: {expected}, Actual: {actual}")


@pytest.mark.stress
def test_owlsight_stress():
    """
    Pytest function to run stress test on Owlsight.
    Tests stability through random operations and verifies clean exit.
    """
    num_iterations = 10  # Number of random operations to perform

    tester = OwlsightStressTester()

    try:
        # Start Owlsight
        startup_success = tester.startup()
        assert startup_success, "Failed to start Owlsight"

        # Run random tests
        for i in range(num_iterations):
            logger.debug(f"Starting iteration {i+1}/{num_iterations}")
            tester.execute_random_action()

        # Run end-to-end test
        success, message = tester.final_end_to_end_test()
        assert success, f"End-to-end test failed: {message}"

    finally:
        tester.cleanup()
        # Verify Owlsight is not running after cleanup
        assert not tester.is_owlsight_running(), "Owlsight process still running after cleanup"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
