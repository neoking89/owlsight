# test_owlsight_stress.py

import asyncio
import functools
import platform
import os
import subprocess
import random
from typing import Optional
import logging

import pytest
import psutil
from pynput.keyboard import Key, Controller

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        async def wrapper(self, *args, **kwargs):
            # Move down n times
            for _ in range(n):
                await self.press_key(Key.down)

            # Execute the original function
            result = await func(self, *args, **kwargs)

            # Move back up n times
            for _ in range(n):
                await self.press_key(Key.up)

            return result

        return wrapper

    return decorator


class OwlsightStressTester:
    def __init__(self):
        self.keyboard = Controller()
        self.system = platform.system()
        self.owlsight_pid: Optional[int] = None

        # Main menu options
        self.main_menu = ["how can I assist you?", "shell", "python", "config", "save", "load", "clear history", "quit"]
        self.in_main_menu = True
        self.main_menu_index = 0  # start at the top

        # Test commands for different modes
        self.python_commands = ["1+1", "print('test')", "owl_show()", "a=42"]
        self.shell_commands = ["pwd", "echo test", "ls", "dir"]
        self.ai_prompts = ["hi", "write a function", "help", "what is Python?"]

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

    async def type_fast(self, text: str):
        """Type text with minimal delay"""
        logger.debug(f"Typing text: {text}")
        for char in text:
            self.keyboard.press(char)
            self.keyboard.release(char)
        await asyncio.sleep(0.01)

    async def press_key(self, key: Key, times=1):
        """Press key with minimal delay"""
        logger.debug(f"Pressing key: {key} {times} times")
        for _ in range(times):
            self.keyboard.press(key)
            self.keyboard.release(key)
            self.set_main_menu_index(key)
        await asyncio.sleep(0.01)

    @move_down_up(2)
    async def test_python(self):
        """Test Python interpreter"""
        logger.debug("Starting test_python")
        if self.main_menu_index != 2:
            self._raise_wrong_mode_error("Python")
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)
        command = random.choice(self.python_commands)
        logger.debug(f"Executing Python command: {command}")
        await self.type_fast(command)
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)
        await self.type_fast("exit()")
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)

    @move_down_up(1)
    async def test_shell(self):
        """Test shell command execution"""
        logger.debug("Starting test_shell")
        if self.main_menu_index != 1:
            self._raise_wrong_mode_error("shell")
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)
        command = random.choice(self.shell_commands)
        logger.debug(f"Executing shell command: {command}")
        await self.type_fast(command)
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)
        await self.type_fast("exit")
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)

    async def test_ai(self):
        """Test AI interaction"""
        logger.debug("Starting test_ai")
        if self.main_menu_index != 0:
            self._raise_wrong_mode_error("AI")
        prompt = random.choice(self.ai_prompts)
        logger.debug(f"Sending AI prompt: {prompt}")
        await self.type_fast(prompt)
        await self.press_key(Key.enter)
        await asyncio.sleep(0.2)

    async def execute_random_action(self):
        """Execute a random action from available options"""
        actions = [(self.test_python, "python test"), (self.test_shell, "shell test"), (self.test_ai, "AI test")]
        action, name = random.choice(actions)
        logger.debug(f"Selected action: {name}")

        try:
            await action()
        except Exception as e:
            # Capture exception details and re-raise
            logger.error(f"Exception occurred during {name}: {e}")
            raise

        if not self.is_owlsight_running():
            # You can capture more information here
            logger.error(f"Owlsight process is not running after {name}")
            raise RuntimeError(f"Owlsight process died during {name}")

    async def final_end_to_end_test(self) -> tuple[bool, str]:
        """Run final end-to-end verification"""
        logger.debug("Starting final_end_to_end_test")
        if self.main_menu_index != 0:
            self._raise_wrong_mode_error("AI")
        try:
            # Navigate to quit
            for _ in range(10):  # Go all the way up
                await self.press_key(Key.up)
                await asyncio.sleep(0.05)

            # Go down to load (7 times from top)
            for _ in range(7):
                await self.press_key(Key.down)
                await asyncio.sleep(0.05)

            # Two more downs to reach quit
            await self.press_key(Key.down)  # To clear history
            await self.press_key(Key.enter)
            await asyncio.sleep(0.05)
            await self.press_key(Key.down)  # To quit
            await asyncio.sleep(0.05)

            # Execute quit
            await self.press_key(Key.enter)
            await asyncio.sleep(1.0)

            # Verify process termination
            for _ in range(10):
                if not self.is_owlsight_running():
                    return True, "Clean exit confirmed"
                await asyncio.sleep(0.1)

            return False, "Owlsight process still running after quit"

        except Exception as e:
            # Re-raise the exception to preserve stack trace
            logger.error(f"End-to-end test failed: {e}")
            raise

    async def startup(self) -> bool:
        """Start Owlsight and verify it's running"""
        logger.debug("Starting up Owlsight")
        # Start terminal
        if self.system == "Windows":
            cmd = f"start powershell -NoExit -Command \"cd '{os.getcwd()}'; $host.UI.RawUI.WindowTitle = 'Owlsight-Terminal'\""
            subprocess.Popen(cmd, shell=True)
            await asyncio.sleep(0.5)
            windows = gw.getWindowsWithTitle("Owlsight-Terminal")
            if windows:
                windows[0].activate()
        else:
            subprocess.Popen(["gnome-terminal", "--working-directory", os.getcwd(), "--title=Owlsight-Terminal", "--"])
            await asyncio.sleep(0.5)
            subprocess.run("xdotool search --name 'Owlsight-Terminal' windowactivate", shell=True)

        await asyncio.sleep(0.1)

        # Start Owlsight
        await self.type_fast("owlsight")
        await self.press_key(Key.enter)
        await asyncio.sleep(5)  # Wait for startup

        # Find process
        self.owlsight_pid = self.find_owlsight_process()
        if self.owlsight_pid:
            logger.debug(f"Owlsight started with PID: {self.owlsight_pid}")
        else:
            logger.error("Failed to find Owlsight process")
        return self.owlsight_pid is not None

    def cleanup(self):
        """Force cleanup if necessary"""
        logger.debug("Cleaning up")
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


@pytest.mark.asyncio
async def test_owlsight_stress():
    """
    Pytest function to run stress test on Owlsight.
    Tests stability through random operations and verifies clean exit.
    """
    num_iterations = 10  # Number of random operations to perform

    tester = OwlsightStressTester()

    try:
        # Start Owlsight
        startup_success = await tester.startup()
        assert startup_success, "Failed to start Owlsight"

        # Run random tests
        for i in range(num_iterations):
            logger.debug(f"Starting iteration {i+1}/{num_iterations}")
            await tester.execute_random_action()

        # Run end-to-end test
        success, message = await tester.final_end_to_end_test()
        assert success, f"End-to-end test failed: {message}"

    finally:
        tester.cleanup()
        # Verify Owlsight is not running after cleanup
        assert not tester.is_owlsight_running(), "Owlsight process still running after cleanup"


# if __name__ == "__main__":
#     # Allow running directly for debugging
#     pytest.main([__file__, "-v", "-s"])
