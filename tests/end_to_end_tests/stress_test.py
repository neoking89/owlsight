# test_owlsight_stress.py

import asyncio
import platform
import os
import subprocess
import random
import pytest
from typing import Optional
import psutil
from pynput.keyboard import Key, Controller

# Import pygetwindow only on Windows
if platform.system() == "Windows":
    try:
        import pygetwindow as gw
    except ImportError:
        subprocess.run(["pip", "install", "pygetwindow"], check=True)
        import pygetwindow as gw

class OwlsightStressTester:
    def __init__(self):
        self.keyboard = Controller()
        self.system = platform.system()
        self.owlsight_pid: Optional[int] = None
        
        # Menu options from README
        self.menu_options = [
            "how can I assist you?",
            "shell",
            "python",
            "config: main",
            "save",
            "load",
            "clear history"
        ]
        
        # Test commands for different modes
        self.python_commands = ["1+1", "print('test')", "owl_show()", "a=42"]
        self.shell_commands = ["pwd", "echo test", "ls", "dir"]
        self.ai_prompts = ["hi", "write a function", "help", "what is Python?"]
        
    def find_owlsight_process(self) -> Optional[int]:
        """Find the Owlsight process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'owlsight' in str(proc.info['cmdline']).lower():
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def is_owlsight_running(self) -> bool:
        """Check if Owlsight process is still running"""
        if self.owlsight_pid:
            try:
                return psutil.pid_exists(self.owlsight_pid)
            except:
                pass
        return False

    async def type_fast(self, text: str):
        """Type text with minimal delay"""
        for char in text:
            self.keyboard.press(char)
            self.keyboard.release(char)
        await asyncio.sleep(0.01)

    async def press_key(self, key, times=1):
        """Press key with minimal delay"""
        for _ in range(times):
            self.keyboard.press(key)
            self.keyboard.release(key)
        await asyncio.sleep(0.01)

    async def test_python(self):
        """Test Python interpreter"""
        await self.type_fast("python")
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)
        await self.type_fast(random.choice(self.python_commands))
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)
        await self.type_fast("exit()")
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)

    async def test_shell(self):
        """Test shell command execution"""
        await self.type_fast("shell")
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)
        await self.type_fast(random.choice(self.shell_commands))
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)
        await self.type_fast("exit")
        await self.press_key(Key.enter)
        await asyncio.sleep(0.1)

    async def test_ai(self):
        """Test AI interaction"""
        await self.type_fast(random.choice(self.ai_prompts))
        await self.press_key(Key.enter)
        await asyncio.sleep(0.2)

    async def execute_random_action(self):
        """Execute a random action from available options"""
        actions = [
            (self.test_python, "python test"),
            (self.test_shell, "shell test"),
            (self.test_ai, "AI test")
        ]
        action, name = random.choice(actions)
        await action()
        
        if not self.is_owlsight_running():
            raise Exception(f"Owlsight process died during {name}")

    async def final_end_to_end_test(self) -> tuple[bool, str]:
        """Run final end-to-end verification"""
        try:
            # Exit Python if we're in it
            await self.type_fast("exit()")
            await self.press_key(Key.enter)
            await asyncio.sleep(0.5)
            
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
            return False, f"End-to-end test failed: {str(e)}"

    async def startup(self) -> bool:
        """Start Owlsight and verify it's running"""
        # Start terminal
        if self.system == "Windows":
            cmd = f'start powershell -NoExit -Command "cd \'{os.getcwd()}\'; $host.UI.RawUI.WindowTitle = \'Owlsight-Terminal\'"'
            subprocess.Popen(cmd, shell=True)
            await asyncio.sleep(0.5)
            windows = gw.getWindowsWithTitle("Owlsight-Terminal")
            if windows:
                windows[0].activate()
        else:
            subprocess.Popen(["gnome-terminal", "--working-directory", os.getcwd(), "--title=Owlsight-Terminal", "--"])
            await asyncio.sleep(0.5)
            subprocess.run(f"xdotool search --name 'Owlsight-Terminal' windowactivate", shell=True)
        
        await asyncio.sleep(0.1)
        
        # Start Owlsight
        await self.type_fast("owlsight")
        await self.press_key(Key.enter)
        await asyncio.sleep(5)  # Wait for startup
        
        # Find process
        self.owlsight_pid = self.find_owlsight_process()
        return self.owlsight_pid is not None

    def cleanup(self):
        """Force cleanup if necessary"""
        if self.is_owlsight_running():
            if self.system == "Windows":
                for window in gw.getWindowsWithTitle("Owlsight-Terminal"):
                    window.close()
            else:
                subprocess.run(["pkill", "-f", "Owlsight-Terminal"])

@pytest.mark.asyncio
async def test_owlsight_stress():
    """
    Pytest function to run stress test on Owlsight.
    Tests stability through random operations and verifies clean exit.
    """
    num_iterations = 50  # Number of random operations to perform
    
    tester = OwlsightStressTester()
    
    try:
        # Start Owlsight
        startup_success = await tester.startup()
        assert startup_success, "Failed to start Owlsight"
        
        # Run random tests
        successful_iterations = 0
        for i in range(num_iterations):
            try:
                await tester.execute_random_action()
                successful_iterations += 1
            except Exception as e:
                pytest.fail(f"Failed at iteration {i+1}: {str(e)}")
        
        # Verify all iterations completed
        assert successful_iterations == num_iterations, \
            f"Not all iterations completed: {successful_iterations}/{num_iterations}"
        
        # Run end-to-end test
        success, message = await tester.final_end_to_end_test()
        assert success, f"End-to-end test failed: {message}"
        
    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")
    
    finally:
        tester.cleanup()
        # Verify Owlsight is not running after cleanup
        assert not tester.is_owlsight_running(), "Owlsight process still running after cleanup"

if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])