import asyncio
import random
import subprocess
import platform
import os
import psutil

from pynput.keyboard import Key, Controller

class OwlsightStressTester:
    def __init__(self):
        self.keyboard = Controller()
        self.system = platform.system()
        self.owlsight_pid = None
        
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
        
        if self.system == "Windows":
            import pygetwindow as gw
            self.gw = gw

    def find_owlsight_process(self):
        """Find the Owlsight process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'owlsight' in str(proc.info['cmdline']).lower():
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def is_owlsight_running(self):
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

    async def final_end_to_end_test(self):
            """Run final end-to-end verification"""
            print("\nRunning final end-to-end verification...")
            
            try:
                # FIRST: Make absolutely sure we exit Python if we're in it
                print("Ensuring we're at main menu...")
                await self.type_fast("exit()")
                await self.press_key(Key.enter)
                await asyncio.sleep(0.5)  # Wait to ensure we're back at main menu
                
                # Now we know we're at the main menu, navigate to quit
                print("Navigating to quit...")
                # First go all the way up
                for _ in range(10):
                    await self.press_key(Key.up)
                    await asyncio.sleep(0.05)
                
                # Now go down to load (7 times from top menu item)
                for _ in range(7):
                    await self.press_key(Key.down)
                    await asyncio.sleep(0.05)
                
                # Two more downs to reach quit from load
                print("Moving from load to quit...")
                await self.press_key(Key.down)  # To clear history
                await asyncio.sleep(0.05)
                await self.press_key(Key.down)  # To quit
                await asyncio.sleep(0.05)
                
                print("Executing quit command...")
                await self.press_key(Key.enter)
                print("Waiting for Owlsight to close...")
                await asyncio.sleep(1.0)

                # Verify process termination
                for _ in range(10):
                    if not self.is_owlsight_running():
                        return True, "End-to-end test successful - clean exit confirmed"
                    await asyncio.sleep(0.1)

                if self.is_owlsight_running():
                    return False, "Owlsight process still running after quit"
                
                return True, "End-to-end test successful"

            except Exception as e:
                return False, f"End-to-end test failed: {str(e)}"
        
    async def run_stress_test(self, num_iterations: int = 50):
        """Run complete stress test with random actions and end-to-end verification"""
        print(f"Starting {num_iterations} rapid test iterations...")
        
        # Start terminal in current directory
        if self.system == "Windows":
            cmd = f'start powershell -NoExit -Command "cd \'{os.getcwd()}\'; $host.UI.RawUI.WindowTitle = \'Owlsight-Terminal\'"'
            subprocess.Popen(cmd, shell=True)
        else:
            subprocess.Popen(["gnome-terminal", "--working-directory", os.getcwd(), "--title=Owlsight-Terminal", "--"])
        
        await asyncio.sleep(0.5)
        
        # Focus window
        if self.system == "Windows":
            windows = self.gw.getWindowsWithTitle("Owlsight-Terminal")
            if windows:
                windows[0].activate()
        else:
            subprocess.run(f"xdotool search --name 'Owlsight-Terminal' windowactivate", shell=True)
        
        await asyncio.sleep(0.1)
        
        # Start Owlsight
        await self.type_fast("owlsight")
        await self.press_key(Key.enter)
        print("Waiting for Owlsight to start...")
        await asyncio.sleep(5)  # Wait for startup
        
        # Find process
        self.owlsight_pid = self.find_owlsight_process()
        if not self.owlsight_pid:
            raise Exception("Failed to find Owlsight process after startup")
        print(f"Owlsight process found with PID: {self.owlsight_pid}")

        # Run random tests
        successful_iterations = 0
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}", end="\r")
            try:
                await self.execute_random_action()
                successful_iterations += 1
            except Exception as e:
                print(f"\nError in iteration {i+1}: {e}")
                break

        # Run end-to-end test
        success, message = await self.final_end_to_end_test()
        
        print("\n=== Test Summary ===")
        print(f"Random Test Iterations: {successful_iterations}/{num_iterations}")
        print(f"End-to-End Test: {'PASSED' if success else 'FAILED'}")
        print(f"Final Status: {message}")
        
        return success

async def main():
    if platform.system() == "Windows":
        try:
            import pygetwindow
        except ImportError:
            subprocess.run(["pip", "install", "pygetwindow"], check=True)
            print("Installed pygetwindow, please restart")
            return

    try:
        subprocess.run(["pip", "install", "psutil"], check=True)
    except:
        print("Failed to install psutil")
        return

    tester = OwlsightStressTester()
    
    try:
        success = await tester.run_stress_test(num_iterations=50)
        if not success:
            print("\nTest suite failed!")
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        # Only force close if still running
        if tester.is_owlsight_running():
            print("WARNING: Forcing Owlsight terminal closure")
            if platform.system() == "Windows":
                for window in pygetwindow.getWindowsWithTitle("Owlsight-Terminal"):
                    window.close()
            else:
                subprocess.run(["pkill", "-f", "Owlsight-Terminal"])

if __name__ == "__main__":
    print(f"Starting Owlsight stress test on {platform.system()}...")
    asyncio.run(main())