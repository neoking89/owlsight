import importlib.util
import os
import builtins
import inspect
import traceback


class OwlDefaultFunctions:
    """
    Define default functions that can be used in the Python interpreter.
    This provides the user with some utility functions to interact with the interpreter.
    Convention is that the functions start with 'owl_' to avoid conflicts with built-in functions.

    This class is open for extension, as possibly more useful functions can be added in the future.
    """

    def __init__(self, globals_dict):
        # Add check to make sure every function starts with 'owl_'
        self.globals_dict = globals_dict
        self._check_method_naming_convention()

    def _check_method_naming_convention(self):
        """Check if all methods in the class start with 'owl_'."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        methods = [method for method in methods if not method[0].startswith("_")]
        for name, _ in methods:
            if not name.startswith("owl_"):
                raise ValueError(f"Method '{name}' does not follow the 'owl_' naming convention!")

    # Function to read a text file
    def owl_read(self, file_path: str) -> str:
        """
        Read the content of a text file.
        """
        try:
            with open(file_path, "r") as file:
                return file.read()
        except FileNotFoundError:
            return f"File not found: {file_path}"

    # Function to dynamically import a Python file and load its contents into the current namespace
    def owl_import(self, file_path: str):
        """
        Import a Python file and load its contents into the current namespace.

        Parameters
        ----------
        file_path : str
            The path to the Python file to import.
        """
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.globals_dict.update(vars(module))
            print(f"Module '{module_name}' imported successfully.")
        except Exception:
            print(f"Error importing module:\n{traceback.format_exc()}")

    # Function to show all currently active imported objects in the namespace except builtins
    def owl_show(self, docs: bool = False):
        """Show all currently active imported objects in the namespace except builtins.

        Parameters:
        -----------
        docs (bool): If True, also display the docstring of each object.
        """
        current_globals = self.globals_dict
        active_objects = {
            name: obj
            for name, obj in current_globals.items()
            if name not in dir(builtins)
        }

        brackets = "#" * 50
        print(brackets)
        print("Active imported objects:\n")
        for name, obj in active_objects.items():
            if not name.startswith("__"):
                obj_type = type(obj).__name__
                print(f"{name} ({obj_type})")

                # Optionally display the docstring if available
                if docs:
                    docstring = obj.__doc__
                    if docstring:
                        print(f"Doc: {docstring.strip()}")
                    else:
                        print(f"Doc: No documentation available")

        print(brackets)

    # Function to write content to a file
    def owl_write(self, file_path: str, content: str):
        """
        Write content to a text file.
        """
        try:
            with open(file_path, "w") as file:
                file.write(content)
            print(f"Content successfully written to {file_path}")
        except Exception as e:
            print(f"Error writing to file: {e}")
