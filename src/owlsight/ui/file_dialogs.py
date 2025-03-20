import subprocess
import os
from typing import Optional


def open_file_dialog(initial_dir: Optional[str] = None):
    """
    Opens a file dialog and returns the selected file path.

    Parameters
    ----------
    initial_dir: str
        The initial directory to open the dialog
    """
    try:
        if initial_dir and os.path.isdir(initial_dir):
            initial_directory = os.path.normpath(initial_dir)
        else:
            initial_directory = os.path.expanduser("~\\Desktop")

        command = f"""
        Add-Type -AssemblyName System.Windows.Forms
        $openFileDialog = New-Object System.Windows.Forms.OpenFileDialog
        $openFileDialog.InitialDirectory = '{initial_directory}'
        $openFileDialog.Filter = 'All Files (*.*)|*.*'
        if ($openFileDialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
            $openFileDialog.FileName
        }}
        """

        result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)

        if result.stdout.strip():
            return result.stdout.strip()
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def save_file_dialog(initial_dir: Optional[str] = None, default_filename: Optional[str] = None) -> Optional[str]:
    """
    Opens a save file dialog and returns the selected file path.

    Parameters
    ----------
    initial_dir: str
        The initial directory to open the dialog
    default_filename: str
        The default filename to display in the dialog
    """
    try:
        if initial_dir and os.path.isdir(initial_dir):
            initial_directory = os.path.normpath(initial_dir)
        else:
            initial_directory = os.path.expanduser("~\\Desktop")

        default_file = default_filename if default_filename else ""

        command = f"""
        Add-Type -AssemblyName System.Windows.Forms
        $saveFileDialog = New-Object System.Windows.Forms.SaveFileDialog
        $saveFileDialog.InitialDirectory = '{initial_directory}'
        $saveFileDialog.FileName = '{default_file}'
        $saveFileDialog.Filter = 'All Files (*.*)|*.*'
        if ($saveFileDialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
            $saveFileDialog.FileName
        }}
        """

        result = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)

        if result.stdout.strip():
            return result.stdout.strip()
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
