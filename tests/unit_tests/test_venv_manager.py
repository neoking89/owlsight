import os
from unittest.mock import patch, call
import subprocess
import pytest

from owlsight.utils.venv_manager import install_python_modules


@patch("subprocess.check_call")
@patch("sys.path", new_callable=list)
def test_single_module_install(mock_sys_path, mock_check_call):
    """
    Test installation of a single module using pytest.
    """
    mock_check_call.return_value = 0  # Simulate successful pip install
    target_dir = os.path.join("path", "to", "target")

    result = install_python_modules("some-package", "pip", target_dir, "--upgrade")

    # Assert that the pip command was called correctly
    mock_check_call.assert_called_once_with(["pip", "install", "--target", target_dir, "some-package", "--upgrade"])

    # Assert that the target_dir was added to sys.path
    assert target_dir in mock_sys_path

    # Assert that the result is True (successful install)
    assert result is True


@patch("subprocess.check_call")
@patch("sys.path", new_callable=list)
def test_multiple_module_install(mock_sys_path, mock_check_call):
    """
    Test installation of multiple modules using pytest.
    """
    mock_check_call.return_value = 0  # Simulate successful pip install
    target_dir = os.path.join("path", "to", "target")

    result = install_python_modules("numpy pandas", "pip", target_dir, "--upgrade")

    # Assert that the pip command was called twice (for each module)
    mock_check_call.assert_has_calls(
        [
            call(["pip", "install", "--target", target_dir, "numpy", "--upgrade"]),
            call(["pip", "install", "--target", target_dir, "pandas", "--upgrade"]),
        ]
    )

    # Assert that the target_dir was added to sys.path
    assert target_dir in mock_sys_path

    # Assert that the result is True (successful install)
    assert result is True


@patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip"))
@patch("sys.path", new_callable=list)
def test_failed_module_install(mock_sys_path, mock_check_call):
    """
    Test a failed installation case using pytest.
    """
    target_dir = os.path.join("path", "to", "target")
    result = install_python_modules("failing-package", "pip", target_dir)

    # Assert that the pip command was called
    mock_check_call.assert_called_once_with(["pip", "install", "--target", target_dir, "failing-package"])

    # Assert that the target_dir was not added to sys.path since installation failed
    assert target_dir not in mock_sys_path

    # Assert that the result is False (installation failed)
    assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
