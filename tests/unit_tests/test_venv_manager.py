from unittest.mock import patch, call
import subprocess

from owlsight.utils.venv_manager import install_python_modules

# Import the install_python_modules function (adjust the import based on where the function is located)
# from your_module import install_python_modules


@patch("subprocess.check_call")
@patch("sys.path", new_callable=list)
def test_single_module_install(mock_sys_path, mock_check_call):
    """
    Test installation of a single module using pytest.
    """
    mock_check_call.return_value = 0  # Simulate successful pip install

    result = install_python_modules("some-package", "pip", "/path/to/target", "--upgrade")

    # Assert that the pip command was called correctly
    mock_check_call.assert_called_once_with(
        ["pip", "install", "--target", "/path/to/target", "some-package", "--upgrade"]
    )

    # Assert that the target_dir was added to sys.path
    assert "/path/to/target" in mock_sys_path

    # Assert that the result is True (successful install)
    assert result is True


@patch("subprocess.check_call")
@patch("sys.path", new_callable=list)
def test_multiple_module_install(mock_sys_path, mock_check_call):
    """
    Test installation of multiple modules using pytest.
    """
    mock_check_call.return_value = 0  # Simulate successful pip install

    result = install_python_modules("numpy pandas", "pip", "/path/to/target", "--upgrade")

    # Assert that the pip command was called twice (for each module)
    mock_check_call.assert_has_calls(
        [
            call(["pip", "install", "--target", "/path/to/target", "numpy", "--upgrade"]),
            call(["pip", "install", "--target", "/path/to/target", "pandas", "--upgrade"]),
        ]
    )

    # Assert that the target_dir was added to sys.path
    assert "/path/to/target" in mock_sys_path

    # Assert that the result is True (successful install)
    assert result is True


@patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip"))
@patch("sys.path", new_callable=list)
def test_failed_module_install(mock_sys_path, mock_check_call):
    """
    Test a failed installation case using pytest.
    """
    result = install_python_modules("failing-package", "pip", "/path/to/target")

    # Assert that the pip command was called
    mock_check_call.assert_called_once_with(["pip", "install", "--target", "/path/to/target", "failing-package"])

    # Assert that the target_dir was not added to sys.path since installation failed
    assert "/path/to/target" not in mock_sys_path

    # Assert that the result is False (installation failed)
    assert result is False
