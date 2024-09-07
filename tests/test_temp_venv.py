import os
import pytest
import tempfile
from contextlib import contextmanager
import venv

# ------------- Unit Test for tempfile.TemporaryDirectory & venv setup ---------


@contextmanager
def create_venv(venv_path: str) -> str:
    venv.create(venv_path, with_pip=True)
    pip_path = os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "pip")
    yield pip_path


def test_tempdir_and_venv_creation():
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = os.path.join(temp_dir, "venv")

        # Test that the temp directory was created
        assert os.path.exists(temp_dir), "Temp directory was not created"

        with create_venv(venv_path) as pip_path:
            # Test that venv was created
            assert os.path.exists(venv_path), "Virtual environment was not created"
            scripts_dir = os.path.dirname(pip_path)
            assert any(os.path.join(scripts_dir, f).startswith(pip_path) for f in os.listdir(scripts_dir)), \
                f"pip was not found in the expected location: {pip_path}"

    # Test that venv and temp_dir are cleaned up after exit
    assert not os.path.exists(
        venv_path
    ), "Virtual environment was not removed after use"
    assert not os.path.exists(temp_dir), "Temp directory was not removed after use"


# ---------------- Test Execution -----------------

if __name__ == "__main__":
    pytest.main()
