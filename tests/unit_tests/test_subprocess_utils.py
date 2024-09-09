import subprocess
import sys

import pytest

sys.path.append(".")
from src.utils.subprocess_utils import parse_globals_from_stdout

def test_parse_command_without_subprocess_run():
    # arrange
    command = 'python -c "print(5)"'
    expected = "5\n"

    # act
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, check=True
    )

    # assert
    assert result.stdout == expected


def test_parse_command_with_spaces_subprocess_run():
    # arrange
    command = 'python -c "a = 5;print(a)"'
    expected = "5\n"

    # act
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, check=True
    )

    # assert
    assert result.stdout == expected


def test_parse_globals_from_stdout():
    # arrange
    stdout = "{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <class '_frozen_importlib.BuiltinImporter'>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, 'a': 5}\n"

    # act
    globals_dict = parse_globals_from_stdout(stdout)

    # assert
    assert isinstance(globals_dict, dict)
    assert globals_dict["a"] == 5
    assert globals_dict["__loader__"] == "<class '_frozen_importlib.BuiltinImporter'>"
    assert globals_dict["__name__"] == "__main__"


if __name__ == "__main__":
    pytest.main([__file__])
