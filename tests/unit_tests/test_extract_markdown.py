# ------------- Unit Test for extract_markdown ----------------
import sys

sys.path.append(".")

from owlsight.utils.helper_functions import extract_markdown


def test_extract_markdown():
    # Sample markdown input for Python, Bash, and CMD
    md_string = """
    ```python
    print("Hello, World!")
    ```

    ```bash
    echo "Hello, World!"
    ```

    ```cmd
    dir
    ```
    """

    # Expected output
    expected = [
        ("python", 'print("Hello, World!")'),
        ("bash", 'echo "Hello, World!"'),
        ("cmd", "dir"),
    ]

    result = extract_markdown(md_string)
    assert result == expected, f"Expected {expected}, but got {result}"
