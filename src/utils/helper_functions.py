from typing import List, Tuple
import re


def extract_markdown(md_string: str) -> List[Tuple[str, str]]:
    """
    Extract language and code blocks from a markdown string.
    """
    pattern = r"```(\w+)([\s\S]*?)```"
    return [
        (match[0].strip(), match[1].strip()) for match in re.findall(pattern, md_string)
    ]
