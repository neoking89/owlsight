"""Prompt AST and parser for OwlScript special syntax.

This module provides an *abstract-syntax-tree* (AST) representation for the
OwlScript CLI prompt language (double-curly Python interpolation and
double-square-bracket tags) plus a small recursive-descent style parser that
converts raw prompt text into a sequence of nodes.

Example
-------
>>> from owlsight.utils.prompt_ast import PromptParser, Literal, PythonInterpolation, MediaTag
>>> parser = PromptParser()
>>> text = 'Value is {{a}} and here is an [[image:cat.jpg||width=256]].'
>>> nodes = parser.parse(text)
>>> nodes
[Literal('Value is '), PythonInterpolation('a'), Literal(' and here is an '),
 MediaTag(tag='image', payload='cat.jpg', options={'width': '256'}), Literal('.')]
"""
import re
from dataclasses import dataclass
from typing import TypeVar

__all__ = [
    "PromptNode",
    "Literal",
    "PythonInterpolation",
    "MediaTag",
    "PromptParser",
]


class PromptNode:  # pragma: no cover
    """Base class for all prompt-AST nodes."""

    def __repr__(self) -> str:  # noqa: D401
        cls = self.__class__.__name__
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls}({attrs})"


@dataclass
class Literal(PromptNode):
    """Plain text segment with no special meaning."""

    value: str

    def __str__(self) -> str:  # pragma: no cover
        return self.value


@dataclass
class PythonInterpolation(PromptNode):
    """`{{ … }}` block containing a Python expression."""

    expression: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{{{{{self.expression}}}}}"


@dataclass
class MediaTag(PromptNode):
    """`[[tag:payload||key=value]]` block."""

    tag: str
    payload: str
    options: dict[str, str]

    def __str__(self) -> str:  # pragma: no cover
        opts = "||".join(f"{k}={v}" for k, v in self.options.items())
        if opts:
            opts = f"||{opts}"
        return f"[[{self.tag}:{self.payload}{opts}]]"


Node = TypeVar("Node", bound=PromptNode)


class PromptParser:
    """Simple scanner-based parser that yields an AST list of *PromptNode*s."""

    _CURY_RE = re.compile(r"\{\{(?P<expr>.*?)}}", re.DOTALL)
    # Regex for [[tag:payload||k=v]] constructs.
    # Designed so that _end_ of the match is **exactly** at the second closing
    # bracket in "]]" – this prevents any stray characters (e.g. a single "[" or
    # whitespace) from being left behind when two tags are adjacent.
    _SQUARE_RE = re.compile(
        r"""
        \[\[                       # opening [[
        (?P<tag>[a-zA-Z0-9_]+)      # tag name
        :                           # colon separator
        (?P<payload>[^|\]]+)       # payload up to || or ]]
        (?:\|\|(?P<opts>[^\]]+))? # optional ||key=value section
        \]\]                       # closing ]]
        """,
        re.VERBOSE | re.DOTALL,
    )

    def parse(self, text: str) -> list[PromptNode]:
        """Parse *text* into an ordered list of *PromptNode*s."""
        idx = 0
        nodes: list[PromptNode] = []
        length = len(text)

        while idx < length:
            c_match = self._CURY_RE.search(text, idx)
            s_match = self._SQUARE_RE.search(text, idx)

            next_kind: str | None = None
            next_match: re.Match[str] | None = None

            # pick earliest upcoming match (if any)
            if c_match and (not s_match or c_match.start() < s_match.start()):
                next_kind, next_match = "curly", c_match
            elif s_match:
                next_kind, next_match = "square", s_match

            # No more special tokens – remainder is literal text.
            if next_match is None:
                nodes.append(Literal(text[idx:].strip()))
                break

            # Text before the special token.
            if next_match.start() > idx:
                nodes.append(Literal(text[idx : next_match.start()].strip()))

            if next_kind == "curly":
                expr = next_match.group("expr").strip()
                nodes.append(PythonInterpolation(expr))
            else:  # square-bracket tag
                tag = next_match.group("tag")
                payload = next_match.group("payload").strip()
                options: dict[str, str] = {}
                raw_opts = next_match.group("opts")
                if raw_opts:
                    for part in raw_opts.split("||"):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            options[k.strip()] = v.strip()
                nodes.append(MediaTag(tag, payload, options))

            idx = next_match.end()

        # Remove all whitespace
        nodes = [node for node in nodes if not isinstance(node, Literal) or node.value.strip()]


        return nodes


if __name__ == "__main__":
    """Quick manual run: parses a variety of special-syntax examples and prints the AST nodes."""
    examples = [
        # Pure Python interpolations
        "{{1 + 1}}",
        "Result: {{2 * 3}}",
        "{{{1, 2, 3}}}",
        "{{[x for x in range(3)]}}",
        "{{{'key': 5}}}",
        "{{(1, 2, 3)}}",
        "{{5*2}} is more than {{2*3}}",
        # Basic media tags
        "[[image:photo.jpg]]",
        "[[audio:recording.mp3]]",
        "[[video:clip.mp4]]",
        # Media with options
        "[[image:photo.jpg||width=512||height=512||pipeline=depth-estimation]]",
        # Python interpolation inside media path
        "[[image:{{folder}}/{{filename}}]]",
        # Multiple media
        "Compare [[image:first.jpg]] with [[image:second.jpg]]",
        # Mixed content
        "The value is {{2 + 2}} and here's an [[image:test.jpg]]",
        # Load / chain tags
        "[[load:path-to-model1.json]]",
        "[[load:path-to-model1.json]] How much stomachs has a cow?",
        "[[load:path-to-model1.json]][[image:path-to-image.jpg]]",
        "[[load:path-to-model1.json]] [[image:path-to-image.jpg]] [[load:path-to-model2.json]] Some question about the output",
        "[[chain:generate.temperature=0.7]]",
    ]

    parser = PromptParser()
    for i, text in enumerate(examples, 1):
        print(f"\nExample {i}: {text}\nParsed nodes:")
        for node in parser.parse(text):
            print("  ", node)
