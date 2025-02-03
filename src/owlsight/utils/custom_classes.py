"""
Created by Nestor Demeure.
This software is released under the Apache License 2.0.
"""

from typing import List, Literal, Dict, Union, Any, Optional, get_args
from dataclasses import dataclass, field
from pathlib import Path
import builtins
import inspect


import torch
from transformers import StoppingCriteria, AutoTokenizer

from owlsight.prompts.helper_functions import function_to_json_for_tool_calling


class StopWordCriteria(StoppingCriteria):
    """
    A stopping criteria that halts the text generation process if any specified stop word is encountered.

    Inspired by https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/9
    And: https://github.com/outlines-dev/outlines/blob/main/outlines/generate/api.py
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        stop_words: List[str] = [],
        check_every: int = 1,
    ):
        """
        Initializes the StopWordCriteria with the necessary parameters for checking stop words during text generation.

        Parameters:
        ----------
            tokenizer (AutoTokenizer): The tokenizer for encoding prompts and stop words.
            prompts (List[str]): Initial prompts used for generation, needed to determine where generated text begins.
            stop_words (List[str]): Words that trigger the stopping of generation when detected.
            check_every (int): Frequency of checking for stop words in the token stream (a performance optimization, use 1 to cut it out).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.input_sizes = [self.tokenizer.encode(prompt, return_tensors="pt").size(-1) for prompt in prompts]
        self.stop_words = stop_words
        self.max_stop_word_size = max(
            (self.tokenizer.encode(word, return_tensors="pt").size(-1) for word in stop_words),
            default=0,
        )
        self.check_every = check_every

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Determines whether to stop generation based on the presence of stop words.

        Stops if a stop word is found in *all* batch elements *and* the sequence length is a multiple of `check_every`.
        Note: Delay in stopping may occur if `check_every > 1`.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            scores (torch.FloatTensor): Generation scores for each token. Not used here.

        Returns:
            bool: True to stop generation, False to continue.
        """
        batch_size, seq_len = input_ids.shape

        # Skip check if no stop words are defined or it is not yet time to check
        if (len(self.stop_words) == 0) or (seq_len % self.check_every != 0):
            return False

        for i in range(batch_size):
            # Calculate starting index for new tokens
            prompt_size = self.input_sizes[i]
            max_new_tokens = (2 * self.max_stop_word_size) + self.check_every
            latest_tokens = input_ids[i, prompt_size:][-max_new_tokens:]

            # Check for stop words in the decoded text
            if not any(
                word in self.tokenizer.decode(latest_tokens, skip_special_tokens=True) for word in self.stop_words
            ):
                return False  # Continue generation if any batch item lacks stop words

        return True  # Stop generation if all conditions are met

    def extract_answers(self, input_ids: torch.LongTensor, strip_stopword: bool = True) -> List[str]:
        """
        Extracts generated answers by removing prompts and optionally stopping at the first stop word.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            strip_stopword (bool): Determines whether the stop word is removed from the output.

        Returns:
            List[str]: Extracted answers, with or without stop words.
        """
        batch_size, _ = input_ids.shape
        result = []

        for i in range(batch_size):
            # Decode generated tokens to text, excluding the prompt
            prompt_size = self.input_sizes[i]
            answer_tokens = input_ids[i, prompt_size:]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            # Find the first occurrence of any stop word
            lower_stop_index = len(answer_text)  # Default to end of text
            for word in self.stop_words:
                stop_index = answer_text.find(word)
                if stop_index != -1:
                    # Adjust stop index based on whether we're stripping the stop word
                    stop_index += 0 if strip_stopword else len(word)
                    lower_stop_index = min(stop_index, lower_stop_index)

            # Cut the text at the first stop word found (if any)
            answer_text = answer_text[:lower_stop_index]
            result.append(answer_text)

        return result

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


class SingletonDict(dict):
    """
    A dictionary that only allows a single instance to be created.
    Meant to be used as a singleton for storing global variables and to share state across different classes.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonDict, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def get_public_keys(self):
        return [k for k in self.keys() if not k.startswith("_")]

    def get_tools(self, exclude_keys: Optional[List[str]] = None) -> List[dict]:
        """
        Get a list of available functions which can be used for tool calling out of the global scope.
        NOTE: ONLY objects that are functions are included.

        Returns
        -------
        str
            String representation of available functions/tools in OPENAI-format.
        """
        globals_dict = self._filter_globals(self)
        if exclude_keys is not None:
            globals_dict = {k: v for k, v in globals_dict.items() if k not in exclude_keys}
        tools = [function_to_json_for_tool_calling(v) for v in globals_dict.values() if inspect.isroutine(v)]

        return tools

    def _filter_globals(self, globals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter the globals dictionary to exclude built-in and private objects.

        Parameters
        ----------
        globals_dict : dict
            The globals dictionary to filter.

        Returns
        -------
        dict
            The filtered globals dictionary.
        """
        return {
            key: value for key, value in globals_dict.items() if not key.startswith("_") and key not in dir(builtins)
        }


DoubleBracketsTag = Literal["image", "audio", "load", "chain"]
_AVAILBLE_DB_TAGS = get_args(DoubleBracketsTag)


@dataclass
class DoubleBracketsObject:
    tag: DoubleBracketsTag
    path: Union[str, Path, bytes]
    options: Dict[str, str] = field(default_factory=lambda: {})


@dataclass
class MediaObject(DoubleBracketsObject):
    """
    Represents a media object with its tag, path, and options.
    Specifically handles image, audio, and video content.

    Attributes
    ----------
    tag : DoubleBracketsTag
        The tag of media (image, audio, or video)
    path : Union[str, Path, bytes]
        The path to the media file or a bytes-like object
    options : Dict[str, str]
        Optional parameters for processing the media
    """
