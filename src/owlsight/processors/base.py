from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizer

from owlsight.utils.logger import logger


class TextGenerationProcessor(ABC):
    """Abstract base class for text generation processors implementing basic generation."""

    def __init__(
        self,
        model_id: str,
        save_history: bool,
        system_prompt: str,
    ):
        """
        Initialize the text generation processor.

        Parameters
        ----------
        model_id : str
            The model ID to use for generation.
            Usually the name of the model or the path to the model.
        save_history : bool
            Whether or not to save the history of inputs and outputs.
        system_prompt : str
            The system prompt to use for generation.
        """
        if not model_id:
            raise ValueError("Model ID cannot be empty.")

        self.model_id = model_id
        self.save_history = save_history
        self.system_prompt = system_prompt
        self.history = []

    def apply_chat_template(
        self,
        input_data: str,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        """
        Apply chat template to the input text.
        This is used to format the input text before generating a response and should be universal across all models.

        Parameters
        ----------
        input_data : str
            The input text to apply the template to.
        tokenizer : PreTrainedTokenizer
            The tokenizer to use for applying the template.

        Returns
        -------
        str
            The formatted text with the chat template applied.
        """
        if tokenizer.chat_template is not None:
            messages = self.get_history()
            messages.append({"role": "user", "content": input_data})
            templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            logger.warning("Chat template not found in tokenizer. Using input text as is.")
            templated_text = input_data

        return templated_text

    def update_history(self, input_data: str, generated_text: str) -> None:
        """
        Update the history with the input and generated text.

        Parameters
        ----------
        input_data : str
            The input text that was provided.
        generated_text : str
            The text that was generated in response.
        """
        if self.save_history:
            self.history.append({"role": "user", "content": input_data})
            self.history.append({"role": "assistant", "content": generated_text.strip()})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get complete chat history of inputs and outputs and system prompt.

        Returns
        -------
        List[Dict[str, str]]
            The chat history including system prompt if present.
        """
        messages = self.history.copy()
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    @abstractmethod
    def generate(
        self,
        input_data: str,
        max_new_tokens: int,
        temperature: float,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text based on input data."""
        raise NotImplementedError("generate method must be implemented in the subclass.")

    @abstractmethod
    def get_max_context_length(self) -> int:
        """
        Retrieve the maximum context length of the model.

        Returns
        -------
        int
            The maximum number of tokens the model can process in a single input.
        """
        raise NotImplementedError("get_max_context_length method must be implemented in the subclass.")

class MultiModalTextGenerationProcessor(TextGenerationProcessor):
    """Abstract base class for multimodal text generation processors."""
