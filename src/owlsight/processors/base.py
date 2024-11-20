from abc import ABC
from typing import Any, Dict, Generator, List, Optional

from transformers import PreTrainedTokenizer

from owlsight.utils.logger import logger


class TextGenerationProcessor(ABC):
    def __init__(
        self,
        model_id: str,
        save_history: bool,
        system_prompt: str,
    ):
        """
        Abstract class for text generation processors.

        Parameters
        ----------
        model_id: str
            The model ID to use for generation.
            Usually the name of the model or the path to the model.
        save_history : bool
            Whether or not to save the history of inputs and outputs.
        """
        if not model_id:
            raise ValueError("Model ID cannot be empty.")

        self.model_id = model_id
        self.save_history = save_history
        self.system_prompt = system_prompt
        self.history = []

    def apply_chat_template(
        self,
        input_text: str,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        """
        Apply chat template to the input text.
        This is used to format the input text before generating a response and should be universal across all models.
        """
        if tokenizer.chat_template is not None:
            messages = self.get_history()
            messages.append({"role": "user", "content": input_text})
            templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            logger.warning("Chat template not found in tokenizer. Using input text as is.")
            templated_text = input_text

        return templated_text

    def update_history(self, input_text: str, generated_text: str):
        """Update the history with the input and generated text."""
        if self.save_history:
            self.history.append({"role": "user", "content": input_text})
            self.history.append({"role": "assistant", "content": generated_text.strip()})

    def get_history(self) -> List[Dict[str, str]]:
        """Get complete chathistory of inputs and outputs and system prompt."""
        messages = self.history.copy()
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages

    def generate(
        self,
        input_text: str,
        max_new_tokens: int,
        temperature: float,
        stopwords: Optional[List[str]],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> str:
        raise NotImplementedError("generate method must be implemented in the subclass.")

    def generate_stream(
        self, input_text: str, max_new_tokens: int, temperature: float, generation_kwargs: Optional[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        raise NotImplementedError("generate_stream method must be implemented in the subclass.")
