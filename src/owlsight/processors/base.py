from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizer

from owlsight.utils.logger import logger


class TextGenerationProcessor(ABC):
    """Abstract base class for text generation processors implementing basic generation."""

    def __init__(
        self,
        model_id: str,
        apply_chat_history: bool,
        system_prompt: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        apply_tools: Optional[List[dict]] = None,
    ):
        """
        Initialize the text generation processor.

        Parameters
        ----------
        model_id : str
            The model ID to use for generation.
            Usually the name of the model or the path to the model.
        apply_chat_history : bool
            Whether or not to save the history of inputs and outputs.
        system_prompt : str
            The system prompt to use for generation.
        model_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments for the model. Default is None.
        apply_tools : Optional[List[dict]]
            A list of tools to call from the processor. Default is None.
            Also see: https://medium.com/@malumbea/function-tool-calling-using-gemma-transform-instruction-tuned-it-model-bc8b05585377
        """
        if not model_id:
            raise ValueError("Model ID cannot be empty.")

        self.model_id = model_id
        self.apply_chat_history = apply_chat_history
        self.system_prompt = system_prompt
        self.chat_history = []
        self.model_kwargs = model_kwargs or {}
        self.apply_tools = apply_tools

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
            messages = self.get_history() if self.apply_chat_history else []
            messages.append({"role": "user", "content": input_data})
            templated_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True if self.apply_tools else False
            )
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
        self.chat_history.append({"role": "user", "content": input_data})
        self.chat_history.append({"role": "assistant", "content": generated_text.strip()})

    def clear_history(self) -> None:
        """
        Clear the chat history.
        This is useful for resetting the state of the processor.
        """
        self.chat_history = []
        logger.info("Chat history cleared.")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get complete chat history of inputs and outputs and system prompt.

        Returns
        -------
        List[Dict[str, str]]
            The chat history including system prompt if present.
        """
        messages = self.chat_history.copy()
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def generate_openai_comp(self, messages: list[dict[str, str]], **generate_kwargs: Any) -> str:
        """
        Generate text based on input data using OpenAI compatible API.

        Returns
        -------
        str
            The generated text.

        NOTE: This method is not thread-safe and should not be called concurrently.
        """
        if not messages:
            return ""

        original_system_prompt = self.system_prompt
        original_chat_history = list(self.chat_history)

        current_call_messages_internal = list(messages)

        if current_call_messages_internal[0]["role"] == "system":
            self.system_prompt = current_call_messages_internal.pop(0)["content"]

        if not current_call_messages_internal:
            self.system_prompt = original_system_prompt
            return ""

        last_message = current_call_messages_internal[-1]
        input_data_for_generate = last_message["content"]
        self.chat_history = current_call_messages_internal[:-1]

        try:
            response = self.generate(input_data_for_generate, **generate_kwargs)
        finally:
            self.system_prompt = original_system_prompt
            self.chat_history = original_chat_history
        
        return response

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
