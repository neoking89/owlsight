from typing import Optional, List, Dict, Any, Union
import traceback
from pathlib import Path
import io
import requests
import numpy as np
from PIL import Image

from owlsight.hugging_face.constants import HUGGINGFACE_MEDIA_TASKS

from owlsight.processors.base import TextGenerationProcessor
from owlsight.processors.text_generation_processors import TextGenerationProcessorTransformers
from owlsight.processors.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from owlsight.utils.logger import logger


class MediaPreprocessor:
    """
    Handles preprocessing for different media types and integrates with text generation.

    This class preprocesses media inputs (images, audio, documents) before passing them
    to the appropriate model pipeline.
    """

    def __init__(self, task: str):
        """
        Initialize preprocessor for specific task.

        Parameters
        ----------
        task : str
            The task to handle. Must be one of HUGGINGFACE_MEDIA_TASKS or a text task.
        """
        self.task = task
        self._validate_task()

    def _validate_task(self) -> None:
        """Validate that the task is supported."""
        if self.task not in HUGGINGFACE_MEDIA_TASKS and not self.task.endswith("generation"):
            raise ValueError(
                f"Task {self.task} is not supported. Must be one of {HUGGINGFACE_MEDIA_TASKS} "
                f"or end with 'generation'"
            )

    def preprocess_input(self, input_data: Union[str, bytes, Path], question: Optional[str] = None) -> Any:
        """
        Preprocess input data based on task type.

        Parameters
        ----------
        input_data : Union[str, bytes, Path]
            The input data. Can be a file path, URL, or bytes.
        question : Optional[str]
            Question for VQA or document QA tasks.

        Returns
        -------
        Dict[str, Any]
            Preprocessed data in format expected by the model.
        """
        if self.task not in HUGGINGFACE_MEDIA_TASKS:
            raise ValueError(
                f"Task {self.task} is not supported for media preprocessing. Should be one of {HUGGINGFACE_MEDIA_TASKS}"
            )

        try:
            if isinstance(input_data, (str, Path)):
                input_data = self._load_from_path_or_url(input_data)

            if self.task == "automatic-speech-recognition":
                return self._preprocess_audio(input_data)
            elif self.task in ["image-to-text", "visual-question-answering", "document-question-answering"]:
                processed = self._preprocess_image(input_data)
                if question and self.task in ["visual-question-answering", "document-question-answering"]:
                    return {"image": processed, "question": question}
            else:
                raise ValueError(f"Task {self.task} is not supported for media preprocessing.")
            return processed

        except Exception:
            logger.error(f"Error preprocessing input for task {self.task}: {traceback.format_exc()}")
            raise

    def _load_from_path_or_url(self, source: Union[str, Path]) -> bytes:
        """Load data from file path or URL."""
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            return response.content
        else:
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {source}")
            return p.read_bytes()

    def _preprocess_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Preprocess audio data."""
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0

        # Convert stereo to mono if needed
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        return {"array": audio_array, "sampling_rate": 16000}  # Standard sampling rate for most models

    def _preprocess_image(self, image_data: bytes) -> Image.Image:
        """Preprocess image data."""
        image = Image.open(io.BytesIO(image_data))
        return image


class MultiModalProcessorTransformers(TextGenerationProcessor):
    # TODO: add docstring from TextGenerationProcessorTransformers
    def __init__(self, model_id: str, task: Optional[str], **kwargs):
        if task not in HUGGINGFACE_MEDIA_TASKS:
            raise ValueError(
                f"Task {task} is not supported for media preprocessing. Should be one of {HUGGINGFACE_MEDIA_TASKS}"
            )

        self.task = task
        self.text_processor = TextGenerationProcessorTransformers(model_id=model_id, task=task, **kwargs)
        self.media_preprocessor = MediaPreprocessor(self.text_processor.task)

    def generate(
        self,
        input_data: Union[str, bytes, Path, List[Union[str, bytes, Path]]],
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        stopwords=None,  # add stop_words for interface compatibility
    ) -> Union[str, List[Dict[str, Any]]]:
        input_data, generate_kwargs = self.text_processor.prepare_generation(
            input_text=input_data,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stopwords=None,
            streaming=False,
            generation_kwargs=generation_kwargs,
            apply_chat_template=False,
        )
        generate_kwargs.pop("eos_token_id", None)
        if isinstance(input_data, list):
            preprocessed = [self.media_preprocessor.preprocess_input(data, question) for data in input_data]
        else:
            preprocessed = self.media_preprocessor.preprocess_input(input_data, question)

        try:
            return self.text_processor.pipe(preprocessed, generate_kwargs=generate_kwargs)
        except Exception as e:
            logger.error(f"Error generating text with media input: {traceback.format_exc()}")
            raise

    def preprocess_input(self, input_data: Union[str, bytes, Path], question: Optional[str] = None) -> Any:
        processed = self.media_preprocessor.preprocess_input(input_data, question)
        return processed
