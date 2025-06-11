from typing import Optional, Dict, Any, Union, List
import traceback
from pathlib import Path
import io
import os
import requests
import numpy as np
import re

from owlsight.huggingface.constants import HUGGINGFACE_MEDIA_TASKS
from owlsight.processors.base import TextGenerationProcessor
from owlsight.processors.text_generation_processors import (
    TextGenerationProcessorTransformers,
    TextGenerationProcessorOnnx,
    TextGenerationProcessorGGUF,
)
from owlsight.processors.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from owlsight.utils.custom_classes import MediaObject
from owlsight.utils.logger import logger

try:
    from PIL import Image
except ImportError:
    logger.warning("Pillow is not installed. Please install it using 'pip install pillow'.")


class MediaPreprocessor:
    """Media preprocessing utility for handling various media types.

    This class provides preprocessing capabilities for different media types (images, audio)
    before passing them to a model. It handles necessary preprocessing for media too.

    Methods
    -------
    preprocess_input(media_obj, question=None)
        Preprocess media input based on its type and task requirements.

    Notes
    -----
    - Supports multiple media types: images, audio, documents
    - Handles format validation and conversion
    - Integrates with various model pipelines
    - Memory-efficient processing with proper cleanup

    Examples
    --------
    >>> preprocessor = MediaPreprocessor()
    >>> media_obj = MediaObject(path="image.jpg", tag="image")
    >>> processed = preprocessor.preprocess_input(media_obj)
    """

    def __init__(self) -> None:
        """Initialize the media preprocessor."""
        pass

    def preprocess_input(self, media_obj: MediaObject, question: Optional[str] = None) -> Any:
        """Preprocess media input based on type and task requirements.

        Parameters
        ----------
        media_obj : MediaObject
            Media object containing the input data and metadata.
            Can be image, audio, or document data.
        question : str, optional
            Question for visual/audio question-answering tasks.

        Returns
        -------
        Any
            Preprocessed media data in the format required by the model.
            Type depends on the media type and model requirements.
        """
        if not isinstance(media_obj, MediaObject):
            raise TypeError("Input data must be a MediaObject instance.")
        input_data = media_obj.path

        try:
            if isinstance(input_data, (str, Path)):
                input_data = self._load_from_path_or_url(input_data)

            if media_obj.tag == "audio":
                return self._preprocess_audio(input_data)
            elif media_obj.tag == "image":
                processed = self._preprocess_image(input_data)
                if question:
                    return {"image": processed, "question": question}
            else:
                raise ValueError(f"Media type {media_obj.tag} is not supported")
            return processed

        except Exception:
            logger.error(f"Error preprocessing input for MediaObject {media_obj}: {traceback.format_exc()}")
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

    def _preprocess_image(self, image_data: bytes):
        """Preprocess image data."""
        image = Image.open(io.BytesIO(image_data))
        return image


class MultiModalProcessor(TextGenerationProcessor):
    """Abstract base class for multimodal  processors."""

    def __init__(
        self,
        model_id: str,
        apply_chat_history: bool = False,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id, apply_chat_history=apply_chat_history, system_prompt=system_prompt)
        text_processor_type = self._get_text_processor_type()
        self.text_processor: TextGenerationProcessor = text_processor_type(model_id=model_id, **kwargs)
        self.media_preprocessor = MediaPreprocessor()

    def preprocess_input(self, input_data: Union[str, bytes, Path], question: Optional[str] = None) -> Any:
        """Preprocess media input data for the model.

        Parameters
        ----------
        input_data : Union[str, bytes, Path]
            Raw input data to preprocess
        question : str, optional
            Question for question-answering tasks

        Returns
        -------
        Any
            Preprocessed input in the format required by the model
        """
        processed = self.media_preprocessor.preprocess_input(input_data, question)
        return processed

    def get_max_context_length(self):
        return self.text_processor.get_max_context_length()

    def _preprocess_media_objects(self, input_data, media_objects, media_refs) -> List[Dict[str, Any]]:
        preprocessed_data = []
        for ref in media_refs:
            media_id = ref.group()
            media_object = media_objects[media_id]

            # Get the question from the input text before the media reference
            text_before = input_data[: ref.start()].strip()
            question = text_before if text_before else None

            media_obj_iter = self._get_media_obj_iter(media_object)
            # Preprocess the media file
            if media_obj_iter:
                for media_obj in media_obj_iter:
                    try:
                        preprocessed = self.media_preprocessor.preprocess_input(media_obj, question)
                        preprocessed_data.append(preprocessed)
                    except Exception as e:
                        logger.error(f"Error preprocessing MediaObject {media_obj}: {e}")
            else:
                preprocessed = self.media_preprocessor.preprocess_input(media_object, question)
                preprocessed_data.append(preprocessed)

        return preprocessed_data

    def _get_media_obj_iter(self, media_object: MediaObject) -> List[MediaObject]:
        """
        Get an iterator of media objects if the path is a directory or list.
        """
        lst = []
        if isinstance(media_object.path, str) and os.path.isdir(media_object.path):
            return [
                MediaObject(
                    tag=media_object.tag, path=os.path.join(media_object.path, file), options=media_object.options
                )
                for file in os.listdir(media_object.path)
            ]

        # If path is a list, treat each element in the list as a separate file path.
        elif isinstance(media_object.path, list):
            for file in media_object.path:
                if not os.path.exists(file):
                    logger.error(f"File not found: '{file}'. Did you provide the complete and correct path?")
                    continue
                try:
                    media_obj = MediaObject(tag=media_object.tag, path=file, options=media_object.options)
                    lst.append(media_obj)
                except Exception as e:
                    logger.error(f"Error processing file {file} to a MediaObject: {e}")

            if not lst:
                raise ValueError("No valid media files found in the list.")

        return lst

    def _handle_media_refs_and_input(self, input_data: str, media_objects: Dict[str, MediaObject]) -> str:
        media_refs = re.finditer(r"__MEDIA_\d+__", input_data)

        # For each media reference, preprocess the media and store question if present
        preprocessed_data = self._preprocess_media_objects(input_data, media_objects, media_refs)

        # If we have only one media object, unpack it
        if len(preprocessed_data) == 1:
            preprocessed_data = preprocessed_data[0]

        return preprocessed_data

    def _get_text_processor_type(self):
        """Dynamically determine the text processor type based on the class name.

        Returns
        -------
        type
            The text processor class to be used.

        Raises
        ------
        ValueError
            If the determined text processor type is not supported.
        """
        _base_class = type(self).__bases__[0]
        # dynamicly select the type of self.text processor during runtime
        text_processor_type = type(self).__name__.removeprefix(_base_class.__name__)
        possible_classes = [
            TextGenerationProcessorOnnx,
            TextGenerationProcessorTransformers,
            TextGenerationProcessorGGUF,
        ]
        text_processor_type = next((i for i in possible_classes if i.__name__.endswith(text_processor_type)), None)
        if text_processor_type is None:
            raise ValueError(
                f"TextGenerationProcessor type {text_processor_type} not supported. Is it in {possible_classes}?"
            )
        return text_processor_type


class MultiModalProcessorTransformers(MultiModalProcessor):
    """Multimodal processor using Hugging Face transformers.

    This processor handles text generation tasks that involve multiple modalities
    (text, images, audio) using Hugging Face transformer models. It combines
    the MediaPreprocessor for handling media inputs with text generation capabilities.

    Parameters
    ----------
    model_id : str
        Identifier for the Hugging Face model to use
    task : str
        Task type, must be one of HUGGINGFACE_MEDIA_TASKS
    apply_chat_history : bool, default=False
        Whether to maintain chat history
    system_prompt : str, default=""
        System prompt to use for generation
    **kwargs : dict
        Additional arguments passed to TextGenerationProcessorTransformers

    Notes
    -----
    - Supports various multimodal tasks (VQA, image captioning, etc.)
    - Handles media preprocessing automatically
    - Integrates with Hugging Face's transformers library
    - Manages memory efficiently for large media files

    Examples
    --------
    >>> processor = MultiModalProcessorTransformers(
    ...     model_id="dandelin/vilt-b32-finetuned-vqa", task="visual-question-answering"
    ... )
    >>> media_obj = MediaObject(path="image-of-car.jpg", tag="image")
    >>> result = processor.generate("What color is the car in this image:", media_objects={"image1": media_obj})
    """

    def __init__(
        self,
        model_id: str,
        task: str,
        apply_chat_history: bool = False,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        if task not in HUGGINGFACE_MEDIA_TASKS:
            raise ValueError(
                f"Task {task} is not supported for media preprocessing. Should be one of {HUGGINGFACE_MEDIA_TASKS}.\nPerhaps we should set the right task for the model in 'config:huggingface:task' inside the CLI?"
            )

        super().__init__(
            model_id=model_id, apply_chat_history=apply_chat_history, system_prompt=system_prompt, task=task
        )
        self.task = task

    def generate(
        self,
        input_data: str,
        media_objects: Dict[str, MediaObject],
        stop_words: Optional[List[str]] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text based on input text and media objects.

        Parameters
        ----------
        input_data : str
            Text prompt or question
        media_objects : Dict[str, MediaObject]
            Dictionary mapping media references to MediaObject instances
        stop_words : List[str], optional
            List of words to stop generation at
        max_new_tokens : int, default=DEFAULT_MAX_TOKENS
            Maximum number of tokens to generate
        temperature : float, default=DEFAULT_TEMPERATURE
            Sampling temperature for generation
        generation_kwargs : dict, optional
            Additional generation parameters

        Returns
        -------
        str
            Generated text incorporating information from media inputs

        Notes
        -----
        - Automatically handles preprocessing
        - A directory of files can also be provided.
        """
        # First prepare the generation parameters
        input_data, generate_kwargs = self.text_processor.prepare_generation(
            input_data=input_data,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_words=stop_words,
            streaming=False,
            generation_kwargs=generation_kwargs,
            apply_chat_template=False,
        )
        generate_kwargs.pop("eos_token_id", None)

        # Handle media references and input
        preprocessed_data = self._handle_media_refs_and_input(input_data, media_objects)

        try:
            response = self.text_processor.pipe_call(preprocessed_data, generate_kwargs=generate_kwargs)
            response = str(response)
            print(response)
        except Exception:
            logger.error(f"Error generating text with media input: {traceback.format_exc()}")
            raise

        self.update_history(str(input_data), response.strip())
        return response


class MultiModalProcessorGGUF(MultiModalProcessor):
    pass


class MultiModalProcessorOnnx(MultiModalProcessor):
    pass
