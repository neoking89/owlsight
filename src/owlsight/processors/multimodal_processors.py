from typing import Optional, Dict, Any, Union, List
import traceback
from pathlib import Path
import io
import os
import requests
import numpy as np
import re

from owlsight.hugging_face.constants import HUGGINGFACE_MEDIA_TASKS
from owlsight.processors.base import TextGenerationProcessor, MultiModalTextGenerationProcessor
from owlsight.processors.text_generation_processors import TextGenerationProcessorTransformers
from owlsight.processors.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from owlsight.utils.custom_classes import MediaObject
from owlsight.utils.logger import logger

try:
    from PIL import Image
except ImportError:
    logger.warning("Pillow is not installed. Please install it using 'pip install pillow'.")

class MediaPreprocessor:
    """
    Handles preprocessing for different media types and integrates with text generation.

    This class preprocesses media inputs (images, audio, documents) before passing them
    to the appropriate model pipeline.
    """

    def __init__(self):
        """
        Initialize preprocessor for specific task.
        """

    def preprocess_input(self, media_obj: MediaObject, question: Optional[str] = None) -> Any:
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
        if not isinstance(media_obj, MediaObject):
            raise TypeError("Input data must be a MediaObject instance.")
        input_data = media_obj.path

        try:
            if isinstance(input_data, (str, Path)):
                input_data = self._load_from_path_or_url(input_data)

            if media_obj.type == "audio":
                return self._preprocess_audio(input_data)
            elif media_obj.type == "image":
                processed = self._preprocess_image(input_data)
                if question:
                    return {"image": processed, "question": question}
            else:
                raise ValueError(f"Media type {media_obj.type} is not supported")
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


class MultiModalProcessorTransformers(MultiModalTextGenerationProcessor):
    def __init__(self, model_id: str, task: str, apply_chat_history: bool = False, system_prompt: str = "", **kwargs):
        if task not in HUGGINGFACE_MEDIA_TASKS:
            raise ValueError(
                f"Task {task} is not supported for media preprocessing. Should be one of {HUGGINGFACE_MEDIA_TASKS}.\nPerhaps we should set the right task for the model in 'config:huggingface:task' inside the CLI?"
            )

        super().__init__(model_id=model_id, apply_chat_history=apply_chat_history, system_prompt=system_prompt)
        self.task = task
        self.text_processor = TextGenerationProcessorTransformers(model_id=model_id, task=task, **kwargs)
        self.media_preprocessor = MediaPreprocessor()

    def generate(
        self,
        input_data: str,
        media_objects: Dict[str, MediaObject],
        stopwords: Optional[List[str]] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        # First prepare the generation parameters
        input_data, generate_kwargs = self.text_processor.prepare_generation(
            input_data=input_data,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stopwords=stopwords,
            streaming=False,
            generation_kwargs=generation_kwargs,
            apply_chat_template=False,
        )
        generate_kwargs.pop("eos_token_id", None)

        media_refs = re.finditer(r"__MEDIA_\d+__", input_data)

        # For each media reference, preprocess the media and store question if present
        preprocessed_data = self._preprocess_media_objects(input_data, media_objects, media_refs)

        # If we have only one media object, unpack it
        if len(preprocessed_data) == 1:
            preprocessed_data = preprocessed_data[0]

        try:
            response = self.text_processor.pipe_call(preprocessed_data, generate_kwargs=generate_kwargs)
            response = str(response)
            print(response)
        except Exception:
            logger.error(f"Error generating text with media input: {traceback.format_exc()}")
            raise

        self.update_history(str(input_data), response.strip())
        return response

    def preprocess_input(self, input_data: Union[str, bytes, Path], question: Optional[str] = None) -> Any:
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
                    type=media_object.type, path=os.path.join(media_object.path, file), options=media_object.options
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
                    media_obj = MediaObject(type=media_object.type, path=file, options=media_object.options)
                    lst.append(media_obj)
                except Exception as e:
                    logger.error(f"Error processing file {file} to a MediaObject: {e}")

            if not lst:
                raise ValueError("No valid media files found in the list.")

        return lst


class MultiModalProcessorGGUF(TextGenerationProcessor):
    pass


class MultiModalProcessorOnnx(TextGenerationProcessor):
    pass
