from pathlib import Path
import requests
import io
import ast
import sys
import pytesseract

sys.path.append("src")

from PIL import Image
import numpy as np
import pytest

from owlsight.hugging_face.constants import HUGGINGFACE_MEDIA_TASKS
from owlsight.processors.multimodal_processors import MultiModalProcessorTransformers
from owlsight.utils.custom_classes import MediaObject
from owlsight.multimodal.tesseract import find_tesseract_installation
pytesseract.pytesseract.tesseract_cmd = find_tesseract_installation()


# Test URLs
TEST_CASES = [
    {
        "task": "image-to-text",
        "path": "tests/data/image-to-text.jpg",
        "question": None,
        "expected_type": list,
        "media_type": "image",
    },
    {
        "task": "visual-question-answering",
        "path": "tests/data/visual-question-answering.jpg",
        "question": "What color is the car?",
        "expected_type": list,
        "media_type": "image",
    },
    {
        "task": "automatic-speech-recognition",
        "path": "tests/data/automatic-speech-recognition.wav",
        "question": None,
        "expected_type": dict,
        "media_type": "audio",
    },
    {
        "task": "document-question-answering",
        "path": "tests/data/document-question-answering.jpg",
        "question": "What is the total number of students?",
        "expected_type": list,
        "media_type": "image",
    },
]


@pytest.fixture(scope="module")
def test_data():
    """Download and cache test data."""
    cached_data = {}
    for case in TEST_CASES:
        with open(case["path"], "rb") as f:
            cached_data[case["task"]] = f.read()
    return cached_data


@pytest.fixture(params=TEST_CASES)
def processor(request, media_model_mappings):
    """Create processor for each test case."""
    return MultiModalProcessorTransformers(
        model_id=media_model_mappings[request.param["task"]], task=request.param["task"]
    )


def test_media_preprocessor_initialization(media_model_mappings):
    """Test MediaPreprocessor initialization with various tasks."""
    for task in HUGGINGFACE_MEDIA_TASKS:
        processor = MultiModalProcessorTransformers(model_id=media_model_mappings[task], task=task)
        assert processor.task == task
        assert processor.media_preprocessor is not None
        assert processor.text_processor.pipe is not None


def test_invalid_task():
    """Test initialization with invalid task."""
    with pytest.raises(ValueError):
        MultiModalProcessorTransformers(model_id="test", task="invalid_task")


@pytest.mark.parametrize("case", TEST_CASES)
def test_generate(case, test_data, media_model_mappings):
    """Test generate method for each task."""
    processor = MultiModalProcessorTransformers(model_id=media_model_mappings[case["task"]], task=case["task"])

    # Create temporary file with test data
    test_file = Path(f"test_{case['task']}_file.tmp")
    test_file.write_bytes(test_data[case["task"]])

    try:
        # Create MediaObject for the test case
        media_objects = {"__MEDIA_0__": MediaObject(type=case["media_type"], path=str(test_file), options={})}

        # Prepare input text based on whether there's a question
        input_text = f"{case['question']} __MEDIA_0__" if case["question"] else "__MEDIA_0__"

        result = processor.generate(input_text, media_objects=media_objects)
        result = ast.literal_eval(result)

        assert isinstance(result, case["expected_type"])
        assert len(result) > 0

    except pytesseract.TesseractNotFoundError:
        pytest.skip(f"Tesseract is not installed. Skipping {case['task']} test or install it to run this test.")

    finally:
        # Clean up temporary file
        if test_file.exists():
            test_file.unlink()


def test_preprocessing(media_model_mappings):
    """Test preprocessing for different input types."""
    processor = MultiModalProcessorTransformers(model_id=media_model_mappings["image-to-text"], task="image-to-text")

    # Create test image
    test_image = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    test_image.save(buffer, format="PNG")

    # Test with bytes
    media_obj = MediaObject(type="image", path=buffer.getvalue(), options={})
    result = processor.media_preprocessor.preprocess_input(media_obj=media_obj)
    assert isinstance(result, Image.Image)

    # Test with Path
    test_image.save("test_image.png")
    media_obj = MediaObject(type="image", path=Path("test_image.png"), options={})
    result = processor.media_preprocessor.preprocess_input(media_obj=media_obj)
    assert isinstance(result, Image.Image)
    # test with invalid media type
    with pytest.raises(ValueError):
        media_obj = MediaObject(type="invalid", path=Path("test_image.png"), options={})
        processor.media_preprocessor.preprocess_input(media_obj=media_obj)
    Path("test_image.png").unlink()

    with pytest.raises(TypeError):
        # Test with non-MediaObject input
        processor.media_preprocessor.preprocess_input(media_obj="not_media_object")

    with pytest.raises(FileNotFoundError):
        # Test with non-existent file
        media_obj = MediaObject(type="invalid", path="124q51q1q.png", options={})
        processor.media_preprocessor.preprocess_input(media_obj=media_obj)


def test_audio_preprocessing(test_data, media_model_mappings):
    """Test audio preprocessing specifically."""
    processor = MultiModalProcessorTransformers(
        model_id=media_model_mappings["automatic-speech-recognition"], task="automatic-speech-recognition"
    )

    media_obj = MediaObject(type="audio", path=test_data["automatic-speech-recognition"], options={})
    result = processor.media_preprocessor.preprocess_input(media_obj=media_obj)

    assert "array" in result
    assert "sampling_rate" in result
    assert isinstance(result["array"], np.ndarray)
    assert result["sampling_rate"] == 16000


def test_error_handling(media_model_mappings):
    """Test error handling for invalid inputs."""
    processor = MultiModalProcessorTransformers(model_id=media_model_mappings["image-to-text"], task="image-to-text")

    # Test with non-existent file
    media_objects = {"__MEDIA_0__": MediaObject(type="image", path="non_existent_file.jpg", options={})}

    with pytest.raises(FileNotFoundError):
        processor.generate("__MEDIA_0__", media_objects=media_objects)

    # Test with invalid URL
    media_objects = {"__MEDIA_0__": MediaObject(type="image", path="https://invalid.url/image.jpg", options={})}

    with pytest.raises(requests.exceptions.RequestException):
        processor.generate("__MEDIA_0__", media_objects=media_objects)


if __name__ == "__main__":
    pytest.main(["-vvv", __file__])
