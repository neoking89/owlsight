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
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Statue_of_Liberty%2C_NY.jpg",
        "question": None,
        "expected_type": list,
        "media_type": "image",
    },
    {
        "task": "visual-question-answering",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/VW_K%C3%A4fer_Baujahr_1966.jpg/420px-VW_K%C3%A4fer_Baujahr_1966.jpg",
        "question": "What color is the car?",
        "expected_type": list,
        "media_type": "image",
    },
    {
        "task": "automatic-speech-recognition",
        "url": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
        "question": None,
        "expected_type": dict,
        "media_type": "audio",
    },
    {
        "task": "document-question-answering",
        "url": "https://vt-vtwa-assets.varsitytutors.com/vt-vtwa/uploads/problem_question_image/image/19791/table.jpg",
        "question": "What is the total number of students?",
        "expected_type": list,
        "media_type": "image",
    },
]


@pytest.fixture(scope="module")
def test_data():
    """Download and cache test data."""
    cached_data = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    for case in TEST_CASES:
        response = requests.get(case["url"], headers=headers, timeout=10)
        response.raise_for_status()
        cached_data[case["task"]] = response.content
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
    result = processor.media_preprocessor.preprocess_input(buffer.getvalue())
    assert isinstance(result, Image.Image)

    # Test with Path
    test_image.save("test_image.png")
    result = processor.media_preprocessor.preprocess_input(Path("test_image.png"))
    assert isinstance(result, Image.Image)
    Path("test_image.png").unlink()


def test_audio_preprocessing(test_data, media_model_mappings):
    """Test audio preprocessing specifically."""
    processor = MultiModalProcessorTransformers(
        model_id=media_model_mappings["automatic-speech-recognition"], task="automatic-speech-recognition"
    )

    result = processor.media_preprocessor.preprocess_input(test_data["automatic-speech-recognition"])

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
