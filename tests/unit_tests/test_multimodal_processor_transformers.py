import pytest

# --- PIL Setup ---
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: Pillow (PIL) library not found. Image processing features will be unavailable.")

# --- Pytesseract Setup ---
PYTESSERACT_CONFIGURED = False
_pytesseract_module = None
_TesseractNotFoundError = None # Will hold the actual or a mock TesseractNotFoundError

try:
    import pytesseract as pt_actual # Try importing pytesseract
    _pytesseract_module = pt_actual
    _TesseractNotFoundError = _pytesseract_module.TesseractNotFoundError # Get the real exception type

    # If pytesseract imported, try to configure it using the project's helper
    try:
        from owlsight.multimodal.tesseract import find_tesseract_installation
        tesseract_cmd_path = find_tesseract_installation()
        if tesseract_cmd_path:
            _pytesseract_module.pytesseract.tesseract_cmd = tesseract_cmd_path
            PYTESSERACT_CONFIGURED = True
        else:
            # This case means find_tesseract_installation returned None or empty
            print("Warning: Pytesseract imported, but Tesseract OCR executable not found by find_tesseract_installation(). Pytesseract remains unconfigured.")
    except ImportError: # Error importing find_tesseract_installation
        print("Warning: Could not import 'find_tesseract_installation' from 'owlsight.multimodal.tesseract'. Pytesseract will not be configured.")
    except _pytesseract_module.TesseractNotFoundError: # Error from find_tesseract_installation if tesseract itself not found
        print("Warning: Tesseract OCR executable not found by find_tesseract_installation() (caught TesseractNotFoundError). Pytesseract will not be configured.")
    except Exception as e_config: # Other errors during configuration
        print(f"Warning: An unexpected error occurred during Pytesseract configuration: {e_config}. Pytesseract remains unconfigured.")

except ImportError: # Error importing pytesseract itself
    print("Warning: Pytesseract library not installed. OCR-dependent tests/features will be affected.")
    class MockTesseractNotFoundError(Exception):
        pass
    _TesseractNotFoundError = MockTesseractNotFoundError # Use mock if pytesseract didn't import

# --- Pytest Skip Condition ---
# Skip all tests in this module if both PIL and Pytesseract are not available/configured.
skip_condition_both_missing = not PIL_AVAILABLE and not PYTESSERACT_CONFIGURED
skip_reason_both_missing = "Both Pillow (PIL) AND Pytesseract are not installed/configured. Skipping all multimodal tests."

pytestmark = pytest.mark.skipif(skip_condition_both_missing, reason=skip_reason_both_missing)

# --- Original Imports (some are now effectively handled or moved) ---
from pathlib import Path
import requests
import io
import ast

import numpy as np

from owlsight.huggingface.constants import HUGGINGFACE_MEDIA_TASKS
from owlsight.processors.multimodal_processors import MultiModalProcessorTransformers
from owlsight.utils.custom_classes import MediaObject
# from owlsight.multimodal.tesseract import find_tesseract_installation # Handled in setup block
# pytesseract.pytesseract.tesseract_cmd = find_tesseract_installation() # Handled in setup block


# Test URLs
TEST_CASES = [
    {
        "task": "image-to-text",
        "path": "tests/data/image-to-text.jpg",
        "question": None,
        "expected_type": list,
        "media_tag": "image",
    },
    {
        "task": "visual-question-answering",
        "path": "tests/data/visual-question-answering.jpg",
        "question": "What color is the car?",
        "expected_type": list,
        "media_tag": "image",
    },
    {
        "task": "automatic-speech-recognition",
        "path": "tests/data/automatic-speech-recognition.wav",
        "question": None,
        "expected_type": dict,
        "media_tag": "audio",
    },
    {
        "task": "document-question-answering",
        "path": "tests/data/document-question-answering.jpg",
        "question": "What is the total number of students?",
        "expected_type": list,
        "media_tag": "image",
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
        media_objects = {"__MEDIA_0__": MediaObject(tag=case["media_tag"], path=str(test_file), options={})}

        # Prepare input text based on whether there's a question
        input_text = f"{case['question']} __MEDIA_0__" if case["question"] else "__MEDIA_0__"

        result = processor.generate(input_text, media_objects=media_objects)
        result = ast.literal_eval(result)

        assert isinstance(result, case["expected_type"])
        assert len(result) > 0

    except _TesseractNotFoundError: # Use the potentially mocked/real TesseractNotFoundError
        pytest.skip(f"Tesseract is not installed or not found. Skipping {case['task']} test or install/configure it to run this test.")

    finally:
        # Clean up temporary file
        if test_file.exists():
            test_file.unlink()


def test_preprocessing(media_model_mappings):
    """Test preprocessing for different input types."""
    processor = MultiModalProcessorTransformers(model_id=media_model_mappings["image-to-text"], task="image-to-text")

    # Create test image
    if PIL_AVAILABLE:
        test_image = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        test_image.save(buffer, format="PNG")

        # Test with bytes
        media_obj = MediaObject(tag="image", path=buffer.getvalue(), options={})
        result = processor.media_preprocessor.preprocess_input(media_obj=media_obj)
        assert isinstance(result, Image.Image)

        # Test with Path
        test_image.save("test_image.png")
        media_obj = MediaObject(tag="image", path=Path("test_image.png"), options={})
        result = processor.media_preprocessor.preprocess_input(media_obj=media_obj)
        assert isinstance(result, Image.Image)
        # test with invalid media type
        with pytest.raises(ValueError):
            media_obj = MediaObject(tag="invalid", path=Path("test_image.png"), options={})
            processor.media_preprocessor.preprocess_input(media_obj=media_obj)
        Path("test_image.png").unlink()

    with pytest.raises(TypeError):
        # Test with non-MediaObject input
        processor.media_preprocessor.preprocess_input(media_obj="not_media_object")

    with pytest.raises(FileNotFoundError):
        # Test with non-existent file
        media_obj = MediaObject(tag="invalid", path="124q51q1q.png", options={})
        processor.media_preprocessor.preprocess_input(media_obj=media_obj)


def test_audio_preprocessing(test_data, media_model_mappings):
    """Test audio preprocessing specifically."""
    processor = MultiModalProcessorTransformers(
        model_id=media_model_mappings["automatic-speech-recognition"], task="automatic-speech-recognition"
    )

    media_obj = MediaObject(tag="audio", path=test_data["automatic-speech-recognition"], options={})
    result = processor.media_preprocessor.preprocess_input(media_obj=media_obj)

    assert "array" in result
    assert "sampling_rate" in result
    assert isinstance(result["array"], np.ndarray)
    assert result["sampling_rate"] == 16000


def test_error_handling(media_model_mappings):
    """Test error handling for invalid inputs."""
    processor = MultiModalProcessorTransformers(model_id=media_model_mappings["image-to-text"], task="image-to-text")

    # Test with non-existent file
    media_objects = {"__MEDIA_0__": MediaObject(tag="image", path="non_existent_file.jpg", options={})}

    with pytest.raises(FileNotFoundError):
        processor.generate("__MEDIA_0__", media_objects=media_objects)

    # Test with invalid URL
    media_objects = {"__MEDIA_0__": MediaObject(tag="image", path="https://invalid.url/image.jpg", options={})}

    with pytest.raises(requests.exceptions.RequestException):
        processor.generate("__MEDIA_0__", media_objects=media_objects)


if __name__ == "__main__":
    pytest.main(["-vvv", __file__])
