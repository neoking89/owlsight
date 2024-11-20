from pathlib import Path
import requests
import io
import sys
sys.path.append("src")

from PIL import Image
import numpy as np
import pytest

from owlsight.hugging_face.constants import HUGGINGFACE_MEDIA_TASKS
from owlsight.processors.multimodal_processors import MultiModalProcessorTransformers

# Test URLs
TEST_CASES = [
    {
        "task": "image-to-text",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Statue_of_Liberty%2C_NY.jpg",
        "question": None,
        "expected_type": list
    },
    {
        "task": "visual-question-answering",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/VW_K%C3%A4fer_Baujahr_1966.jpg/420px-VW_K%C3%A4fer_Baujahr_1966.jpg",
        "question": "What color is the car?",
        "expected_type": list
    },
    {
        "task": "automatic-speech-recognition",
        "url": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
        "question": None,
        "expected_type": dict
    },
    {
        "task": "document-question-answering",
        "url": "https://vt-vtwa-assets.varsitytutors.com/vt-vtwa/uploads/problem_question_image/image/19791/table.jpg",
        "question": "What is the total number of students?",
        "expected_type": list
    }
]


@pytest.fixture(scope="module")
def test_data():
    """Download and cache test data."""
    cached_data = {}
    for case in TEST_CASES:
        response = requests.get(case["url"], timeout=10)
        response.raise_for_status()
        cached_data[case["task"]] = response.content
    return cached_data

@pytest.fixture(params=TEST_CASES)
def processor(request, media_model_mappings):
    """Create processor for each test case."""
    return MultiModalProcessorTransformers(
        model_id=media_model_mappings[request.param["task"]],
        task=request.param["task"]
    )

def test_media_preprocessor_initialization(media_model_mappings):
    """Test MediaPreprocessor initialization with various tasks."""
    for task in HUGGINGFACE_MEDIA_TASKS:
        processor = MultiModalProcessorTransformers(
            model_id=media_model_mappings[task],
            task=task
        )
        assert processor.task == task
        assert processor.media_preprocessor is not None
        assert processor.text_processor.pipe is not None

def test_invalid_task():
    """Test initialization with invalid task."""
    with pytest.raises(ValueError):
        MultiModalProcessorTransformers(
            model_id="test",
            task="invalid_task"
        )

@pytest.mark.parametrize("case", TEST_CASES)
def test_generate(case, test_data, media_model_mappings):
    """Test generate method for each task."""
    processor = MultiModalProcessorTransformers(
        model_id=media_model_mappings[case["task"]],
        task=case["task"]
    )
    
    result = processor.generate(
        test_data[case["task"]],
        question=case["question"]
    )
    
    assert isinstance(result, case["expected_type"])
    assert len(result) > 0

def test_preprocessing(media_model_mappings):
    """Test preprocessing for different input types."""
    processor = MultiModalProcessorTransformers(
        model_id=media_model_mappings["image-to-text"],
        task="image-to-text"
    )
    
    # Test with bytes
    test_image = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    test_image.save(buffer, format='PNG')
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
        model_id=media_model_mappings["automatic-speech-recognition"],
        task="automatic-speech-recognition"
    )
    
    result = processor.media_preprocessor.preprocess_input(
        test_data["automatic-speech-recognition"]
    )
    
    assert "array" in result
    assert "sampling_rate" in result
    assert isinstance(result["array"], np.ndarray)
    assert result["sampling_rate"] == 16000

def test_error_handling(media_model_mappings):
    """Test error handling for invalid inputs."""
    processor = MultiModalProcessorTransformers(
        model_id=media_model_mappings["image-to-text"],
        task="image-to-text"
    )
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        processor.generate("non_existent_file.jpg")
    
    # Test with invalid URL
    with pytest.raises(requests.exceptions.RequestException):
        processor.generate("https://invalid.url/image.jpg")

if __name__ == "__main__":
    pytest.main(["-vvv", __file__])