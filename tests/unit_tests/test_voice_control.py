import pytest
from unittest.mock import Mock, call
import queue

from owlsight.voice.voice_control import VoiceControl

@pytest.fixture
def voice_control():
    """Create a VoiceControl instance with mocked queues and callback."""
    vc = VoiceControl(
        word_to_key_map={
            "left": "left",
            "right": "right",
            "select all": ["ctrl", "a"],
            "delete": "delete"
        },
        word_to_word_map={
            "exit": "exit()",
            "print": "print()"
        },
        debug=True
    )
    
    # Mock the queues
    vc.key_press_queue = Mock(spec=queue.Queue)
    vc.typing_queue = Mock(spec=queue.Queue)
    
    # Mock the callback
    vc.on_command_processed = Mock()
    
    return vc

@pytest.mark.parametrize("input_text, expected_output", [
    # Basic word transformations
    ("exit", "exit()"),
    ("print", "print()"),
    
    # With punctuation
    ("exit.", "exit()"),
    ("exit!", "exit()"),
    
    # Mixed case
    ("EXIT", "exit()"),
    ("ExIt", "exit()"),
    
    # In a sentence
    ("I want to exit.", "I want to exit()"),
    
    # Multiple words
    ("exit print", "exit() print()"),
])
def test_word_transformations(voice_control, input_text, expected_output):
    """Test various word transformation scenarios."""
    voice_control._process_text(input_text)
    voice_control.typing_queue.put.assert_called_once_with(expected_output)

# Test commands
@pytest.mark.parametrize("input_text, expected_commands", [
    ("left", [("left", "left")]),
    ("select all", [("select all", ["ctrl", "a"])]),
])
def test_commands(voice_control, input_text, expected_commands):
    """Test command recognition and key press queueing."""
    voice_control._process_text(input_text)
    
    expected_calls = [call(key) for _, key in expected_commands]
    voice_control.key_press_queue.put.assert_has_calls(expected_calls)
    assert voice_control.key_press_queue.put.call_count == len(expected_commands)
    
    expected_callback_calls = [call(word, key) for word, key in expected_commands]
    voice_control.on_command_processed.assert_has_calls(expected_callback_calls)
    assert voice_control.on_command_processed.call_count == len(expected_commands)