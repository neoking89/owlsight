"""
Simple tests for TextGenerationManager.load_model_processor method.
Testing specifically for:
1. When it returns an exception
2. When it returns None (both for missing model_id and success cases)
"""
import pytest
from unittest.mock import MagicMock, patch

# Simple mock class for tests
class MockException(Exception):
    pass

def test_load_model_processor_returns_exception():
    """Test that load_model_processor returns exception when processor initialization fails."""
    # Import inside function to avoid import errors during collection
    from owlsight.processors.text_generation_manager import TextGenerationManager
    
    # Mock the config manager
    config_manager = MagicMock()
    config_manager.get.side_effect = lambda key, default=None: {
        "model.model_id": "test_model",
        "model": {"model_id": "test_model"},
        "huggingface.task": "text-generation",
        "agentic.active": False,
    }.get(key, default)
    
    # Reset the singleton and create instance
    TextGenerationManager._reset_instance()
    manager = TextGenerationManager(config_manager)
    
    # Force an exception during processor creation
    with patch("owlsight.processors.text_generation_manager.select_processor_type") as mock_select:
        # Return a class that will raise an exception when instantiated
        mock_processor_class = MagicMock()
        mock_processor_class.side_effect = MockException("Test exception")
        mock_select.return_value = mock_processor_class
        
        # Call the method
        result = manager.load_model_processor()
        
        # Check the result
        assert isinstance(result, Exception)
        assert "Test exception" in str(result)

def test_load_model_processor_returns_none_when_no_model_id():
    """Test that load_model_processor returns None when no model_id is provided."""
    # Import inside function
    from owlsight.processors.text_generation_manager import TextGenerationManager
    
    # Mock the config manager to return an empty model_id
    config_manager = MagicMock()
    config_manager.get.side_effect = lambda key, default=None: {
        "model.model_id": "",  # Empty model_id
        "model": {},
        "huggingface.task": "text-generation",
        "agentic.active": False,
    }.get(key, default)
    
    # Reset the singleton and create instance
    TextGenerationManager._reset_instance()
    manager = TextGenerationManager(config_manager)
    
    # Mock the logger to verify error is logged
    with patch("owlsight.processors.text_generation_manager.logger") as mock_logger:
        # Call the method
        result = manager.load_model_processor()
        
        # Check the results
        assert result is None
        mock_logger.error.assert_called_once_with(
            "No model_id provided. Please set a model_id in the configuration."
        )

if __name__ == "__main__":
    # Run directly if needed
    pytest.main(["-xvs", __file__])
