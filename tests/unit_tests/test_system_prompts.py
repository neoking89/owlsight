import json
import os
import tempfile
import pytest
import sys
from unittest.mock import patch

sys.path.append("src")

from owlsight.prompts.system_prompts import SystemPrompts, PromptWriter


class MockSystemPrompts(SystemPrompts):
    """Mock class for testing SystemPrompts"""

    @property
    def test_role(self) -> str:
        return """
# ROLE:
Test role description

# TASK:
Some task description
"""

    other_role = """
# ROLE:
Other role description

# TASK:
Some other task description
"""


@pytest.fixture
def prompt_writer():
    """Fixture for PromptWriter tests"""
    test_prompt = "Test prompt content"
    return PromptWriter(test_prompt), test_prompt


@pytest.fixture
def mock_prompts():
    """Fixture for SystemPrompts tests"""
    return MockSystemPrompts()


def test_prompt_writer_init(prompt_writer):
    """Test PromptWriter initialization"""
    writer, test_prompt = prompt_writer
    assert writer.prompt == test_prompt


def test_prompt_writer_to_with_valid_json(prompt_writer):
    """Test writing prompt to a valid JSON file"""
    writer, _ = prompt_writer

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tf:
        # Write initial content
        json.dump({"model": {"system_prompt": "old prompt"}}, tf)
        tf.flush()
        temp_path = tf.name

    try:
        # Test writing to the file
        writer.to(temp_path)

        # Verify the content was updated
        with open(temp_path, "r") as f:
            content = json.load(f)
            assert content["model"]["system_prompt"] == writer.prompt
    finally:
        # Clean up
        os.unlink(temp_path)


def test_prompt_writer_to_with_nonexistent_file(prompt_writer: PromptWriter):
    """Test writing to a nonexistent file raises FileNotFoundError"""
    writer, _ = prompt_writer
    with pytest.raises(FileNotFoundError):
        writer.to("/nonexistent/path/file.json")


def test_prompt_writer_to_with_invalid_json(prompt_writer):
    """Test writing to a file with invalid JSON content"""
    writer, _ = prompt_writer
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tf:
        tf.write("invalid json content")
        tf.flush()
        temp_path = tf.name

    try:
        with pytest.raises(ValueError):
            writer.to(temp_path)
    finally:
        os.unlink(temp_path)


def test_system_prompts_list_roles(mock_prompts: SystemPrompts):
    """Test that list_roles returns all available roles"""
    roles = mock_prompts.list_roles()
    # Check that we have the mock roles
    assert "test_role" in roles
    assert "other_role" in roles
    # Check that all roles are strings and don't start with underscore
    for role in roles:
        assert isinstance(role, str)
        assert not role.startswith("_")


def test_system_prompts_as_dict(mock_prompts: SystemPrompts):
    """Test converting roles to dictionary"""
    roles_dict = mock_prompts.as_dict()

    # Check that we have expected roles
    assert "test_role" in roles_dict
    assert "other_role" in roles_dict

    # Check content of roles
    assert "Test role description" in roles_dict["test_role"]
    assert "Other role description" in roles_dict["other_role"]


def test_system_prompts_getattr_with_valid_role(mock_prompts: SystemPrompts):
    """Test accessing valid roles through __getattr__"""
    # Test property role
    role = mock_prompts.test_role
    assert isinstance(role, str)
    assert "Test role description" in role

    # Test string role
    role = mock_prompts.other_role
    assert isinstance(role, str)
    assert "Other role description" in role


def test_system_prompts_getattr_with_invalid_role(mock_prompts: SystemPrompts):
    """Test accessing invalid roles through __getattr__"""
    with pytest.raises(AttributeError) as exc_info:
        _ = mock_prompts.nonexistent_role
    assert "Available roles are:" in str(exc_info.value)


def test_show_available_tools(mock_prompts: SystemPrompts):
    """Test that show_available_tools instantiates SingletonDict and returns a string"""
    with patch('owlsight.prompts.system_prompts.SingletonDict') as global_dict_mock:
        # Test without providing globals_dict
        result = mock_prompts.show_available_tools()
        assert isinstance(result, str)
        assert global_dict_mock.call_count == 1, "SingletonDict should be initialized exactly once"

        global_dict_mock.reset_mock()

        # Test with providing globals_dict
        result = mock_prompts.show_available_tools(global_dict_mock)
        assert isinstance(result, str)
        assert global_dict_mock.call_count == 0, "SingletonDict should not be initialized"
    
if __name__ == "__main__":
    pytest.main([__file__])
