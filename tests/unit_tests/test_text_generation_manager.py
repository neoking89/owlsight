import pytest
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.configurations.config_manager import ConfigManager


@pytest.fixture
def config_manager():
    return ConfigManager()


@pytest.fixture(autouse=True)
def reset_text_generation_manager():
    TextGenerationManager._reset_instance()
    yield


def test_singleton_behavior(config_manager):
    """Test that TextGenerationManager enforces singleton behavior."""
    manager1 = TextGenerationManager(config_manager)
    assert isinstance(manager1, TextGenerationManager)

    with pytest.raises(RuntimeError) as exc_info:
        TextGenerationManager(config_manager)

    assert str(exc_info.value) == "Only one instance of TextGenerationManager can exist at the same time."