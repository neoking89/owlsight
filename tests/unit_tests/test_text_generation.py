import pytest


# Test with default response
def test_mock_text_generator_default(mock_text_generator):
    response = mock_text_generator.generate("Hello")
    assert response == "Default mock response"


# Test with single custom response
@pytest.mark.parametrize("mock_text_generator", [["Custom response"]], indirect=True)
def test_mock_text_generator_single_custom(mock_text_generator):
    response = mock_text_generator.generate("Hello")
    assert response == "Custom response"


# Test with multiple custom responses
@pytest.mark.parametrize(
    "mock_text_generator", [["Response 1", "Response 2", "Response 3"]], indirect=True
)
def test_mock_text_generator_multiple_custom(mock_text_generator):
    assert mock_text_generator.generate("test") == "Response 1"
    assert mock_text_generator.generate("test") == "Response 2"
    assert mock_text_generator.generate("test") == "Response 3"
    # It should cycle back to the first response
    assert mock_text_generator.generate("test") == "Response 1"


# Test with different responses for different test cases
@pytest.mark.parametrize(
    "mock_text_generator, expected_response",
    [
        (["Response A"], "Response A"),
        (["Response B"], "Response B"),
    ],
    indirect=["mock_text_generator"],
)
def test_mock_text_generator_different_responses(
    mock_text_generator, expected_response
):
    response = mock_text_generator.generate("test")
    assert response == expected_response


# Test history saving
@pytest.mark.parametrize(
    "mock_text_generator", [["Response X", "Response Y"]], indirect=True
)
def test_mock_text_generator_history(mock_text_generator):
    mock_text_generator.generate("Input X")
    mock_text_generator.generate("Input Y")
    assert mock_text_generator.history == [
        ("Input X", "Response X"),
        ("Input Y", "Response Y"),
    ]

if __name__ == "__main__":
    pytest.main([__file__])