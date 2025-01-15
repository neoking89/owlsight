import sys
from io import StringIO
import pytest
from unittest.mock import patch, MagicMock

sys.path.append("src")
from owlsight.app.run_app import run_code_generation_loop
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor


@pytest.fixture
def manager():
    """Fixture for TextGenerationManager mock"""
    return MagicMock(spec=TextGenerationManager)


@pytest.fixture
def code_executor():
    """Fixture for CodeExecutor mock"""
    return MagicMock(spec=CodeExecutor)


@pytest.fixture
def captured_stdout(monkeypatch):
    """Fixture to capture stdout"""
    string_io = StringIO()
    monkeypatch.setattr(sys, 'stdout', string_io)
    return string_io


def test_main_menu_navigation(manager, code_executor):
    """Test that the main menu navigation works correctly"""
    with patch('owlsight.app.run_app.get_user_input') as mock_get_user_input, \
         patch('owlsight.app.run_app.logger') as mock_logger:
        
        # Mock user selecting 'quit' option
        mock_get_user_input.return_value = ('quit', None)
        
        try:
            run_code_generation_loop(code_executor, manager)
        except SystemExit:
            pass  # Expected behavior when quitting

        # Verify the get_user_input was called with correct start_index
        mock_get_user_input.assert_called_with(start_index=0)
        
        # Verify that logger.info was called with "Quitting..."
        mock_logger.info.assert_called_once_with("Quitting...")


# @pytest.mark.parametrize("command,expected_calls", [
#     ('config', [('config', 'config')]),
#     ('clear history', [('clear history', 'clear history')]),
#     ('python', [('python', 'python')]),
#     ('shell', [('shell', 'shell')])
# ])
# def test_special_command_handling(command, expected_calls, manager, code_executor):
#     """Test that special commands are properly handled"""
#     with patch('owlsight.app.run_app.get_user_input') as mock_get_user_input, \
#          patch('owlsight.app.run_app.handle_special_commands') as mock_handle_special:
        
#         mock_get_user_input.side_effect = [
#             (command, None),  # First return the command
#             ('quit', None)    # Then quit to exit loop
#         ]
#         mock_handle_special.return_value = CommandResult.CONTINUE

#         try:
#             run_code_generation_loop(code_executor, manager)
#         except SystemExit:
#             pass

#         # Verify handle_special_commands was called with correct arguments
#         assert mock_handle_special.call_args_list == [
#             pytest.call(call[0], call[1], code_executor, manager) 
#             for call in expected_calls
#         ]


# @pytest.mark.parametrize("question", [
#     "How do I create a list in Python?",
#     "What is a dictionary?",
#     "How to use pandas?"
# ])
# def test_question_processing(question, manager, code_executor):
#     """Test that user questions are properly processed"""
#     with patch('owlsight.app.run_app.get_user_input') as mock_get_user_input, \
#          patch('owlsight.app.run_app.process_user_question') as mock_process_question:
        
#         mock_get_user_input.side_effect = [
#             (None, question),  # Return the question
#             ('quit', None)     # Then quit to exit loop
#         ]

#         try:
#             run_code_generation_loop(code_executor, manager)
#         except SystemExit:
#             pass

#         # Verify the question was processed
#         mock_process_question.assert_called_with(question, code_executor, manager)


# def test_clear_history_command(manager, code_executor):
#     """Test that the clear history command works"""
#     with patch('owlsight.app.run_app.get_user_input') as mock_get_user_input, \
#          patch('owlsight.app.run_app.clear_history') as mock_clear_history:
        
#         mock_get_user_input.side_effect = [
#             ('clear history', None),  # First return clear history
#             ('quit', None)           # Then quit to exit loop
#         ]

#         try:
#             run_code_generation_loop(code_executor, manager)
#         except SystemExit:
#             pass

#         # Verify clear_history was called
#         mock_clear_history.assert_called_once_with(code_executor, manager)


# def test_main_menu_options_validity():
#     """Test that all main menu options are valid and properly formatted"""
#     # Verify that required options exist in MAIN_MENU
#     required_options = {'quit', 'config', 'clear history', 'python', 'shell'}
#     assert all(opt in MAIN_MENU.keys() for opt in required_options)

#     # Verify each menu item has a description
#     for key, value in MAIN_MENU.items():
#         assert isinstance(key, str)
#         assert isinstance(value, str)
#         assert len(value) > 0

if __name__ == "__main__":
    pytest.main([__file__])