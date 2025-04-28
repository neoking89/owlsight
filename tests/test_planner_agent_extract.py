from unittest.mock import MagicMock, patch
import pytest

from owlsight.agentic.core import PlanAgent



class TestPlanAgentExtract:
    """Tests for the _extract method in PlanAgent."""

    @pytest.fixture
    def plan_agent(self):
        """Fixture providing a PlanAgent instance."""
        return PlanAgent()

    @pytest.fixture
    def logger_mock(self):
        """Fixture providing a mock for the logger."""
        return MagicMock()
        
    @patch('owlsight.agentic.core.logger')
    def test_valid_json_extraction(self, mock_logger, plan_agent):
        """Test extraction of a valid JSON response with correct format."""
        # Create a valid JSON response as if from the LLM
        valid_json = """
```json
{
  "plan": [
    {
      "description": "Search the web for recent news on AI advancements",
      "agent": "ToolSelectionAgent",
      "reason": "To gather up-to-date information on recent AI developments"
    },
    {
      "description": "Provide the final answer to the user",
      "agent": "FinalAgent",
      "reason": "To synthesize the information and answer the user's question"
    }
  ]
}
```
"""
        result = plan_agent._extract(valid_json)
        assert len(result) == 2
        assert result[0].description == "Search the web for recent news on AI advancements"
        assert result[0].agent_name == "ToolSelectionAgent"
        assert result[0].reason == "To gather up-to-date information on recent AI developments"
        assert result[1].agent_name == "FinalAgent"

    @patch('owlsight.agentic.helper_functions.logger')
    def test_valid_json_without_markdown(self, mock_logger, plan_agent):
        """Test extraction from a valid JSON string without markdown formatting."""
        json_without_markdown = """{
          "plan": [
            {
              "description": "Create a function to calculate Fibonacci numbers",
              "agent": "ToolCreationAgent",
              "reason": "The user requested a function to calculate Fibonacci numbers"
            },
            {
              "description": "Test the created Fibonacci function",
              "agent": "ToolSelectionAgent", 
              "reason": "To verify the function works correctly"
            },
            {
              "description": "Provide the final answer to the user",
              "agent": "FinalAgent",
              "reason": "To explain the implementation and provide the results"
            }
          ]
        }"""
        
        result = plan_agent._extract(json_without_markdown)
        assert len(result) == 3
        assert result[0].agent_name == "ToolCreationAgent"
        assert result[1].agent_name == "ToolSelectionAgent"
        assert result[2].agent_name == "FinalAgent"
        # Verify that warning was logged
        mock_logger.warning.assert_called_once()

    @patch('owlsight.agentic.core.logger')
    def test_invalid_json_format(self, mock_logger, plan_agent):
        """Test extraction from an invalid JSON string."""
        invalid_json = """
```json
{
  "invalid_key": [
    {
      "not_a_plan": "This should fail"
    }
  ]
}
```
"""
        result = plan_agent._extract(invalid_json)
        assert len(result) == 0, "Should return empty list for invalid JSON format"

    @patch('owlsight.agentic.core.logger')
    def test_completely_invalid_json(self, mock_logger, plan_agent):
        """Test extraction from a completely malformed JSON string."""
        completely_invalid = """
```json
This is not JSON at all!
```
"""
        result = plan_agent._extract(completely_invalid)
        assert len(result) == 0, "Should return empty list for malformed JSON"

    @patch('owlsight.agentic.core.logger')
    def test_empty_json(self, mock_logger, plan_agent):
        """Test extraction from an empty JSON object."""
        empty_json = """```json
{}
```"""
        result = plan_agent._extract(empty_json)
        assert len(result) == 0, "Should return empty list for empty JSON"

    @patch('owlsight.agentic.core.logger')
    def test_missing_required_fields(self, mock_logger, plan_agent):
        """Test extraction when required fields are missing from steps."""
        missing_fields = """
```json
{
  "plan": [
    {
      "description": "Search the web"
    },
    {
      "agent": "FinalAgent",
      "reason": "To provide the final answer"
    }
  ]
}
```
"""
        result = plan_agent._extract(missing_fields)
        assert len(result) == 0, "Should return empty list when required fields are missing"

    @patch('owlsight.agentic.core.logger')
    def test_valid_plan_with_partial_fields(self, mock_logger, plan_agent):
        """Test extraction when some steps have all required fields but others don't."""
        partial_fields = """
```json
{
  "plan": [
    {
      "description": "Search for information",
      "agent": "ToolSelectionAgent",
      "reason": "To find relevant data"
    },
    {
      "description": "Process the results",
      "agent": "ToolSelectionAgent"
    }
  ]
}
```
"""
        result = plan_agent._extract(partial_fields)
        assert len(result) == 2
        assert result[0].reason == "To find relevant data"
        assert result[1].reason == "", "Optional reason field should default to empty string"

    @patch('owlsight.agentic.core.logger')
    def test_invalid_agent_names(self, mock_logger, plan_agent):
        """Test extraction when agent names are not in the allowed set."""
        # Mock the error log
        mock_logger.error = MagicMock()
        
        invalid_agents = """
```json
{
  "plan": [
    {
      "description": "Search for information",
      "agent": "InvalidAgent",
      "reason": "This agent doesn't exist"
    },
    {
      "description": "Provide the final answer",
      "agent": "FinalAgent",
      "reason": "Valid agent"
    }
  ]
}
```
"""
        result = plan_agent._extract(invalid_agents)
        assert len(result) == 0, "Should return empty list when any agent name is invalid"
        # Verify that error was logged
        mock_logger.error.assert_called_once()

    @patch('owlsight.agentic.core.logger')
    def test_multiple_markdown_blocks(self, mock_logger, plan_agent):
        """Test extraction when there are multiple markdown blocks in the response."""
        multiple_blocks = """
Here's some explanation text.

```python
# This is a code block that should be ignored
def hello():
    print("Hello")
```

And here's the actual plan:

```json
{
  "plan": [
    {
      "description": "Create a tool",
      "agent": "ToolCreationAgent",
      "reason": "To implement the requested functionality"
    },
    {
      "description": "Provide the final answer",
      "agent": "FinalAgent",
      "reason": "To explain the implementation"
    }
  ]
}
```
"""
        result = plan_agent._extract(multiple_blocks)
        assert len(result) == 2
        assert result[0].agent_name == "ToolCreationAgent"
        assert result[1].agent_name == "FinalAgent"

    @patch('owlsight.agentic.core.logger')
    def test_empty_plan_list(self, mock_logger, plan_agent):
        """Test extraction when the plan list is empty."""
        empty_plan = """
```json
{
  "plan": []
}
```
"""
        result = plan_agent._extract(empty_plan)
        assert len(result) == 0, "Should return empty list for empty plan"

    @patch('owlsight.agentic.core.logger')
    def test_plan_not_list(self, mock_logger, plan_agent):
        """Test extraction when the plan is not a list."""
        plan_not_list = """
```json
{
  "plan": "This is not a list"
}
```
"""
        result = plan_agent._extract(plan_not_list)
        assert len(result) == 0, "Should return empty list when plan is not a list"

    @patch('owlsight.agentic.helper_functions.logger')
    def test_missing_markdown_code_block(self, mock_logger, plan_agent):
        """Test extraction when there's no markdown code block."""
        no_code_block = "This response has no code blocks at all."
        result = plan_agent._extract(no_code_block)
        assert len(result) == 0, "Should return empty list when no code blocks exist"
        # Verify that warning was logged
        mock_logger.warning.assert_called_once()

    @patch('owlsight.agentic.core.logger')
    def test_mixed_valid_and_invalid_steps(self, mock_logger, plan_agent):
        """Test extraction when some steps are valid and others are invalid."""
        # Mock the error log
        mock_logger.error = MagicMock()
        
        mixed_steps = """
```json
{
  "plan": [
    {
      "description": "Valid step",
      "agent": "ToolSelectionAgent",
      "reason": "Valid reason"
    },
    {
      "description": "Invalid step",
      "agent": "NonExistentAgent",
      "reason": "Invalid agent"
    },
    {
      "description": "Another valid step",
      "agent": "FinalAgent",
      "reason": "Valid final step"
    }
  ]
}
```
"""
        result = plan_agent._extract(mixed_steps)
        assert len(result) == 0, "Should return empty list when any step has an invalid agent"
        # Verify that error was logged
        mock_logger.error.assert_called_once()
    
    @patch('owlsight.agentic.core.logger')
    def test_toolexecution_follows_toolcreation_guardrail(self, mock_logger, plan_agent):
        """Test extraction that would trigger the ToolExecutionFollowsToolCreationGuardrail."""
        # This test verifies the extraction of a plan that would later trigger the guardrail 
        # (where ToolSelectionAgent tries to use a tool that hasn't been created yet)
        incorrect_order = """
```json
{
  "plan": [
    {
      "description": "Execute the fibonacci function to calculate the 10th Fibonacci number",
      "agent": "ToolSelectionAgent",
      "reason": "To get the result of the Fibonacci calculation"
    },
    {
      "description": "Create a function to calculate Fibonacci numbers",
      "agent": "ToolCreationAgent",
      "reason": "To implement the Fibonacci algorithm"
    },
    {
      "description": "Provide the final answer",
      "agent": "FinalAgent",
      "reason": "To explain the implementation and show the result"
    }
  ]
}
```
"""
        result = plan_agent._extract(incorrect_order)
        # The extraction itself should work, even though the plan is logically flawed
        assert len(result) == 3
        assert result[0].agent_name == "ToolSelectionAgent"
        assert result[1].agent_name == "ToolCreationAgent"
        assert result[2].agent_name == "FinalAgent"
        
    @patch('owlsight.agentic.core.logger')
    def test_final_agent_is_last_guardrail(self, mock_logger, plan_agent):
        """Test extraction that would trigger the FinalAgentIsLastGuardrail."""
        # This test verifies the extraction of a plan that would later trigger the guardrail
        # (where FinalAgent is not the last step)
        final_agent_not_last = """
```json
{
  "plan": [
    {
      "description": "Create a function to calculate Fibonacci numbers",
      "agent": "ToolCreationAgent",
      "reason": "To implement the Fibonacci algorithm"
    },
    {
      "description": "Provide a preliminary answer",
      "agent": "FinalAgent",
      "reason": "To explain what will be done"
    },
    {
      "description": "Execute the fibonacci function",
      "agent": "ToolSelectionAgent",
      "reason": "To get the result of the calculation"
    }
  ]
}
```
"""
        result = plan_agent._extract(final_agent_not_last)
        # The extraction itself should work, even though the plan is logically flawed
        assert len(result) == 3
        assert result[0].agent_name == "ToolCreationAgent"
        assert result[1].agent_name == "FinalAgent"
        assert result[2].agent_name == "ToolSelectionAgent"
    
    @patch('owlsight.agentic.core.logger')
    def test_json_with_extra_fields(self, mock_logger, plan_agent):
        """Test extraction when JSON contains extra fields that should be ignored."""
        extra_fields = """
```json
{
  "plan": [
    {
      "description": "Search for information",
      "agent": "ToolSelectionAgent",
      "reason": "To find relevant data",
      "extra_field": "This should be ignored",
      "another_extra": 123
    },
    {
      "description": "Provide the final answer",
      "agent": "FinalAgent",
      "reason": "To synthesize the information",
      "irrelevant": "This doesn't matter"
    }
  ],
  "extra_top_level": "This should also be ignored"
}
```
"""
        result = plan_agent._extract(extra_fields)
        assert len(result) == 2
        assert result[0].agent_name == "ToolSelectionAgent"
        assert result[1].agent_name == "FinalAgent"
        # Extra fields should be ignored and not affect the extraction
    
    @patch('owlsight.agentic.helper_functions.logger')
    def test_malformed_markdown_format(self, mock_logger, plan_agent):
        """Test extraction when markdown format is malformed but JSON is valid."""
        malformed_markdown = """
Here's the plan:
json
{
  "plan": [
    {
      "description": "Create a function",
      "agent": "ToolCreationAgent",
      "reason": "To implement the functionality"
    },
    {
      "description": "Provide the final answer",
      "agent": "FinalAgent",
      "reason": "To explain the implementation"
    }
  ]
}
```
"""  # Note the missing opening backticks
        result = plan_agent._extract(malformed_markdown)
        # Should attempt to parse as JSON directly
        assert len(result) == 0, "Should fail with malformed markdown"
        # Verify that warning was logged
        mock_logger.warning.assert_called_once()
    
    @patch('owlsight.agentic.core.logger')
    def test_json_with_comments(self, mock_logger, plan_agent):
        """Test extraction when JSON contains comments that would be invalid in standard JSON."""
        json_with_comments = """
```json
{
  "plan": [
    {
      "description": "Search for information",
      "agent": "ToolSelectionAgent",
      "reason": "To find relevant data"
    },
    {
      "description": "Provide the final answer",
      "agent": "FinalAgent",
      "reason": "To synthesize the information"
    }
  ]
}
```
"""
        # Manually modify the json string to insert an invalid comment
        json_with_comments = json_with_comments.replace('"reason": "To find relevant data"', 
                                                       '// This is a comment that would break standard JSON parsing\n      "reason": "To find relevant data"')
        
        result = plan_agent._extract(json_with_comments)
        assert len(result) == 0, "Should return empty list for JSON with comments"
    
    @patch('owlsight.agentic.core.logger')
    def test_large_complex_plan(self, mock_logger, plan_agent):
        """Test extraction of a large complex plan with many steps."""
        large_plan = """
```json
{
  "plan": [
    {
      "description": "Create a function to scrape weather data",
      "agent": "ToolCreationAgent",
      "reason": "To implement a custom scraper for weather information"
    },
    {
      "description": "Search for weather APIs",
      "agent": "ToolSelectionAgent",
      "reason": "To find available weather data sources"
    },
    {
      "description": "Create a function to parse weather data",
      "agent": "ToolCreationAgent",
      "reason": "To process the data from weather APIs"
    },
    {
      "description": "Execute the weather scraper function",
      "agent": "ToolSelectionAgent",
      "reason": "To retrieve current weather data"
    },
    {
      "description": "Execute the weather parser function",
      "agent": "ToolSelectionAgent",
      "reason": "To format the retrieved weather data"
    },
    {
      "description": "Create a function to generate weather visualizations",
      "agent": "ToolCreationAgent",
      "reason": "To create visual representations of the weather data"
    },
    {
      "description": "Execute the visualization function",
      "agent": "ToolSelectionAgent",
      "reason": "To generate the weather visualization"
    },
    {
      "description": "Provide the final weather report to the user",
      "agent": "FinalAgent",
      "reason": "To present the complete weather analysis with visualizations"
    }
  ]
}
```
"""
        result = plan_agent._extract(large_plan)
        assert len(result) == 8
        assert result[0].agent_name == "ToolCreationAgent"
        assert result[-1].agent_name == "FinalAgent"
        # Check that all steps were properly extracted
        for step in result:
            assert step.description, "Each step should have a description"
            assert step.agent_name, "Each step should have an agent name"
            assert step.reason, "Each step should have a reason"
    
    @patch('owlsight.agentic.core.logger')
    def test_handling_of_excessive_whitespace(self, mock_logger, plan_agent):
        """Test extraction when there's excessive whitespace in the JSON."""
        excessive_whitespace = """
```json


   {

      "plan"   :   [
         {
            "description"  :   "Search for information"   ,
            "agent"   :   "ToolSelectionAgent"   ,
            "reason"   :   "To find relevant data"
         }   ,
         {
            "description"   :   "Provide the final answer"   ,
            "agent"   :   "FinalAgent"   ,
            "reason"   :   "To synthesize the information"
         }
      ]

   }


```
"""
        result = plan_agent._extract(excessive_whitespace)
        assert len(result) == 2
        assert result[0].agent_name == "ToolSelectionAgent"
        assert result[1].agent_name == "FinalAgent"
        # Excessive whitespace should not affect the extraction
