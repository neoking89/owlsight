import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append("src")

import pytest

from owlsight.utils.helper_functions import replace_bracket_placeholders


# Fixture for test data
@pytest.fixture
def test_context():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["x", "y", "z"]})

    test_array = np.array([1, 2, 3])
    test_dict = {"name": "Alice", "scores": [85, 90, 95]}
    test_date = datetime(2024, 1, 1)

    return {
        "pd": pd,
        "np": np,
        "df": df,
        "test_array": test_array,
        "test_dict": test_dict,
        "test_date": test_date,
        "datetime": datetime,
        "timedelta": timedelta,
    }


# Basic expressions
@pytest.mark.parametrize(
    "test_id, input_string, expected",
    [
        ("basic_1", "{{1 + 1}}", 2),
        ("basic_2", "Result: {{2 * 3}}", "Result: 6"),
    ],
)
def test_basic_expressions(test_context, test_id, input_string, expected):
    result = replace_bracket_placeholders(input_string, test_context)
    assert result == expected


# Data structures
@pytest.mark.parametrize(
    "test_id, input_string, expected",
    [
        ("struct_1", "{{{1, 2, 3}}}", {1, 2, 3}),
        ("struct_2", "{{[x for x in range(3)]}}", [0, 1, 2]),
            ("struct_3", "{{{'key': 5}}}", {"key": 5}),
        ("struct_4", "{{(1, 2, 3)}}", (1, 2, 3)),
    ],
)
def test_data_structures(test_context, test_id, input_string, expected):
    result = replace_bracket_placeholders(input_string, test_context)
    assert result == expected


# String operations
@pytest.mark.parametrize(
    "test_id, input_string, expected",
    [
        ("str_1", "{{'hello'.upper()}}", "HELLO"),
        ("str_2", '{{", ".join(["a", "b", "c"])}}', "a, b, c"),
    ],
)
def test_string_operations(test_context, test_id, input_string, expected):
    result = replace_bracket_placeholders(input_string, test_context)
    assert result == expected


# Dictionary and complex operations
@pytest.mark.parametrize(
    "test_id, input_string, expected",
    [
        ("dict_1", '{{test_dict["name"]}} scored {{max(test_dict["scores"])}}', "Alice scored 95"),
        (
            "dict_2",
            '{{test_dict["name"]}} has average score of {{sum(test_dict["scores"])/len(test_dict["scores"])}}',
            "Alice has average score of 90.0",
        ),
    ],
)
def test_dict_operations(test_context, test_id, input_string, expected):
    result = replace_bracket_placeholders(input_string, test_context)
    assert result == expected


# Pandas operations
@pytest.mark.parametrize(
    "test_id, input_string, expected",
    [
        ("pd_1", '{{df["A"].mean()}}', 2.0),
        ("pd_2", '{{df["B"].max()}}', 6),
        ("pd_3", '{{df["C"].value_counts().to_dict()}}', {"x": 1, "y": 1, "z": 1}),
        ("pd_4", 'Average of A: {{df["A"].mean()}}', "Average of A: 2.0"),
        ("pd_5", '{{df.groupby("C")["A"].mean().to_dict()}}', {"x": 1.0, "y": 2.0, "z": 3.0}),
    ],
)
def test_pandas_operations(test_context, test_id, input_string, expected):
    result = replace_bracket_placeholders(input_string, test_context)
    assert result == expected


# Numpy operations
@pytest.mark.parametrize(
    "test_id, input_string, expected",
    [
        ("np_1", "{{np.mean(test_array)}}", 2.0),
        ("np_2", "{{np.sum(test_array)}}", 6),
    ],
)
def test_numpy_operations(test_context, test_id, input_string, expected):
    result = replace_bracket_placeholders(input_string, test_context)
    assert result == expected


# Date operations
@pytest.mark.parametrize(
    "test_id, input_string, expected",
    [
        ("date_1", "{{test_date.year}}", 2024),
        ("date_2", "{{test_date + timedelta(days=1)}}", datetime(2024, 1, 2)),
    ],
)
def test_date_operations(test_context, test_id, input_string, expected):
    result = replace_bracket_placeholders(input_string, test_context)
    assert result == expected


# Error handling
@pytest.mark.parametrize(
    "test_id, input_string, expected_error",
    [
        ("error_1", "{{undefined_variable}}", NameError),
        ("error_2", "{{1/0}}", ZeroDivisionError),
        ("error_3", '{{int("not a number")}}', ValueError),
    ],
)
def test_error_handling(test_id, input_string, expected_error):
    with pytest.raises(expected_error):
        replace_bracket_placeholders(input_string, {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
