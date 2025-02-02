import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append("src")

import pytest

from owlsight.utils.helper_functions import parse_python_placeholders, parse_media_tags
from owlsight.utils.custom_classes import MediaObject

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
    result = parse_python_placeholders(input_string, test_context)
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
    result = parse_python_placeholders(input_string, test_context)
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
    result = parse_python_placeholders(input_string, test_context)
    assert result == expected


# complex operations
@pytest.mark.parametrize(
    "test_id, input_string, expected",
    [
        ("complex_1", "{{5*2}} is more than {{2*3}}", "10 is more than 6"),
        ("complex_2", '{{" ".join([str(x) for x in range(3)])}}', "0 1 2"),
    ],
)
def test_dict_operations(test_context, test_id, input_string, expected):
    result = parse_python_placeholders(input_string, test_context)
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
    result = parse_python_placeholders(input_string, test_context)
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
    result = parse_python_placeholders(input_string, test_context)
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
    result = parse_python_placeholders(input_string, test_context)
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
        parse_python_placeholders(input_string, {})


@pytest.mark.parametrize(
    "test_id, input_string, var_dict, expected_text, expected_media",
    [
        (
            "basic_image",
            "[[image:photo.jpg]]",
            {},
            "__MEDIA_0__",
            {"__MEDIA_0__": MediaObject(tag="image", path="photo.jpg", options={})},
        ),
        (
            "basic_audio",
            "[[audio:recording.mp3]]",
            {},
            "__MEDIA_0__",
            {"__MEDIA_0__": MediaObject(tag="audio", path="recording.mp3", options={})},
        ),
        (
            "basic_video",
            "[[video:clip.mp4]]",
            {},
            "__MEDIA_0__",
            {"__MEDIA_0__": MediaObject(tag="video", path="clip.mp4", options={})},
        ),
    ],
)
def test_basic_media_syntax(test_id, input_string, var_dict, expected_text, expected_media):
    result_text, result_media = parse_media_tags(input_string, var_dict)
    assert result_text == expected_text
    assert result_media == expected_media

# Python expression integration
@pytest.mark.parametrize(
    "test_id, input_string, var_dict, expected_text, expected_media",
    [
        (
            "python_expr",
            "[[image:{{folder}}/{{filename}}]]",
            {"folder": "photos", "filename": "cat.jpg"},
            "__MEDIA_0__",
            {"__MEDIA_0__": MediaObject(tag="image", path="photos/cat.jpg", options={})},
        ),
        (
            "python_expr_options",
            "[[image:{{folder}}/test.jpg|width={{size}}]]",
            {"folder": "images", "size": 512},
            "__MEDIA_0__",
            {"__MEDIA_0__": MediaObject(tag="image", path="images/test.jpg", options={"width": "512"})},
        ),
    ],
)
def test_python_expression_integration(test_id, input_string, var_dict, expected_text, expected_media):
    result_text, result_media = parse_media_tags(input_string, var_dict)
    assert result_text == expected_text
    assert result_media == expected_media

# Multiple media objects and options
@pytest.mark.parametrize(
    "test_id, input_string, var_dict, expected_text, expected_media",
    [
        (
            "multiple_media",
            "Compare [[image:first.jpg]] with [[image:second.jpg]]",
            {},
            "Compare __MEDIA_0__ with __MEDIA_1__",
            {
                "__MEDIA_0__": MediaObject(tag="image", path="first.jpg", options={}),
                "__MEDIA_1__": MediaObject(tag="image", path="second.jpg", options={}),
            },
        ),
        (
            "complex_options",
            "[[image:photo.jpg|width=512|height=512|pipeline=depth-estimation]]",
            {},
            "__MEDIA_0__",
            {
                "__MEDIA_0__": MediaObject(
                    tag="image",
                    path="photo.jpg",
                    options={"width": "512", "height": "512", "pipeline": "depth-estimation"},
                )
            },
        ),
    ],
)
def test_multiple_media_and_options(test_id, input_string, var_dict, expected_text, expected_media):
    result_text, result_media = parse_media_tags(input_string, var_dict)
    assert result_text == expected_text
    assert result_media == expected_media

# Mixed content
@pytest.mark.parametrize(
    "test_id, input_string, var_dict, expected_text, expected_media",
    [
        (
            "mixed_content",
            "The value is {{2 + 2}} and here's an [[image:test.jpg]]",
            {},
            "The value is 4 and here's an __MEDIA_0__",
            {"__MEDIA_0__": MediaObject(tag="image", path="test.jpg", options={})},
        ),
        (
            "mixed_complex",
            """Process this [[image:{{folder}}/{{filename}}|width={{size}}]] 
            with value {{x + 1}} and [[audio:recording.mp3|language=en]]""",
            {"folder": "imgs", "filename": "test.jpg", "size": 256, "x": 5},
            """Process this __MEDIA_0__ 
            with value 6 and __MEDIA_1__""",
            {
                "__MEDIA_0__": MediaObject(tag="image", path="imgs/test.jpg", options={"width": "256"}),
                "__MEDIA_1__": MediaObject(tag="audio", path="recording.mp3", options={"language": "en"}),
            },
        ),
    ],
)
def test_mixed_content(test_id, input_string, var_dict, expected_text, expected_media):
    result_text, result_media = parse_media_tags(input_string, var_dict)
    assert result_text == expected_text
    assert result_media == expected_media

# Error handling
def test_invalid_media_type():
    with pytest.raises(ValueError):
        parse_media_tags("[[invalid:test.jpg]]", {})

def test_missing_path():
    with pytest.raises(ValueError):
        parse_media_tags("[[image:]]", {})

def test_invalid_option_format():
    with pytest.raises(ValueError):
        parse_media_tags("[[image:test.jpg|invalid_option]]", {})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])