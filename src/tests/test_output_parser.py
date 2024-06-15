import pytest
from llama_cpp_agent.output_parser import (
    sanitize_and_load_json,
    is_empty_or_whitespace,
    parse_json_response,
    parse_json_response_with_markdown_code_block_or_triple_quoted_string,
    extract_object_from_response,
)


def test_sanitize_and_load_json_correct_format():
    # Test JSON in correct format
    input_json = '{"key": "value"}'
    expected_output = {"key": "value"}
    assert sanitize_and_load_json(input_json) == expected_output


def test_sanitize_and_load_json_with_newlines():
    # Test JSON with newlines
    input_json = '{"key": "value\nwith newline"}'
    expected_output = {"key": "value with newline"}
    assert sanitize_and_load_json(input_json) == expected_output


def test_is_empty_or_whitespace():
    # Test empty string
    assert is_empty_or_whitespace("") is True
    # Test string with only whitespace
    assert is_empty_or_whitespace("   ") is True
    # Test non-empty string
    assert is_empty_or_whitespace("hello") is False


def test_parse_json_response():
    # Test JSON response parsing
    response = '{"key": "value"}'
    expected_output = {"key": "value"}
    assert parse_json_response(response) == expected_output


def test_parse_json_response_with_markdown_code_block_or_triple_quoted_string():
    # Test parsing JSON response with Markdown code block or triple-quoted string
    response = """{"key": "value"}
    ```python
    print("Hello, world!")
    ```"""
    expected_json = {"key": "value"}
    expected_markdown = 'print("Hello, world!")'
    json_obj, markdown_content = parse_json_response_with_markdown_code_block_or_triple_quoted_string(response, "```")
    assert json_obj == expected_json
    assert markdown_content == expected_markdown


def test_extract_object_from_response():
    # Test extracting object from JSON response
    class TestObject:
        def __init__(self, key):
            self.key = key

    response = '{"key": "value"}'
    expected_object = TestObject("value")
    extracted_object = extract_object_from_response(response, TestObject)
    assert vars(extracted_object) == vars(expected_object)
