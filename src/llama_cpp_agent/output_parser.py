import json


def sanitize_and_load_json(input_json):
    try:
        # Directly load the JSON if it's already in a correct format
        return json.loads(input_json)
    except json.JSONDecodeError:
        # If there's a decode error, attempt to fix and re-parse

        # Replace problematic parts of the JSON string here
        # Note: This is a simplified approach and may need adjustments based on actual LLM output
        fixed_json = input_json.replace('\n', '\\n').replace('\"', '\\"').replace('\r', '\\r')

        # Try loading the JSON again after making replacements
        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError as e:
            # If there's still an error, print or log the error message
            print(f"Error parsing JSON: {e}")
            return None


def is_empty_or_whitespace(s):
    """
    Checks if a string is empty or contains only whitespace.

    Args:
        s (str): The input string.

    Returns:
        bool: True if the string is empty or contains only whitespace, False otherwise.
    """
    return not s.strip()


def parse_json_response(response: str):
    """
    Parses a JSON response string.

    Args:
        response (str): The JSON response string.

    Returns:
        dict: The parsed JSON object.
    """

    response_lines = response.split("\n")
    while is_empty_or_whitespace(response_lines[0]):
        response_lines.pop(0)
    response = "\n".join(response_lines)
    json_object = sanitize_and_load_json(response.strip())
    return json_object


def parse_json_response_with_markdown_code_block_or_triple_quoted_string(json_response, marker):
    """
    Parses a JSON response string followed by a Markdown code block or triple-quoted string.

    Args:
        json_response (str): The JSON response string followed by a Markdown code block or triple-quoted string.
        marker(str): Triple quotes (''') or triple back ticks (```)

    Returns:
        Tuple[dict, str]: The parsed JSON object and the content of the Markdown code block or triple-quoted string.
    """
    response_lines = json_response.split("\n")

    if is_empty_or_whitespace(response_lines[0]):
        response_lines.pop(0)

    # Get the first line JSON object
    response = response_lines[0]
    # Remove the first line
    response_lines.pop(0)
    count = len(response_lines)
    for _ in range(count):
        if response_lines[0].startswith(marker):
            break
        else:
            line = response_lines.pop(0)
            response += "\n" + line

    if len(response_lines) == 0:
        return None, None
    # Remove the first line Markdown code block marker
    response_lines.pop(0)
    # Remove the last line Markdown code block marker
    response_lines.pop(-1)
    response_lines.pop(-1)
    # Combine lines into a single string
    markdown_code_block_content = "\n".join(response_lines)
    json_object = parse_json_response(response)

    return json_object, markdown_code_block_content


def extract_object_from_response(response: str, object_clas: type):
    """
    Extracts an object of the specified class from a JSON response string.

    Args:
        response (str): The JSON response string.
        object_clas (type): The class of the object to be extracted.

    Returns:
        object_clas: An instance of the specified class.
    """
    obj = parse_json_response(response)
    cls = object_clas
    obj = cls(**obj)
    return obj
