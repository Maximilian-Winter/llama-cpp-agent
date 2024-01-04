import json


def sanitize_json_string(s):
    # Escaping newlines, quotes, and backslashes
    return s.replace('\n', '\\n').replace('\r', '\\r')


def is_empty_or_whitespace(s):
    return not s.strip()


def parse_json_response(response: str):
    sanitized = sanitize_json_string(response.strip())
    json_object = json.loads(sanitized)
    return json_object


def parse_json_response_with_file_string(file_string):
    response_lines = file_string.split("\n")

    if is_empty_or_whitespace(response_lines[0]):
        response_lines.pop(0)
    # Get the first line JSON object
    response = response_lines[0]
    # Remove the first line
    response_lines.pop(0)
    # Remove the first line Markdown code block marker
    response_lines.pop(0)
    # Remove the last line Markdown code block marker
    response_lines.pop(-1)
    # Combine lines into a single string
    file_string_content = "\n".join(response_lines)
    json_object = parse_json_response(response)

    return json_object, file_string_content


def extract_object_from_response(response: str, object_clas: type):
    obj = parse_json_response(response)
    cls = object_clas
    obj = cls(**obj)
    return obj
