import json


def sanitize_json_string(s):
    # Escaping newlines, quotes, and backslashes
    return s.replace('\n', '\\n').replace('\r', '\\r')

def parser_json_with_file_string(file_string):
    response_lines = file_string.split("\n")

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
    sanitized = sanitize_json_string(response)
    json_object = json.loads(sanitized)

    return json_object, file_string_content