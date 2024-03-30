import json


def load_text_from_file(file_path):
    # Reading text from the given file path
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        return "File not found. Please check the file path."


def load_files_and_generate_json(file_path1, file_path2):
    # Using the previously defined function to load text from the files
    text1 = load_text_from_file(file_path1)
    text2 = load_text_from_file(file_path2)

    # Check if both files were successfully read
    if text1.startswith("File not found") or text2.startswith("File not found"):
        return "One or both files could not be found. Please check the file paths."

    # Constructing the JSON structure
    data = {
        "persona": {
            "personality": text1
        },
        "human": {
            "name": text2
        }
    }

    # Converting the data to JSON format
    json_data = json.dumps(data, ensure_ascii=False, indent=4)
    with open("core_memory.json", 'w', encoding='utf-8') as file:
        file.write(json_data)

    return json_data


file_name_ai = "agent.txt"
file_name_human = "human.txt"

result = load_files_and_generate_json(file_name_ai, file_name_human)
