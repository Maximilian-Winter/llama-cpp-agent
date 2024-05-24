import logging
import os
import subprocess
from enum import Enum
from io import StringIO

import cssutils
import html5lib
from lxml import etree

from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings


class FileEditorMenu(Enum):
    file = "file"
    folder = "folder"

    close_file_editor = "close-editor"


class EditorFileOperation(Enum):
    open_file = "open-and-edit-file"
    create_file = "create-file"
    delete_file = "delete-file"
    move_file = "move-file"
    rename_file = "rename-file"
    back = "go-back"


class EditorFolderOperation(Enum):
    open_folder = "open-folder"
    create_folder = "create-folder"
    delete_folder = "delete-folder"
    move_folder = "move-folder"
    rename_folder = "rename-folder"
    back = "go-back"


class FileProcessor:
    def __init__(self, agent, llm_sampling_settings, file_path):
        """
        Initialize the FileProcessor with the given file path and read the text from the file.

        :type agent: agent that reads and edit the file.
        :param file_path: str, path to the file
        """
        self.lines = []
        self.agent = agent
        self.file_path = file_path
        self.llm_sampling_settings = llm_sampling_settings
        self._read_file()
        self.start_line = 1
        self.end_line = len(self.lines)

    def _read_file(self):
        """Read the text from the file and split it into lines."""
        with open(self.file_path, 'r', encoding="utf-8") as file:
            self.text = file.read()
        self.lines = self.text.splitlines()

    def _write_file(self):
        """Write the current text back to the file."""
        self.text = '\n'.join(self.lines)
        with open(self.file_path, 'w') as file:
            file.write(self.text)

    def view_lines(self, start_line: int, end_line: int):
        """
        View lines of text from start_line to end_line (inclusive).
        Args:
            start_line (int): the starting line number
            end_line (int): the ending line number
        Returns:
             (str) lines of text with line numbers
        """
        if start_line < 1:
            start_line = 1
        if end_line > len(self.lines) or end_line == -1:
            end_line = len(self.lines)
        if start_line > end_line:
            if end_line == 0:
                start_line = 0
            else:
                start_line = end_line - 1

        self.start_line = start_line
        self.end_line = end_line

        # Adjust for 0-based index
        start_idx = self.start_line - 1
        end_idx = self.end_line

        selected_lines = self.lines[start_idx:end_idx]
        numbered_lines = [f"| {'  ' if i + 1 < 10 else ' ' if i + 1 < 100 else ''}{i + 1} |{line}" for i, line in
                          enumerate(selected_lines, start=start_idx)]

        return '\n'.join(numbered_lines)

    def number_of_lines(self):
        """
        Get the total number of lines in the text.

        :return: int, number of lines
        """
        return len(self.lines)

    def edit_lines(self, start_line: int, end_line: int, new_text: str):
        """
        Edit lines of text from start_line to end_line (inclusive) with new_text.

        Args:
            start_line (int): the starting line number
            end_line (int): the ending line number
            new_text (str): the new text to replace the specified line range
        """
        if start_line < 1:
            start_line = 1
        if end_line > len(self.lines):
            end_line = len(self.lines)
        if start_line > end_line:
            if end_line == 0:
                start_line = 0
            else:
                start_line = end_line - 1

        if len(self.lines) == 0:
            self.add_lines(new_text)
        else:

            # Split the new text into lines
            new_lines = new_text.split('\n')

            # Adjust for 0-based index
            start_idx = start_line - 1
            end_idx = end_line

            # Replace the specified range with the new lines
            self.lines = self.lines[:start_idx] + new_lines + self.lines[end_idx:]

            # Write changes back to the file
            self._write_file()

    def add_lines(self, new_text: str):
        """
        Add new lines to the end of the text.

        Args:
            new_text (str): the new text to be added at the end
        """
        new_lines = new_text.split('\n')
        self.lines.extend(new_lines)
        self._write_file()

    def insert_lines(self, position: int, new_text: str):
        """
        Insert new lines at a specific position in the text.

        Args:
            position (int): the line number before which the new text will be inserted
            new_text (str): the new text to be inserted
        """
        if position < 1 or position > len(self.lines) + 1:
            return "Invalid position specified."

        new_lines = new_text.split('\n')

        # Adjust for 0-based index
        insert_idx = position - 1

        # Insert the new lines
        self.lines = self.lines[:insert_idx] + new_lines + self.lines[insert_idx:]

        # Write changes back to the file
        self._write_file()

    def close_file(self):
        """
        Close the file.
        :return: None
        """
        self._write_file()

    def read_and_edit_file(self):
        """
        Opens a file for reading and editing.
        """
        response = None
        while True:
            structured_output_settings = LlmStructuredOutputSettings.from_functions(
                [self.view_lines, self.edit_lines,
                 self.add_lines, self.insert_lines, self.close_file],
                add_thoughts_and_reasoning_field=True)

            file_text = self.view_lines(self.start_line, self.end_line)
            response = self.agent.get_chat_response(llm_sampling_settings=self.llm_sampling_settings,
                                                    message=f"## File Editor\nFile: {self.file_path}\nNumber of lines: {self.number_of_lines()}\nCurrently visible file content: Line {self.start_line} to Line {self.end_line}\nVisible File Content:\n{file_text}\n\nSelect file editor function. Use the 'view_lines' function to display a range of lines of the file. Use the 'edit_lines' function to replace lines with new content. Use the 'add_lines' function to add lines to the file. Use the 'insert_lines' function to insert new lines in a file at a specific position. And use 'close_file' once you are finished with the work on the file.",
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            validation = self.validate_file()
            if not validation.endswith("validation passed."):
                response = self.agent.get_chat_response(llm_sampling_settings=self.llm_sampling_settings,
                                                        message=f"Your last action caused a validation error: {validation}",
                                                        structured_output_settings=structured_output_settings,
                                                        role=Roles.tool)
            if response[0]["function"] == "close_file":
                return "File Editor closed."

    def validate_file(self):
        file_extension = self.file_path.split('.')[-1]
        try:
            if file_extension == 'html':
                parser = html5lib.HTMLParser(strict=True)
                parser.parse(self.text)
                return self.validate_embedded_css()
            elif file_extension == 'css':
                parser = cssutils.CSSParser()
                result = parser.parseString(self)
                if "validation errors" in result:
                    return result
                return "CSS validation passed."
            elif file_extension == 'js':
                result = subprocess.run(['jshint', self.file_path], capture_output=True, text=True)
                if result.returncode == 0:
                    return "JavaScript validation passed."
                else:
                    return result.stdout
            elif file_extension in ['xml', 'xhtml']:
                etree.fromstring(self.text)
                return "XML validation passed."
            else:
                return f"No validator available for .{file_extension} files."
        except Exception as e:
            return f"Validation error: {e}"

    def validate_embedded_css(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(self.text, 'html.parser')
        styles = soup.find_all('style')

        for style in styles:
            css_text = style.string
            if css_text:
                parser = cssutils.CSSParser()
                result = parser.parseStyle(css_text)
                if "validation errors" in result:
                    return result

        return "HTML and embedded CSS validation passed."


def input_file_path(file_path: str):
    """
    Input the file path.
    Args:
        file_path (str): The file path to use.
    """
    return file_path


def input_source_and_destination_paths(source_path: str, destination_path: str):
    """
    Input the sourcer and destination path.
    Args:
        source_path (str): The source path to use.
        destination_path (str): The destination path to use.
    """
    return source_path, destination_path


def input_rename(old_name: str, new_name: str):
    """
    Input the old and new name.
    Args:
        old_name (str): The name to rename.
        new_name (str): The new name to use.
    """
    return old_name, new_name


def input_folder_path(folder_path: str):
    """
    Input the folder path.
    Args:
        folder_path (str): The folder path to use.
    """
    return folder_path


file = FileProcessor(None, None, "index.html")
print(file.validate_file())
