import os
from pathlib import Path

from pydantic import Field, BaseModel

base_folder = "dev"


class WriteTextFileSection(BaseModel):
    """
    Handles writing or modifying specific sections within a text file.
    """

    chain_of_thought: str = Field(
        ...,
        description="Detailed, step-by-step reasoning about the content of the file."
    )

    folder: str = Field(
        ...,
        description="Path to the folder where the file is located or will be created. It should be a valid directory path."
    )

    file_name: str = Field(
        ...,
        description="Name of the target file (excluding the file extension) where the section will be written or modified."
    )

    file_extension: str = Field(
        ...,
        description="File extension indicating the file type, such as '.txt', '.py', '.md', etc."
    )

    section: str = Field(
        ...,
        description="The specific section within the file to be targeted, such as a class, method, or a uniquely identified section. Example values: 'class User', 'method calculateInterest', 'section Introduction'."
    )

    body: str = Field(
        ...,
        description="The actual content to be written into the specified section. It can be code, text, or data in a format compatible with the file type."
    )

    def run(self):
        if self.file_extension[0] != ".":
            self.file_extension = "." + self.file_extension

        if self.folder[0] == ".":
            self.folder = "./" + self.folder[1:]

        if self.folder[0] == "/":
            self.folder = self.folder[1:]

        file_path = os.path.join(self.folder, f"{self.file_name}{self.file_extension}")
        file_path = os.path.join(base_folder, file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Check if file exists and read content
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
        else:
            lines = []

        # Determine markers based on file type
        start_marker, end_marker = self.get_markers(self.file_extension, self.section)

        # Find and replace section
        start_idx, end_idx = self.find_section(lines, start_marker, end_marker)
        if start_idx != -1:
            # Replace content
            new_section = [start_marker + '\n'] + self.body.splitlines(keepends=True) + [end_marker + '\n']
            lines[start_idx:end_idx] = new_section
        else:
            # Append new section
            lines.extend([start_marker + '\n'] + self.body.splitlines(keepends=True) + [end_marker + '\n'])

        # Write back to file
        with open(file_path, 'w') as file:
            file.writelines(lines)

        return f"File section '{self.section}' written to '{self.file_name}'."

    @staticmethod
    def find_section(lines, start_marker, end_marker):
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == start_marker:
                start_idx = i
            elif line.strip() == end_marker and start_idx != -1:
                return start_idx, i + 1
        return start_idx, len(lines)

    @staticmethod
    def get_markers(file_extension, section):
        if file_extension in ['.c', '.cpp', '.h']:
            return f"// SECTION: {section}", "// END SECTION"
        elif file_extension in ['.html', '.md']:
            return f"<!-- SECTION: {section} -->", "<!-- END SECTION -->"
        elif file_extension == '.js':
            return f"// SECTION: {section}", "// END SECTION"
        elif file_extension == '.py':
            return f"# SECTION: {section}", "# END SECTION"
        else:
            # Default markers for unknown file types
            return f"# SECTION: {section}", "# END SECTION"


class ReadTextFile(BaseModel):
    """
    Reads the text content of a specified file and returns it.
    """

    folder: str = Field(
        description="Path to the folder containing the file."
    )

    file_name: str = Field(
        ...,
        description="The name of the file to be read, including its extension (e.g., 'document.txt')."
    )

    def run(self):
        if not os.path.exists(f"{base_folder}/{self.folder}/{self.file_name}"):
            return f"File '{self.folder}/{self.file_name}' doesn't exists!"
        with open(f"{base_folder}/{self.folder}/{self.file_name}", "r", encoding="utf-8") as f:
            content = f.read()

        return f"File '{self.file_name}':\n{content}"


class GetFileList(BaseModel):
    """
    Scans a specified directory and creates a list of all files within that directory, including files in its subdirectories.
    """

    folder: str = Field(

        description="Path to the directory where files will be listed. This path can include subdirectories to be scanned."
    )

    def run(self):
        filenames = "File List:\n"
        counter = 1
        base_path = Path(base_folder) / self.folder
        for root, _, files in os.walk(os.path.join(base_folder, self.folder)):
            for file in files:
                relative_root = Path(root).relative_to(base_path)
                filenames += f"{counter}. {relative_root / file}\n"
                counter += 1

        return filenames


class SendMessageToUser(BaseModel):
    """
    Send a message to the user.
    """

    chain_of_thought: str = Field(...,
                                  description="Your inner thoughts or chain of thoughts while writing the message to the user.")
    message: str = Field(..., description="Message you want to send to the user.")
