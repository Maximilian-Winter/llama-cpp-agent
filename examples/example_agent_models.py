import datetime
import os
from enum import Enum
from pathlib import Path

from pydantic import Field, BaseModel

base_folder = "dev"


def agent_dev_folder_setup():
    global base_folder
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

    base_folder += f"_{timestamp}"
    os.mkdir(base_folder)


agent_dev_folder_setup()


class WriteOperation(Enum):
    CREATE_FILE = "create-file"
    APPEND_FILE = "append-file"
    OVERWRITE_FILE = "overwrite-file"


class WriteTextFile(BaseModel):
    """
    Handles writing or modifying specific files.
    """

    chain_of_thought: str = Field(
        ...,
        description="Detailed, step-by-step reasoning about the content of the file."
    )

    reasoning: str = Field(..., description="Reasoning for the content of this file.")

    folder: str = Field(
        ...,
        description="Path to the folder where the file is located or will be created. It should be a valid directory path."
    )

    file_name_without_extension: str = Field(
        ...,
        description="Name of the target file without the file extension."
    )

    file_extension: str = Field(
        ...,
        description="File extension indicating the file type, such as '.txt', '.py', '.md', etc."
    )

    write_operation: WriteOperation = Field(...,
                                            description="Write operation performed, 'create-file', 'append-file' or 'overwrite-file'")

    content: str = Field(
        ...,
        description="The actual content to be written into the file."
    )

    def run(self):

        if self.folder == "":
            self.folder = "./"
        if self.file_extension[0] != ".":
            self.file_extension = "." + self.file_extension
        if self.folder[0] == "." and len(self.folder) == 1:
            self.folder = "./"

        if self.folder[0] == "." and len(self.folder) > 1 and self.folder[1] != "/":
            self.folder = "./" + self.folder[1:]

        if self.folder[0] == "/":
            self.folder = self.folder[1:]

        file_path = os.path.join(self.folder, f"{self.file_name_without_extension}{self.file_extension}")
        file_path = os.path.join(base_folder, file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Determine the write mode based on the write_operation attribute
        if self.write_operation == WriteOperation.CREATE_FILE:
            write_mode = 'w'  # Create a new file, error if file exists
        elif self.write_operation == WriteOperation.APPEND_FILE:
            write_mode = 'a'  # Append if file exists, create if not
        elif self.write_operation == WriteOperation.OVERWRITE_FILE:
            write_mode = 'w'  # Overwrite file if it exists, create if not
        else:
            raise ValueError(f"Invalid write operation: {self.write_operation}")

        # Write back to file
        with open(file_path, write_mode) as file:
            file.writelines(self.content)

        return f"Content written to '{self.file_name_without_extension}{self.file_extension}'."


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
        if content.strip() == "":
            return f"File '{self.file_name}' is empty!"
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

        if counter == 1:
            return f"Folder '{self.folder}' is empty!"
        return filenames


class SendMessageToUser(BaseModel):
    """
    Send a message to the Developer.
    """

    chain_of_thought: str = Field(...,
                                  description="Your inner thoughts or chain of thoughts while writing the message to the user.")
    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Chain of Thought: " + self.chain_of_thought)
        print("Message: " + self.message)
