import os
import re
import subprocess
import traceback

import venv
import tempfile
from enum import Enum
from pathlib import Path
from typing import List

from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.providers import LlamaCppServerProvider
from file_processor import FileEditorMenu, EditorFileOperation, FileProcessor, input_file_path, \
    input_source_and_destination_paths, input_rename, EditorFolderOperation, input_folder_path


class LlmComputerInterface:
    def __init__(self, agent: LlamaCppAgent, sampling_settings, operating_system, venv_path, default_packages=None,
                 additional_packages=None, additional_blocking_commands=None):
        self.current_open_file = None
        self.cwd = os.getcwd()
        self.agent = agent
        self.venv_path = venv_path
        self.sampling_settings = sampling_settings
        if not os.path.exists(venv_path):
            self.create_venv(venv_path)
            if default_packages is None:
                default_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn']
            if additional_packages is not None:
                default_packages.extend(additional_packages)
            self.install_packages(default_packages)
        self.blocking_commands = ["npm start", "npm run", "node server.js"]
        if additional_blocking_commands is not None:
            self.blocking_commands.extend(additional_blocking_commands)
        self.operating_system = operating_system
        self.last_chosen_editor_operation = FileEditorMenu.file

    def execute_cli_command(self, command: str) -> str:
        """
        Execute a single CLI command.

        Args:
            command (str): The CLI command to execute.

        Returns:
            str: The output or error of the command.
        """
        try:
            if self.is_blocking_command(command):
                subprocess.Popen(command, shell=True, cwd=self.cwd)
                return f"Started blocking command: {command}\n"
            else:
                result = subprocess.run(command, shell=True, cwd=self.cwd, capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout
                else:
                    return f"Error executing command: {command}\n{result.stderr}"
        except Exception as e:
            tb = traceback.format_exc()
            return f"Exception occurred while executing command: {str(e)}\n{tb}"

    def is_blocking_command(self, command: str) -> bool:
        """
        Determine if a command is a blocking command.

        Args:
            command (str): The command to check.

        Returns:
            bool: True if the command is blocking, False otherwise.
        """
        return any(blocking_cmd in command for blocking_cmd in self.blocking_commands)

    def change_directory(self, new_dir: str) -> str:
        """
        Change the current working directory if it is different from the current one.

        Args:
            new_dir (str): The directory to change to.

        Returns:
            str: A message indicating the result of the directory change.
        """
        try:
            if not os.path.isabs(new_dir):
                new_dir = os.path.join(self.cwd, new_dir)
            new_dir = os.path.normpath(new_dir)

            if os.path.isdir(new_dir):
                if self.cwd != new_dir:
                    self.cwd = new_dir
                    return f"Changed directory to {self.cwd}\n"
                else:
                    return f"Already in directory: {self.cwd}\n"
            else:
                return f"Directory does not exist: {new_dir}\n"
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error changing directory: {str(e)}\n{tb}"

    def create_venv(self, venv_path: str):
        """
        Create a virtual environment at the specified path.

        Args:
            venv_path (str): The path to create the virtual environment.
        """
        self.venv_path = venv_path
        venv.create(self.venv_path, with_pip=True)
        print(f"Virtual environment created at {self.venv_path}")

    def run_code_in_venv(self, code: str) -> str:
        """
        Run the provided Python code in the specified virtual environment.

        Args:
            code (str): The Python code to execute.

        Returns:
            str: The output or error of the executed code.
        """
        if not self.venv_path:
            raise ValueError("Virtual environment path must be specified.")

        python_executable = os.path.join(self.venv_path, 'Scripts', 'python')
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_script:
                temp_script.write(code.encode('utf-8'))
                temp_script_path = temp_script.name

            result = subprocess.run([python_executable, temp_script_path], cwd=self.cwd, capture_output=True, text=True)

            os.remove(temp_script_path)
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error during Python code execution:\n{result.stderr}"
        except Exception as e:
            tb = traceback.format_exc()
            return f"Exception occurred:\n{str(e)}\n{tb}"

    @staticmethod
    def extract_python_code(text):
        # Pattern to capture Python code within Markdown code blocks
        pattern = r"```python\s*(.*?)\s*```"

        # Using re.DOTALL to make '.' match newline characters as well
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # Assuming you want all occurrences concatenated, or you can handle them separately
            return "\n".join(matches)
        else:
            return "No Python code block found."

    @staticmethod
    def view_folder_as_string(folder_path):
        """
        Returns a view of the given folder as a string, including only files and directories
        directly within the given folder.

        Parameters:
        folder_path (str): The path of the folder to view.

        Returns:
        str: A formatted string representing the contents of the folder.
        """
        folder = Path(folder_path)

        if not folder.exists():
            return f"Error: The folder '{folder_path}' does not exist."

        if not folder.is_dir():
            return f"Error: The path '{folder_path}' is not a directory."

        contents = []

        for item in folder.iterdir():
            if item.is_dir():
                contents.append(f"{item.name}/")
            else:
                contents.append(f"{item.name}")

        return "\n".join(contents)

    def write_and_execute_python_code(self) -> str:
        """
        Lets you write Python code and automatically execute it in the virtual environment.
        """
        response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                prompt_suffix="\n```python\n",
                                                message='Python Code Interpreter activated. Provide only the valid Python code you want to execute in a markdown codeblock.',
                                                role=Roles.tool)

        # Logging the response for debugging
        print(f"Received response for Python execution: {response}")

        if "```python" not in response:
            response = "\n```python\n" + response
        if "```" in response:
            response = self.extract_python_code(response)
        try:
            if isinstance(response, str):
                return self.run_code_in_venv(response)
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error during Python code execution: {str(e)}\n{tb}"

    def install_packages(self, packages: List[str]):
        """
        Installs additional Python packages to the virtual environment.

        Args:
            packages (List[str]): packages to install.
        """
        if not self.venv_path:
            raise ValueError("Virtual environment path must be specified.")

        pip_executable = os.path.join(self.venv_path, 'Scripts', 'pip')
        command = [pip_executable, 'install']
        command.extend(packages)
        subprocess.check_call(command)

    def execute_cli_commands(self, commands: List[str]) -> str:
        """
        Executes the given CLI commands on the system. Keeps its state between calls and works like an open terminal window to you.

        Args:
            commands (List[str]): The CLI commands to execute.

        Returns:
            str: The combined results of the executed commands.
        """
        results = []

        for command in commands:
            sub_commands = command.split('&&')
            for sub_command in sub_commands:
                sub_command = sub_command.strip()
                if sub_command.startswith("cd "):
                    new_dir = sub_command[3:].strip()
                    normalized_path = os.path.normpath(new_dir)
                    base_name = os.path.basename(normalized_path)
                    if not self.cwd.endswith(base_name):
                        results.append(self.change_directory(new_dir))
                else:
                    results.append(self.execute_cli_command(sub_command))

        return '\n'.join(results)

    def set_editor_mode(self, mode: FileEditorMenu):
        """
        Set the editor mode to file or folder, or close the editor.
        Args:
            mode (FileEditorMenu): The mode to switch to.
        """
        if mode == FileEditorMenu.file:
            structured_output_settings = LlmStructuredOutputSettings.from_functions([self.select_file_editor_action],
                                                                                    add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Select File Editor Action.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            return mode, response[0]["return_value"]
        elif mode == FileEditorMenu.folder:
            structured_output_settings = LlmStructuredOutputSettings.from_functions([self.select_folder_editor_action],
                                                                                    add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Select Folder Editor Action.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            return mode, response[0]["return_value"]
        elif mode == FileEditorMenu.close_file_editor:
            return mode, None

    def select_next_file_editor_action(self):
        structured_output_settings = LlmStructuredOutputSettings.from_functions([self.select_file_editor_action],
                                                                                add_thoughts_and_reasoning_field=True)
        response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                message=f'Successfully performed last action. Select next file editor action.',
                                                structured_output_settings=structured_output_settings,
                                                role=Roles.tool)

    def select_file_editor_action(self, action: EditorFileOperation):
        """
        Selects the file editor action.
        Args:
            action (EditorFileOperation): The action to perform on a file or to go back.
        """

        if action == EditorFileOperation.open_file:
            structured_output_settings = LlmStructuredOutputSettings.from_functions([input_file_path],
                                                                                    add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Open file: Please Input Filepath.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            filepath = response[0]["return_value"]
            self.current_open_file = FileProcessor(self.agent,sampling_settings, filepath)

            self.current_open_file.read_and_edit_file()
            self.select_next_file_editor_action()
        elif action == EditorFileOperation.create_file:
            structured_output_settings = LlmStructuredOutputSettings.from_functions([input_file_path],
                                                                                    add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Create file: Please input filename.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            filepath = response[0]["return_value"]
            with open(filepath, "w") as file:
                file.write("")
            self.current_open_file = FileProcessor(self.agent, sampling_settings, filepath)
            self.current_open_file.read_and_edit_file()
            self.select_next_file_editor_action()
        elif action == EditorFileOperation.delete_file:
            structured_output_settings = LlmStructuredOutputSettings.from_functions([input_file_path],
                                                                                    add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Delete file: Please Input Filepath.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            filepath = response[0]["return_value"]
            if os.path.exists(filepath):
                os.remove(filepath)
            self.select_next_file_editor_action()
        elif action == EditorFileOperation.move_file:
            structured_output_settings = LlmStructuredOutputSettings.from_functions(
                [input_source_and_destination_paths], add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Move file: Please Input Source and Destination Filepaths.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            src_path, dest_path = response[0]["return_value"]
            os.rename(src_path, dest_path)
            self.select_next_file_editor_action()
        elif action == EditorFileOperation.rename_file:
            structured_output_settings = LlmStructuredOutputSettings.from_functions(
                [input_rename], add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Rename file: Please Input Filepath and New Filename.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            filepath, new_name = response[0]["return_value"]
            os.rename(filepath, new_name)
            self.select_next_file_editor_action()
        elif action == EditorFileOperation.back:
            self.open_editor()

    def select_next_folder_editor_action(self):
        structured_output_settings = LlmStructuredOutputSettings.from_functions([self.select_folder_editor_action],
                                                                                add_thoughts_and_reasoning_field=True)
        response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                message=f'Successfully performed last action. Select next folder editor action.',
                                                structured_output_settings=structured_output_settings,
                                                role=Roles.tool)

    def select_folder_editor_action(self, action: EditorFolderOperation):
        """
        Selects the folder editor action.
        Args:
            action (EditorFolderOperation): The action to perform on a folder or to go back.
        """

        if action == EditorFolderOperation.open_folder:
            structured_output_settings = LlmStructuredOutputSettings.from_functions([input_folder_path],
                                                                                    add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Open Folder: Please input path to folder.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            folder_path = response[0]["return_value"]
            if os.path.exists(folder_path):
                self.cwd = folder_path
                self.open_editor()

        elif action == EditorFolderOperation.create_folder:
            structured_output_settings = LlmStructuredOutputSettings.from_functions([input_folder_path],
                                                                                    add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Create Folder: Please Input Folder Name.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            folder_path = response[0]["return_value"]
            os.makedirs(folder_path, exist_ok=True)

            self.select_next_folder_editor_action()
        elif action == EditorFolderOperation.delete_folder:
            structured_output_settings = LlmStructuredOutputSettings.from_functions([input_folder_path],
                                                                                    add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Delete Folder: Please Input Folder path.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            folder_path = response[0]["return_value"]
            if os.path.exists(folder_path):
                os.rmdir(folder_path)

            self.select_next_folder_editor_action()
        elif action == EditorFolderOperation.move_folder:
            structured_output_settings = LlmStructuredOutputSettings.from_functions(
                [input_source_and_destination_paths], add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Move Folder: Please Input Source and Destination Paths.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            src_path, dest_path = response[0]["return_value"]
            os.rename(src_path, dest_path)
            self.select_next_folder_editor_action()
        elif action == EditorFolderOperation.rename_folder:
            structured_output_settings = LlmStructuredOutputSettings.from_functions(
                [input_rename], add_thoughts_and_reasoning_field=True)
            response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                    message=f'Rename Folder: Please Input Folder Path and New Folder Name.',
                                                    structured_output_settings=structured_output_settings,
                                                    role=Roles.tool)
            folder_path, new_name = response[0]["return_value"]
            os.rename(folder_path, new_name)
            self.select_next_folder_editor_action()
        elif action == EditorFolderOperation.back:
            return self.open_editor()

    def open_editor(self):
        """
        Opens the editor in the current working directory.
        """
        structured_output_settings = LlmStructuredOutputSettings.from_functions([self.set_editor_mode],
                                                                                add_thoughts_and_reasoning_field=True)
        response = self.agent.get_chat_response(llm_sampling_settings=self.sampling_settings,
                                                message=f"## Editor\nOperating System: {self.operating_system}\nCurrent working directory:{self.cwd}\n\nCurrent working directory content:\n{self.view_folder_as_string(self.cwd).strip()}\n\nSelect Editor Mode. 'file' for creating, opening, moving or deleting files. 'folder'  for creating, opening, moving or deleting folders. And 'close-editor' for closing the editor.",
                                                structured_output_settings=structured_output_settings,
                                                role=Roles.tool)
        print(response)

provider = LlamaCppServerProvider("http://hades.hq.solidrust.net:8084")
#provider = LlamaCppServerProvider("http://localhost:8080")
agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt="You are Funky, an advanced AI agent that can call functions by responding with JSON objects representing the function call with its parameters. You have access to functions to control and work with a computer environment and its filesystem.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
    add_tools_and_structures_documentation_to_system_prompt=True,
)
sampling_settings = provider.get_provider_default_settings()
sampling_settings.top_k = 40
sampling_settings.top_p = 0.85
sampling_settings.tfs_z = 1.0
sampling_settings.min_p = 0.0
sampling_settings.temperature = 0.55

llm_computer_interface = LlmComputerInterface(agent=agent, sampling_settings=sampling_settings, operating_system="Windows 11", venv_path="./venv_agent")

output_settings = LlmStructuredOutputSettings.from_functions([llm_computer_interface.open_editor],
                                                             add_thoughts_and_reasoning_field=True)

agent.get_chat_response(
    "Edit the colour theme of my personal webpage in the 'index.html' file to a dark minimalist theme.",
    structured_output_settings=output_settings)
