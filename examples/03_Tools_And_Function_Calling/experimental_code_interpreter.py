import os
import re
import subprocess
import traceback

import venv
import tempfile
from typing import List

from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.prompt_templates import function_calling_thoughts_and_reasoning
from llama_cpp_agent.providers import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://localhost:8080")

import os
import re
import subprocess
import traceback

import venv
import tempfile
from typing import List


class ComputerLlmInterface:
    def __init__(self, agent: LlamaCppAgent, venv_path, additional_packages=None,
                 additional_blocking_commands=None):
        self.cwd = os.getcwd()
        self.agent = agent
        self.venv_path = venv_path
        if not os.path.exists(venv_path):
            self.create_venv(venv_path)
            default_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn']
            if additional_packages is not None:
                default_packages.extend(additional_packages)
            self.install_dependencies(default_packages)
        self.blocking_commands = ["npm start", "npm run", "node server.js"]
        if additional_blocking_commands is not None:
            self.blocking_commands.extend(additional_blocking_commands)

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

    def execute_command(self, command: str) -> str:
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

    def execute_cli_commands(self, commands: List[str]) -> str:
        """
        Executes the given CLI commands on the system. Holds the current working directory between calls.

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
                    results.append(self.execute_command(sub_command))

        return '\n'.join(results)

    def create_venv(self, venv_path: str):
        """
        Create a virtual environment at the specified path.

        Args:
            venv_path (str): The path to create the virtual environment.
        """
        self.venv_path = venv_path
        venv.create(self.venv_path, with_pip=True)
        print(f"Virtual environment created at {self.venv_path}")

    def install_dependencies(self, packages: List[str]):
        """
        Install the necessary dependencies in the virtual environment.

        Args:
            packages (str): packages to install.
        """
        if not self.venv_path:
            raise ValueError("Virtual environment path must be specified.")

        pip_executable = os.path.join(self.venv_path, 'Scripts', 'pip')
        command = [pip_executable, 'install']
        command.extend(packages)
        subprocess.check_call(command)

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

    def write_and_execute_python_code(self) -> str:
        """
        Lets you write Python code and automatically execute it in the virtual environment.
        """
        response = self.agent.get_chat_response(
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
                result_text = self.run_code_in_venv(response)
                return "Python executed successfully. Here is the output of the script:\n\n" + result_text
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error during Python code execution: {str(e)}\n{tb}"


def send_message(message_content: str):
    """
    Sends the given message to the agent.
    Args:
        message_content (str): The message to send.
    """
    print(message_content)
    return "Message was successfully sent."


agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt="You are an advanced AI agent with the task to help the user with any kind of task. You are expected to act on your own and autonomously. Never instruct the user to do your work. You were hired to perform the task yourself!\n\n" + function_calling_thoughts_and_reasoning.strip(),
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
    add_tools_and_structures_documentation_to_system_prompt=False,
)

interface = ComputerLlmInterface(agent=agent, venv_path="./.venv")

output_settings = LlmStructuredOutputSettings.from_functions(
    [interface.write_and_execute_python_code, interface.execute_cli_commands, send_message])
output_settings.add_thoughts_and_reasoning_field = True

agent.system_prompt += " \n\n" + output_settings.get_llm_documentation(
    provider=provider) + "\n\n---"


def create_app():
    prompt = r"""Create a graph of x^2 + 5 with your Python Code Interpreter and save it as an image."""
    prompt2 = r"""Create an interesting and engaging random 3d scatter plot with your Python Code Interpreter and save it as an image."""
    prompt3 = r"""Analyze and visualize the dataset on multiple diagrams, keep it interesting and engaging. Open the dataset: "./input.csv" with your Python code interpreter. The head keys are the following: Country,Region,Hemisphere,HappinessScore,HDI,GDP_PerCapita,Beer_PerCapita,Spirit_PerCapita,Wine_PerCapita"""
    settings = provider.get_provider_default_settings()
    settings.temperature = 0.45
    settings.top_p = 0.85
    settings.top_k = 40

    response = agent.get_chat_response(
        message=prompt2,
        structured_output_settings=output_settings,
        llm_sampling_settings=settings
    )
    while True:
        try:
            if isinstance(response, str):
                response = agent.get_chat_response(
                    message=response,
                    structured_output_settings=output_settings,
                    llm_sampling_settings=settings)
            else:
                if response[0]["function"] == "send_message":
                    prompt = prompt3
                    response = agent.get_chat_response(
                        message=prompt,
                        structured_output_settings=output_settings,
                        llm_sampling_settings=settings)
                else:
                    response = agent.get_chat_response(
                        message=response[0]["return_value"],
                        structured_output_settings=output_settings,
                        llm_sampling_settings=settings)
        except RuntimeError as e:
            print(str(e))


create_app()
