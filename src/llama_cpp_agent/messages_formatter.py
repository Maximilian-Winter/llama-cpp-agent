import json
from enum import Enum
from typing import List, Dict, Tuple, Literal

SYS_PROMPT_START_MIXTRAL = """"""
SYS_PROMPT_END_MIXTRAL = """\n\n"""
USER_PROMPT_START_MIXTRAL = """[INST] """
USER_PROMPT_END_MIXTRAL = """ """
ASSISTANT_PROMPT_START_MIXTRAL = """[/INST] """
ASSISTANT_PROMPT_END_MIXTRAL = """</s>"""
FUNCTION_PROMPT_START_MIXTRAL = """"""
FUNCTION_PROMPT_END_MIXTRAL = """"""
DEFAULT_MIXTRAL_STOP_SEQUENCES = ["</s>"]

MIXTRAL_8x22_TOOL_JINJA_TEMPLATE = "{{bos_token}}{% set user_messages = messages | selectattr('role', 'equalto', 'user') | list %}{% for message in messages %}{% if message['role'] == 'user' %}{% if message == user_messages[-1] %}{{ '[AVAILABLE_TOOLS]'}}{% for tool in tools %}{{ tool }}{% endfor %}{{ '[/AVAILABLE_TOOLS]'}}{{ '[INST]' + message['content'] + '[/INST]' }}{% else %}{{ '[INST]' + message['content'] + '[/INST]' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% elif message['role'] == 'tool_results' %}{{'[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]'}}{% elif message['role'] == 'tool_calls' %}{{'[TOOL_CALLS]' + message['content']|string + eos_token}}{% endif %}{% endfor %}"
MIXTRAL_8x22_DEFAULT_JINJA_TEMPLATE = "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
CHATML_JINJA_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
SYS_PROMPT_START_CHATML = """<|im_start|>system\n"""
SYS_PROMPT_END_CHATML = """<|im_end|>\n"""
USER_PROMPT_START_CHATML = """<|im_start|>user\n"""
USER_PROMPT_END_CHATML = """<|im_end|>\n"""
ASSISTANT_PROMPT_START_CHATML = """<|im_start|>assistant\n"""
ASSISTANT_PROMPT_END_CHATML = """<|im_end|>\n"""

FUNCTION_PROMPT_START_CHATML = """<|im_start|>function\n"""
FUNCTION_PROMPT_END_CHATML = """<|im_end|>\n"""
DEFAULT_CHATML_STOP_SEQUENCES = [
    "<|im_end|>",
    "<|im_start|>assistant",
    "<|im_start|>user",
    "<|im_start|>system",
]

SYS_PROMPT_START_VICUNA = """"""
SYS_PROMPT_END_VICUNA = """\n\n"""
USER_PROMPT_START_VICUNA = """USER:"""
USER_PROMPT_END_VICUNA = """\n"""
ASSISTANT_PROMPT_START_VICUNA = """ASSISTANT:"""
ASSISTANT_PROMPT_END_VICUNA = """"""
FUNCTION_PROMPT_START_VICUNA = """"""
FUNCTION_PROMPT_END_VICUNA = """"""
DEFAULT_VICUNA_STOP_SEQUENCES = ["</s>", "USER:"]
USER_PROMPT_START_LLAMA_2, USER_PROMPT_END_LLAMA_2 = "[INST] ", " [/INST]"

ASSISTANT_PROMPT_START_LLAMA_2, ASSISTANT_PROMPT_END_LLAMA_2 = " ", " </s>"
SYS_PROMPT_START_LLAMA_2, SYS_PROMPT_END_LLAMA_2 = "<<SYS>>\n", "\n<</SYS>>\n\n"
FUNCTION_PROMPT_START_LLAMA_2, FUNCTION_PROMPT_END_LLAMA_2 = "", ""
DEFAULT_LLAMA_2_STOP_SEQUENCES = ["</s>", "[INST]"]

SYS_PROMPT_START_SYNTHIA = """SYSTEM: """
SYS_PROMPT_END_SYNTHIA = """\n"""
USER_PROMPT_START_SYNTHIA = """USER: """
USER_PROMPT_END_SYNTHIA = """\n"""
ASSISTANT_PROMPT_START_SYNTHIA = """ASSISTANT:"""
ASSISTANT_PROMPT_END_SYNTHIA = """\n"""
FUNCTION_PROMPT_START_SYNTHIA = """"""
FUNCTION_PROMPT_END_SYNTHIA = """"""

SYS_PROMPT_START_ALPACA = """### Instruction:\n"""
SYS_PROMPT_END_ALPACA = """\n"""
USER_PROMPT_START_ALPACA = """### Input:\n"""
USER_PROMPT_END_ALPACA = """ \n"""
ASSISTANT_PROMPT_START_ALPACA = """### Response:\n"""
ASSISTANT_PROMPT_END_ALPACA = """\n"""
FUNCTION_PROMPT_START_ALPACA = """"""
FUNCTION_PROMPT_END_ALPACA = """"""
DEFAULT_ALPACA_STOP_SEQUENCES = ["### Instruction:", "### Input:", "### Response:"]

SYS_PROMPT_START_NEURAL_CHAT = """### System:\n"""
SYS_PROMPT_END_NEURAL_CHAT = """\n"""
USER_PROMPT_START_NEURAL_CHAT = """### User:\n"""
USER_PROMPT_END_NEURAL_CHAT = """ \n"""
ASSISTANT_PROMPT_START_NEURAL_CHAT = """### Assistant:\n"""
ASSISTANT_PROMPT_END_NEURAL_CHAT = """\n"""
FUNCTION_PROMPT_START_NEURAL_CHAT = """"""
FUNCTION_PROMPT_END_NEURAL_CHAT = """"""
DEFAULT_NEURAL_CHAT_STOP_SEQUENCES = ["### User:"]

SYS_PROMPT_START_22B = """### System: """
SYS_PROMPT_END_22B = """\n"""
USER_PROMPT_START_22B = """### User: """
USER_PROMPT_END_22B = """ \n"""
ASSISTANT_PROMPT_START_22B = """### Assistant:"""
ASSISTANT_PROMPT_END_22B = """\n"""
FUNCTION_PROMPT_START_22B = """"""
FUNCTION_PROMPT_END_22B = """"""
DEFAULT_22B_STOP_SEQUENCES = ["### User:"]

SYS_PROMPT_START_CODE_DS = """"""
SYS_PROMPT_END_CODE_DS = """\n\n"""
USER_PROMPT_START_CODE_DS = """@@ Instruction\n"""
USER_PROMPT_END_CODE_DS = """\n\n"""
ASSISTANT_PROMPT_START_CODE_DS = """@@ Response\n"""
ASSISTANT_PROMPT_END_CODE_DS = """\n\n"""

DEFAULT_CODE_DS_STOP_SEQUENCES = ["@@ Instruction"]

SYS_PROMPT_START_LLAMA3 = """<|start_header_id|>system<|end_header_id|>\n"""
SYS_PROMPT_END_LLAMA3 = """<|eot_id|>"""
USER_PROMPT_START_LLAMA3 = """<|start_header_id|>user<|end_header_id|>\n"""
USER_PROMPT_END_LLAMA3 = """<|eot_id|>"""
ASSISTANT_PROMPT_START_LLAMA3 = """<|start_header_id|>assistant<|end_header_id|>\n"""
ASSISTANT_PROMPT_END_LLAMA3 = """<|eot_id|>"""
FUNCTION_PROMPT_START_LLAMA3 = (
    """<|start_header_id|>function_calling_results<|end_header_id|>\n"""
)
FUNCTION_PROMPT_END_LLAMA3 = """<|eot_id|>"""
DEFAULT_LLAMA3_STOP_SEQUENCES = ["assistant", "<|eot_id|>"]

SYS_PROMPT_START_SOLAR = """"""
SYS_PROMPT_END_SOLAR = """\n"""
USER_PROMPT_START_SOLAR = """### User:\n"""
USER_PROMPT_END_SOLAR = """ \n"""
ASSISTANT_PROMPT_START_SOLAR = """### Assistant:\n"""
ASSISTANT_PROMPT_END_SOLAR = """\n"""
FUNCTION_PROMPT_START_SOLAR = """"""
FUNCTION_PROMPT_END_SOLAR = """"""
DEFAULT_SOLAR_STOP_SEQUENCES = ["### User:"]

SYS_PROMPT_START_OPEN_CHAT = """"""
SYS_PROMPT_END_OPEN_CHAT = """  """
USER_PROMPT_START_OPEN_CHAT = """GPT4 Correct User: """
USER_PROMPT_END_OPEN_CHAT = """<|end_of_turn|>"""
ASSISTANT_PROMPT_START_OPEN_CHAT = """GPT4 Correct Assistant: """
ASSISTANT_PROMPT_END_OPEN_CHAT = """<|end_of_turn|>"""
FUNCTION_PROMPT_START_OPEN_CHAT = """"""
FUNCTION_PROMPT_END_OPEN_CHAT = """"""
DEFAULT_OPEN_CHAT_STOP_SEQUENCES = ["<|end_of_turn|>"]

SYS_PROMPT_START_PHI_3 = """"""
SYS_PROMPT_END_PHI_3 = """\n\n"""
USER_PROMPT_START_PHI_3 = """<|user|>"""
USER_PROMPT_END_PHI_3 = """<|end|>\n"""
ASSISTANT_PROMPT_START_PHI_3 = """<|assistant|>"""
ASSISTANT_PROMPT_END_PHI_3 = """<|end|>\n"""
FUNCTION_PROMPT_START_PHI_3 = """"""
FUNCTION_PROMPT_END_PHI_3 = """"""
DEFAULT_PHI_3_STOP_SEQUENCES = ["<|end|>", "<|end_of_turn|>"]


class MessagesFormatterType(Enum):
    """
    Enum representing different types of predefined messages formatters.
    """

    MIXTRAL = 1
    CHATML = 2
    VICUNA = 3
    LLAMA_2 = 4
    SYNTHIA = 5
    NEURAL_CHAT = 6
    SOLAR = 7
    OPEN_CHAT = 8
    ALPACA = 9
    CODE_DS = 10
    B22 = 11
    LLAMA_3 = 12
    PHI_3 = 13


class MessagesFormatter:
    """
    Class representing a messages formatter for LLMs.
    """

    def __init__(
            self,
            PRE_PROMPT: str,
            SYS_PROMPT_START: str,
            SYS_PROMPT_END: str,
            USER_PROMPT_START: str,
            USER_PROMPT_END: str,
            ASSISTANT_PROMPT_START: str,
            ASSISTANT_PROMPT_END: str,
            INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE: bool,
            DEFAULT_STOP_SEQUENCES: List[str],
            USE_USER_ROLE_FUNCTION_CALL_RESULT: bool = True,
            FUNCTION_PROMPT_START: str = "",
            FUNCTION_PROMPT_END: str = "",
            STRIP_PROMPT: bool = True,
    ):
        """
        Initializes a new MessagesFormatter object.

        Args:
            PRE_PROMPT (str): The pre-prompt content.
            SYS_PROMPT_START (str): The system prompt start.
            SYS_PROMPT_END (str): The system prompt end.
            USER_PROMPT_START (str): The user prompt start.
            USER_PROMPT_END (str): The user prompt end.
            ASSISTANT_PROMPT_START (str): The assistant prompt start.
            ASSISTANT_PROMPT_END (str): The assistant prompt end.
            INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE (bool): Indicates whether to include the system prompt
                                                             in the first user message.
            DEFAULT_STOP_SEQUENCES (List[str]): List of default stop sequences.
            USE_USER_ROLE_FUNCTION_CALL_RESULT (bool): Indicates whether to use user role for function call results.
            FUNCTION_PROMPT_START (str): The function prompt start.
            FUNCTION_PROMPT_END (str): The function prompt end.
        """
        self.PRE_PROMPT = PRE_PROMPT
        self.SYS_PROMPT_START = SYS_PROMPT_START
        self.SYS_PROMPT_END = SYS_PROMPT_END
        self.USER_PROMPT_START = USER_PROMPT_START
        self.USER_PROMPT_END = USER_PROMPT_END
        self.ASSISTANT_PROMPT_START = ASSISTANT_PROMPT_START
        self.ASSISTANT_PROMPT_END = ASSISTANT_PROMPT_END
        self.INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE = (
            INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE
        )
        self.DEFAULT_STOP_SEQUENCES = DEFAULT_STOP_SEQUENCES
        self.FUNCTION_PROMPT_START = FUNCTION_PROMPT_START
        self.FUNCTION_PROMPT_END = FUNCTION_PROMPT_END
        self.USE_USER_ROLE_FUNCTION_CALL_RESULT = USE_USER_ROLE_FUNCTION_CALL_RESULT
        self.STRIP_PROMPT = STRIP_PROMPT

    def format_messages(
            self,
            messages: List[Dict[str, str]],
            response_role: Literal["user", "assistant"] | None = None,
    ) -> Tuple[str, str]:
        """
        Formats a list of messages into a conversation string.

        Args:
            messages (List[Dict[str, str]]): List of messages with role and content.
            response_role(Literal["system", "user", "assistant", "function"]|None): Forces the response role to be "system", "user" or "assistant".
        Returns:
            Tuple[str, str]: Formatted conversation string and the role of the last message.
        """
        formatted_messages = self.PRE_PROMPT
        last_role = "assistant"

        no_user_prompt_start = False
        for message in messages:
            if message["role"] == "system":
                formatted_messages += (
                        self.SYS_PROMPT_START + message["content"] + self.SYS_PROMPT_END
                )
                last_role = "system"
                if self.INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE:
                    formatted_messages = self.USER_PROMPT_START + formatted_messages
                    no_user_prompt_start = True
            elif message["role"] == "user":
                if no_user_prompt_start:
                    no_user_prompt_start = False
                    formatted_messages += message["content"] + self.USER_PROMPT_END
                else:
                    formatted_messages += (
                            self.USER_PROMPT_START
                            + message["content"]
                            + self.USER_PROMPT_END
                    )
                last_role = "user"
            elif message["role"] == "assistant":
                if self.STRIP_PROMPT:
                    message["content"] = message["content"].strip()
                formatted_messages += (
                        self.ASSISTANT_PROMPT_START
                        + message["content"]
                        + self.ASSISTANT_PROMPT_END
                )
                last_role = "assistant"
            elif message["role"] == "function":
                if isinstance(message["content"], list):
                    message["content"] = "\n".join(
                        [json.dumps(m, indent=2) for m in message["content"]]
                    )
                if self.USE_USER_ROLE_FUNCTION_CALL_RESULT:
                    formatted_messages += (
                            self.USER_PROMPT_START
                            + message["content"]
                            + self.USER_PROMPT_END
                    )
                    last_role = "user"
                else:
                    formatted_messages += (
                            self.FUNCTION_PROMPT_START
                            + message["content"]
                            + self.FUNCTION_PROMPT_END
                    )
                    last_role = "function"
        if last_role == "system" or last_role == "user" or last_role == "function":
            if self.STRIP_PROMPT:
                if response_role is not None:
                    if response_role == "assistant":
                        return (
                            formatted_messages + self.ASSISTANT_PROMPT_START.strip(),
                            "assistant",
                        )
                    if response_role == "user":
                        return (
                            formatted_messages + self.USER_PROMPT_START.strip(),
                            "user",
                        )
                else:
                    return (
                        formatted_messages + self.ASSISTANT_PROMPT_START.strip(),
                        "assistant",
                    )
            else:
                if response_role is not None:
                    if response_role == "assistant":
                        return (
                            formatted_messages + self.ASSISTANT_PROMPT_START,
                            "assistant",
                        )
                    if response_role == "user":
                        return formatted_messages + self.USER_PROMPT_START, "user"
                else:
                    return formatted_messages + self.ASSISTANT_PROMPT_START, "assistant"
        if self.STRIP_PROMPT:
            if response_role is not None:
                if response_role == "assistant":
                    return (
                        formatted_messages + self.ASSISTANT_PROMPT_START.strip(),
                        "assistant",
                    )
                if response_role == "user":
                    return formatted_messages + self.USER_PROMPT_START.strip(), "user"
            else:
                return formatted_messages + self.USER_PROMPT_START.strip(), "user"
        else:
            if response_role is not None:
                if response_role == "assistant":
                    return formatted_messages + self.ASSISTANT_PROMPT_START, "assistant"
                if response_role == "user":
                    return formatted_messages + self.USER_PROMPT_START, "user"
            else:
                return formatted_messages + self.USER_PROMPT_START, "user"

    def save(self, file_path: str):
        """
        Saves the messages formatter configuration to a file.

        Args:
           file_path (str): The file path to save the configuration.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "MessagesFormatter":
        """
        Loads a messages formatter configuration from a file.

        Args:
            file_path (str): The file path to load the configuration from.

        Returns:
            MessagesFormatter: Loaded messages formatter.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_messages_formatter = json.load(file)
            return MessagesFormatter(**loaded_messages_formatter)

    @staticmethod
    def load_from_dict(loaded_messages_formatter: dict) -> "MessagesFormatter":
        """
        Creates a messages formatter from a dictionary.

        Args:
            loaded_messages_formatter (dict): Dictionary representing the messages formatter.

        Returns:
            MessagesFormatter: Created messages formatter.
        """
        return MessagesFormatter(**loaded_messages_formatter)

    def as_dict(self) -> dict:
        """
        Converts the messages formatter to a dictionary.

        Returns:
            dict: Dictionary representation of the messages formatter.
        """
        return self.__dict__


mixtral_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_MIXTRAL,
    SYS_PROMPT_END_MIXTRAL,
    USER_PROMPT_START_MIXTRAL,
    USER_PROMPT_END_MIXTRAL,
    ASSISTANT_PROMPT_START_MIXTRAL,
    ASSISTANT_PROMPT_END_MIXTRAL,
    True,
    DEFAULT_MIXTRAL_STOP_SEQUENCES,
)
chatml_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_CHATML,
    SYS_PROMPT_END_CHATML,
    USER_PROMPT_START_CHATML,
    USER_PROMPT_END_CHATML,
    ASSISTANT_PROMPT_START_CHATML,
    ASSISTANT_PROMPT_END_CHATML,
    False,
    DEFAULT_CHATML_STOP_SEQUENCES,
    False,
    FUNCTION_PROMPT_START_CHATML,
    FUNCTION_PROMPT_END_CHATML,
)
vicuna_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_VICUNA,
    SYS_PROMPT_END_VICUNA,
    USER_PROMPT_START_VICUNA,
    USER_PROMPT_END_VICUNA,
    ASSISTANT_PROMPT_START_VICUNA,
    ASSISTANT_PROMPT_END_VICUNA,
    False,
    DEFAULT_VICUNA_STOP_SEQUENCES,
)

llama_2_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_LLAMA_2,
    SYS_PROMPT_END_LLAMA_2,
    USER_PROMPT_START_LLAMA_2,
    USER_PROMPT_END_LLAMA_2,
    ASSISTANT_PROMPT_START_LLAMA_2,
    ASSISTANT_PROMPT_END_LLAMA_2,
    True,
    DEFAULT_LLAMA_2_STOP_SEQUENCES,
)

llama_3_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_LLAMA3,
    SYS_PROMPT_END_LLAMA3,
    USER_PROMPT_START_LLAMA3,
    USER_PROMPT_END_LLAMA3,
    ASSISTANT_PROMPT_START_LLAMA3,
    ASSISTANT_PROMPT_END_LLAMA3,
    False,
    DEFAULT_LLAMA3_STOP_SEQUENCES,
    USE_USER_ROLE_FUNCTION_CALL_RESULT=False,
    FUNCTION_PROMPT_START=FUNCTION_PROMPT_START_LLAMA3,
    FUNCTION_PROMPT_END=FUNCTION_PROMPT_END_LLAMA3,
    STRIP_PROMPT=True,
)

synthia_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_SYNTHIA,
    SYS_PROMPT_END_SYNTHIA,
    USER_PROMPT_START_SYNTHIA,
    USER_PROMPT_END_SYNTHIA,
    ASSISTANT_PROMPT_START_SYNTHIA,
    ASSISTANT_PROMPT_END_SYNTHIA,
    False,
    DEFAULT_VICUNA_STOP_SEQUENCES,
)

neural_chat_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_NEURAL_CHAT,
    SYS_PROMPT_END_NEURAL_CHAT,
    USER_PROMPT_START_NEURAL_CHAT,
    USER_PROMPT_END_NEURAL_CHAT,
    ASSISTANT_PROMPT_START_NEURAL_CHAT,
    ASSISTANT_PROMPT_END_NEURAL_CHAT,
    False,
    DEFAULT_NEURAL_CHAT_STOP_SEQUENCES,
    STRIP_PROMPT=False,
)
code_ds_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_CODE_DS,
    SYS_PROMPT_END_CODE_DS,
    USER_PROMPT_START_CODE_DS,
    USER_PROMPT_END_CODE_DS,
    ASSISTANT_PROMPT_START_CODE_DS,
    ASSISTANT_PROMPT_END_CODE_DS,
    True,
    DEFAULT_CODE_DS_STOP_SEQUENCES,
)

solar_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_SOLAR,
    SYS_PROMPT_END_SOLAR,
    USER_PROMPT_START_SOLAR,
    USER_PROMPT_END_SOLAR,
    ASSISTANT_PROMPT_START_SOLAR,
    ASSISTANT_PROMPT_END_SOLAR,
    True,
    DEFAULT_SOLAR_STOP_SEQUENCES,
)

open_chat_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_OPEN_CHAT,
    SYS_PROMPT_END_OPEN_CHAT,
    USER_PROMPT_START_OPEN_CHAT,
    USER_PROMPT_END_OPEN_CHAT,
    ASSISTANT_PROMPT_START_OPEN_CHAT,
    ASSISTANT_PROMPT_END_OPEN_CHAT,
    True,
    DEFAULT_OPEN_CHAT_STOP_SEQUENCES,
    True,
    FUNCTION_PROMPT_START_OPEN_CHAT,
    FUNCTION_PROMPT_END_OPEN_CHAT,
)

alpaca_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_ALPACA,
    SYS_PROMPT_END_ALPACA,
    USER_PROMPT_START_ALPACA,
    USER_PROMPT_END_ALPACA,
    ASSISTANT_PROMPT_START_ALPACA,
    ASSISTANT_PROMPT_END_ALPACA,
    False,
    DEFAULT_ALPACA_STOP_SEQUENCES,
    False,
    FUNCTION_PROMPT_START_CHATML,
    FUNCTION_PROMPT_END_CHATML,
)

b22_chat_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_22B,
    SYS_PROMPT_END_22B,
    USER_PROMPT_START_22B,
    USER_PROMPT_END_22B,
    ASSISTANT_PROMPT_START_22B,
    ASSISTANT_PROMPT_END_22B,
    False,
    DEFAULT_22B_STOP_SEQUENCES,
    STRIP_PROMPT=False,
)

phi_3_chat_formatter = MessagesFormatter(
    "",
    SYS_PROMPT_START_PHI_3,
    SYS_PROMPT_END_PHI_3,
    USER_PROMPT_START_PHI_3,
    USER_PROMPT_END_PHI_3,
    ASSISTANT_PROMPT_START_PHI_3,
    ASSISTANT_PROMPT_END_PHI_3,
    True,
    DEFAULT_PHI_3_STOP_SEQUENCES,
    True,
    FUNCTION_PROMPT_START_PHI_3,
    FUNCTION_PROMPT_END_PHI_3,
)

predefined_formatter = {
    MessagesFormatterType.MIXTRAL: mixtral_formatter,
    MessagesFormatterType.CHATML: chatml_formatter,
    MessagesFormatterType.VICUNA: vicuna_formatter,
    MessagesFormatterType.LLAMA_2: llama_2_formatter,
    MessagesFormatterType.SYNTHIA: synthia_formatter,
    MessagesFormatterType.NEURAL_CHAT: neural_chat_formatter,
    MessagesFormatterType.SOLAR: solar_formatter,
    MessagesFormatterType.OPEN_CHAT: open_chat_formatter,
    MessagesFormatterType.ALPACA: alpaca_formatter,
    MessagesFormatterType.CODE_DS: code_ds_formatter,
    MessagesFormatterType.B22: b22_chat_formatter,
    MessagesFormatterType.LLAMA_3: llama_3_formatter,
    MessagesFormatterType.PHI_3: phi_3_chat_formatter
}


def get_predefined_messages_formatter(
        formatter_type: MessagesFormatterType,
) -> MessagesFormatter:
    """
    Gets a predefined messages formatter based on the formatter type.

    Args:
        formatter_type (MessagesFormatterType): The type of messages formatter.

    Returns:
        MessagesFormatter: The predefined messages formatter.
    """
    return predefined_formatter[formatter_type]
