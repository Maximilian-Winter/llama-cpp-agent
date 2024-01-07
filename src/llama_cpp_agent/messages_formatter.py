import json
from enum import Enum
from typing import List, Dict, Tuple

SYS_PROMPT_START_MIXTRAL = """"""
SYS_PROMPT_END_MIXTRAL = """\n"""
USER_PROMPT_START_MIXTRAL = """[INST] """
USER_PROMPT_END_MIXTRAL = """ [/INST]"""
ASSISTANT_PROMPT_START_MIXTRAL = """"""
ASSISTANT_PROMPT_END_MIXTRAL = """"""
DEFAULT_MIXTRAL_STOP_SEQUENCES = ["</s>"]

SYS_PROMPT_START_CHATML = """<|im_start|>system\n"""
SYS_PROMPT_END_CHATML = """<|im_end|>\n"""
USER_PROMPT_START_CHATML = """<|im_start|>user\n"""
USER_PROMPT_END_CHATML = """<|im_end|>\n"""
ASSISTANT_PROMPT_START_CHATML = """<|im_start|>assistant\n"""
ASSISTANT_PROMPT_END_CHATML = """<|im_end|>\n"""
DEFAULT_CHATML_STOP_SEQUENCES = ["<|im_end|>"]

SYS_PROMPT_START_VICUNA = """"""
SYS_PROMPT_END_VICUNA = """\n\n"""
USER_PROMPT_START_VICUNA = """USER:"""
USER_PROMPT_END_VICUNA = """\n"""
ASSISTANT_PROMPT_START_VICUNA = """ASSISTANT:"""
ASSISTANT_PROMPT_END_VICUNA = """"""

DEFAULT_VICUNA_STOP_SEQUENCES = ["</s>", "USER:"]
USER_PROMPT_START_LLAMA_2, USER_PROMPT_END_LLAMA_2 = "[INST]", "[/INST]"

ASSISTANT_PROMPT_START_LLAMA_2, ASSISTANT_PROMPT_END_LLAMA_2 = "", "</s>"
SYS_PROMPT_START_LLAMA_2, SYS_PROMPT_END_LLAMA_2 = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_LLAMA_2_STOP_SEQUENCES = ["</s>", "[INST]"]

SYS_PROMPT_START_SYNTHIA = """SYSTEM: """
SYS_PROMPT_END_SYNTHIA = """\n"""
USER_PROMPT_START_SYNTHIA = """USER: """
USER_PROMPT_END_SYNTHIA = """\n"""
ASSISTANT_PROMPT_START_SYNTHIA = """ASSISTANT:"""
ASSISTANT_PROMPT_END_SYNTHIA = """\n"""

SYS_PROMPT_START_NEURAL_CHAT = """### System:\n"""
SYS_PROMPT_END_NEURAL_CHAT = """\n"""
USER_PROMPT_START_NEURAL_CHAT = """### User:\n"""
USER_PROMPT_END_NEURAL_CHAT = """ \n"""
ASSISTANT_PROMPT_START_NEURAL_CHAT = """### Assistant:\n"""
ASSISTANT_PROMPT_END_NEURAL_CHAT = """\n"""
DEFAULT_NEURAL_CHAT_STOP_SEQUENCES = ["### User:"]

SYS_PROMPT_START_SOLAR = """"""
SYS_PROMPT_END_SOLAR = """\n"""
USER_PROMPT_START_SOLAR = """### User:\n"""
USER_PROMPT_END_SOLAR = """ \n"""
ASSISTANT_PROMPT_START_SOLAR = """### Assistant:\n"""
ASSISTANT_PROMPT_END_SOLAR = """\n"""
DEFAULT_SOLAR_STOP_SEQUENCES = ["### User:"]

SYS_PROMPT_START_OPEN_CHAT = """"""
SYS_PROMPT_END_OPEN_CHAT = """  """
USER_PROMPT_START_OPEN_CHAT = """GPT4 Correct User:"""
USER_PROMPT_END_OPEN_CHAT = """<|end_of_turn|>"""
ASSISTANT_PROMPT_START_OPEN_CHAT = """GPT4 Correct Assistant:"""
ASSISTANT_PROMPT_END_OPEN_CHAT = """<|end_of_turn|>"""
DEFAULT_OPEN_CHAT_STOP_SEQUENCES = ["<|end_of_turn|>"]


class MessagesFormatterType(Enum):
    MIXTRAL = 1
    CHATML = 2
    VICUNA = 3
    LLAMA_2 = 4
    SYNTHIA = 5
    NEURAL_CHAT = 6
    SOLAR = 7
    OPEN_CHAT = 8


class MessagesFormatter:
    def __init__(self, PRE_PROMPT: str, SYS_PROMPT_START: str, SYS_PROMPT_END: str, USER_PROMPT_START: str,
                 USER_PROMPT_END: str,
                 ASSISTANT_PROMPT_START: str,
                 ASSISTANT_PROMPT_END: str,
                 INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE: bool,
                 DEFAULT_STOP_SEQUENCES: List[str]):
        self.PRE_PROMPT = PRE_PROMPT
        self.SYS_PROMPT_START = SYS_PROMPT_START
        self.SYS_PROMPT_END = SYS_PROMPT_END
        self.USER_PROMPT_START = USER_PROMPT_START
        self.USER_PROMPT_END = USER_PROMPT_END
        self.ASSISTANT_PROMPT_START = ASSISTANT_PROMPT_START
        self.ASSISTANT_PROMPT_END = ASSISTANT_PROMPT_END
        self.INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE = INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE
        self.DEFAULT_STOP_SEQUENCES = DEFAULT_STOP_SEQUENCES

    def format_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        formatted_messages = self.PRE_PROMPT
        last_role = "assistant"
        no_user_prompt_start = False
        for message in messages:
            if message["role"] == "system":
                formatted_messages += self.SYS_PROMPT_START + message["content"] + self.SYS_PROMPT_END
                last_role = "system"
                if self.INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE:
                    formatted_messages = self.USER_PROMPT_START + formatted_messages
                    no_user_prompt_start = True
            elif message["role"] == "user":
                if no_user_prompt_start:
                    no_user_prompt_start = False
                    formatted_messages += message["content"] + self.USER_PROMPT_END
                else:
                    formatted_messages += self.USER_PROMPT_START + message["content"] + self.USER_PROMPT_END
                last_role = "user"
            elif message["role"] == "assistant":
                formatted_messages += self.ASSISTANT_PROMPT_START + message["content"] + self.ASSISTANT_PROMPT_END
                last_role = "assistant"
        if last_role == "system" or last_role == "user":
            return formatted_messages + self.ASSISTANT_PROMPT_START.strip(), "assistant"
        return formatted_messages + self.USER_PROMPT_START.strip(), "user"

    def save(self, file_path: str):
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "MessagesFormatter":
        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_messages_formatter = json.load(file)
            return MessagesFormatter(**loaded_messages_formatter)

    @staticmethod
    def load_from_dict(loaded_messages_formatter: dict) -> "MessagesFormatter":
        return MessagesFormatter(**loaded_messages_formatter)

    def as_dict(self) -> dict:
        return self.__dict__


mixtral_formatter = MessagesFormatter("", SYS_PROMPT_START_MIXTRAL, SYS_PROMPT_END_MIXTRAL, USER_PROMPT_START_MIXTRAL,
                                      USER_PROMPT_END_MIXTRAL, ASSISTANT_PROMPT_START_MIXTRAL,
                                      ASSISTANT_PROMPT_END_MIXTRAL, True, DEFAULT_MIXTRAL_STOP_SEQUENCES)
chatml_formatter = MessagesFormatter("", SYS_PROMPT_START_CHATML, SYS_PROMPT_END_CHATML, USER_PROMPT_START_CHATML,
                                     USER_PROMPT_END_CHATML, ASSISTANT_PROMPT_START_CHATML,
                                     ASSISTANT_PROMPT_END_CHATML, False, DEFAULT_CHATML_STOP_SEQUENCES)
vicuna_formatter = MessagesFormatter("", SYS_PROMPT_START_VICUNA, SYS_PROMPT_END_VICUNA, USER_PROMPT_START_VICUNA,
                                     USER_PROMPT_END_VICUNA, ASSISTANT_PROMPT_START_VICUNA,
                                     ASSISTANT_PROMPT_END_VICUNA, False, DEFAULT_VICUNA_STOP_SEQUENCES)

llama_2_formatter = MessagesFormatter("", SYS_PROMPT_START_LLAMA_2, SYS_PROMPT_END_LLAMA_2, USER_PROMPT_START_LLAMA_2,
                                      USER_PROMPT_END_LLAMA_2, ASSISTANT_PROMPT_START_LLAMA_2,
                                      ASSISTANT_PROMPT_END_LLAMA_2, True, DEFAULT_LLAMA_2_STOP_SEQUENCES)

synthia_formatter = MessagesFormatter("", SYS_PROMPT_START_SYNTHIA, SYS_PROMPT_END_SYNTHIA, USER_PROMPT_START_SYNTHIA,
                                      USER_PROMPT_END_SYNTHIA, ASSISTANT_PROMPT_START_SYNTHIA,
                                      ASSISTANT_PROMPT_END_SYNTHIA, False, DEFAULT_VICUNA_STOP_SEQUENCES)

neural_chat_formatter = MessagesFormatter("", SYS_PROMPT_START_NEURAL_CHAT, SYS_PROMPT_END_NEURAL_CHAT,
                                          USER_PROMPT_START_NEURAL_CHAT,
                                          USER_PROMPT_END_NEURAL_CHAT, ASSISTANT_PROMPT_START_NEURAL_CHAT,
                                          ASSISTANT_PROMPT_END_NEURAL_CHAT, False, DEFAULT_NEURAL_CHAT_STOP_SEQUENCES)

solar_formatter = MessagesFormatter("", SYS_PROMPT_START_SOLAR, SYS_PROMPT_END_SOLAR, USER_PROMPT_START_SOLAR,
                                    USER_PROMPT_END_SOLAR, ASSISTANT_PROMPT_START_SOLAR,
                                    ASSISTANT_PROMPT_END_SOLAR, True, DEFAULT_SOLAR_STOP_SEQUENCES)

open_chat_formatter = MessagesFormatter("", SYS_PROMPT_START_OPEN_CHAT, SYS_PROMPT_END_OPEN_CHAT,
                                        USER_PROMPT_START_OPEN_CHAT, USER_PROMPT_END_OPEN_CHAT,
                                        ASSISTANT_PROMPT_START_OPEN_CHAT, ASSISTANT_PROMPT_END_OPEN_CHAT, True,
                                        DEFAULT_OPEN_CHAT_STOP_SEQUENCES)
predefined_formatter = {
    MessagesFormatterType.MIXTRAL: mixtral_formatter,
    MessagesFormatterType.CHATML: chatml_formatter,
    MessagesFormatterType.VICUNA: vicuna_formatter,
    MessagesFormatterType.LLAMA_2: llama_2_formatter,
    MessagesFormatterType.SYNTHIA: synthia_formatter,
    MessagesFormatterType.NEURAL_CHAT: neural_chat_formatter,
    MessagesFormatterType.SOLAR: solar_formatter,
    MessagesFormatterType.OPEN_CHAT: open_chat_formatter
}


def get_predefined_messages_formatter(formatter_type: MessagesFormatterType) -> MessagesFormatter:
    return predefined_formatter[formatter_type]
