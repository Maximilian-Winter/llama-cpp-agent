from enum import Enum
from typing import List, Dict, Tuple

SYS_PROMPT_START_MIXTRAL = """[INST] """
SYS_PROMPT_END_MIXTRAL = """ """
USER_PROMPT_START_MIXTRAL = """"""
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


class MessagesFormatterType(Enum):
    MIXTRAL = 1
    CHATML = 2
    VICUNA = 3
    LLAMA_2 = 4


class MessagesFormatter:
    def __init__(self, PRE_PROMPT: str, SYS_PROMPT_START: str, SYS_PROMPT_END: str, USER_PROMPT_START: str,
                 USER_PROMPT_END: str,
                 ASSISTANT_PROMPT_START: str,
                 ASSISTANT_PROMPT_END: str,
                 INCLUDE_SYS_PROMPT_IN_FIRST_MESSAGE: bool,
                 DEFAULT_STOP_SEQUENCES: List[str]):
        self.PRE_PROMPT = PRE_PROMPT
        self.SYS_PROMPT_START = SYS_PROMPT_START
        self.SYS_PROMPT_END = SYS_PROMPT_END
        self.USER_PROMPT_START = USER_PROMPT_START
        self.USER_PROMPT_END = USER_PROMPT_END
        self.ASSISTANT_PROMPT_START = ASSISTANT_PROMPT_START
        self.ASSISTANT_PROMPT_END = ASSISTANT_PROMPT_END
        self.INCLUDE_SYS_PROMPT_IN_FIRST_MESSAGE = INCLUDE_SYS_PROMPT_IN_FIRST_MESSAGE
        self.DEFAULT_STOP_SEQUENCES = DEFAULT_STOP_SEQUENCES

    def format_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        formatted_messages = self.PRE_PROMPT
        last_role = "assistant"
        no_user_prompt_start = False
        for message in messages:
            if message["role"] == "system":
                formatted_messages += self.SYS_PROMPT_START + message["content"] + self.SYS_PROMPT_END
                last_role = "system"
                if self.INCLUDE_SYS_PROMPT_IN_FIRST_MESSAGE:
                    formatted_messages = self.USER_PROMPT_START + formatted_messages
                    no_user_prompt_start = True
            elif message["role"] == "user":
                if no_user_prompt_start:
                    no_user_prompt_start = False
                    formatted_messages += " " + message["content"] + self.USER_PROMPT_END
                else:
                    formatted_messages += self.USER_PROMPT_START + message["content"] + self.USER_PROMPT_END
                last_role = "user"
            elif message["role"] == "assistant":
                formatted_messages += self.ASSISTANT_PROMPT_START + message["content"] + self.ASSISTANT_PROMPT_END
                last_role = "assistant"
        if last_role == "system" or last_role == "user":
            return formatted_messages + self.ASSISTANT_PROMPT_START.strip(), "assistant"
        return formatted_messages + self.USER_PROMPT_START.strip(), "user"


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

predefined_formatter = {
    MessagesFormatterType.MIXTRAL: mixtral_formatter,
    MessagesFormatterType.CHATML: chatml_formatter,
    MessagesFormatterType.VICUNA: vicuna_formatter,
    MessagesFormatterType.LLAMA_2: llama_2_formatter
}


def get_predefined_messages_formatter(formatter_type: MessagesFormatterType) -> MessagesFormatter:
    return predefined_formatter[formatter_type]
