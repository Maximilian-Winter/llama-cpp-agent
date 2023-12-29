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


class MessagesFormatterType(Enum):
    MIXTRAL = 1
    CHATML = 2
    VICUNA = 3


class MessagesFormatter:
    def __init__(self, SYS_PROMPT_START: str, SYS_PROMPT_END: str, USER_PROMPT_START: str, USER_PROMPT_END: str,
                 ASSISTANT_PROMPT_START: str,
                 ASSISTANT_PROMPT_END: str,
                 DEFAULT_STOP_SEQUENCES: List[str]):
        self.SYS_PROMPT_START = SYS_PROMPT_START
        self.SYS_PROMPT_END = SYS_PROMPT_END
        self.USER_PROMPT_START = USER_PROMPT_START
        self.USER_PROMPT_END = USER_PROMPT_END
        self.ASSISTANT_PROMPT_START = ASSISTANT_PROMPT_START
        self.ASSISTANT_PROMPT_END = ASSISTANT_PROMPT_END
        self.DEFAULT_STOP_SEQUENCES = DEFAULT_STOP_SEQUENCES

    def format_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        formatted_messages = ""
        last_role = "assistant"
        for message in messages:
            if message["role"] == "system":
                formatted_messages += self.SYS_PROMPT_START + message["content"] + self.SYS_PROMPT_END
                last_role = "system"
            elif message["role"] == "user":
                formatted_messages += self.USER_PROMPT_START + message["content"] + self.USER_PROMPT_END
                last_role = "user"
            elif message["role"] == "assistant":
                formatted_messages += self.ASSISTANT_PROMPT_START + message["content"] + self.ASSISTANT_PROMPT_END
                last_role = "assistant"
        if last_role == "system" or last_role == "user":
            return formatted_messages + self.ASSISTANT_PROMPT_START.strip(), "assistant"
        return formatted_messages + self.USER_PROMPT_START.strip(), "user"


mixtral_formatter = MessagesFormatter(SYS_PROMPT_START_MIXTRAL, SYS_PROMPT_END_MIXTRAL, USER_PROMPT_START_MIXTRAL,
                                      USER_PROMPT_END_MIXTRAL, ASSISTANT_PROMPT_START_MIXTRAL,
                                      ASSISTANT_PROMPT_END_MIXTRAL, DEFAULT_MIXTRAL_STOP_SEQUENCES)
chatml_formatter = MessagesFormatter(SYS_PROMPT_START_CHATML, SYS_PROMPT_END_CHATML, USER_PROMPT_START_CHATML,
                                     USER_PROMPT_END_CHATML, ASSISTANT_PROMPT_START_CHATML,
                                     ASSISTANT_PROMPT_END_CHATML, DEFAULT_CHATML_STOP_SEQUENCES)
vicuna_formatter = MessagesFormatter(SYS_PROMPT_START_VICUNA, SYS_PROMPT_END_VICUNA, USER_PROMPT_START_VICUNA,
                                     USER_PROMPT_END_VICUNA, ASSISTANT_PROMPT_START_VICUNA,
                                     ASSISTANT_PROMPT_END_VICUNA, DEFAULT_VICUNA_STOP_SEQUENCES)


predefined_formatter = {
    MessagesFormatterType.MIXTRAL: mixtral_formatter,
    MessagesFormatterType.CHATML: chatml_formatter,
    MessagesFormatterType.VICUNA: vicuna_formatter
}


def get_predefined_messages_formatter(formatter_type: MessagesFormatterType) -> MessagesFormatter:
    return predefined_formatter[formatter_type]
