import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Literal

from llama_cpp_agent.chat_history.messages import Roles


class MessagesFormatterType(Enum):
    """
    Enum representing different types of predefined messages formatters.
    """

    MISTRAL = 1
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
    OPEN_INTERPRETER = 14
    AUTOCODER = 15

@dataclass
class PromptMarkers:
    start: str
    end: str


class MessagesFormatter:
    def __init__(
            self,
            pre_prompt: str,
            prompt_markers: Dict[Roles, PromptMarkers],
            include_sys_prompt_in_first_user_message: bool,
            default_stop_sequences: List[str],
            use_user_role_for_function_call_result: bool = True,
            strip_prompt: bool = True,
            bos_token: str = "<s>",
            eos_token: str = "</s>"
    ):
        self.pre_prompt = pre_prompt
        self.prompt_markers = prompt_markers
        self.include_sys_prompt_in_first_user_message = include_sys_prompt_in_first_user_message
        self.default_stop_sequences = default_stop_sequences
        self.use_user_role_for_function_call_result = use_user_role_for_function_call_result
        self.strip_prompt = strip_prompt
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.added_system_prompt = False

    def get_bos_token(self) -> str:
        return self.bos_token

    def format_conversation(
            self,
            messages: List[Dict[str, str]],
            response_role: Literal[Roles.user, Roles.assistant] | None = None,
    ) -> Tuple[str, Roles]:
        formatted_messages = self.pre_prompt
        last_role = Roles.assistant
        self.added_system_prompt = False
        for message in messages:
            role = Roles(message["role"])
            content = self._format_message_content(message["content"], role)

            if role == Roles.system:
                formatted_messages += self._format_system_message(content)
                last_role = Roles.system
            elif role == Roles.user:
                formatted_messages += self._format_user_message(content)
                last_role = Roles.user
            elif role == Roles.assistant:
                formatted_messages += self._format_assistant_message(content)
                last_role = Roles.assistant
            elif role == Roles.tool:
                formatted_messages += self._format_tool_message(content)
                last_role = Roles.tool

        return self._format_response(formatted_messages, last_role, response_role)

    def _format_message_content(self, content: str, role: Roles) -> str:
        if self.strip_prompt:
            return content.strip()
        return content

    def _format_system_message(self, content: str) -> str:
        formatted_message = self.prompt_markers[Roles.system].start + content + self.prompt_markers[Roles.system].end
        self.added_system_prompt = True
        if self.include_sys_prompt_in_first_user_message:
            formatted_message = self.prompt_markers[Roles.user].start + formatted_message
        return formatted_message

    def _format_user_message(self, content: str) -> str:
        if self.include_sys_prompt_in_first_user_message and self.added_system_prompt:
            self.added_system_prompt = False
            return content + self.prompt_markers[Roles.user].end
        return self.prompt_markers[Roles.user].start + content + self.prompt_markers[Roles.user].end

    def _format_assistant_message(self, content: str) -> str:
        return self.prompt_markers[Roles.assistant].start + content + self.prompt_markers[Roles.assistant].end

    def _format_tool_message(self, content: str) -> str:
        if isinstance(content, list):
            content = "\n".join(json.dumps(m, indent=2) for m in content)
        if self.use_user_role_for_function_call_result:
            return self._format_user_message(content)
        else:
            return self.prompt_markers[Roles.tool].start + content + self.prompt_markers[Roles.tool].end

    def _format_response(
            self,
            formatted_messages: str,
            last_role: Roles,
            response_role: Literal[Roles.user, Roles.assistant] | None = None,
    ) -> Tuple[str, Roles]:
        if response_role is None:
            response_role = Roles.assistant if last_role != Roles.assistant else Roles.user

        prompt_start = self.prompt_markers[response_role].start.strip() if self.strip_prompt else self.prompt_markers[
            response_role].start
        return formatted_messages + prompt_start, response_role


mixtral_prompt_markers = {
    Roles.system: PromptMarkers("", """\n\n"""),
    Roles.user: PromptMarkers("""[INST] """, """ [/INST]"""),
    Roles.assistant: PromptMarkers("""""", """</s>"""),
    Roles.tool: PromptMarkers("", ""),
}

chatml_prompt_markers = {
    Roles.system: PromptMarkers("""<|im_start|>system\n""", """<|im_end|>\n"""),
    Roles.user: PromptMarkers("""<|im_start|>user\n""", """<|im_end|>\n"""),
    Roles.assistant: PromptMarkers("""<|im_start|>assistant\n""", """<|im_end|>\n"""),
    Roles.tool: PromptMarkers("""<|im_start|>function\n""", """<|im_end|>\n"""),
}

vicuna_prompt_markers = {
    Roles.system: PromptMarkers("", """\n\n"""),
    Roles.user: PromptMarkers("""USER: """, """\n"""),
    Roles.assistant: PromptMarkers("""ASSISTANT:""", ""),
    Roles.tool: PromptMarkers("", ""),
}

llama_2_prompt_markers = {
    Roles.system: PromptMarkers("<<SYS>>\n", "\n<</SYS>>\n\n"),
    Roles.user: PromptMarkers("[INST] ", " [/INST]"),
    Roles.assistant: PromptMarkers(" ", " </s>"),
    Roles.tool: PromptMarkers("", ""),
}

llama_3_prompt_markers = {
    Roles.system: PromptMarkers("""<|start_header_id|>system<|end_header_id|>\n""", """<|eot_id|>"""),
    Roles.user: PromptMarkers("""<|start_header_id|>user<|end_header_id|>\n""", """<|eot_id|>"""),
    Roles.assistant: PromptMarkers("""<|start_header_id|>assistant<|end_header_id|>\n""", """<|eot_id|>"""),
    Roles.tool: PromptMarkers("""<|start_header_id|>function_calling_results<|end_header_id|>\n""", """<|eot_id|>"""),
}

synthia_prompt_markers = {
    Roles.system: PromptMarkers("""SYSTEM: """, """\n"""),
    Roles.user: PromptMarkers("""USER: """, """\n"""),
    Roles.assistant: PromptMarkers("""ASSISTANT:""", """\n"""),
    Roles.tool: PromptMarkers("", ""),
}

neural_chat_prompt_markers = {
    Roles.system: PromptMarkers("""### System:\n""", """\n"""),
    Roles.user: PromptMarkers("""### User:\n""", """ \n"""),
    Roles.assistant: PromptMarkers("""### Assistant:\n""", """\n"""),
    Roles.tool: PromptMarkers("", ""),
}

code_ds_prompt_markers = {
    Roles.system: PromptMarkers("", """\n\n"""),
    Roles.user: PromptMarkers("""@@ Instruction\n""", """\n\n"""),
    Roles.assistant: PromptMarkers("""@@ Response\n""", """\n\n"""),
    Roles.tool: PromptMarkers("", ""),
}

solar_prompt_markers = {
    Roles.system: PromptMarkers("", """\n"""),
    Roles.user: PromptMarkers("""### User:\n""", """ \n"""),
    Roles.assistant: PromptMarkers("""### Assistant:\n""", """\n"""),
    Roles.tool: PromptMarkers("", ""),
}

open_chat_prompt_markers = {
    Roles.system: PromptMarkers("", """  """),
    Roles.user: PromptMarkers("""GPT4 Correct User: """, """<|end_of_turn|>"""),
    Roles.assistant: PromptMarkers("""GPT4 Correct Assistant: """, """<|end_of_turn|>"""),
    Roles.tool: PromptMarkers("", ""),
}

alpaca_prompt_markers = {
    Roles.system: PromptMarkers("""### Instruction:\n""", """\n"""),
    Roles.user: PromptMarkers("""### Input:\n""", """ \n"""),
    Roles.assistant: PromptMarkers("""### Response:\n""", """\n"""),
    Roles.tool: PromptMarkers("""<|im_start|>function\n""", """<|im_end|>\n"""),
}

b22_chat_prompt_markers = {
    Roles.system: PromptMarkers("""### System: """, """\n"""),
    Roles.user: PromptMarkers("""### User: """, """ \n"""),
    Roles.assistant: PromptMarkers("""### Assistant:""", """\n"""),
    Roles.tool: PromptMarkers("", ""),
}

phi_3_chat_prompt_markers = {
    Roles.system: PromptMarkers("", """\n\n"""),
    Roles.user: PromptMarkers("""<|user|>""", """<|end|>\n"""),
    Roles.assistant: PromptMarkers("""<|assistant|>""", """<|end|>\n"""),
    Roles.tool: PromptMarkers("", ""),
}
open_interpreter_chat_prompt_markers = {
    Roles.system: PromptMarkers("", "\n\n"),
    Roles.user: PromptMarkers("### Instruction:\n", "\n"),
    Roles.assistant: PromptMarkers("### Response:\n", "\n"),
    Roles.tool: PromptMarkers("", ""),
}
autocoder_chat_prompt_markers = {
    Roles.system: PromptMarkers("", "\n"),
    Roles.user: PromptMarkers("Human: ", "\n"),
    Roles.assistant: PromptMarkers("Assistant: ", "<|EOT|>\n"),
    Roles.tool: PromptMarkers("", ""),
}

"""
### Instruction:
{prompt}
### Response:"""
mixtral_formatter = MessagesFormatter(
    "",
    mixtral_prompt_markers,
    True,
    ["</s>"],
)

chatml_formatter = MessagesFormatter(
    "",
    chatml_prompt_markers,
    False,
    ["<|im_end|>", "</s>"],
    use_user_role_for_function_call_result=False,
    strip_prompt=True,
)

vicuna_formatter = MessagesFormatter(
    "",
    vicuna_prompt_markers,
    False,
    ["</s>", "USER:"],
)

llama_2_formatter = MessagesFormatter(
    "",
    llama_2_prompt_markers,
    True,
    ["</s>", "[INST]"],
)

llama_3_formatter = MessagesFormatter(
    "",
    llama_3_prompt_markers,
    False,
    ["assistant", "<|eot_id|>"],
    use_user_role_for_function_call_result=False,
    strip_prompt=True,
)

synthia_formatter = MessagesFormatter(
    "",
    synthia_prompt_markers,
    False,
    ["</s>", "USER:"],
)

neural_chat_formatter = MessagesFormatter(
    "",
    neural_chat_prompt_markers,
    False,
    ["### User:"],
    strip_prompt=False,
)

code_ds_formatter = MessagesFormatter(
    "",
    code_ds_prompt_markers,
    True,
    ["@@ Instruction"],
)

solar_formatter = MessagesFormatter(
    "",
    solar_prompt_markers,
    True,
    ["### User:"],
)

open_chat_formatter = MessagesFormatter(
    "",
    open_chat_prompt_markers,
    True,
    ["<|end_of_turn|>"],
    use_user_role_for_function_call_result=True,
)

alpaca_formatter = MessagesFormatter(
    "",
    alpaca_prompt_markers,
    False,
    ["### Instruction:", "### Input:", "### Response:"],
    use_user_role_for_function_call_result=False,
)

b22_chat_formatter = MessagesFormatter(
    "",
    b22_chat_prompt_markers,
    False,
    ["### User:"],
    strip_prompt=False,
)

phi_3_chat_formatter = MessagesFormatter(
    "",
    phi_3_chat_prompt_markers,
    True,
    ["<|end|>", "<|end_of_turn|>"],
    use_user_role_for_function_call_result=True,
)

open_interpreter_chat_formatter = MessagesFormatter(
    "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n",
    open_interpreter_chat_prompt_markers,
    True,
    ["<|EOT|>", "### Instruction:"],
    use_user_role_for_function_call_result=True,
)

autocoder_chat_formatter = MessagesFormatter(
    "",
    autocoder_chat_prompt_markers,
    True,
    ["<|EOT|>"],
    bos_token="<｜begin▁of▁sentence｜>",
    eos_token="<|EOT|>",
)

predefined_formatter = {
    MessagesFormatterType.MISTRAL: mixtral_formatter,
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
    MessagesFormatterType.PHI_3: phi_3_chat_formatter,
    MessagesFormatterType.OPEN_INTERPRETER: open_interpreter_chat_formatter,
    MessagesFormatterType.AUTOCODER: autocoder_chat_formatter,
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
