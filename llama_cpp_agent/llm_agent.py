import json
from typing import List, Dict, Literal

from llama_cpp import Llama, LlamaGrammar

from .messages_formatter import MessagesFormatterType, get_predefined_messages_formatter, MessagesFormatter


class LlamaCppAgent:
    name: str
    system_prompt: str
    model: Llama
    messages: List[Dict[str, str]] = []
    debug_output: bool
    messages_formatter: MessagesFormatter

    def __init__(self, model, name="llamacpp_model", system_prompt="You are helpful assistant.",
                 predefined_messages_formatter_type: MessagesFormatterType = None, debug_output=False):
        self.model = model
        self.name = name
        self.system_prompt = system_prompt
        self.debug_output = debug_output
        if predefined_messages_formatter_type:
            self.messages_formatter = get_predefined_messages_formatter(predefined_messages_formatter_type)
        else:
            self.messages_formatter = get_predefined_messages_formatter(MessagesFormatterType.CHATML)

    def get_chat_response(
            self,
            message: str,
            role: Literal["system"] | Literal["user"] | Literal["assistant"] = "user",
            system_prompt=None,
            grammar: LlamaGrammar = None,
            max_tokens: int = 0,
            temperature: float = 0.4,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.05,
            typical_p: float = 1.0,
            repeat_penalty: float = 1.0,
            mirostat_mode: int = 0,
            mirostat_tau: float = 5.0,
            mirostat_eta: float = 0.1,
            tfs_z: float = 1.0,
            stop_sequences: List[str] = None,
            stream: bool = True,
            add_response_to_chat_history: bool = True,
            print_output: bool = True
    ):
        if system_prompt is None:
            system_prompt = self.system_prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
        ]

        self.messages.append(
            {
                "role": role,
                "content": message.strip(),
            },
        )
        messages.extend(self.messages)

        prompt, response_role = self.messages_formatter.format_messages(messages)
        if self.debug_output:
            print(prompt, end="")

        if stop_sequences is None:
            stop_sequences = self.messages_formatter.DEFAULT_STOP_SEQUENCES

        if self.model:
            completion = self.model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                stream=stream,
                stop=stop_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                tfs_z=tfs_z,
                repeat_penalty=repeat_penalty,
                grammar=grammar
            )
            if stream and print_output:
                full_response = ""
                for out in completion:
                    text = out['choices'][0]['text']
                    full_response += text
                    print(text, end="")
                print("")
                if add_response_to_chat_history:
                    self.messages.append(
                        {
                            "role": response_role,
                            "content": full_response.strip(),
                        },
                    )
                return full_response.strip()
            if stream:
                full_response = ""
                for out in completion:
                    text = out['choices'][0]['text']
                    full_response += text
                if add_response_to_chat_history:
                    self.messages.append(
                        {
                            "role": response_role,
                            "content": full_response.strip(),
                        },
                    )
                return full_response.strip()
            if print_output:
                text = completion['choices'][0]['text']
                print(text)

                if add_response_to_chat_history:
                    self.messages.append(
                        {
                            "role": response_role,
                            "content": text.strip(),
                        },
                    )
                return text.strip()
            text = completion['choices'][0]['text']
            if add_response_to_chat_history:
                self.messages.append(
                    {
                        "role": response_role,
                        "content": text.strip(),
                    },
                )
            return text.strip()
        return "Error: No model loaded!"

    def remove_last_k_chat_messages(self, k):
        # Ensure k is not greater than the length of the messages list
        k = min(k, len(self.messages))

        # Remove the last k elements
        self.messages = self.messages[:-k] if k > 0 else self.messages

    def save_messages(self, file_path: str):
        """
        Save the current state of messages to a file in JSON format.
        :param file_path: The path of the file where messages will be saved.
        """
        with open(file_path, 'w') as file:
            json.dump(self.messages, file, indent=4)

    def load_messages(self, file_path: str):
        """
        Load messages from a file and append them to the current messages list.
        :param file_path: The path of the file from which messages will be loaded.
        """
        with open(file_path, 'r') as file:
            loaded_messages = json.load(file)
            self.messages.extend(loaded_messages)
