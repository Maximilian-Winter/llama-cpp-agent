import json
from typing import List, Dict, Literal

from llama_cpp import Llama, LlamaGrammar

from .messages_formatter import MessagesFormatterType, get_predefined_messages_formatter, MessagesFormatter
from .function_call_tools import LlamaCppFunctionTool, LlamaCppFunctionToolRegistry


class LlamaCppAgent:

    def __init__(self, model, name="llamacpp_model", system_prompt="You are helpful assistant.",
                 predefined_messages_formatter_type: MessagesFormatterType = None, debug_output=False):
        self.model = model
        self.name = name
        self.system_prompt = system_prompt
        self.debug_output = debug_output
        self.messages = []
        if predefined_messages_formatter_type:
            self.messages_formatter = get_predefined_messages_formatter(predefined_messages_formatter_type)
        else:
            self.messages_formatter = get_predefined_messages_formatter(MessagesFormatterType.CHATML)

    @staticmethod
    def get_function_tool_registry(function_tool_list: List[LlamaCppFunctionTool]):
        function_tool_registry = LlamaCppFunctionToolRegistry()

        for function_tool in function_tool_list:
            function_tool_registry.register_function_tool(function_tool)
        function_tool_registry.finalize()
        return function_tool_registry

    def add_message(self, message: str, role: Literal["system"] | Literal["user"] | Literal["assistant"] = "user",
                    auto_format=False):
        if len(self.messages) == 0:
            self.messages.append(
                {
                    "role": "user",
                    "content": message.strip(),
                },
            )
        if auto_format:
            role = "user" if (self.messages[-1]["role"] == "assistant" or self.messages[-1][
                "role"] == "system") else "assistant"
        self.messages.append(
            {
                "role": role,
                "content": message.strip(),
            },
        )

    def get_chat_response(
            self,
            message: str = None,
            role: Literal["system"] | Literal["user"] | Literal["assistant"] = "user",
            system_prompt=None,
            grammar=None,
            function_tool_registry=None,
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
            k_last_messages: int = -1,
            add_response_to_chat_history: bool = True,
            add_message_to_chat_history: bool = True,
            print_output: bool = True
    ):
        if function_tool_registry is not None:
            grammar = function_tool_registry.get_grammar()
        if system_prompt is None:
            system_prompt = self.system_prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
        ]
        if message is not None and add_message_to_chat_history:
            self.messages.append(
                {
                    "role": role,
                    "content": message.strip(),
                },
            )
        if not add_message_to_chat_history and message is not None:
            messages.append(
                {
                    "role": role,
                    "content": message.strip(),
                },
            )
        if k_last_messages > -1:
            messages.extend(self.messages[-k_last_messages:])
        else:
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
                if function_tool_registry is not None:
                    full_response = function_tool_registry.handle_function_call(full_response)
                    return full_response if full_response else None
                return full_response.strip() if full_response else None
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
                if function_tool_registry is not None:
                    full_response = function_tool_registry.handle_function_call(full_response)
                    return full_response if full_response else None
                return full_response.strip() if full_response else None
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
                if function_tool_registry is not None:
                    text = function_tool_registry.handle_function_call(text)
                    return text if text else None
                return text.strip() if text else None
            text = completion['choices'][0]['text']
            if add_response_to_chat_history:
                self.messages.append(
                    {
                        "role": response_role,
                        "content": text.strip(),
                    },
                )
            if function_tool_registry is not None:
                text = function_tool_registry.handle_function_call(text)
                return text if text else None
            return text.strip() if text else None
        return "Error: No model loaded!"

    def remove_last_k_chat_messages(self, k):
        # Ensure k is not greater than the length of the messages list
        k = min(k, len(self.messages))

        # Remove the last k elements
        self.messages = self.messages[:-k] if k > 0 else self.messages

    def remove_first_k_chat_messages(self, k):
        # Ensure k is not greater than the length of the messages list
        k = min(k, len(self.messages))

        # Remove the first k elements
        self.messages = self.messages[k:] if k > 0 else self.messages

    def save_messages(self, file_path: str):
        with open(file_path, 'w') as file:
            json.dump(self.messages, file, indent=4)

    def load_messages(self, file_path: str):
        with open(file_path, 'r') as file:
            loaded_messages = json.load(file)
            self.messages.extend(loaded_messages)

    @staticmethod
    def agent_conversation(
            agent_1,
            agent_2,
            agent_1_initial_message,
            number_of_exchanges=15
    ):
        current_message = agent_1_initial_message
        current_agent, next_agent = agent_2, agent_1

        for _ in range(number_of_exchanges):
            # Current agent responds to the last message
            response = current_agent.get_chat_response(
                message=current_message,
                role="user",
                add_response_to_chat_history=True,
                print_output=True,
                top_p=0.8,
                top_k=40
            )

            # Update the message for the next turn
            current_message = response

            # Swap the agents for the next turn
            current_agent, next_agent = next_agent, current_agent

        print("Conversation ended.")
