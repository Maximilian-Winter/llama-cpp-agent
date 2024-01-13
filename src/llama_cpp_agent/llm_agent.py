import json
from dataclasses import dataclass
from typing import List, Dict, Literal, Callable, Union

from llama_cpp import Llama, LlamaGrammar

from .llm_settings import LlamaLLMSettings
from .messages_formatter import MessagesFormatterType, get_predefined_messages_formatter, MessagesFormatter
from .function_calling import LlamaCppFunctionTool, LlamaCppFunctionToolRegistry
from .providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, LlamaCppGenerationSettings
from .providers.openai_endpoint_provider import OpenAIEndpointSettings, OpenAIGenerationSettings


@dataclass
class StreamingResponse:
    """
    Represents a streaming response with text and an indicator for the last response.
    """
    text: str
    is_last_response: bool

    def __init__(self, text: str, is_last_response: bool):
        """
        Initializes a new StreamingResponse object.

        Args:
            text (str): The text content of the streaming response.
            is_last_response (bool): Indicates whether this is the last response in the stream.
        """
        self.text = text
        self.is_last_response = is_last_response


class LlamaCppAgent:
    """
    A base agent that can be used for chat, structured output and function calling.
     Is used as part of all other agents.
    """

    def __init__(self, model: Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings],
                 name: str = "llamacpp_agent",
                 system_prompt: str = "You are helpful assistant.",
                 predefined_messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 custom_messages_formatter: MessagesFormatter = None, debug_output: bool = False):
        """
        Initializes a new LlamaCppAgent object.

        Args:
           model (Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings]):The underlying Llama model or settings.
           name (str): The name of the agent.
           system_prompt (str): The system prompt used in chat interactions.
           predefined_messages_formatter_type (MessagesFormatterType): The type of predefined messages formatter.
           custom_messages_formatter (MessagesFormatter): Custom messages formatter.
           debug_output (bool): Indicates whether debug output should be enabled.
        """
        if isinstance(model, LlamaLLMSettings):
            model = Llama(**model.as_dict())
        self.model = model
        self.name = name
        self.system_prompt = system_prompt
        self.debug_output = debug_output
        self.messages = []
        self.grammar_cache = {}
        if custom_messages_formatter is not None:
            self.messages_formatter = custom_messages_formatter
        else:
            self.messages_formatter = get_predefined_messages_formatter(predefined_messages_formatter_type)
        self.last_response = ""

    @staticmethod
    def get_function_tool_registry(function_tool_list: List[LlamaCppFunctionTool], allow_parallel_function_calling=False):
        """
        Creates and returns a function tool registry from a list of LlamaCppFunctionTool instances.

        Args:
            function_tool_list (List[LlamaCppFunctionTool]): List of function tools to register.
            allow_parallel_function_calling: Allow parallel function calling (Default=False)

        Returns:
            LlamaCppFunctionToolRegistry: The created function tool registry.
        """
        function_tool_registry = LlamaCppFunctionToolRegistry(allow_parallel_function_calling)

        for function_tool in function_tool_list:
            function_tool_registry.register_function_tool(function_tool)
        function_tool_registry.finalize()
        return function_tool_registry

    def add_message(self, message: str, role: Literal["system"] | Literal["user"] | Literal["assistant"] = "user"):
        """
        Adds a message to the chat history.

        Args:
            message (str): The content of the message.
            role (Literal["system"] | Literal["user"] | Literal["assistant"]): The role of the message sender.
        """
        self.messages.append(
            {
                "role": role,
                "content": message.strip(),
            },
        )

    def get_chat_response(
            self,
            message: str = None,
            role: Literal["system", "user", "assistant", "function"] = "user",
            system_prompt: str = None,
            add_message_to_chat_history: bool = True,
            add_response_to_chat_history: bool = True,
            grammar: str = None,
            function_tool_registry: LlamaCppFunctionToolRegistry = None,
            streaming_callback: Callable[[StreamingResponse], None] = None,
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
            print_output: bool = True,
            k_last_messages: int = 0,
            # Llama Cpp Server and Open AI endpoint settings
            n_predict: int = -1,
            n_keep: int = 0,
            repeat_last_n: int = 64,
            penalize_nl: bool = True,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            penalty_prompt: Union[None, str, List[int]] = None,
            seed: int = -1,
            ignore_eos: bool = False,
            suffix: str = None,
            echo: bool = False,
            logprobs: int = None,
            logit_bias: Dict[str, float] = None,
            logit_bias_type: Literal["input_ids", "tokens"] = None
    ):
        """
        Gets a chat response based on the input message and context.

        Args:
            message (str): The input message.
            role (Literal["system", "user", "assistant", "function"]): The role of the message sender.
            system_prompt (str): The system prompt used in chat interactions.
            add_message_to_chat_history (bool): Indicates whether to add the input message to the chat history.
            add_response_to_chat_history (bool): Indicates whether to add the generated response to the chat history.
            grammar (str): The grammar for generating responses in string format.
            function_tool_registry (LlamaCppFunctionToolRegistry): The function tool registry for handling function calls.
            streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.
            max_tokens (int): The maximum number of tokens in the generated response.
            temperature (float): The temperature parameter for response generation.
            top_k (int): Top-k parameter for response generation.
            top_p (float): Top-p parameter for response generation.
            min_p (float): Minimum probability parameter for response generation.
            typical_p (float): Typical probability parameter for response generation.
            repeat_penalty (float): Penalty for repeating tokens in response generation.
            mirostat_mode (int): Mirostat mode for response generation.
            mirostat_tau (float): Mirostat tau parameter for response generation.
            mirostat_eta (float): Mirostat eta parameter for response generation.
            tfs_z (float): TFS Z parameter for response generation.
            stop_sequences (List[str]): List of stop sequences for response generation.
            stream (bool): Indicates whether to stream the response.
            print_output (bool): Indicates whether to print the generated response.
            k_last_messages (int): Number of last messages to consider from the chat history.


            Additional parameters for llama.cpp server backends and OpenAI endpoints
            n_predict (int): Number of predictions to generate for each completion.
            n_keep (int): Number of completions to keep.
            repeat_last_n (int): Number of tokens to consider for repeat penalty.
            penalize_nl (bool): Indicates whether to penalize newline characters in response generation.
            presence_penalty (float): Presence penalty parameter for response generation.
            frequency_penalty (float): Frequency penalty parameter for response generation.
            penalty_prompt (Union[None, str, List[int]]): Penalty prompt for response generation.
            seed (int): Seed for random number generation.
            ignore_eos (bool): Indicates whether to ignore end-of-sequence tokens.
            suffix: str = None
            echo: bool = False,
            logprobs: int = None,
            logit_bias: Dict[str, float] = None,
            logit_bias_type:Literal["input_ids", "tokens"] = None

        Returns:
            str: The generated chat response.
        """
        if function_tool_registry is not None:
            grammar = function_tool_registry.gbnf_grammar

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
        if k_last_messages > 0:
            messages.extend(self.messages[-k_last_messages:])
        else:
            messages.extend(self.messages)

        prompt, response_role = self.messages_formatter.format_messages(messages)
        if self.debug_output:
            print(prompt, end="")

        if stop_sequences is None:
            stop_sequences = self.messages_formatter.DEFAULT_STOP_SEQUENCES

        if self.model:
            if isinstance(self.model, LlamaCppEndpointSettings):
                completion = self.model.create_completion(
                    prompt=prompt,
                    grammar=grammar,
                    generation_settings=LlamaCppGenerationSettings(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        n_predict=n_predict,
                        n_keep=n_keep,
                        stream=stream,
                        stop_sequences=stop_sequences,
                        tfs_z=tfs_z,
                        typical_p=typical_p,
                        repeat_penalty=repeat_penalty,
                        repeat_last_n=repeat_last_n,
                        penalize_nl=penalize_nl,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        penalty_prompt=penalty_prompt,
                        mirostat_mode=mirostat_mode,
                        mirostat_tau=mirostat_tau,
                        mirostat_eta=mirostat_eta,
                        seed=seed,
                        ignore_eos=ignore_eos)
                )
            elif isinstance(self.model, OpenAIEndpointSettings):
                completion = self.model.create_completion(
                    prompt=prompt,
                    grammar=grammar,
                    generation_settings=OpenAIGenerationSettings(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        suffix=suffix,
                        stream=stream,
                        stop_sequences=stop_sequences,
                        echo=echo,
                        repeat_penalty=repeat_penalty,
                        logprobs=logprobs,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        logit_bias=logit_bias,
                        logit_bias_type=logit_bias_type,
                        mirostat_mode=mirostat_mode,
                        mirostat_tau=mirostat_tau,
                        mirostat_eta=mirostat_eta,
                        seed=seed)
                )
            else:
                if isinstance(grammar, str):
                    if grammar in self.grammar_cache:
                        grammar = self.grammar_cache[grammar]
                    else:
                        grammar_string = grammar
                        grammar = LlamaGrammar.from_string(grammar, False)
                        self.grammar_cache[grammar_string] = grammar
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
                    if streaming_callback is not None:
                        streaming_callback(StreamingResponse(text=text, is_last_response=False))
                    print(text, end="")
                if streaming_callback is not None:
                    streaming_callback(StreamingResponse(text="", is_last_response=True))
                print("")
                self.last_response = full_response.strip()
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
                    if streaming_callback is not None:
                        streaming_callback(StreamingResponse(text=text, is_last_response=False))
                if streaming_callback is not None:
                    streaming_callback(StreamingResponse(text="", is_last_response=True))
                self.last_response = full_response.strip()
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
                self.last_response = text.strip()
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
            self.last_response = text.strip()
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

    def remove_last_k_chat_messages(self, k: int):
        """
        Removes the last k messages from the chat history.

        Args:
            k (int): Number of last messages to remove.
        """
        # Ensure k is not greater than the length of the messages list
        k = min(k, len(self.messages))

        # Remove the last k elements
        self.messages = self.messages[:-k] if k > 0 else self.messages

    def remove_first_k_chat_messages(self, k: int):
        """
        Removes the first k messages from the chat history.

        Args:
            k (int): Number of first messages to remove.
        """
        # Ensure k is not greater than the length of the messages list
        k = min(k, len(self.messages))

        # Remove the first k elements
        self.messages = self.messages[k:] if k > 0 else self.messages

    def save_messages(self, file_path: str):
        """
        Saves the chat history messages to a file.

        Args:
           file_path (str): The file path to save the messages.
        """
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.messages, file, indent=4)

    def load_messages(self, file_path: str):
        """
        Loads chat history messages from a file.

        Args:
            file_path (str): The file path to load the messages from.
        """
        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_messages = json.load(file)
            self.messages.extend(loaded_messages)

    @staticmethod
    def agent_conversation(
            agent_1: "LlamaCppAgent",
            agent_2: "LlamaCppAgent",
            agent_1_initial_message: str,
            number_of_exchanges: int = 15
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

    @staticmethod
    def group_conversation(
            agent_list: list["LlamaCppAgent"],
            initial_message: str,
            number_of_turns: int = 4
    ):
        responses = [{
            "role": "user",
            "content": initial_message,
        }]
        last_role = "user"
        for _ in range(number_of_turns):

            for a in agent_list:
                a.messages = responses
                response = a.get_chat_response(add_response_to_chat_history=False, add_message_to_chat_history=False)
                response = f"{a.name}: {response}" if not response.strip().startswith(a.name) else response
                responses.append({
                    "role": "user" if last_role == "assistant" else "assistant",
                    "content": response,
                })
                last_role = responses[-1]["role"]
        print("Conversation ended.")
