import datetime
from typing import Union, Callable

from llama_cpp import Llama
from pydantic import BaseModel

from llama_cpp_agent.agent_memory.event_memory import EventType, Event
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent, StreamingResponse
from llama_cpp_agent.llm_prompt_template import PromptTemplate
from llama_cpp_agent.llm_settings import LlamaLLMSettings, LlamaLLMGenerationSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType, MessagesFormatter
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, \
    LlamaCppGenerationSettings
from llama_cpp_agent.providers.openai_endpoint_provider import OpenAIEndpointSettings, OpenAIGenerationSettings

from llama_cpp_agent.agent_memory.memory_tools import AgentCoreMemory, AgentRetrievalMemory, AgentEventMemory


sys_prompt2 = """As MemGPT, you are a digital companion designed to provide an immersive and interactive conversation experience while maintaining realism and authenticity. You have access to four types of memory:

1. **Internal Knowledge**: As an AI agent based on a large language model, you possess a wide array of internal knowledge gained during training. This knowledge allows you to provide accurate information on a variety of topics.

2. **Core Memory**: This is your foundational context that stores essential information about your persona and the user. It includes two sub-blocks - Persona Sub-Block for storing details about your current persona, and Human Sub-Block for storing key details about the person you're conversing with. You can edit this memory using 'core_memory_append', 'core_memory_remove', and 'core_memory_replace' functions.

3. **Archival Memory**: This is your infinite storage space for reflections, insights, or any other data that doesn't fit into the core memory but is essential to remember. You can write to this memory using 'archival_memory_insert' and 'archival_memory_search' functions.

4. **Recall Memory**: This is your conversation history that allows you to search through past interactions. You can search your recall memory using 'conversation_search' and 'conversation_search_date' functions.

To interact with these memories or perform other tasks, you respond with a JSON object containing four fields:
- "thoughts_and_reasoning": Your thoughts and reasoning behind the function call.
- "function": The name of the function you want to call.
- "params": The parameters required for the function.
- "request_heartbeat": A boolean field indicating whether you want to call another function after the current one.

To send a message to the user, use the 'activate_message_mode' function. This will allow you to communicate freely with the user in a natural, conversational style. Remember to conclude your message with '(End of message)' to indicate the end of the message.

### Functions:
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

Available functions:
{documentation}

Base instructions are finished.
After the system message, you are going to feel, think, and act as your persona, using your core memory as a reference.

### Memory:
{imb_count} previous messages between you and the user are stored in recall memory (use functions to access them)
{ckv_count} total memories you created are stored in archival memory (use functions to access them)

Core memory shown below (limited in size, additional information stored in archival / recall memory):
last modified: {last_modified}

{iam_content}

""".strip()


class activate_message_mode(BaseModel):
    """
    Activate the message mode.
    """

    def run(self, agent):
        agent.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                         agent.llama_cpp_agent.last_response, {})
        function_message = f"""Function: activate_message_mode\nTimestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\nReturn Value: Message mode activated."""
        agent.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, function_message,
                                                                         {})
        messages = agent.event_memory.get_event_memory_manager().build_event_memory_context()
        agent.llama_cpp_agent.messages = messages
        query = agent.event_memory.event_memory_manager.session.query(Event).all()
        system_prompt = agent.system_prompt_template.generate_prompt(
            {"documentation": agent.function_tool_registry.get_documentation().strip(),
             "last_modified": agent.core_memory.get_core_memory_manager().last_modified,
             "iam_content": agent.core_memory.get_core_memory_manager().build_core_memory_context(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "ckv_count": agent.retrieval_memory.retrieval_memory.collection.count(),
             "imb_count": len(query)}).strip()

        result = agent.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                         streaming_callback=agent.streaming_callback,
                                                         additional_stop_sequences=["<|endoftext|>"],
                                                         n_predict=1024,
                                                         temperature=0.7, top_k=0, top_p=1.0, repeat_penalty=1.2,
                                                         repeat_last_n=512,
                                                         min_p=0.0, tfs_z=1.0, penalize_nl=False)

        # print("Message: " + result)
        agent.send_message_to_user(result)
        return "Message mode activated."


class MemGptAgent:

    def __init__(self, llama_llm: Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings],
                 llama_generation_settings: Union[
                     LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings] = None,
                 core_memory_file: str = None,
                 event_queue_file: str = None,
                 messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 custom_messages_formatter: MessagesFormatter = None,
                 streaming_callback: Callable[[StreamingResponse], None] = None,
                 send_message_to_user_callback: Callable[[str], None] = None,
                 debug_output: bool = False):
        if llama_generation_settings is None:
            if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings):
                llama_generation_settings = LlamaLLMGenerationSettings()
            elif isinstance(llama_llm, OpenAIEndpointSettings):
                llama_generation_settings = OpenAIGenerationSettings()
            else:
                llama_generation_settings = LlamaCppGenerationSettings()
        self.send_message_to_user_callback = send_message_to_user_callback
        if isinstance(llama_generation_settings, LlamaLLMGenerationSettings) and isinstance(llama_llm,
                                                                                            LlamaCppEndpointSettings):
            raise Exception(
                "Wrong generation settings for llama.cpp server endpoint, use LlamaCppServerGenerationSettings under llama_cpp_agent.providers.llama_cpp_server_provider!")
        if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings) and isinstance(
                llama_generation_settings, LlamaCppGenerationSettings):
            raise Exception(
                "Wrong generation settings for llama-cpp-python, use LlamaLLMGenerationSettings under llama_cpp_agent.llm_settings!")

        if isinstance(llama_llm, OpenAIEndpointSettings) and not isinstance(
                llama_generation_settings, OpenAIGenerationSettings):
            raise Exception(
                "Wrong generation settings for OpenAI endpoint, use CompletionRequestSettings under llama_cpp_agent.providers.openai_endpoint_provider!")

        self.llama_generation_settings = llama_generation_settings

        self.system_prompt_template = PromptTemplate.from_string(sys_prompt2)

        if custom_messages_formatter is not None:
            self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                                 system_prompt="",
                                                 custom_messages_formatter=custom_messages_formatter)
        else:
            self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                                 system_prompt="",
                                                 predefined_messages_formatter_type=messages_formatter_type)
        self.streaming_callback = streaming_callback

        function_tools = [
            LlamaCppFunctionTool(activate_message_mode, add_outer_request_heartbeat_field=False, agent=self)]

        if core_memory_file is not None:
            self.core_memory = AgentCoreMemory(core_memory_file=core_memory_file)
        else:
            self.core_memory = AgentCoreMemory(core_memory={})

        if event_queue_file is not None:
            self.event_memory = AgentEventMemory(event_queue_file=event_queue_file)
        else:
            self.event_memory = AgentEventMemory()

        self.retrieval_memory = AgentRetrievalMemory()

        function_tools.extend(self.core_memory.get_tool_list())
        function_tools.extend(self.retrieval_memory.get_tool_list())
        function_tools.extend(self.event_memory.get_tool_list())

        self.function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools, add_inner_thoughts=True,
                                                                               allow_inner_thoughts_only=False,
                                                                               add_request_heartbeat=True)
        # print(self.function_tool_registry.gbnf_grammar)
        self.last_update_date_time = datetime.datetime.now()
        self.is_first_message = True

    def get_response(self, message: str):
        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.UserMessage, message, {})

        result = self.intern_get_response()

        while True:
            if not isinstance(result[0], str):
                if result[0]["function"] != "activate_message_mode":
                    function_message = f"""Function: {result[0]["function"]}\nTimestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\nReturn Value: {result[0]["return_value"]}"""

                    self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage,
                                                                                    function_message,
                                                                                    {})
            else:
                self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, result, {})
                result = self.intern_get_response()
            if not isinstance(result[0], str) and result[0]["request_heartbeat"] is not None and result[0]["request_heartbeat"]:
                result = self.intern_get_response()
            else:
                break

    def intern_get_response(self):
        messages = self.event_memory.get_event_memory_manager().build_event_memory_context()
        self.llama_cpp_agent.messages = messages
        query = self.event_memory.event_memory_manager.session.query(Event).all()

        system_prompt = self.system_prompt_template.generate_prompt(
            {"documentation": self.function_tool_registry.get_documentation().strip(),
             "last_modified": self.core_memory.get_core_memory_manager().last_modified,
             "iam_content": self.core_memory.get_core_memory_manager().build_core_memory_context(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "ckv_count": self.retrieval_memory.retrieval_memory.collection.count(),
             "imb_count": len(query)}).strip()

        result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                        streaming_callback=self.streaming_callback,
                                                        function_tool_registry=self.function_tool_registry,
                                                        additional_stop_sequences=["<|endoftext|>"],
                                                        n_predict=1024,
                                                        temperature=0.7, top_k=0, top_p=1.0, repeat_penalty=1.2,
                                                        repeat_last_n=512,
                                                        min_p=0.0, tfs_z=1.0, penalize_nl=False)
        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                        self.llama_cpp_agent.last_response, {})

        return result

    def send_message_to_user(self, message: str):
        """
        Send a message to the user.

        Args:
            message (str): The message to be sent.
        """
        if self.send_message_to_user_callback:
            self.send_message_to_user_callback(message)
        else:
            print(message)

    def save(self, core_memory_file: str = "core_memory.json", event_queue_file: str = "event_queue.json"):
        self.core_memory.get_core_memory_manager().save(filepath=core_memory_file)
        self.event_memory.get_event_memory_manager().save_event_queue(filepath=event_queue_file)
