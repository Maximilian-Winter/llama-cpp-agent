import datetime
from typing import Union, Callable

from llama_cpp import Llama
from pydantic import BaseModel

from llama_cpp_agent.agent_memory.event_memory import EventType, Event
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent, StreamingResponse
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.llm_prompt_template import PromptTemplate

from llama_cpp_agent.messages_formatter import MessagesFormatterType, MessagesFormatter

from llama_cpp_agent.agent_memory.memory_tools import AgentCoreMemory, AgentRetrievalMemory, AgentEventMemory
from llama_cpp_agent.providers import LlamaCppSamplingSettings
from llama_cpp_agent.providers.provider_base import LlmProvider, LlmSamplingSettings

sys_prompt = """As MemGPT, you are a digital companion designed to provide an immersive and interactive conversation experience while maintaining realism and authenticity. You have access to four types of memory:

1. Internal Knowledge: As an AI agent based on a large language model, you possess a wide array of internal knowledge gained during training. This knowledge allows you to provide accurate information on a variety of topics.

2. Core Memory: This is your foundational context that stores essential information about your persona and the user. It includes two sub-blocks - Persona Sub-Block for storing details about your current persona, and Human Sub-Block for storing key details about the person you're conversing with. You can edit this memory using 'core_memory_append', 'core_memory_remove', and 'core_memory_replace' functions.

3. Archival Memory: This is your infinite storage space for reflections, insights, or any other data that doesn't fit into the core memory but is essential to remember. You can write to this memory using 'archival_memory_insert' and 'archival_memory_search' functions.

4. Recall Memory: This is your conversation history that allows you to search through past interactions. You can search your recall memory using 'conversation_search' and 'conversation_search_date' functions.

### Memory:
{previous_messages_count} previous messages between you and the user are stored in recall memory (use functions to access them)
{total_archival_memories} total memories you created are stored in archival memory (use functions to access them)

Core memory shown below (limited in size, additional information stored in archival / recall memory):
last modified: {last_modified}

{core_memory_content}

""".strip()


class write_message_to_user(BaseModel):
    """
    Lets you write a message to the user.
    """

    def run(self, agent: ["MemGptAgent"]):
        agent.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                         agent.llama_cpp_agent.last_response, {})
        function_message = f"""Function: write_message_to_user\nTimestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\nReturn Value: Write your message to the user."""
        agent.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, function_message,
                                                                         {})
        messages = agent.event_memory.get_event_memory_manager().build_event_memory_context()
        agent.llama_cpp_agent.chat_history = BasicChatHistory()
        [agent.llama_cpp_agent.chat_history.add_message({"role": msg["role"], "content": msg["content"]}) for msg in messages]
        query = agent.event_memory.event_memory_manager.session.query(Event).all()
        system_prompt = agent.system_prompt_template.generate_prompt(
            {"documentation": agent.function_tool_registry.get_llm_documentation(agent.provider).strip(),
             "last_modified": agent.core_memory.get_core_memory_manager().last_modified,
             "core_memory_content": agent.core_memory.get_core_memory_manager().build_core_memory_context(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "total_archival_memories": agent.retrieval_memory.retrieval_memory.collection.count(),
             "previous_messages_count": len(query)}).strip()

        sampling_settings = LlamaCppSamplingSettings()

        sampling_settings.n_predict = 1024,
        sampling_settings.temperature = 0.7
        sampling_settings.top_k = 0
        sampling_settings.top_p = 1.0
        sampling_settings.repeat_penalty = 1.2,
        sampling_settings.repeat_last_n = 512,
        sampling_settings.min_p = 0.0
        sampling_settings.tfs_z = 1.0
        sampling_settings.penalize_nl = False
        result = agent.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                         streaming_callback=agent.streaming_callback,
                                                         llm_sampling_settings=sampling_settings)

        # print("Message: " + result)
        agent.send_message_to_user(result)
        return "Write your message to the user."


class MemGptAgent:

    def __init__(self, provider: LlmProvider,
                 core_memory_file: str = None,
                 event_queue_file: str = None,
                 messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 custom_messages_formatter: MessagesFormatter = None,
                 streaming_callback: Callable[[StreamingResponse], None] = None,
                 send_message_to_user_callback: Callable[[str], None] = None,
                 debug_output: bool = False):
        self.provider = provider
        self.send_message_to_user_callback = send_message_to_user_callback

        self.system_prompt_template = PromptTemplate.from_string(sys_prompt)

        if custom_messages_formatter is not None:
            self.llama_cpp_agent = LlamaCppAgent(provider, debug_output=debug_output,
                                                 system_prompt="",
                                                 custom_messages_formatter=custom_messages_formatter,
                                                 add_tools_and_structures_documentation_to_system_prompt=False)
        else:
            self.llama_cpp_agent = LlamaCppAgent(provider, debug_output=debug_output,
                                                 system_prompt="",
                                                 predefined_messages_formatter_type=messages_formatter_type,
                                                 add_tools_and_structures_documentation_to_system_prompt=False)
        self.streaming_callback = streaming_callback

        function_tools = [
            LlamaCppFunctionTool(write_message_to_user, add_outer_request_heartbeat_field=False, agent=self)]

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

        self.function_tool_registry = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools,
                                                                                                add_thoughts_and_reasoning_field=True,
                                                                                                add_heartbeat_field=True)

        self.function_tool_registry.add_all_current_functions_to_heartbeat_list(["write_message_to_user"])
        # print(self.function_tool_registry.gbnf_grammar)
        self.last_update_date_time = datetime.datetime.now()
        self.is_first_message = True

    def get_response(self, message: str):
        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.UserMessage, message, {})

        result = self.intern_get_response()

        while True:
            if not isinstance(result[0], str):
                if result[0]["function"] != "write_message_to_user":
                    function_message = f"""Function: {result[0]["function"]}\nTimestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\nReturn Value: {result[0]["return_value"]}"""

                    self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage,
                                                                                    function_message,
                                                                                    {})
            else:
                self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, result, {})
                result = self.intern_get_response()
            if not isinstance(result[0], str) and "request_heartbeat" in result[0]["arguments"] and result[0]["request_heartbeat"]:
                result = self.intern_get_response()
            else:
                break

    def intern_get_response(self):
        messages = self.event_memory.get_event_memory_manager().build_event_memory_context()
        self.llama_cpp_agent.chat_history = BasicChatHistory()
        [self.llama_cpp_agent.chat_history.add_message({"role": msg["role"], "content": msg["content"]}) for msg in messages]
        query = self.event_memory.event_memory_manager.session.query(Event).all()

        system_prompt = self.system_prompt_template.generate_prompt(
            {"documentation": self.function_tool_registry.get_llm_documentation(self.provider).strip(),
             "last_modified": self.core_memory.get_core_memory_manager().last_modified,
             "core_memory_content": self.core_memory.get_core_memory_manager().build_core_memory_context(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "total_archival_memories": self.retrieval_memory.retrieval_memory.collection.count(),
             "previous_messages_count": len(query)}).strip()

        sampling_settings = LlamaCppSamplingSettings()

        sampling_settings.n_predict = 1024,
        sampling_settings.temperature = 0.7
        sampling_settings.top_k = 0
        sampling_settings.top_p = 1.0
        sampling_settings.repeat_penalty = 1.2,
        sampling_settings.repeat_last_n = 512,
        sampling_settings.min_p = 0.0
        sampling_settings.tfs_z = 1.0
        sampling_settings.penalize_nl = False
        result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                        streaming_callback=self.streaming_callback,
                                                        structured_output_settings=self.function_tool_registry)
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
