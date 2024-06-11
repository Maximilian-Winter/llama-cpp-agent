from llama_cpp_agent.agent_memory.event_memory import Event
from llama_cpp_agent.agent_memory.memory_tools import AgentCoreMemory, AgentRetrievalMemory, AgentEventMemory
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings


def write_message_to_player():
    """
    Lets you write a message to player.
    """
    return "Please write your response to the player, nothing else, only what the player should read!"

agent_core_memory = AgentCoreMemory(["general_game_information", "players", "game_progress", "miscellaneous"], core_memory_file="core_memory.json")
agent_retrieval_memory = AgentRetrievalMemory()
agent_event_memory = AgentEventMemory()

memory_tools = agent_core_memory.get_tool_list()
memory_tools.extend(agent_retrieval_memory.get_tool_list())
memory_tools.extend(agent_event_memory.get_tool_list())

output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(memory_tools,
                                                                            add_thoughts_and_reasoning_field=True,
                                                                            add_heartbeat_field=True)
output_settings.add_all_current_functions_to_heartbeat_list()
output_settings.add_function_tool(write_message_to_player)


def update_memory_section(section):
    query = agent_event_memory.event_memory_manager.session.query(Event).all()
    section.set_content(
        f"Archival Memories:{agent_retrieval_memory.retrieval_memory.collection.count()}\nConversation History Entries:{len(query)}\n\nCore Memory Content:\n{agent_core_memory.get_core_memory_view().strip()}")
