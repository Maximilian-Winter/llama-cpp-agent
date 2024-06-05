import datetime

from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_agent import SystemPromptModules, SystemPromptModule
from llama_cpp_agent.providers import LlamaCppServerProvider
from memory import output_settings, agent_core_memory, agent_retrieval_memory, agent_event_memory
from prompts import game_master_prompt
provider = LlamaCppServerProvider("http://localhost:8080")

agent = LlamaCppAgent(
    provider,
    system_prompt=game_master_prompt,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
)

settings = provider.get_provider_default_settings()
settings.n_predict = 512
settings.temperature = 0.75
settings.top_k = 0
settings.top_p = 0.5

core_memory_section = SystemPromptModule("core_memory", "The following section shows the current content of your core memory with information about the current game:")
date_time_section = SystemPromptModule("current_date_time", "The following section shows the current date and time:")

while True:
    user_input = input(">")
    if user_input == "exit":
        break

    core_memory_section.set_content(agent_core_memory.get_core_memory_view().strip())
    date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y") + "\nFormat: dd.mm.yyyy")

    agent_output = agent.get_chat_response("<player_message>\n" + user_input + "\n</player_message>", llm_sampling_settings=settings, system_prompt_additions=SystemPromptModules([core_memory_section, date_time_section]), structured_output_settings=output_settings)

    while True:
        core_memory_section.set_content(agent_core_memory.get_core_memory_view().strip())
        date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S") + "\nFormat: dd.mm.yyyy")

        if agent_output[0]["function"] == "write_message_to_player":
            output = agent.get_chat_response("<function_response>\n" + agent_output[0]["return_value"] + "\n</function_response>", role=Roles.tool, system_prompt_additions=SystemPromptModules([core_memory_section, date_time_section]), llm_sampling_settings=settings)
            print(output)
            break
        agent_output = agent.get_chat_response("<function_response>\n" + agent_output[0]["return_value"] + "\n</function_response>", role=Roles.tool, llm_sampling_settings=settings, system_prompt_additions=SystemPromptModules([core_memory_section, date_time_section]), structured_output_settings=output_settings)
