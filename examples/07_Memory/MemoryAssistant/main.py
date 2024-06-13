import datetime

from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.agent_memory.event_memory import Event
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_agent import SystemPromptModule, SystemPromptModulePosition
from llama_cpp_agent.providers import LlamaCppServerProvider
from memory import output_settings, agent_core_memory, agent_retrieval_memory, agent_event_memory, update_memory_section
from prompts import assistant_prompt,  memory_prompt, wrap_function_response_in_xml_tags_json_mode, \
    generate_write_message, generate_write_message_with_examples, wrap_user_message_in_xml_tags_json_mode

provider = LlamaCppServerProvider("http://localhost:8080")

agent = LlamaCppAgent(
    provider,
    system_prompt=assistant_prompt,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

settings = provider.get_provider_default_settings()
settings.n_predict = 1024
settings.temperature = 0.65
settings.top_k = 40
settings.top_p = 0.85

memory_section = SystemPromptModule("memory",
                                    "The following section shows the count of memories in archival memory and chat history memory and the current content of your core memory:")
date_time_section = SystemPromptModule("current_date_time", "The following section shows the current date and time:")

memory_intro_section = SystemPromptModule("memory_intro",
                                          "To support you in your task as a game master and to help you remembering things, you have access to 3 different types of memory.",
                                          position=SystemPromptModulePosition.after_system_instructions)
memory_intro_section.set_content(memory_prompt)
output_settings.output_structured_output_and_raw_json_string = True
while True:
    user_input = input(">")
    if user_input == "exit":
        break
    update_memory_section(memory_section)
    date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y") + "\nFormat: dd.mm.yyyy")

    agent_event_memory.add_event(Roles.user, wrap_user_message_in_xml_tags_json_mode(user_input))
    agent_output, json_output = agent.get_chat_response(
        chat_history=agent_event_memory.get_event_memory_manager().build_chat_history(),
        llm_sampling_settings=settings,
        system_prompt_modules=[memory_intro_section, memory_section, date_time_section],
        structured_output_settings=output_settings)

    agent_event_memory.add_event(Roles.assistant, json_output)
    while True:
        update_memory_section(memory_section)
        date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y") + "\nFormat: dd.mm.yyyy")

        if agent_output[0]["function"] == "write_message_to_user":
            agent_event_memory.add_event(Roles.tool, generate_write_message())
            output = agent.get_chat_response(
                chat_history=agent_event_memory.get_event_memory_manager().build_chat_history(),
                add_message_to_chat_history=False, add_response_to_chat_history=False,
                system_prompt_modules=[memory_intro_section, memory_section, date_time_section],
                llm_sampling_settings=settings)
            agent_event_memory.add_event(Roles.assistant, output)
            print(output)
            break

        agent_event_memory.add_event(Roles.tool, wrap_function_response_in_xml_tags_json_mode(
            agent_output[0]["return_value"]))
        agent_output, json_output = agent.get_chat_response(
            chat_history=agent_event_memory.get_event_memory_manager().build_chat_history(),
            llm_sampling_settings=settings,
            system_prompt_modules=[memory_intro_section, memory_section,
                                   date_time_section],
            structured_output_settings=output_settings)
        agent_event_memory.add_event(Roles.assistant, json_output)
