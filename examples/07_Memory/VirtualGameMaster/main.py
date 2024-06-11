import datetime

from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.agent_memory.event_memory import Event
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_agent import SystemPromptModule, SystemPromptModulePosition
from llama_cpp_agent.providers import LlamaCppServerProvider
from memory import output_settings, agent_core_memory, agent_retrieval_memory, agent_event_memory, update_memory_section
from prompts import game_master_prompt, examples, memory_prompt, wrap_function_response_in_xml_tags_json_mode, generate_fake_write_message, generate_write_message_with_examples, wrap_player_message_in_xml_tags_json_mode

provider = LlamaCppServerProvider("http://localhost:8080")

agent = LlamaCppAgent(
    provider,
    system_prompt=game_master_prompt,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
)

settings = provider.get_provider_default_settings()
settings.n_predict = 1024
settings.temperature = 0.35
settings.top_k = 0
settings.top_p = 0.5

memory_section = SystemPromptModule("memory",
                                    "The following section shows the count of memories in archival memory and chat history memory and the current content of your core memory:")
date_time_section = SystemPromptModule("current_date_time", "The following section shows the current date and time:")

example_section = SystemPromptModule("examples",
                                     "The following examples show you which kind of responses you should write to the user based on the current scenario:", suffix="Always remember to never write actions or dialogue for the user! Always let the user decide on actions or dialogue!")
example_section.set_content(examples)
memory_intro_section = SystemPromptModule("memory_intro",
                                          "To support you in your task as a game master and to help you remembering things, you have access to 3 different types of memory.",
                                          position=SystemPromptModulePosition.after_system_instructions)
memory_intro_section.set_content(memory_prompt)
while True:
    user_input = input(">")
    if user_input == "exit":
        break
    update_memory_section(memory_section)
    date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y") + "\nFormat: dd.mm.yyyy")

    agent_event_memory.add_event_to_queue(Roles.user, wrap_player_message_in_xml_tags_json_mode(user_input))
    agent_output = agent.get_chat_response(
        chat_history=agent_event_memory.get_event_memory_manager().build_chat_history(),
        llm_sampling_settings=settings,
        system_prompt_modules=[memory_intro_section, memory_section, date_time_section],
        structured_output_settings=output_settings)

    while True:
        update_memory_section(memory_section)
        date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y") + "\nFormat: dd.mm.yyyy")

        if agent_output[0]["function"] == "write_message_to_player":
            agent_event_memory.add_event_to_queue(Roles.tool, generate_write_message_with_examples(examples=example_section.get_formatted_content()))
            output = agent.get_chat_response(
                chat_history=agent_event_memory.get_event_memory_manager().build_chat_history(),
                add_message_to_chat_history=False, add_response_to_chat_history=False,
                system_prompt_modules=[memory_intro_section, memory_section, date_time_section],
                llm_sampling_settings=settings)
            agent_event_memory.add_event_to_queue(Roles.tool, generate_fake_write_message())
            agent_event_memory.add_event_to_queue(Roles.assistant, output)

            print(output)
            break

        agent_event_memory.add_event_to_queue(Roles.tool, wrap_function_response_in_xml_tags_json_mode(agent_output[0]["return_value"]))
        agent_output = agent.get_chat_response(chat_history=agent_event_memory.get_event_memory_manager().build_chat_history(),
                                               llm_sampling_settings=settings,
                                               system_prompt_modules=[memory_intro_section, memory_section,
                                                                      date_time_section],
                                               structured_output_settings=output_settings)
