import datetime

from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.agent_memory.event_memory import Event
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_agent import SystemPromptModule, SystemPromptModulePosition
from llama_cpp_agent.providers import LlamaCppServerProvider
from memory import output_settings, agent_core_memory, agent_retrieval_memory, agent_event_memory
from prompts import game_master_prompt, examples, memory_prompt

provider = LlamaCppServerProvider("http://localhost:8080")

agent = LlamaCppAgent(
    provider,
    system_prompt=game_master_prompt,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
)

settings = provider.get_provider_default_settings()
settings.n_predict = 1024
settings.temperature = 0.65
settings.top_k = 40
settings.top_p = 0.85

memory_section = SystemPromptModule("memory",
                                    "The following section shows the count of memories in archival memory and chat history memory and the current content of your core memory:")
date_time_section = SystemPromptModule("current_date_time", "The following section shows the current date and time:")

example_section = SystemPromptModule("example",
                                     "The following examples show you which kind of responses you should write to the user based on the current scenario:\n\n")
example_section.set_content(examples)
memory_intro_section = SystemPromptModule("memory_intro",
                                          "To support you in your task as a game master and to help you remembering things, you have access to 3 different types of memory.",
                                          position=SystemPromptModulePosition.after_system_instructions)
memory_intro_section.set_content(memory_prompt)
while True:
    user_input = input(">")
    if user_input == "exit":
        break
    query = agent_event_memory.event_memory_manager.session.query(Event).all()
    memory_section.set_content(
        f"Archival Memories:{agent_retrieval_memory.retrieval_memory.collection.count()}\nConversation History Entries:{len(query)}\n\nCore Memory Content:\n{agent_core_memory.get_core_memory_view().strip()}")
    date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y") + "\nFormat: dd.mm.yyyy")

    agent_output = agent.get_chat_response(
        "<player_message>\n" + user_input + "\n</player_message>\n<response_format>\nJSON function call.\n</response_format>",
        llm_sampling_settings=settings,
        system_prompt_modules=[memory_intro_section, memory_section, date_time_section],
        structured_output_settings=output_settings)

    while True:
        query = agent_event_memory.event_memory_manager.session.query(Event).all()
        memory_section.set_content(
            f"Archival Memories:{agent_retrieval_memory.retrieval_memory.collection.count()}\nConversation History Entries:{len(query)}\n\nCore Memory Content:\n{agent_core_memory.get_core_memory_view().strip()}")
        date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y") + "\nFormat: dd.mm.yyyy")

        if agent_output[0]["function"] == "write_message_to_player":
            output = agent.get_chat_response(
                f"<function_response>\nWrite your message to the user.\n{example_section.prefix + example_section.content}</function_response>\n<response_format>\nText\n</response_format>",
                role=Roles.tool, add_message_to_chat_history=False, add_response_to_chat_history=False,
                system_prompt_modules=[memory_intro_section, memory_section, date_time_section],
                llm_sampling_settings=settings)
            agent.chat_history.add_message({"role": Roles.tool,
                                            "content": f"<function_response>\nWrite your message to the user.\n</function_response>\n<response_format>\nText\n</response_format>"})
            agent.chat_history.add_message({"role": Roles.assistant, "content": output})
            print(output)
            break
        agent_output = agent.get_chat_response("<function_response>\n" + agent_output[0][
            "return_value"] + "\n</function_response>\n<response_format>\nJSON function call.\n</response_format>",
                                               role=Roles.tool, llm_sampling_settings=settings,
                                               system_prompt_modules=[memory_intro_section, memory_section,
                                                                      date_time_section],
                                               structured_output_settings=output_settings)
