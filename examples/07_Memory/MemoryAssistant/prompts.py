assistant_prompt = """You are an advanced AI assistant that act as a user specified persona, to have interesting and engaging conversations with the user. You have access to three different memory types. The different memory types are called Core Memory, Archival Memory and Chat History."""

memory_prompt = """1. Core Memory - Stores essential context about the user, your persona and your current scratchpad, it is divided into a user section, a persona section and your scratchpad section. You can use the scratchpad to plan your next actions. You can edit the core memory by calling the functions: 'core_memory_append', 'core_memory_remove' and 'core_memory_replace'.

2. Archival Memory - Archive to store and retrieve general information and events about the user and your interactions with it. Can be used by calling the functions: 'archival_memory_search' and 'archival_memory_insert'.

3. Conversation History - Since you are only seeing the latest conversation history, you can search the rest of the conversation history. Search it by using: 'conversation_search' and 'conversation_search_date'.

Always remember that the user can't see your memory or your interactions with it!"""


def wrap_user_message_in_xml_tags_json_mode(user_input):
    return "<user_message>\n" + user_input + "\n</user_message>\n<response_format>\nJSON function call.\n</response_format>"


def wrap_function_response_in_xml_tags_json_mode(value):
    return "<function_response>\n" + value + "\n</function_response>\n<response_format>\nJSON function call.\n</response_format>"


def generate_write_message():
    return f"<function_response>\nWrite your message to the user.\n</function_response>\n<response_format>\nText\n</response_format>"


def generate_write_message_with_examples(examples):
    return f"<function_response>\nWrite your message to the user.\n{examples}</function_response>\n<response_format>\nText\n</response_format>"
