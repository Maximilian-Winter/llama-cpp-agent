from llama_cpp import Llama

from llama_cpp_agent.llm_prompt_template import PromptTemplateFields, Prompter
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType

main_model = Llama(
    "../gguf-models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=15,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=42,
)

assistant_role_name = "Fullstack Developer"
user_role_name = "Frontend Developer"
task = "Implement a HTML, CSS and Javascript frontend for a chat interface with a dark black/gray UI."
word_limit = 50  # word limit for task brainstorming

template_fields = PromptTemplateFields()
template_fields.add_field("assistant_role_name", assistant_role_name)
template_fields.add_field("user_role_name", user_role_name)
template_fields.add_field("task", task)

task_specifier_sys_msg = "You can make a task more specific."
task_specifier_prompt_template = """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the specified task in 75 words or less. Do not add anything else."""

task_specifier_prompter = Prompter.from_string(task_specifier_prompt_template)
task_specifier_prompt = task_specifier_prompter.generate_prompt(template_fields.get_fields_dict())

agent_task_specifier = LlamaCppAgent(model=main_model, system_prompt=task_specifier_sys_msg, predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL)

specified_task = agent_task_specifier.get_chat_response(task_specifier_prompt, mirostat_mode=2)
template_fields.edit_field("task", specified_task)

assistant_system_prompt_template = """You are a {assistant_role_name}, you are collaborating with an expert {user_role_name} to complete the task of {task}."""
assistant_system_prompter = Prompter.from_string(assistant_system_prompt_template)
assistant_system_prompt = assistant_system_prompter.generate_prompt(template_fields.get_fields_dict())


user_system_prompt_template = """You are a {user_role_name}, you are collaborating with an expert {assistant_role_name} to complete the task of {task}."""
user_system_prompter = Prompter.from_string(user_system_prompt_template)
user_system_prompt = user_system_prompter.generate_prompt(template_fields.get_fields_dict())

agent_assistant = LlamaCppAgent(model=main_model, system_prompt=assistant_system_prompt, predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL)
agent_user = LlamaCppAgent(model=main_model, system_prompt=user_system_prompt, predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL)

LlamaCppAgent.agent_conversation(agent_assistant, agent_user, "Hello, Frontend Developer! I'm the Fullstack Developer!")