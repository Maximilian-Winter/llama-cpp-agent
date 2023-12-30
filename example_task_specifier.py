from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_prompt_template import Prompter, PromptTemplateFields
from llama_cpp_agent.messages_formatter import MessagesFormatterType

main_model = Llama(
    "../gguf-models/openhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=30,
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

task = 'Implement a chat bot frontend in HTML, CSS and Javascript with a dark UI under the "./workspace" folder without.'

template_fields = PromptTemplateFields()

template_fields.add_field("task", task)

task_specifier_sys_msg = "You are an advanced AI agent that working as a task creator and is responsible to split a given task into sub-tasks."
task_specifier_prompt_template = """Here is a task: {task}.
Please split it into manageable sub task. Do not add anything else."""

task_specifier_prompter = Prompter.from_string(task_specifier_prompt_template)
task_specifier_prompt = task_specifier_prompter.generate_prompt(template_fields.get_fields_dict())
agent_task_specifier = LlamaCppAgent(model=main_model, debug_output=True, system_prompt=task_specifier_sys_msg, predefined_messages_formatter_type=MessagesFormatterType.SYNTHIA)