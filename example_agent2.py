import json

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation, sanitize_json_string, map_grammar_names_to_pydantic_model_class

from example_agent_models import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType

from llama_cpp_agent.llm_prompt_template import PromptTemplateFields, Prompter

pydantic_function_models = [SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile]

gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
    pydantic_function_models, "function", "function_parameters", "Function",
    "Function Parameter")
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

output_to_pydantic_model = map_grammar_names_to_pydantic_model_class(pydantic_function_models)

main_model = Llama(
    "../gguf-models/mistral-7b-instruct-v0.2.Q8_0.gguf",
    n_gpu_layers=45,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)

system_prompt = f'''You are an advanced AI agent called AutoCoder. As AutoCoder your primary task is to autonomously plan, outline and implement complete software projects based on user specifications. You have to use JSON objects to perform functions.
Here are your available functions:
{documentation}'''.strip()

task = 'Implement a chat bot frontend in HTML, CSS and Javascript with a dark UI under the "./workspace" folder.'

template_fields = PromptTemplateFields()

template_fields.add_field("task", task)

task_specifier_sys_msg = "You are an advanced AI agent that working as a task creator and is responsible to split a given task into sub-tasks."
task_specifier_prompt_template = """Here is a task: {task}.
Please split it into manageable sub task. Do not add anything else."""

task_specifier_prompter = Prompter.from_string(task_specifier_prompt_template)
task_specifier_prompt = task_specifier_prompter.generate_prompt(template_fields.get_fields_dict())
# agent_task_specifier = LlamaCppAgent(model=main_model, debug_output=True, system_prompt=task_specifier_sys_msg, predefined_messages_formatter_type=MessagesFormatterType.SYNTHIA)

# specified_task = agent_task_specifier.get_chat_response(task_specifier_prompt, temperature=0.65, mirostat_mode=2)


wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt,
                              predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL)

user_input = task
while True:

    if user_input is None:
        user_input = "Proceed with next step."

    response = wrapped_model.get_chat_response(
        user_input,
        temperature=0.0, mirostat_mode=0, mirostat_tau=3.0, mirostat_eta=0.1, grammar=grammar)

    sanitized = sanitize_json_string(response)
    function_call = json.loads(sanitized)
    cls = output_to_pydantic_model[function_call["function"]]
    call_parameters = function_call["function_parameters"]
    call = cls(**call_parameters)
    user_input = call.run()
