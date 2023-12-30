import json

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation, sanitize_json_string, map_grammar_names_to_pydantic_model_class

from example_agent_models import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType

pydantic_function_models = [SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile]

gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
    pydantic_function_models, "function", "function_params", "Function",
    "Function Parameter")
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

output_to_pydantic_model = map_grammar_names_to_pydantic_model_class(pydantic_function_models)

main_model = Llama(
    "../gguf-models/synthia-v3.0-11b.Q5_K_M.gguf",
    n_gpu_layers=35,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=42,
)

system_prompt = f'''You are the physical embodiment of the Developer who is working on solving a task.
You can call functions on a physical computer including reading and writing files.
Your job is to call the functions necessary to fulfill the task.
You will receive thoughts from the Developer and you will need to perform the actions described in the thoughts.
You can write a function call as JSON object to act.
You should perform function calls based on the descriptions of the functions.

Here is your action space:
{documentation}'''.strip()

wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt,
                              predefined_messages_formatter_type=MessagesFormatterType.SYNTHIA)


first_input = 'Implement a chat bot frontend in HTML, CSS and Javascript under "./workspace".'
user_input = None
while True:

    if user_input is None:
        user_input = input(">")

    response = wrapped_model.get_chat_response(
        user_input,
        temperature=0.35, grammar=grammar)

    sanitized = sanitize_json_string(response)
    function_call = json.loads(sanitized)
    cls = output_to_pydantic_model[function_call["function"]]
    call_parameters = function_call["function_params"]
    call = cls(**call_parameters)
    user_input = call.run()

