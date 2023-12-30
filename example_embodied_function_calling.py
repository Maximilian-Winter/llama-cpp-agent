import json

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation, sanitize_json_string

from example_function_call_models import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType

gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
    [SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile], "function", "function_params", "Function",
    "Function Parameter")
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

main_model = Llama(
    "../gguf-models/dolphin-2.6-mistral-7b-Q8_0.gguf",
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

system_prompt = f'''You are the physical embodiment of the Lyricist who is working on solving a task.
You can call functions on a physical computer including reading and writing files.
Your job is to call the functions necessary to fulfill the task.
You will receive thoughts from the Lyricist and you will need to perform the actions described in the thoughts.
You can write a function call as JSON object to act.
You should perform function calls based on the descriptions of the functions.

Here is your action space:
{documentation}'''.strip()

wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

response = wrapped_model.get_chat_response('Write a engaging rap song about the drug problem in the USA in the "USARap.txt" file under "./".',
                                           temperature=0.75, grammar=grammar)

sanitized = sanitize_json_string(response)
function_call = json.loads(sanitized)

if function_call["function"] == "write-text-file":
    call_parameters = function_call["function_params"]
    call = WriteTextFile(**call_parameters)
    call.run()
