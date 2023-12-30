import datetime
import json

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation, sanitize_json_string, map_grammar_names_to_pydantic_model_class

from example_agent_models_auto_coder import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile, \
    agent_dev_folder_setup
from llama_cpp_agent.messages_formatter import MessagesFormatterType

pydantic_function_models = [GetFileList, ReadTextFile, WriteTextFile]

gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
    pydantic_function_models, True, "function", "function_parameters", "Function",
    "Function Parameter")

grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

output_to_pydantic_model = map_grammar_names_to_pydantic_model_class(pydantic_function_models)

main_model = Llama(
    "../gguf-models/neuralhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=46,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=16384,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)

system_prompt_planner = f'''You are an advanced AI agent called AutoPlanner. As AutoPlanner your primary task is to autonomously plan complete software projects based on user specifications. You will create a complete development plans.  Your output is constrained to write JSON function call objects. The content of files is constrained to markdown code blocks with different content types like HTML, CSS, Javascript, Python or  Markdown. Here are your available functions:

{documentation}'''.strip()
system_prompt_coder = f'''You are an advanced AI agent called AutoCoder. As AutoCoder your primary task is to autonomously implement complete software projects based on a development plan. Your output is constrained to write JSON function call objects. The content of files is constrained to markdown code blocks with different content types like HTML, CSS, Javascript, Python or  Markdown. Here are your available functions:

{documentation}'''.strip()

task = 'Develop a complete development plan for a fullstack chat bot system with a backend using huggingface transformers library in the current working directory, which is empty.'
task_implement = 'Implement the existing development plan in the "./" folder, for a chat bot frontend in HTML, CSS and Javascript with a dark UI.'
timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

agent_dev_folder_setup(f"dev_{timestamp}")
# agent_dev_folder_setup("agent_auto_coder_auto_planner_output")
planner_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt_planner,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

coder_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt_coder,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)
user_input = task
while True:

    if user_input is None:
        user_input = "Proceed with next step."

    response = planner_agent.get_chat_response(
        user_input,
        temperature=0.45, top_p=1.0, top_k=0, tfs_z=0.95, repeat_penalty=1.12, grammar=grammar)

    if "write-text-file" in response:
        response_lines = response.split("\n")

        # Get the first line JSON object
        response = response_lines[0]
        # Remove the first line
        response_lines.pop(0)
        # Remove the first line Markdown code block marker
        response_lines.pop(0)
        # Remove the last line Markdown code block marker
        response_lines.pop(-1)
        # Combine lines into a single string
        content = "\n".join(response_lines)
        sanitized = sanitize_json_string(response)
        function_call = json.loads(sanitized)
        cls = output_to_pydantic_model[function_call["function"]]
        function_call["function_parameters"]["file_string"] = content

    else:
        sanitized = sanitize_json_string(response)
        function_call = json.loads(sanitized)
        cls = output_to_pydantic_model[function_call["function"]]

    call_parameters = function_call["function_parameters"]
    call = cls(**call_parameters)
    user_input = call.run()
