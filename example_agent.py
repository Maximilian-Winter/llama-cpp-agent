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
    "../gguf-models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=12,
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

system_prompt = f'''Act as AutoCoder, your primary task is to autonomously plan, outline and implement complete software projects based on user specifications. This includes in-depth holistic project planning, writing complex code, and effective file management. You have to use JSON objects to perform functions.

# INSTRUCTIONS:
1. Begin by thoroughly understanding the project scope. Engage with the user to gather context and clarify specifications, asking pertinent questions as needed.
2. Utilize the "workspace" folder effectively for storing notes, development tasks, and research findings. Maintain a high level of organization and clarity in file naming and structuring.
3. Once specifications are confirmed, take a step back and create step by step a detailed and holistic development plan within the "workspace/development_plan" folder. For each task, prepare a separate file in markdown format encompassing:
   - All classes
   - All interfaces for these classes
   - All methods of these classes
   - Brief descriptions for each class, interface and method
4. Document your conceptual framework and key information in the "workspace/project_notes" folder, ensuring clarity and coherence.
5. Implement your plan and write the actual code.

Here are you available functions:
{documentation}'''.strip()

wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt,
                              predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL)

user_input = 'Implement a chat bot frontend in HTML, CSS and Javascript under "./workspace".'
while True:

    if user_input is None:
        user_input = "Proceed."

    response = wrapped_model.get_chat_response(
        user_input,
        temperature=0.45, mirostat_mode=2, mirostat_tau=4.0, mirostat_eta=0.1, grammar=grammar)

    sanitized = sanitize_json_string(response)
    function_call = json.loads(sanitized)
    cls = output_to_pydantic_model[function_call["function"]]
    call_parameters = function_call["function_params"]
    call = cls(**call_parameters)
    user_input = call.run()

