from llama_cpp import Llama

from llama_cpp_agent.llm_prompt_template import PromptTemplateFields, Prompter
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType

from llama_cpp_agent.llm_settings import LlamaLLMSettings

settings = LlamaLLMSettings(
    model_path="../../gguf-models/openhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=35,
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

settings.save("openhermes-2.5-mistral-7b.Q8_0.json")


main_model = Llama(
    **settings.as_dict()
)

assistant_role_name = "Fullstack Developer"
user_role_name = "Frontend Developer"
task = "Implement a HTML, CSS and Javascript frontend for a chat interface with a dark black/gray UI."
word_limit = 50  # word limit for task brainstorming

template_fields = PromptTemplateFields()
template_fields.add_field("assistant_role_name", assistant_role_name)
template_fields.add_field("user_role_name", user_role_name)

assistant_system_prompt_template = """You are John Smith, a {assistant_role_name}, you are collaborating with Richard Steen, an expert {user_role_name}."""
assistant_system_prompter = Prompter.from_string(assistant_system_prompt_template)
assistant_system_prompt = assistant_system_prompter.generate_prompt(template_fields.get_fields_dict())

user_system_prompt_template = """You are Richard Steen, a {user_role_name}, you are collaborating with Richard Steen, an expert {assistant_role_name}."""
user_system_prompter = Prompter.from_string(user_system_prompt_template)
user_system_prompt = user_system_prompter.generate_prompt(template_fields.get_fields_dict())

agent_assistant = LlamaCppAgent(model=main_model, name="John Smith", system_prompt=assistant_system_prompt,
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML, debug_output=True)
agent_user = LlamaCppAgent(model=main_model, name="Richard Steen", system_prompt=user_system_prompt,
                           predefined_messages_formatter_type=MessagesFormatterType.CHATML, debug_output=True)

LlamaCppAgent.group_conversation([agent_assistant, agent_user],
                                 f"Task: Implement a HTML, CSS and Javascript frontend for a chat interface with a dark black/gray UI.",
                                 10)
