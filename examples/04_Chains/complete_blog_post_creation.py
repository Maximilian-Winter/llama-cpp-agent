# Example: Blog Post Creation (Topic, Outline, Introduction, Body, Conclusion, and Meta Description)
from llama_cpp_agent import AgentChainElement, AgentChain
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://127.0.0.1:8080")

agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)
blog_post_topic = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a blog post topic generator",
    prompt="Generate a catchy title for a blog post about {subject}."
)

blog_post_outline = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a blog post outliner",
    prompt="Create an outline for a blog post titled '{out_0}'. Include main sections and subpoints."
)

blog_post_intro = AgentChainElement(
    output_identifier="out_2",
    system_prompt="You are a blog post introduction writer",
    prompt="Write an engaging introduction for a blog post based on the following title and outline:\n--\nTitle: {out_0}\nOutline:\n{out_1}"
)

blog_post_body = AgentChainElement(
    output_identifier="out_3",
    system_prompt="You are a blog post body writer",
    prompt="Write the main body of the blog post based on the following title, outline, and introduction:\n--\nTitle: {out_0}\nOutline:\n{out_1}\nIntroduction:\n{out_2}"
)

blog_post_conclusion = AgentChainElement(
    output_identifier="out_4",
    system_prompt="You are a blog post conclusion writer",
    prompt="Write a compelling conclusion for the blog post based on the following title, outline, introduction, and body:\n--\nTitle: {out_0}\nOutline:\n{out_1}\nIntroduction:\n{out_2}\nBody:\n{out_3}"
)

blog_post_meta = AgentChainElement(
    output_identifier="out_5",
    system_prompt="You are a blog post meta description writer",
    prompt="Write a concise and engaging meta description for the blog post based on the following title, outline, introduction, body, and conclusion:\n--\nTitle: {out_0}\nOutline:\n{out_1}\nIntroduction:\n{out_2}\nBody:\n{out_3}\nConclusion:\n{out_4}"
)

chain = [blog_post_topic, blog_post_outline, blog_post_intro, blog_post_body, blog_post_conclusion, blog_post_meta]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"subject": "The Benefits of Regular Exercise"})

