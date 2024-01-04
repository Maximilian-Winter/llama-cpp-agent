from setuptools import setup

setup(
    name='llama-cpp-agent',
    packages=["llama_cpp_agent", "llama_cpp_agent.agent_memory", "llama_cpp_agent.gbnf_grammar_generator"],
    include_package_data=True
)
