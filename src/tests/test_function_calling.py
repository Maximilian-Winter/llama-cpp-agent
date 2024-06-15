from llama_cpp import LlamaGrammar
import pytest

from llama_cpp_agent.function_calling import (
    LlamaCppFunctionTool,
    LlamaCppFunctionToolRegistry,
    BaseModel,
)


class MockModel(BaseModel):
    arg1: int
    arg2: int

    def run(self) -> int:
        return self.arg1 * self.arg2


@pytest.fixture
def llama_tool_registry():
    return LlamaCppFunctionToolRegistry(allow_parallel_function_calling=False)


def test_llama_cpp_function_tool_init():
    llama_tool = LlamaCppFunctionTool(MockModel)
    assert llama_tool.model == MockModel
    assert not llama_tool.add_params_to_result
    assert not llama_tool.look_for_field_string
    assert not llama_tool.has_markdown_code_block
    assert not llama_tool.has_triple_quoted_string
    assert llama_tool.additional_parameters == {}


def test_llama_cpp_function_tool_call():
    llama_tool = LlamaCppFunctionTool(MockModel)
    result = llama_tool(arg1=10, arg2=2)
    assert isinstance(result, MockModel)
    assert result.arg1 == 10
    assert result.arg2 == 2


def test_llama_cpp_function_tool_registry_register(llama_tool_registry):
    llama_tool = LlamaCppFunctionTool(MockModel)
    llama_tool_registry.register_function_tool(llama_tool)
    assert llama_tool_registry.function_tools == {"MockModel": llama_tool}


def test_llama_cpp_function_tool_registry_get_function_tool(llama_tool_registry):
    llama_tool = LlamaCppFunctionTool(MockModel)
    llama_tool_registry.register_function_tool(llama_tool)
    assert llama_tool_registry.get_function_tool("MockModel") == llama_tool


def test_llama_cpp_function_tool_registry_finalization(llama_tool_registry):
    llama_tool = LlamaCppFunctionTool(MockModel)
    llama_tool_registry.register_function_tool(llama_tool)
    llama_tool_registry.finalize()
    assert isinstance(llama_tool_registry.grammar, LlamaGrammar)
    print(llama_tool_registry.get_documentation())
    assert (
        llama_tool_registry.get_documentation() == "function: MockModel\n  params:\n    arg1 (int)\n    arg2 (int)\n\n"
    )


def test_llama_cpp_function_tool_registry_handle_function_call(llama_tool_registry):
    llama_tool = LlamaCppFunctionTool(MockModel)
    llama_tool_registry.register_function_tool(llama_tool)
    llama_tool_registry.finalize()
    response = '{"function": "MockModel", "params": {"arg1": 10, "arg2": 2}}'
    output = llama_tool_registry.handle_function_call(response)
    assert output[0]["function"] == "MockModel"
    assert output[0]["return_value"] == 20
