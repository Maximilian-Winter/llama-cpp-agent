from typing import List, Callable

from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.llm_prompt_template import PromptTemplate
from llama_cpp_agent.providers.provider_base import LlmSamplingSettings


class AgentChainElement:
    """
    Represents a single element in the chain of an agent-based framework. This element holds all necessary data to
    manage the prompt, process the input/output, and adjust the processing behavior based on given parameters.

    Attributes:
        output_identifier (str): Unique identifier for the output of this chain element.
        system_prompt (str): Template string for the system prompt.
        prompt (str): Template string for the main prompt to the language model.

        preprocessor (Callable[[str, str, dict], tuple[str, str, dict]], optional): Function to preprocess the input
                before sending it to the language model. Takes the system prompt with template fields replaced, the prompt
                with template fields replaced, and the additional information dictionary as arguments and returns the
                modified tuple of it.
        postprocessor (Callable[[str, str, dict, str], str], optional): Function to postprocess the output from
                the language model. Takes the system prompt, the prompt, the additional information dictionary and the
                result as arguments and returns the modified result.
        function_tool_registry (LlamaCppFunctionToolRegistry): Registry for LlamaCppFunctionTool and enables function calling.
        add_prompt_to_chat_history (bool): Flag to determine if the prompt should be added to the chat history.
        add_response_to_chat_history (bool): Flag to determine if the response should be added to the chat history.

    Methods:
        __init__: Constructs an instance of the AgentChain class.
    """

    def __init__(
        self,
        output_identifier: str,
        system_prompt: str,
        prompt: str,
        tools: List[LlamaCppFunctionTool] = None,
        structured_output_settings: LlmStructuredOutputSettings = None,
        preprocessor: Callable[[str, str, dict], tuple[str, str, dict]] = None,
        postprocessor: Callable[[str, str, dict, str], str] = None,
        add_prompt_to_chat_history: bool = False,
        add_response_to_chat_history: bool = False,
        llm_sampling_settings: LlmSamplingSettings = None,
    ):
        """
        Constructs an instance of the AgentChainElement class.

        Args:
            output_identifier (str): Unique identifier for the output of this chain element.
            system_prompt (str): Template string for the system prompt.
            prompt (str): Template string for the main prompt to the language model.
            tools (List[LlamaCppFunctionTool], optional): List of function tools available for use in this chain element.
            preprocessor (Callable[[str, str, dict], tuple[str, str, dict]], optional): Function to preprocess the input
                before sending it to the language model. Takes the system prompt with template fields replaced, the prompt
                with template fields replaced, and the additional information dictionary as arguments and returns the
                modified tuple of it.
            postprocessor (Callable[[str, str, dict, str], str], optional): Function to postprocess the output from
                the language model. Takes the system prompt, the prompt, the additional information dictionary and the
                result as arguments and returns the modified result.
            add_prompt_to_chat_history (bool): Flag to determine if the prompt should be added to the chat history.
            add_response_to_chat_history (bool): Flag to determine if the response should be added to the chat history.
        """
        self.output_identifier = output_identifier
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.add_prompt_to_chat_history = add_prompt_to_chat_history
        self.add_response_to_chat_history = add_response_to_chat_history
        self.structured_output_settings = structured_output_settings
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.function_tool_registry = None
        if tools is not None:
            self.structured_output_settings = (
                LlmStructuredOutputSettings.from_llama_cpp_function_tools(tools)
            )
        self.llm_sampling_settings = llm_sampling_settings


class AgentChain:
    """
    Represents a chain of AgentChainElements that are processed in a sequence to handle an interaction within an
    agent-based system.

    Attributes:
        agent (LlamaCppAgent): The agent responsible for managing and processing the chain.
        chain (List[AgentChainElement]): A list of AgentChainElement instances that make up the chain.

    Methods:
        __init__: Constructs an instance of the AgentChain class.
        run_chain: Processes the entire chain of elements using provided inputs.
    """

    def __init__(self, agent: LlamaCppAgent, chain_elements: List[AgentChainElement]):
        """
        Constructs an instance of the AgentChain class.
        Args:
            agent (LlamaCppAgent): The agent responsible for managing and processing the chain.
            chain_elements (List[AgentChainElement]): A list of AgentChainElement instances that make up the chain.
        """
        self.agent = agent
        self.chain = chain_elements

    def run_chain(self, user_message: str = None, additional_fields: dict = None):
        """
        Executes the chain of agent elements using the initial user message and additional fields, and
        returns the final output and state of the outputs dictionary.

        Args:
            user_message (str, optional): Initial user message to be processed by the chain.
            additional_fields (dict, optional): Dictionary of additional data to be used in the processing of the chain.

        Returns:
            tuple[str, dict]: A tuple containing the concatenated output string from the final element and
            the dictionary of all outputs.
        """
        outputs = {}

        if additional_fields is not None:
            for field, value in additional_fields.items():
                outputs[field] = value

        for chain_element in self.chain:
            sys_prompter = PromptTemplate.from_string(chain_element.system_prompt)
            sys_prompt = sys_prompter.generate_prompt(outputs)
            prompter = PromptTemplate.from_string(chain_element.prompt)
            prompt = prompter.generate_prompt(outputs)
            if chain_element.function_tool_registry is not None:
                sys_prompt += f"\n\nYou can call functions in JSON format.\n\nAvailable Function Tools:\n\n{chain_element.function_tool_registry.get_documentation().strip()}"
            if chain_element.preprocessor is not None:
                sys_prompt, prompt, outputs = chain_element.preprocessor(
                    sys_prompt, prompt, outputs
                )

            outputs[chain_element.output_identifier] = self.agent.get_chat_response(
                user_message if user_message is not None else prompt,
                system_prompt=sys_prompt,
                structured_output_settings=chain_element.structured_output_settings,
                add_response_to_chat_history=chain_element.add_response_to_chat_history,
                add_message_to_chat_history=chain_element.add_prompt_to_chat_history,
                llm_sampling_settings=chain_element.llm_sampling_settings,
            )
            if chain_element.function_tool_registry is not None:
                outputs[chain_element.output_identifier] = outputs[
                    chain_element.output_identifier
                ][0]["return_value"]
            if chain_element.postprocessor is not None:
                outputs[
                    chain_element.output_identifier
                ] = chain_element.postprocessor = chain_element.postprocessor(
                    sys_prompt,
                    prompt,
                    outputs,
                    outputs[chain_element.output_identifier],
                )
        output = "\n".join(
            [val if isinstance(val, str) else str(val) for val in outputs.values()]
        )
        return output, outputs


class MapChain:
    """
    Represents a specialized chain that maps over a list of items and then combines the results using another chain.

    Attributes:
        agent (LlamaCppAgent): The agent responsible for managing and processing the map and combine chains.
        map_chain (AgentChain): An AgentChain instance used to process each item in the list.
        combine_chain (AgentChain): An AgentChain instance used to combine the results of the map chain.
        item_identifier (str): The identifier used to insert each item into the additional_fields dictionary.
        map_output_identifier (str): The identifier used to store the results of the map chain before passing to the
        combine chain.

    Methods:
        __init__: Constructs an instance of the MapChain class.
        run_map_chain: Executes the map chain on a list of items and then processes the results with the combine chain.
    """

    def __init__(
        self,
        agent: LlamaCppAgent,
        map_chain: List[AgentChainElement],
        combine_chain: List[AgentChainElement],
        item_identifier: str = "item",
        map_output_identifier: str = "map_output",
    ):
        """
        Constructs an instance of the MapChain class. This class is designed to process a list of items through
        a mapping chain and then combine the results using a combining chain.

        Args:
            agent (LlamaCppAgent): The agent responsible for managing and processing the map and combine chains.
            map_chain (List[AgentChainElement]): A list of AgentChainElement instances that make up the map chain.
            combine_chain (List[AgentChainElement]): A list of AgentChainElement instances that make up the combine chain.
            item_identifier (str, optional): The identifier used to insert each item into the additional_fields dictionary.
            map_output_identifier (str, optional): The identifier used to store the results of the map chain before passing to the combine chain.
        """
        self.agent = agent
        self.map_chain = AgentChain(agent, map_chain)
        self.combine_chain = AgentChain(agent, combine_chain)
        self.item_identifier = item_identifier
        self.map_output_identifier = map_output_identifier

    def run_map_chain(
        self,
        items_to_map: list[str],
        user_message: str = None,
        additional_fields: dict = None,
    ):
        """
        Executes the map chain over a list of items and then uses the combine chain to process the concatenated
        results.

        Args:
            items_to_map (list[str]): List of strings to be individually processed by the map chain.
            user_message (str, optional): Initial user message to be included in the processing.
            additional_fields (dict, optional): Additional data to be used throughout the map and combine chains.

        Returns:
            tuple: A tuple containing the final output string from the combine chain and the outputs dictionary.
        """
        outputs = {}

        if additional_fields is not None:
            for field, value in additional_fields.items():
                outputs[field] = value
        else:
            additional_fields = {}
        results = []
        for item in items_to_map:
            additional_fields[self.item_identifier] = item
            result, result_dic = self.map_chain.run_chain(
                user_message=user_message, additional_fields=additional_fields
            )
            results.append(result_dic[self.map_chain.chain[-1].output_identifier])
        additional_fields[self.map_output_identifier] = "\n\n".join(results)
        return self.combine_chain.run_chain(additional_fields=additional_fields)
