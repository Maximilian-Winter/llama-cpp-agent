# Example Recipe Recommendation and Nutrition Analysis
from llama_cpp_agent import AgentChainElement, AgentChain
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL
)

recipe_element = AgentChainElement(
    output_identifier="recipe_recommendation",
    system_prompt="You are a recipe recommendation assistant that suggests recipes based on user preferences.",
    prompt="Recommend a {meal_type} recipe that includes {ingredient1} and {ingredient2}."
)

nutrition_element = AgentChainElement(
    output_identifier="nutrition_analysis",
    system_prompt="You are a nutrition analysis assistant that provides nutritional information for recipes.",
    prompt="Analyze the nutritional content of the following recipe: {recipe_recommendation}. Provide a breakdown of calories, protein, carbohydrates, and fat."
)

chain = [recipe_element, nutrition_element]

agent_chain = AgentChain(agent, chain)

output, _ = agent_chain.run_chain(additional_fields={
    "meal_type": "dinner",
    "ingredient1": "chicken",
    "ingredient2": "spinach"
})