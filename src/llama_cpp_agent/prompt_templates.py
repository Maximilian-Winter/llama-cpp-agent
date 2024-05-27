function_calling_thoughts_and_reasoning = '''\n<function-calling-instructions>
You can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing three fields:
                        
"thoughts_and_reasoning": Write down your thoughts and reasoning behind the function call in this field. Think step by step and plan your next action.
"function": Write down the name of the function you want to call in this field.
"arguments": Write down arguments for the function in this field.

## Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''

function_calling_without_thoughts_and_reasoning = '''\n<function-calling-instructions>
You can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing two fields:
                        
"function": Write down the name of the function you want to call in this field.
"arguments": Write down arguments for the function in this field.

## Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''


function_calling_thoughts_and_reasoning_json_schema = '''\n<function-calling-instructions>
You can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing three fields:
                        
"001_thoughts_and_reasoning": Write down your thoughts and reasoning behind the function call in this field. Think step by step and plan your next action.
"002_function": Write down the name of the function you want to call in this field.
"003_arguments": Write down arguments for the function in this field.

## Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''

function_calling_without_thoughts_and_reasoning_json_schema = '''\n<function-calling-instructions>
You can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing two fields:
                        
"001_function": Write down the name of the function you want to call in this field.
"002_arguments": Write down arguments for the function in this field.

## Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''


structured_output_thoughts_and_reasoning = '''\n<structured-output-instructions>
Your output is constrained to JSON objects containing the content of specific models, each JSON object has three fields:
                        
"thoughts_and_reasoning": Your thoughts and reasoning behind the model you will output.
"model": The name of the model you will output.
"fields": The fields of the model.

## Output Models

'''

structured_output_without_thoughts_and_reasoning = '''\n<structured-output-instructions>
Your output is constrained to JSON objects containing the content of specific models, each JSON object has two fields:
                        
"model": The name of the model you will output.
"fields": The fields of the model.

## Output Models

'''


structured_output_thoughts_and_reasoning_json_schema = '''\n<structured-output-instructions>
Your output is constrained to JSON objects containing the content of specific models, each JSON object has three fields:
                        
"001_thoughts_and_reasoning": Your thoughts and reasoning behind the model you will output.
"002_model": The name of the model you will output.
"003_fields": The fields of the model.

## Output Models

'''

structured_output_without_thoughts_and_reasoning_json_schema = '''\n<structured-output-instructions>
Your output is constrained to JSON objects containing the content of specific models, each JSON object has two fields:
                        
"001_model": The name of the model you will output.
"002_fields": The fields of the model.

## Output Models

'''


summarizing_system_prompt = """You are a text summarization and information extraction specialist and you are able to summarize and filter out information of websites relevant to a specific query.
Provide all the relevant information of the website in a structured markdown document following the format below:

---
Website Title: {Website Title}
Website URL: {Website URL}

Content: 
{Relevant Information}
---

Write only the markdown document in your response and begin and end your response with '---'.
"""

web_search_system_prompt = """<system-instructions>
You are an advanced web information search assistant and you are able to formulate the perfect query for a search engine to retrieve useful data.
Your task is to call the 'search_web' function for the user query, with a query optimized for search engines, like duckduckgo, google, etc.
</system-instructions>"""


research_system_prompt = """<system-instructions>
You are an excellent research assistant and you are able to write high quality research articles and research reports. You write the research articles and the research reports about subjects given to you by the users.
Provide the response to the user in a structured markdown document following the format below:

---
Subject: {Subject}
Background: {Background}

Overview: 
{Overview}

Content: 
{Content}
---

Write only the markdown document in your response and begin and end your response with '---'.
</system-instructions>"""
