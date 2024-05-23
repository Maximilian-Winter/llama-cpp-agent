function_calling_thoughts_and_reasoning = '''\n\nYou can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing three fields:
                        
"thoughts_and_reasoning": Write down your thoughts and reasoning behind the function call in this field. Think step by step and plan your next action.
"function": Write down the name of the function you want to call in this field.
"arguments": Write down arguments for the function in this field.

## Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''

function_calling_without_thoughts_and_reasoning = '''\n\nYou can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing two fields:
                        
"function": Write down the name of the function you want to call in this field.
"arguments": Write down arguments for the function in this field.

## Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''


function_calling_thoughts_and_reasoning_json_schema = '''\n\nYou can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing three fields:
                        
"001_thoughts_and_reasoning": Write down your thoughts and reasoning behind the function call in this field. Think step by step and plan your next action.
"002_function": Write down the name of the function you want to call in this field.
"003_arguments": Write down arguments for the function in this field.

## Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''

function_calling_without_thoughts_and_reasoning_json_schema = '''\n\nYou can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing two fields:
                        
"001_function": Write down the name of the function you want to call in this field.
"002_arguments": Write down arguments for the function in this field.

## Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''


structured_output_thoughts_and_reasoning = '''\n\nYour output is constrained to JSON objects containing the content of specific models, each JSON object has three fields:
                        
"thoughts_and_reasoning": Your thoughts and reasoning behind the model you will output.
"model": The name of the model you will output.
"fields": The fields of the model.

## Output Models

'''

structured_output_without_thoughts_and_reasoning = '''\n\nYour output is constrained to JSON objects containing the content of specific models, each JSON object has two fields:
                        
"model": The name of the model you will output.
"fields": The fields of the model.

## Output Models

'''


structured_output_thoughts_and_reasoning_json_schema = '''\n\nYour output is constrained to JSON objects containing the content of specific models, each JSON object has three fields:
                        
"001_thoughts_and_reasoning": Your thoughts and reasoning behind the model you will output.
"002_model": The name of the model you will output.
"003_fields": The fields of the model.

## Output Models

'''

structured_output_without_thoughts_and_reasoning_json_schema = '''\n\nYour output is constrained to JSON objects containing the content of specific models, each JSON object has two fields:
                        
"001_model": The name of the model you will output.
"002_fields": The fields of the model.

## Output Models

'''
