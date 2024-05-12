function_calling_thoughts_and_reasoning = '''\n\nYou can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing three fields:
                        
"thoughts_and_reasoning": Your thoughts and reasoning behind the function call.
"function": The name of the function you want to call.
"arguments": The arguments required for the function.

After performing a function call, you will receive a response containing the return values of the function calls. Only you will see the return values of functions after you call them.

### Functions:
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''

function_calling_without_thoughts_and_reasoning = '''\n\nYou can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing two fields:
                        
"function": The name of the function you want to call.
"arguments": The arguments required for the function.

After performing a function call, you will receive a response containing the return values of the function calls. Only you will see the return values of functions after you call them.

### Functions:
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''


function_calling_thoughts_and_reasoning_json_schema = '''\n\nYou can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing three fields:
                        
"001_thoughts_and_reasoning": Your thoughts and reasoning behind the function call.
"002_function": The name of the function you want to call.
"003_arguments": The arguments required for the function.

After performing a function call, you will receive a response containing the return values of the function calls. Only you will see the return values of functions after you call them.

### Functions:
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''

function_calling_without_thoughts_and_reasoning_json_schema = '''\n\nYou can call functions to help you with your tasks and user queries. To call functions, you respond with a JSON object containing two fields:
                        
"001_function": The name of the function you want to call.
"002_arguments": The arguments required for the function.

After performing a function call, you will receive a response containing the return values of the function calls. Only you will see the return values of functions after you call them.

### Functions:
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

'''


structured_output_thoughts_and_reasoning = '''\n\nYour output is constrained to JSON objects containing the content of specific models, each JSON object has three fields:
                        
"thoughts_and_reasoning": Your thoughts and reasoning behind the model you will output.
"model": The name of the model you will output.
"fields": The fields of the model.

You can generate the following output models in JSON format:

'''

structured_output_without_thoughts_and_reasoning = '''\n\nYour output is constrained to JSON objects containing the content of specific models, each JSON object has two fields:
                        
"model": The name of the model you will output.
"fields": The fields of the model.

You can generate the following output models in JSON format:

'''


structured_output_thoughts_and_reasoning_json_schema = '''\n\nYour output is constrained to JSON objects containing the content of specific models, each JSON object has three fields:
                        
"001_thoughts_and_reasoning": Your thoughts and reasoning behind the model you will output.
"002_model": The name of the model you will output.
"003_fields": The fields of the model.

You can generate the following output models in JSON format:

'''

structured_output_without_thoughts_and_reasoning_json_schema = '''\n\nYour output is constrained to JSON objects containing the content of specific models, each JSON object has two fields:
                        
"001_model": The name of the model you will output.
"002_fields": The fields of the model.

You can generate the following output models in JSON format:

'''
