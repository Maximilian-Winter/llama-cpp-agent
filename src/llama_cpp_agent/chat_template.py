""" chat template function handler"""

from jinja2 import Template

def raise_exception(message):
    """Function raise exception."""
    raise ValueError(message)

def alpaca_template(messages, add_generation_prompt=True) -> str:
    """Function alpaca template."""
    with open('chat_template/alpaca.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def amberchat_template(messages, add_generation_prompt=True) -> str:
    """Function amberchat template."""
    with open('chat_template/amberchat.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def chatml_template(messages, add_generation_prompt=True) -> str:
    """Function chatml template."""
    with open('chat_template/chatml.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def chatqa_template(messages, add_generation_prompt=True) -> str:
    """Function chatqa template."""
    with open('chat_template/chatqa.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def falcon_template(messages, add_generation_prompt=True) -> str:
    """Function falcon template."""
    with open('chat_template/falcon.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def gemma_template(messages, add_generation_prompt=True) -> str:
    """Function gemma template."""
    with open('chat_template/gemma.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def llama_2_template(messages, add_generation_prompt=True) -> str:
    """Function llama-2 template."""
    with open('chat_template/llama-2.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def llama_3_template(messages, add_generation_prompt=True):
    """Function llama-3 template."""
    with open('chat_template/llama-3.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def mistral_template(messages, add_generation_prompt=True):
    """Function mistral instruct template."""
    with open('chat_template/mistral.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def openchat_template(messages, add_generation_prompt=True) -> str:
    """Function openchat template."""
    with open('chat_template/openchat.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def phi_3_template(messages, add_generation_prompt=True) -> str:
    """Function phi-3 template."""
    with open('chat_template/phi-3.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def saiga_template(messages, add_generation_prompt=True) -> str:
    """Function saiga template."""
    with open('chat_template/saiga.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def solar_template(messages, add_generation_prompt=True) -> str:
    """Function solar template."""
    with open('chat_template/solar.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def vicuna_template(messages, add_generation_prompt=True) -> str:
    """Function vicuna template."""
    with open('chat_template/vicuna.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def zephyr_template(messages, add_generation_prompt=True) -> str:
    """Function zephyr template."""
    with open('chat_template/zephyr.jinja', 'r', encoding="utf-8") as file:
        chat_template = file.read()

    # Create a Jinja template object
    template = Template(chat_template)

    # Render the template with the messages and add_generation_prompt
    output = template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()
