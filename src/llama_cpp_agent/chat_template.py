""" chat template function handler"""

from jinja2 import Template

def raise_exception(message):
    """raise exception."""
    raise ValueError(message)

def alpaca_template(messages, add_generation_prompt=True) -> str:
    """Alpaca template."""
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
    """Amberchat template."""
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
    """Chatml template."""
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
    """Chatqa template."""
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
    """Falcon template."""
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
    """Gemma template."""
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
    """Llama-2 template."""
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
    """Llama-3 template."""
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
    """Mistral instruct template."""
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
    """Openchat template."""
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
    """Phi-3 template."""
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
    """Saiga template."""
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
    """Solar template."""
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
    """Vicuna template."""
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
    """Zephyr template."""
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
