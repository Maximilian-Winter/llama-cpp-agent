""" chat template function handler"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

current_file_path = Path(__file__).resolve()  # Resolve to get the absolute path
directory_path = current_file_path.parent  # Get the directory
env = Environment(loader=FileSystemLoader(directory_path))


def raise_exception(message):
    """raise exception."""
    raise ValueError(message)


def alpaca_template(messages, add_generation_prompt=True) -> str:
    """Alpaca template."""
    chat_template = env.get_template('alpaca.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def amberchat_template(messages, add_generation_prompt=True) -> str:
    """Amberchat template."""
    chat_template = env.get_template('amberchat.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def chatml_template(messages, add_generation_prompt=True) -> str:
    """Chatml template."""
    chat_template = env.get_template('chatml.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def chatqa_template(messages, add_generation_prompt=True) -> str:
    """Chatqa template."""
    chat_template = env.get_template('chatqa.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def falcon_template(messages, add_generation_prompt=True) -> str:
    """Falcon template."""
    chat_template = env.get_template('falcon.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def gemma_template(messages, add_generation_prompt=True) -> str:
    """Gemma template."""
    chat_template = env.get_template('gemma.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def llama_2_template(messages, add_generation_prompt=True) -> str:
    """Llama-2 template."""
    chat_template = env.get_template('llama-2.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def llama_3_template(messages, add_generation_prompt=True):
    """Llama-3 template."""
    chat_template = env.get_template('llama-3.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def mistral_template(messages, add_generation_prompt=True):
    """Mistral instruct template."""
    chat_template = env.get_template('mistral.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def openchat_template(messages, add_generation_prompt=True) -> str:
    """Openchat template."""
    chat_template = env.get_template('openchat.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def phi_3_template(messages, add_generation_prompt=True) -> str:
    """Phi-3 template."""
    chat_template = env.get_template('phi-3.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def saiga_template(messages, add_generation_prompt=True) -> str:
    """Saiga template."""
    chat_template = env.get_template('saiga.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def solar_template(messages, add_generation_prompt=True) -> str:
    """Solar template."""
    chat_template = env.get_template('solar.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def vicuna_template(messages, add_generation_prompt=True) -> str:
    """Vicuna template."""
    chat_template = env.get_template('vicuna.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()

def zephyr_template(messages, add_generation_prompt=True) -> str:
    """Zephyr template."""
    chat_template = env.get_template('zephyr.jinja')

    # Render the template with the messages and add_generation_prompt
    output = chat_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        raise_exception=raise_exception
    )

    return output.strip()
