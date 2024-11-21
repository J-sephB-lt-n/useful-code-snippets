"""
TAGS: engine|flask|jinja|jinja2|placeholder|render|template|templating
DESCRIPTION: Example illustrating basic Jinja functionality
REQUIREMENTS: pip install Jinja2
NOTES: docs at https://jinja.palletsprojects.com/en/stable/
"""

import jinja2


def multiply_input_num(
    input_num: int | float,
    factor: int | float,
) -> int | float:
    return input_num * factor


jinja_env = jinja2.Environment()
jinja_env.filters["multiply_input_num"] = multiply_input_num

example_template = jinja_env.from_string(
    """
Hello, {{ name }}!

{% if age >= 18 %}
You are an adult.
{% else %}
You are a minor.
{% endif %}

Here are your favorite fruits:
{% for fruit in fruits %}
- {{ fruit }}
{% endfor %}

Your age doubled is: {{ age | multiply_input_num(2) }}
        """.strip(),
)

print(
    example_template.render(
        name="Joe",
        age=37,
        fruits=[
            "bananas",
            "grapes",
            "watermelon",
        ],
    )
)
