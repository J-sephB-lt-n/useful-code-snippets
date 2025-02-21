"""
TAGS: ai|calling|func|function|function-use|function-calling|genai|language|large-language-model|llm|model|nlp|prompt|prompting|string|text|tool|tool-use|tool-calling
DESCRIPTION: Get a string representation of a python function (e.g. to include in a prompt to a Large Language Model)
"""

import inspect
import textwrap
from typing import Callable, Optional


def func_definition_as_string(func: Callable, include_body_code: bool) -> str:
    """
    Renders the function definition as text (optionally including the body code)
    """
    if include_body_code:
        return inspect.getsource(func)

    docstring = inspect.getdoc(func)
    if docstring:
        dedented_docstring = textwrap.dedent(docstring)
        indented_docstring = textwrap.indent(f'"""\n{dedented_docstring}\n"""', "    ")
    else:
        indented_docstring = ""

    func_signature = inspect.signature(func)
    func_def: str = f"def {func.__name__}{func_signature}:"

    return f"{func_def}\n{indented_docstring}\n    ..."


if __name__ == "__main__":

    def example_function(
        a: int,
        b,
        x: bool | int = 69,
        y: Optional[str] = None,
    ) -> set[str]:
        """
        A function which definitely does something

        Args:
            a (int): An integer argument
            b: Anything you like
            x (bool|int): Something fishy
            y (Optional[str]): A string, if you like

        Returns:
            set[str]: A set containing an important message
        """
        print(a, b, x, y)
        return {"joe", "is", "wonderful"}

    example_function(a=69, b=420)

    print(func_definition_as_string(example_function, include_body_code=False))
