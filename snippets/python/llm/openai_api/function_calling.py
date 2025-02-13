"""
TAGS: agent|agentic|api|client|function|function-calling|large-language-model|llm|openai|openai-api|openai-client|nlp|tool|tool-calling|tool-use
DESCRIPTION: Experimenting with LLM function calling (tool use) using the OpenAI API spec and using manual prompting
NOTES: https://platform.openai.com/docs/guides/function-calling
"""

import inspect
import json
import os
from typing import Callable

import openai
import pydantic

openai_client = openai.OpenAI(
    base_url=os.environ["LLM_API_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

USER_QUERY = "Please tell me about large language models"
MODEL_NAME = "gpt-4o-mini"

def web_search(search_query: str) -> str:
    """Runs `search_query` through an internet search engine and returns a text summary of the results

    Args:
        search_query (str): The text to be searched for online
    Returns:
        str: a high quality summary of the search results
    """
    return "TODO"


def wikipedia_search(search_query: str) -> str:
    """
    Runs `search_query` through the search function on en.wikipedia.org and returns
    the contents of the first wikipedia page in the search results

        Args:
            search_query (str): The text to be searched for on en.wikipedia.org
        Returns:
            str: the contents of the wikipedia page
    """
    return "TODO"


class WebSearchToolSchema(pydantic.BaseModel):
    search_query: str = pydantic.Field(
        ...,
        description="Text to search on web search engine",
    )


class WikipediaSearchToolSchema(pydantic.BaseModel):
    search_query: str = pydantic.Field(
        ...,
        description="Text to search on en.wikipedia.org",
    )


web_search_tool = openai.pydantic_function_tool(
    name="web_search",
    model=WebSearchToolSchema,
)
wikipedia_search_tool = openai.pydantic_function_tool(
    name="wikipedia_search",
    model=WikipediaSearchToolSchema,
)

print(
    json.dumps(
        {
            "web_search_tool_definition": web_search_tool,
            "wikipedia_search_tool_definition": wikipedia_search_tool,
        },
        indent=4,
    )
)

# using the `tools` parameter in the OpenAI API #
completion = openai_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": """
You are an effective worker and you follow instructions to the letter.
""".strip(),
        },
        {
            "role": "user",
            "content": f"""
Using ONLY the tools available to you, please answer the following user query:
<user-query>
{USER_QUERY}
</user-query>
You must also justify your choice of tool(s)
            """.strip(),
        },
    ],
    tools=[
        # tool schemas according to OpenAI spec
        web_search_tool,
        wikipedia_search_tool,
    ],
)

for tool_call in completion.choices[0].message.tool_calls:
    print(tool_call)

# (annoying) actual message content appears to be always None for tool calls #
print(completion.choices[0].message.content)


# using manual tool parsing #
def func_def_string(func: Callable) -> str:
    """Converts a function definition into a human-readable string using the function docstring"""
    def_str: str = f"def {func.__name__}(\n"
    for param in inspect.signature(func).parameters.values():
        def_str += f"\t{param.name}: {param.annotation.__name__},\n"
    def_str += "):"
    def_str += f'''
    """
    {func.__doc__.strip()}
    """
    ...
'''.rstrip()

    return def_str


system_prompt: str = f"""
You are an effective worker and you follow instructions to the letter.

You have the following python functions available to you to assist you in your work:

1. "web_search"

2. "wikipedia_search"

These functions are defined as follows:

<web-search-definition>
{func_def_string(web_search)}
</web-search-definition>

<wikipedia-search-definition>
{func_def_string(wikipedia_search)}
</wikipedia-search-definition>

For every function which you wish to run, include a JSON markdown code block, formatted as follows:
```json
{{
    "function_name": "<function-name-here>",
    "function_kwargs": {{
        "<param-name-here>": "<param-value-here>",
        ...
    }}
}}       
```

You may call multiple functions by including multiple JSON markdown code blocks in your response.

Each function will be run as follows and the results reported back to you:
```python
result = funcs[function_name](**function_kwargs)
```
""".strip()

print(system_prompt)

completion = openai_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"""
Using ONLY the tools available to you, please answer the following user query:
<user-query>
{USER_QUERY}
</user-query>
You must also justify your choice of tool(s)
            """.strip(),
        },
    ],
)

print(completion.choices[0].message.content)
