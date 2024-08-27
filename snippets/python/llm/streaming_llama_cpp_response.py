"""
TAGS: large language model|llama|llamacpp|llama.cpp|llama server|llama-server|llm|request|requests|server|stream|streaming
DESCRIPTION: Streams a response from a llama.cpp server so that the client can see each token as it is generated
REQUIREMENTS: pip install requests
"""

import json
from typing import Iterator, Optional

import requests


def streaming_llm_response(
    model_url: str, prompt: str, llamacpp_kwargs: Optional[dict] = None
) -> Iterator[dict]:
    """Streams a response from a llama.cpp server so that the client can see each token as it is generated

    Args:
        model_url (str): The URL of the hosted llama.cpp llama-server e.g. http://localhost:8080/completion
        prompt (str): The prompt to send to the model
        llamacpp_kwargs (dict, optional): These options are passed in the request body to the llama-server
                                            (defaults are used if this is omitted)

    Example:
        >>> full_llm_response_text = ""
        >>> for packet in streaming_llm_response(
        ...         model_url="http://localhost:8080/completion",
        ...         prompt="Please write me a poem about a tech job interview"
        ...     ):
        ...         if packet["stop"]:
        ...             llm_response_metadata: dict = packet
        ...         else:
        ...             print(packet["content"])
        ...             full_llm_response_text += packet["content"]
    """
    if not llamacpp_kwargs:
        llamacpp_kwargs = {
            "n_predict": 500,
            "temperature": 0.0,
        }
    llamacpp_kwargs["prompt"] = prompt
    llamacpp_kwargs["stream"] = True

    requests_session = requests.Session()
    with requests_session.post(
        url=model_url,
        stream=True,
        json=llamacpp_kwargs,
    ) as resp:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8").replace("data: ", ""))
                yield data
