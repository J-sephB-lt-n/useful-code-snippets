"""
TAGS: abstract|abstraction|client|gpt|interface|language|large language model|llm|nlp
DESCRIPTION: A client managing usage and tracking of a Large Language Model which is accessed using the openai API spec
REQUIREMENTS: pip install openai pydantic
"""

import os
from typing import Optional

import openai
import pydantic


class TokenUsageRecord(pydantic.BaseModel):
    """An immutable record of token usage resulting from a single model call"""

    model_config = pydantic.ConfigDict(frozen=True)

    usage_description: str = pydantic.Field(
        ...,
        min_length=3,
        max_length=99,
        description="Description of the LLM call",
    )
    input_tokens: int = pydantic.Field(
        ...,
        ge=0,
        description="Number of input (prompt) tokens used",
    )
    output_tokens: int = pydantic.Field(
        ...,
        ge=0,
        description="Number of output (completion) tokens used",
    )


class LLM:
    """
    A client managing usage and tracking of a Large Language Model which is accessed using the openai API spec

    Example:
        >>> llm = LLM(model_name="gpt-4o", temperature=0.2, save_chat_history=True)
        >>> llm.chat(
        ...     messages=[
        ...         {"role": "system", "content": "you are a funny comedian"},
        ...         {"role": "user", "content": "please tell me a very short joke"}
        ...     ],
        ...     usage_description="joke generation"
        ... )
        "Why don't skeletons fight each other? They don't have the guts!"
        >>> llm.chat(
        ...     messages=[
        ...         {"role": "system", "content": "you are very foolish"},
        ...         {"role": "user", "content": "please give me some advice"}
        ...     ],
        ...     usage_description="advice generation"
        ... )
        "Of course! I\'d be happy to help. What kind of advice are you looking for? It could be related to \
personal development, career, relationships, or anything else you have in mind. Let me know!'
        >>> llm.token_usage_history
        (   
            TokenUsageRecord(usage_description='joke generation', input_tokens=23, output_tokens=14),
            TokenUsageRecord(usage_description='advice generation', input_tokens=20, output_tokens=42),
        )
        >>> import json; print( json.dumps(llm.chat_history, indent=4) )
        ...
    """

    def __init__(
        self,
        model_name: str,
        temperature: float | int,
        save_chat_history: bool = False,
    ) -> None:
        self.llm_client = openai.OpenAI(
            base_url=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.model_name: str = model_name
        self.chat_history: Optional[list] = [] if save_chat_history else None
        self.temperature: float | int = temperature
        self.token_usage_history: tuple[TokenUsageRecord, ...] = tuple()

    def chat(self, messages: list[dict], usage_description: str) -> str:
        """Create a chat completion and return the text string"""
        llm_response: openai.types.chat.chat_completion.ChatCompletion = (
            self.llm_client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=messages,
            )
        )
        llm_response_text: Optional[str] = llm_response.choices[0].message.content
        if self.chat_history is not None:  # if chat_history is being saved
            self.chat_history.append(
                messages
                + [
                    {
                        "role": "assistant",
                        "content": llm_response_text,
                    }
                ]
            )
        self.token_usage_history = tuple(
            list(self.token_usage_history)
            + [
                TokenUsageRecord(
                    usage_description=usage_description,
                    input_tokens=llm_response.usage.prompt_tokens,
                    output_tokens=llm_response.usage.completion_tokens,
                )
            ]
        )

        return llm_response_text
