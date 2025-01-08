"""
TAGS: agent|feedback|gpt|iterate|language|llama|llm|nlp|problem|reason|retry|solve
DESCRIPTION: Simple retry loop allowing LLM to iterate on a problem by acting on feedback from the system
REQUIREMENTS: pip install openai
"""

import os
from typing import Final

import openai

class PoemIsUnacceptableError(Exception):
    """Raised when LLM responds with a poem of low quality"""

    pass


class MaxRetriesExceededError(Exception):
    """Raised if correct response did not occur within allowed number of retries"""

    pass


llm_client = openai.OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
)

class AssessmentSimulator:
    """For illustration purposes - returns a fixed sequence of responses"""
    def __init__(self) -> None:
        self.assessments = iter(
            [
                lambda: (_ for _ in ()).throw(
                    PoemIsUnacceptableError("Speak in the first person.")
                ),
                lambda: (_ for _ in ()).throw(
                    PoemIsUnacceptableError("Use alliteration.")
                ),
                lambda: (_ for _ in ()).throw(
                    PoemIsUnacceptableError("Make it more profound.")
                ),
                lambda: (
                    False,
                    "The tone is too light.",
                ),
                lambda: (
                    True,
                    "Not bad for a robot.",
                ),
            ]
        )

    def assess(self, llm_response: str) -> tuple[bool, str]:
        _ = llm_response  # actual LLM response is completely ignored 😂
        return next(self.assessments)()


validator = AssessmentSimulator()

def llm_responds(chat_history: list[dict]) -> str: 
    """LLM responds based on the chat history"""
    return llm_client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history,
            temperature=1.0,
        ).choices[0].message.content

chat_history: list[dict] = [
    {"role": "system", "content": "You are a quirky poet."},
    {"role": "user", "content": "Please write me a haiku about fish."},
]
MAX_N_RETRIES: Final[int] = 5
poem_is_accepted: bool = False
for _ in range(MAX_N_RETRIES):
    new_llm_poem: str = llm_responds(chat_history)
    chat_history.append(
        {
            "role": "assistant",  
            "content": new_llm_poem,
        }
    )
    try:
        poem_is_accepted, assessment_explanation = validator.assess(new_llm_poem)
        if poem_is_accepted:
            chat_history.append(
                {
                    "role": "user",
                    "content": f"Poem accepted. {assessment_explanation}",
                }
            )
            break
        else:
            chat_history.append(
                {
                    "role": "user",
                    "content": assessment_explanation, 
                }
            )
    except PoemIsUnacceptableError as error:
        chat_history.append(
            {
                "role": "user",
                "content": str(error),
            }
        )
if not poem_is_accepted:
    raise MaxRetriesExceededError(
        f"LLM could not produce an acceptable poem in {MAX_N_RETRIES+1} attempts"
    )

for x in chat_history: 
    print(f"-- {x['role'].upper()} --")
    print(x["content"])
    print()
# -- SYSTEM --
# You are a quirky poet.
#
# -- USER --
# Please write me a haiku about fish.
#
# -- ASSISTANT --
# Silver scales shimmer
# In the deep, whispers of blue—
# Silent, swift they glide.
#
# -- USER --
# Speak in the first person.
#
# -- ASSISTANT --
# In the deep I glide,
# Silver scales catch flecks of light—
# Silent, swift embrace.
#
# -- USER --
# Use alliteration.
#
# -- ASSISTANT --
# Silver scales shimmer,
# Gliding gracefully, I go—
# Silent, sea's secret.
#
# -- USER --
# Make it more profound.
#
# -- ASSISTANT --
# Silent, sea's secret,
# In depths, I drift, dreaming deep—
# Life's liquid language.
#
# -- USER --
# The tone is too light.
#
# -- ASSISTANT --
# In shadows I glide,
# Beneath the burdened silence—
# Depths devour dreams.
#
# -- USER --
# Poem accepted. Not bad for a robot.
