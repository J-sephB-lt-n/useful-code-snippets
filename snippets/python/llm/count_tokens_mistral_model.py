"""
TAGS: context|context length|count|counts|count token|count tokens|large language model|llm|mistral|token|tokens
DESCRIPTION: Python code for counting the number of tokens in an input string
REQUIREMENTS: pip install mistral-common
NOTES: Different models use different tokenization schemes - refer to the model documentation in order to identify which tokenizer version is correct
REFERENCE: https://docs.mistral.ai/guides/tokenization/
"""

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest

input_text: str = "The quick brown fox jumped over lazy dog. Apparently."
tokenizer_v1 = MistralTokenizer.v1()
tokenized = tokenizer_v1.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(content=input_text),
        ],
        model="test",
    )
)
tokens, text = tokenized.tokens, tokenized.text
print(f"There are {len(tokens):,} tokens in the text '{input_text}'")
