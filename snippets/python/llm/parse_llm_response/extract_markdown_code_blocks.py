"""
TAGS: ai|block|blocks|code|code-block|code-blocks|extract|json|language|llm|markdown|nlp|parse|response|xml
DESCRIPTION: Code for extracting markdown code blocks from an input string (e.g. for parsing a Large Language Model response for code)
"""

import json
import re
from collections import defaultdict

def extract_markdown_code_blocks(
        input_text: str
    ) -> dict:
    """TODO"""
    regex_pattern: str = r"```(\w+)\s*\n(.*?)\n```"
    matches = re.findall(regex_pattern, input_text, re.DOTALL)
    
    code_blocks = defaultdict(list)
    for language, code in matches:
        code_blocks[language].append(code.strip())
    
    return dict(code_blocks)

if __name__ == "__main__":
    input_text: str = """
Here is some JSON:

```json
[
    {
        "a": "thing"
    }
]
```

Here is some python code:

```python
print("hello")
print("world")
```

Here's another JSON:

```json
[69, 420]
```

Thank you for your time
""".strip()
    print("input_text:", "\n", input_text, "\n")
    extracted_code_blocks: dict = extract_markdown_code_blocks(input_text)
    print("extracted code blocks:")
    print(json.dumps(extracted_code_blocks, indent=4))
