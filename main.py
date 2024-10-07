import os

import anthropic
from anthropic.types import (
    Message,
    ToolParam,
)
from lmnr import Laminar
from pydantic import BaseModel

assert (
    anthropic_api_key_env := os.getenv("ANTHROPIC_API_KEY")
) is not None, "ANTHROPIC_API_KEY is not set"
ANTHROPIC_CLIENT = anthropic.AsyncClient(
    api_key=anthropic_api_key_env,
)

# commenting this out makes it work
assert (lmnr_project_api_key_env := os.getenv("LMNR_PROJECT_API_KEY")) is not None
Laminar.initialize(project_api_key=lmnr_project_api_key_env)


class SemanticCheckOutput(BaseModel):
    number: int


async def test():
    messages = [
        {
            "role": "user",
            "content": "return a number",
        },
    ]

    tools = [
        ToolParam(
            name="number",
            description="return a number",
            input_schema=SemanticCheckOutput.model_json_schema(),
        )
    ]

    input = {
        "model": "claude-3-5-sonnet-20240620",
        "tools": tools,
        "messages": messages,
        "temperature": 0.0,
        "tool_choice": {"name": "number", "type": "tool"},
        "max_tokens": 4096,
    }

    response: Message = await ANTHROPIC_CLIENT.messages.create(**input)
    print(response)


import asyncio

if __name__ == "__main__":
    asyncio.run(test())
