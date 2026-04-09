import os
import asyncio
from dotenv import load_dotenv

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel
)

from agents.tracing import set_tracing_disabled  # ✅ FIXED

# Disable tracing
set_tracing_disabled(True)

# Load environment variables
load_dotenv()

# Model setup
model = OpenAIChatCompletionsModel(
    model=os.environ.get("MODEL_NAME"),
    openai_client=AsyncOpenAI(
        api_key=os.environ.get("MODEL_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
)

# Agent
history_agent = Agent(
    model=model,
    name="History Tutor",
    instructions="You are an amazing history tutor."
)

# Main async function
async def main():
    query = "Who is the president of the United States?"
    result = await Runner.run(history_agent, query)
    print(result.final_output)
if __name__ == "__main__":
    asyncio.run(main())