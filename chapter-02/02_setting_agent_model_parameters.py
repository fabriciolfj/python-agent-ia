from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

# Load environment variables from .env file (expects ANTHROPIC_API_KEY)
load_dotenv()

# Agent Instructions
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""

agent = Agent(
    name="Research Planner", 
    instructions=instructions,
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),  # Claude Opus/Sonnet 5 force adaptive thinking, which requires temperature=1
    model_settings=ModelSettings(
        temperature=0.0,  # Set the temperature for repeatability
        max_tokens=150,  # Set the maximum number of tokens in the response
        # Claude does not support setting top_p together with temperature,
        # and does not support frequency_penalty / presence_penalty
    )
)

input = "learn about AI agents"

result = Runner.run_sync(
    agent, 
    input=input,
    )

print(result.final_output)
