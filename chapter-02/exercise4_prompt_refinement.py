from typing import TypedDict

from agents import Agent, Runner, ModelSettings, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file (expects ANTHROPIC_API_KEY)
load_dotenv()

class Task(TypedDict):
    id: int
    description: str

class ResearchPlanModel(BaseModel):
    tasks: list[Task]


# Agent Instructions
instructions = """
saiba mais sobre agentes de IA.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""

@function_tool
def get_resource_sources() -> list[str]:
    sources = ["google", "yahoo"]
    return sources


agent = Agent(
    name="Research Planner",
    instructions=instructions,
    model=LitellmModel(model="anthropic/claude-opus-5"),
    model_settings=ModelSettings(max_tokens=600, temperature=1),
    output_type=ResearchPlanModel,
    tools=[get_resource_sources])

input = "learn about AI agents"

result = Runner.run_sync(
    agent,
    input=input,
)

print(result.final_output)
