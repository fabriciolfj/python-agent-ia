from agents import Agent, Runner, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from litellm import max_tokens

# Load environment variables from .env file (expects ANTHROPIC_API_KEY)
load_dotenv()

# Agent Instructions
instructions = """
saiba mais sobre agentes de IA.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""


agent = Agent(
    name="Research Planner",
    instructions=instructions,
    model=LitellmModel(model="anthropic/claude-opus-5"),
    model_settings=ModelSettings(max_tokens=150, temperature=1),
)

input = "learn about AI agents"

result = Runner.run_sync(
    agent,
    input=input,
)

print(result.final_output)
