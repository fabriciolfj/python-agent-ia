import asyncio

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

# Agente para gerar os pensamentos do próximo passo
generator = Agent(
    name="ToT-Generator",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    instructions="Dada a situação atual, faça um brainstorm de um possível próximo passo ou ação para alcançar o objetivo.",
)
# Agente para avaliar soluções parciais
evaluator = Agent(
    name="ToT-Evaluator",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    instructions="Avalie a probabilidade de o plano proposto resolver o problema. Responda com 'promissor' ou 'improvável'.",
)

problem = """
Você precisa chegar ao ano de 1800 a partir de 2025 usando uma máquina do tempo
que pode saltar -100 ou -30 anos.
"""


async def main():
    # Gera os pensamentos candidatos iniciais
    initial_thoughts = []
    for i in range(3):
        resp = await Runner.run(
            generator, input=f"Problema: {problem}\nPense em um primeiro passo."
        )
        initial_thoughts.append(resp.final_output.strip())

    # Avalia e expande cada pensamento (uma iteração de expansão BFS)
    promising_branches = []
    for thought in initial_thoughts:
        eval_resp = await Runner.run(
            evaluator, input=f"Plano: {thought}\nIsso é promissor?"
        )
        if "promissor" in eval_resp.final_output.lower():
            # Expande este pensamento com um segundo passo
            next_step = await Runner.run(
                generator, input=f"Ideia atual: {thought}\nPróximo passo?"
            )
            promising_branches.append(f"{thought} -> {next_step.final_output.strip()}")

    print("Pensamentos candidatos iniciais:", initial_thoughts)
    print(
        "Ramo promissor expandido:",
        promising_branches[0] if promising_branches else "Nenhum",
    )


if __name__ == "__main__":
    asyncio.run(main())
