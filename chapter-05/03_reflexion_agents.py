import asyncio

from agents import Agent, RunContextWrapper, Runner
from agents.extensions.models.litellm_model import LitellmModel


def get_reflexion_solver_instructions(
    run_context: RunContextWrapper[str], agent: Agent[str]
) -> str:
    """Gera as instruções para o agente solucionador com reflexão."""
    instructions = (
        "Você é um especialista em viagem no tempo. Resolva o problema passo a passo "
        "e tome cuidado para evitar erros."
    )
    return instructions + "\nDICA:\n" + run_context.context


# --- Agentes base -------------------------------------------------------------
solver = Agent(
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    name="TimeTravelerReflexion", instructions=get_reflexion_solver_instructions
)
critic = Agent(
    name="TimeTravelCritic",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    instructions=(
        "Você é um tutor especialista. Se a solução estiver errada, explique o erro "
        "e dê uma dica concisa para melhorar."
    ),
)

# --- Especificação do problema ------------------------------------------------

problem = """
Em um filme de ficção científica, Alex é um viajante do tempo que decide voltar no tempo
para presenciar um famoso evento histórico que aconteceu há 100 anos
e que durou 10 dias. Ele chega três dias antes do início do evento.
Porém, depois de passar seis dias no passado, ele salta para o futuro
50 anos e fica lá por 20 dias. Então, ele viaja de volta para
presenciar o fim do evento.
Quantos dias Alex passa no passado antes de ver o fim do evento?
"""
TARGET_DAYS = "26"  # resposta final esperada
MAX_ATTEMPTS = 5  # limite de segurança para as tentativas


async def main():
    # --- Loop de reflexão ---------------------------------------------------------
    feedback_hint = ""
    for attempt_no in range(1, MAX_ATTEMPTS + 1):
        # Executa o solucionador
        result = await Runner.run(solver, input=problem, context=feedback_hint)
        answer = result.final_output.strip()
        print(f"\nTentativa {attempt_no}:\n{answer}")

        # --- Verificação simples de correção -------------------------------------
        has_correct_days = TARGET_DAYS in answer
        says_claim_correct = "sim" in answer.lower() or "corret" in answer.lower()
        solved = has_correct_days and says_claim_correct

        if solved:
            print("✅ Solução aceita.")
            break

        # --- Não resolvido: gera feedback e tenta novamente ----------------------
        feedback_prompt = (
            f"Solução dada:\n{answer}\n\n"
            f"Dias finais esperados: {TARGET_DAYS}\n"
            "Explique o erro brevemente e dê uma dica útil."
        )
        feedback_resp = await Runner.run(critic, input=feedback_prompt)
        hint = feedback_resp.final_output.strip()

        print(f"Feedback:\n{hint}")
        feedback_hint = hint
    else:
        print("\n⚠️  Número máximo de tentativas atingido sem uma solução correta.")


if __name__ == "__main__":
    asyncio.run(main())
