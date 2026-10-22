import asyncio
import os

from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel, Field
from agents import Agent, Runner
from agents.mcp import MCPServerStdio


class SubTopic(BaseModel):
    name: str
    status: str = "pending"
    notes: str = ""


class ResearchPlan(BaseModel):
    sub_topics: list[SubTopic] = Field(default_factory=list)
    strategy_notes: str = ""

    def to_context(self) -> str:
        return str(dict(
            sub_topics=[
                dict(name=st.name, status=st.status, notes=st.notes)
                for st in self.sub_topics
            ],
            strategy_notes=self.strategy_notes,
        ))

    @property
    def progress_summary(self) -> str:
        if not self.sub_topics:
            return "Nenhum plano ainda"
        total = len(self.sub_topics)
        complete = sum(1 for st in self.sub_topics
                       if st.status == "complete")
        in_progress = sum(1 for st in self.sub_topics
                          if st.status == "in_progress")
        return (f"{complete}/{total} concluídos, "
                f"{in_progress} em andamento")


class ResearchState(BaseModel):
    goal: str = ""
    findings: list[str] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    plan: ResearchPlan = Field(default_factory=ResearchPlan)
    iteration_count: int = 0
    max_iterations: int = 10
    status: str = "in_progress"

    @property
    def should_continue(self) -> bool:
        return (
            self.status == "in_progress"
            and self.iteration_count < self.max_iterations
            and len(self.follow_up_questions) > 0
        )

    def to_context(self) -> str:
        return str(dict(
            goal=self.goal,
            findings=self.findings[-5:],
            sources=self.sources_consulted,
            pending_questions=self.follow_up_questions,
            plan=self.plan.to_context(),
            iteration=self.iteration_count,
            remaining=self.max_iterations - self.iteration_count,
        ))


class SubTopicUpdate(BaseModel):
    name: str
    status: str
    notes: str = ""


class ResearchIteration(BaseModel):
    summary_of_findings: str
    sources_used: list[str]
    follow_up_questions: list[str]
    goal_satisfied: bool
    confidence: float
    reasoning: str
    plan_updates: list[SubTopicUpdate] = []
    new_sub_topics: list[SubTopicUpdate] = []
    strategy_notes: str = ""


research_agent = Agent(
    name="Agente de Pesquisa Profunda",
    instructions="""
Você é um agente de pesquisa profunda.
Seu objetivo é pesquisar um tema a fundo, buscando
informações, lendo os resultados e sintetizando as descobertas.

PLANO DE PESQUISA:
- Na sua PRIMEIRA iteração (quando ainda não existe plano), analise o
  objetivo da pesquisa e crie um plano retornando subtópicos em
  new_sub_topics. Cada subtópico deve ser um aspecto distinto do
  objetivo. Defina o status como "pending" para tópicos não iniciados e
  "in_progress" para aquele que você está pesquisando no momento.
- Nas iterações SEGUINTES, atualize os subtópicos existentes via
  plan_updates (altere o status para "in_progress" ou "complete" e
  adicione notas resumindo o que foi aprendido). Você também pode
  adicionar novos subtópicos via new_sub_topics se descobrir que a
  pesquisa exige áreas que não estavam no plano original.
- Use strategy_notes para registrar observações de alto nível sobre
  sua abordagem de pesquisa ou ajustes de estratégia.

A cada iteração, revise seu plano e o estado acumulado, identifique
lacunas, busque novas informações e atualize suas descobertas.
follow_up_questions deve conter consultas de busca específicas para
as próximas iterações. O plano fornece a direção estratégica, enquanto
follow_up_questions conduz a execução tática.

Quando acreditar que tem informações suficientes para responder
ao objetivo da pesquisa de forma abrangente, defina goal_satisfied como true.

Responda sempre em português.
""",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    output_type=ResearchIteration,
)


def apply_plan_updates(
    plan: ResearchPlan, iteration: ResearchIteration
) -> None:
    """Aplica as mudanças de plano de uma iteração ao plano mutável."""
    existing = {st.name: st for st in plan.sub_topics}
    for update in iteration.plan_updates:
        if update.name in existing:
            existing[update.name].status = update.status
            if update.notes:
                existing[update.name].notes = update.notes

    for new_st in iteration.new_sub_topics:
        if new_st.name not in existing:
            plan.sub_topics.append(
                SubTopic(
                    name=new_st.name,
                    status=new_st.status,
                    notes=new_st.notes,
                )
            )

    if iteration.strategy_notes:
        plan.strategy_notes = iteration.strategy_notes


async def run_research_loop(
    goal: str, max_iterations: int = 10
) -> ResearchState:
    search_server = MCPServerStdio(
        name="Brave Search",
        params={
            "command": "npx",
            "args": ["-y", "@anthropic/brave-search-mcp"]
        },
    )
    async with search_server:
        agent = research_agent.clone(
            mcp_servers=[search_server]
        )
        state = ResearchState(
            goal=goal,
            max_iterations=max_iterations,
            follow_up_questions=[goal],
        )

        while state.should_continue:
            state.iteration_count += 1
            question = state.follow_up_questions.pop(0)
            input_context = dict(
                current_question=question,
                accumulated_state=state.to_context(),
                research_plan=state.plan.to_context(),
            )
            print(f"\n--- Iteração {state.iteration_count} ---")
            print(f"Investigando: {question}")

            result = await Runner.run(
                agent, input=str(input_context)
            )
            iteration = result.final_output

            state.findings.append(iteration.summary_of_findings)
            state.sources_consulted.extend(iteration.sources_used)
            state.follow_up_questions.extend(
                iteration.follow_up_questions
            )
            apply_plan_updates(state.plan, iteration)

            if iteration.goal_satisfied:
                state.status = "complete"
                print(f"Objetivo atingido (confiança: "
                      f"{iteration.confidence:.0%})")

            print(f"Encontrado: {iteration.summary_of_findings[:100]}...")
            print(f"Novas perguntas: {len(iteration.follow_up_questions)}")
            print(f"Plano: {state.plan.progress_summary}")

        return state


async def main():
    state = await run_research_loop(
        goal="Quais são os avanços mais recentes em baterias de estado "
             "sólido e quais empresas estão liderando a comercialização?",
        max_iterations=8,
    )
    print(f"\n{'='*50}")
    print(f"Pesquisa concluída após {state.iteration_count} iterações")
    print(f"Fontes consultadas: {len(state.sources_consulted)}")
    print(f"Status: {state.status}")
    print(f"Progresso do plano: {state.plan.progress_summary}")
    if state.plan.sub_topics:
        for st in state.plan.sub_topics:
            print(f"  [{st.status}] {st.name}")
    for i, finding in enumerate(state.findings, 1):
        print(f"\nDescoberta {i}: {finding[:200]}...")


asyncio.run(main())
