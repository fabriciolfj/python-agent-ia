import asyncio
from pathlib import Path

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStdio

# ──────────────────────────────────────────────────────────────
# 1  “Blocos de memória” em JSON que queremos que o agente execute
# ──────────────────────────────────────────────────────────────
memories = [
    # ---------- ENTIDADES ------------------------------------------------
    {
        "name": "create_entities",
        "arguments": {
            "entities": [
                #  pessoas
                {"name": "Marty McFly", "entityType": "Person", "observations": []},
                {
                    "name": "Professor Emmett Brown",
                    "entityType": "Person",
                    "observations": [],
                },
                {"name": "George McFly", "entityType": "Person", "observations": []},
                {
                    "name": "Eileen McFly (née Baines)",
                    "entityType": "Person",
                    "observations": [],
                },
                {"name": "Suzy Parker", "entityType": "Person", "observations": []},
                {"name": "Biff Tannen", "entityType": "Person", "observations": []},
                {"name": "Mr. Arky", "entityType": "Person", "observations": []},
                {
                    "name": "Reginald Washington",
                    "entityType": "Person",
                    "observations": [],
                },
                {"name": "Dick Wilson", "entityType": "Person", "observations": []},
                {
                    "name": "Shemp (organ-grinder monkey)",
                    "entityType": "Person",
                    "observations": [],
                },
                #  lugares / item
                {
                    "name": "Orpheum Theatre Building",
                    "entityType": "Location",
                    "observations": [],
                },
                {"name": "Brown’s Lab", "entityType": "Location", "observations": []},
                {"name": "Wilson’s Café", "entityType": "Location", "observations": []},
                {
                    "name": "Hill Valley High School",
                    "entityType": "Location",
                    "observations": [],
                },
                {"name": "Power Converter", "entityType": "Item", "observations": []},
            ]
        },
    },
    # ---------- RELAÇÕES -------------------------------------------------
    {
        "name": "create_relations",
        "arguments": {
            "relations": [
                {
                    "from": "Marty McFly",
                    "to": "George McFly",
                    "relationType": "child_of",
                },
                {
                    "from": "Marty McFly",
                    "to": "Eileen McFly (née Baines)",
                    "relationType": "child_of",
                },
                {"from": "Marty McFly", "to": "Mr. Arky", "relationType": "student_of"},
                {
                    "from": "Marty McFly",
                    "to": "Professor Emmett Brown",
                    "relationType": "mentored_by",
                },
                {
                    "from": "Professor Emmett Brown",
                    "to": "Brown’s Lab",
                    "relationType": "owns_lab",
                },
                {
                    "from": "George McFly",
                    "to": "Eileen McFly (née Baines)",
                    "relationType": "married_to",
                },
                {
                    "from": "Biff Tannen",
                    "to": "George McFly",
                    "relationType": "bullies",
                },
                {
                    "from": "Reginald Washington",
                    "to": "Marty McFly",
                    "relationType": "represents",
                },
                {
                    "from": "Power Converter",
                    "to": "Professor Emmett Brown",
                    "relationType": "invented_by",
                },
                {
                    "from": "Brown’s Lab",
                    "to": "Orpheum Theatre Building",
                    "relationType": "located_in",
                },
            ]
        },
    },
    # ---------- OBSERVAÇÕES ----------------------------------------------
    {
        "name": "add_observations",
        "arguments": {
            "observations": [
                {
                    "entityName": "Power Converter",
                    "contents": [
                        "Marty derrama Coca-Cola sem querer no Power Converter de 30 anos, "
                        "causando um enorme pico de tensão que convence Brown de que a fórmula funciona."
                    ],
                },
                {
                    "entityName": "Hill Valley High School",
                    "contents": [
                        "Mr. Arky repreende Marty por demonstrar curiosidade sobre uma explosão atômica "
                        "enquanto Suzy passa um bilhete de paquera para Marty."
                    ],
                },
                {
                    "entityName": "George McFly",
                    "contents": [
                        "Biff invade a cozinha dos McFly e humilha George por causa de uma furadeira queimada "
                        "enquanto Eileen e Marty assistem."
                    ],
                },
            ]
        },
    },
]

# ──────────────────────────────────────────────────────────────
# 2  Instruções mínimas do agente
#    Toda mensagem do usuário *é* um payload JSON de ferramenta – basta repassá-lo.
# ──────────────────────────────────────────────────────────────
instructions = """
Você é um agente de povoamento de memória.
Toda mensagem do usuário já será um objeto JSON com as chaves `name` e `arguments`.

Passos:
1. Trate `name` como o nome exato da ferramenta MCP a ser chamada.
2. Chame essa ferramenta com `arguments` (encapsulado como `input_json=`).
3. **Não** converse nem explique – apenas retorne o resultado da ferramenta.
"""


# ──────────────────────────────────────────────────────────────
# 3  Runner
# ──────────────────────────────────────────────────────────────
async def main() -> None:
    memory_srv = MCPServerStdio(
        name="memory",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory@latest"],
            "env": {"MEMORY_FILE_PATH": str(Path(__file__).parent / "memory.jsonl")},
        },
    )

    async with memory_srv:
        agent = Agent(
            model=LitellmModel(model="anthropic/claude-haiku-4-5"),
            name="Agente de Povoamento de Memória",
            instructions=instructions,
            mcp_servers=[memory_srv],
        )

        for payload in memories:
            # Cada payload entra como mensagem do usuário (o Runner.run cuida do encapsulamento)
            resp = await Runner.run(agent, str(payload))
            print(resp)  # opcional: ver a confirmação do servidor


if __name__ == "__main__":
    asyncio.run(main())
