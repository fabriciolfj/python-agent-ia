import asyncio
import base64
import os
import subprocess
import sys

from agents import Agent, ImageGenerationTool, Runner, trace
from agents.extensions.models.litellm_model import LitellmModel


def open_file(path: str) -> None:
    if sys.platform.startswith("darwin"):
        subprocess.run(["open", path], check=False)  # macOS
    elif os.name == "nt":  # Windows
        os.startfile(path)  # type: ignore
    elif os.name == "posix":
        subprocess.run(["xdg-open", path], check=False)  # Linux/Unix
    else:
        print(f"Não sei como abrir arquivos nesta plataforma: {sys.platform}")


async def main():
    agent = Agent(
        name="Image generator",
        model=LitellmModel(model="anthropic/claude-haiku-4-5"),
        instructions=""",
## Diretrizes de Estilo para Todas as Imagens:

- **Consistência**: Cada imagem deve manter a mesma qualidade fotográfica, com elementos renderizados em 3D integrados de forma natural
- **Paleta de Cores**: Use um esquema consistente de azuis, roxos, dourados quentes e verdes, com toques de cores vibrantes de destaque
- **Iluminação**: Iluminação de fotografia profissional, com tons dramáticos, porém quentes
- **Infográficos**: Projeções holográficas semitransparentes que não sobrecarreguem os elementos principais
- **Personagens**: Os robôs de IA devem ser fofos, acessíveis e claramente diferentes entre si, mantendo uma semelhança de família
- **Ícones**: Use símbolos universalmente reconhecíveis (lâmpadas para ideias, engrenagens para processamento, corações para alinhamento etc.)
- **Clima**: Otimista, educativo e levemente futurista, sem ser frio ou intimidador
- Responda sempre em português.
""",
        tools=[
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "high",
                    "model": "gpt-image-1",
                    "size": "1536x1024",
                }
            )
        ],
    )

    image_description = "um agente gerando uma imagem"
    image_name = "agent_image_generation"

    with trace("Geração de imagem"):
        result = await Runner.run(agent, image_description)
        print(result.final_output)
        for item in result.new_items:
            if (
                item.type == "tool_call_item"
                and item.raw_item.type == "image_generation_call"
                and (img_result := item.raw_item.result)
            ):
                os.makedirs("gen_images", exist_ok=True)
                image_path = os.path.join("gen_images", f"{image_name}.png")
                with open(image_path, "wb") as img_file:
                    img_file.write(base64.b64decode(img_result))
                open_file(image_path)


if __name__ == "__main__":
    asyncio.run(main())
