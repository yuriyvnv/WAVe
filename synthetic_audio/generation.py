"""
Gera 25 arquivos MP3 simultaneamente (TTS‑1, voz 'nova').
Requer: `pip install --upgrade openai`
Coloque sua chave na variável OPENAI_API_KEY (ou use variável de ambiente).
"""
import asyncio, os
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOICE = "alloy"          # outras vozes: alloy, echo, fable, onyx, shimmer
MODEL = "tts-1"         # modo “normal”

texts = [
    "A manhã amanheceu com brisa suave, anunciando um dia cheio de possibilidades.",
    "Lembre‑se de beber água e alongar o corpo antes de começar seu trabalho.",
    "O café na mesa ainda solta vapor, perfumando toda a cozinha.",
    "Escreva seus objetivos do dia e revise‑os quando o sol se pôr.",
    "A luz dourada do pôr do sol reflete nos prédios da cidade.",
    "A música suave ao fundo cria o ambiente perfeito para concentração.",
    "Cada pequeno passo aproxima você de um grande resultado.",
    "Respire fundo, conte até cinco e deixe a ansiedade escoar.",
    "O som das ondas lembra que tudo na vida é movimento.",
    "Compartilhe um sorriso hoje; ele pode mudar o dia de alguém.",
    "Abra a janela e deixe o vento trazer novas ideias.",
    "Organize sua mesa; um espaço limpo clareia a mente.",
    "Aprender algo novo expande horizontes e renova a criatividade.",
    "A noite estrelada convida a sonhar acordado e planejar o amanhã.",
    "Valorize cada encontro; toda pessoa traz uma história única.",
    "Uma pausa de cinco minutos pode render horas de produtividade.",
    "Ler um bom livro é viajar sem sair do lugar.",
    "A gratidão diária transforma desafios em oportunidades.",
    "Movimentar o corpo libera energia e melhora o humor.",
    "Faça hoje algo que seu futuro vai agradecer.", 
    "O silêncio também é música quando a mente está em paz.",
    "Plantar uma semente de bondade gera florestas de empatia.",
    "Pequenas vitórias merecem grande celebração interior.",
    "Confie no processo; a jornada molda o destino.",
    "Cada amanhecer traz a chance de recomeçar melhor.",
]

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def synth(index: int, text: str) -> None:
    """Gera um MP3 e salva em disco."""
    async with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=text,
        response_format="mp3",
    ) as response:
        path = f"audios/audio_{index:02d}.mp3"
        await response.stream_to_file(path)
    print(f"✔️  {path}")

async def main():
    # Limita a 5 requisições paralelas p/ evitar rate‑limit
    semaphore = asyncio.Semaphore(25)
    async def wrapped(i, t):
        async with semaphore:
            await synth(i, t)
    await asyncio.gather(*(wrapped(i, t) for i, t in enumerate(texts, 1)))

if __name__ == "__main__":
    asyncio.run(main())
