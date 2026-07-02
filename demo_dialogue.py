"""Quick smoke test: load Aldric persona and run 3 dialogue turns."""
from backend.llm.dialogue import DialogueHandler
from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.generator import PersonaGenerator

llm = OllamaClient()
npc = PersonaGenerator.load("data/personas/aldric.json")
handler = DialogueHandler(llm, npc)

print(f"\nNPC: {npc.core.name} ({npc.core.occupation})")
print(f"Speech style: {npc.core.speech_style}")
print(f"Goal: {npc.dynamic.current_goal}")
print("-" * 50)

test_inputs = [
    "Good day! Can you make me a sword?",
    "What do you think of Lord Vane?",
    "Do you know the healer Mira?",
]

for player_input in test_inputs:
    print(f"\nPlayer: {player_input}")
    reply = handler.respond(player_input)
    print(f"Aldric: {reply}")
