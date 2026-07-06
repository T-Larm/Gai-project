import json
import random
import uuid
import os
import sys

# Ensure we can import config relative to this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

# --- HELPER FUNCTIONS ---
def get_reply(action_type):
    candidates = config.replies_pool.get(action_type, ["..."])
    raw_reply = random.choice(candidates)
    if "{price}" in raw_reply:
        raw_reply = raw_reply.replace("{price}", random.choice(config.prices))
    return raw_reply

# --- NPC LOGIC ---
def create_npc_with_history():
    p = random.choice(list(config.personalities.keys()))
    r = random.choice(config.roles)
    
    npc = {
        "id": str(uuid.uuid4())[:8],
        "role": r,
        "personality": p,
        "stress": random.randint(0, 40), 
        "trust": random.randint(30, 70), 
        "memory": []
    }
    
    # 80% chance to have memory
    if random.random() < 0.8:
        # Role appropriate pool
        pool = config.role_memories.get(r, config.role_memories["General"])
        # Mix with General
        full_pool = list(set(pool + config.role_memories["General"]))
        
        # Select 1-3 random memories
        num_memories = random.randint(1, 3)
        selected_memories = random.sample(full_pool, min(len(full_pool), num_memories))
        npc["memory"].extend(selected_memories)
        
        for mem in selected_memories:
            # Update mood based on memory content
            if any(word in mem.lower() for word in ["sinirli", "kızdı", "kavga", "bozuk", "kötü", "öldü", "çaldı", "isyan", "tepti"]):
                npc["stress"] += 15
            if any(word in mem.lower() for word in ["güzel", "kazandım", "kurtardım", "seviyor", "eğlendim", "bahşiş"]):
                npc["trust"] += 10
            
    return npc

def interact(npc):
    pdata = config.personalities[npc["personality"]]
    trait = random.choice(pdata["traits"])
    scenario = random.choice(["TRADE", "COMBAT"])
    
    # Memory string
    current_memory = " ".join(npc["memory"]) if npc["memory"] else "Sıradan bir gün, aklımda özel bir şey yok."

    if scenario == "TRADE":
        item = random.choice(config.items)
        player = random.choice([
            f"Bu {item} satılık mı?",
            f"Şu {item} için ne kadar istiyorsun?",
            f"Tezgâhtaki {item} ilgimi çekti.",
            f"Bana o {item} lazım, fiyatı nedir?"
        ])
        
        if pdata["trade"] == "reject":
            action = "Trade_Reject"
            npc["stress"] += 5
        elif pdata["trade"] == "scam":
            action = "Trade_Scam"
            npc["trust"] -= 5
        elif pdata["trade"] == "accept_low":
            action = "Trade_Accept"
            npc["stress"] += 10
        else:
            action = "Trade_Fair"

    else: # COMBAT
        player = random.choice(["Yolumdan çekil!", "Paraları ver!", "Teslim ol!"])
        
        if pdata["combat"] == "attack":
            action = "Attack"
            npc["stress"] += 10
        elif pdata["combat"] == "flee":
            action = "Flee"
            npc["stress"] += 15
        elif pdata["combat"] == "trick":
            action = "Trick"
        elif pdata["combat"] == "bribe":
            action = "Bribe"
        else:
            action = "Defend"

    reply = get_reply(action)

    thought = (
        f"Durum: {current_memory} "
        f"Ruh Halim: Stres {npc['stress']}, Güven {npc['trust']}. "
        f"Kişiliğim '{npc['personality']}' olduğu için '{action}' eylemini seçiyorum."
    )

    return {
        "role": npc["role"],
        "trait": trait,
        "player": player,
        "thought": thought,
        "speech": reply,
        "action": action,
        "emotion": npc["personality"],
        "memory_str": current_memory 
    }

def generate_dataset(filename, count):
    data = []
    print(f"{filename} için {count} adet veri üretiliyor...")
    
    for _ in range(count):
        npc = create_npc_with_history() 
        result = interact(npc)

        # Build prompt using template
        text = config.SYSTEM_PROMPT_TEMPLATE.format(
            memory=result['memory_str'],
            trait=result['trait'],
            role=result['role'],
            player=result['player'],
            json_output=json.dumps({
                "thought": result["thought"],
                "speech": result["speech"],
                "action": result["action"],
                "emotion": result["emotion"]
            }, ensure_ascii=False)
        )

        data.append({"text": text})
    
    # Save as JSONL (or JSON Array, but user requested .jsonl and usage is dump per line)
    # The original was JSON lines: json.dump(d) + newline
    with open(filename, "w", encoding="utf-8") as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")
    print(f"[OK] {filename} tamamlandı.")

# --- RUN ---
if __name__ == "__main__":
    # Ensure data directory exists relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)

    print("Starting generation...")
    # Generate Training Data
    generate_dataset(os.path.join(data_dir, "train.jsonl"), 20000)
    
    # Generate Test Data
    generate_dataset(os.path.join(data_dir, "test.jsonl"), 2000)

    print("\nTüm işlemler bitti. Yeni datasetler hazır!")
