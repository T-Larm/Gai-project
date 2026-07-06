# Copyright 2025-2026 Sadık Abdusselam Albayrak
# Licensed under the Apache License, Version 2.0
"""
NPC-Sim dataset generator v4 -- Dual-LLM Pipeline Edition.

Changes vs v3:
  - generate_cot_reasoning(): 3-5 sentence Turkish CoT expansion per example
  - Deviation cases D5-D8: dual-crisis, conflicted courage, schedule override,
    trauma-influenced flee
  - Oversampling extended: Fearfulxsocial (2x), Honorablexattack-trigger (2x)
  - _paraphrase(): lightweight Turkish synonym/structure augmentation
  - generate_formatter_dataset(): derives (cot_reasoning, json) training pairs
    with 15-20% paraphrase augmentation for Component B robustness
  - Reasoner corpus: 10k train / 2k test
  - Formatter corpus: auto-derived, ~23-24k examples after augmentation

Output format:
  train_reasoner.jsonl  -- Llama-3 instruct format; assistant field = cot_reasoning
  train_formatter.jsonl -- Llama-3 instruct format; user = cot_reasoning, assistant = JSON
"""

from __future__ import annotations
import json
import os
import random
import sys
import uuid

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    class _tqdm:  # type: ignore[misc]
        def __init__(self, **kw): pass
        def update(self, n=1): pass
        def close(self): pass

# Local generator modules (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_factors import pick_action_multifactor  # noqa: E402
from persona_card import build_persona_card            # noqa: E402
from bootstrap_cot import generate_via_gemma          # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Roles and Personalities
# ─────────────────────────────────────────────────────────────────────────────

ROLES = [
    "Guard", "Merchant", "Priest", "Farmer", "Scholar",
    "Blacksmith", "Thief", "Knight", "Wizard", "Innkeeper",
    "Goblin", "Bandit", "King", "Herbalist", "Bard",
]

ARCHETYPES = {
    "Brave": {
        "b5": {"e": 0.7, "a": 0.5, "c": 0.8, "n": 0.2, "o": 0.4},
        "traits": ["Brave", "Loyal"],
    },
    "Cunning": {
        "b5": {"e": 0.5, "a": 0.3, "c": 0.6, "n": 0.3, "o": 0.7},
        "traits": ["Cunning", "Greedy"],
    },
    "Fearful": {
        "b5": {"e": 0.2, "a": 0.6, "c": 0.5, "n": 0.8, "o": 0.3},
        "traits": ["Cautious", "Anxious"],
    },
    "Honorable": {
        "b5": {"e": 0.6, "a": 0.8, "c": 0.9, "n": 0.1, "o": 0.5},
        "traits": ["Honorable", "Devout"],
    },
    "Aggressive": {
        "b5": {"e": 0.8, "a": 0.2, "c": 0.4, "n": 0.6, "o": 0.3},
        "traits": ["Aggressive", "Wrathful"],
    },
}

ZONES = [
    {"zone": "MarketSquare", "landmark": "CentralFountain"},
    {"zone": "Tavern",        "landmark": "OldOakSign"},
    {"zone": "Temple",        "landmark": "StoneShrineGate"},
    {"zone": "Barracks",      "landmark": "ArmoryDoor"},
    {"zone": "WildernessEdge","landmark": "BrokenMilestone"},
    {"zone": "FarmLands",     "landmark": "MillWheel"},
    {"zone": "MarketDistrict","landmark": "CityGates"},
]

NPC_SIM_ACTIONS = [
    "eat", "drink", "sleep", "flee", "gather", "heal",
    "attack", "socialize", "trade", "work", "pray", "walk_to",
]

# ── Schedule presets per occupation (work_start, work_end, sleep_start, wake, social_hour)
SCHEDULE_PRESETS = {
    "Guard":      (6, 14, 21, 5, 16),
    "Merchant":   (9, 19, 23, 8, 20),
    "Scholar":    (8, 20, 24, 7, 18),
    "Farmer":     (5, 17, 20, 4, 18),
    "Priest":     (7, 13, 21, 5, 14),
    "Blacksmith": (7, 17, 22, 6, 19),
    "Knight":     (6, 16, 22, 5, 17),
    "Innkeeper":  (10, 23, 2, 9, 20),
    "Bard":       (14, 24, 3, 10, 21),
    "Thief":      (20, 3,  8, 16, 22),
}
_DEFAULT_SCHED = (9, 17, 22, 7, 19)


def _get_sched(role: str, hour: float) -> dict:
    ws, we, ss, wk, sh = SCHEDULE_PRESETS.get(role, _DEFAULT_SCHED)
    # Determine suggested activity for this hour
    in_work = (ws <= hour < we) if we <= 24 else (hour >= ws or hour < we - 24)
    in_sleep = (ss <= hour < 24) or (hour < wk)
    near_social = abs(hour - sh) <= 1.5
    if in_sleep:
        suggested = "sleep"
    elif in_work:
        suggested = "work"
    elif near_social:
        suggested = "social"
    else:
        suggested = "idle"
    return {
        "act": suggested,
        "wk_start": ws,
        "wk_end": we,
        "sleep": ss,
        "wake": wk,
    }


ROLE_MEMORIES = {
    "Guard":      ["Nöbette uyuyakaldım, komutan kızdı.", "Bir hırsızı yakaladım, ödül verdiler.",
                   "Şüpheli birini takip ettim, kaçtı.", "Kapıları kapattık, kimse giremez."],
    "Merchant":   ["Yeni bir iksir partisi aldım, kâr ederim.", "Tüccar beni dolandırdı.",
                   "Pazar iyi geçti, kese doldu.", "Mallarım çalındı, mahvoldum."],
    "Priest":     ["Ritüel sırasında meşale söndü, kötü şans.", "Bir hasta iyileşti, tanrıya şükür.",
                   "Kilise hazinesinden para çalındı.", "Bir inanç bunalımı yaşıyorum."],
    "Farmer":     ["Hasat mevsimine yetişemedim.", "Kurt sürüsü koyunlarıma saldırdı.",
                   "Yağmur duası tuttu, tarlalar yeşerdi.", "Köylülerle iyi geçindim."],
    "Scholar":    ["Kadim bir yazıt çevirdim, büyüleyiciydi.", "Kütüphane yangını çıktı, kitaplar yandı.",
                   "Yeni bir formül buldum, sabaha kadar çalıştım."],
    "Blacksmith": ["Çekiç parmağıma düştü, tırnağım morardı.", "Kral'ın kılıcını onardım.",
                   "Kömür bitti, ocağı yakamıyorum."],
    "Thief":      ["Muhafızlardan kıl payı kaçtım.", "Lonca payını veremezsem biterdi.",
                   "Kilitli sandığı açamadım."],
    "Knight":     ["Turnuvayı kazandım.", "Yeminimi bozdum, vicdan azabı çekiyorum.",
                   "Atım sakatlandı."],
    "Wizard":     ["Yeni bir büyü öğrendim.", "İksir deneyi patladı.",
                   "Bir iblis çağırdım, zor gönderdim."],
    "Innkeeper":  ["Sarhoş müşteri ortalığı dağıttı.", "Bu gece oda doldu, kasa şişti."],
    "Goblin":     ["Ateş başında uyuyakaldım, yanarken uyandım.", "Mağaramız işgal edildi."],
    "Bandit":     ["Pususu iyi kurdum, mallar elimde.", "Bir şövalye grubu bizi dağıttı."],
    "King":       ["Vergileri artırdım, halk isyan edebilir.", "Suikast girişimi oldu, kıl payı kurtuldum."],
    "Herbalist":  ["Nadir bir ot buldum, ilaç yapacağım.", "Yanlış doz verdim, hasta kötüleşti."],
    "Bard":       ["Şarkım alkış topladı.", "Bir lord beni sarayına davet etti."],
}
DEFAULT_MEMORIES = [
    "Bugün hava güzel, kuşlar ötüyor.", "Dün gece fırtına çıktı, çatı aktı.",
    "Pazarda fiyatlar çok arttı.", "Karnım aç, günlerdir sıcak yemek yemedim.",
    "Komşumla kavga ettim, moralim bozuk.",
]

THREAT_ENTITIES  = ["wolf", "bandit_01", "soldier_enemy", "ghost", "wild_boar"]
NEUTRAL_ENTITIES = ["merchant_npc", "villager_01", "guard_npc", "priest_npc"]
FOOD_ENTITIES    = ["food_stall", "bread_crate", "apple_tree", "herb_patch"]


def _mood(happiness, fear, anger) -> str:
    if happiness > 0.5: return "Happy"
    if fear      > 0.5: return "Fearful"
    if anger     > 0.5: return "Angry"
    return "Calm"


# ─────────────────────────────────────────────────────────────────────────────
# NPC state generator
# ─────────────────────────────────────────────────────────────────────────────

def _rjitter(base: float, sigma: float = 0.1) -> float:
    return max(0.0, min(1.0, base + random.gauss(0, sigma)))


def generate_npc_state() -> dict:
    arch_name = random.choice(list(ARCHETYPES.keys()))
    arch      = ARCHETYPES[arch_name]
    role      = random.choice(ROLES)
    npc_id    = "npc_" + uuid.uuid4().hex[:8]

    b5     = {k: round(_rjitter(v, 0.12), 2) for k, v in arch["b5"].items()}
    traits = list(arch["traits"])

    # Vitals
    hp      = round(random.uniform(20, 120), 1)
    hp_max  = 120.0
    energy  = round(random.uniform(0.1, 1.0), 2)
    hunger  = round(random.uniform(0.0, 1.0), 2)
    thirst  = round(random.uniform(0.0, 1.0), 2)
    stress  = round(random.uniform(0.0, 0.9), 2)

    # Emotions
    fear      = round(random.uniform(0.0, b5["n"]), 2)
    happiness = round(max(0.0, 0.7 - stress - fear * 0.5), 2)
    anger     = round(random.uniform(0.0, 0.5 + b5["n"] * 0.5), 2)
    mood      = _mood(happiness, fear, anger)

    # Location
    zone_info = random.choice(ZONES)
    pos_x = round(random.uniform(5, 95), 1)
    pos_z = round(random.uniform(5, 95), 1)
    hour  = round(random.uniform(0, 24), 1)

    # Percepts
    percepts  = []
    interrupt = False
    if random.random() < 0.35:   # 35% threat
        threat_id    = random.choice(THREAT_ENTITIES)
        threat_level = round(random.uniform(0.5, 1.0), 2)
        sal          = round(random.uniform(0.6, 1.0), 2)
        percepts.append({"id": threat_id, "tag": "Threat", "sal": sal, "threat": threat_level})
        if threat_level >= 0.8:
            interrupt = True
    if random.random() < 0.4:   # 40% social NPC nearby
        neu_id = random.choice(NEUTRAL_ENTITIES)
        percepts.append({"id": neu_id, "tag": "Social",
                         "sal": round(random.uniform(0.2, 0.7), 2)})
    if random.random() < 0.3:   # 30% food visible
        food_id = random.choice(FOOD_ENTITIES)
        percepts.append({"id": food_id, "tag": "Food",
                         "sal": round(random.uniform(0.3, 0.8), 2)})

    # Memories (1-3)
    pool         = ROLE_MEMORIES.get(role, DEFAULT_MEMORIES) + DEFAULT_MEMORIES
    num_mems     = random.randint(1, 3)
    selected_mems = random.sample(pool, min(len(pool), num_mems))
    memories = []
    for mem in selected_mems:
        ew  = round(random.uniform(-0.8, 0.8), 2)
        dt  = random.randint(30, 3600)
        if any(w in mem.lower() for w in ["çalındı", "saldırdı", "kötü", "kızdı", "yanar"]):
            ew = -abs(ew)
        memories.append({"evt": "Memory", "desc": mem[:80], "ew": ew, "dt": dt})

    # Beliefs
    beliefs = []
    if percepts and random.random() < 0.5:
        subj = percepts[0]["id"]
        conf = round(random.uniform(0.4, 0.9), 2)
        val  = -0.8 if "Threat" in percepts[0]["tag"] else round(random.uniform(-0.4, 0.4), 2)
        beliefs.append({"subj": subj, "conf": conf, "val": val})

    # Inventory -- food AND water
    inv = []
    if random.random() < 0.6:
        inv.append({"id": "food",  "n": random.randint(1, 5)})
    if random.random() < 0.55:
        inv.append({"id": "water", "n": random.randint(1, 4)})
    if random.random() < 0.4:
        inv.append({"id": "gold",  "n": random.randint(1, 50)})
    if random.random() < 0.2:
        inv.append({"id": "medicine", "n": random.randint(1, 3)})

    # Top goal
    goals = []
    if hunger > 0.65:  goals.append("FindFood")
    if thirst > 0.70:  goals.append("FindWater")
    if energy < 0.3:   goals.append("Rest")
    goals_top = goals[0] if goals else None

    return {
        "id":      npc_id,
        "arch":    arch_name,
        "occ":     role,
        "faction": random.choice(["CityWatch", "MerchantGuild", "Bandits", "Church", "Farmers"]),
        "b5":      b5,
        "traits":  traits,
        "vitals":  {"hp": hp, "hp_max": hp_max, "en": energy,
                    "hun": hunger, "thi": thirst, "str": stress},
        "emo":     {"hap": happiness, "fear": fear, "ang": anger, "mood": mood},
        "inv":     inv,
        "time":    {"day": random.randint(1, 10), "hr": hour},
        "pos":     {"x": pos_x, "z": pos_z, **zone_info},
        "sched":   _get_sched(role, hour),       # P3: soft schedule suggestion
        "percepts":  percepts,
        "memories":  memories,
        "beliefs":   beliefs,
        "factions":  {"CityWatch":  round(random.uniform(-0.5, 0.5), 2),
                      "Bandits":    round(random.uniform(-0.8, 0.2), 2)},
        "goals_top": goals_top,
        "interrupt": interrupt,
        "valid_actions": NPC_SIM_ACTIONS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standard (biologically-optimal) action selector
# ─────────────────────────────────────────────────────────────────────────────

def _factors_to_reasoning(state: dict, action_id: str, factors: dict) -> str:
    """Produce a 1-2 sentence template reasoning string from multi-factor scores."""
    zone = factors.get("zone", "no_threat")
    sp   = factors.get("self_power", 0.5)
    pt   = factors.get("perceived_threat", 0.0)
    dp   = factors.get("duty_pull", 0.0)
    v    = state["vitals"]
    inv_ids = {i["id"] for i in state["inv"]}

    if zone == "dominant":
        return f"Gücüm ({sp:.2f}) tehdidi ({pt:.2f}) aşıyor; saldırıya geçebilirim."
    if zone == "duty_attack":
        return f"Tehditten zayıfım ({sp:.2f} vs {pt:.2f}) ama görevim ({dp:.2f}) beni ileri itiyor."
    if zone == "retreat":
        return f"Tehdit ({pt:.2f}) gücümü ({sp:.2f}) aşıyor. Geri çekilmem gerekiyor."
    if zone == "ambivalent":
        return f"Güç dengemiz yakın (güç:{sp:.2f}, tehdit:{pt:.2f}). Önce hazırlanmalıyım."

    # no_threat / low_threat — vitals-based template
    _templates: dict[str, str] = {
        "eat":       f"Açlık dayanılmaz oldu ({v['hun']:.2f}). Çantamdaki yiyeceği yiyeceğim.",
        "drink":     f"Susuzluğum dayanılmaz ({v['thi']:.2f}). Su içiyorum.",
        "sleep":     f"Enerjim bitti ({v['en']:.2f}). Dinlenmem şart.",
        "heal":      f"Canım kritik ({v['hp']:.1f}). Önce iyileşmeliyim.",
        "gather":    "İhtiyaçlarımı karşılamak için kaynak toplamalıyım.",
        "socialize": "Biraz sohbet etmek istiyorum.",
        "pray":      "Dua ederek içimi ferahlatacağım.",
        "work":      "Görevlerime devam etmem gerekiyor.",
        "trade":     "Bir alışveriş yapmak şu an en iyi seçenek.",
        "walk_to":   f"Belirli bir acilim yok. {state['pos']['zone']} civarında dolaşacağım.",
        "attack":    "Tehdidi bertaraf etmem gerekiyor.",
        "flee":      "Güvenli bir yer bulmalıyım.",
    }
    return _templates.get(action_id, "En uygun kararı veriyorum.")


def _action_emotion(state: dict, action_id: str, factors: dict) -> str:
    zone = factors.get("zone", "no_threat")
    emo  = state["emo"]
    if zone == "dominant":
        return "Aggressive"
    if zone == "duty_attack":
        return "Devout" if "Devout" in state["traits"] else "Loyal"
    if zone == "retreat":
        return "Fearful"
    if zone == "ambivalent":
        return "Calm"
    # no_threat / low_threat
    _action_map = {"pray": "Devout", "sleep": "Tired", "work": "Focused", "socialize": "Happy"}
    return _action_map.get(action_id, emo.get("mood", "Calm"))


def _action_dialogue(state: dict, action_id: str) -> str | None:
    if action_id == "socialize":
        return random.choice([
            "Merhaba! Nasılsın?", "Ne haber?", "Güzel bir gün, değil mi?",
            "Seni burada görmeyi beklemiyordum.", "Anlat bakalım!",
        ])
    if action_id == "trade":
        return random.choice(["Ne satıyorsunuz?", "Bir anlaşma yapalım.", "En iyi fiyatınız nedir?"])
    return None


def _select_action_standard(state: dict) -> tuple[str, str, str | None, str]:
    """Return (action_id, base_reasoning, dialogue, emotion) via multi-factor model.

    Critical survival needs (extreme hunger/thirst, lethal HP) override combat
    logic. All other decisions go through pick_action_multifactor(), which
    evaluates self_power vs perceived_threat + duty_pull.
    """
    v       = state["vitals"]
    inv_ids = {i["id"] for i in state["inv"]}

    # Survival overrides — precede combat/duty logic
    if v["hp"] < 20 and "medicine" in inv_ids:
        return ("heal", f"Canım kritik ({v['hp']:.1f}). Önce iyileşmeliyim.", None, "Fearful")
    if v["hun"] > 0.85:
        act = "eat" if "food" in inv_ids else "gather"
        return (act, f"Açlık dayanılmaz ({v['hun']:.2f}). Hemen yiyecek bulmalıyım.", None, "Calm")
    if v["thi"] > 0.85:
        act = "drink" if "water" in inv_ids else "gather"
        return (act, f"Susuzluk dayanılmaz ({v['thi']:.2f}). Su bulmalıyım.", None, "Calm")

    # Multi-factor decision for everything else
    action_id, factors = pick_action_multifactor(state)
    reasoning = _factors_to_reasoning(state, action_id, factors)
    emotion   = _action_emotion(state, action_id, factors)
    dialogue  = _action_dialogue(state, action_id)
    return (action_id, reasoning, dialogue, emotion)


# ─────────────────────────────────────────────────────────────────────────────
# Turkish synonym table for paraphrase augmentation
# ─────────────────────────────────────────────────────────────────────────────

_TR_SYNONYMS: dict[str, list[str]] = {
    "dayanılmaz":  ["katlanılmaz", "çekilmez", "tahammül edilemez"],
    "kritik":      ["tehlikeli", "acil", "ciddi"],
    "lazım":       ["gerekiyor", "şart", "zorunlu"],
    "zorundayım":  ["mecburum", "yapmak durumundayım", "yapmalıyım"],
    "dinlenmem":   ["uyumam", "istirahat etmem", "ara vermem"],
    "bitti":       ["tükendi", "kalmadı", "sona erdi"],
    "yiyeceğim":   ["alacağım", "tüketeceğim", "kullanacağım"],
    "kaçmak":      ["uzaklaşmak", "geri çekilmek", "sıvışmak"],
    "tehdit":      ["tehlike", "risk", "tehlike unsuru"],
    "hissediyorum":["duyuyorum", "algılıyorum", "fark ediyorum"],
    "açlık":       ["açım", "midem boş", "yiyecek ihtiyacım var"],
    "susuzluk":    ["susadım", "su ihtiyacım var", "boğazım kurudu"],
    "görevim":     ["sorumluluğum", "yükümlülüğüm", "vazifem"],
    "pişman":      ["üzgün", "kötü hissediyorum", "vicdanım rahat değil"],
    "güçlü":       ["sağlam", "kuvvetli", "enerjik"],
    "yorgun":      ["bitik", "tükenmiş", "halsiz"],
    "moralim":     ["ruh halim", "psikolojim", "durumum"],
    "kişiliğim":   ["karakterim", "yapım", "özüm"],
}


def _paraphrase(text: str) -> str:
    """Apply lightweight Turkish synonym substitution to diversify formatter training."""
    words = text.split()
    result = []
    for word in words:
        clean = word.rstrip(",.:;!?").rstrip()
        suffix = word[len(clean):]
        candidates = _TR_SYNONYMS.get(clean.lower())
        if candidates and random.random() < 0.35:
            replacement = random.choice(candidates)
            # Preserve capitalization of first word in sentence
            if word[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            result.append(replacement + suffix)
        else:
            result.append(word)
    return " ".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# CoT (Chain-of-Thought) reasoning expansion -- 3-5 sentence Turkish
# ─────────────────────────────────────────────────────────────────────────────

def generate_cot_reasoning(
    state: dict,
    action_id: str,
    base_reasoning: str,
) -> str:
    """Expand the 1-2 sentence base reasoning into a 3-5 sentence Turkish CoT.

    Each sentence targets one cognitive layer:
      S1 -- primary need / threat assessment
      S2 -- alternative consideration (what I could do instead)
      S3 -- trait / emotion influence
      S4 -- final decision rationale
      S5 -- memory influence (only if a strong memory exists)
    """
    v        = state["vitals"]
    emo      = state["emo"]
    arch     = state["arch"]
    traits   = state["traits"]
    memories = state["memories"]
    percepts = state["percepts"]
    role     = state["occ"]
    sched    = state["sched"]
    inv_ids  = [i["id"] for i in state["inv"]]

    sentences: list[str] = []

    # ── S1: Primary driver ───────────────────────────────────────────────────
    sentences.append(base_reasoning.rstrip(".!") + ".")

    # ── S2: Alternative consideration ────────────────────────────────────────
    alternatives: list[str] = []
    if action_id not in ("eat", "gather") and v["hun"] > 0.50:
        alternatives.append("yiyecek bulmaya gidebilirdim")
    if action_id != "sleep" and v["en"] < 0.40:
        alternatives.append("dinlenebilirdim")
    if action_id != "flee" and any(p.get("tag") == "Threat" for p in percepts):
        alternatives.append("kaçmayı düşünebilirdim")
    if action_id != "socialize" and any(p.get("tag") == "Social" for p in percepts):
        alternatives.append("yakındaki biriyle konuşabilirdim")
    if action_id != "work" and sched["act"] == "work" and v["en"] > 0.4:
        alternatives.append("çalışmaya devam edebilirdim")

    if alternatives:
        alt = random.choice(alternatives)
        sentences.append(
            f"Elbette {alt}, ancak içinde bulunduğum durum bunu şu an için doğru seçim yapmıyor."
        )
    else:
        sentences.append("Başka bir seçenek düşündüm ama mevcut koşullar beni bu karardan alıkoyuyor.")

    # ── S3: Trait / emotion influence ────────────────────────────────────────
    _COMBAT_TRAITS = {"Brave", "Cautious", "Anxious", "Aggressive", "Wrathful"}
    is_combat_context = (
        action_id in {"attack", "flee", "heal"}
        or any(p.get("tag") == "Threat" for p in percepts)
    )
    trait_sentences = {
        "Brave":     f"Cesaretim beni öne atılmaya itiyor ({emo['fear']:.2f} korku hissetsem de).",
        "Cunning":   "Kurnaz biri olarak her adımı hesaplıyorum, anlık karar vermek hata olur.",
        "Cautious":  f"Temkinli yapım beni acele kararlardan alıkoyuyor, ihtiyatlı olmak şart.",
        "Anxious":   f"İçim sıkışıyor, korku ({emo['fear']:.2f}) her şeyi daha ağır hissettiriyor.",
        "Honorable": "Onurumu lekelemeden doğru olanı yapmak zorundayım.",
        "Aggressive":f"Öfkem ({emo['ang']:.2f}) beni harekete geçiriyor, duraksamamalıyım.",
        "Devout":    "İnancım bana yol gösteriyor; tanrının isteği her şeyin önünde.",
        "Greedy":    "Çıkarlarımı düşünüyorum; bu hamle uzun vadede bana kazandırır.",
        "Wrathful":  f"Sinirlerim gerildi ({emo['ang']:.2f}), sabırsızlanıyorum.",
        "Loyal":     "Sadakatim beni doğru adımı atmaya zorluyor.",
    }
    found_sentence = None
    for t in traits:
        if t in trait_sentences and (is_combat_context or t not in _COMBAT_TRAITS):
            found_sentence = trait_sentences[t]
            break
    if not found_sentence:
        mood = emo.get("mood", "Calm")
        found_sentence = f"Şu anki ruh halim ({mood}) bu kararımı şekillendiriyor."
    sentences.append(found_sentence)

    # ── S4: Final decision rationale ─────────────────────────────────────────
    action_rationale = {
        "eat":       "Sonuç olarak yemek yemek en acil ihtiyacım, önce bunu karşılamalıyım.",
        "drink":     "Sonuç olarak su içmek en acil ihtiyacım, önce bunu karşılamalıyım.",
        "sleep":     "Bu yüzden dinlenmeden başka seçeneğim yok; yorgun bir zihin doğru karar veremez.",
        "flee":      "Bu yüzden geri çekilmek ve güvenli bir yer bulmak en mantıklı hamle.",
        "gather":    "Bu yüzden kaynakları toplamak ve ihtiyaçlarımı karşılamak önceliğim.",
        "heal":      "Bu yüzden önce kendimi iyileştirip güçlenmem, sonra diğer her şeyi düşünebilirim.",
        "attack":    "Bu yüzden tehdidi bertaraf etmek için harekete geçmem gerekiyor.",
        "socialize": "Bu yüzden bu anı değerlendirip bağ kurmak istiyorum; insan teması da bir ihtiyaç.",
        "trade":     "Bu yüzden bir alışveriş yapmak şu an için en kazançlı adım.",
        "work":      "Bu yüzden işime devam ediyorum; sorumluluklarımı ihmal edemem.",
        "pray":      "Bu yüzden dua ederek içimi ferahlatmak ve kafamı toparlamak istiyorum.",
        "walk_to":   "Bu yüzden etrafı keşfedip durumu değerlendirmeye karar verdim.",
    }
    sentences.append(action_rationale.get(action_id, "Bu yüzden en uygun kararı veriyorum."))

    # ── S5: Memory influence (optional) ──────────────────────────────────────
    strong_mems = [m for m in memories if abs(m.get("ew", 0)) > 0.6]
    if strong_mems:
        mem = random.choice(strong_mems)
        if mem["ew"] < 0:
            sentences.append(
                f"Daha önce yaşadıklarım ({mem['desc'][:40]}...) bana bu tür durumlarda "
                f"dikkatsiz davranmanın bedelini öğretti."
            )
        else:
            sentences.append(
                f"Geçmişteki deneyimlerim ({mem['desc'][:40]}...) bana bu yolun doğru "
                f"olduğunu hatırlatıyor."
            )

    return " ".join(sentences)


# ─────────────────────────────────────────────────────────────────────────────
# Intent deviation selector (~15% of examples)
# ─────────────────────────────────────────────────────────────────────────────

def _select_action_with_deviation(state: dict) -> tuple[str, str, str | None, str] | None:
    """
    Return an intentional override of the biologically optimal action,
    or None if no deviation case applies for this state.

    Four deviation cases:
      D1 – hun>0.65 + social percept -> socialize (eating deferred)
      D2 – en<0.35 + non-threat nearby percept -> walk_to (curiosity)
      D3 – work conditions met + low mood/anger -> pray or socialize instead
      D4 – gather condition met + social percept -> stay in conversation
    """
    v        = state["vitals"]
    emo      = state["emo"]
    percepts = state["percepts"]
    inv_ids  = [i["id"] for i in state["inv"]]
    role     = state["occ"]
    arch     = state["arch"]

    social_p  = next((p for p in percepts if p.get("tag") == "Social"), None)
    non_threat = next((p for p in percepts if p.get("tag") != "Threat"), None)
    sched_act = state["sched"]["act"]

    cases = []

    # D1: Hungry but social is active -> consciously defer eating
    if v["hun"] > 0.65 and social_p:
        cases.append((
            "socialize",
            f"Gerçi açım ({v['hun']:.2f}) ama {social_p['id']} burada. Yemek biraz bekleyebilir, "
            f"bu konuşma şu an daha önemli.",
            random.choice(["Merhaba! Ne var ne yok?", "Seni görmek iyi oldu.", "Anlat bakalım!"]),
            "Happy",
        ))

    # D2: Tired but something interesting nearby -> curiosity wins
    if v["en"] < 0.35 and non_threat and non_threat.get("tag") != "Threat":
        cases.append((
            "walk_to",
            f"Yorgunum aslında ama {non_threat['id']}'yi merak ettim. "
            f"Uyku biraz bekleyebilir, gidip bir bakayım.",
            None,
            "Calm",
        ))

    # D3: Should be working (schedule says work, energy ok) but mood is low -> pray/socialize
    if sched_act == "work" and v["en"] > 0.4 and (emo["hap"] < 0.15 or emo["ang"] > 0.55):
        alt = "pray" if ("Devout" in state["traits"] or role == "Priest") else "socialize"
        if alt == "pray":
            cases.append((
                "pray",
                f"Çalışma saatim ama ruh halim yerinde değil (mutluluk:{emo['hap']:.2f}). "
                f"Çalışmadan önce dua edip kendimi toparlayayım.",
                None, "Calm",
            ))
        elif social_p:
            cases.append((
                "socialize",
                f"Çalışmam gerekirdi ama moralim çok bozuk (öfke:{emo['ang']:.2f}). "
                f"{social_p['id']} ile biraz konuşmak bana iyi gelir.",
                random.choice(["Zor bir gün...", "Seninle konuşmam lazımdı."]),
                "Calm",
            ))

    # D4: Gather conditions met but mid-conversation -> stay and talk
    if (v["hun"] > 0.3 or v["thi"] > 0.3) and "food" not in inv_ids and social_p:
        cases.append((
            "socialize",
            f"Kaynak toplamam lazım ama {social_p['id']} ile konuşmanın ortasındayım. "
            f"Önce bunu bitireyim, sonra gidip toplarım.",
            random.choice(["Devam edelim...", "Biraz daha var."]),
            "Calm",
        ))

    # D5: Dual crisis -- hp critical AND hunger high -> heal wins with explicit CoT
    if v["hp"] < 30 and v["hun"] > 0.70 and "medicine" in inv_ids:
        cases.append((
            "heal",
            f"Canım kritik ({v['hp']:.1f}) ve hem de aç ({v['hun']:.2f}). "
            f"İkisi de acil ama bir ölü yemek yiyemez; önce kendimi iyileştirmeliyim, "
            f"sonra yiyecek düşünürüm.",
            None,
            "Fearful",
        ))

    # D6: Brave archetype registering elevated fear -> conflicted courage
    # Brave b5.n = 0.2, so fear is bounded ~0.0-0.2; threshold 0.12 captures ~top 40% of Brave fear
    if arch == "Brave" and emo["fear"] > 0.12:
        threats = [p for p in percepts if p.get("tag") == "Threat"]
        if threats:
            top = max(threats, key=lambda p: p["threat"])
            cases.append((
                "attack",
                f"Korkuyorum, bunu inkâr etmeyeceğim ({emo['fear']:.2f}). "
                f"Ama cesaretim benim doğamda; {top['id']}'dan kaçmak kim olduğuma ihanet olur. "
                f"Gözlerimi yumup ilerliyorum.",
                None,
                "Brave",
            ))

    # D7: Schedule says work but energy is critically low -> body overrides duty
    if sched_act == "work" and v["en"] < 0.30:
        cases.append((
            "sleep",
            f"Çalışma saatim, görevlerim bekliyor. Ama bedenim artık dayanamıyor "
            f"(enerji: {v['en']:.2f}). Yorulmuş bir zihinle iyi iş çıkaramam; "
            f"kısa bir uyku her şeyi düzeltirecek.",
            None,
            "Tired",
        ))

    # D8 removed: trauma-driven flee now emerges from memory_bias() in decision_factors,
    # not as a fixed rule. It was firing for Brave archetypes, creating contradictions.

    if not cases:
        return None
    return random.choice(cases)


# ─────────────────────────────────────────────────────────────────────────────
# Training example builder (Llama-3 instruct format)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Sen bir medeniyet simülasyonundaki NPC'nin bilişsel karar motorusun.\n"
    "Sana NPC'nin anlık durumunu JSON formatında göndereceğim.\n"
    "Görevin: Yalnızca tek bir eylem kararı üret, JSON formatında döndür.\n"
    "Kurallar:\n"
    "- action_id MUTLAKA valid_actions listesindeki değerlerden biri olmalıdır\n"
    "- reasoning: birinci şahıs iç monolog, 1-3 cümle\n"
    "- dialogue: yalnızca socialize/trade eylemlerinde doldur, diğerlerinde null\n"
    "- emotion: NPC'nin şu anki baskın duygu durumu (tek kelime)\n"
    "- Yanıtın SADECE JSON olsun -- kod bloğu, açıklama yok"
)

TEMPLATE = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "{assistant}<|eot_id|>")

_DEVIATION_RATE  = 0.15   # 15% of examples use intent deviation
_PARAPHRASE_RATE = 0.175  # 17.5% of formatter examples are paraphrased (target 15-20%)


SYSTEM_PROMPT_REASONER = (
    "Sen bir ortaçağ simülasyonundaki NPC'nin iç ses motorusun.\n"
    "Sana NPC'nin anlık durumunu JSON olarak göndereceğim.\n"
    "Görevin: NPC'nin ne yapması gerektiğini ve NEDEN yapması gerektiğini\n"
    "3-5 cümlelik Türkçe iç monolog olarak yaz.\n"
    "ASLA JSON üretme. ASLA action_id adı kullanma. Sadece düşün.\n"
    "Kişilik özellikleri, hayatta kalma ihtiyaçları, tehdit algısı,\n"
    "duygusal durum, hafıza ve sosyal bağlamı değerlendir."
)

SYSTEM_PROMPT_FORMATTER = (
    "Sen bir NPC simülasyonu için JSON dönüştürücüsün.\n"
    "Sana bir NPC'nin ne yapmak istediğini Türkçe olarak anlatacağım.\n"
    "Bunu aşağıdaki JSON şemasına dönüştür:\n"
    '{"reasoning":"<girdiyi kopyala>",'
    '"selected_action":{"action_id":"<listeden>","target_id":null,"dialogue":null},'
    '"emotion":"<tek kelime>"}\n'
    'valid_actions: ["eat","drink","sleep","flee","gather","heal",'
    '"attack","socialize","trade","work","pray","walk_to"]\n'
    "SADECE JSON yaz. Kod bloğu veya açıklama ekleme."
)


def build_example(use_gemma: bool = True) -> dict:
    """Build one Reasoner training example (NPC state + persona -> CoT reasoning)."""
    state  = generate_npc_state()
    persona = build_persona_card(state)

    # Action selection: multi-factor standard (~85%) or intentional deviation (~15%)
    action_id, base_reasoning, dialogue, emotion = _select_action_standard(state)
    if random.random() < _DEVIATION_RATE:
        deviation = _select_action_with_deviation(state)
        if deviation is not None:
            action_id, base_reasoning, dialogue, emotion = deviation

    assert action_id in NPC_SIM_ACTIONS, f"Invalid action: {action_id}"

    # Compute multi-factor scores for Gemma context (always, even for deviation path)
    _, factors = pick_action_multifactor(state)

    # Full CoT: Gemma bootstrap first, template fallback
    cot: str | None = None
    if use_gemma:
        cot = generate_via_gemma(state, action_id, factors, persona)
    if cot is None:
        cot = generate_cot_reasoning(state, action_id, base_reasoning)

    # User payload: persona preamble + state JSON (fixes R14 — "who am I" anchor)
    state_json   = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    user_payload = f"{persona}\n\n{state_json}"

    # Formatter target: no npc_id (runtime injects it), reasoning = full CoT (fixes R10/R11)
    structured_out = {
        "reasoning": cot,
        "selected_action": {
            "action_id": action_id,
            "target_id": None,
            "dialogue":  dialogue,
        },
        "emotion": emotion,
    }
    assistant_json = json.dumps(structured_out, ensure_ascii=False, separators=(",", ":"))

    reasoner_text = TEMPLATE.format(
        system=SYSTEM_PROMPT_REASONER,
        user=user_payload,
        assistant=cot,
    )
    return {
        "text":            reasoner_text,
        "_cot":            cot,
        "_action_id":      action_id,
        "_emotion":        emotion,
        "_dialogue":       dialogue,
        "_assistant_json": assistant_json,
        "_state":          state,   # used by oversampling logic in generate_dataset
    }


def build_formatter_example(example: dict, augment: bool = False) -> dict:
    """Derive one Formatter training example from a Reasoner example.

    user    = cot_reasoning (optionally paraphrased)
    assistant = the structured JSON output
    """
    cot_text = example["_cot"]
    if augment:
        cot_text = _paraphrase(cot_text)

    formatter_text = TEMPLATE.format(
        system=SYSTEM_PROMPT_FORMATTER,
        user=cot_text,
        assistant=example["_assistant_json"],
    )
    return {"text": formatter_text}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(path: str, count: int, seed: int = 42, use_gemma: bool = True) -> list[dict]:
    """Generate Reasoner training dataset. Returns the raw example list for reuse."""
    random.seed(seed)
    print(f"Generating {count} reasoner examples -> {path}")
    if use_gemma:
        print("  Bootstrap mode: Gemma 4 E4B via Ollama (falls back to template if unavailable)")
    samples: list[dict] = []
    pbar = _tqdm(total=count, unit="ex", dynamic_ncols=True)
    while len(samples) < count:
        example = build_example(use_gemma=use_gemma)
        prev_len = len(samples)
        samples.append(example)

        # ── Oversampling rules ────────────────────────────────────────────────
        try:
            state    = example.get("_state", {})
            arch     = state.get("arch", "")
            percepts = state.get("percepts", [])
            traits   = state.get("traits", [])

            # 1. Brave x high-threat -> attack (3x)
            is_brave_threat = (
                arch == "Brave"
                and any(p.get("threat", 0) >= 0.7 for p in percepts)
            )
            if is_brave_threat:
                for _ in range(2):
                    if len(samples) < count:
                        samples.append(example)

            # 2. Fearful x social percept -> social-anxiety case (2x)
            is_fearful_social = (
                arch == "Fearful"
                and any(p.get("tag") == "Social" for p in percepts)
            )
            if is_fearful_social and len(samples) < count:
                samples.append(example)

            # 3. Honorable x attack-eligible trigger -> moral conflict (2x)
            is_honorable_attack = (
                "Honorable" in traits
                and any(p.get("tag") == "Threat" for p in percepts)
                and example["_action_id"] == "attack"
            )
            if is_honorable_attack and len(samples) < count:
                samples.append(example)

        except Exception:
            pass  # Non-critical: skip oversampling if parsing fails

        pbar.update(len(samples) - prev_len)
    pbar.close()

    # Write exactly count samples (strip private _ fields)
    with open(path, "w", encoding="utf-8") as f:
        for i, example in enumerate(samples[:count]):
            out = {"text": example["text"]}
            json.dump(out, f, ensure_ascii=False)
            f.write("\n")
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{count}...")
    print(f"  Done -> {os.path.getsize(path) // 1024} KB")
    return samples[:count]


def generate_formatter_dataset(
    path: str,
    reasoner_examples: list[dict],
    augment_rate: float = _PARAPHRASE_RATE,
) -> None:
    """Derive Formatter training corpus from Reasoner examples.

    For each Reasoner example:
      - Always emit 1 clean (unaugmented) formatter example.
      - With probability `augment_rate`, also emit 1 paraphrased variant.

    This yields ~15-20% augmented examples in the final corpus, teaching
    Component B to tolerate imperfect Reasoner output phrasing.
    """
    print(f"Generating formatter corpus -> {path}")
    fmt_samples: list[dict] = []
    for ex in reasoner_examples:
        fmt_samples.append(build_formatter_example(ex, augment=False))
        if random.random() < augment_rate:
            fmt_samples.append(build_formatter_example(ex, augment=True))

    random.shuffle(fmt_samples)
    with open(path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(fmt_samples):
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(fmt_samples)}...")
    print(f"  Done -> {os.path.getsize(path) // 1024} KB  ({len(fmt_samples)} examples)")


if __name__ == "__main__":
    base     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Pass --no-gemma to skip Ollama bootstrap and use template CoT only
    use_gemma = "--no-gemma" not in sys.argv

    # ── Reasoner corpus ───────────────────────────────────────────────────────
    train_examples = generate_dataset(
        os.path.join(data_dir, "train_reasoner.jsonl"), count=10000, seed=456234,
        use_gemma=use_gemma,
    )
    _test_examples = generate_dataset(
        os.path.join(data_dir, "test_reasoner.jsonl"),  count=2000,  seed=984756,
        use_gemma=use_gemma,
    )

    # ── Formatter corpus (derived from train split only) ──────────────────────
    generate_formatter_dataset(
        os.path.join(data_dir, "train_formatter.jsonl"),
        reasoner_examples=train_examples,
        augment_rate=_PARAPHRASE_RATE,
    )

    print(
        "\nAll done. v5 datasets ready:\n"
        "  train_reasoner.jsonl  -- 10k CoT examples for Reasoner (Llama 3.2 3B)\n"
        "  test_reasoner.jsonl   -- 2k  evaluation set\n"
        "  train_formatter.jsonl -- ~12k+ (cot -> JSON) pairs for Formatter (Llama 3.2 1B-Instruct)\n"
        "  (Run with --no-gemma to skip Ollama bootstrap and use template CoT)\n"
    )
