# Copyright 2025-2026 Sadık Abdusselam Albayrak
# Licensed under the Apache License, Version 2.0
"""
Persona card builder for NPC training data (v1.6.0).

Generates a 2-3 sentence Turkish narrative preamble from NPC state.
This gives the Reasoner model a "who am I" anchor before the raw JSON,
fixing R14 (pure-JSON input forces the 3B model to build identity on the fly).

Public API:
  build_persona_card(state) -> str   # 2-3 sentence Turkish paragraph
"""

from __future__ import annotations
import random

# ─────────────────────────────────────────────────────────────────────────────
# Name pools  (~30 male, ~30 female)
# ─────────────────────────────────────────────────────────────────────────────

_MALE_NAMES = [
    "Ahmet", "Mehmet", "Ali", "Hasan", "Mustafa", "İbrahim", "Yusuf",
    "Ömer", "Hüseyin", "Kadir", "Emre", "Kaan", "Mert", "Burak", "Serkan",
    "Tolga", "Oğuz", "Erdal", "Selim", "Tarık", "Kemal", "Cengiz", "Baran",
    "Furkan", "Berkay", "Çağrı", "Uğur", "Alp", "Doğan", "Ercan",
]

_FEMALE_NAMES = [
    "Ayşe", "Fatma", "Zeynep", "Hatice", "Elif", "Emine", "Selin",
    "Büşra", "Merve", "Gül", "Leyla", "Derya", "Ceren", "İrem", "Pınar",
    "Özlem", "Tuğba", "Arzu", "Nilüfer", "Şeyma", "Beyza", "Nehir",
    "Melisa", "Defne", "Serap", "Nazan", "Rüya", "Gökçe", "Bahar", "Işıl",
]

# ─────────────────────────────────────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────────────────────────────────────

_OCC_TR: dict[str, str] = {
    "Guard": "muhafız",
    "Merchant": "tüccar",
    "Priest": "rahip",
    "Farmer": "çiftçi",
    "Scholar": "akademisyen",
    "Blacksmith": "demirci",
    "Thief": "hırsız",
    "Knight": "şövalye",
    "Wizard": "büyücü",
    "Innkeeper": "hancı",
    "Goblin": "goblin",
    "Bandit": "haydut",
    "King": "kral",
    "Herbalist": "otacı",
    "Bard": "ozan",
}

_FACTION_TR: dict[str, str] = {
    "CityWatch": "Şehir Muhafızları",
    "MerchantGuild": "Tüccar Loncası",
    "Bandits": "Haydutlar",
    "Church": "Kilise",
    "Farmers": "Çiftçi Kooperatifi",
}

_TRAIT_ADJ_TR: dict[str, str] = {
    "Brave": "cesur",
    "Cunning": "kurnaz",
    "Cautious": "temkinli",
    "Anxious": "endişeli",
    "Honorable": "onurlu",
    "Aggressive": "saldırgan",
    "Devout": "dindar",
    "Greedy": "açgözlü",
    "Wrathful": "öfkeli",
    "Loyal": "sadık",
    "Coward": "korkak",
    "Pacifist": "barışsever",
}

# Big Five key → Turkish name used in sentences
_B5_LABEL_TR: dict[str, str] = {
    "e": "dışadönüklük",
    "a": "uyumluluk",
    "c": "öz-denetim",
    "n": "nevrotiklik",
    "o": "açıklık",
}

# High-value b5 → descriptive sentence added as S3
_B5_HIGH_SENTENCE: dict[str, str] = {
    "e": "Çevresiyle kolayca kaynaşır; sessiz kalmak ona göre değildir.",
    "a": "Başkalarıyla anlaşmayı tercih eder, çatışmadan kaçınır.",
    "c": "Sorumluluklarını hiç ihmal etmez; düzen ve disiplin onun için vazgeçilmezdir.",
    "n": "Duygusal dalgalanmalara yatkındır; stres altında kendini zor toplar.",
    "o": "Meraklı ve yaratıcıdır; bilmediği her şey onu çeker.",
}

_B5_LOW_SENTENCE: dict[str, str] = {
    "e": "Kalabalıktan uzak durmayı sever; sessizlikte huzur bulur.",
    "a": "Anlaşmazlık çıkarmaktan çekinmez; doğru bildiğinden şaşmaz.",
    "c": "Kurallara fazla bağlı değildir; anlık kararlarla hareket eder.",
    "n": "Baskı altında bile soğukkanlılığını kaybetmez; duygularını sıkı tutar.",
    "o": "Tanıdık ve bildik olan onu rahatlatır; belirsizlikten hoşlanmaz.",
}


def _b5_decile(val: float) -> str:
    if val >= 0.70:
        return "yüksek"
    if val >= 0.45:
        return "orta"
    return "düşük"


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_persona_card(state: dict) -> str:
    """
    Return a 2-3 sentence Turkish narrative about this NPC.

    S1 — Identity: name + role (+ faction if notable)
    S2 — Personality: dominant traits + b5 decile summary
    S3 — Optional: extreme b5 elaboration (only when max b5 >= 0.75 or <= 0.25)
    """
    occ = state["occ"]
    traits = state["traits"]
    b5 = state["b5"]
    faction = state.get("faction", "")

    name = random.choice(_MALE_NAMES if random.random() < 0.5 else _FEMALE_NAMES)
    occ_tr = _OCC_TR.get(occ, occ.lower())

    # ── S1: identity ─────────────────────────────────────────────────────────
    faction_tr = _FACTION_TR.get(faction, "")
    if faction_tr:
        s1 = f"{name}, {faction_tr} bünyesinde görev yapan bir {occ_tr}."
    else:
        s1 = f"{name}, bu diyarda {occ_tr} olarak geçimini sağlar."

    # ── S2: personality ───────────────────────────────────────────────────────
    trait_adjs = [_TRAIT_ADJ_TR[t] for t in traits if t in _TRAIT_ADJ_TR]
    b5_parts = [f"{_B5_LABEL_TR[k]} {_b5_decile(v)}" for k, v in b5.items()]

    if trait_adjs:
        adj_str = " ve ".join(trait_adjs[:2])
        s2 = f"{adj_str.capitalize()} biri olarak tanınır; {', '.join(b5_parts[:3])}."
    else:
        s2 = f"Sıradan biri gibi görünse de {', '.join(b5_parts[:3])} değerleri onu şekillendirir."

    # ── S3: extreme b5 elaboration (optional) ────────────────────────────────
    sentences = [s1, s2]
    max_k = max(b5, key=b5.get)
    min_k = min(b5, key=b5.get)
    if b5[max_k] >= 0.75:
        sentences.append(_B5_HIGH_SENTENCE[max_k])
    elif b5[min_k] <= 0.25:
        sentences.append(_B5_LOW_SENTENCE[min_k])

    return " ".join(sentences)
