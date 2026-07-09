"""
Dialogue handler: takes player input + NPC state → generates response,
then updates the NPC's dynamic situation layer (memory, goal, emotion).
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from backend.config.settings import (
    DATA_DIR,
    DYNAMIC_UPDATE_EVERY,
    HISTORY_MAX_MESSAGES,
    SHORT_TERM_MEMORY_SIZE,
    VOICES_DIR,
)
from backend.behavior.policy import RuleBasedPolicy
from backend.behavior.schemas import PolicyAction, StateFeatures
from backend.behavior.state_encoder import StateEncoder
from backend.behavior.supervised_policy import SupervisedPolicy
from backend.llm.json_utils import coerce_str, parse_llm_json
from backend.llm.ollama_client import OllamaClient
from backend.llm.persona.memory import MemoryStream
from backend.llm.persona.models import NPC

if TYPE_CHECKING:
    from backend.behavior.dialogue_guard import DialogueGuard, GuardResult
    from backend.tts.xtts_client import XTTSClient


_SYSTEM_TEMPLATE = """\
You are {name}, an NPC in an RPG game. Stay completely in character.

## Who you are
Occupation: {occupation}
Backstory: {backstory}
Values: {values}
Speech style: {speech_style}
Knowledge: {knowledge}

## Your world
Faction: {faction}
Reputation: {reputation}

## Your current state
Goal: {goal}
Emotional state: {emotional_state}

{memory_block}Rules:
- Never break character or mention you are an AI.
- Keep responses concise (2–4 sentences) unless pressed for detail.
- Reflect your speech style in every reply.
- If asked about things outside your knowledge domains, say so in character.
"""

_DYNAMIC_UPDATE_PROMPT = """\
You are the game engine for an RPG NPC named {name} ({occupation}).
Current goal: {goal}
Current emotional state: {emotional_state}

Recent conversation:
{recent}

Based on the recent conversation, update the NPC's state. Reply ONLY with JSON:
  {{"current_goal": "...", "emotional_state": "..."}}
Keep values short. If nothing meaningful changed, return the current values.
"""


_PROMPT_STYLES = ("layered", "flat", "none")
_POLICY_MODES = ("llm_only", "rule", "trained")

_RULES_BLOCK = """\
Rules:
- Never break character or mention you are an AI.
- Keep responses concise (2–4 sentences) unless pressed for detail.
- If asked about things outside your knowledge, say so in character.
"""


class DialogueHandler:
    def __init__(
        self,
        llm: OllamaClient,
        npc: NPC,
        tts: Optional["XTTSClient"] = None,
        use_memory: bool = True,
        dynamic_updates: bool = True,
        prompt_style: str = "layered",
        system_prompt_text: Optional[str] = None,
        guard: Optional["DialogueGuard"] = None,
        policy_mode: str = "llm_only",
        behavior_policy=None,
        trained_policy_checkpoint: Optional[str] = None,
    ):
        # use_memory / dynamic_updates / prompt_style / system_prompt_text are
        # evaluation-condition switches (baselines and ablations); defaults
        # reproduce the full system. guard is the optional rule-based dialogue
        # guard (secret/injection protection); None keeps legacy behavior.
        # policy_mode / behavior_policy / trained_policy_checkpoint drive the
        # optional dialogue-side policy action injection; "llm_only" (default)
        # keeps the plain LLM dialogue contract.
        if prompt_style not in _PROMPT_STYLES:
            raise ValueError(
                f"Unknown prompt_style '{prompt_style}', expected one of {_PROMPT_STYLES}"
            )
        if policy_mode not in _POLICY_MODES:
            raise ValueError(
                f"Unknown policy_mode '{policy_mode}', expected one of {_POLICY_MODES}"
            )
        self.llm = llm
        self.npc = npc
        self.tts = tts
        self.use_memory = use_memory
        self.dynamic_updates = dynamic_updates
        self.prompt_style = prompt_style
        self.system_prompt_text = system_prompt_text
        self.guard = guard
        self.last_guard: Optional["GuardResult"] = None
        self.policy_mode = policy_mode
        self.behavior_policy = behavior_policy
        self.trained_policy_checkpoint = (
            trained_policy_checkpoint
            or os.path.join(DATA_DIR, "behavior_policy", "checkpoints", "stateful_rpg_a40")
        )
        self.state_encoder = StateEncoder()
        self._rule_policy = RuleBasedPolicy()
        self._trained_policy = None
        self.last_policy_action: Optional[PolicyAction] = None
        self.last_state_features: Optional[StateFeatures] = None
        self.memory = MemoryStream.from_list(npc.memory_log)
        self.history: List[Dict[str, str]] = []   # [{role, content}, ...]
        self._turn_count = 0

    def _build_system_prompt(
        self,
        query: str,
        policy_action: Optional[PolicyAction] = None,
        memories: Optional[List[str]] = None,
    ) -> str:
        if self.system_prompt_text is not None:
            return self.system_prompt_text + self._policy_action_block(policy_action)
        if self.prompt_style == "none":
            prompt = (
                f"You are {self.npc.core.name}, an NPC in an RPG game "
                f"({self.npc.core.occupation}). Stay in character.\n\n" + _RULES_BLOCK
            )
            return prompt + self._policy_action_block(policy_action)
        if self.prompt_style == "flat":
            return self._build_flat_prompt() + self._policy_action_block(policy_action)
        return self._build_layered_prompt(query, memories=memories) + self._policy_action_block(policy_action)

    def _build_flat_prompt(self) -> str:
        """Same persona facts as the layered prompt, as one unstructured paragraph."""
        c, s, d = self.npc.core, self.npc.social, self.npc.dynamic
        paragraph = (
            f"You are {c.name}, a {c.occupation} NPC in an RPG game. {c.backstory} "
            f"You value {', '.join(c.values)}. Your speech style is {c.speech_style}. "
            f"You know about {', '.join(c.knowledge_domains)}. "
            f"You belong to {s.faction} and are known as {s.reputation}. "
            f"Your current goal is {d.current_goal} and you feel {d.emotional_state}."
        )
        return paragraph + "\n\n" + _RULES_BLOCK

    def _build_layered_prompt(self, query: str, memories: Optional[List[str]] = None) -> str:
        if self.use_memory:
            memories = self._retrieve_memories(query) if memories is None else memories
            memory_block = (
                "## What you remember\n"
                + ("\n".join(f"- {m}" for m in memories)
                   if memories else "No relevant memories yet.")
                + "\n\n"
            )
        else:
            memory_block = ""
        c, s, d = self.npc.core, self.npc.social, self.npc.dynamic
        return _SYSTEM_TEMPLATE.format(
            name=c.name,
            occupation=c.occupation,
            backstory=c.backstory,
            values=", ".join(c.values),
            speech_style=c.speech_style,
            knowledge=", ".join(c.knowledge_domains),
            faction=s.faction,
            reputation=s.reputation,
            goal=d.current_goal,
            emotional_state=d.emotional_state,
            memory_block=memory_block,
        )

    def _policy_action_block(self, policy_action: Optional[PolicyAction]) -> str:
        if policy_action is None:
            return ""
        action = policy_action.to_dict()
        return f"""

## Policy action
Dialogue act: {action["dialogue_act"]}
Emotion: {action["emotion"]}
Allowed disclosure level: {action["disclosure_level"]}
Required gesture: {action["gesture"]}
Quest update: {action["quest_update"]}
Memory write type: {action["memory_write_type"]}

Policy rules:
- Do not change the policy action.
- Do not exceed the allowed disclosure level.
- Return JSON only: {{"reply": "...", "emotion": "...", "used_facts": ["..."], "memory_to_store": "..."}}
"""

    def _retrieve_memories(self, query: str) -> List[str]:
        if not self.use_memory:
            return []
        return self.memory.retrieve(query)

    def _resolve_policy_mode(self, policy_mode: Optional[str]) -> str:
        mode = policy_mode or self.policy_mode
        if mode not in _POLICY_MODES:
            raise ValueError(f"Unknown policy_mode '{mode}', expected one of {_POLICY_MODES}")
        return mode

    def _policy_for_mode(self, mode: str):
        if mode == "llm_only":
            return None
        if self.behavior_policy is not None:
            return self.behavior_policy
        if mode == "rule":
            return self._rule_policy
        if mode == "trained":
            if self._trained_policy is None:
                checkpoint = Path(self.trained_policy_checkpoint)
                if not checkpoint.exists():
                    raise FileNotFoundError(
                        f"Trained policy checkpoint not found: {checkpoint}"
                    )
                self._trained_policy = SupervisedPolicy(checkpoint)
            return self._trained_policy
        return None

    def _compute_policy_action(
        self,
        player_input: str,
        memories: List[str],
        game_state: Optional[Dict] = None,
        policy_mode: Optional[str] = None,
    ) -> Optional[PolicyAction]:
        mode = self._resolve_policy_mode(policy_mode)
        policy = self._policy_for_mode(mode)
        if policy is None:
            self.last_state_features = None
            self.last_policy_action = None
            return None
        state = self.state_encoder.encode(
            player_text=player_input,
            npc=self.npc,
            retrieved_memories=memories,
            game_state=game_state,
        )
        action = policy.predict(state)
        self.last_state_features = state
        self.last_policy_action = action
        return action

    def _coerce_policy_reply(
        self,
        raw_reply: str,
        policy_action: Optional[PolicyAction],
    ) -> tuple[str, str]:
        if policy_action is None:
            return raw_reply, ""
        try:
            data = parse_llm_json(raw_reply)
        except ValueError:
            return raw_reply, ""
        reply = coerce_str(data.get("reply", raw_reply))
        memory_to_store = coerce_str(data.get("memory_to_store", ""))
        return reply, memory_to_store

    def respond(
        self,
        player_input: str,
        game_state: Optional[Dict] = None,
        policy_mode: Optional[str] = None,
    ) -> str:
        return self.respond_with_metadata(
            player_input, game_state=game_state, policy_mode=policy_mode
        )["reply"]

    def respond_with_metadata(
        self,
        player_input: str,
        game_state: Optional[Dict] = None,
        policy_mode: Optional[str] = None,
    ) -> Dict:
        memories = self._retrieve_memories(player_input)
        action = self._compute_policy_action(
            player_input,
            memories=memories,
            game_state=game_state,
            policy_mode=policy_mode,
        )
        system = self._build_system_prompt(player_input, policy_action=action, memories=memories)

        # Rule-based guard: the decision to refuse (secret probing, prompt
        # injection) is made here; the LLM only phrases it in character.
        # Injection text is replaced before it reaches the LLM or the history,
        # so there is nothing for the model to obey.
        self.last_guard = self.guard.assess(player_input, self.npc) if self.guard else None
        llm_input = player_input
        if self.last_guard is not None:
            system = system + "\n" + self.last_guard.instruction
            if self.last_guard.sanitized_input:
                llm_input = self.last_guard.sanitized_input

        self.history.append({"role": "user", "content": llm_input})
        self.history = self.history[-HISTORY_MAX_MESSAGES:]
        raw_reply = self.llm.chat(self.history, system=system)
        reply, policy_memory = self._coerce_policy_reply(raw_reply, action)
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-HISTORY_MAX_MESSAGES:]

        # Update dynamic layer: memory stream, short-term mirror, persisted log.
        # Injection attempts are not memorized — they would poison retrieval.
        injection_blocked = (
            self.last_guard is not None and self.last_guard.reason == "prompt_injection"
        )
        if not injection_blocked:
            self.memory.add(f"Player said: {player_input}", importance=0.4)
        self.memory.add(f"I ({self.npc.core.name}) replied: {reply}", importance=0.5)
        if policy_memory:
            self.memory.add(policy_memory, importance=0.6)
        self.npc.dynamic.short_term_memory = self.memory.recent(SHORT_TERM_MEMORY_SIZE)
        self.npc.memory_log = self.memory.to_list()

        self._turn_count += 1
        if self.dynamic_updates and self._turn_count % DYNAMIC_UPDATE_EVERY == 0:
            self._update_dynamic_state()

        if self.tts is not None:
            voice_path = os.path.join(
                VOICES_DIR, f"{self.npc.core.name.lower().replace(' ', '_')}.wav"
            )
            self.tts.speak(reply, voice_path)

        return {
            "reply": reply,
            "policy_mode": self._resolve_policy_mode(policy_mode),
            "action": action.to_dict() if action is not None else None,
            "state": (
                self.last_state_features.to_dict()
                if self.last_state_features is not None else None
            ),
        }

    def _update_dynamic_state(self) -> None:
        """Ask the LLM to re-evaluate goal/emotion; keep old state on failure."""
        d = self.npc.dynamic
        prompt = _DYNAMIC_UPDATE_PROMPT.format(
            name=self.npc.core.name,
            occupation=self.npc.core.occupation,
            goal=d.current_goal,
            emotional_state=d.emotional_state,
            recent="\n".join(self.memory.recent(SHORT_TERM_MEMORY_SIZE)),
        )
        try:
            data = parse_llm_json(self.llm.generate(prompt))
        except ValueError:
            return
        d.current_goal = coerce_str(data.get("current_goal", d.current_goal))
        d.emotional_state = coerce_str(data.get("emotional_state", d.emotional_state))

    def reset(self) -> None:
        self.history.clear()
        self.memory = MemoryStream()
        self.npc.memory_log = []
        self.npc.dynamic.short_term_memory = []
        self._turn_count = 0
        self.last_policy_action = None
        self.last_state_features = None
