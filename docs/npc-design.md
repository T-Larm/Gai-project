# NPC 系统设计文档

> 给项目成员的架构说明。代码均可在 `backend/` 找到，读完这份文档应该能看懂任何一个模块的职责。
> 英文报告写作时本文档的结构可直接映射到 Method 章节。

## 总体思路

一条"**最小种子 → 自动生成三层人格 → 运行时动态演化**"的流水线：

```
seed(几行JSON) ──离线,3次LLM调用──> 三层persona JSON
                                       │
                            运行时每轮拼装 system prompt
                            (persona + 记忆检索top-5 + 规则)
                                       │
玩家语音 → Whisper STT → 文本 ────────> LLM(llama3) → 回复文本 → XTTS v2克隆音色 → 播放
                                       │
                          记忆流写入 / 每4轮dynamic层更新 / 退出时落盘
```

核心主张（对应 proposal 的 persona scalability）：**几百个 NPC 只需写几百份几行的种子，不用手写几百篇人设散文**。

## 1. Minimal Seed（唯一的人工输入）

`data/seeds/example_seeds.json`，每个 NPC 一条：

```json
{
  "name": "Aldric",
  "occupation": "Blacksmith",
  "personality_tags": ["gruff", "honest", "proud", "hardworking"],
  "relationships": {
    "Mira": "childhood friend, now the town healer — trusts her completely",
    "Lord Vane": "distrusts him, believes the lord overtaxes the common folk"
  },
  "extra": {
    "location": "The Iron Hearth forge, market district",
    "secret": "Once forged a blade for an assassin unknowingly; haunted by it"
  }
}
```

目前有 3 个 NPC：Aldric（铁匠）、Mira（治疗师）、Lord Vane（腐败贵族），关系互相交叉。

## 2. 离线生成：三层 Persona

`backend/llm/persona/generator.py` 的 `PersonaGenerator.generate(seed)`，3 次 LLM 调用依次生成三层，结果存 `data/personas/{name}.json`：

| 层 | 字段 | 变化频率 | 生成方式 |
|----|------|---------|---------|
| **Core** 核心层 | backstory、values、speech_style（含口头禅）、knowledge_domains | 生成一次，不变 | LLM 从 seed 扩写 |
| **Social** 社交层 | faction、reputation、relationships（LLM 把 seed 里一句话的关系扩写成丰富描述） | 很少变 | LLM 从 seed 扩写 |
| **Dynamic** 动态层 | current_goal、emotional_state、short_term_memory | **每次对话演化** | LLM 生成初始值 |

数据类定义在 `backend/llm/persona/models.py`（`PersonaSeed` / `CorePersona` / `SocialPersona` / `DynamicSituation` / `NPC`）。

LLM 输出的 JSON 解析有三层容错（`backend/llm/json_utils.py`）：剥 markdown 围栏 → 正则提取 `{...}` → 让 LLM 修复自己的输出；仍失败抛含原文的 `ValueError`。嵌套 dict/list 用 `coerce_str` 压平成字符串。

生成命令：
```bash
python -m backend.main --seed data/seeds/example_seeds.json --name Aldric
```

## 3. 运行时：每轮对话的 prompt 组装

`backend/llm/dialogue.py` 的 `DialogueHandler`。每次玩家说话，system prompt 现场拼装：

```
## Who you are      ← Core 层
## Your world       ← Social 层
## Your current state ← Dynamic 层（goal + emotion，随对话变化）
## What you remember  ← 记忆检索 top-5（见下）
Rules: 不出戏 / 2-4句 / 体现speech_style / 知识域外承认不懂
```

**关键设计：对话历史滑动窗口**。发给 LLM 的 history 只保留最近 16 条消息（`HISTORY_MAX_MESSAGES`），更早的内容只能靠记忆检索找回。这使记忆系统真正承担长期记忆职责（也是消融实验 `no_memory` 条件能成立的前提）。

## 4. 记忆系统（Phase 2 核心）

`backend/llm/persona/memory.py` 的 `MemoryStream`，参考 Generative Agents (Park et al., 2023)：

- 每轮写入两条：玩家说的话（importance 0.4）、NPC 的回复（importance 0.5）
- 检索时三因子加权排序（权重在 `settings.IMPORTANCE_WEIGHTS`）：
  - **semantic 0.4**：sentence-transformers（all-MiniLM-L6-v2）余弦相似度
  - **recency 0.4**：指数衰减，半衰期 5 分钟
  - **importance 0.2**：写入时的重要度
- 每条记忆的 embedding 只编码一次（缓存在 entry 上）
- 上限 200 条，超出丢最旧的

## 5. 对话中的演化（Dynamic 层"活着"的证据)

- **每 4 轮**（`DYNAMIC_UPDATE_EVERY`）让 LLM 根据最近对话重估 `current_goal` 和 `emotional_state`。实测：4 轮"强盗抢了铁料"对话后，Aldric 的目标从 "Maintain current workload" 自动变为 "Retrieve stolen iron"，情绪从 Contentment 变为 Determination。解析失败则保持原状态（不会因一次坏输出崩溃）。
- **跨会话持久化**：退出时（CLI 的 finally / server 每轮）把记忆流序列化进 persona JSON 的 `memory_log` 字段，下次加载还原。实测：新会话中 Aldric 记得上次玩家的名字和订单。

## 6. 声音身份（Phase 3）

每个 NPC 按命名约定对应一个参考音色 `data/voices/{name小写下划线}.wav`（与 persona JSON 同一命名规则）。`backend/tts/xtts_client.py` 用 Coqui XTTS v2 零样本克隆该音色朗读回复。当前是 pyttsx3 生成的占位音（`python -m scripts.generate_placeholder_voices`），之后换成 VCTK 真人片段只需替换 wav 文件，代码零改动。

## 7. 对外接口

**CLI**（调试/用户研究）：
```bash
python -m backend.main --npc aldric --text            # 文字模式
python -m backend.main --npc aldric --text --speak    # + 语音输出
python -m backend.main --npc aldric                   # 麦克风输入
```

**HTTP server**（Unity 用，`backend/server.py`）：
```bash
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```
- `GET /npc/{name}` — persona 摘要 + 当前状态
- `POST /chat` `{npc, text, speak?}` — 回复文本，`speak=true` 附 base64 WAV
- `POST /transcribe` — WAV 上传 → Whisper 转文本

## 8. 实验条件开关（评估用）

`DialogueHandler` 构造参数即实验条件（详见 `evaluation/README.md`）：

| 条件 | 参数 | 含义 |
|------|------|------|
| full | 默认 | 完整系统 |
| no_memory | `use_memory=False` | 关记忆检索（RQ3 消融） |
| flat | `prompt_style="flat"` | 同样事实压成无结构单段（RQ1） |
| none | `prompt_style="none"` | 只给名字+职业（RQ1 下界） |
| handwritten | `system_prompt_text=...` | 人工手写 persona（RQ2 对照） |

## 9. 代码地图

```
backend/
├── main.py                     CLI 入口（--npc/--seed/--text/--speak/--no-memory/--prompt-style）
├── server.py                   FastAPI（Unity 桥接）
├── audio_utils.py              WAV 编解码
├── config/settings.py          所有常量（模型名/权重/窗口大小/路径）
├── stt/whisper_stt.py          麦克风录音 + Whisper
├── tts/xtts_client.py          XTTS v2 合成/播放
└── llm/
    ├── ollama_client.py        Ollama 封装（chat/generate）
    ├── dialogue.py             DialogueHandler：prompt组装/窗口/记忆写入/dynamic更新/TTS
    ├── json_utils.py           LLM-JSON 容错解析 + coerce_str
    └── persona/
        ├── models.py           三层 persona 数据类
        ├── generator.py        seed→persona 生成 + save/load
        └── memory.py           MemoryStream 三因子检索

evaluation/                     实验脚手架（run_dialogues/judge_consistency/measure_latency）
scripts/                        占位语音生成
tests/                          69 个 pytest（不依赖 Ollama/XTTS，stub 隔离）
unity/                          （Phase 4b 待做）
```

## 已知设计局限（报告 limitations 素材）

- Social 层只在 persona 生成时被展开，运行时整体注入 prompt，没有按对话对象做关系检索（多 NPC 场景可改进）
- Dynamic 更新每 4 轮一次是固定节奏，不是事件驱动
- 记忆没有做 reflection/summarization（Park et al. 有），长对话记忆条目会碎
- judge 与被测同为 llama3，有 self-preference bias，报告要说明
