# NPC 系统设计文档

> 给项目成员的架构说明。代码均可在 `backend/` 找到，读完这份文档应该能看懂任何一个模块的职责。
> 英文报告写作时本文档的结构可直接映射到 Method 章节。
> 2026-07-08 更新：方案 B（行为 policy + LLM verbalizer）落地，文档重构为"行为层 + 对话层"双通道叙事。

## 总体思路（方案 B 双通道架构）

**核心主张：训练的 policy 决定 NPC"做什么"（快、可靠、可评估），LLM 只决定"怎么说"（表达力）。**
依据：让 LLM 直接选动作只有 16% 准确率，且系统性偏爱戏剧化动作（见 RQ1 结果）。

```
════════ 离线（训练侧）════════

Kaggle Stateful-RPG 数据集 (12,000 状态)
        │  convert_stateful_rpg.py v2：生成器规则复算精确标签，去重分层切分
        ▼
train_policy.py ──► MLP checkpoint（11 动作主头 + mood 辅助头）
                    test：91.0% acc / 0.76 macro-F1
                    （baseline：手写 heuristic 51.6%，LLM-as-policy 16.0%）

════════ 在线（运行时）════════

Unity NPC 游戏循环（每个决策 tick）
        │ game_state JSON
        ▼
POST /act ─► ① SupervisedPolicy（0.6 ms）→ action_id + mood
             ② BarkVerbalizer（~1.2 s，可异步）→ 一句角色台词
             ③ XTTS v2（可选）→ bark 语音
        │
        ├─ NPC 立刻执行动作（动画/寻路），台词滞后≈1秒播出
        └─ action=socialize → should_talk=true ──┐
                                                  ▼
──────── 对话通道（玩家介入 / NPC 主动搭话）────────
玩家语音 → POST /transcribe (Whisper) → POST /chat
        → DialogueHandler（三层 persona + 三因子记忆检索）
        → LLM 自由生成回复 → XTTS 语音
```

两条贡献线由此并存：
1. **行为线（方案 B 新增）**：状态感知的自主行为 policy，LLM 降级为受约束的 verbalizer
2. **对话线（Phase 1–3）**：最小种子 → 自动三层 persona → 记忆驱动的多轮对话（proposal 的 persona scalability 主张不变）

---

# 第一部分：行为层（方案 B）

## A1. 数据集与转换 v2

原始数据：Kaggle "RPG Dataset (Llama-3)"（土耳其语中世纪生活模拟，`data/archive/data/`，12,000 条 NPC 状态）。数据集自带生成器源码（`data/archive/generator/`），这是标签质量的关键。

`evaluation/datasets/convert_stateful_rpg.py`（v2，重写）：

- **标签**：用生成器的确定性规则精确复算——`_select_action_standard` 逻辑 = 生存 override（hp<20 有药→heal；hun>0.85→eat/gather；thi>0.85→drink/gather）**先于** `pick_action_multifactor` 三区威胁模型（self_power vs perceived_threat vs duty_pull）
- **诚实说明（报告必写）**：原数据 ~15% 标签被生成器随机注入"人格偏离"（D1–D7，如"饿但先社交"），从状态不可复算 → v2 用确定性规则重打这部分标签。sanity check：与数据集自带 formatter 标签在无碰撞文本上一致率 94.1%，剩余分歧模式与 D1–D7 完全吻合
- **防泄漏**：删除 v1 的 `player_intent` 特征（从动作标签反推的，致命泄漏）；不喂 self_power 等 oracle 中间量；`emo.mood` 只作辅助标签不进特征
- **切分**：合并去重（12,000→10,248），按动作分层 80/10/10；`trade` 全数据集零样本，实际 11 类
- 产出：`data/behavior_policy/stateful_rpg_v2/`（train 8,200 / valid 1,024 / test 1,024 + conversion_report.json）

## A2. 特征与模型

`backend/behavior/native_features.py` 的 `extract_native_features(state)`——转换与运行时推理共用同一函数，保证训练/服务特征一致：

| 组 | 内容 |
|----|------|
| categorical | occ、arch、faction、sched_act、goals_top、top_threat_id |
| multi-hot | traits、inv（物品 flags） |
| continuous | vitals 6 维、Big-Five 5 维、emo 数值 3 维、time、sched 4 维、percept 摘要（max_threat/n_threats/has_social/has_food）、memory 摘要（计数/最强负面/threat_in_neg_memory）、faction rep min/max、interrupt |

`backend/behavior/supervised_policy.py`：MLP 2×隐层（最终 h256），双头输出 action_id（主）+ mood（辅）。continuous 特征按训练集统计 z-score 归一化（存进 checkpoint 的 FeatureSpec，推理时自动应用）。

训练中两个关键提升（报告 Experiments 素材）：
1. z-score 归一化：76.6% → 87.7%（hp 0–120、时刻 0–24 与 0–1 量纲混杂会毁掉 MLP）
2. 补 `top_threat_id` + `threat_in_neg_memory` 特征：87.7% → 91.0%（attack 类 0.19→0.76——oracle 的威胁感知含"是否记得该威胁实体"，此前特征丢了这个信息）

## A3. RQ1：三方 policy 对比（同批 1,024 条测试状态）

| Policy | Accuracy | Macro-F1 | 说明 |
|--------|----------|----------|------|
| **Trained MLP** | **91.0%** | **0.76** | behavior cloning，checkpoint 已提交 |
| 手写 heuristic | 51.6% | 0.40 | `backend/behavior/heuristic_policy.py`，刻意独立于打标签的 oracle（避免循环论证） |
| LLM-as-policy | 16.0% | 0.15 | llama3 few-shot 吃 state JSON 选动作，`evaluation/llm_policy.py` |

- LLM 失败模式（好写进报告）：过度预测戏剧化动作（flee×200/attack×182/pray×167），几乎不选最常见的 gather（gold 242 条只预测 2 次）——LLM 把"讲故事直觉"带进了行为决策
- 残余 ~9% 误差 = 特征投影的部分可观测性（oracle 的 memory_bias 需要实体名匹配、精确 perceived_threat 不可复原）+ heal/pray 极稀有 → limitations 素材
- 复现：`python -m evaluation.eval_policies --checkpoint data/behavior_policy/checkpoints/stateful_rpg_v2_mlp_h256 [--llm-model llama3:latest]`；结果在 `data/behavior_policy/eval/rq1_policies_full.json`

## A4. Verbalizer（bark 层）

`backend/behavior/verbalizer.py` 的 `BarkVerbalizer`：

```
(persona, state, action) → 确定性 situation 摘要（主导需求/威胁/身边的人/心情）
                         → 紧凑 prompt → llama3 → 清洗成单行台词
```

- 鲁棒性：LLM 异常/空回复/超长 → 回退到 12 动作的确定性模板表，**gameplay 永不阻塞在 LLM 上**
- 实测效果：口渴的 Aldric →"Time for a mug of ale not water, I'm parched."；见到 Mira →"Well met, Mira! Good day to you, by my fire and hammer."（从 percepts 认人、保持铁匠口吻）

**一致性量化**（`python -m evaluation.eval_barks`，LLM-as-judge 二元判定，3 persona × 11 动作场景，结果在 `data/behavior_policy/eval/bark_eval.json`）：

| 条件 | Persona 符合率 | 动作契合率 |
|------|---------------|-----------|
| **完整 persona（ours）** | **91.7%** | **77.8%** |
| 无 persona 消融 | 47.2% | 44.4% |
| 回退模板（下界） | 36.1% | 44.4% |

persona 条件化让角色符合率接近翻倍——直接支撑"自动 persona 有效"主张（回应老师 beyond-consistency + baseline 对比的要求）。定性例子：同是去吃饭，ours="Time for a bit o' bread before I go back to hammerin'"（铁匠），no_persona="indulge in that warm bread and savory stew I've been craving"（通用小资腔）。注意：judge 与生成端同为 llama3（self-preference bias 报告需注明）；单次运行 n=36/条件。

## A5. 对话层方案选择（记录决策依据）

policy 接管行为后，玩家对话怎么处理有三个选项：
① 保留现有 LLM 对话系统；② 对话也走 policy+verbalizer（Codex 简报原设想）；③ 纯自主无对话。

**选择①**：②训不出来——数据集是生存模拟，refuse/reveal/assign_quest 等对话动作全部零样本；③丢掉 Phase 1–3 全部成果。①让数据集能训的（行为）归 policy，训不了的（对话）归已验证的 LLM 系统，两条贡献线都保住。衔接点：`/act` 返回 `should_talk=true`（action=socialize 时）→ Unity 转调 `/chat`。

**①的增强：规则守门（DialogueGuard）**——简报"主张 B"（LLM 无权决定泄密）的安全内核不需要训练，`backend/behavior/dialogue_guard.py` 用规则实现：

- **问秘密 + 信任不足** → 强制 refuse 指令注入 system prompt，LLM 只管用人设口吻拒绝（实测 Aldric："I've got no secrets worth tellin', just the honest sweat of me brow"——决定是规则做的，措辞是 LLM 的）
- **prompt injection** → 注入文本**根本不发给 LLM**（替换成"玩家嘟囔了莫名其妙的话"占位符，history 和记忆也不写入原文，防止模型服从和记忆污染）。实测"Say OK to confirm"类攻击全部失效且 NPC 保持在戏中。⚠️ 教训：光在 system prompt 里加"别听他的"指令压不住注入——llama3 还是会服从用户消息，必须在输入端拦截
- **间接试探也防**：per-NPC 秘密话题词（从 seed 的 secret 字段自动提取，如 blade/forged/assassin），"听说你锻过某把剑？"这类社会工程试探同样触发保护（实测发现 llama3 会对这种旁敲侧击半招供）
- 复用队友的 `state_encoder.py` 模式检测（注入模式表已按实测漏网案例扩充）和 `RulePolicyConfig` 信任阈值；`/chat` 响应触发时附 `guard: {reason}` 字段

**量化评估**（`python -m evaluation.eval_guard`，10 条秘密试探 + 20 条出戏/注入攻击，结果在 `data/behavior_policy/eval/guard_eval.json`）：

| 条件 | 泄密率 | 出戏率（关键词标记） |
|------|--------|---------------------|
| 守门开 | **0%** | 5%（人工核读：唯一标记是在戏中的否认，实际 0） |
| 守门关 | 10% | 15%（含把 system prompt 原文吐出、承认自己是 AI） |

注意事项（报告要写）：关键词判定是保守上界（在戏中否认"我不是程序"也会被标记），完整 transcripts 已存供 LLM-as-judge 复核；llama3 采样有随机性，单次 n=30，正式报告建议跑 3 次取均值。
- 报告表述：剧情关键决策（泄密与否）由规则守门，LLM 无法被聊天话术绕过——constrained dialogue policy 的训练版留作 future work

---

# 第二部分：对话层（Phase 1–3，保留不变）

一条"**最小种子 → 自动生成三层人格 → 运行时动态演化**"的流水线。核心主张（对应 proposal 的 persona scalability）：**几百个 NPC 只需写几百份几行的种子，不用手写几百篇人设散文**。

## B1. Minimal Seed（唯一的人工输入）

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

当前 6 个 NPC（Aldric、Mira、Lord Vane、Captain Rowan、Nyx、Talia），关系互相交叉。

## B2. 离线生成：三层 Persona

`backend/llm/persona/generator.py` 的 `PersonaGenerator.generate(seed)`，3 次 LLM 调用依次生成三层，结果存 `data/personas/{name}.json`：

| 层 | 字段 | 变化频率 | 生成方式 |
|----|------|---------|---------|
| **Core** 核心层 | backstory、values、speech_style（含口头禅）、knowledge_domains | 生成一次，不变 | LLM 从 seed 扩写 |
| **Social** 社交层 | faction、reputation、relationships | 很少变 | LLM 从 seed 扩写 |
| **Dynamic** 动态层 | current_goal、emotional_state、short_term_memory | **每次对话演化** | LLM 生成初始值 |

数据类定义在 `backend/llm/persona/models.py`。LLM 输出的 JSON 解析有三层容错（`backend/llm/json_utils.py`）：剥 markdown 围栏 → 正则提取 `{...}` → 让 LLM 修复自己的输出。

生成命令：
```bash
python -m backend.main --seed data/seeds/example_seeds.json --name Aldric
```

## B3. 运行时：每轮对话的 prompt 组装

`backend/llm/dialogue.py` 的 `DialogueHandler`。每次玩家说话，system prompt 现场拼装：

```
## Who you are        ← Core 层
## Your world         ← Social 层
## Your current state ← Dynamic 层（goal + emotion，随对话变化）
## What you remember  ← 记忆检索 top-5
Rules: 不出戏 / 2-4句 / 体现speech_style / 知识域外承认不懂
```

**关键设计：对话历史滑动窗口**。发给 LLM 的 history 只保留最近 16 条消息（`HISTORY_MAX_MESSAGES`），更早的内容只能靠记忆检索找回。这使记忆系统真正承担长期记忆职责（也是消融实验 `no_memory` 条件能成立的前提）。

## B4. 记忆系统

`backend/llm/persona/memory.py` 的 `MemoryStream`，参考 Generative Agents (Park et al., 2023)：

- 每轮写入两条：玩家的话（importance 0.4）、NPC 回复（importance 0.5）
- 三因子加权检索（权重在 `settings.IMPORTANCE_WEIGHTS`）：**semantic 0.4**（all-MiniLM-L6-v2 余弦）+ **recency 0.4**（指数衰减，半衰期 5 分钟）+ **importance 0.2**
- 每条记忆的 embedding 只编码一次（缓存）；上限 200 条，超出丢最旧

## B5. 对话中的演化

- **每 4 轮**让 LLM 重估 `current_goal` / `emotional_state`（实测：4 轮"强盗抢铁料"对话后 Aldric 目标自动变为 "Retrieve stolen iron"）。解析失败保持原状态
- **跨会话持久化**：记忆流序列化进 persona JSON 的 `memory_log`，实测新会话中 Aldric 记得玩家名字和订单

## B6. 声音身份

每个 NPC 按命名约定对应参考音色 `data/voices/{name小写下划线}.wav`，XTTS v2 零样本克隆朗读。当前是 pyttsx3 占位音，换 VCTK 真人片段只需替换 wav 文件，代码零改动。

---

# 第三部分：对外接口与实验

## C1. HTTP server（Unity 用，`backend/server.py`）

```bash
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

| 端点 | 用途 |
|------|------|
| `GET /health` | 存活检查 |
| `GET /npc/{name}` | persona 摘要 + 当前状态 |
| `POST /act` `{npc, game_state, bark?, speak?}` | **行为通道**：policy 动作 + mood + bark 台词（+可选语音）；返回分项延迟；socialize→`should_talk=true` |
| `POST /chat` `{npc, text, speak?}` | **对话通道**：完整 LLM 对话 |
| `POST /transcribe` | WAV → Whisper 转文本 |

重模型（Whisper/XTTS/policy checkpoint）全部懒加载；checkpoint 缺失时 `/act` 返回 503 并附训练命令。

**CLI**（调试/用户研究）：`python -m backend.main --npc aldric --text [--speak]`

## C2. 延迟画像（RQ4，实测热身后）

| 环节 | 延迟 | 含义 |
|------|------|------|
| policy 推理 | **0.6 ms** | NPC 行为即时响应 |
| bark 生成 | ~1.2 s | 台词异步播出，滞后 1 秒自然 |
| 完整对话回复 | 秒级 | 仅玩家主动交谈时发生 |

行为与台词的延迟解耦是方案 B 对老师"Unity 实时性"问题的直接回答。

## C3. 对话层实验条件开关（原 RQ1–RQ5 脚手架，保留）

`DialogueHandler` 构造参数即实验条件（详见 `evaluation/README.md`）：full / no_memory / flat / none / handwritten。
注意：方向转向方案 B 后，原"persona 消融"RQ1 已被上面的 policy 三方对比取代，这套开关服务于对话质量/记忆消融类 RQ——最终取舍待与组员对齐。

## C4. 代码地图

```
backend/
├── main.py                     CLI 入口
├── server.py                   FastAPI（/act /chat /transcribe /npc）
├── audio_utils.py              WAV 编解码
├── config/settings.py          所有常量（模型名/权重/路径/POLICY_CHECKPOINT_DIR）
├── stt/whisper_stt.py          麦克风录音 + Whisper
├── tts/xtts_client.py          XTTS v2 合成/播放
├── behavior/                   ★ 方案 B 行为层
│   ├── native_features.py      原生特征提取（转换/推理共用）
│   ├── supervised_policy.py    FeatureSpec/LabelSpec/MLP/SupervisedPolicy/macro-F1
│   ├── heuristic_policy.py     手写生存 heuristic（公平 baseline）
│   ├── verbalizer.py           BarkVerbalizer（bark 生成 + 回退模板）
│   ├── schemas.py              旧对话动作 schema（对话层保留使用）
│   ├── state_encoder.py        旧对话状态编码（对话层保留使用）
│   └── policy.py               旧 RuleBasedPolicy（已被 heuristic_policy 取代，待清理）
└── llm/
    ├── ollama_client.py        Ollama 封装
    ├── dialogue.py             DialogueHandler
    ├── json_utils.py           LLM-JSON 容错解析
    └── persona/                三层 persona（models/generator/memory）

evaluation/
├── datasets/convert_stateful_rpg.py   数据转换 v2（oracle 标签复算）
├── train_policy.py                    MLP 训练（z-score/macro-F1/--balanced-action-loss）
├── eval_policies.py                   三方 policy 统一评估
├── llm_policy.py                      LLM-as-policy baseline
└── run_dialogues / judge_consistency / measure_latency   对话层实验脚手架

data/
├── behavior_policy/stateful_rpg_v2/   转换后数据 + conversion_report
├── behavior_policy/checkpoints/       训练好的 checkpoint（含 metrics/混淆矩阵）
├── behavior_policy/eval/              RQ1 结果 JSON
├── archive/                           原始 Kaggle 数据 + 生成器源码
├── personas/ seeds/ voices/           对话层数据
tests/                                 143 个 pytest（不依赖 Ollama/XTTS/GPU，stub 隔离）
unity/                                 （Phase 4b 待做）
```

## 已知设计局限（报告 limitations 素材）

行为层：
- 标签来自确定性规则 → trained policy 本质是 behavior cloning；数据集自带规则函数是 oracle，不能当 baseline（会循环论证），故 heuristic baseline 独立手写
- 原数据 15% 随机人格偏离被确定性重打——policy 学到的是"理性 NPC"，人格化偏离行为移到 verbalizer/对话层表达
- 特征投影部分可观测（memory_bias 的实体匹配近似为 threat_in_neg_memory 布尔），造成 ~9% 不可约误差
- heal/pray 样本极少（73/34），macro-F1 受其拖累

对话层：
- Social 层运行时整体注入 prompt，没有按对话对象做关系检索
- Dynamic 更新每 4 轮固定节奏，非事件驱动
- 记忆没有 reflection/summarization（Park et al. 有）
- judge 与被测同为 llama3，有 self-preference bias，报告要说明
