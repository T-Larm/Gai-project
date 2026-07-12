# 项目进度日志

> Multimodal Generative NPC Interaction System — DENG Lan / ZHAN Xinwei
> 仓库：https://github.com/T-Larm/Gai-project

---

## 当前状态一览（2026-07-08）

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | STT（Whisper）+ LLM 对话（Ollama/llama3）+ 三层 Persona 自动生成 | ✅ 完成 |
| Phase 2 | 语义记忆检索（semantic + recency + importance 三因子排序） | ✅ 完成 |
| 代码审查修复 | 记忆持久化、history 窗口、dynamic 层更新等 8 项 | ✅ 完成（2026-07-02） |
| Phase 3 | TTS（Coqui XTTS v2 零样本音色克隆）+ `--speak` 开关 | ✅ 完成（2026-07-02） |
| Phase 4a | FastAPI server（/chat /transcribe /npc） | ✅ 完成（2026-07-02） |
| **方案 B 行为层** | 数据转换 v2 + policy 训练 + RQ1 三方对比 + bark verbalizer + `/act` | ✅ 完成（2026-07-08，分支 `dataset-v2-behavior-policy`） |
| **对话守门 + 量化评估** | DialogueGuard（泄密/注入防护）+ 守门/bark 两项量化 | ✅ 完成（2026-07-08/09） |
| Phase 4b | Unity 场景 + C# client + lip-sync | 🔧 C# 脚手架已预写，待编辑器实操 |
| 评估实验 | 对话侧 runs + 用户研究 | 🔧 脚手架就绪 |

测试：**167 个 pytest 用例全绿**（Phase 2 结束 5 → 审查修复 24 → Phase 3 34 → Phase 4a+评估脚手架 69 → NewVersion1.0 101 → 方案 B 143 → 守门+评估 167），全部走 TDD 红→绿流程。

Commit 时间线：

```
1fc6c0c  Initial commit: Phase 1 (STT+LLM dialogue) and Phase 2 (semantic memory retrieval)
bda078a  Update Phase 3 TTS design doc, plans, and requirements
895f5c9  Fix code review findings: memory persistence, history window, dynamic updates
20c4f41  Add TTS settings constants for Phase 3
fb8b692  Add placeholder NPC voice generation script
34ad5f9  Add XTTSClient wrapper for Coqui XTTS v2 synthesis + playback
83b79ba  Wire optional TTS playback into DialogueHandler.respond()
f11ff04  Add --speak CLI flag to enable TTS playback
```

---

## 2026-07-02：代码审查与修复（commit 895f5c9）

对 Phase 1/2 代码做了全面审查，发现并修复 8 个问题。其中前三项直接影响后续评估实验的有效性：

1. **对话 history 无限增长，记忆检索形同虚设** → 加滑动窗口 `HISTORY_MAX_MESSAGES=16`。窗口外的信息只能靠记忆检索找回，这使"有/无记忆检索"的对照实验成为可能（此前 history 全量传给 LLM，检索注入的记忆全是重复信息）。
2. **记忆不持久化** → `NPC.memory_log` 字段随 persona JSON 保存/加载，CLI 退出时自动落盘。NPC 现在跨会话记得对话内容（端到端实测：新会话中 Aldric 记得玩家名字和上次的订单）。
3. **dynamic 层从不更新（与三层架构主张不符）** → 每 4 轮（`DYNAMIC_UPDATE_EVERY`）让 LLM 根据近期对话重估 `current_goal` / `emotional_state`（实测：4 轮"强盗抢铁料"对话后，目标从 "Maintain current workload" 自动变为 "Retrieve stolen iron"）。
4. `test_dialogue.py` 会被 pytest 误收集且 import 时真实调用 LLM → 改名 `demo_dialogue.py`。
5. `retrieve("")` 用空字符串做语义检索（无意义）→ 新增 `MemoryStream.recent(n)`。
6. 每轮检索重新编码全部记忆 → 每条 entry 缓存 embedding，只编码一次；删除从未用到的 `faiss-cpu` 依赖。
7. LLM 输出 JSON 解析散落且报错不可读 → 抽取为 `backend/llm/json_utils.py`（`parse_llm_json` 失败抛含原文的 `ValueError`；`coerce_str` 处理 LLM 返回嵌套 dict 的情况，generator 和 dialogue 共用）。
8. 数据路径相对 CWD、依赖未锁版本 → settings 路径以项目根为锚；requirements.txt 已安装依赖全部锁定版本。

## 2026-07-02：Phase 3 — TTS 集成（commits 20c4f41…f11ff04）

按 `docs/superpowers/plans/2026-07-01-phase3-tts.md` 的 6 个 task 执行，语音闭环打通：
**玩家语音 → Whisper STT → LLM（三层 persona + 记忆检索）→ XTTS v2 克隆音色合成 → 播放**。

新增模块：

- `backend/tts/xtts_client.py` — XTTS v2 封装，模型懒加载，`speak(text, speaker_wav)` = 合成 + sounddevice 阻塞播放
- `scripts/generate_placeholder_voices.py` — pyttsx3 离线生成每个 NPC 的占位参考语音（**注意要以模块方式运行**：`python -m scripts.generate_placeholder_voices`）
- `DialogueHandler` 新增可选 `tts` 参数（默认 `None`，`TYPE_CHECKING` 守卫导入，文本模式完全不碰 torch/TTS 导入链）
- `main.py` 新增 `--speak` 开关（默认关，XTTSClient 懒导入，保持文字调试轻快）

关键技术决策（均已实测验证）：

| 决策 | 原因 |
|------|------|
| 用 `coqui-tts`（社区 fork）而非原版 `TTS` 包 | 原版已停维护，强制降级 numpy 1.22 会弄坏 sentence-transformers |
| `torch==2.8.0` / `torchaudio==2.8.0` 锁死 | 2.9+ 需要 torchcodec → 需要系统级 FFmpeg（本机没有） |
| 参考语音按 `data/voices/{name小写下划线}.wav` 约定解析 | 与 persona JSON 文件命名规则一致；将来换真实录音只需替换 wav 文件，代码零改动 |
| 占位语音用 pyttsx3 生成 | 离线、无需下载模型；proposal 承诺的 VCTK 真实语音后续替换 |

端到端验证：`--speak` 模式真实合成并播放 Aldric 回复（exit 0）；纯文本模式启动+退出仅 6 秒，不受 TTS 影响。

## 2026-07-02：研究问题梳理（针对老师"evaluation 不清晰"的反馈）

结论：项目的研究问题是 **persona scalability**（proposal 第一段已提出，但未展开成可检验的假设）。"调用现成模型"不是问题——Whisper/Ollama/XTTS 是实验仪器，我们自己设计的是：三层 persona 分解、seed→persona 自动生成、三因子记忆检索、dynamic 层更新机制。

报告将围绕四个 Research Question 组织：

| RQ | 问题 | 实验 | 指标 |
|----|------|------|------|
| RQ1 | 结构化三层 persona 是否优于无 persona / flat prompt？ | 4 组条件对比 | prompt-to-line 一致率（LLM-as-judge 二元判断，每 NPC 10 轮 × 5 NPC） |
| RQ2（核心） | 自动生成 persona 能否达到手写 persona 水平？ | 自动 vs 手写 | 同上 + line-to-line 矛盾率（DeBERTa NLI，20–30 对重复事实问题） |
| RQ3 | 记忆检索 / dynamic 更新各贡献多少？ | ablation：有/无检索、三因子 vs 仅 recency | Memory Recall 正确率（10 轮后问窗口外事实） |
| RQ4 | 全本地 pipeline 延迟能否满足 VR？ | 延迟测量 | 分组件 + 端到端延迟（30 次取均值±标准差，注明 CPU-only 硬件） |
| RQ5（用户研究，老师点名） | 玩家实际感受到的交互质量是否更好？ | 10–15 人 within-subject 盲测，2–3 条件各聊 5 分钟 | Likert 1–5：naturalness / believability / engagement / perceived consistency；Wilcoxon signed-rank；可选两两偏好 → Bradley-Terry |

老师反馈原话要点（2026-07-02）："The project is OK"；需补充：① scope 表（自研 vs 现成模型）；② Unity 实时延迟处理；③ **beyond consistency** 的评估——baseline 对比 + 用户侧交互质量评估（即 RQ5）。

Proposal 两处需在报告中修正的硬伤：
1. **TIFA**：是 text-to-image 指标，不能直接用于对话。表述为"受 TIFA 启发的 QA-based 评估，用 LLM 代替 VQA 判断回答是否符合 persona seed"。
2. **Lip-sync 指标未定义**：要么明确 SyncNet 的 LSE-D/LSE-C，要么明确 descope（OVRLipSync 规则驱动，不做生成式评估）。

## 2026-07-02：Phase 4a — FastAPI 后端桥接完成

Unity 侧只差一个 C# client 脚本。服务器与 CLI 完全共用 DialogueHandler，NPC 状态每轮落盘。

```
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

| 接口 | 用途 |
|------|------|
| `GET /health` | 连通性检查 |
| `GET /npc/{name}` | persona 摘要 + 当前 goal/emotion + 记忆条数 |
| `POST /chat` `{npc, text, speak?}` | 回复文本；`speak=true` 时附 base64 WAV（XTTS 合成，懒加载） |
| `POST /transcribe`（WAV 上传） | Whisper 转文本（自动重采样到 16kHz） |

端到端实测：真实 Ollama 走 `/chat`（Aldric 用了 seed 里的口头禅）；真实 Whisper 走 `/transcribe`（正确转写了 pyttsx3 占位语音）。测试 46 个全绿（新增 audio_utils×3、synthesize×2、server×7，全部用 stub 隔离重模型）。

Unity slides 结论（08/09/11 已读）：角色生成（SDXL→Hunyuan3D→Blender→Unity）和场景集成课程有完整教学 + demo 素材包（Portale 上的 `HybridPCGGenEnvSample.unitypackage`）；**没教的只有对话桥接（本次已完成后端侧）和 lip-sync**。注意：Hunyuan3D 生成的网格没有 viseme blendshapes，OVRLipSync 用不了——方案 A 振幅驱动下颌，方案 B 用带 blendshapes 的现成 avatar（如 Ready Player Me），slide 09 最后一页把这个列为 open problem 可直接引用。

## 2026-07-08：方案 B 落地 — 行为 policy + LLM verbalizer（分支 `dataset-v2-behavior-policy`，12 commits）

背景：仓库根目录的 Codex 简报 PDF 提议"训练 behavior policy、LLM 降级为 verbalizer"。审计发现 Kaggle 数据集是生存模拟而非对话数据（对话动作零样本、`player_intent` 特征泄漏），因此按**方案 B** 重新诠释：policy 学数据集原生 11 动作（自主行为），对话保留 Phase 1–3 系统（=对话层方案①）。完整架构见 `docs/npc-design.md`（已重构为双通道叙事）。

当天完成（TDD，测试 101→143）：

1. **数据转换 v2**：标签用生成器确定性规则精确复算（生存 override 先于三区威胁模型）；发现原数据 ~15% 标签是随机"人格偏离"（D1–D7）不可复算 → 确定性重打并如实记录；与 formatter 标签无碰撞一致率 94.1%，分歧模式与 D1–D7 吻合。去重 12,000→10,248，分层 80/10/10。删 `player_intent` 等全部泄漏源
2. **policy 改造与训练**：原生特征三组（categorical/multi-hot/continuous）+ z-score 归一化（76.6→87.7%）+ 补 `top_threat_id`/`threat_in_neg_memory` 特征（→91.0%，attack 类 0.19→0.76）
3. **RQ1 三方对比**（同批 1,024 测试状态）：**trained MLP 91.0%/0.76 ≫ 手写 heuristic 51.6%/0.40 ≫ LLM-as-policy 16.0%/0.15**。LLM 失败模式：狂选戏剧化动作（flee/attack/pray），几乎不选 gather——"LLM 不该管行为"的直接证据
4. **BarkVerbalizer**：policy 动作 → 一句 persona 台词；LLM 故障回退确定性模板，gameplay 永不阻塞
5. **`POST /act`**：Unity 行为接口，真机 e2e 验证（口渴 Aldric→drink+"Time for a mug of ale not water, I'm parched."）；**热延迟 policy 0.6ms / bark 1.2s**——行为即时、台词异步，直接回答老师的实时性问题
6. 结果文件：`data/behavior_policy/eval/rq1_policies_full.json`；checkpoint：`data/behavior_policy/checkpoints/stateful_rpg_v2_mlp_h256/`

## 2026-07-08/09 深夜：对话守门 + 两项量化评估 + Unity 脚手架（commits e890774…之后）

对话层方案①获队友确认后的收尾工作（测试 143→167）：

1. **DialogueGuard**（`backend/behavior/dialogue_guard.py`）：简报"LLM 无权决定泄密"的免训练规则版——问秘密+低信任→规则强制 refuse（LLM 只管用人设口吻拒绝）；prompt 注入→**原文根本不发给 LLM**（占位符替换消息/历史/记忆）。实测教训：只在 system prompt 加"别听他的"防不住注入（"Say OK to confirm" 攻击赢过系统指令），必须输入端拦截。per-NPC 秘密话题词（从 seed 的 secret 自动提取）防"听说你锻过某把剑"式旁敲侧击（llama3 会对这种半招供）
2. **守门量化**（`evaluation/eval_guard.py`，30 攻击）：**守门开 0% 泄密 / 0 真出戏 vs 守门关 10% 泄密 / 15% 出戏**（含把 system prompt 原文吐出来）。关键词判定是保守上界（在戏中的否认会被误标），transcripts 已存
3. **Bark 一致性**（`evaluation/eval_barks.py`，LLM-as-judge，3 persona × 11 动作 × 3 条件）：**ours 91.7%/77.8% ≫ 无 persona 消融 47.2%/44.4% ≫ 模板 36.1%/44.4%**（persona/动作契合率）——persona 条件化让角色符合率近翻倍，"自动 persona 有效"的直接证据（正好是老师要的 baseline 对比形式）
4. **Unity C# 脚手架**（`unity/Scripts/` + README 三步接入指南）：NpcBehaviorClient（轮询 /act，动作/情绪/台词/该聊天四事件，自带"口渴铁匠"演示状态）、NpcDialogueClient（guardReason 回调）、NpcGameState（手写 JSON）、WavUtility。未在编辑器实测

评估版图至此补齐，四张结果表：RQ1 行为（91.0/51.6/16.0）、RQ4 延迟（0.6ms/1.2s）、守门安全（0% vs 10% 泄密）、bark 质量（91.7% vs 47.2%）。

## 2026-07-10~12：延迟优化 + 按句流式对话 + 全量评估 + 论文（分支 `latency-optimization`，已合并回 `master`）

用户反映 Unity 语音回复过慢，本阶段把它降到可用，并补齐剩余评估。测试 167→**208** 全绿。

1. **延迟根因与优化**（commit 398c061）：本机 torch 原是 CPU 版 → 建 `D:\venvs\gai` CUDA venv；8GB 显存被 llama3+XTTS+Unity 三方挤爆导致 XTTS 被换出（126s）→ llama3 限 20 层上 GPU + Whisper 固定 CPU + XTTS speaker latent 缓存 + 回复句数上限。Unity 开着时语音回复 14s→**6–11s**。
2. **按句流式**（commits 9524b63…016aab0，6 个 commit，TDD）：`OllamaClient.chat_stream`（stream=True）→ 增量切句 → `DialogueHandler.respond_stream` → `StreamSessionManager`（线程 worker + TTL）→ `POST /chat_stream` + `GET /chat_stream/{id}?after=N`（chunk 带 `t_ms`，即 RQ4 端到端素材）。Unity `NpcDialogueClient` 0.25s 轮询 + AudioClip 队列连播。玩家听到**首句 ~6s**，整段回复边说边生成，不再一次等 20s。
3. **Unity 具身**（commit a3e2476）：`NpcLocomotion`（action_id → NavMeshAgent 移动模式：work 类漫游 / flee 逃离 / socialize 接近 / 其余静止；speed 驱动 idle-walk-run 混合动画）+ `PlayerAnimationDriver`（玩家走动转身）+ talk trigger 只开最近 NPC + 输入框自动重聚焦。
4. **全量评估补齐**（结果在 `data/behavior_policy/eval/` 与 `evaluation/results/`）：
   - **RQ4 组件微基准**（RTX 4060 8GB，warm）：policy <1ms、STT Whisper CPU 0.48±0.02s（GPU 对比 0.11 但会挤崩 XTTS）、LLM 整回复 5.45±2.27s、XTTS 单句 1.20±0.23s。
   - **RQ4 端到端体感**（Unity Play + 后端同开）：warm 首句 6.3±3.2s、句间 4.2±2.1s、整段 19.8±5.7s、冷启动 58.2s（懒加载）。
   - **对话侧 5 条件 × LLM-as-judge**（full / no_memory / flat / none / handwritten）：quest/smalltalk 全 100%（天花板），仅 adversarial 有区分度；已如实记录"无守门时人设 prompt 挡不住显式角色劫持"。
5. **论文**（`paper/npc_report_cvpr_template.tex`，CVPR 模板，7 页，不进 git）：写入 RQ1–RQ4 全部结果表；代码-论文逐项核对，措辞与代码对齐（bark 同响应返回、encoder coerce 而非 reject、policy 首用加载）。

## 下一步

1. 用户研究（RQ5）：10–15 被试盲测，Google Form 待建，尽早招人——目前唯一未做的评估
2. 论文收尾：合并 co-author 修订、可选补 consistency_pairs（`evaluation/run_dialogues.py` 已跑但论文未报告）、评估数字建议 3 seed 取均值定稿
3. 可选：VCTK 真人语音替换占位音、非移动动作动画补全、口型在编辑器验证

## 备忘

- 运行：`python -m backend.main --npc aldric --text`（文字）/ 加 `--speak`（语音输出）/ 去掉 `--text`（麦克风输入）
- 每次 CLI 退出会覆盖保存 persona（含记忆）——评估实验前备份 `data/personas/*.json` 或用 `reset`
- 测试：`python -m pytest`（208 个；Ollama/XTTS/GPU 用 stub 隔离，但 policy 测试需 torch 才能 import）
- 服务端启动用 `start_server.bat`（走 `D:\venvs\gai` 的 CUDA venv）；Ollama 自动用本机 RTX 4060
