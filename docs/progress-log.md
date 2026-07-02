# 项目进度日志

> Multimodal Generative NPC Interaction System — DENG Lan / ZHAN Xinwei
> 仓库：https://github.com/T-Larm/Gai-project

---

## 当前状态一览（2026-07-02）

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | STT（Whisper）+ LLM 对话（Ollama/llama3）+ 三层 Persona 自动生成 | ✅ 完成 |
| Phase 2 | 语义记忆检索（semantic + recency + importance 三因子排序） | ✅ 完成 |
| 代码审查修复 | 记忆持久化、history 窗口、dynamic 层更新等 8 项 | ✅ 完成（2026-07-02） |
| Phase 3 | TTS（Coqui XTTS v2 零样本音色克隆）+ `--speak` 开关 | ✅ 完成（2026-07-02） |
| Phase 4 | FastAPI server + Unity 集成 + OVRLipSync | ⬜ 未开始 |
| 评估实验 | 四组 baseline 对比 + ablation | ⬜ 未开始（计划已定） |

测试：**34 个 pytest 用例全绿**（Phase 2 结束时 5 个 → 审查修复后 24 个 → Phase 3 后 34 个），全部走 TDD 红→绿流程。

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

## 下一步

1. **评估脚手架**（优先，RQ2/RQ3 最划算）：`--no-memory` / `--flat-persona` 开关、50–100 条评估对话集（quest / small talk / adversarial 各约 1/3）、LLM-as-judge 打分脚本
2. Phase 4b：Unity 场景（用课程素材包）+ C# client 脚本 + lip-sync（方案 A 保底）
3. 用 slide 09 的 pipeline 生成 Aldric 的 3D 形象（需要 ComfyUI 环境，看课程 demo 材料）
4. 占位语音替换为 VCTK 真实语音片段
5. 报告撰写（Experiments 章节按 RQ1–4 组织）

## 备忘

- 运行：`python -m backend.main --npc aldric --text`（文字）/ 加 `--speak`（语音输出）/ 去掉 `--text`（麦克风输入）
- 每次 CLI 退出会覆盖保存 persona（含记忆）——评估实验前备份 `data/personas/*.json` 或用 `reset`
- 测试：`python -m pytest`（34 个，不依赖 Ollama/XTTS，用 stub 隔离）
