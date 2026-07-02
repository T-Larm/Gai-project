# Evaluation

针对报告 RQ1–RQ5 的实验脚手架。所有命令在项目根目录执行，需要 Ollama 在跑。

## 实验条件（baseline / ablation）

| 条件 | 说明 | 服务于 |
|------|------|--------|
| `full` | 三层 persona + 记忆检索 + dynamic 更新（完整系统） | 所有 RQ |
| `no_memory` | 关闭记忆检索注入（其余同 full） | RQ3 ablation |
| `flat` | 同样的 persona 事实压成单段无结构 prompt，无记忆/dynamic | RQ1 |
| `none` | 只给名字+职业 | RQ1 下界 |
| `handwritten` | 人工手写 persona（`handwritten_personas/{npc}.txt`），无记忆/dynamic | RQ2 |

同一套开关也可在 CLI 使用（用户研究 RQ5 切条件）：
`python -m backend.main --npc aldric --text --no-memory --prompt-style flat`

## 数据集（test_data/）

- `dialogue_prompts.json` — 60 条（quest / smalltalk / adversarial 各 20）
- `memory_probes.json` — 6 个记忆探针（事实注入 → 5 轮填充 → 回忆提问 → 关键词命中）
- `consistency_pairs.json` — 15 对同义事实问题（先问完所有 A 再问所有 B，间隔 15 轮）

## 跑实验

```bash
# 1. 生成 transcripts（每条件一次；all = prompts + memory + pairs）
python -m evaluation.run_dialogues --npc aldric --condition full --suite all
python -m evaluation.run_dialogues --npc aldric --condition no_memory --suite all
python -m evaluation.run_dialogues --npc aldric --condition flat --suite all
python -m evaluation.run_dialogues --npc aldric --condition none --suite all
python -m evaluation.run_dialogues --npc aldric --condition handwritten --suite all

# 2. LLM-as-judge 打分（prompt-to-line 一致率，输出各条件×类别表格）
python -m evaluation.judge_consistency evaluation/results/aldric_*.jsonl

# 3. 延迟测量（RQ4；n=30，报告 mean±std）
python -m evaluation.measure_latency --n 30 --components llm
python -m evaluation.measure_latency --n 10 --components stt,tts
```

结果落在 `evaluation/results/`（gitignore 掉原始 transcripts，汇总表进报告）。

## 输出与 RQ 的对应

- **RQ1/RQ2（prompt-to-line 一致率）**：judge 表格，`full` vs `flat`/`none`，`full` vs `handwritten`
- **RQ3（记忆贡献）**：runner 输出的 memory 命中率，`full` vs `no_memory`
- **line-to-line 矛盾率**：`pairs` 记录的 reply_a/reply_b，用 NLI（DeBERTa CrossEncoder）或人工标注判矛盾——脚本待补
- **RQ4（延迟）**：measure_latency 表格
- **RQ5（用户研究）**：CLI 条件开关 + Google Form 问卷（Likert 1–5：naturalness / believability / engagement / perceived consistency），10–15 人 within-subject 盲测，Wilcoxon signed-rank

## 注意

- judge 默认 `llama3:latest`（`--judge-model` 可换）；judge 与被测是同一个基座模型，报告 limitations 里要提 self-preference bias，有预算可换 GPT-4 复核一遍
- 跑批用 `PersonaGenerator.load` 的新副本且不落盘，不会污染 `data/personas/`
