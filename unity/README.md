# Unity 客户端接入指南（Phase 4b）

后端已就绪，Unity 侧只需三步接线。脚本在 `Scripts/`，直接拖进工程的 `Assets/Scripts/` 即可（无第三方依赖，用的都是 UnityWebRequest + JsonUtility）。

> ⚠️ 这些脚本是在 Unity 之外预写的，尚未在编辑器里实测——第一次导入如遇编译报错（Unity 版本 API 差异），报错行一般很好改。

## 0. 启动后端

```bash
# 项目根目录（先确认 Ollama 在跑）
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

冒烟：浏览器开 `http://127.0.0.1:8000/health` 应返回 `{"status":"ok"}`。

## 1. 最小可跑 demo（10 分钟目标）

1. 场景里放个胶囊体，命名 Aldric，挂 `NpcBehaviorClient`
2. 加一个 World-Space Canvas + Text 当头顶气泡
3. 写一行胶水脚本把事件接到气泡：

```csharp
using GaiNpc;
using UnityEngine;
using UnityEngine.UI;

public class BarkBubble : MonoBehaviour
{
    public NpcBehaviorClient npc;
    public Text bubble;

    void Start()
    {
        npc.OnActionChanged.AddListener(a => Debug.Log($"action: {a}"));
        npc.OnBark.AddListener(line => bubble.text = line);
        npc.OnShouldTalk.AddListener(() => Debug.Log("open dialogue UI here"));
    }
}
```

4. 运行。`NpcBehaviorClient` 自带一个"口渴的铁匠"演示状态，每 5 秒问一次 `/act`——预期看到 action=drink 和一句铁匠口吻的台词。

## 2. 接真实场景状态

演示状态跑通后，把 `StateProvider` 换成读真实场景的函数：

```csharp
npc.StateProvider = () =>
{
    var s = new NpcGameState();
    s.thi = thirstSystem.Value;          // 你的口渴数值
    s.hour = timeOfDay.Hour;             // 游戏时钟
    if (wolfNearby)
        s.percepts.Add(new NpcGameState.Percept
            { id = "wolf", tag = "Threat", threat = 0.8f, sal = 0.9f });
    if (playerNearby)
        s.percepts.Add(new NpcGameState.Percept
            { id = "player", tag = "Social", sal = 0.7f });
    return s;
};
```

字段随便缺——后端对缺失字段按 0 处理。`OnActionChanged` 接 Animator/NavMesh（drink→走向井、flee→跑离威胁源……）。

### 位移（NpcLocomotion，已实现）

`NpcLocomotion` 把动作 id 变成 NavMesh 上的真实移动：walk_to/gather/work→出生点附近随机游荡，flee→远离玩家加速跑，socialize→走向玩家停在聊天距离，其余动作原地不动；任何对话框打开时全村冻结（与 /act 暂停同一规则），附近 NPC 转身面向玩家。前置条件：场景烘焙过 NavMesh、NPC 挂 `NavMeshAgent`（GAINpcDemo 工程里 "GAI NPC → Bake NavMesh" 和 "Setup Character NPCs" 两个菜单各点一次即可）。注意：暂无走路动画，位移时是滑行观感，动画后补。

## 3. 对话 UI

玩家按 E 或 `OnShouldTalk` 触发时，用 `NpcDialogueClient`：

```csharp
dialogue.Send(inputField.text, (reply, guardReason) =>
{
    replyText.text = reply;
    if (guardReason == "secret_low_trust")
        ; // 可选：NPC 眯眼/后退等警觉演出
});
```

`guardReason` 非空表示这轮被规则守门约束过（问秘密/注入攻击），可以做表现层反馈。

### 流式语音回复（默认路径）

`DialogueUI` 现在走 `POST /chat_stream`：后端逐句生成+逐句合成，Unity 每 0.25s
轮询 `GET /chat_stream/{session_id}?after=N` 取新句子，字幕逐句追加，语音按队列
连续播放。实测（RTX 4060、无 Unity 占显存、预热后）：首句 ~5s，之后每句
0.6–0.9s 连续到达——旧的整段路径要等全部生成+合成完才出声。旧的阻塞式
`/chat` 仍在，`NpcDialogueClient.Send()` 未删，评估脚本继续用它。
每个 chunk 带 `t_ms`（距请求开始的毫秒数），是 RQ4 延迟表的原始数据。
首次请求多付 ~40s 模型懒加载，属正常。

## 4. 语音（可选）

`speak=true` 时响应会带 base64 WAV（XTTS 克隆音色，约多 2–5 秒延迟）。给客户端脚本的 `voiceSource` 拖一个 AudioSource 即可自动播放（`WavUtility` 负责解码）。建议 bark 常关、对话按需开。

## 5. Lip-sync 备忘（课程结论）

Hunyuan3D 生成的网格**没有 viseme blendshapes，OVRLipSync 用不了**。两个方案：
- **A（保底）**：用 AudioSource 输出振幅驱动下颌骨旋转（十几行脚本）
- **B**：用自带 blendshapes 的现成 avatar（如 Ready Player Me）

## 接口速查

| 端点 | 用途 | 关键返回 |
|------|------|---------|
| `POST /act` | 行为决策 + bark | `action_id`, `mood`, `bark`, `should_talk`, `latency_ms` |
| `POST /chat` | 完整对话（阻塞） | `reply`, `guard.reason?`, `audio_base64?` |
| `POST /chat_stream` | 完整对话（逐句流式） | `session_id` |
| `GET /chat_stream/{id}?after=N` | 轮询流式句子 | `chunks[{index,text,t_ms,audio_base64?}]`, `done`, `guard?`, `error?` |
| `POST /transcribe` | 语音转文字 | `text` |
| `GET /npc/{name}` | persona 摘要 | 调试用 |
