# SocialRewind：双人轮回交易谈判

SocialRewind 是一个面向 **AI 社会策略探索**的可重启多智能体科研原型。项目通过构建具有客观约束、利益冲突与信息不对称的社会环境，让 LLM Agent 在相同初始条件下经历不同的互动过程，并探索跨时间线经验如何影响后续决策。

首个实验场景基于 [AgenticPay](https://arxiv.org/abs/2602.06008) 的双边价格谈判任务设计，采用可控的二手商品交易设置，实现了真实 LLM 自主谈判、双方轮回记忆、可信状态管理与确定性回放。具体商品与估值由本项目构造，并非从 AgenticPay 原始数据集中抽取。

**The World Resets. Experience Remains.**

## 初步实验成果

项目已实现从单次社会互动到双人轮回的完整实验闭环，并进一步建立了无记忆对照组（Control）与双人轮回组（Rewind）。

在一次真实 LLM 对照实验中，三个分支均成功达成交易：

| 指标 | 第一世 | Control | Rewind |
|---|---:|---:|---:|
| 成交价格 | 3900 | 3900 | 3700 |
| 买方收益 | 0 | 0 | 200 |
| 卖方收益 | 600 | 600 | 400 |
| 谈判轮数 | 2 | 4 | 3 |

Rewind 组的 Agent 在决策中明确引用了上一世的谈判经历，并据此调整报价策略。这一初步观察展示了跨时间线记忆参与多智能体决策的可能性。

实验使用真实模型 API，完整保存模型交互与世界事件，并通过确定性回放验证运行过程的可审查性。详细结果见 [对照实验记录](EXPERIMENT_CONTROL.md)。

## 最短运行

```powershell
# Python 3.11+；纯标准库，离线演示不联网
python -m social_rewind.trade_demo --log examples/trade_offline_run.json
python -m social_rewind.trade_demo --compare --log examples/trade_compare_offline.json
python -m unittest discover -s tests -v
```

真实 OpenAI-compatible 模型：

```powershell
$env:SOCIALREWIND_API_BASE="https://your-api.example/v1"
$env:SOCIALREWIND_MODEL="your-model"
$env:SOCIALREWIND_API_KEY="..."
python -m social_rewind.trade_demo --compare --live --log examples/trade_compare_live.json
```

若服务的推理模式会耗尽结构化输出预算，可显式加 `--disable-thinking`。也可用 `--api-key-env DEEPSEEK_API_KEY` 指向其他已配置变量；密钥不会写入日志。原配音社基座演示仍可运行：`python -m social_rewind.demo`。

## 最小规则与指标

- 商品、挂牌价公开；Buyer 只知道自己的最高愿付价，Seller 只知道自己的最低接受价。
- `message / offer / accept / reject / exit` 共用现有 `Action + DecisionNote` 信封。自然语言暗示不成交，只有明确接受当前对方有效报价才成交。
- 价格、行动顺序、保留价、20 轮上限与终止状态由确定性代码执行；报价不会推进原有模拟日。
- 成交价为 `p`、可交易剩余为 `Z = buyer_max - seller_min`：原始效用分别为 `buyer_max-p` 与 `p-seller_min`，归一化效用为各自效用除以 `Z`，交易质量 `Q = 4*u_b*u_s`，轮次效率为 `gamma^(rounds-1)`。指标分别保留，不合成单一总分。

## 架构与轮回边界

- `trade.py`：独立的轻量交易领域、角色观察、完整审计、精确 reset、授权记忆和确定性 replay；未把交易逻辑塞入原有 `world.py`。
- `llm.py`：最小 OpenAI-compatible `chat/completions` 客户端，配置 API、模型与密钥；不记录鉴权头。
- `negotiation.py`：两角色 Runner、离线 ScriptedAgent、严格 JSON LLM Agent；格式最多修复一次，非法世界行动显式失败。
- `trade_demo.py`：保留原有 `Timeline 1 → Rewind` 演示；`--compare` 从同一 Timeline 1 派生两个独立世界：Control 知道发生过重置但没有可用前世经历，Rewind 双方获得各自合法记忆。
- 模型调用输入、输出、元数据和世界事件分开审计；回放只消费已记录的结构化行动，不再次请求模型。

Control 与 Rewind 使用相同商品、估值、规则、模型参数和轮数上限，并各自创建全新 Agent 会话。该设计仍同时改变了“记忆内容”和“轮回叙事情境”，且单次 API 采样不能排除模型随机性；详见 [对照实验记录](EXPERIMENT_CONTROL.md)。

## 与 AgenticPay 的关系

本项目以 AgenticPay 的双边谈判任务为基础，参考 [AgenticPay 论文](https://arxiv.org/abs/2602.06008) 及其 [`Task1_basic_price_negotiation`](https://github.com/SAIL-Research-Lab/AgenticPay)，借鉴私人保留价、多轮协商、经济效用与交易质量等实验设计，并将其整合进 SocialRewind 的可信社会世界。二手手机、角色和具体估值是本项目的可控实验设置，不是 AgenticPay 原始数据的复刻。

在此基础上，SocialRewind 进一步引入**精确世界重置、双边跨时间线记忆、角色信息隔离与分支对照实验**，将研究重点从单次谈判表现拓展到社会经验对后续策略选择的影响。

## 研究范围与后续方向

当前版本聚焦于最小且完整的研究闭环：以双边价格谈判作为可控实验环境，验证 LLM Agent 的自主互动、跨时间线记忆与策略变化。系统采用轻量化设计，将客观规则与 Agent 决策分离，保留完整审计与回放能力。

下一阶段将重点研究轮回经验的策略价值及其跨情境迁移能力，并逐步拓展到多方竞争、多维协商与更复杂的社会合作场景。

当前对照结果属于初步探索，尚不能单独证明普遍的策略改进或因果效应；后续仍需通过更多场景与重复实验进一步验证。当前实现不包含身份认证、数据库、Web UI、并发市场、长期记忆压缩或自动策略优化；真实模型输出格式错误或提出非法行动时，实验会显式停止。

详见 [完成报告](COMPLETION_REPORT.md) 与 [对照实验记录](EXPERIMENT_CONTROL.md)。
