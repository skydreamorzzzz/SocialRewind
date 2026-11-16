# SocialRewind：可重启的微型社会实验室

SocialRewind 是一个面向 **AI 社会策略探索**的可重启多智能体科研原型。项目通过构建具有客观约束、利益冲突与信息不对称的社会环境，让 LLM Agent 在相同初始条件下经历不同的互动过程，并探索跨时间线经验如何影响后续决策。

首个实验场景基于 [AgenticPay](https://arxiv.org/abs/2602.06008) 的双边价格谈判任务设计，采用可控的二手商品交易设置，实现了真实 LLM 自主谈判、双方轮回记忆、可信状态管理与确定性回放。具体商品与估值由本项目构造，并非从 AgenticPay 原始数据集中抽取。

**The World Resets. Experience Remains.**

## v2 最小科研 Demo

Buyer 与 Seller 自主协商**价格、交付时间、售后保障**。可自由询问、交换条件、提出或修改完整合同、接受、拒绝和退出；世界只验证规则、计算收益，不预设谈判流程。默认合同空间为 3000–4500 元（步长 50）、1/3/7 天交付、0/30/90 天保障。

```powershell
# Python 3.11+，纯标准库。离线三世对照并生成可直接打开的单文件页面
python -m social_rewind.experiment
Invoke-Item runs/v2.html

# 两组私人偏好、各重复三次；Control 与 Rewind 每世都使用全新 Agent 会话
python -m social_rewind.experiment --lives 3 --profiles urgent warranty --repeats 3
python -m unittest discover -s tests -v
```

真实模型通过现有 OpenAI-compatible API 接入；配置环境变量后运行：

```powershell
$env:SOCIALREWIND_API_BASE="https://api.deepseek.com"
$env:SOCIALREWIND_MODEL="deepseek-flash"
# 预先设置 DEEPSEEK_API_KEY，不要把密钥写入文件
python -m social_rewind.experiment --live --api-key-env DEEPSEEK_API_KEY --disable-thinking --lives 1
python -m social_rewind.experiment --live --api-key-env DEEPSEEK_API_KEY --disable-thinking --lives 3
# 同一命令加 --profiles urgent warranty --repeats 3 即为真实批量实验
```

`--lives 3` 共运行五场：共同第一世，然后 Control 第 2/3 世、Rewind 第 2/3 世。每次重启恢复同一 S0；Control 只知道发生过重置，Rewind 累积自己实际可见的全部往世经历。不同重复不共享经验，重复间交替两组运行顺序。`--lives 1` 为单次双 Agent 谈判。离线脚本仅验证机制，两组使用相同脚本，不模拟记忆带来的改进。

私人效用：买方 `value-price-delay*(delivery_days-1)+warranty*warranty_days`，卖方 `price-value+delay*(delivery_days-1)-warranty*warranty_days`；各自仅见自己的系数。默认 `urgent` 买方更在意速度，`warranty` 买方更重保障。正式报价/接受须满足自身收益 ≥ 0；未成交收益为 0。世界在公开有限合同网格上枚举可行协议，判断是否存在双方不变差且至少一方变好的替代方案，计算 Pareto Efficiency；退出点也纳入比较。

批量输出按**配置/组/世数**分别统计 Agreement Rate、Buyer/Seller Utility、Social Welfare（双方效用之和）、Pareto Efficiency 比例、Negotiation Rounds 的均值与样本标准差。效用均值包含未成交的 0；轮数是每两次交替行动一轮，消息也计行动。重复数为 1 时标准差显示 0，仅表示无法估计重复间波动。

`--log runs/v2.json --html runs/v2.html` 指定输出。JSON 保存完整配置、角色授权记忆、实际模型请求/响应、行动、拒绝与状态差异；API/输出失败时保留部分日志、明确停止，不回退脚本。HTML 是本地**审计页面**，可切换配置/重复、并列查看各世对话、条款、收益、首案变化与显式决策说明，包含双方私人记录，不应提供给 Agent。页面无外部依赖，双击即可打开。

```powershell
# 只消费已记录行动，无需密钥，不重新调用 LLM
python -m social_rewind.experiment --replay runs/v2.json --html runs/replayed.html
```

可通过 `--scenario path.json` 读取完整自定义场景（格式见结果 JSON 的 `scenario`）；`--max-rounds` 设置内置场景轮数，`--temperature`、`--max-tokens` 调整采样配置。`--seed` 仅在服务支持时使用并写入调用记录，不保证确定性采样。三维合同与批量入口复用 `trade.py` / `negotiation.py`；`experiment.py` 负责简单串行实验，`timeline_view.py` 只生成静态审计页。

已完成 50 项离线测试、两种偏好的重复批量演示和一组真实三世对照。真实运行中五场均成交，社会福利均为 660，未达到 Pareto 效率；详情、原始日志与首次失败记录见 [v2 实验记录](EXPERIMENT_V2.md)。可直接在本地打开 [真实时间线页面](examples/v2_live.html)。

## v1 初步实验成果

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

## v1 运行与兼容接口

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

## v1 最小规则与指标

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

当前 v2 已加入三维协商、多世轮回与批量实验。下一阶段可研究轮回经验的策略价值及跨情境迁移，再考虑多方竞争与更复杂的社会合作场景。

当前对照结果属于初步探索，尚不能单独证明普遍的策略改进或因果效应；记忆可用性与轮回叙事情境同时变化。合同效用是简化的确定性模拟，不模拟真实履约风险。当前实现不包含身份认证、数据库、在线 Web 服务、并发市场、长期记忆压缩或自动策略优化；全部历史直接注入，世数增加会受到模型上下文窗口和成本限制。

详见 [完成报告](COMPLETION_REPORT.md) 与 [对照实验记录](EXPERIMENT_CONTROL.md)。
