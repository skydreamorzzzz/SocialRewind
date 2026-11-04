# SocialRewind：双人轮回交易谈判

这是研究“相同社会初始条件下，双方保留各自合法前世记忆后是否改变选择”的最小科研原型。当前主案例是**完全虚构的二手手机交易**：一个 Buyer、一个 Seller、一个商品，最多 20 轮。两次谈判只能展示机制，不能证明策略学习有效。

## 最短运行

```powershell
# Python 3.11+；纯标准库，离线演示不联网
python -m social_rewind.trade_demo --log examples/trade_offline_run.json
python -m unittest discover -s tests -v
```

真实 OpenAI-compatible 模型：

```powershell
$env:SOCIALREWIND_API_BASE="https://your-api.example/v1"
$env:SOCIALREWIND_MODEL="your-model"
$env:SOCIALREWIND_API_KEY="..."
python -m social_rewind.trade_demo --live --log examples/trade_live_run.json
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
- `trade_demo.py`：`Timeline 1 → Rewind → Timeline 2`。第二世双方都知道彼此重生，但各自只获得上一世亲眼见过的事件、自己的决策说明、自己的效用与私人价格。
- 模型调用输入、输出、元数据和世界事件分开审计；回放只消费已记录的结构化行动，不再次请求模型。

## AgenticPay 参考与差异

任务设定与评价定义参考 [AgenticPay 论文](https://arxiv.org/abs/2602.06008) 及其 [`Task1_basic_price_negotiation`](https://github.com/SAIL-Research-Lab/AgenticPay)。本项目仅采用双边保留价、20 轮上限、原始/归一化效用、Nash 型交易质量与轮次折扣；没有复制其环境注册、多模态、合同模式、任务集或加权总分。SocialRewind 额外强调事件权限、S0 重置、双方轮回记忆隔离与精确回放。

## 明确局限

没有身份认证、数据库、Web UI、并发市场、长期记忆压缩或自动策略优化。真实模型输出仍可能格式错误或提出非法行动，此时实验会显式停止；单次两世结果不具统计意义。仓库中的 live 日志全部是虚构案例，但包含完整模型提示与响应，应按审计数据对待。

详见 [完成报告](COMPLETION_REPORT.md)。
