# SocialRewind：可重启的微型社会实验室

**The World Resets. Experience Remains.**

两个 LLM Agent 在同一初始世界 S₀ 中自由谈判，经历重置后，探索各自保留的经历如何影响下一世选择。世界负责客观规则、信息权限和收益，Agent 自主沟通、交换条件、接受或退出。没有最低谈判轮数，也不要求轮回后一定改进。

## 一个有利益冲突的手机案例

星云 X1 二手手机，使用 10 个月，屏幕无裂纹，电池健康度 91%，含保护壳和充电线；挂牌 4200 元。可协商价格（3000–4500 元，步长 50）、交付（1/3/7 天）、售后保障（0/30/90 天）。

买方希望价格合理、交付较快、有至少 30 天保障，担心隐藏故障；卖方希望收益较高，因迁移个人数据更适合 3 天交付，希望控制售后责任。双方只见自己的角色卡、效用和最低接受条件，可以选择披露需求。

效用为离散的确定性设置：买方 `4100 - 价格 + 交付效用 + 保障效用`；卖方 `价格 - 3300 + 交付效用 + 保障效用`。以下是研究者审计设定，不会整体提供给任何 Agent：

| 条款 | 买方效用变化 | 卖方效用变化 |
|---|---:|---:|
| 1 天交付 | +120 | -220 |
| 3 天交付 | 0 | 0 |
| 7 天交付 | -250 | +20 |
| 0 天保障 | 0 | 0 |
| 30 天保障 | +140 | -50 |
| 90 天保障 | +170 | -170 |

默认最低可接受效用为 0。下限仅用于合法性检查，不代替角色的理想目标。私人观察提供自身条款效用、合法价格边界和当前方案的自身收益，不提供对方参数、联合指标或理论最优合同。自然语言的同意不自动成交，必须明确接受当前对方的有效 `offer_id`；新完整报价替代旧报价。

## 运行与查看

新增三组第二世消融入口，冻结本案例、角色卡、效用、合同网格和基础谈判 Prompt。每次独立重复共四场：共同第一世，再从同一 S₀ 分出 **Control**（无记忆）、**Rewind-H**（本人完整授权历史）、**Rewind-HR**（本人历史及 Reflection）。三个分支共享完全相同的重启叙事，仅独立记忆数据块不同；轮数和谈判动作不受额外限制。

```powershell
# 离线检查与真实批次；真实运行需预先设置 DEEPSEEK_API_KEY
python -m social_rewind.experiment --ablation --repeats 2
python -m social_rewind.experiment --ablation --live --api-base https://api.deepseek.com --model deepseek-flash --api-key-env DEEPSEEK_API_KEY --disable-thinking --repeats 10 --log runs/ablation.json --html runs/ablation.html
Invoke-Item runs/ablation.html
```

每场结束后用独立调用，要求双方只根据本世授权观察估计对手在**3 天交付、30 天保障**下的客观合法接受边界。买方估计卖方最低价，卖方估计买方最高价；分别记录依据、不确定性和绝对误差。研究者由世界效用枚举得到真值 **3350 / 4200 元**，不将它们作为答案输入 Agent。这不是对手下一轮报价、理想价或愿意立即成交的价格。

Reflection 在估计之前生成；估计结果、真值和误差仅用于事后分析，均不注入 Reflection、世界事件或后续记忆。统计按指标报告有效 n、均值和样本标准差，计算 H−Control、HR−Control、HR−H 的配对差值及 95% bootstrap 区间（独立重复为重采样单位，10000 次，seed=2026），并报告各组第一世→第二世估计误差变化。n<2 不估计标准差/区间，结果仅供探索且未作多重比较校正。

每次独立重复轮换三组执行顺序；单场失败显式记录，第一世失败则不创建后续分支，不换样本。配对分析仅采用同次重复中该指标双方均有效的结果。配额/鉴权/限流失败会停止批次并报告实际数量，其他失败保留并继续独立重复。试运行单独保存，不并入正式统计。全部原始请求、响应、估计、事件与派生统计保存在 JSON；页面包含三组收益和双方估计误差图。

原有三世入口继续保留：

Python 3.11+，纯标准库。离线脚本只验证机制，不作为策略探索证据：

```powershell
python -m social_rewind.experiment
Invoke-Item runs/v2.html
python -m unittest discover -s tests -v
```

真实三世对照（需预先配置 `DEEPSEEK_API_KEY`；密钥不写入文件）：

```powershell
python -m social_rewind.experiment --live --api-base https://api.deepseek.com --model deepseek-flash --api-key-env DEEPSEEK_API_KEY --disable-thinking
```

API、模型也可通过 `SOCIALREWIND_API_BASE`、`SOCIALREWIND_MODEL` 配置，默认密钥变量为 `SOCIALREWIND_API_KEY`。`--lives 1` 运行单次谈判，默认 `--lives 3` 共运行五场：共同第一世，Control 第 2/3 世，Rewind 第 2/3 世。`--repeats N` 重复同一案例，重复间不共享经验。可用 `--max-rounds`、`--temperature`、`--max-tokens`、`--seed` 记录实验设置；相同种子不保证相同在线输出。

每世结束，双方只依据自己可见的事件和结果生成简短个人总结：观察、对对方需求的判断、下一世可能的策略。总结允许出错，只校验格式与证据编号；它保存在世界之外。Rewind 继承本人授权历史和总结，Control 只知道发生过重置，不继承具体经历或总结。所有世使用新会话，角色目标、规则及采样参数一致。

输出默认是 `runs/v2.json` 和单文件 `runs/v2.html`，可用 `--log` / `--html` 指定。页面按“前世谈判 → 保留经验 → 本世行动 → 合同与收益”展示，可切换第 2/3 世、查看公开对话、报价变化、个人总结和原始审计信息。页面含双方私人记录，仅供研究者本地查看，不作为 Agent 观察接口。

```powershell
# 不联网，校验世界回放、记忆来源和总结继承，并重新生成页面
python -m social_rewind.experiment --replay examples/phone_case_live.json --html runs/replayed.html
```

## 本轮三组消融结果

正式预定10次尝试：9次形成完整配对，第2次因第一世报4125元（违反50元步长）停止，原始记录保留、不替换。正式319次调用，另有25次机制试运行；有效第二世各组n=9，均成交。均值 ± 样本标准差：

| 指标 | Control | Rewind-H | Rewind-HR |
|---|---:|---:|---:|
| 买方收益 | 207.8 ± 98.3 | 192.2 ± 80.9 | 184.4 ± 76.8 |
| 卖方收益 | 648.9 ± 92.7 | 686.7 ± 96.7 | 705.6 ± 76.8 |
| 买方估计卖方最低合法价：误差 | 661.1 ± 124.4 | 661.1 ± 74.1 | 666.7 ± 70.7 |
| 卖方估计买方最高合法价：误差 | 133.3 ± 100.0 | 138.9 ± 99.3 | 77.8 ± 75.5 |

HR相对H的卖方估计误差差值为 -61.1，探索性95%配对区间 [-116.7, -11.1]；但HR相对Control区间包含0，买方误差也没有改善。H的9/9场最终合同复用了前世，HR为8/9；计划有被执行的例子，也有生成Reflection后重复原方案的例子。**客观合法边界是3350/4200，不能把对方报价、底线主张或最终成交价当成真值。**

全部指标、配对差值/区间和第一世→第二世误差变化见 [实验记录](EXPERIMENT_V2.md)，[正式日志](examples/ablation_live_10.json) 与 [三组时间线页面](examples/ablation_live_10.html) 保留所有成功/失败。世界、授权记忆、估计原始输出和统计均已离线验证，68项测试通过。区间未作多重比较校正，短对话未必能识别隐藏效用，不能宣称普遍策略学习或因果改善。

## 上一轮单组真实观察（保留）

已保存一组真实三世对照，共 33 次调用（23 次谈判、10 次总结）。五场都在 3 天交付、30 天保障成交；Rewind 第二、三世都以 4150 元成交，买方收益 90；Control 分别谈到 4050、4100 元，买方收益 190、140。所有场次社会福利均为 890，达到当前有限网格的 Pareto 效率。

角色通过沟通披露保障需求与加急成本；Rewind 明确引用了前世经历，却也重复了原有价格锚点和保守接受策略。Rewind 未提高买方收益或总福利，卖方收益则高于两场 Control；这里只观察到收益分配差异。详细对话、判断错误与局限见 [实验记录](EXPERIMENT_V2.md)，可直接在本地打开 [真实时间线页面](examples/phone_case_live.html)。

## 实现、评测与来源

- `trade.py`：三维合同、角色观察、确定性效用与权限、只追加事件、精确重置及回放。`negotiation.py`：自主 LLM / 离线 Agent、结构化行动、本人经验总结。
- `experiment.py`：共享第一世的 Control/Rewind、总结注入、重复与统计；`timeline_view.py`：无依赖的本地审计页面；`llm.py`：OpenAI-compatible API 与脱敏调用记录。
- 指标沿用 Agreement Rate、双方 Utility、Social Welfare、Pareto Efficiency、Negotiation Rounds。Pareto 判断枚举合法有限合同空间及退出点；未成交计 0 收益；每两次交替行动算一轮，消息也计行动。统计按组/世汇总均值和重复间样本标准差。
- 交互方式参考 Stanford [CraigslistBargain 原论文](https://nlp.stanford.edu/pubs/he2018decouple.pdf) 中通过送货、自取等附加条件议价的做法；本案例不增加自取维度，保障效用由本项目构造。此前价格任务与评价参考 [AgenticPay](https://arxiv.org/abs/2602.06008)。**手机、角色及估值均为虚构设置，不是这些数据集的原样样本或真实成交流水。**

保留原有测试和日志兼容路径：`python -m social_rewind.trade_demo --compare`、`python -m social_rewind.demo`。旧线性配置仅用于历史记录/回归测试，当前 Demo 只运行上述一个手机案例。

当前结果只有一次重复，记忆、个人总结及轮回叙事情境同时变化，不能证明因果改善；个人判断未经事实核查，完整历史注入也受上下文窗口限制。尚不模拟真实履约风险，不包含身份认证、数据库或额外服务。API/格式失败及非法行动会显式停止并保留记录，不伪造结果。
