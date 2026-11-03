# SocialRewind：可审查的最小社会世界

这是研究“相同社会初始条件下，跨时间线经验能否改变选择”的世界基座。当前案例是**完全虚构的配音社项目**；本仓库只证明硬约束、信息权限、结算与回放机制可工作，不宣称发现了社会策略。

```powershell
# Python 3.11+
python -m social_rewind.demo
python -m unittest discover -s tests -v
```

写出脱敏的虚构演示日志：

```powershell
python -m social_rewind.demo --log examples/demo_log.jsonl
```

## 最小架构

- `models.py`：统一的 `Action(kind, actor, payload, decision)` 信封与短决策说明；自由文本事实明确标为角色自述，可引用当前时间线内可见的事件编号。
- `world.py`：确定性执行预算、权限、档期、截止日期和状态差异；普通观察按角色及当前时间线过滤，完整跨时间线审计仅 `auditor` 可读。`act()` 只返回裁剪后的行动回执。
- `judge.py`：受限 Judge 契约与离线确定性 `MockJudge`。Judge 只评有效工时和质量，不能修改硬状态。
- `scenario.py`：三名目标不完全一致、私有信息不同的虚构角色，一个受预算与截止日期约束的共同任务。
- JSONL 审计事件记录输入、合法性、结果、状态差异及 Judge 输入/输出；回放直接消费已记录判定，不再次调用模型。
- `TimelineMemory` 不属于 `World`，因此能改变脚本策略但不能污染不可变初始状态 S0。

## 明确局限

没有实时多 Agent 编排、真实 LLM 适配器、数据库、Web UI、并发行动或长期日志存储。身份由调用方传入，尚无密码学认证；JSONL 需要由实验运行器妥善保存。当前结算是单任务、按天批处理，服务于可复现实验演示而非生产系统。

详见 [完成报告](COMPLETION_REPORT.md)。
