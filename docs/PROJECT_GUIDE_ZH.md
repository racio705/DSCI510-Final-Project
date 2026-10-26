# 简历与面试说明：最终验证版本

建议项目名称：**纽约 Airbnb 长住报价基准模型与时间外验证**。

主成果是按预先锁定规则完成的后续时间快照验证，以及可运行的 JSON/CSV 报价比较工具。先读 [时间外验证结论](VERIFIED_CONCLUSIONS_ZH.md)；原 2025 年探索结果作为分析附录保留。

## 中文简历版本

- 整合 NYC Airbnb 三期快照共 **96,288 条房源快照记录**及 MTA、Census 空间数据，构建带版本校验、空间连接、房东账号分组和数据质量审计的长住挂牌价分析流程。
- 基于 17,100 条训练期长住房源构建按房东均衡加权的梯度提升模型；在 **311 个历史快照未出现过的账号**上进行锁定时间外检验，MAE **$96.03/晚**，较两种分层可比房源基准降低 **13.6%–16.3%**；名义 80% 区间实测覆盖率 **82.6%**。
- 在两期后续快照中复现固定住宿类别的跨区报价差异，完成最大账号剔除敏感性分析，并交付支持 JSON/CSV 输入、地理特征推导、参考价区间和可比样本支持度的本地工具。

96,288 是三期原始记录之和，存在跨期重复房源，不能写成 96,288 套独立房屋。17,100 是原 2025 快照中非酒店且最低入住至少 30 晚的训练样本。311 是严格主要测试账号数；不能用当期全部 14,152 条合格记录冒充未见账号测试量。

## English resume version

- Built a versioned geospatial pipeline combining 96,288 NYC Airbnb listing-snapshot records with MTA and Census data, including source checks, tract joins, host-account grouping and quality audits.
- Trained a host-balanced gradient-boosted asking-price model on 17,100 long-stay listings and evaluated a locked temporal holdout of 311 previously unobserved accounts; achieved $96.03 nightly MAE, 13.6%–16.3% below two hierarchical comparable-price baselines, with 82.6% empirical coverage for nominal 80% intervals.
- Replicated a fixed accommodation-category price benchmark in two later snapshots, tested sensitivity to dominant host accounts, and delivered a local JSON/CSV comparison tool with geographic enrichment, prediction intervals and comparable-sample support.

## 面试中最值得讲的两个结论

**结论一：固定住宿类别的区域报价差异跨时间重复出现。** 条件为一卧整套、标注可住恰好 2 人、最低入住恰好 30 晚、距地铁不超过 500 米。Brooklyn 在 2026 年 6 月与 9 月的挂牌价中位数分别比 Manhattan 低 35.9% 和 31.9%；移除每区最大账号后仍分别低 22.3% 和 18.4%。这是分层报价基准，不是同一房屋换区的因果效果。

**结论二：模型的优势在时间外且未见账号的测试中得到支持。** 训练只用 2025 年 10 月，2026 年 6 月只校准区间，9 月不调参。测试账号与房源 ID 在前两份全部原始记录中均不存在，每账号按与价格无关的哈希选一条。主模型误差降幅的配对重抽样区间均排除零；预测区间宽度和所有失败子组同时公开。

## 需要解释清楚的细节

- **冻结发生在什么时候？** 协议 commit `b75e5a69daf625312ca2a4c582d3ddbf36657447` 早于两份未来快照的下载。模型、比较层级、账号抽样、通过标准与分析方向都先写定；不是注册实验。
- **对照有多强？** 两个基线按社区、房型、容量、卧室、入住门槛等层级寻找训练期可比组，组内不足 20 条或 5 个账号就按固定顺序回退。一个使用房源中位数，另一个使用账号内中位数的中位数。
- **区间说明什么？** 311 个主要测试账号中覆盖 257 个。它是后续快照上的实测覆盖，不保证每个地区、房型或未来时期都达到 80%。整体区间宽度中位数约 $169，适合初筛参考。
- **有哪些失效范围？** Queens 独立房间的误差百分比中位数为 32.9%，超过整体预设的 30% 精度目标；若干地区/房型样本不足 30。只说整体达到预设离线标准，不说所有房源都可靠。
- **“地铁没作用”成立吗？** 没有得到支持。加入地铁后的预测等效区间没有完全落在预设 ±2% 范围，不能把不显著当成证明没有作用，更不能声称因果效果。
- **有没有商业收益？** 没有真实订单或运营实验，不能写营收提升、提高入住率或自动定价收益。实际交付是经过离线检验的挂牌价比较工具。

## 运行与示例

先运行 `python src/validate_temporal.py`，再运行 `python src/quote_benchmark.py --input examples/quote_request.json`。该输入是明确标注的假设案例，输出约 $144/晚、名义区间约 $83–$249 和可比中位数 $153，不能当作实际在售/可订房源。

原课程工作在 2025 年完成，后续完善与时间外验证在 2026 年完成。简历日期应按真实经历填写。旧版空间 CV 的 33.8% 改善与当前时间外 13.6%–16.3% 改善使用不同样本、模型加权和基线，不应混写或择高宣传。
