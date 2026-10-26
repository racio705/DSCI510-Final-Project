# 项目定位与简历表述

项目名称建议：**纽约 Airbnb 挂牌价格建模与地铁可达性分析**。

它适合展示 Python 数据清洗、GeoPandas 空间连接、地理特征工程、机器学习评估和结果解释。分析对象是挂牌价，不是房屋售价、成交价或营收。原课程项目在 2025 年完成；这次完善属于后续扩展，简历日期应按你的真实经历填写。

## 可直接使用的中文项目经历

- 整合 Inside Airbnb、MTA 地铁站及 Census ACS/TIGER 数据，构建带 SHA256 校验、离线缓存和数据质量审计的地理分析流程；从 36,111 条房源记录形成 20,308 条非酒店住宿样本，关联 472 个地铁站点及 NYC 普查区社会经济特征。
- 比较 6 类价格基线与机器学习候选模型，采用 2 公里空间分组五折交叉验证；表现最好的候选模型 MAE 为 **$79.22/晚**，较“房型＋行政区中位数”基线降低 **33.8%**，log1p-price R² 为 **0.742**。
- 通过社区固定效应回归、普查区聚类标准误和特征消融，发现地铁距离在控制社区及房源特征后缺乏稳定的独立价格关联，并交付可复现 Notebook、交互地图与数据来源记录。

如果版面只能放两条，优先第一、第二条；面试时补充第三条，不要把统计关联写成因果溢价。

## English resume version

- Built a reproducible geospatial pipeline combining Inside Airbnb, MTA and Census ACS/TIGER data, with source hashes, offline caching and quality checks; prepared 20,308 non-hotel listings from 36,111 records and engineered access features for 472 subway station locations.
- Compared six candidate price models using 5-fold cross-validation grouped by 2-km spatial cells; the best evaluated candidate achieved **$79.22 nightly-price MAE**, **33.8% below** a room-type/borough median baseline, with **0.742 log1p-price R²**.
- Used neighborhood fixed effects, tract-clustered uncertainty estimates and transit-feature ablation to distinguish observed location-price associations from incremental predictive signal; delivered an executed notebook, interactive map and traceable source documentation.

## 面试需要讲清楚的内容

**为什么排除酒店？** 原始目录中存在大量每晚 4 万、5 万美元的酒店挂牌价，部分酒店又被归为 Private room。按住宿产品类型划分研究范围，比为提高分数任意截断价格更有依据。酒店/旅舍/度假村排除 1,018 条，其余正价格全部保留；完整目录的模型结果另行公开。主样本最少入住天数中位数为 30，不能概括为纯短租市场。

**为什么不能只报随机划分？** 相近位置可能共享环境和价格特征。主结果让同一个 2 公里网格完全落在同一折；网格相邻且房东可能跨折，因此这仍是内部地理迁移测试，不是未来价格或完全独立新区域验证。

**33.8% 是怎么算的？** `(119.650844 - 79.217146) / 119.650844`。这比较的是相同主样本、相同空间折中的平均绝对误差，不是营收提升。中位绝对误差为 $31.32，而 RMSE 为 $545.23，说明保留的极端值仍有很大误差。

**地铁到底有没有影响？** 在 14,001 条完整记录上，加入社区固定效应后，每远离地铁 500 米的估计关联为 +0.80%，95% 区间为 [-0.87%, 2.49%]。区间跨过零，且加入地铁特征没有改善空间验证 MAE。可以说当前数据和方法没有提供稳定的独立溢价证据，不能说证明地铁完全无影响。

**还有什么没做？** 尚未部署成定价服务，也没有未来时间快照测试、房东分组验证、步行路网可达性或真实订单营收。保留这些边界，面试可信度会更高。

## 你需要理解并能修改的代码

依次读 `scope.py` 的样本定义、`data.py` 的 Census 缺失值/投影距离/空间连接、`modeling.py` 的同折基线比较与固定效应回归，最后检查 Notebook 的结果。简历中可以使用本次已验证结果，但应在能解释和复现这些步骤后提交。不要把 2026 年补充的方法描述成 2025 年已经做过。
