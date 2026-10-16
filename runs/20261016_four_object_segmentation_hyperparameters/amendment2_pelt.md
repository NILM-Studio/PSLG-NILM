# Amendment 2: PELT computational step

记录时间：2026-10-16。初始 PELT-RBF 搜索固定 `jump=1`，REFIT H20 开发活动 ID 47 长 1,757 点；其执行在前 6/12 条活动完成后，第 7 条运行约 31 分钟仍未结束。该执行经人工终止，`run.json` 保留 `exit_code=-15`，不是候选算法的结构失败，也不计入质量排名。

改用作者库 `ruptures.Pelt(model="rbf", jump=5)` 的**官方默认步长**，其余惩罚与最小段长网格不变，重跑四对象的完整开发筛参并另行登记验证。新的候选配置位于 `configs/search_space_amendment2.json`，每条记录显式包含 `jump=5`；旧 `jump=1` 执行保留但不与新配置混排。`jump=5` 的切点位置分辨率为五个采样点，应在十子图人工审核及最终报告中注明。搜索规则仍按既定成功数和综合得分排序，只有人审后才可冻结。
