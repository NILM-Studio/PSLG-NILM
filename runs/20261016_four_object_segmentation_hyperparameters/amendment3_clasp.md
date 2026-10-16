# Amendment 3: ClaSP significance threshold range

记录时间：2026-10-16。首轮 ClaSP 开发筛参仅测试官方默认阈值（p=1e-15 或更严格）与 1e-8。H7 成功运行的 68 个活动×配置中只有 3 个产生内部切点；H20 为 54 个中 4 个。它们虽可执行，但多数候选无法供 CLaP 初始化出状态转换。根据官方 `claspy.BinaryClaSPSegmentation` 源码，`validation="significance_test"` 时 `threshold` 是 p 值上限，因此在 `configs/search_space_amendment3.json` 中扩为 `['default',1e-8,1e-4,0.01]`；窗口与其他参数不变。

本修订依据开发侧切点稀少现象提出。首轮验证曾开始执行，但没有用其数值决定新网格；旧执行保留为历史，不与新网格混排。新网格四对象重新登记完整开发筛参及验证。较大 p 值可能引入假阳性，必须通过同活动十子图逐切点人工审核；即使机器得分较好也不自动冻结。CLaP 的探索性图库使用新网格的暂定 ClaSP 初始边界，最终仍依赖用户选定 ClaSP 参数。
