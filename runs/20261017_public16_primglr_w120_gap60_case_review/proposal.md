# PrimGLR W=120秒、gap=60秒：16对象十例图审

用户授权生成gap=60秒图表。沿用gap=30秒批次datasets中的活动片段及每对象原定10个review_case，逐条核对活动ID及输入SHA256一致。唯一切分参数变化为gap_seconds从30改为60；W_seconds=120，alpha=.90，dur_sigma=.35，g_wf=16，g_env=8，seed=0。点数按各对象nominal_seconds向上取整，短序列沿用模型的W缩小规则。

四个数据集队列并行，每进程2线程，CPU运行。逐对象登记执行，生成5×2 PNG/SVG，图中显示功率、上下文、切点、日期和划分。不重做数据提取、不改变活动片段，图审不使用evaluation。结果须人工审核，不能自动冻结参数。最终记录与gap30的案例一致性、输入哈希、参数差异及图表可读性。
