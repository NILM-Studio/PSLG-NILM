# 真实状态块组合：Unit Selection 第一阶段

服务器回传已暴露全时期筛选后验证/测试为空的问题。下一轮请使用[按时期修复评价协议入口](cohort_composition_repair_20260917.md)，不要直接重跑下文的全时期默认入口。

本轮在 `af7dbd2` 的四组严格配对实验上增加一个有文献依据的选择框架，并固定第一轮指标。仅使用师兄已有切分、特征和聚类产物；不改上游算法，不训练师弟的波形生成模型。真实 UK-DALE 效果需要服务器报告确认，fixture 成功只说明实现可运行。

## 方法来源与本轮实现

- Hunt & Black (ICASSP 1996), [Unit Selection in a Concatenative Speech Synthesis System Using a Large Speech Database](https://www.cstr.inf.ed.ac.uk/downloads/publications/1996/Hunt_1996_a.pdf)：目标匹配成本、相邻连接成本和全局 Viterbi 选择。
- Clark, Richmond & King (2007), [Multisyn: Open-domain unit selection for the Festival speech synthesis system](https://doi.org/10.1016/j.specom.2007.01.014)：该方法家族的公开实现参考；[Festival/MultiSyn](https://github.com/festvox/festival/tree/master/src/modules/MultiSyn)。

这里只移植目标成本加连接成本的选择框架，不宣称复现语音系统的声学成本，也没有复制 Festival 代码。沿用已验证的分层动态规划即可，不需新训练一个生成网络。

五组均共享锚点、状态顺序、目标时长、每位置候选池、首块和插值，均无平滑：

| 方法 | 路径选择 |
|---|---|
| `random` | 随机选择，保留原行为 |
| `boundary_greedy` | 逐边端点贪心，保留原行为 |
| `boundary_dp` | 全局端点成本，保留原行为 |
| `transition_dp` | 全局经验转移成本，保留原行为 |
| `unit_selection` | 全局经验转移成本 + 时长目标成本 |

对候选 `u_i`，目标成本为 `abs(log(target_length_i / source_length_i))`，对所有状态块求和。它倾向使用变形较少的供体，不使用锚点真实功率来选候选。连接成本复用仅由同预算、同 Class/Mode、非锚点训练供体建立的 `TransitionReference`。总目标为：

`unit_selection_objective = transition_objective + target_weight * duration_target_cost`

首轮 `target_weight=1`，是预先固定的诊断设置，不是调参后的最优值。为零时，新方法应与 `transition_dp` 完全一致。状态与时长固定，因此仅增加状态转移概率或时长概率不能区分候选；这里使用的是每个供体的原始时长与目标时长的差异。

仍需保留的限制：目标、经验距离、边界回退三者没有统计尺度校准。新方法同样可能偏好单个供体重放；全回退时，新方法是“时长 + 边界”选择，不再等价于 `boundary_dp`。全/部分/无支持分别汇报，不混成一个效果结论。支持状态按整个案例标记；partial 案例中的某种状态转移不能据此视为已经获得足够供体支持。

本轮不加入 DTW-Merge 或整周期重放作为第六组：原始版本不满足相同状态模板和候选池，后续应以单独协议对照。现有整周期最近供体指标先用于识别重放。

## 第一轮固定指标

不设置加权总分、不按训练优化成本选最佳方法。生成完成后才读取验证集进行以下描述；测试集不用于方法评价或选择。新的统计位于 `composition_summary.json` 的 `evaluation`。

| 类别 | 指标和口径 | 解读 |
|---|---|---|
| 覆盖与可比性 | 每预算 requested/completed/skipped、跳过原因；全/部分/无支持数；至少3状态块且中间位置有多个候选的案例数 | 先确认是否有足够案例实际区分方法；两个状态块无法区分固定首块的边界贪心与DP |
| 主要接缝诊断 | 按有向状态对、Class/Mode、预算、种子和支持状态分别比较**带符号功率跳变**的验证集 Wasserstein-1，单位 W | 越小表示这一分布更接近；不能把跳变本身越小当成越好 |
| 局部动态 | 同样分组的左右局部斜率 W1，单位 W/s；取块内末/首5点相邻差均值除以采样周期 | 不跨接缝求斜率、不平滑，单点块记0；与选择模型的可配 window 参数无关 |
| 周期形态统计 | 按 Class/Mode、预算、种子和支持状态比较能量 Wh、平均功率 W、峰值 W 的验证集 W1 | 必须联合判断，单一能量接近不代表状态转移合理 |
| 重放与来源集中 | 单供体比例；最大供体样本占比；最近供体整周期重采样 NRMSE及来源ID | NRMSE接近0提示重放，越大也不自动更好；无“超过某阈值就新颖”的判定 |
| 工程完整性 | NPZ哈希、候选来源、状态顺序、时长、首块共享、诊断重算和目标成本算术检查 | 审计通过不证明物理正确，也不重新读取源CSV重建供体 |
| 优化诊断 | 三类优化成本及新方法时长目标成本 | 只能解释选择行为，不作为独立效果证据 |

所有分布保留 count/median/p10/p90；没有同组或同转移验证数据时输出 null 距离和明确状态，少于3个不同验证周期标记 `low_support`。3只是报告提示阈值，不意味着3例就有统计充分性。重复接缝不能算多个独立周期；不同生成锚点共享供体，也不能当独立重复实验计算显著性。

最近供体 NRMSE 定义：将每个可用训练供体整周期线性适配到输出总长度，计算 RMSE，除以 `max(该适配供体峰值, 1 W)`，取最小值。它可能掩盖局部差异，因此必须结合逐块来源和图查看。

第一轮候选结论只能是“覆盖足够且值得进一步验证”或“覆盖/回退/重放/分布偏差需要处理”，不能仅凭成本下降宣布方法有效。后续多种子/NILM需另行决定，且不能用测试集选权重。

## 已有报告优先：只生成图表，不重跑

如果服务器已经有旧四组或新五组 `composition` 目录，先执行只读报告命令：

```bash
python -m scripts.report_primitive_composition \
  --study-dir log/你的已有run/composition \
  --output-dir log/你的已有run/composition_review_v1 \
  --max-cases 6
```

报告工具先独立审计，然后在全新目录写 `report.md`、`cases.csv` 和 PNG。已有输出目录会拒绝，旧 manifest/summary/NPZ 不被修改。旧四组 schema 1 仍可审计和画图，但不会伪造不存在的新指标或第五组结果。首轮图先按种子、预算、类别和锚点排序，再轮流从不同预算取图，并优先覆盖不同类别；不按分数挑好看的案例。

PNG 用相同时间轴和功率范围排列各方法，新版首行加入真实训练锚点，仅供观察模板形态，不把它作为重建目标或候选功率匹配目标。图中标出状态边界、状态编号及供体ID；图中状态仍是经验标签，不是人工确认的电器动作。完整案例仍保留在 `cases.csv` 与 manifest 中。

## 没有本轮结果时：由用户在服务器启动新run

继续使用开发端修改/测试/推送、用户服务器拉取/运行/回传的流程。不要在开发端猜测服务器执行情况，也不要自动启动服务器实验。

```bash
cd /home/scnu2024024563/NILM/PSLG-NILM
git switch feature/primitive-generation
git pull --ff-only origin feature/primitive-generation
conda activate pslg-nilm
bash scripts/run_composition_validation.sh
```

默认引用 `ukdale_wm_primglr_detsec_3789`，新建 `ukdale_wm_composition_units_20260917`，避免与旧四组run混淆。CPU运行，无需 `slurm/env.sh`。预算5%、10%、20%、100%，seed42，每预算最多30锚点、每位置最多8候选、时长比例[0.5,2]、至少3独立供体周期支持每种转移。

可用 `SOURCE_RUN_ID`、全新的 `RUN_ID`、`PYTHON_BIN` 指定环境；`TARGET_WEIGHT` 默认1、`REPORT_CASES` 默认6。不要为了得到好看结果不断改变权重。目标run存在会拒绝，不覆盖失败或成功产物。初始化发生在日志重定向之前，若创建run阶段失败，回传终端报错即可。

回传：

1. `composition_review/report.md` 与代表性 PNG；
2. `composition/composition_summary.json`；
3. `upstream_validation.json` 和 `composition_execution.log`；
4. 如需追踪具体案例，再提供 `composition/composition_manifest.json`。

旧上游覆盖/时间接口不通过时停止，不擅自修补或重新切分聚类。零有效案例仍保留诊断报告，并以非零退出码结束。输出依然只有电器支路功率，尚不能直接送入 Seq2Point，不要求本轮准备总表文件。

## 开发端验证记录

2026-09-17：全量 `pytest tests -q -p no:cacheprovider` 通过 **252项测试**，`bash -n scripts/run_composition_validation.sh` 和 `git diff --check` 通过。覆盖含新路径的穷举对照、零目标权重退化、原四组路径保持、预算隔离、评价分层、旧四组报告兼容、产物篡改拒绝和已有结果保护。

已用测试夹具走通拼接、独立审计、统计报告与 PNG，并人工检查图像布局。这些是工程验证，**不是服务器真实数据结果**。真实数据尚未在本轮启动或重新运行。
