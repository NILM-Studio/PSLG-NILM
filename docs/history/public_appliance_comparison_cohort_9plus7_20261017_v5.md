# 公开电器对比实验共用清单：9＋7

日期：2026-10-17，修订v5（16对象主步骤提取与数据双副本已完成）。数据完整性校验通过；PrimGLR切分参数及片段语义仍待人工图审。

## 结论与适用边界

原九对象中保留八个，将序号3的REDD洗碗机替换为REFIT House13洗碗机，再新增七对象，共16个主要比较候选，来自UK-DALE、REFIT、SmartNIALMeter、DARCK四个数据集，约十类电器。文件名保留9plus7以便引用；原REDD对象仅保留于历史报告，不进入本轮清单。

该规模适合作为资源受限的、多种工作机制覆盖的方法比较基准；不足以宣称覆盖所有NILM电器或证明所有类别的跨住宅泛化。多个类别只有一个设备；DARCK对象来自同一住宅；UK-DALE house_1/meter5两设备共享住宅背景。多个数据集上分别训练不等于跨数据集迁移。

所有非生成实验原则上引用本表作为共同候选对象范围，但不应机械地将16对象全部纳入每项主指标。活动质量、独立样本数、方法支持和人工参考标签决定实际评价集合；排除理由必须登记，不可按方法优劣事后选样。

## 活动状态的定义

本项目本轮所说的“活动状态”，指活动状态提取步骤根据阈值、跌落容忍时间`t_drop`、最小时长`t_min_work`等参数提取、并带有效上下文支持的时间片段。以冻结的提取规则和数据支撑为准；允许它对应设备运行的一个子过程。整次洗涤、冲泡或压缩机工作周期不是本轮纳入条件，因此不因片段只覆盖局部过程而额外合并或排除；短片段只按已经声明的最小时长和质量规则处理。片段的数据完整性指核心及所需上下文的观测支撑、缺测／截断处理和边界索引正确，不代表必须覆盖一次完整设备程序。

主项目入口为`src/steps/extract_active_data_step.py`，本次全部使用`models/extract_active_data/simple_threshold.py`。已实际调用主步骤并保存逐条CSV及执行记录。逐对象阈值、跌落容忍、最小时长和上下文参数冻结于本轮configs/plan.json及各source/selected_interval.json。本次增加可选的真实时间上下文展开，并保留小数秒和实际末点支持边界；旧的默认点数上下文方式保留，合约测试通过。

## 对象总表与当前数据交付

本轮注册实验：`runs/20261017_public16_primglr_w120_gap30_case_review`。按UK-DALE、REFIT、SmartNIALMeter、DARCK四路并行执行。16对象共2,824条入选活动状态，核心及上下文全部落在各自选择区间内。咖啡机和冰箱保留连续范围内全部合格片段，无随机抽样；其余14对象按时间块划分后日期分层确定性选样至既定上限。

独立数据根目录：`/home/avis/pslg-nilm/datasets/public16_activity_states_20261017/`。表中对象ID后接`/source/`是连续时间范围的全部原始时间／功率副本；后接`/activities/`是主步骤输出的有上下文支持片段副本。两组均为实体文件，已核验哈希及独立副本。完整路径及读取说明见[数据指引](public16_activity_states_20261017/DATA_GUIDE.md)，机器清单见[manifest.json](public16_activity_states_20261017/manifest.json)。服务器数据根目录也有同名指引和清单。

|序号|对象ID|实际时间范围（起含终不含）|本次主步骤支持数|实际入选数|原定上限|
|---:|---|---|---:|---:|---:|
|1|ukdale_h2_washing_machine|[2013-05-20 21:28:39, 2013-10-10 05:15:26) UTC|54|54|54|
|2|ukdale_h2_microwave|[2013-05-20 21:28:39, 2013-10-10 05:15:32) UTC|439|100|100|
|3|refit_b13_m5_dish_washer|[2014-01-17 22:26:20, 2015-05-31 11:06:15) UTC|101|86|86|
|4|refit_h7_dishwasher|[2013-11-01 22:01:18, 2015-07-08 02:11:02) UTC|531|150|150|
|5|refit_h9_dishwasher|[2013-12-17 17:26:20, 2015-07-08 18:52:58) UTC|254|150|150|
|6|refit_h18_bread_maker|[2014-03-06 16:23:19, 2015-06-20 03:33:02) UTC|211|150|150|
|7|refit_h20_tumble_dryer|[2014-03-07 16:20:20, 2015-07-10 11:56:43) UTC|90|90|91|
|8|ukdale_h1_washing_machine_hotpoint|[2012-11-09 22:28:18, 2015-09-07 23:00:02) UTC|223|150|150|
|9|ukdale_h1_washing_machine_samsung|[2015-09-07 23:00:04, 2017-04-26 17:35:30) UTC|327|150|150|
|10|ukdale_b1_m6_dish_washer|[2012-11-09 22:28:18, 2017-04-26 17:35:36) UTC|184|150|150|
|11|snm_building_01_dishwasher|[2018-03-01 13:22:39.799000, 2020-12-03 00:00:01.097000) 源时区未标注|105|100|100|
|12|snm_building_05_coffee_maker|[2020-03-21 00:00:00, 2020-12-03 00:00:00) 源时区未标注|599|599|599|
|13|snm_building_03_boiler|[2020-02-21 00:00:01.006000, 2020-09-26 08:20:14.976000) 源时区未标注|104|100|100|
|14|darck_kettle|[2025-03-05 00:30:00, 2025-09-05 00:00:00) 源时区未标注|252|100|100|
|15|darck_sherlockhairdryer|[2025-03-05 00:30:00, 2025-09-05 00:00:00) 源时区未标注|133|100|100|
|16|darck_fridge|[2025-07-20 00:00:00, 2025-09-05 00:00:00) 源时区未标注|595|595|599|

REFIT历史h18对应原House19面包机，历史h20对应原House21烘干机；对象ID沿用项目映射。UK-DALE两台house_1/meter5设备按各自设备时期分开。UK-DALE／REFIT显示时间统一为UTC；SmartNIALMeter／DARCK保留未标注时区的源日历口径。

咖啡机范围仍为[2020-03-21, 2020-12-03)，本次599条；冰箱仍为[2025-07-20, 2025-09-05)，本次595条。历史筛选冰箱596条、烘干机91条属于旧实现计数；本次按主步骤及完整上下文范围核验后分别为595、90条，未补样。冰箱配置的599是严格低于600的数量上限，实际交付595条。历史筛选计数和实现差异保存在v4归档，不能直接当作本次执行结果。

原始source保留NaN及观测间断；这表示所选时间范围内的原始记录完整复制，不表示该范围无缺测。活动提取在有效连续支撑段内执行。REFIT活动输入使用既定8秒支撑网格，已从原始副本独立重建核验；source仍保存原生点。各活动CSV与真实主步骤输出逐字节一致，NPZ提供核心和上下文索引。数据分组避免同一时间组跨训练／验证／测试划分。

每对象一张5×2十例图，共16张PNG及16张SVG，使用不同的10个活动片段检查PrimGLR W=120秒、gap=30秒、alpha=.90、dur_sigma=.35。短序列沿用原生模型缩小有效W的规则，逐例记录于figure_manifest.json。图审未使用evaluation侧；本次不自动冻结论文最终切分参数。

## 新增选择依据与质量限制

- UK-DALE及SmartNIALMeter洗碗机：与REFIT组成同类跨来源比较，避免新增类别和新增来源完全混杂。
- 咖啡机：补充初次加热、保温重复脉冲；符合冻结提取规则且有上下文支持的片段可作为活动状态，不另按整次冲泡合并，也不把片段数解释为独立冲泡次数。
- boiler与水壶：补充简单开关负载；检验过度切分、过度聚类和合并恢复简单结构的能力。
- 吹风机：补充离散功率档位；不是所有活动都有档位切换。
- 冰箱：补充启动峰、缓变平台和短负荷片段；符合冻结阈值、最小时长与上下文支撑规则的短片段同样保留，不额外按“完整制冷周期”排除。保留同日片段相关性。
- SmartNIALMeter洗碗机：既有窗口含低功率子过程；核验提取边界和数据支撑，不要求覆盖整次洗碗。
- 替换对象REFIT House13/meter5：筛选报告有1246个阈值片段、86个上下文支持片段；十例中八例呈约1.8–2kW双加热、约57–70分钟过程，另两例较短较弱。86按活动状态口径核验提取规则和数据支撑，不按是否整次洗碗排除短例。600规则依据支持数86而非全部阈值片段1246，不触发本次连续范围限制。
- DARCK：发布端存在插值、前填及补零，原始缺测掩码不可恢复；单列排除DARCK的敏感性汇总。
- 本轮不新增DEDDIAG、GREEND及其他REDD对象，以控制事件流适配、缺测处理与审核成本。不使用三个自采CSV或缺少时间戳的旧NPZ。

## 各实验如何使用本表

章节号对应当前essay/proposal.md；本清单不自动改写该文件的实验矩阵。

| 实验 | 建议对象范围 | 评价与共同支持要求 |
|---|---|---|
| 活动状态提取及可视化审核 | 全16对象 | 每对象一张10子图原生时间序列图；不足10例留空；展示核心边界、上下文、日期、缺测和截断。审核规则先冻结，不按下游分数筛选，不另按完整设备程序选样。 |
| 实验一：六种切分比较 | 16主候选 | 使用相同活动状态片段及时间支持；简单开关型及符合提取规则的局部过程同样保留。若需要边界F1，必须另有独立人工边界，不能拿PrimGLR输出作真值。 |
| 实验二：七种特征表示比较 | 16主候选 | 使用相同PrimGLR片段池及活动划分；每个模型仅在训练侧拟合。表示退化或邻域指标无定义如实报告，不补零。 |
| 实验三：表示＋K-means，K=2–8 | 16主候选 | 固定片段池；按对象报告K曲线、支持量和退化情况，不要求所有对象都有多个真实状态。K=1建议仅作另行冻结的诊断参照，不替换主K范围。 |
| 实验四：不同切分方法下的聚类 | 16主候选 | 使用同一总体选定的表示方案、相同活动输入；各方法片段数不同是实验结果，不强行匹配。模型失败及共同有效覆盖同时报告。 |
| 实验五：状态合并 | 16主候选 | 每个对象固定上游，对相同活动做合并前后对照；简单对象检验冗余合并，复杂对象检验结构保留。无定义指标不能直接当0或删去对象。 |
| 实验七：独立人工评价及标签对齐 | 以16主候选为抽样母集 | 独立盲评尽量覆盖全部16对象，按机制分层并与选K样本分离。ARI/边界F1等仅在有独立人工标注的共同支持子集计算；没有标签的设备不能直接用于标签对齐。既有非电器跨域有标签补充集不由本表替代。 |
| RQ2受控扰动验证 | 16主候选中的预声明机制分层子集 | 扰动必须适用于相应结构；简单对象不强行造多阶段任务。报告适用对象、扰动程度及无定义情况，不能只选有利对象。 |
| 实验六：生成比较 | 后续从16主候选中冻结10–12对象 | 按合格活动状态片段数量、机制配额、模型支持选取，不按P−U提升选样；不承诺当前所有对象均满足生成统计需求。 |

“共同对象范围”不意味着各实验必须使用同一派生片段：切分比较共用原活动，表示比较共用固定片段池，合并比较共用冻结上游；人工对齐另需独立参考标注。

## 数量控制、划分与统计

1. 序号12、16先限定在总表的连续范围内，整条排除核心跨边界的活动状态片段，再按整条片段和时间块确定训练／验证／测试；范围内所有合格活动状态片段均保留，不随机抽样。上下文是否允许读取范围外的邻接数据须在执行协议中冻结并核验。其余对象先依据整条活动状态片段和时间块确定划分，再在各划分内按日期分层抽样，合计不超过表中上限。相邻且上下文重叠的片段同组，设备更换边界不跨接。实际日期、数量、选样规则和对象标识写入运行清单；仅实际使用随机抽样的对象记录抽样种子。
2. 归一化、编码器、DTW原型和K-means中心只在训练侧拟合；验证侧选模型、K与参数，测试侧冻结应用。已有全历史目视筛选和开发使用必须披露，重新切分不等于获得完全未见的确认集。
3. 所有比较方法引用同一冻结输入清单及其哈希。失败、不支持及无定义值均登记；共同有效子集的比较须同时列出全体覆盖，不能通过排除失败对象隐藏方法局限。
4. 保留原生时间及有效支持；REFIT沿用明确的8秒支持网格，不强制所有来源为6秒。计算量优先通过片段数和并发控制，不裁断已提取的活动状态片段。生成沿用512点相位表示，但不把512点当固定秒数。
5. 对象内成对比较，按活动／时间块重采样；再按设备、类别汇总，补充来源分层和排除DARCK结果。种子、片段、K值不当独立设备；同住宅依赖需说明。
6. 各对象上限（多数100–150，序号12、16在固定连续范围内目前为599、596）属于有限样本预算，不是统计充分性保证。两个连续范围的结论仅覆盖所选时期，不能直接外推到该设备全部历史。尤其生成实验须报告实际真实测试活动数与不确定性，重复生成500条不能补足独立真实测试活动。
7. 后续实验在runs内按README登记；本文件同时记录共用选样规范及本次实际数据准备执行；本轮未启动训练。各实验protocol.md应引用本文并冻结具体输入，不仅引用设备名称。

## 历史片段位置与保存状态（当前交付以上表及datasets清单为准）

以下位置均于2026-10-17在`ssh zhouslab`实际核实，绝对仓库根目录为`/home/avis/pslg-nilm/`。本地同版文档用于索引，不表示服务器全量波形已复制到本地。这里列的是活动提取结果；后续PrimGLR切分得到的基元属于另一层数据。

### 既有八对象：已保存逐活动波形

公共前缀A：`/home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/`。下表路径接在A后。每个目录均含：`activities.json`（边界、活动ID、左右截断标记、核心在波形内的索引）、`activities/<五位activity_id>.npz`（逐活动原生时间／功率及上下文）、`activity_review.csv`和`audit.json`。NPZ字段为`timestamp`、`power`、`end_timestamp`；核心范围由清单的`core_a:core_b`指定。全量文件数包含截断活动，取总表候选须先按`left_censored=false`且`right_censored=false`过滤，再核验核心／上下文数据支撑及文件对应关系。

| 位置编号／总表序号 | 对象ID | A后相对目录 | 全量逐活动NPZ数／未截断候选数 |
|---|---|---|---:|
| A01／1 | ukdale_h2_washing_machine | `prepare_ukdale_h2_washing_machine/20261016T120241Z_1c3a3021/analysis/ukdale_h2_washing_machine/` | 56／54 |
| A02／2 | ukdale_h2_microwave | `prepare_ukdale_h2_microwave/20261016T120241Z_bf9046f8/analysis/ukdale_h2_microwave/` | 428／421 |
| A04／4 | refit_h7_dishwasher | `prepare_refit_h7_dishwasher/20261016T120249Z_4254abca/analysis/refit_h7_dishwasher/` | 725／568 |
| A05／5 | refit_h9_dishwasher | `prepare_refit_h9_dishwasher/20261016T120254Z_a5ccb756/analysis/refit_h9_dishwasher/` | 456／269 |
| A06／6 | refit_h18_bread_maker | `prepare_refit_h18_bread_maker/20261016T120347Z_27e20277/analysis/refit_h18_bread_maker/` | 321／256 |
| A07／7 | refit_h20_tumble_dryer | `prepare_refit_h20_tumble_dryer/20261016T120349Z_e153b2e3/analysis/refit_h20_tumble_dryer/` | 136／91 |
| A08／8 | ukdale_h1_washing_machine_hotpoint | `prepare_ukdale_h1_washing_machine_hotpoint/20261016T120440Z_daf29836/analysis/ukdale_h1_washing_machine_hotpoint/` | 890／265 |
| A09／9 | ukdale_h1_washing_machine_samsung | `prepare_ukdale_h1_washing_machine_samsung/20261016T120440Z_26aed33b/analysis/ukdale_h1_washing_machine_samsung/` | 506／358 |

入口映射：`/home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/data/prepared_map.json`。该映射还保留历史REDD对象，本轮只使用上表八对象。

### 替换及新增八对象：全量边界索引＋十例展示波形

公共前缀B：`/home/avis/pslg-nilm/runs/20261016_public_appliance_signature_screening/executions/actual_time_metric_harmonization/20261016T141009Z_69e251aa/analysis/object_results/`。下表对象目录接在B后。

每个目录实际包含`all_fragment_bounds.json`（全量阈值片段的核心／上下文索引、时间和支撑标记）、`result.json`（原始源、通道、提取参数和读取记录）、`frozen_plot_selection.json`（十例选样和哈希）、`selected_native_waveforms.npz`（仅十例的`t0/p0`至`t9/p9`时间／功率数组）。总表支持数按`observed_supported_complete=true`统计，属于本轮活动状态候选。**该历史筛选批次没有每对象全量逐活动NPZ目录**，十例包也不是按本轮截取范围重新导出的数据。

| 位置编号／总表序号 | B后对象目录 | 全量边界数／全历史支持数 | 当前保存的展示波形数 |
|---|---|---:|---:|
| B03／3 | `refit_b13_m5_dish_washer/` | 1246／86 | 10 |
| B10／10 | `ukdale_b1_m6_dish_washer/` | 838／184 | 10 |
| B11／11 | `snm_building_01_dishwasher/` | 594／105 | 10 |
| B12／12 | `snm_building_05_coffee_maker/` | 6434／5311；指定范围内支持599 | 10 |
| B13／13 | `snm_building_03_boiler/` | 881／104 | 10 |
| B14／14 | `darck_kettle/` | 254／254 | 10 |
| B15／15 | `darck_sherlockhairdryer/` | 132／132 | 10 |
| B16／16 | `darck_fridge/` | 2245／2244；指定范围内支持596 | 10 |

B03、B10、B11、B12、B13的`result.json.reader_record`引用的历史`native_scratch`二进制缓存当前已不存在；后续导出全量活动须按源数据和适配器重新读取，不能把旧缓存路径当作可用输入。原始文件实际存在：B03为`datasets/refit/refit.h5`，B10为`datasets/ukdale/ukdale.h5`，B11／B12／B13分别为`datasets/smartnialmeter/native_raw/raw/building_01/dishwasher.h5`、`building_05/coffee_maker.h5`、`building_03/boiler.h5`。具体住宅／通道与功率列见各自`result.json.object`；SmartNIALMeter两台三相设备仅在同一时刻三相均有效时求和。

DARCK三个对象的可用原生缓存公共前缀C：`/home/avis/pslg-nilm/runs/20261016_public_appliance_signature_screening/executions/darck_shared_decode_cache/20261016T131444Z_4c9495dc/features/native_cache/`。三者共用`darck_shared.i64`时间数组，功率分别为`darck_kettle.f64`、`darck_sherlockhairdryer.f64`、`darck_fridge.f64`；可配合B中各对象的边界索引读取完整片段。缓存属于发布者已填补的1秒数据，原始文件为`/home/avis/pslg-nilm/datasets/darck/DARCK.csv.xz`，不能据缓存恢复原始缺测掩码。

上述A／B为历史证据。本轮连续截取副本、主步骤活动片段及PrimGLR十例图已经保存于文首当前交付目录。

## 版本记录

- v5（2026-10-17）：完成16对象主步骤提取、datasets中两组独立副本及全部校验，共2824条；总表落实实际区间、支持数和入选数，生成16张十例图，参数待人工审核。[历史v4](history/public_appliance_comparison_cohort_9plus7_20261017_v4.md)。

- v2（2026-10-17）：替换REDD对象，并将咖啡机／冰箱的原100上限改为600和日期分层抽样。原文SHA256：`e3ac68d324dbf658eadddd3ab0bcd4dd393e4095adb47ba36982dc6f59d5cb7e`；保存于[历史v2](history/public_appliance_comparison_cohort_9plus7_20261017_v2.md)。
- v3（2026-10-17）：依据用户要求，序号12、16改为总表固定连续范围，当前支持片段599／596；增加16对象的实际提取结果路径、全量／展示包区别及失效缓存说明。未执行范围数据导出或PrimGLR新实验。
- v4（2026-10-17）：按用户澄清，活动状态指有上下文支持的提取片段；移除整次冲泡／制冷／洗涤程序的额外选样要求，保留片段数据支撑核验；说明筛选提取器与主步骤实现差异。连续范围与现有计数不变，未执行重新提取或全量导出。v3原文SHA256为`1c7e1e49ead7493ae74a9cb39099728102798a949e77a457cfbfa7fcc28e9527`，保存于[历史v3](history/public_appliance_comparison_cohort_9plus7_20261017_v3.md)。

## 证据来源

- [论文实验设计](../essay/proposal.md)，重点5.1、5.2、5.8、5.9。
- [原九对象全量报告](../runs/20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_house1_increment_final_20261016T084158Z_01699ad3/report.md)。
- [新增电器筛选报告](../runs/20261016_public_appliance_signature_screening/reports/final_20261016T142157Z_497430b2/report.md)。

引用时建议注明：`docs/public_appliance_comparison_cohort_9plus7.md`（2026-10-17 v5版），并保存该文件及实际输入清单的SHA256。调整对象、上限或资格规则时保留版本记录，不回写已执行批次的冻结输入。
