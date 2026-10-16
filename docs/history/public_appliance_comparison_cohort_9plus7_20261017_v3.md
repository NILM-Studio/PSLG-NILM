# 公开电器对比实验共用清单：9＋7

日期：2026-10-17，修订v3（落实咖啡机／冰箱连续截取范围，补充16对象提取结果位置）。用途：供后续实验方案和输入清单引用；本文是选样设计，不是执行结果，也不是全部活动已经审核通过的声明。

## 结论与适用边界

原九对象中保留八个，将序号3的REDD洗碗机替换为REFIT House13洗碗机，再新增七对象，共16个主要比较候选，来自UK-DALE、REFIT、SmartNIALMeter、DARCK四个数据集，约十类电器。文件名保留9plus7以便引用；原REDD对象仅保留于历史报告，不进入本轮清单。

该规模适合作为资源受限的、多种工作机制覆盖的方法比较基准；不足以宣称覆盖所有NILM电器或证明所有类别的跨住宅泛化。多个类别只有一个设备；DARCK对象来自同一住宅；UK-DALE house_1/meter5两设备共享住宅背景。多个数据集上分别训练不等于跨数据集迁移。

所有非生成实验原则上引用本表作为共同候选对象范围，但不应机械地将16对象全部纳入每项主指标。活动质量、独立样本数、方法支持和人工参考标签决定实际评价集合；排除理由必须登记，不可按方法优劣事后选样。

## 对象总表

序号1、2、4–9的数量为既有未截断核心数；替换后的序号3及新增七行是筛选报告的上下文支持片段数，两者不具有完全相同含义。支持片段不保证完整工作程序。上限指全部训练、验证、测试活动之和，不是每个划分的数量；既有全量结果保持原样。已有核心／支持数超过600的序号12、16，各使用下表的一个连续时间范围，保留完整落入范围的合格活动，不再随机抽样；其余对象保持原定上限及选样规则。两段范围内目前分别有599、596个支持片段，最终整次活动／周期数仍须审核。

| 序号 | 状态 | 数据集 | 住宅／通道或对象标识 | 电器 | 既有核心／支持数 | 本轮使用上限 | 本轮连续截取范围（起点含，终点不含） | 提取结果位置编号 |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | 保留 | UK-DALE | house_2 / meter12 | 洗衣机 | 54 | 54 | 原设备历史；按上限选样 | A01 |
| 2 | 保留 | UK-DALE | house_2 / meter15 | 微波炉 | 421 | 100 | 原设备历史；按上限选样 | A02 |
| 3 | 替换REDD，需边界审核 | REFIT | 原House13 / meter5；refit_b13_m5_dish_washer | 洗碗机 | 86（支持片段） | 86 | 原设备历史；按上限选样 | B03 |
| 4 | 保留 | REFIT | 原House7 / meter7 | 洗碗机 | 568 | 150 | 原设备历史；按上限选样 | A04 |
| 5 | 保留 | REFIT | 原House9 / meter5 | 洗碗机 | 269 | 150 | 原设备历史；按上限选样 | A05 |
| 6 | 保留 | REFIT | 原House19 / meter8，历史h18 | 面包机 | 256 | 150 | 原设备历史；按上限选样 | A06 |
| 7 | 保留 | REFIT | 原House21 / meter3，历史h20 | 滚筒烘干机 | 91 | 91 | 原设备历史；按上限选样 | A07 |
| 8 | 保留 | UK-DALE | house_1 / meter5，Hotpoint时期 | 洗烘一体机 | 265 | 150 | 仅Hotpoint设备时期；按上限选样 | A08 |
| 9 | 保留 | UK-DALE | house_1 / meter5，Samsung时期 | 洗衣／洗烘，保留元数据口径 | 358 | 150 | 仅Samsung设备时期；按上限选样 | A09 |
| 10 | 新增 | UK-DALE | ukdale_b1_m6_dish_washer | 洗碗机 | 184 | 150 | 原设备历史；按上限选样 | B10 |
| 11 | 新增，需边界审核 | SmartNIALMeter | snm_building_01_dishwasher | 洗碗机 | 105 | 100 | 原设备历史；按上限选样 | B11 |
| 12 | 新增，需完整活动定义 | SmartNIALMeter | snm_building_05_coffee_maker | 咖啡机 | 5311；范围内599 | 599（完整活动审核后可减少） | **[2020-03-21 00:00:00, 2020-12-03 00:00:00)** | B12 |
| 13 | 新增 | SmartNIALMeter | snm_building_03_boiler | 电热水器／锅炉，原标签boiler | 104 | 100 | 原设备历史；按上限选样 | B13 |
| 14 | 新增 | DARCK | darck_kettle | 电水壶 | 254 | 100 | 原设备历史；按上限选样 | B14 |
| 15 | 新增 | DARCK | darck_sherlockhairdryer | 吹风机 | 132 | 100 | 原设备历史；按上限选样 | B15 |
| 16 | 新增，需周期审核 | DARCK | darck_fridge | 冰箱 | 2244；范围内596 | 596（周期审核后可减少） | **[2025-07-20 00:00:00, 2025-09-05 00:00:00)** | B16 |

按当前范围及上限，主要比较最多2826个审核合格活动／周期。实际合格数量不足时减少，不重复抽样充数。新旧对象都需检查活动完整性，不能将未截断自动等同于完整程序。A／B位置编号见下文“片段位置与保存状态”。

两个截取范围按源记录的日历日期解释；SmartNIALMeter、DARCK源元数据未标注时区，不额外声称为UTC或北京时间。2026-10-17依据B12／B16的`all_fragment_bounds.json`复核：只计`observed_supported_complete=true`且`start_ns >= 范围起点`、`end_ns < 范围终点`的片段。咖啡机范围内全部阈值片段为1019个，其中支持片段599个；冰箱范围内596个均有上下文支持。冰箱另有2025-07-19 22:02:00至2025-07-20 00:21:28的跨起点片段，整条排除，不能裁成新活动。范围前一日开始时，支持片段分别为603及611个，故上述边界是以既有观察末端为终点、按整日向前扩展且不超过600个支持片段的最早起点。

599／596不等于完整冲泡／制冷周期已审核合格。固定范围内仍须合并同次冲泡脉冲、排除辅助短片段并核验缺测和上下文；若活动定义修订导致数量超过600，应另行修订范围和版本记录，不暗中随机删样。这里落实的是输入范围与可追溯位置；本轮范围专用的全量活动波形副本尚未生成，后续注册执行需保存实际输入清单及哈希。

## 新增选择依据与质量限制

- UK-DALE及SmartNIALMeter洗碗机：与REFIT组成同类跨来源比较，避免新增类别和新增来源完全混杂。
- 咖啡机：补充初次加热、保温重复脉冲；先定义整次活动，不把每个保温脉冲当一次独立冲泡。
- boiler与水壶：补充简单开关负载；检验过度切分、过度聚类和合并恢复简单结构的能力。
- 吹风机：补充离散功率档位；不是所有活动都有档位切换。
- 冰箱：补充压缩机周期、启动峰和缓变平台；排除辅助秒级片段，保留同日周期相关性。
- SmartNIALMeter洗碗机：既有窗口含低功率子过程，需审核是否覆盖整次洗碗。
- 替换对象REFIT House13/meter5：筛选报告有1246个阈值片段、86个上下文支持片段；十例中八例呈约1.8–2kW双加热、约57–70分钟程序，另两例较短较弱，故86仍须审核。600规则依据支持数86而非全部阈值片段1246，不触发本次连续范围限制。
- DARCK：发布端存在插值、前填及补零，原始缺测掩码不可恢复；单列排除DARCK的敏感性汇总。
- 本轮不新增DEDDIAG、GREEND及其他REDD对象，以控制事件流适配、缺测处理与审核成本。不使用三个自采CSV或缺少时间戳的旧NPZ。

## 各实验如何使用本表

章节号对应当前essay/proposal.md；本清单不自动改写该文件的实验矩阵。

| 实验 | 建议对象范围 | 评价与共同支持要求 |
|---|---|---|
| 活动状态提取及可视化审核 | 全16对象 | 每对象一张10子图原生时间序列图；不足10例留空；展示边界、上下文、日期及缺测。审核规则先冻结，不按下游分数筛选。截断和不完整程序均明确标记。 |
| 实验一：六种切分比较 | 16主候选 | 使用相同完整活动及时间支持；简单开关型同样保留。若需要边界F1，必须另有独立人工边界，不能拿PrimGLR输出作真值。 |
| 实验二：七种特征表示比较 | 16主候选 | 使用相同PrimGLR片段池及活动划分；每个模型仅在训练侧拟合。表示退化或邻域指标无定义如实报告，不补零。 |
| 实验三：表示＋K-means，K=2–8 | 16主候选 | 固定片段池；按对象报告K曲线、支持量和退化情况，不要求所有对象都有多个真实状态。K=1建议仅作另行冻结的诊断参照，不替换主K范围。 |
| 实验四：不同切分方法下的聚类 | 16主候选 | 使用同一总体选定的表示方案、相同活动输入；各方法片段数不同是实验结果，不强行匹配。模型失败及共同有效覆盖同时报告。 |
| 实验五：状态合并 | 16主候选 | 每个对象固定上游，对相同活动做合并前后对照；简单对象检验冗余合并，复杂对象检验结构保留。无定义指标不能直接当0或删去对象。 |
| 实验七：独立人工评价及标签对齐 | 以16主候选为抽样母集 | 独立盲评尽量覆盖全部16对象，按机制分层并与选K样本分离。ARI/边界F1等仅在有独立人工标注的共同支持子集计算；没有标签的设备不能直接用于标签对齐。既有非电器跨域有标签补充集不由本表替代。 |
| RQ2受控扰动验证 | 16主候选中的预声明机制分层子集 | 扰动必须适用于相应结构；简单对象不强行造多阶段任务。报告适用对象、扰动程度及无定义情况，不能只选有利对象。 |
| 实验六：生成比较 | 后续从16主候选中冻结10–12对象 | 按完整活动数量、机制配额、模型支持选取，不按P−U提升选样；不承诺当前所有对象均满足生成统计需求。 |

“共同对象范围”不意味着各实验必须使用同一派生片段：切分比较共用原活动，表示比较共用固定片段池，合并比较共用冻结上游；人工对齐另需独立参考标注。

## 数量控制、划分与统计

1. 序号12、16先限定在总表的连续范围内，整条排除跨边界活动，再按完整活动和时间块确定训练／验证／测试；范围内所有合格活动均保留，不随机抽样。其余对象先依据完整活动和时间块确定划分，再在各划分内按日期分层抽样，合计不超过表中上限。相邻且上下文重叠的活动同组，设备更换边界不跨接。实际日期、数量、选样规则和对象标识写入运行清单；仅实际使用随机抽样的对象记录抽样种子。
2. 归一化、编码器、DTW原型和K-means中心只在训练侧拟合；验证侧选模型、K与参数，测试侧冻结应用。已有全历史目视筛选和开发使用必须披露，重新切分不等于获得完全未见的确认集。
3. 所有比较方法引用同一冻结输入清单及其哈希。失败、不支持及无定义值均登记；共同有效子集的比较须同时列出全体覆盖，不能通过排除失败对象隐藏方法局限。
4. 保留原生时间及有效支持；REFIT沿用明确的8秒支持网格，不强制所有来源为6秒。计算量优先通过活动数和并发控制，不裁断活动程序。生成沿用512点相位表示，但不把512点当固定秒数。
5. 对象内成对比较，按活动／时间块重采样；再按设备、类别汇总，补充来源分层和排除DARCK结果。种子、片段、K值不当独立设备；同住宅依赖需说明。
6. 各对象上限（多数100–150，序号12、16在固定连续范围内目前为599、596）属于有限样本预算，不是统计充分性保证。两个连续范围的结论仅覆盖所选时期，不能直接外推到该设备全部历史。尤其生成实验须报告实际真实测试活动数与不确定性，重复生成500条不能补足独立真实测试活动。
7. 后续实验在runs内按README登记；本文件只是共用选样规范，本次不启动训练、不登记虚构执行。各实验protocol.md应引用本文并冻结具体输入，不仅引用设备名称。

## 片段位置与保存状态

以下位置均于2026-10-17在`ssh zhouslab`实际核实，绝对仓库根目录为`/home/avis/pslg-nilm/`。本地同版文档用于索引，不表示服务器全量波形已复制到本地。这里列的是活动提取结果；后续PrimGLR切分得到的基元属于另一层数据。

### 既有八对象：已保存逐活动波形

公共前缀A：`/home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/`。下表路径接在A后。每个目录均含：`activities.json`（边界、活动ID、左右截断标记、核心在波形内的索引）、`activities/<五位activity_id>.npz`（逐活动原生时间／功率及上下文）、`activity_review.csv`和`audit.json`。NPZ字段为`timestamp`、`power`、`end_timestamp`；核心范围由清单的`core_a:core_b`指定。全量文件数包含截断活动，取总表候选须先按`left_censored=false`且`right_censored=false`过滤，再做程序完整性审核。

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

每个目录实际包含`all_fragment_bounds.json`（全量阈值片段的核心／上下文索引、时间和支撑标记）、`result.json`（原始源、通道、提取参数和读取记录）、`frozen_plot_selection.json`（十例选样和哈希）、`selected_native_waveforms.npz`（仅十例的`t0/p0`至`t9/p9`时间／功率数组）。总表支持数按`observed_supported_complete=true`统计。这些文件未证明完整工作程序；**目前没有每对象全量逐活动NPZ目录**，十例包也不是按本轮截取范围重新导出的数据。

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

本轮连续截取后的完整活动副本、质量审核及PrimGLR `W=120s,gap=30s`结果须由后续注册执行另行保存；本表仅将已经存在的文件与待导出内容如实登记。

## 版本记录

- v2（2026-10-17）：替换REDD对象，并将咖啡机／冰箱的原100上限改为600和日期分层抽样。原文SHA256：`e3ac68d324dbf658eadddd3ab0bcd4dd393e4095adb47ba36982dc6f59d5cb7e`；保存于[历史v2](history/public_appliance_comparison_cohort_9plus7_20261017_v2.md)。
- v3（2026-10-17）：依据用户要求，序号12、16改为总表固定连续范围，当前支持片段599／596；增加16对象的实际提取结果路径、全量／展示包区别及失效缓存说明。未执行范围数据导出或PrimGLR新实验。

## 证据来源

- [论文实验设计](../essay/proposal.md)，重点5.1、5.2、5.8、5.9。
- [原九对象全量报告](../runs/20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_house1_increment_final_20261016T084158Z_01699ad3/report.md)。
- [新增电器筛选报告](../runs/20261016_public_appliance_signature_screening/reports/final_20261016T142157Z_497430b2/report.md)。

引用时建议注明：`docs/public_appliance_comparison_cohort_9plus7.md`（2026-10-17 v3版），并保存该文件及实际输入清单的SHA256。调整对象、上限或资格规则时保留版本记录，不回写已执行批次的冻结输入。
