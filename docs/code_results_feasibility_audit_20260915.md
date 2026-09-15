# PSLG-NILM：代码、结果用途、可行性与下游任务复核

复核日期：2026-09-15。代码基线：`feature/primitive-generation`，提交 `ff722ca`。范围：本地源代码、配置、测试、版本化实验记录、代表图像，以及 UK-DALE 官方元数据和相关原始论文。未登录服务器、未重新训练模型。本报告中的“已确认”指源码/本地材料可验证；历史实验数值另行注明。

贡献边界：切分和聚类是师兄的已有工作，本项目在其产物上实现周期组织、生成与下游评价，不能把上游方法写成本人的创新。以下是 `ff722ca` 的审计快照；后续已实施的修复和服务器验证步骤见 [本轮交接](server_generation_validation_20260915.md)，不应将本报告的问题清单误读为最新代码仍未修复。

## 1. 结论与本轮决策

研究方向仍有可行性：以电器基元为单位、保留周期结构进行数据增强，然后用真实连续总表上的负荷分解验证价值。当前已经具备上游发现、重组、评价和下游训练代码，工程起点较好；正式论文结论尚缺可靠的实验闭环。

优先下游任务确定为：**相同目标电器真实数据预算下，比较增强方法对真实连续时间线洗衣机功率分解的作用，并同时评价关机误报。** Seq2Point 继续作为第一个固定评估器；状态聚类和周期分类属于生成器内部组织方式，不作为下游真实标签。

开始正式模型实验前，第一项工作应是恢复实验证据并校正数据定义，随后处理时间隔离、预算口径和生成器版本一致性。现有结果尚不足以支持“端到端无泄漏的小样本增益”“类别等同真实程序”或“跨设备泛化已成立”。

## 2. 目前实际掌握了什么

| 项目 | 本轮核验结果 | 用途和限制 |
|---|---|---|
| 当前代码 | 子仓库工作树在复核前干净，HEAD 为 `ff722ca` | 可核对现有实现；根目录旧交接不能代表此版本 |
| 自动测试 | `82 passed, 6 warnings`，4.08 秒 | 验证已有小规模测试覆盖的工程行为；不等同真实数据、GPU或科学有效性验证 |
| 本地科学环境 | Anaconda Python 3.13.9，可用 NumPy/Pandas/SciPy/sklearn；无 TensorFlow | 本轮没有验证 DETSEC/Seq2Point 实际训练，环境也不是 requirements 声明的正式环境 |
| 新生成图 | `primitive_synthesis/` 20 张 PNG | 检查周期形状、Class/Mode和边界；缺源CSV、运行哈希，不能确认对应当前预算内版本 |
| 旧生成图 | `primite_synthesis1/` 20 张 PNG | 含后来排除的 Class 7，属于旧阶段，不应混入当前图组 |
| 模式代表图 | `cycle_validation_modes/` 10 张 PNG | 显示真实代表/近邻/远邻周期；文档记载11模式，本地缺Class 3/Mode 0图 |
| 原始实验产物 | 本地无 `input/`、`log/`、`output/`，无实际实验 manifest、指标CSV、NPZ、模型checkpoint | 历史指标目前只能追溯到文档，不能重算；新预算内结果尚未回传 |

测试命令：在仓库目录运行 `PYTHONDONTWRITEBYTECODE=1 /Users/kyrie/IDE/anaconda3/bin/python -m pytest tests -q -p no:cacheprovider`。警告涉及核心数检测和小型测试重复点的聚类收敛，不是测试失败。

目前可信的上下文入口是 [最新实验记录](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/docs/ukdale_washing_machine_primitive_generation.md:635)。根 [HANDOFF.md](/Users/kyrie/Documents/Thesis_research_proposal/HANDOFF.md:3) 的旧路径、1396周期/3736基元、统计特征与“生成器尚未实现”等内容只适合历史参考。

## 3. 中间产物到底指向什么

下表文件名是源码定义的产物，不代表它们目前都在本地。路径由 `run_manifest.json` 解析。

| 阶段 | 关键产物 | 实际含义 | 下游用途 | 不能据此得出的结论 |
|---|---|---|---|---|
| 数据对齐 | `timestamp,mains,appliance`、对齐audit | 总表和目标支路的配对观测 | 背景构造、NILM输入与监督目标 | 匹配率高不证明计量类型相同 |
| 工作段提取 | 活动CSV、时间范围 | 阈值算法检测出的候选活动及上下文 | 分割、原始波形回读、来源追踪 | 不保证每段恰好一个完整程序 |
| 基元切分 | `X.npy`、`lengths.npy`、`indices.npy` | 片段张量、有效长度、源活动及起点 | 表征学习、聚类和定位原始波形 | 四通道不代表四个独立传感器；prim-glr分支基本为重复功率通道加零通道 |
| 表征学习 | `features.npy`、`training_history.json`、cache key | 用于比较片段的向量 | 聚类；检查训练过程 | 重构损失低不证明生成好或NILM好 |
| 聚类 | 标签、索引、簇文件、内部指标 | 波形表示相近的片段集合 | 状态合并、基元库 | Cluster 0/1等不是人工标注的加热/漂洗 |
| 状态合并 | `blocks.json`、`state_sequences.json`、修正标签 | 每个周期内的状态顺序、跨度、合并来源 | 类别发现、周期画像、重组 | 合并规则并非真实设备状态标注 |
| 时间留出 | assignments、时间范围summary | 周期所属的训练/验证/测试时段 | 后续fit范围限制、连续数据划分 | 当前位置不能追溯约束此前已训练的编码器与聚类 |
| 周期类别发现 | `cycle_classes.json`、assignments | 离散状态签名的常见组合 | 周期筛选、结构条件 | 相同签名不必然对应相同洗衣程序 |
| 周期验证 | validity report、`validated_cycle_classes.json`、mode summary、grammar | 规则筛选和GMM/MAD定义的经验类别/模式 | 生成候选库、逐组质量检查 | 通过规则不等于物理真值认证，GMM模式不等于设备设置 |
| 周期拆分 | train/validation/test catalogs | 可用于各阶段的周期目录 | 限制生成来源、建立比较集 | 编号不交叉不保证原始时间点不交叉 |
| 全局重组 | 合成CSV、`synthesis_manifest.json`、库/转移/条件摘要 | 来自真实基元的组合及其逐块来源 | 质量消融、绘图；旧版NILM路径 | 不是训练好的VAE/GAN，也不是凭空产生独立观测 |
| 质量评价 | distribution/state-duration/novelty CSV、quality summary | 分布偏差、最近邻形状差异、覆盖诊断 | 选择生成策略，寻找失败模式 | 距离下降不自动意味着增强有效；新颖性越大也非越好 |
| 预算内NILM构建 | 真实/传统/合成NPZ、`budget_synthesis_manifest.json`、dataset manifest | 按所选真实周期预算构建A/B/C | Seq2Point训练和来源审计 | 仅波形来源在预算内，不代表所有上游学习都在预算内 |
| 连续数据构建 | 连续验证/测试NPZ、OFF片段与引用列表 | 含真实关机背景的评价序列 | 功率回归、开关检测、误报与能量评价 | 当前训练输入仍主要是活动片段+独立OFF片段 |
| 下游训练 | checkpoint、history、metrics、predictions | 一个确定数据/方法/种子的模型与输出 | 重算论文表格、配对比较 | 缺数据版本绑定时，旧metrics不能代表新数据实验 |

当前有两条需要明确区分的生成路径：

- `primitive_synthesis → synthesis_evaluation`：全局训练库上的独立/周期近邻条件重组，历史 `k=10` 质量消融属于这条链。
- `nilm_dataset(synthesis_scope=budget_local) → nilm_continuous → Seq2Point`：在NILM构建器内部再次生成预算内周期；不消费前一条链的合成CSV，也没有调用周期近邻索引。

因此，**历史被评估的生成器与当前真正送入C组的生成器存在方法差异**。这不是路径名问题，需要统一方法接口或分别报告。

另外，`fewshot → pam → split` 是旧的少样本簇/活动映射与knockout划分支路；当前A/B/C训练器消费的是NILM dataset manifest，两者不能当作同一个小样本协议。[流程顺序](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/main.py:48)

## 4. 会改变实验结论的主要问题

### P0-1：总表和支路功率类型不同，背景相减缺少物理依据

官方元数据把 House 1 `channel_1.dat` 对应到 `EcoManagerWholeHouseTx`，测量视在功率；`channel_5.dat` 对应 `EcoManagerTxPlug`，测量有功功率。前者单位VA，后者单位W。[房屋通道映射](https://raw.githubusercontent.com/JackKelly/UK-DALE_metadata/master/building1.yaml)、[仪表测量类型](https://raw.githubusercontent.com/JackKelly/UK-DALE_metadata/master/meter_devices.yaml)

当前增强采用 `max(mains-appliance,0)+synthetic_appliance`。[代码](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/nilm_dataset_step.py:305)

物理分析：视在总功率不是各支路有功功率的直接可加总和，故 `S_total-P_target` 不能直接解释为其他电器的真实有功背景。用真实视在总功率学习预测真实支路有功功率仍可作为回归任务，但这种人工替换不能声称严格保持功率叠加关系；截断负值只改变数值，不解决计量类型问题。

优先核查服务器是否具有 House 1 `mains.dat` 的有功列。官方把它登记为 SoundCardPowerMeter，包含有功/视在功率和电压；需要确认实际版本、字段、有效覆盖，再选择有功量重采样对齐。当前读取脚本只按两列解析，不能直接替换文件名使用多列mains数据。[读取器](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/scripts/prepare_ukdale_nilm_pair.py:20)

### P0-2：同一通道包含换机前后的两台机器

官方记录：2015-09-08前为 Hotpoint WD420，之后为 Samsung wf80f5e5u4x，均使用meter 5。[官方设备实例及有效日期](https://raw.githubusercontent.com/JackKelly/UK-DALE_metadata/master/building1.yaml)

本地图像也包含两时期：Class0/Mode0样例来自2012、2014和2015年7月；Class1/Mode0样例来自2015年10月及2016年。可直接查看 [Class0/Mode0](/Users/kyrie/Documents/Thesis_research_proposal/cycle_validation_modes/class_0_mode_0.png) 与 [Class1/Mode0](/Users/kyrie/Documents/Thesis_research_proposal/cycle_validation_modes/class_1_mode_0.png)。

这证明现有图组跨设备实例；“类别/模式差异有多大比例由换机造成”仍须成员级时间交叉表验证。主实验应先在同一设备实例内做时间留出；跨换机留出可作为后续单独的设备变化实验。不能把两种问题混成同机小样本提升。

### P0-3：时间留出太晚，且上下文可跨越切分边界

`main.py` 强制顺序为 feature、cluster、state_merge，然后才 temporal_holdout；调整CLI中步骤排列不会改变该顺序。[入口](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/main.py:48)

FeatureExtract读取整个X训练，当前缺按训练活动筛选的路径。DETSEC还在全部输入上求MinMax，KMeans和状态合并的特征标准化也由输入全体拟合。因此 `structure_fit_scope=train_only` 只描述后续周期结构层，不能证明端到端训练隔离。[FeatureExtract](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/feature_extract_step.py:181)、[DETSEC训练](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/models/feature_extract/detsec_model.py:150)

另外，holdout仅按周期开始时间排序，没有验证上一个split的最后结束时间早于下一个split。[holdout](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/temporal_holdout_step.py:78) 当前90秒前后context与150秒停机容忍允许相邻活动重叠。本轮纯内存诊断得到合法区间 `[30,390]` 和 `[366,726]`，共享5个6秒采样时间点。若位于split分界，两侧会共享真实数据。服务器实际是否触发及规模尚未知。

应先在原始时间轴冻结训练/验证/测试边界，再提取或剔除越界活动；所有拟合仅使用允许的训练来源。为窗口保留上下文时，也要审计其原始时间范围。验证/测试映射只能使用已冻结的编码器、归一化和类别模型。

### P0-4：基元长度上限可能截掉真实波形并使后续状态统计错位

配置启用 `max_seg_len: 1536`。[配置](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/config/config_ukdale_detsec.yaml:20) 代码直接截掉长片段尾部并缩短length，但保留下一片段原始start。[切分代码](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/time_segmentation.py:231)

后续状态块以长度累加，`cycle_profile` 按cursor顺序读取原始波形，不使用显式start/end；若中间有截尾缺口，状态画像就会读错区段。本轮构造例中第二状态真实20W，画像会读成缺口区域的100W。[画像实现](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/generation/cycle_conditioning.py:11)

触发条件和错误机制已确认，历史受影响片段数因缺实际indices/lengths未知。应审计每个活动的片段区间并集是否完整覆盖；模型长度限制用分块、保留映射的重采样或仅表征视图解决，原始来源跨度不能被覆盖成截短长度。

### P0-5：预算内重组尚不能证明严格少标签学习，且方法与k10消融不一致

当前预算内代码确实约束基元来自所选周期，且保存逐块来源，这是有效进展。但类别/模式及分层选择仍依赖全训练期结构，DETSEC/KMeans范围也未隔离；OFF池从完整训练期按支路真值筛出，需要单列其标签资源成本。[预算生成](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/nilm_dataset_step.py:238)、[连续训练OFF引用](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/nilm_continuous_dataset_step.py:204)

预算内sampler从同Class/Mode的预算周期抽样，允许锚点自身，未执行 `CycleNeighborIndex(k=10)`。某组只有一个源周期时可能近似重放。历史k10质量结果不能转用为当前C组质量证据。

还需明确分母：当前1%等比例是“通过验证且对齐成功的训练活动周期数”的比例。`D_full_real`也只含这些周期再加OFF，不是完整真实训练时间线。[比例与D组](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/nilm_dataset_step.py:448)

### P1：评价和复现需要补强的环节

1. **测试集被用于选生成超参。** 质量评价直接读取test_catalog，历史记录又用这些距离比较并选择k。此集合已经承担开发集作用；正式测试应另行冻结或采用预先固定的嵌套验证。[评价入口](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/synthesis_evaluation_step.py:154)
2. **旧结果可能被静默复用。** 训练器只因同名目录有metrics.json就跳过，未核验数据、代码和参数哈希；预算重建后复用旧输出根目录会混淆结果。[跳过条件](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/scripts/train_nilm_seq2point.py:125)
3. **窗口上下文不一致。** 599点×6秒约60分钟；短活动只有90秒实测外侧context，剩余补零。连续数据还按100000点分块，每块重新补零，人为块边界丢掉实际邻域。需采用halo上下文/中心区评分，并报告有效评分覆盖。[窗口加载](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/nilm/seq2point.py:15)
4. **平滑后跳变小是算法直接作用。** 不能独立证明物理合理；还要对比真实边界、能量变化和瞬态保留。生成质量指标也应报告原始W/Wh尺度及真实—真实基线，而非只报告归一化均值。
5. **模型与数据应整体可复现。** 当前特征流程保存向量和history，没有登记可用于新周期映射的编码器；DETSEC的save/load也未保存MinMax。正式fit/transform接口应保存模型、scaler、训练来源和版本。[模型保存](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/models/feature_extract/detsec_model.py:223)
6. **同run重跑有残留和依赖混用风险。** Extract写入现有目录，Segment随后扫描所有CSV；改阈值后旧活动可能残留。Manifest更新上游也不会让旧下游失效，聚类tag没有绑定当前特征指纹。应使用稳定活动ID、明确文件清单和上游内容哈希，不依赖目录排序恢复来源。[活动写入](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/extract_active_data_step.py:146)、[manifest更新](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/framework/run_manifest.py:91)
7. **缺失与来源检查存在盲区。** Extract填充NaN后，cycle_validate只能看到填充后的缺失率；应保留原始missing mask。synthesis_eval尚未逐条验证合成来源与heldout不相交，且其summary将结构范围写死为all_validated_cycles；应基于实际来源和上游元数据计算，不靠名称或固定字段认证。[缺失处理](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/extract_active_data_step.py:176)、[评价scope](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/src/steps/synthesis_evaluation_step.py:322)

旁支说明：当前 `detsec` 是BiGRU门控注意力自编码器，MSE训练后外接聚类；根HANDOFF中的“pretrain_epochs与联合聚类训练”属于旧实现。PC-DETSEC旁支的 `softplus(v)-ln2` 在v<0时可为负；所谓scheduled-sampling-style目前是将上一真实输入置零，并非回馈上一预测。若论文采用这条旁支，名称与物理约束需另审。[PC实现](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/models/feature_extract/detsec_pc.py:99)

## 5. 历史结果能支持到哪里

以下来自版本化实验记录，未由本轮实验原件重算。

| 历史证据 | 记录结果 | 合理解释 |
|---|---|---|
| 筛选与生成 | 920有效周期，5类11模式；220均衡生成周期 | 说明已有可执行组织方法；按设备实例和新split重建后数量会改变 |
| 质量五种子 | Independent归一化Wasserstein `0.1052±0.0034`，k10为`0.0924±0.0027` | 在当时全局库协议下，近邻条件改善这项分布距离约12.2% |
| 下游10%五种子 | A/B/C的NDE分别`0.0533/0.0542/0.0466`；C的F1为`0.9178`，低于A/B | 存在功率误差改善线索，同时开关误报可能恶化 |
| 统计显著性 | 一项配对t检验p=0.0446，精确Wilcoxon p=0.0625 | 五种子加多指标/多比例不足以宣称全面稳健显著提升 |
| 周期测试与连续测试 | 文档所述ON比例约86.9%与3.67% | 工作段内性能无法直接代表全天连续NILM性能 |
| 最新预算内版本 | 代码已存在，服务器结果未回传 | 尚无可核验的最终预算内连续评价结论 |

数值出处：[质量消融与下游实验记录](/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM/docs/ukdale_washing_machine_primitive_generation.md:260)。上述“改善”均限于对应旧协议，不能越过第4节限制推断。

## 6. 文献检索与论文可行性

本轮为定向检索，不是穷尽的系统综述。采用官方数据说明、作者仓库和论文原始页面；部分出版商全文不可获取，只用可核对的摘要界定范围。

| 原始来源 | 与本课题的关系 | 对当前路线的判断 |
|---|---|---|
| [Seq2Point论文](https://arxiv.org/abs/1612.09106)及[作者代码](https://github.com/MingjunZhong/NeuralNetNilm) | 总表窗口预测电器中心点功率 | 继续作为固定评估器有依据，但本项目预处理/缩放与完整原文复现应区别表述 |
| [SynD，2020](https://www.nature.com/articles/s41597-020-0434-6) | 用真实电器轨迹模拟合成负荷，考虑多种运行模式 | “使用真实轨迹合成”已有先例；贡献需落在基元结构、条件一致性及严格预算效果 |
| [MATNilm，2023](https://arxiv.org/abs/2307.14778) | 有限标注数据下的NILM样本增强 | 小样本增强本身不是新问题；相关工作需对齐可用操作曲线/预训练资源的口径 |
| [HiFAKES，2025修订版](https://arxiv.org/abs/2409.00062v2) | 高频合成、模型诊断、保真/多样性/真实性评价 | 可借鉴多维质量评价；高频电流任务不能直接当6秒功率重组的同条件性能对手 |
| [扩散增强NILM，Energy 2025](https://www.sciencedirect.com/science/article/pii/S0360544225010655) | 已有扩散模型用于NILM数据增强的研究 | 加入深度生成器本身不构成新贡献；本轮仅核对摘要所述范围 |

综合判断：

- 工程可行性较高：核心模块、来源清单和训练器已具备，测试覆盖有基础。
- 物理/数据有效性目前不足：功率类型、设备实例、时间覆盖要先校正。
- 统计有效性尚未建立：最终预算内连续结果缺失，旧数据集又已参与方法选择。
- 极小预算的算法可行性有疑问：当前类别支持阈值30、模式支持阈值10，严格限定1%/2%来源后不能假定还能发现原来11种模式。应先报告每预算的真实周期数和每状态支持，再决定回退规则及可评估范围。
- 论文差异点可以收敛为“保留周期结构和物理尺度关系的可追踪基元重组，在明确真实数据预算下的连续NILM验证”。是否最终成立取决于受控实验，不能由视觉好看或内部聚类指标决定。

## 7. 冻结下游任务及评价口径

### 7.1 主任务

预测对象为未来真实连续时间线中的洗衣机有功功率。先在同一设备实例内做时间留出；有功总表可用时优先统一有功量。首个评估器固定Seq2Point，保持相同窗口、阈值、训练/验证策略。中心窗口使用未来上下文，因此本版本属于离线分解，不宣称实时因果预测。

主指标设为连续时间线NDE；同时报告MAE、SAE、Precision/Recall/F1、关机误报率、OFF误报能量。补充ON区间MAE和实际事件能量误差，避免大量OFF让整体MAE掩盖漏检。全零预测作为廉价基线可检验指标是否具有辨别力。

训练分组：

| 组别 | 训练资源 | 要回答的问题 |
|---|---|---|
| A | 固定预算真实数据 | 原始小样本能力 |
| A-repeat | 与增强组等训练曝光的真实重放 | 收益是否仅来自重复样本/更多梯度更新 |
| B | 相同真实数据+传统扰动 | 普通增强的收益 |
| C-independent | 相同真实数据+预算内独立基元重组 | 基元重组本身的作用 |
| C-conditioned | 相同真实数据+预算内周期条件重组 | 周期条件机制的增量作用 |
| D | 完整允许训练时间线 | 全量真实参考，不视作数学上的性能上界 |

A/B/C共享真实活动成员、可用背景、验证和测试。相同epoch数不等于相同优化步数，需要记录样本窗口数与梯度更新数；主比较B与C应匹配这些资源。

### 7.2 预算定义

正式“严格少标签”结论要求所有使用目标支路信息的步骤都受预算约束：编码器、归一化、聚类、类别/模式、基元库、背景筛选，以及数据驱动的选样。无监督训练若输入目标支路，也仍使用了该支路采集资源。

如采用独立历史参考库，必须报告其房屋、机器、时间和规模，并改称“参考库辅助的目标域小样本适配”；不能把参考库资源隐去后仍称全部流程仅用1%标签。若OFF和验证标签作为各组公共资源，明确单列，主张限定为“活动周期预算”，不能泛称总标签预算。

当前建议：先建立严格来源核算；以5%/10%/20%验证受控机制，1%/2%作为预先规定的低支持压力测试。发生模式无法拟合时如实记录或采用预先定义回退，不借用预算外周期补齐。百分比同时给出周期数、ON小时、OFF小时及预训练资源。

### 7.3 参数与统计规则

- 参数只在训练/验证集选择。历史已反复查看的测试集降为开发证据，另设最终测试或固定的外层时间验证。
- 将“抽到哪几个真实周期”“生成随机性”“模型初始化随机性”分开记录。只重复模型种子，不能证明对不同少样本子集稳健。
- 每组配对比较，报告差值和不确定性；天/事件为时间块评估，避免把相邻采样点当独立重复。跨种子的结果也不等同跨家庭泛化。
- 默认报告所有预定组和失败情况。验证阶段若C无稳定收益，先分析来源不足、边界或分布偏差，再决定是否增加模型复杂度。

## 8. 已确定的执行任务与验收

| 顺序 | 工作内容 | 交付物 | 完成条件 |
|---|---|---|---|
| T0 | 恢复最小实验证据；核查计量、设备及日期 | 数据与artifact清单、时间×设备×Class×Mode统计 | 源文件、单位、覆盖、运行提交、指标来源可追踪；历史/当前方法分开 |
| T1 | 修正来源跨度、分界重叠、拟合范围 | 完整区间审计、冻结split、可保存的fit/transform模型 | 源区间无缺口/重复；split无共享时间点；训练来源与模型参数完整登记 |
| T2 | 统一预算内生成和质量评价 | 同一sampler接口、预算support表、每样本来源和质量表 | 实际C组与被评估方法一致；不超预算；单源/回退/复制比例显式报告 |
| T3 | 构建连续训练/评价和版本绑定 | dataset hash、训练配置hash、背景/窗口审计 | 无过期metrics复用；真实背景单位一致；人为块边界不改变有效输入 |
| T4 | 固定验证协议做小矩阵，再最终评估 | 配对结果表、预测与失败分析 | 先验证工程与预算，再扩大种子/比例；最终测试一次性运行冻结方案 |
| T5 | 第二机器/家庭或不同电器扩展 | 独立泛化实验 | 主闭环成立后扩展；设备变化和同机时间泛化分开报告 |

T0所需最小数据包包括：实际数据文件头和字段元信息、数据版本与日期范围、run_manifest、运行时配置及commit、活动索引/长度/时间表、holdout与cycle_split清单、budget来源清单、连续dataset manifest、质量CSV、每组metrics/history及代表预测NPZ。原始大数据可继续留在服务器，由只读审计产生摘要；不要求先复制全量19M点到本地。

已有 `scripts/audit_budget_dataset.py` 可作为T0的一部分，但它主要验证来源编号、文件存在及引用关系，不能代替NPZ数值、计量类型、时间交叉和上游fit范围检查。

本轮完成复核、自动测试及任务定义，新增此报告；业务代码和实验输出未改动。T0仍缺服务器实际证据，不能据此报告正式重跑已完成。
