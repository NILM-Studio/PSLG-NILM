# 实验说明与增量完成登记

本文件只追加。新实验登记目的、模型/数据、条件与预期产物；阶段结束登记实际完成时间和登记时间、实验/执行 ID、完成范围、配置与输入、状态和退出码、验证依据、报告、图表、限制和后续事项。
未知填写 unknown。目录迁移不补造历史审批、完成日期或科学结论。

## 2026-10-15：目录重构登记

- 范围：将实验归入 runs，共享代码归入 src/models/runtime，生成模型迁至 models/downstream_generation_models。
- 授权：用户确认“可以按照上述方式重构”；仅表示目录重构授权。
- 实际迁移日期/登记日期：2026-10-15；原历史运行完成时间和退出码 unknown。
- 证据：archive/20261015_structure_v1/moves.json、source_before.tar.gz、source_sha256.json。
- 历史实验：20260915_sequence_workflow、20260915_primitive_vs_direct、20261013_multidomain_generation；保留原证据，本次未重训。
- F4：移出活动代码和实验树，归档保留，不永久删除。
- 重构后的测试与真实数据 smoke 单独登记，不能从迁移成功推断模型质量。

## 2026-10-15T13:06:39.191098+00:00：重构后验证完成登记

- `20261015_mainline_restoration/regression/20261015T130215Z_b99d3eee`：failed，退出码 1，实际完成 2026-10-15T13:02:43.005750+00:00；命令/输入/产物见该执行目录。
- `20261015_mainline_restoration/post_migration_smoke/20261015T130216Z_bed9c122`：passed，退出码 0，实际完成 2026-10-15T13:02:25.728317+00:00；命令/输入/产物见该执行目录。
- `20261015_downstream_generation_validation/post_migration_smoke/20261015T130217Z_7d55b8e2`：failed，退出码 1，实际完成 2026-10-15T13:02:22.734343+00:00；命令/输入/产物见该执行目录。
- `20261015_downstream_generation_validation/post_migration_gpu_smoke/20261015T130309Z_de38b63a`：passed，退出码 0，实际完成 2026-10-15T13:03:40.400881+00:00；命令/输入/产物见该执行目录。
- `20261015_mainline_restoration/regression_isolated/20261015T130401Z_1155d225`：passed，退出码 0，实际完成 2026-10-15T13:04:43.727556+00:00；命令/输入/产物见该执行目录。
- 主线报告：runs/20261015_mainline_restoration/reports/20261015_structure_validation/report.md；图为 pipeline_counts.svg。
- 生成报告：runs/20261015_downstream_generation_validation/reports/20261015_structure_validation/report.md；图为 smoke_duration.svg，表为 model_smoke.csv。
- 完成范围：目录和执行路径重构、65 项回归、真实数据主线、七个公开生成实现的最小验证；两次初始失败保留且说明原因。C-TimeGAN 未实现；未重训历史实验、未验证生成质量。

## 2026-10-15：公开数据集 PrimGLR/DTW/KMeans 方案与盘点登记

- 实验：`20261015_multidataset_primglr_dtw_kmeans`。目的为公开数据原生采样率与 6 秒合理性检验，经过活动提取可视化验收后比较 PrimGLR + DTW 原型距离 + KMeans(k=2–8) + 时间合并。种子建议 0/1/2，正式方案待用户审核。
- 范围修订：用户明确不使用三个自采 CSV。此前 `inventory_cadence_preview/20261015T132844Z_8dd40617` CPU 预览于 2026-10-15 完成、退出码 0，现排除其全部自采结论；历史执行原样保留，报告预览与旧脚本移入本实验 `data/excluded_self_collected_audit/`，不用于公开实验方案。
- 已完成：公开数据盘点 `public_inventory_npz_schema/20261015T133219Z_c35affb4`，2026-10-15 完成、退出码 0；245/245 文件哈希匹配，288,079 点。原始开始/完成时间详见执行 run.json，登记日期为 2026-10-15。
- 数据：UK-DALE 洗衣机 40、REDD H2 洗碗机 5、REFIT 四对象各 40、GREEND B4 洗碗机 40 个预提取 NPZ；均仅有 x/labels，所有 labels 为 unlabeled，声明 6 s 网格但无原始时间戳。原始连续公开文件尚未找到，原 source 在其他服务器的路径当前不可访问。
- 证据及配置：本实验 `configs/experiment_plan.json`、`executions/public_inventory_npz_schema/20261015T133219Z_c35affb4/{inputs,environment,artifacts,run}.json`；公开数组只读核验。
- 交付：`proposal.md`、`protocol.md`；`reports/20261015_public_inventory/report.md`、`figures/public_npz_relative_time.png`、`tables/public_inventory.json`、`tables/public_files.csv`。实验注册表与报告索引已更新。
- 限制：当前完成的是数据勘查与方案，不是完整流程实验。原始 Δt、采样率损失、开启提取图表验收必须补齐连续公开数据后开展；REFIT 发布者 8 s 与转换元数据 7 s 差异待实际时间戳核验。没有有效状态真值，不计算状态准确率。
- 后续：审核方案后先补齐原始公开数据并进行采样/开启提取验收，再适配长度感知 DTW、物理时间 PrimGLR 参数和合并连续性检查，保留 k=2–8 全部结果。本轮未占用 GPU；后续资源限制物理 GPU1–2。

## 2026-10-16：完整公开低频 HDF5 获取与验证完成

- 实验：`20261015_multidataset_primglr_dtw_kmeans`；用户授权网页搜索并下载完整 UK-DALE、REDD、REFIT H5，允许本地 7897 代理转发；该阶段不等于正式聚类方案已审核。
- 实际完成时间：2026-10-16T05:48:00.585809+00:00；登记日期：2026-10-16。
- UK-DALE：发布者指向 CEDA 的 2017 完整低频 HDF5；5 住宅、111 meter、6,330,578,146 字节；SHA-256 `511853bdec05beb182927e95ca5a508f475ffd34f09503d3d6441a855fe66d07`。
- REDD：Zenodo 13917372 研究者镜像；6 住宅、116 meter、401,317,829 字节；发布 MD5 校验通过，SHA-256 `757e694a420157aa035c8fd345915080c2396cb7a37bfeb5f925bcb84f9532b8`。
- REFIT：Strathclyde 完整 CLEAN_REFIT_081116.7z，经固定官方 NILMTK 转换，20 住宅、200 meter、7,320,292,291 字节；SHA-256 `d6149238cf523da02fd84800206dcfa31be51e668d97a5ff3223c0aee21622ab`。所有源 CSV 与 200 meter 行数和首末记录核对通过：119,495,879 时间行、1,194,958,790 功率值。
- 位置：`datasets/{ukdale,redd,refit}/<dataset>.h5`；原包、来源、原始 CSV、许可说明和 validation.json 保留。根 `datasets/public_h5_manifest.json` 与本实验 `data/public_h5_manifest.json` 为总清单。
- 成功执行（退出码均为 0）：`public_full_h5_resumable/20261016T051541Z_d2d58332`、`refit_complete_h5_conversion/20261016T050925Z_8f4adbd5`、`refit_complete_source_rows/20261016T053635Z_f11c5096`、`h5_validation_ukdale_redd_refit/20261016T054720Z_980b05e5`、`complete_public_h5_report/20261016T054759Z_2a237a60`；独立 REDD/REFIT 验证也通过，详见执行索引。
- 初始失败记录保留：`public_full_h5_download/20261016T050256Z_ac41fd81`（中止慢速 UK/REDD 直连、改用代理）、`public_full_h5_proxy/20261016T050640Z_b0a849a0`（传输 SSL EOF，中断后从校验前缀续传）、`refit_complete_source_rows/20261016T053338Z_76ffa3a8`（核验结果清单含 NumPy int64，修复 JSON 序列化后重跑通过）；这些不作为最终数据验证失败。
- 产物验证：UK ZIP 长度和解压 CRC、REDD 发布 MD5、REFIT 7z 完整性；三份 H5 均检查全部住宅/meter、行数/时间范围、采样时间差样本和 NILMTK 读取。转换保留源时间戳，不重采样、不裁日期。
- 报告：`reports/20261016_complete_public_h5/report.md`；图 `figures/validated_coverage.png`；表 `tables/{datasets,households,appliances}.csv` 和完整核验 JSON。来源及环境要求见 `data/PUBLIC_H5_SOURCES.md`、`configs/data_environment_requirements.txt`，独立 `.venv-nilm-data` 的 freeze 随执行保存。
- 限制：完整性范围为完整低频功率 HDF5，未获取多 TB 高频电压/电流波形；REDD 为可追溯镜像；REFIT cleaned 已受发布者缺测填补/置零影响，原 House14 缺失，NILMTK building14–20 对应原 House15–21，后续按 original_name 选通道。
- 收尾：数据/报告索引及方案数据依赖说明已更新；临时 SSH 代理转发已关闭；未使用 GPU、未读取自采 CSV、未启动正式活动提取/DTW/聚类实验。

## 2026-10-16T05:27:25.907618+00:00：COSCI-GAN 补充部署完成

- 实验：20261016_cosci_gan_validation；方案 proposal.md；输入为三设备真实同步 6 秒功率网格。
- 执行 mlp_central_mlp/20261016T052459Z_ac32ccee：passed，退出码 0，实际完成 2026-10-16T05:25:03.072287+00:00；参数/环境/输入/产物清单在执行目录。
- 执行 lstm_central_mlp/20261016T052503Z_e4156453：passed，退出码 0，实际完成 2026-10-16T05:25:06.750464+00:00；参数/环境/输入/产物清单在执行目录。
- 范围：官方 MLP/LSTM 局部网络 + 中央 MLP 判别器，1 epoch；七个优化器实际更新、生成器重载和有限值采样通过。
- Diffusion-TS / TimeVQVAE 已部署跳过；未重复安装或重训。COSCI-GAN 复用现有 Torch 环境，无新增依赖。
- 报告：runs/20261016_cosci_gan_validation/reports/20261016_official_smoke/report.md；图 figures/{mlp,lstm}_sample.svg；表 tables/smoke_results.csv。
- 限制与后续：接口 smoke 不是论文质量复现；正式实验需冻结数据划分、评价耦合保真度，并独立设计基元条件接口。

- 2026-10-16T06:52:01.497588+00:00 `20261015_multidataset_primglr_dtw_kmeans`：七对象计算147个k结果，0个支持不足，产物独立验证见`runs/20261015_multidataset_primglr_dtw_kmeans/reports/20261016_native_refit8s_20261016T064850Z_2c0cabe3`；活动人工验收pending。

- 2026-10-16T07:09:23.852387+00:00 `20261015_multidataset_primglr_dtw_kmeans`：七对象计算147个k结果，0个支持不足，产物独立验证见`runs/20261015_multidataset_primglr_dtw_kmeans/reports/20261016_fullhouse_refit8s_20261016T070750Z_874e990d`；活动人工验收pending。

- 2026-10-16T07:11:08.535660+00:00 全量公开电器实验：用户已核对UK-DALE House2；按全量源记录及累计有效观测时间划分完成。六对象串行产物按并行请求只读复用并逐对象验证；串行worker主动终止见parallel_handoff.json，烘干机新独立执行完成。7对象、147个聚类结果、零切分异常；96项训练参数敏感性见最新报告。

- 本次冻结结果新增hyperparameters.md与activity_extraction_validation.md，7对象并行只读时间污染审计；明确微波炉上下文和面包机低功率风险，语义验收未通过/未完成，不改历史模型。

- 2026-10-16T07:48:38.638410+00:00 `20261015_multidataset_primglr_dtw_kmeans`：只读冻结结果，完成294幅全成员基元叠加图及SVG，K=2–8、3种子、ALL/TEST双范围，逐簇shown=n独立核对；索引`runs/20261015_multidataset_primglr_dtw_kmeans/reports/20261016_fullhouse_refit8s_20261016T070750Z_874e990d/stacked_waveforms.md`。

- 2026-10-16T08:09:48.634800+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：七对象计算通过；首次集合校验 `20261016T080821Z_64234301` 因原生不规则时间轴末点支持口径差异失败，采样点和算法产物未改变。修正校验口径，登记新集合 `20261016T080948Z_3d95c3cd`；失败执行与未完成报告保留，不覆盖重跑。

- 2026-10-16T08:10:50.406991+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：用户授权取消训练/验证/测试划分；新注册七对象并行全量活动拟合，27,622,660原始点、2,129活动、47,026原语、147个k/seed结果。结构验证通过，活动低功率语义仍待审核。[报告](20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_20261016T080948Z_3d95c3cd/report.md)。旧实验未覆盖。

- 2026-10-16T08:14:18.492365+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：新登记文档定稿 `20261016T081417Z_583a4cd3`，区分Windows部署CRLF文件与规范LF源码哈希；七个实际代码快照均一致，不改变原算法/参数/产物，不重复科学计算。[最终报告](20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_final_20261016T081417Z_583a4cd3/report.md)。

- 2026-10-16T08:35:28.417081+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：用户授权新增House1 meter5，按2015-09-08当地时间设备更换拆成Hotpoint/Samsung两个对象；元数据与覆盖检查通过，原七对象结果只读；两对象各2 CPU线程并行。

- 2026-10-16T08:38:26.624116+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：House1增补完成，Hotpoint/Samsung新增活动1396、原语28426、42组k/seed结果；九对象合计3525活动、75452原语、189结果。新两对象结构核验通过，原七对象147结果只读复用，原报告文件哈希未改变。语义审核仍待完成。[增补报告](20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_house1_increment_20261016T083744Z_a5b35b4a/report.md)。

- 2026-10-16T08:41:59.022834+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：House1增补定稿 `20261016T084158Z_01699ad3`；逐字保留原七对象五张CSV中的原始行，仅追加两新对象行。九对象/189组合结构验证沿用已通过汇总，不重算模型。[定稿报告](20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_house1_increment_final_20261016T084158Z_01699ad3/report.md)。

- 2026-10-16T08:44:17.650322+00:00 `20261015_multidataset_primglr_dtw_kmeans`：首次代表性图索引执行20261016T084238Z_4959176e失败（索引CSV构造重复profiles关键字），模型和冻结输入未变；初始图与表保留为representative_sequences_initial_render，原始脚本保存在scripts/snapshots。修正索引，并稳定选样浮点求和后登记重新制图。

- 2026-10-16T08:45:01.790173+00:00 `20261015_multidataset_primglr_dtw_kmeans`：只读冻结raw结果，完成147幅代表性活动状态图；逐对象并行、单线程CPU；固定选样、原始切分、色块及全部波形点验证通过。索引`runs/20261015_multidataset_primglr_dtw_kmeans/reports/20261016_fullhouse_refit8s_20261016T070750Z_874e990d/representative_sequences.md`。

- 2026-10-16T09:12:24.953057+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：只读冻结raw结果，完成189幅代表性活动状态图；逐对象并行、单线程CPU；固定选样、原始切分、色块及全部波形点验证通过。索引`runs/20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_house1_increment_final_20261016T084158Z_01699ad3/representative_sequences.md`。

- 2026-10-16T09:12:36.012365+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：只读冻结全量结果，完成189幅全成员基元叠加图及SVG，全部945个簇面板shown=n；索引`runs/20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_house1_increment_final_20261016T084158Z_01699ad3/stacked_waveforms.md`。

- 2026-10-16T09:12:51.450251+00:00 `20261016_public_all_activities_primglr_dtw_kmeans`：完成九对象两类图378幅PNG及SVG，原报告科学表图和冻结模型输入校验不变，补充report.md方法、限制和示例。联合验证`runs/20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_house1_increment_final_20261016T084158Z_01699ad3/tables/public_plots_validation.json`。

## 2026-10-16：GREEND完整原始包获取与68项NILM资源检索完成

- 实验：`20261016_greend_acquisition_nilm_catalog`；用户授权继续下载GREEND、全面网页检索其他NILM数据，允许本地7897代理。方案与协议见本实验proposal.md/protocol.md；本阶段未批量下载其他完整数据、未做聚类、未用GPU/Slurm/自采CSV。
- GREEND完整官方0.2 ZIP：1,933,462,293字节，SHA256 `77c3d9dc0c8864bde861fe317868e84a06868ebcfbe87546a416b4ea60468dce`；4806条目全部CRC通过，8住宅、2682个日CSV，展开17,031,047,852字节。位置datasets/greend/GREEND_0-2_300615.zip；保留ZIP，未全量展开/转换H5。
- 发布greend.h5：52,497,559字节、73个Series表，有历史NILMTK根metadata，但覆盖仅子集（某building1条，多数小时/一天）。不将其列为全量H5；原有三数据集public_h5_manifest.json范围不变。原始包building4/5含2000-01-01文件名异常，已标注而未强改时间。
- 本阶段只对GREEND完整包做字节/摘要/CRC；原始CSV每户抽3文件、每文件最多10000行，未全行语义验证。发布H5逐表全读检查索引、缺测与Inf；详细结果datasets/greend/validation.json。
- 检索68资源/家族，含既有数据、真实住宅/工业、高频/EMI、合成/生成器、受限与负对照；等级A4、B1前缀7、B2入口/元数据42、C受限3、D待核/无独立数据12，不宣称68个均可完整匿名下载。7个B1分别DARCK、DEDDIAG、SmartNIALMeter、HIPE、PLAID、LARCO、UALM6，均正确返回206与合理格式前缀，仅1024字节。
- 补齐ENERTALK Figshare集合全部22户文件元数据：36,594,457,137字节；作者代码读Parquet，不是原始现成H5。DRED旧址跳作者主页，明确标不可直接下载；Pecan Street/HES等未代申请或接受许可。
- 最终成功执行（退出码0）：official_release_verified/20261016T091344Z_705051ed；repository_availability_probe/20261016T091827Z_3b5683f4；expanded_catalogue_checks/20261016T092524Z_7c5bb90e；enertalk_full_collection_metadata/20261016T093301Z_2963d888；final_report_with_enertalk_inventory/20261016T093459Z_99c7608e。实际起止/完成时间以各run.json为准；登记日期2026-10-16。早期慢直连中止、Series列接口和NumPy JSON序列化失败均保留，不能作为最终数据损坏结论。
- 报告：runs/20261016_greend_acquisition_nilm_catalog/reports/20261016_download_and_catalog/report.md；完整目录catalogue.md；图figures/greend_raw_vs_published_h5.png和catalogue_verification_levels.png；表tables/nilm_dataset_catalogue.json、enertalk_full_collection.json、greend_validation.json。精确freeze、输入哈希、日志、退出码、产物在最终报告execution；环境要求configs/environment_requirements.txt。
- 后续优先DARCK及一个多户秒级集；先完成GREEND时间/设备ID审计及全量H5适配，再做开启可视化验收。8/10/30/60秒原始数据不插值冒充6秒实测，高频短启动窗不代替完整活动周期。目录总索引datasets/public_dataset_catalogue.json；源码与已有实验改动未覆盖。

## 2026-10-16：三个补充公开数据集获取启动，DARCK阶段验证通过

- 实验：20261016_darck_deddiag_snm_acquisition，用户授权补充DARCK、DEDDIAG、SmartNIALMeter；方案proposal.md/protocol.md，官方清单冻结freeze_official_sources/20261016T094231Z_bcda30eb（退出码0）。目标为DARCK全发布、DEDDIAG全部15住宅及说明、SmartNIALMeter完整raw版，共25,537,486,971字节；不重复下载SNM预处理版。
- DARCK获取acquire_darck/20261016T094339Z_8a3c9fa6及完整读取validate_darck/20261016T095333Z_e0a660f5均passed、退出码0；实际起止以run.json为准，登记2026-10-16。两个官方文件90,958,966字节；CSV.xz的发布MD5 b10e208b719188276a9cc5278247c4d1通过。全部15,895,800行、57列读到XZ EOF，2025-03-05 00:30:00至2025-09-04 23:59:59，全部时间步1秒、NaN/Inf=0；CSV时间列无时区，不擅自当UTC。
- 语义限制：DARCK发布者已插值/前填/补0/手修，原始缺测掩码不可恢复；aggr_*和inaccuracy不是独立电器。DEDDIAG源数据按变化点存储，只有一户有总表，不等于15户完整总表+分表；当前未导入PostgreSQL或展开>139GB的数据库数据。
- DEDDIAG与SNM仍在下载，不能记作已完成。直连/代理路径探测后调整并发、保留通过Content-Range验证的已接收前缀续传。被主动终止的早期执行保留failed/-15，不将其视为数据损坏；以最新双路径执行状态为准。验证与最终报告执行会接续运行。
- 位置datasets/{darck,deddiag,smartnialmeter}/；source_manifest.json/publisher_metadata.json记录来源许可与发布校验和，acquisition.json记逐文件完成状态，validation.json只在验证通过后写出。查看进度：本实验scripts/status.py --compact。当前未聚类、未用GPU/Slurm/自采CSV；既有代码与实验改动未覆盖。

## 2026-10-16：九对象切分参数复核方案登记待审核

- 实验：20261016_public_activity_segmentation_recalibration；用户要求先提交方案，审核后执行。已复制模板，保存proposal/protocol、待审核参数矩阵、父实验文件哈希、既有结果只读统计及固定展示活动ID。
- 父报告为/home/avis/pslg-nilm/runs/20261016_public_all_activities_primglr_dtw_kmeans/reports/20261016_all_activities_house1_increment_final_20261016T084158Z_01699ad3/report.md；既有3525活动、75452原语，仅汇总旧JSON，未重新提取、切分或执行模型。
- 规划测试W/min_gap/alpha/dur_sigma，限活动提取与切分、统计及每对象10子图；REDD仅7条并留3格。CPU最多2进程各2线程；详细阈值与预算等待方案审核。
- 当前审核pending，executions/index.json为空；没有创建计算execution，也没有启动create_execution.py/run_execution.py。实际运行时间、退出码及新实验结果均unknown。父实验与源数据保持原样。

## 2026-10-16：九对象切分参数复核获准执行

- 实验20261016_public_activity_segmentation_recalibration；用户审核通过“可以执行”，补充最终报告保留hyperparameters.md。参数化入口与精确缓存检查已实现，接下来通过登记入口运行契约检查、原H5活动重新提取及全量基线复现，再筛参及制图。

## 2026-10-16T12:02:40.964575+00:00 segmentation recalibration contracts

- Experiment: 20261016_public_activity_segmentation_recalibration; target: None; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/contracts/20261016T120237Z_14fece23; stage completed, structural status in analysis/validation.json; duration 2.56 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:02:48.841949+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_microwave; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_ukdale_h2_microwave/20261016T120241Z_bf9046f8; stage completed, structural status in analysis/validation.json; duration 7.24 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:02:49.698514+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_ukdale_h2_washing_machine/20261016T120241Z_1c3a3021; stage completed, structural status in analysis/validation.json; duration 8.09 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:02:53.980948+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: redd_h2_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_redd_h2_dishwasher/20261016T120249Z_21524c8d; stage completed, structural status in analysis/validation.json; duration 4.34 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:03:46.171239+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h7_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_refit_h7_dishwasher/20261016T120249Z_4254abca; stage completed, structural status in analysis/validation.json; duration 55.8 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:03:48.156383+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h9_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_refit_h9_dishwasher/20261016T120254Z_a5ccb756; stage completed, structural status in analysis/validation.json; duration 53.45 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:04:39.424464+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h18_bread_maker; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_refit_h18_bread_maker/20261016T120347Z_27e20277; stage completed, structural status in analysis/validation.json; duration 51.28 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:04:39.654910+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h20_tumble_dryer; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_refit_h20_tumble_dryer/20261016T120349Z_e153b2e3; stage completed, structural status in analysis/validation.json; duration 49.56 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:05:14.191190+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_samsung; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_ukdale_h1_washing_machine_samsung/20261016T120440Z_26aed33b; stage completed, structural status in analysis/validation.json; duration 33.1 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:05:29.199410+00:00 segmentation recalibration prepare

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_hotpoint; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/prepare_ukdale_h1_washing_machine_hotpoint/20261016T120440Z_daf29836; stage completed, structural status in analysis/validation.json; duration 48.29 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:05:35.773872+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_ukdale_h2_washing_machine/20261016T120532Z_0e13240c; stage completed, structural status in analysis/validation.json; duration 2.61 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:05:37.210509+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_microwave; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_ukdale_h2_microwave/20261016T120532Z_6c04291e; stage completed, structural status in analysis/validation.json; duration 4.0 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:05:37.353804+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: redd_h2_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_redd_h2_dishwasher/20261016T120535Z_c2ce8aae; stage completed, structural status in analysis/validation.json; duration 0.99 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:06:15.256882+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h9_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_refit_h9_dishwasher/20261016T120537Z_34d62ddf; stage completed, structural status in analysis/validation.json; duration 37.3 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:06:22.744326+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h7_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_refit_h7_dishwasher/20261016T120537Z_f154c024; stage completed, structural status in analysis/validation.json; duration 44.93 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:06:31.674596+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h20_tumble_dryer; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_refit_h20_tumble_dryer/20261016T120622Z_4a0839c4; stage completed, structural status in analysis/validation.json; duration 8.4 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:06:35.406370+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h18_bread_maker; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_refit_h18_bread_maker/20261016T120615Z_826059f5; stage completed, structural status in analysis/validation.json; duration 19.67 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:07:02.999415+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_samsung; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_ukdale_h1_washing_machine_samsung/20261016T120635Z_16b5f1be; stage completed, structural status in analysis/validation.json; duration 27.04 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:07:15.352441+00:00 segmentation recalibration baseline

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_hotpoint; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/baseline_ukdale_h1_washing_machine_hotpoint/20261016T120631Z_305f8d79; stage completed, structural status in analysis/validation.json; duration 43.2 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:07:19.535715+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_microwave; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_ukdale_h2_microwave/20261016T120715Z_f200b76f; stage completed, structural status in analysis/validation.json; duration 3.68 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:07:23.319894+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: redd_h2_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_redd_h2_dishwasher/20261016T120719Z_cdd6b7b8; stage completed, structural status in analysis/validation.json; duration 3.26 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:07:40.002853+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_ukdale_h2_washing_machine/20261016T120715Z_b035bbd6; stage completed, structural status in analysis/validation.json; duration 24.13 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:08:01.186042+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h7_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_refit_h7_dishwasher/20261016T120723Z_34779bc8; stage completed, structural status in analysis/validation.json; duration 37.35 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:08:28.638310+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h9_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_refit_h9_dishwasher/20261016T120740Z_558aae04; stage completed, structural status in analysis/validation.json; duration 48.09 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:08:31.729225+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h18_bread_maker; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_refit_h18_bread_maker/20261016T120801Z_33d161c6; stage completed, structural status in analysis/validation.json; duration 30.06 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:02.621048+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_hotpoint; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_ukdale_h1_washing_machine_hotpoint/20261016T120831Z_3f37a328; stage completed, structural status in analysis/validation.json; duration 30.3 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:05.678189+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h20_tumble_dryer; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_refit_h20_tumble_dryer/20261016T120828Z_13b2b570; stage completed, structural status in analysis/validation.json; duration 36.52 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:35.498805+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_samsung; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_ukdale_h1_washing_machine_samsung/20261016T120902Z_20fdc98e; stage completed, structural status in analysis/validation.json; duration 32.37 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:39.392731+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_microwave; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h2_microwave_w060_g012_a095_d035/20261016T120935Z_e32ec78b; stage completed, structural status in analysis/validation.json; duration 3.41 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:40.206841+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_microwave; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h2_microwave_w060_g018_a080_d020/20261016T120935Z_874c6bd1; stage completed, structural status in analysis/validation.json; duration 4.14 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:41.503322+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: redd_h2_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_redd_h2_dishwasher_w060_g012_a090_d035/20261016T120940Z_fc78bca8; stage completed, structural status in analysis/validation.json; duration 0.78 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:42.804123+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: redd_h2_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_redd_h2_dishwasher_w060_g090_a080_d035/20261016T120941Z_661cd719; stage completed, structural status in analysis/validation.json; duration 0.79 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:43.934799+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_microwave; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h2_microwave_w060_g012_a098_d020/20261016T120939Z_238b8580; stage completed, structural status in analysis/validation.json; duration 3.97 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:44.220087+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: redd_h2_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_redd_h2_dishwasher_w180_g012_a098_d020/20261016T120942Z_d9c7c467; stage completed, structural status in analysis/validation.json; duration 0.94 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:46.835485+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h2_washing_machine_w060_g030_a095_d035/20261016T120944Z_b6d626a9; stage completed, structural status in analysis/validation.json; duration 2.13 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:47.280926+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h2_washing_machine_w120_g012_a080_d020/20261016T120944Z_0fb9aad2; stage completed, structural status in analysis/validation.json; duration 2.8 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:09:49.371736+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h2_washing_machine_w120_g030_a095_d020/20261016T120946Z_c78d8cdb; stage completed, structural status in analysis/validation.json; duration 2.03 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:10:21.596782+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h7_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h7_dishwasher_w120_g090_a090_d035/20261016T120949Z_834faac9; stage completed, structural status in analysis/validation.json; duration 31.66 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:10:24.418493+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h7_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h7_dishwasher_w060_g030_a090_d035/20261016T120947Z_06b0af7f; stage completed, structural status in analysis/validation.json; duration 36.61 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:10:51.493940+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h9_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h9_dishwasher_w060_g060_a090_d020/20261016T121024Z_f86fc0ef; stage completed, structural status in analysis/validation.json; duration 26.59 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:10:51.999948+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h7_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h7_dishwasher_w120_g060_a095_d060/20261016T121021Z_ce09fbde; stage completed, structural status in analysis/validation.json; duration 29.93 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:11:15.436712+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h9_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h9_dishwasher_w060_g090_a098_d020/20261016T121052Z_52811ef7; stage completed, structural status in analysis/validation.json; duration 22.95 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:11:17.193568+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h9_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h9_dishwasher_w120_g090_a080_d035/20261016T121051Z_ed77dbda; stage completed, structural status in analysis/validation.json; duration 25.16 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:11:33.621720+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h18_bread_maker; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h18_bread_maker_w060_g030_a080_d060/20261016T121115Z_29cc9e09; stage completed, structural status in analysis/validation.json; duration 17.66 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:11:34.421221+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h18_bread_maker; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h18_bread_maker_w060_g030_a090_d035/20261016T121117Z_a20e4c4e; stage completed, structural status in analysis/validation.json; duration 16.62 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:11:50.275077+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h18_bread_maker; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h18_bread_maker_w180_g030_a080_d035/20261016T121133Z_63ea1a66; stage completed, structural status in analysis/validation.json; duration 16.13 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:12:13.128870+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_hotpoint; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h1_washing_machine_hotpoint_w060_g012_a090_d060/20261016T121134Z_f28215b0; stage completed, structural status in analysis/validation.json; duration 38.17 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:12:24.460935+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_hotpoint; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h1_washing_machine_hotpoint_w060_g090_a090_d035/20261016T121150Z_ac3b006c; stage completed, structural status in analysis/validation.json; duration 33.68 seconds. Quality acceptance remains separately recorded.

## 2026-10-16：九对象活动生成方案待审核

- 实验：20261016_nine_object_activity_generation。按用户最终范围仅规划 U 无状态条件；D、BP、P 暂缓。八个非 REDD 对象主分析，REDD 极小样本探索。
- 已保存 proposal、protocol、hyperparameters、待审核配置与报告表头；指标定义含 DS、PS、C-FID、双向最近邻 DTW。论文提案仍保留更大的 U/D/BP/P 远期矩阵，本轮不执行其余条件。
- 当前 executions/index.json 为空，未调用 create_execution.py 或 run_execution.py；新训练、采样和评价结果均 unknown。

## 2026-10-16T12:12:32.625243+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h20_tumble_dryer; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h20_tumble_dryer_w060_g090_a080_d020/20261016T121224Z_96b214e9; stage completed, structural status in analysis/validation.json; duration 7.63 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:12:39.625608+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h20_tumble_dryer; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h20_tumble_dryer_w060_g090_a090_d035/20261016T121232Z_8555a80c; stage completed, structural status in analysis/validation.json; duration 6.52 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:12:43.523280+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_hotpoint; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h1_washing_machine_hotpoint_w120_g090_a095_d020/20261016T121213Z_868ad6c4; stage completed, structural status in analysis/validation.json; duration 29.9 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:12:45.914834+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h20_tumble_dryer; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h20_tumble_dryer_w060_g060_a098_d020/20261016T121239Z_57d9c359; stage completed, structural status in analysis/validation.json; duration 5.77 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:13:08.361835+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_samsung; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h1_washing_machine_samsung_w060_g090_a090_d035/20261016T121246Z_9fcf3b25; stage completed, structural status in analysis/validation.json; duration 21.94 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:13:08.390745+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_samsung; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h1_washing_machine_samsung_w060_g060_a080_d035/20261016T121243Z_de387053; stage completed, structural status in analysis/validation.json; duration 24.3 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:13:27.140115+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h1_washing_machine_samsung; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h1_washing_machine_samsung_w060_g030_a098_d035/20261016T121308Z_2eb10c2a; stage completed, structural status in analysis/validation.json; duration 18.23 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:13:41.311949+00:00 selection replay validation

- Experiment 20261016_public_activity_segmentation_recalibration; execution /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/selection_replay_initial/20261016T121340Z_51c0aaa9; targets requiring canonical selection replay: ['redd_h2_dishwasher', 'ukdale_h2_washing_machine', 'refit_h7_dishwasher', 'refit_h9_dishwasher']. Early scientific outputs retained; no source or baseline modification.

## 2026-10-16T12:13:41.389569+00:00 selection implementation reconciliation

- Initial selector replay requires canonical rerun for: ['redd_h2_dishwasher', 'ukdale_h2_washing_machine', 'refit_h7_dishwasher', 'refit_h9_dishwasher']. Original registered runs retained; fixed Pareto selector reruns only affected object searches and required full candidates. Approved parameter grid and guards unchanged.

## 2026-10-16T12:13:45.263595+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: redd_h2_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_redd_h2_dishwasher/20261016T121341Z_69ba53cc; stage completed, structural status in analysis/validation.json; duration 3.28 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:14:05.374570+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_ukdale_h2_washing_machine/20261016T121341Z_90824b6e; stage completed, structural status in analysis/validation.json; duration 23.39 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:14:22.727304+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h7_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_refit_h7_dishwasher/20261016T121345Z_e2d58fc3; stage completed, structural status in analysis/validation.json; duration 36.96 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:14:51.344685+00:00 segmentation recalibration screen

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h9_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/screen_refit_h9_dishwasher/20261016T121405Z_7914b7e4; stage completed, structural status in analysis/validation.json; duration 45.42 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:14:52.634087+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: redd_h2_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_redd_h2_dishwasher_w060_g012_a095_d020/20261016T121451Z_c3b2c055; stage completed, structural status in analysis/validation.json; duration 0.76 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:14:54.569905+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h2_washing_machine_w060_g030_a080_d020/20261016T121451Z_7e2ae7c5; stage completed, structural status in analysis/validation.json; duration 2.68 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:14:55.036348+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: ukdale_h2_washing_machine; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_ukdale_h2_washing_machine_w060_g012_a098_d020/20261016T121452Z_2a70704b; stage completed, structural status in analysis/validation.json; duration 1.93 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:15:21.842737+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h9_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h9_dishwasher_w060_g060_a090_d035/20261016T121455Z_a7977447; stage completed, structural status in analysis/validation.json; duration 26.26 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:15:31.547336+00:00 segmentation recalibration full

- Experiment: 20261016_public_activity_segmentation_recalibration; target: refit_h7_dishwasher; execution: /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/full_refit_h7_dishwasher_w060_g030_a090_d060/20261016T121454Z_1772e012; stage completed, structural status in analysis/validation.json; duration 36.39 seconds. Quality acceptance remains separately recorded.

## 2026-10-16T12:15:32.209502+00:00 selection replay validation

- Experiment 20261016_public_activity_segmentation_recalibration; execution /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/selection_replay_final/20261016T121531Z_817254f0; targets requiring canonical selection replay: []. Early scientific outputs retained; no source or baseline modification.

## 2026-10-16T12:16:38.273660+00:00 segmentation recalibration report

- Experiment 20261016_public_activity_segmentation_recalibration; registered report execution /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/final_report/20261016T121532Z_2222a238; report /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/reports/20261016_segmentation_recalibration_20261016T121532Z_2222a238/report.md; hyperparameters /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/reports/20261016_segmentation_recalibration_20261016T121532Z_2222a238/hyperparameters.md; 3525 activities and baseline 75452 segments exact; screening 456, full conditions 36; recommended segments 59326; figures 45 PNG + SVG; semantic review pending; raster QA pending external inspection.

## 2026-10-16：九对象 U 条件生成获执行授权

- 用户确认执行。范围仅 U；D、BP、P 暂缓。按已审核方案登记数据准备、模型训练、生成及评价执行；GPU1/2 当前被其他用户作业占满，启动前按服务器指引重查。

## 2026-10-16T12:18:57.623755+00:00 segmentation recalibration visual audit and closeout

- Experiment 20261016_public_activity_segmentation_recalibration; registered QA execution /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/final_visual_qa/20261016T121856Z_3db7ab57; 45 PNG decoded, 45 SVG parsed, 450 panels and 7683 cut lines traced to 87 frozen activity arrays; four representative figures visually inspected. Main README and report index updated. Scientific outcome: 59326 recommended total primitives versus 75452 baseline; only REFIT H7/H9 pass the predefined transition and fidelity guards while reducing segments; seven objects retain baseline. Human semantic review remains pending.

## 2026-10-16 primitive condition interfaces
User authorized P adapters and small-scale CPU validation. U experiment remains separate.

## 2026-10-16T12:30:02.901047+00:00 W/min_gap ten-panel gallery QA

- Experiment 20261016_public_activity_segmentation_recalibration; gallery report /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/reports/20261016_w_min_gap_ten_panel_20261016T122837Z_8280bf63; QA execution /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/w_min_gap_gallery_visual_qa/20261016T123001Z_017bbeed. Nine PNG and nine SVG validated, 90 panels and 925 cut lines traced; microwave, REDD dishwasher, and REFIT H7 dishwasher rasters visually reviewed. Microwave activity 197 selected for variation among frozen screening conditions; its tenth panel is a single-activity diagnostic. Hyperparameters retained; original science/recommendations unchanged; human semantic review pending.

## 2026-10-16 primitive interface validation completed
Three P adapters passed CPU 512-phase forward/backward/sample/reload checks; TimeVQVAE 20-step no-response retained and bounded 100/200-step diagnostic responds. U Stage-1 smoke bridge passed; no formal U/P quality claim. Report: [20261016_primitive_condition_interfaces](20261016_primitive_condition_interfaces/reports/20261016T123522Z_c24e28d1/report.md). Failed dependency/deepcopy attempts retained.

## 2026-10-16T12:36:48.086844+00:00：三个补充数据集完整获取与原生格式验证完成

- 实验20261016_darck_deddiag_snm_acquisition：DARCK 2文件90,958,966字节、DEDDIAG 20文件15,301,180,457字节、SNM raw 1文件10,145,347,548字节，共25,537,486,971字节。全部源文件大小/发布MD5通过，另留SHA256；生成说明后再次核对MD5，DEDDIAG原README/SQL/import.sh未覆盖或执行。
- DARCK完整读取15,895,800行57列、全步1秒、NaN/Inf=0；发布者已补缺，不能恢复原始缺测掩码，CSV时区未声明。DEDDIAG 15 ZIP逐条目CRC，house IDs0–14；测量表只抽前最多10000行/设备，非139GB全行质量审计；变化点保存，只有House8有总表。
- SNM全包7z测试、路径/链接安全预检及全部45,238,647,971字节展开通过，120原生H5结构/代表值可读。另查每文件前最多10000行与全部value blocks：92文件含Active Power，7文件仅On标签；其余为其他测量。含功率文件抽样中位间隔约4.947–5.158秒，不将1秒标签或约60秒其他通道当5秒功率。原生H5不是项目统一NILMTK格式，未转换或列入public_h5_manifest.json。
- [获取报告](20261016_darck_deddiag_snm_acquisition/reports/20261016_complete_acquisition/report.md)；[通道与采样报告](20261016_darck_deddiag_snm_acquisition/reports/20261016_snm_channel_semantics/report.md)。方案、环境要求、精确freeze、日志/退出码/哈希、图表在本实验。当前定稿执行executions/finalize_dataset_documentation/20261016T110744Z_b1c18627；此前成功执行executions/freeze_official_sources/20261016T094231Z_bcda30eb, executions/acquire_darck/20261016T094339Z_8a3c9fa6, executions/validate_darck/20261016T095333Z_e0a660f5, executions/validate_deddiag/20261016T095333Z_d50ce6bd, executions/validate_snm_with_archive_safety/20261016T101856Z_2afba307, executions/complete_native_acquisition_report_v2/20261016T103650Z_35494a87, executions/acquire_snm_final_proxy_tail/20261016T103832Z_205b822a, executions/audit_snm_native_sampling/20261016T110339Z_d4ab925e, executions/audit_snm_channel_semantics/20261016T110454Z_9a02a68c, executions/acquire_deddiag_official_s3_finish/20261016T114938Z_d28a3a25。
- 下载布局重排/安全验证更新时主动中止的执行保留failed状态：executions/acquire_deddiag/20261016T094339Z_6615c342, executions/acquire_smartnialmeter/20261016T094339Z_6744e805, executions/acquire_deddiag_parallel_resume/20261016T094912Z_c5a5d494, executions/acquire_snm_parallel_resume/20261016T094912Z_73cf719f, executions/validate_smartnialmeter/20261016T095333Z_df78b150, executions/complete_native_acquisition_report/20261016T100058Z_4642b9f3, executions/acquire_snm_dual_route_resume/20261016T100402Z_a8280ca0, executions/acquire_deddiag_dual_route_resume/20261016T100657Z_23cbc5bb, executions/acquire_deddiag_all_houses_flow/20261016T105412Z_6622af1b, executions/acquire_deddiag_balanced_route_finish/20261016T111439Z_7da03804, executions/acquire_deddiag_official_s3_finish/20261016T114109Z_e7df38fb, executions/acquire_deddiag_official_s3_finish/20261016T114154Z_3721629e。以最终passed执行为准，不将中止误记为数据损坏。科学质量限制见报告；未训练、聚类、用GPU/Slurm/自采CSV。保留既有源码与其他实验改动。

- 2026-10-16T12:37:08.611117+00:00 `20261016_darck_deddiag_snm_acquisition`：定稿后追加官方S3 virtual-hosted直连与临时代理配置说明NETWORK.md，15个HEAD大小/ETag逐项匹配官方MD5；不改变原数据与科学表。新执行executions/network_report_addendum/20261016T114338Z_a33c7f93，报告补充实际最后下载路径及旧前缀快照范围。

## 2026-10-16 九对象 U 生成实验执行进度

- Experiment `20261016_nine_object_activity_generation`; nine registered activity-preparation executions passed. Main eight targets use uncensored cores; REDD seven activities remain exploratory. Diffusion-TS, TimeVQVAE and COSCI-GAN CPU training/sample smoke executions passed. Full DS/PS/C-FID/bidirectional DTW contract test passed after an indexed-window bug was fixed in a new execution; the failed attempt is retained.
- User revised compute placement to two concurrent generation-training processes on physical GPU 1. Both workers are running there; GPU 2 has no experiment process. Shared TS2Vec fitting and downstream analysis queues are active. Progress report: `/home/avis/pslg-nilm/runs/20261016_nine_object_activity_generation/reports/execution_progress/report.md`; no formal model-quality result yet. `hyperparameters.md` remains with proposed values and explicit execution revisions.

## 2026-10-16T12:43:52.874179+00:00 W/min_gap activity gallery

- Experiment 20261016_public_activity_segmentation_recalibration; report /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/reports/20261016_w_min_gap_activity_gallery_20261016T124025Z_5ba440cf; QA execution /home/avis/pslg-nilm/runs/20261016_public_activity_segmentation_recalibration/executions/w_min_gap_activity_gallery_qa_and_cleanup/20261016T124344Z_ef2e806b. 87 ten-panel figures, 870 panels, 11409 cut lines; 87 PNG and 87 SVG verified. Eight objects have ten distinct activities each; REDD has its complete set of seven. Hyperparameters retained. Two superseded w_min_gap_ten_panel report directories removed per user request; registered executions preserved. Original scientific report and recommendations unchanged.

## 2026-10-16T12:59:04.861918+00:00：新增有功电器工作特征筛选方案登记（待审核）

- 实验：20261016_public_appliance_signature_screening。用户要求先方案、审核后执行；本轮仅阅读README、目录和既有设备/验证元数据，完成方案、协议、机器配置及九对象排除清单。没有打开新波形、提取活动、运行筛选、切分、聚类或绘图；没有创建计算execution，开始/结束/退出码为unknown。
- 范围：datasets/内DARCK、DEDDIAG、SmartNIALMeter、UK-DALE、REDD、REFIT、GREEND。根目录data/不存在。自采CSV、无直接有功、总表、重复版本与已确认九对象不进入新增候选；混合/未知/缺映射单列待核。
- 已确认九对象按用户结论作为正向参考，不重新否定。规范去重排除八个唯一源通道，含UK-DALE House1 meter5两设备寿命；REFIT内部building18/20对应原House19/21，映射经已有元数据核对。
- 拟议流程：元数据去重 -> 预冻结日期的有界探索 -> 潜在明显/稀有对象全历史复核 -> 按开关、多阶段、周期、连续调节归类与目视复核 -> 每新对象5×2十子图及完整候选/待核清单。少样本保留，不凑数重复；不把缺测/填补或On标签当有功特征。
- 方案[20261016_public_appliance_signature_screening/proposal.md](20261016_public_appliance_signature_screening/proposal.md)，协议protocol.md，配置configs/screening_plan.json，依据data/planning_evidence.json，排除表data/known_positives.json。审核状态pending_user_review。审核后CPU最多2进程×2线程、预算12小时；无GPU/Slurm，不运行PrimGLR/DTW/KMeans/生成训练，不改其他并行实验。

- 2026-10-16T13:05:54.231696+00:00 `20261016_public_appliance_signature_screening`：用户明确批准“可以执行”；开始既定元数据/适配/筛选/十子图流程。原九对象只读，CPU最多2进程×2线程，不使用GPU/Slurm、不启动下游聚类或生成训练。

- 2026-10-16T13:09:47.154162+00:00 `20261016_public_appliance_signature_screening`：首次metadata_inventory因GREEND同meter重复设备元数据导致唯一性断言失败，保留执行20261016T130554Z_f5801aea；修正为同meter集合的待映射组后新登记，不改源数据。

- 2026-10-16T13:09:48.332990+00:00 `20261016_public_appliance_signature_screening`：元数据盘点完成，eligible=482；排除/待映射=184；九对象八源排除已应用，未将元数据资格当波形结论。执行executions/metadata_inventory/20261016T130947Z_a8985160。

- 2026-10-16T13:13:03.422222+00:00 `20261016_public_appliance_signature_screening`：接入/活动边界/缺测/相线/选样合成单元测试：6项，passed=True；执行20261016T131302Z_66619c79。

- 2026-10-16T13:14:53.309280+00:00 `20261016_public_appliance_signature_screening`：适配pilot通过：6个格式/来源前缀，各<=30000行；源未修改。GREEND待序列号映射；执行executions/adapter_pilot/20261016T131433Z_94884ef6。前缀不是筛选结论。

- 2026-10-16T13:19:46.973554+00:00 `20261016_public_appliance_signature_screening`：DARCK单次XZ解码完成，50物理候选；共享时间轴，原始float64功率，未补缺/重采样。执行executions/darck_shared_decode_cache/20261016T131444Z_4c9495dc；仅适配缓存，未宣称筛选通过。

- 2026-10-16T13:23:14.641934+00:00 `20261016_public_appliance_signature_screening`：GREEND本地数字/时间交叉匹配尝试：确认13个单设备映射，其他保留待核；未按列序猜标签。执行executions/greend_serial_mapping_crosscheck/20261016T132310Z_f38faafb。

- 2026-10-16T13:27:45.044979+00:00 `20261016_public_appliance_signature_screening`：GREEND原始包单次/住宅接入完成，13数字匹配对象；时间异常/重叠隔离数见manifest，未用不完整小H5替代原始包。执行executions/greend_raw_shared_decode/20261016T132508Z_dbf61a64。
- 2026-10-16T13:30:53Z 20261015_mainline_restoration: post_segmentation_refactor_smoke execution 20261016T133044Z_88ec9cc5 passed (exit 0); real microwave mainline ran PrimGLR through state_sequence after four-method adapter refactor. This is an integration smoke, not experiment 5.3 results.
- 2026-10-16T13:39Z 20261016_feature_representation_deployment: four author-source snapshots pinned; four registered 16x64 training smokes and four frozen 79x64 exports passed, finite vectors and exact first-16 replay; see reports/deployment_validation/report.md. Interface only, not experiment 5.4 results.
- 2026-10-16T13:44Z 20261016_feature_representation_deployment: DTW reference adapter added under models/feature_extract; 79x16 raw distances and seed-0 prototype rows exactly match the frozen nine-object pipeline (registered dtw_equivalence execution).

- 2026-10-16T13:46:47.624863+00:00 `20261016_public_appliance_signature_screening`：接入/活动边界/缺测/相线/选样合成单元测试：9项，passed=True；执行20261016T134646Z_f99893fd。

- 2026-10-16T13:47:17.885389+00:00 `20261016_public_appliance_signature_screening`：十子图渲染预览完成：8 PNG/SVG，失败0；结构核验不等于视觉/状态真值。执行executions/ten_panel_visual_preview/20261016T134704Z_f36b736b。
- 2026-10-16T13:46Z 20261016_feature_representation_deployment: existing CNN-AE and DeTSEC first-stage smoke checks passed on 16 real segments, each exporting finite 16x64 vectors; seven candidate methods now have vector interfaces. See deployment_validation report; no comparative metrics computed.

- 2026-10-16T13:48:09.503486+00:00 `20261016_public_appliance_signature_screening`：筛选/全历史复核分片0结束，处理248/248、技术失败0；执行executions/bounded_and_full_confirmation_shard0/20261016T132634Z_c47b706c。自动候选尚待视觉审核，无聚类/生成训练。
- 2026-10-16T13:44Z 20261016_feature_representation_deployment: first dtw_equivalence execution 20261016T134428Z_cc654ce2 failed before import because PYTHONPATH omitted project root; corrected in new execution 20261016T134444Z_f2530c36, original failure retained.

- 2026-10-16T13:56:01.829972+00:00 `20261016_public_appliance_signature_screening`：筛选/全历史复核分片0结束，处理170/170、技术失败0；执行executions/resource_recovery_non_deddiag/20261016T134800Z_8d982f68。自动候选尚待视觉审核，无聚类/生成训练。

- 2026-10-16T13:56:55.033931+00:00 `20261016_public_appliance_signature_screening`：十子图渲染预览完成：17 PNG/SVG，失败0；结构核验不等于视觉/状态真值。执行executions/expanded_signature_visual_preview/20261016T135628Z_1ee81eb6。

- 2026-10-16T13:57:43.022484+00:00 `20261016_public_appliance_signature_screening`：接入/活动边界/缺测/相线/选样合成单元测试：11项，passed=True；执行20261016T135740Z_dd63da4c。

- 2026-10-16T14:01:45.616116+00:00 `20261016_public_appliance_signature_screening`：十子图渲染预览完成：5 PNG/SVG，失败0；结构核验不等于视觉/状态真值。执行executions/deddiag_signature_visual_preview/20261016T140135Z_c0e18b29。

- 2026-10-16T14:09:02.954320+00:00 `20261016_public_appliance_signature_screening`：筛选/全历史复核分片0结束，处理33/33、技术失败0；执行executions/deddiag_eventtime_chunked_recovery/20261016T134916Z_49eed88c。自动候选尚待视觉审核，无聚类/生成训练。

- 2026-10-16T14:10:16.515163+00:00 `20261016_public_appliance_signature_screening`：实际时间描述指标对齐完成：481对象；仅修正描述代理的时间分箱/事件保持语义，未改变阈值、活动边界、判定状态或展示选样；执行executions/actual_time_metric_harmonization/20261016T141009Z_69e251aa。

- 2026-10-16T14:21:56.006485+00:00 `20261016_public_appliance_signature_screening`：十子图渲染正式批次完成：481 PNG/SVG，失败0；结构核验不等于视觉/状态真值。执行executions/all_objects_ten_panel_figures/20261016T141019Z_42149fe5。

- 2026-10-16T14:21:57.235004+00:00 `20261016_public_appliance_signature_screening`：实际目视记录绑定完成：30对象，预览到正式图的原生波形/核心选样哈希一致；尚未逐图审阅的对象保持待审，不自动提升；执行executions/visual_review_binding/20261016T142156Z_92e89cd8。

- 2026-10-16T14:21:59.714111+00:00 `20261016_public_appliance_signature_screening`：筛选正式交付：全部495可辨识对象完成；481张PNG+SVG十子图、目视建议新增6、条件性21；所有源身份未变、九对象八源通道排除通过。报告runs/20261016_public_appliance_signature_screening/reports/final_20261016T142157Z_497430b2/report.md；用户审核pending，无下游训练。

- 2026-10-16T14:23:13.331032+00:00 `20261016_public_appliance_signature_screening`：独立交付完整性核验通过：495对象、481 PNG/SVG均可解码且SVG恰有10个axes，报告相对链接1006项有效，11测试通过；不等于工作状态真值。执行executions/independent_delivery_integrity/20261016T142304Z_08e7dd5c。

- 2026-10-16T14:28:30.046408+00:00 `20261016_public_appliance_signature_screening`：轻量执行溯源归档完成：2910文件、3947117字节；不含原始源/大型缓存，供本地报告镜像追溯；执行executions/portable_execution_provenance_archive/20261016T142827Z_56e78d84。

- 2026-10-16T18:17:20.540723+00:00 `20261017_public16_primglr_w120_gap30_case_review`：注册16对象数据导出与十例图审；两组数据独立复制到datasets，图审pending。

- 2026-10-16T18:21:30.157705+00:00 `20261017_public16_primglr_w120_gap30_case_review`：16对象两组数据独立复制与QA完成，入选2824条；16张十例图供人审。数据清单`/home/avis/pslg-nilm/datasets/public16_activity_states_20261017/manifest.json`，未自动冻结PrimGLR论文最终参数。

- 20261017_public16_primglr_w120_gap60_case_review：用户授权，复用gap30同一批片段并注册gap60图审。

- 20261017_public16_primglr_w120_gap60_case_review：16对象160例gap60图审完成；与gap30输入逐例哈希相同，参数待人工审核。

- 2026-10-16T18:49:16.139410+00:00 `20261017_public16_primglr_feature_pca16`：仅方案与只读元数据/资源核查，输入16对象2824活动，继承1700/556/568划分；固定候选PrimGLR、原生表示＋PCA16、两学习率三种子及GPU1/2/6指数监控方案。审核pending，无计算execution/训练；GPU1现有训练和后续分析队列待完成。

- 2026-10-16T18:52:39.288856+00:00 `20261017_public16_primglr_feature_pca16`：用户补充PrimGLR W120s/gap30s、按来源/对象已有采样间隔不统一Hz；指数监控上限改30min，间隔2..1024再1800s，事件重置。仅更新方案、协议、配置与16对象采样率表；整体审核pending，无计算execution。

- 20261017_public16_all_supported_max600_correction：用户要求修正；统一最多600，范围内全部保留；历史限额版保留，新版独立保存。

- 2026-10-16T19:03:50.620400+00:00 `20261017_public16_all_supported_max600_correction`：16对象两组数据独立复制与QA完成，入选4202条；gap30/gap60各16张十例图供人审。数据清单`/home/avis/pslg-nilm/datasets/public16_activity_states_20261017_v2/manifest.json`，未自动冻结PrimGLR论文最终参数。

- 2026-10-16T19:05:27.336646+00:00 `20261017_public16_primglr_feature_pca16`：按用户全量交付目标补充G阶段，参数/训练预算由留出阶段冻结后，在全部2824活动重新拟合编码器、尺度、DTW、PCA和聚类；另保留留出泛化证据。正式神经训练最多672次，整体仍待审核，无计算execution。

- 2026-10-17T03:16:42.936128+00:00 `20261017_public16_segmentation_hyperparameter_review`：用户确认public16_activity_states_20261017_v2，16对象4202既有活动；每对象每方法合计2图（开发1、验证1），共160张完整参数总图。冻结32条输入及既有PrimGLR边界，不重跑提取/PrimGLR；已登记准备执行通过，五方法切分与图审开始，尚无人审冻结。

- 2026-10-17T03:26:21.935429+00:00 `20261017_public16_segmentation_hyperparameter_review`：用户指定CPU；已停止E2USD GPU工作进程并归档旧图，全部E2USD改用作者cuda=False路径重算。真实数据CPU两次重放切点一致，作者源码与除cuda外网络/训练配置同原GPU配置核验通过；其余四方法既有CPU结果复用。六个单线程CPU进程推进160图，候选/输入/划分哈希不变，参数仍待人审。

- 2026-10-17T03:42:24.973307+00:00 `20261017_public16_segmentation_hyperparameter_review`：CPU完整参数图库完成，16对象×5模型×开发/验证各1图=160张，1952格；32条public16 v2既有活动，不重新提取或重跑PrimGLR。三格式、全网格、源记录/SVG切点、Clasp/CLaP初始化一致性和E2USD CPU核验通过。人工审核pending，未冻结其他模型或运行正式实验一。

- 2026-10-17T04:57:59.793736+00:00 `20261017_public16_segmentation_hyperparameter_review`：按用户要求在原MD报告增量加入PrimGLR，复用同32活动；普通对象144组、H20 240组完整W/gap/alpha/sigma笛卡尔积，新增32图4800格，总计192图6752格。原160图逐文件哈希不变，96个直接调用与缓存调用切点一致，32个已选参考与已有边界一致。仅CPU；不自动改选配置。
