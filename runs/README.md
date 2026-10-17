# 实验注册表

参照 `/home/avis/post_training_rep/README.md` 的实验、执行、报告三级结构。

| 实验 | 内容与边界 |
|---|---|
| [20261016_public_appliance_signature_screening](20261016_public_appliance_signature_screening/README.md) | 七公开数据筛选完成：495对象、481张逐对象十子图，目视建议新增6、条件性21；九既有对象排除，用户审核pending，CPU-only，无聚类/生成训练 |
| [20261016_darck_deddiag_snm_acquisition](20261016_darck_deddiag_snm_acquisition/README.md) | 三数据集完整选定发布物25.54GB均核验；DEDDIAG全部15住宅，SNM 120原生H5可读及采样/通道审计；报告/环境齐全，未启动聚类或统一H5转换 |
| [20261016_greend_acquisition_nilm_catalog](20261016_greend_acquisition_nilm_catalog/README.md) | GREEND官方0.2原始ZIP完整性通过；发布小H5为子集；68个NILM/相关资源分级检索与来源、环境和覆盖图；未启动聚类 |
| [20261016_cosci_gan_validation](20261016_cosci_gan_validation/README.md) | COSCI-GAN 官方 MLP/LSTM 训练与采样 smoke；Diffusion-TS、TimeVQVAE 已部署跳过 |
| [20261015_multidataset_primglr_dtw_kmeans](20261015_multidataset_primglr_dtw_kmeans/README.md) | 已确认七对象全量记录，REFIT 8秒、UK-DALE/REDD原生点；147个k/种子组合完成，独立验证通过，活动语义待人工审核 |
| [20260915_sequence_workflow](20260915_sequence_workflow/README.md) | 历史 NILM 协议；原其他服务器的数据路径未在本机验证 |
| [20260915_primitive_vs_direct](20260915_primitive_vs_direct/README.md) | 历史基元与直接监督协议；不是本次重跑 |
| [20261013_multidomain_generation](20261013_multidomain_generation/README.md) | 非 F4 多域条件生成；保留原执行内部层级 |
| [20261015_mainline_restoration](20261015_mainline_restoration/README.md) | 主路线恢复与真实数据 smoke |
| [20261015_downstream_generation_validation](20261015_downstream_generation_validation/README.md) | 七个公开生成实现的接口 smoke；C-TimeGAN 无公开源码 |

新实验使用 [_template](_template/)。每次阶段结束追加 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。
F4 和含 F4 的混合历史实验移到 `../archive/retired_f4/experiments/`，不再作为活动实验入口。

| [20261016_public_all_activities_primglr_dtw_kmeans](20261016_public_all_activities_primglr_dtw_kmeans/README.md) | 用户授权七对象全量活动拟合，无训练/验证/测试划分，按对象并行；状态以执行索引为准 |

| [20261016_public_activity_segmentation_recalibration](20261016_public_activity_segmentation_recalibration/README.md) | 九对象活动提取与PrimGLR切分复核完成；456配置、36全量条件、45张十子图；超参数与图形核验归档 |
| [20261016_nine_object_activity_generation](20261016_nine_object_activity_generation/README.md) | 九对象活动功率波形生成实验执行中；仅 U 条件，Diffusion-TS / TimeVQVAE / COSCI-GAN；DS、PS、C-FID、双向最近邻 DTW；九对象输入已完成，两条 GPU 1 训练队列及评价队列运行中，质量结果待正式报告 |

| [20261016_primitive_condition_interfaces](20261016_primitive_condition_interfaces/README.md) | 三模型P条件接口及CPU小规模验证；U由独立实验执行 |

| [20261017_public16_primglr_w120_gap30_case_review](20261017_public16_primglr_w120_gap30_case_review/README.md) | 用户授权16对象原始区间与主步骤活动片段双副本、PrimGLR W120s/gap30s十例图审，执行状态以索引为准 |

- [20261017_public16_primglr_w120_gap60_case_review](20261017_public16_primglr_w120_gap60_case_review/README.md)：同160例PrimGLR W120/gap60图审。

| [20261017_public16_primglr_feature_pca16](20261017_public16_primglr_feature_pca16/README.md) | 待用户审核：16对象2824活动，PrimGLR共同核心片段，七表示＋训练侧PCA16，GPU1/2/6与2的指数监控；未启动计算 |

- [20261017_public16_all_supported_max600_correction](20261017_public16_all_supported_max600_correction/README.md)：纠正100/150限额，16对象全量支持片段<=600及两版图审。

| [20261017_public16_segmentation_hyperparameter_review](20261017_public16_segmentation_hyperparameter_review/README.md) | public16 v2既有活动，六模型完整参数图审；每对象每模型开发/验证各1张，共192张，全部CPU，参数待人审 |
