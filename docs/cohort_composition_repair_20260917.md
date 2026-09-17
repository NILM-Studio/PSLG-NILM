# 先修复评价协议：按时期建立新的基元拼接实验

## 为什么要另开一轮

用户回传的 `ukdale_wm_composition_20260917` 使用提交 `af7dbd2`，完成了四组拼接，但没有可用验证集。按活动 ID 关联时间划分、类别、逐周期检查与最终白名单后得到：

| 阶段 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 原始时间划分 | 1043 | 149 | 298 |
| 匹配类别 | 921 | 135 | 261 |
| 单周期检查通过 | 776 | 103 | 190 |
| 整类准入后保留 | 619 | 0 | 0 |

留出侧单周期通过的成员全部属于被标为 uncertain 的类别 3、5、9、11。按原报告配置的 `2015-09-08T00:00:00+01:00` 分界，原始1490个周期中，925个在前、565个在后；保留的619个全部位于分界前。原验证和测试周期全部位于分界后。

这说明时期分布与整类筛选发生冲突，不证明后期波形全部损坏。分界日期来自既有实验配置，不在本轮按拼接分数搜索；日期过滤本身不证明设备更换或时期内严格平稳。

## 本次改什么、不改什么

1. 从继承上游的**原始周期清单**出发，先选择完全落在指定时期内的活动，再做70/10/20全局时间划分，随后仅用训练侧建立下游类别、模式和筛选规则。
2. 全部活动保留原始ID与源文件索引。时期外活动标为 `outside_cohort`；跨边界活动整段排除，不裁剪、不重新编号、不复制原始数据。时间窗口为 `[start, end)`，包含起点，排除终点；比较的是含上下文的整个提取区间。
3. 增加 `cycle_cohort_audit.json`，逐阶段核对成员、拟合范围、时间隔离、类别排除原因及各Class/Mode覆盖。筛选后任一划分为空，或训练与验证/测试没有任何同组数据时，入口在拼接前退出并留下诊断包。
4. 保留已有五组选择方法与默认参数：random、boundary_greedy、boundary_dp、transition_dp、unit_selection。它们来自 `5953f4f`，不是这次为了改善结果新增的方法。
5. 不修改师兄的切分、特征、聚类、状态合并算法，不重新训练这些上游模块。不训练师弟的生成模型，不训练NILM。
6. **不修改配置中的类别/成员筛选门槛，不强制放行类别，不改为随机划分。** “公共状态应否成为整类否决条件”等方法学问题另行审议。本次先隔离时期混合因素，不能宣称所有筛选偏差已经解决。

连续NILM数据入口目前不支持这个有界时期协议，已增加显式拒绝，防止其沿用全时期背景区间而越过窗口。未限定时期的旧入口保持兼容；本轮不要把新目录直接送入连续NILM训练。

不要把旧的619个合格周期直接拆成新训练/验证/测试。它们已经参与旧下游结构与规则拟合；新的划分必须在新的下游目录中、在结构拟合前进行。

## 服务器命令

在允许CPU计算的环境中执行，无需GPU：

```bash
cd /home/scnu2024024563/NILM/PSLG-NILM
git switch feature/primitive-generation
git pull --ff-only origin feature/primitive-generation
conda activate pslg-nilm
bash scripts/run_cohort_composition_validation.sh
```

默认继承 `ukdale_wm_primglr_detsec_3789`，全新目录为：

`log/ukdale_wm_composition_prechange_20260917/`

默认只纳入**结束时间严格早于** `2015-09-08T00:00:00+01:00` 的完整周期。先运行下游时间划分/分类/验证/最终目录，再审计评价覆盖，通过后才拼接。仍为5%、10%、20%、100%预算，seed42，每预算最多30锚点，不做大规模调参。

原目录已存在则拒绝覆盖。需要保留失败产物重跑时，显式换新名字：

```bash
RUN_ID=ukdale_wm_composition_prechange_20260917_r2 \
bash scripts/run_cohort_composition_validation.sh
```

`SOURCE_RUN_ID`、`PYTHON_BIN`沿用旧入口。`COHORT_START`和`COHORT_END`可指定明确时区的ISO时间，但首轮请保持默认边界，不根据结果反复挑时期。旧 `run_composition_validation.sh` 默认不限定时期，如不传窗口仍覆盖全时期；它现在也会在评价覆盖不满足时停止，不应反复运行它来替代本轮入口。

直接使用主程序时支持 `--cohort-start` / `--cohort-end`，也支持配置的 `temporal_holdout.cohort_start/cohort_end`。窗口只影响该下游时间划分步骤，不改变上游原始文件。

## 回传一个文件即可

初始化成功后，无论下游检查通过还是因筛选失败退出，入口都会尝试生成：

`log/ukdale_wm_composition_prechange_20260917_diagnostics.tar.gz`

下载这个文件并回传。它包含已有JSON、CSV、日志及生成成功时的代表性PNG，排除NPY/NPZ大数组。打包不会上传Git，也不会修改源报告。若初始化阶段失败或打包失败，直接回传终端报错。

入口还保存并使用 `downstream_config.yaml` 配置快照，审计记录其哈希，便于确认服务器实际使用的筛选阈值。

最先看 `cycle_cohort_audit.json`：

- `metadata_integrity_passed`：保存的阶段产物能否按ID、来源和数量对齐；不是原始波形重建审计。
- `evaluation_ready`：本轮是否有非空且至少部分组匹配的训练/验证/测试。不是每个组都充分，须结合 `groups` 和缺失组列表。
- `stages`：每个划分从分配、分类、硬检查、单周期有效到最终保留的数量；`valid_but_class_excluded`单列“周期合格但类别不准入”。
- `blockers`：明确指出阻断条件。

如需单独复核已经存在的目录，指定一个**未存在**的报告路径：

```bash
python -m scripts.audit_cycle_cohort \
  --run-id ukdale_wm_composition_20260917 \
  --output log/ukdale_wm_composition_20260917/cohort_recheck.json
```

这个审计不修改旧划分或拟合模型。加 `--require-evaluation-ready` 时，完整性正常但评价不可用会返回退出码2。

## 拼接结果状态修正

- `generation_only_no_validation`：生成完成，但没有验证周期；退出码2。
- `generation_only_no_matching_validation`：有验证周期，但没有生成案例所属Class/Mode的验证参考；退出码2。
- `partial_validation_coverage`：只覆盖部分案例，未覆盖组不能混入效果结论。
- `ready_for_descriptive_review`：生成、审计和同组验证覆盖可用于描述性检查，不证明独立样本数量足够或方法有效。

新的summary记录 `validation_availability`、`source_protocol` 与 `distinct_anchor_cycles`。旧四组报告也会根据实际验证条目补充缺失警告，不因旧版保存了ready状态就忽略空验证集。

## 本地验证与剩余限制

全套测试通过：`275 passed, 6 warnings`。警告来自本机物理CPU核数探测及测试数据重复点下的GMM收敛提示；两个Bash入口均通过语法检查。

测试包括实际Bash入口的完整临时CSV/NPY工作流：时期外样本不能参与类别拟合；原ID不变；含上下文的跨界周期不裁剪；成功与失败均留下诊断包；旧结果不覆盖；空验证、无同组参考及部分覆盖分别报告。

还用用户回传的真实元数据包对审计器进行只读复核，复现了619/0/0及103/190个单周期通过但被整类排除的计数。**这不是在本地跑过新的真实UK-DALE拼接实验。** 新时期范围下的真实覆盖与效果仍须服务器运行确认。

本轮仍继承上游表示，完整上游拟合范围没有得到训练侧认证，不能声称端到端无泄漏。此前全时期结果用于协议排查，不应包装为完全未查看的最终测试。先固定修订协议并检查覆盖，后续效果实验、多种子与NILM评价另行安排。测试侧这里只看覆盖/筛选元数据，不用其性能分数调参。
