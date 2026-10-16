# 部署方案与环境决策

## 1. 结论

报告列出的 8 个模型中，7 个存在可核验公开仓库并已固定提交；C-TimeGAN 没有报告可指向的公开代码，因此不进入“已复现”列表。当前部署把上游源码、适配代码、数据快照、运行时补丁和结果分开，保证后续可以替换模型而不改原始数据与上游仓库。

## 2. 目录职责

```text
models/downstream_generation_models/
├── vendor/        # 上游 git 仓库，禁止直接改源码
├── scripts/       # 本项目数据适配、smoke test、兼容处理
├── config/        # 仓库地址、提交、状态
├── environment/   # 完整参考环境规格
└── docs/          # 方案与已知限制
```

依赖 overlay 位于项目根 `runtime/generation/torch_site`，TimeGAN 环境位于 `runtime/envs/timegan`。
数据和结果位于 `runs/20261015_downstream_generation_validation/`；新运行必须通过根目录的 create_execution.py / run_execution.py 登记，见 [当前部署入口](../README.md)。

## 3. 为什么分三个运行时

当前项目已有 Torch 2.5.1/CUDA 11.8 与 TensorFlow 2.17.1 环境。Torch 模型共用现有解释器，`loguru`、线性注意力等额外依赖位于项目根 `runtime/generation/torch_site`。TimeVAE 使用现有 TensorFlow 2 环境。ImagenTime 官方逆变换硬编码 CUDA，整套 smoke 必须使用可用 GPU。

TimeGAN 官方提交仍使用 TensorFlow 1.x 的 `tf.contrib`、图会话和旧 RNN cell，不能把它硬塞进 TF 2.17/Keras 3。它使用独立的 Python 3.7 + TF 1.15 环境。官方末尾把 Python list 直接乘 ndarray，是可确认的兼容 bug；smoke 脚本只在临时目录复制源码并将该行改为 `np.asarray(generated_data) * max_val`，原始 vendor 不变。

上游环境与部署环境并非逐字相同：Diffusion-TS 原 requirements 固定 Torch 2.0.1；ImagenTime 给出 Torch 2.0/CUDA 11.8；TimeVQVAE 当前 pyproject 要求 Python >=3.12、Torch >=2.10、NumPy >=2.4.2。最小验证以服务器已有 Torch 2.5.1 为共同兼容基线，实际通过 smoke 才标为可用。若正式训练出现数值或算子差异，再为相应模型按其锁文件建立独立环境。

ImagenTime 的固定提交中 `utils/utils_data.py` 无条件导入了仓库里并不存在的 `data.data_provider` 与 `data.long_range`，但 delay embedding 实际只需同文件里的两个缩放函数。适配器在进程内提供等价的两个函数后加载官方模型，不修改上游源码；STFT 与仓库自带数据加载器不在本次 smoke 的通过范围。

## 4. 最小验证口径

最小验证不追求收敛，只验证以下链路：

1. 真实 `datasets/self_collected` CSV 可读取，时间轴可对齐到 6 秒。
2. 同一份 `(batch, length, feature)` 数据能进入各适配器。
3. 模型能构建，至少完成一次前向；可训练模型再完成一次反传/优化。
4. 生成型 API 至少返回一个有限值样本；可逆变换检查形状和误差。
5. 每个模型独立记录 pass/fail、运行时、输入/输出形状和损失。

这只验证工程可运行性。正式论文实验还需要按活动而不是随机点划分数据，防止相邻窗口泄漏；缩放参数只能用训练集拟合；至少报告真实性、覆盖度/多样性、条件一致性和 NILM 下游增益。

## 5. 与“活动状态提取—切分—特征—聚类—合并”的衔接

生成模型不应直接取代上游活动发现流程。建议接口为：上游输出 `appliance_id/activity_id/state_id/start/end/duration/power_level` 与定长或带 mask 的片段；生成层使用状态、时长、设备与功率水平作为条件；生成后反缩放并执行非负功率、边界连续性和状态转移合法性检查；最后才合并为活动或整日轨迹。

阶段顺序建议：

1. 用 TimeVAE、Fourier-flow 建立无条件低成本基线。
2. 用 CSDI 验证观测 mask 条件，用 Diffusion-TS 验证片段生成。
3. 用 ImagenTime 比较长活动；用 TimeVQVAE Stage-1 学离散基元，再训练 Stage-2 先验。
4. TimeGAN 只作为经典 GAN 对照，不让旧环境成为主流水线依赖。
5. C-TimeGAN 等获得作者实现后再评估；若自行重实现，结果必须标注“reimplementation”，不能标为官方复现。

## 6. 网络与代理

部署文件不保存代理。下载慢时，由本地 SSH 反向转发 `127.0.0.1:7897` 到远端临时端口，并只给 clone/pip/conda 命令设置 `HTTP_PROXY`/`HTTPS_PROXY`。完成后退出 SSH 会话即失效，避免训练任务意外依赖本地电脑。
