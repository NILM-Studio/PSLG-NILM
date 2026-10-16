# COSCI-GAN 环境、部署与测试

论文：Generating multivariate time series with COmmon Source CoordInated GAN，NeurIPS 2022，[arXiv:2205.13741](https://arxiv.org/abs/2205.13741)，[OpenReview](https://openreview.net/pdf?id=RP1CtZhEmR)。
官方代码：https://github.com/aliseyfi75/COSCI-GAN 。固定提交 `1c2c8b340599c149af44a2bc0e96116025770fe9`。

## 目录与安装

源码位于 `models/downstream_generation_models/vendor/COSCI-GAN/Code`，项目适配入口为 `models/downstream_generation_models/scripts/smoke_cosci_gan.py`。
使用 Git sparse checkout 获取 Code 及根目录 README/LICENSE；不下载与本次自采数据验证无关的官方 Dataset/Experiments/Figures。需要完整资源时可在代理可用的情况下扩展 sparse checkout，本次不是缺失训练源码的替代实现。

复用 `.venv-generation-torch`：Python 3.12.0、Torch 2.5.1+cu118、NumPy 1.26.4、Pandas 2.2.3；绘图使用 Matplotlib 3.9.2。无新增包、无需 overlay，不修改 TF1/TF2 环境。
上游没有本次所用版本的依赖锁文件；`environment/cosci-smoke-requirements.txt` 是本项目实际验证配方，不是论文原始环境声明。其他主机建议新建 Python 3.12 venv 后安装：

```bash
python3.12 -m venv .venv-cosci
.venv-cosci/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
.venv-cosci/bin/pip install -r models/downstream_generation_models/environment/cosci-smoke-requirements.txt
```

## 最小测试

```bash
cd /home/avis/pslg-nilm
EXEC=$(python3 scripts/create_execution.py 20261016_cosci_gan_validation lstm_central_mlp)
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 scripts/run_execution.py "$EXEC" -- .venv-generation-torch/bin/python \
  models/downstream_generation_models/scripts/smoke_cosci_gan.py --architecture lstm
```

另建执行将 `--architecture` 改为 `mlp`，不要覆盖已有执行。CPU 验证不要求 GPU、CUDA 驱动或 Slurm。
官方 `COSCIGAN` 的 Dataset/Results 相对路径通过独立工作目录接入；不改 vendor、不改损失。
仅用 Adam 观测包装检查梯度有限和参数更新，所有实际优化仍调用原 Adam.step。

三个真实设备功率按 datetime 同步对齐：首 24 小时、6 秒网格、均匀选取 16 个长度 64 的三通道窗口。逐通道 min/max 归一化；batch=8、1 epoch、gamma=5、noise_len=32，中央判别器为 MLP。
已通过 MLP 与 LSTM 两种局部网络配置：七个优化器各更新 2 次、梯度有限、参数确实变化；三份生成器权重严格重载成功；共同噪声生成有限的 `4×64×3` 数组。
输入来源/哈希与完整参数在每次执行的 inputs.json/config.resolved.json，实际环境在 environment.json，权重在 checkpoints/，采样在 generations/。

## 适用性与限制

共同噪声协调各通道，中央判别器检查联合分布，适合探索多设备功率的协同生成；但本次没有实现“基元状态＋时长＋功率水平”的条件接口，不能把共同噪声称为已实现基元条件生成。
1 epoch smoke 不证明相关性保真、训练收敛或论文指标；没有正式测试集。样本图只展示产物形状和数值，不作生成质量结论。
未验证 Pairwise/LSTM 中央判别器、GPU/多卡和上游全部任务。保留官方训练逻辑中的实现细节，不擅自修正成另一种 GAN。
F4 不作为依赖或恢复对象。其余列出的 C-VAE、WGAN-GP 不在本次三个模型的部署范围内。

Diffusion-TS 和 TimeVQVAE 已存在，固定提交分别为 `566307e6cf2d8095e58de4c6e3a6ae965b69b5b5`、`39f98e99cb103ea00694d01c65fd6e7a49416d24`，沿用 2026-10-15 的通过记录。
