# 下游生成模型

代码位于 `/home/avis/pslg-nilm/models/downstream_generation_models`。
实验入口为 [下游验证实验](../../runs/20261015_downstream_generation_validation/README.md)，结果不写回本目录。

| 模型 | 官方代码位置 | 最小验证边界 |
|---|---|---|
| Diffusion-TS | vendor/Diffusion-TS | 前向、损失、反传、采样 |
| CSDI | vendor/CSDI | 条件掩码损失与反传 |
| ImagenTime | vendor/ImagenTime | delay embedding 可逆性、损失、反传；不验证 STFT |
| TimeVAE | vendor/timeVAE | train_step 与先验采样 |
| TimeVQVAE | vendor/TimeVQVAE | Stage-1 反传、Stage-2 条件损失与采样 |
| TimeGAN | vendor/TimeGAN | TF1 隔离环境中的 1 轮 smoke |
| Fourier-flows | vendor/Fourier-flows | 2 轮拟合与采样 |
| COSCI-GAN | vendor/COSCI-GAN | 官方训练函数，MLP/LSTM 两配置，中央 MLP 判别器参与更新、权重重载与共同噪声三通道采样 |
| C-TimeGAN | 无公开代码 | blocked-no-public-code，不用自制模型冒充复现 |

仓库与提交号见 `config/model_registry.json`。`vendor/` 保留官方实现；兼容层在 `scripts/`，TimeGAN 的列表反归一化修复仅作用于临时副本。
TimeVQVAE 使用微型未收敛模型；所有 smoke 都不是生成质量评价，也不构成论文指标复现。

## 环境要求

- Torch：项目根 `.venv-generation-torch`，Python 3.12 / Torch 2.5.1+cu118；附加依赖位于 `runtime/generation/torch_site`，启动器自动设置 PYTHONPATH。
- TimeVAE：项目根 `.venv-generation-tf`，TensorFlow 2.17.1。
- TimeGAN：项目根 `runtime/envs/timegan`，Python 3.7 / TensorFlow 1.15 / NumPy 1.18.5 / protobuf 3.20.3。不能用 TF2 替代其 tf.contrib API。默认 CPU 验证。
- 配方见 `environment/`，安装器为 `scripts/setup_runtime.sh` 和 `scripts/setup_timegan_legacy.sh`。完整说明见 [服务器指引](../../guides/server-usage.md) 和 `docs/DEPLOYMENT_PLAN.md`。
- 当前环境为兼容性 smoke 环境，不保证所有上游完整训练脚本无修改可用。
- ImagenTime 上游逆变换硬编码 `.cuda()`，整套 smoke 需要可用 CUDA GPU；不能隐藏全部 GPU。TimeGAN 子进程仍强制使用 CPU。

## 数据与执行

输入为 `datasets/self_collected/{Air_condition,Microwave,WashMachine}.csv`；每设备前 24 小时，以 6 秒均值重采样，长度 128、每设备 32 窗口；全体选中窗口的 q01/q99 缩放至 [-1,1]。该缩放只用于接口测试，正式评估必须只在训练集拟合缩放参数。

```bash
cd /home/avis/pslg-nilm
EXEC=$(python3 scripts/create_execution.py 20261015_downstream_generation_validation smoke)
python3 scripts/run_execution.py "$EXEC" -- bash models/downstream_generation_models/scripts/smoke_all.sh
```

输出保存到执行的 `analysis/`，数据 manifest 保存到 `inputs.json`，控制台日志在 `logs/console.log`。
单模型脚本要求显式 `--output`，避免污染源码目录；可传 `--data` 使用已有 npz。
共享准备数据保留于 `runs/20261015_downstream_generation_validation/data/`，迁移前结果在 `executions/smoke/pre_migration/analysis/`。
下载慢时使用本地 `ssh -N -R 17897:127.0.0.1:7897 zhouslab` 临时转发，不永久改代理。

## 2026-10-16 补充

Diffusion-TS（[论文](https://arxiv.org/abs/2403.01742)）与 TimeVQVAE（[论文](https://arxiv.org/abs/2303.04743)）已有固定提交与通过记录，本次按要求跳过重复部署，不声称今天重新训练了这两个模型。
新增 COSCI-GAN（[NeurIPS 2022 论文](https://arxiv.org/abs/2205.13741)、[官方代码](https://github.com/aliseyfi75/COSCI-GAN)），完整环境和运行说明见 [COSCI_GAN.md](docs/COSCI_GAN.md)。
本次测试报告见 [COSCI-GAN 验证实验](../../runs/20261016_cosci_gan_validation/README.md)。现有 `smoke_all.sh` 保持原七模型口径；COSCI-GAN 使用独立登记执行，避免覆盖不同数据协议的输入清单。
