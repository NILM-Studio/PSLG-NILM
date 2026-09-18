# SGN 可复用训练环境

## 范围

本环境用于 `nilm_lab` 中的 SGN 适配实现。Python 3.12、PyTorch 2.5.1、CUDA 11.8 runtime；四个模型独立 Conda 环境，共用固定依赖版本。NILMFormer 禁用可选 xformers。FCN 使用 NILMFormer 基准实现；BERT4NILM 为统一回归训练版；SGN 为论文结构的 seq2seq 门控适配，并不安装整个 NILMTK 社区框架。原版完整训练流程不在本环境验证范围内。

## 已安装环境复用

```bash
source /home/scnu202438025446/miniconda3/etc/profile.d/conda.sh
conda activate sgn-cu118-v1
cd /home/scnu202438025446/pslg-nilm/nilm_experiments
python -m pip check
python runtime/verify_runtime.py --model SGN
python -m nilm_lab.cli --help
```

登录节点没有 GPU，CPU 检查通过时 CUDA available=false 正常。GPU 验证应通过 Slurm 分配计算节点；共享文件系统上的 Conda 环境可直接复用。全部四模型检查脚本：`sbatch runtime/verify_gpu.slurm`。它只使用合成张量，不读取真实数据。

2026-09-15 实测：`h103-slurm-a` 的 CUDA 驱动初始化返回 999，`h104-slurm-a` 正常。因此本次验证和训练提交脚本默认设置 `#SBATCH --nodelist=h104-slurm-a`。确认其他节点 CUDA 正常后可移除此限制；仅 nvidia-smi 能显示 GPU 不足以确认可训练。

## 从文件重建

从 nilm_experiments 目录执行（同名环境已存在时请先修改 YAML 名称，勿覆盖）：

```bash
conda env create -f third_party/nilmtk_contrib/runtime/environment.yml
conda activate sgn-cu118-v1
python runtime/verify_runtime.py --model SGN
```

`runtime/pip-freeze.txt` 与 `runtime/conda-explicit.txt` 是首次成功安装后导出的实际锁定清单；YAML 是跨机器重建配方。PyTorch wheel 自带 CUDA 用户态运行库，无须系统安装 CUDA toolkit；GPU 主机必须有兼容 CUDA 11.8 的 NVIDIA 驱动。

精确版本重建（Linux x86_64，使用新环境名）：

```bash
conda create -n sgn-cu118-rebuilt --file third_party/nilmtk_contrib/runtime/conda-explicit.txt
conda run -n sgn-cu118-rebuilt python -m pip install --no-index --find-links runtime/wheelhouse -r runtime/requirements-lock.txt
conda run -n sgn-cu118-rebuilt python runtime/verify_runtime.py --model SGN
```

Conda 基础包仍需本地缓存或联网取得；上述 Python wheel 安装可离线完成。镜像和 wheel 包未包含真实数据或训练好的模型权重。

## Docker 构建与使用

本次生成的完整镜像归档位于 `../../runtime/images/nilm-cu118-v1.tar`（相对于模型目录）。在有 Docker 的机器，从 nilm_experiments 根目录执行 `docker load -i runtime/images/nilm-cu118-v1.tar`，即可直接使用四个模型标签。实际完成状态及 SHA256 以 `../../runtime/STATUS.md` 为准。

离线重建使用 `docker build -t nilm-runtime:torch2.5.1-cu118-v1 -f runtime/Dockerfile.offline .`，需要保留共享 `runtime/wheelhouse/` 与 `runtime/requirements-lock.txt`。Python 基础镜像也须已缓存；其摘要已固定。下面的普通 Dockerfile 是联网重建方式。Conda 的 Python 为 3.12.14，当前容器基础 Python 为 3.12.9；Python 包版本以共同锁定清单为准。

所有构建从 nilm_experiments 目录执行，先构建一次公共基础镜像：

```bash
docker build -t nilm-runtime:torch2.5.1-cu118-v1 -f runtime/Dockerfile .
docker build -t nilm-sgn:torch2.5.1-cu118-v1 -f third_party/nilmtk_contrib/runtime/Dockerfile .
docker run --rm -v "$PWD:/workspace/nilm_experiments" nilm-sgn:torch2.5.1-cu118-v1
docker run --rm --gpus all -v "$PWD:/workspace/nilm_experiments" nilm-sgn:torch2.5.1-cu118-v1 python runtime/verify_runtime.py --model SGN --device cuda
```

镜像只包含运行环境，通过挂载整个 nilm_experiments 提供模型代码与共享适配器。不要仅挂载当前模型子目录。GPU 容器要求主机 Docker Engine 和 NVIDIA Container Toolkit；本服务器登录节点未安装 Docker，因此优先使用 Conda/Slurm。

迁移：在构建机执行 `docker save -o nilm-runtime-images.tar nilm-runtime:torch2.5.1-cu118-v1 nilm-sgn:torch2.5.1-cu118-v1`，目标机执行 `docker load -i nilm-runtime-images.tar`。镜像实际构建/验证状态见共享 `runtime/STATUS.md`。

## 后续训练入口

使用 O 组，状态概率实际门控功率；不是普通辅助分类。

准备真实配置、训练标签及现有框架要求的配置/数据清单授权文件后，在分配的 GPU 计算节点执行：

```bash
python -m nilm_lab.cli train --config /absolute/path/config.yaml --approval /absolute/path/APPROVAL.json --model SGN --arm O --seed 0 --weight 0.1
```

也可复用提交脚本：`sbatch runtime/train.slurm SGN /absolute/path/config.yaml /absolute/path/APPROVAL.json O 0 0.1`。脚本默认一张 RTX3090、4 CPU、16 GB 内存、12 小时，可按实验资源需求调整。

配置中设置 `training.device: cuda`；真实数据路径必须在主机/容器内可访问。不要把测试数据用于选模。TensorFlow 的 PrimGLR/DeTSEC-PC 发现阶段使用单独环境，本环境只负责四个 PyTorch NILM 模型。此次环境配置不会启动真实训练。
