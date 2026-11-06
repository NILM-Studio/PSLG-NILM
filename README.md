# PSLG-NILM

主流程代码、模型实现与精选发布模型权重。该仓库不包含论文实验编排、对照实验和论文图表生成脚本，也不宣称独立复现论文全部结果。

## 目录

- `main.py`：工作流入口。
- `src/`：主流程框架、分割/表征/聚类/状态构建、NILM 及共享表征工具。
- `models/`：模型实现、适配器及必要第三方源码和许可证。
- `config/config.yaml`：基础流程配置；数据路径由使用者指定。
- `requirements.txt`：主流程依赖声明。
- `checkpoints/`：通过 Git LFS 保存的发布权重。

## 获取权重

```bash
git lfs install
git clone https://github.com/NILM-Studio/PSLG-NILM.git
cd PSLG-NILM
git lfs pull
```

全部权重约 57 GiB。只获取源码时，在克隆命令前设置 `GIT_LFS_SKIP_SMUDGE=1`；之后可使用 `git lfs pull --include="checkpoints/<group>/**"` 按组下载。

权重按 E1 结构扰动、E2 表征/分割、E3 生成骨干/条件/字典、E4 编辑/排列、E5 NILM/增广，以及评价编码器分组。共 1,085 套 checkpoint、1,127 个权重文件；TensorFlow 的 `.index` 与 `.data-*` 分片必须共同下载。TimeVQVAE Stage-1 和 Stage-2 分开保留。文件夹保留任务名称及唯一标识，以区分设备、方法与随机种子。

发布的是原始序列化权重文件，不另附审计报告、来源清单、校验脚本、数据集或实验结果。部分 checkpoint 内含训练状态；不同模型需要对应架构、超参数与运行环境，不能统一由 `main.py` 自动加载所有历史权重。第三方源码的授权以其随附许可证为准。

## 主流程

建议在 Linux、Python 3.12 环境安装依赖。神经网络后端按机器的 CPU/CUDA 环境单独安装。

```bash
python -m pip install -r requirements.txt
python main.py --help
python main.py --steps extract,segment,feature,cluster   --segment-method prim-glr --feature-model dtw   --raw-series /path/to/power.csv --appliance example --run-id example
```

输入格式和阶段参数见 `main.py` 与 `config/config.yaml`。完整 NILM 流程还需要使用者提供匹配的数据及配置。运行输出写入本地 `runs/`，不会纳入此发布范围。

`runs/`、论文材料、数据准备与划分脚本、测试、虚拟环境和完整运行快照均不在当前发布树中。已有 Git 历史保留。
