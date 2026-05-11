# PSLG-NILM: 可扩展机器学习模型工作流框架 (ML Workflow Framework)

这是一个基于步骤（Step）的顺序执行机器学习模型工作流框架，旨在提供一个模块化、可配置且易于扩展的系统。

## 目录结构说明

- `input/`: 原始输入数据文件夹（支持 `.csv` / `.npy` / `.txt` 格式，实际读取格式以 `DataLoaderStep` 为准）。
- `doc/`: 项目文档目录，包含各 Step 的详细输入输出说明。
- `log/`: 存储工作流执行过程中的缓存文件和日志。
  - 每个执行序列号（时间戳标识）创建独立子文件夹。
  - 按步骤（Step）分类存储中间生成内容。
- `output/`: 最终结果输出文件夹。
  - 每个序列号创建独立子文件夹。
  - `output/`: 存储生成的预测结果或处理后的数据。
  - `figure/`: 存储生成的图表。
- `models/`: 模型文件夹，存放所有模型相关的 Python 脚本和子文件夹。
- `src/`: 核心框架代码实现。
  - `framework/`: 包含 Workflow 基类、Step 抽象基类和日志记录功能。
  - `steps/`: 实现具体的工作流步骤（如数据加载、处理、模型训练等）。
- `config/`: 配置文件模板（YAML 格式）。

## 开发规范

请务必阅读项目主文件夹下的 [GUIDELINES.md](file:///f:/B__ProfessionProject/PSLG-NILM/GUIDELINES.md) 文档，以了解如何扩展 Step、添加 Model 以及遵循目录结构和输出规范。

## 工作流执行架构

1. **唯一标识符**: 每个工作流进程会自动生成一个唯一的时间戳作为序列号（例如：`20260415_221145`）。
2. **顺序执行**: 每个步骤必须按预定顺序执行，上下文（Context）在步骤之间共享。
3. **流程**:
   - 从 `input`（或上游 Step 生成的缓存目录）读取数据（常见为 `.csv`）。
   - 处理数据并生成中间缓存文件到 `log` 文件夹。
   - 调用模型进行训练并直接生成结果。
   - 将最终结果和图表保存到 `output` 文件夹。

## 使用方法

### 1. 安装依赖项

确保已安装必要的 Python 库：

```bash
pip install -r requirements.txt
```

### 2. 准备数据

本框架支持两种常见的数据输入方式（推荐使用配置文件控制）：

#### 方式 A：直接把 CSV 放入 input/（最简单）

1. 将一个或多个电器的 CSV 文件放到 `input/` 目录下（每个文件至少包含 `timestamp,power` 两列）。
2. 在 `config/config.yaml` 中开启 `steps.data_loader.enabled: true`。

`DataLoaderStep` 会把 `input/` 下的 CSV 拷贝到本次运行的缓存目录，并加载到 `context['data']`。

#### 方式 B：用 ExtractActiveDataStep 从外部大 CSV 切割工作区间（UKDALE 场景）

适用于输入是 UKDALE 预提取目录（例如 `/home/scnu2023024258/data/datasets/ukdale_extracted/house1`）中某个电器的整段 CSV。

1. 在 `config/config.yaml` 中开启 `steps.extract_active_data.enabled: true`。
2. 设置 `steps.extract_active_data.input_file` 为目标电器的 `.csv` 文件路径（必须是文件路径，不是目录）。
3. 建议同时开启 `steps.data_loader.enabled: true`，并保持 `steps.extract_active_data.set_input_root: true`（默认值）。

这样会先在 `log/.../ExtractActiveData/segments/` 生成切割后的多个 CSV，再由 `DataLoaderStep` 直接加载这些切割结果供后续步骤使用。

### 3. 配置工作流

在 `config/config.yaml` 中配置所需的步骤和模型参数。

### 4. 运行工作流

使用以下命令运行工作流：

```bash
python main.py --sample  # 创建示例数据并运行
```

或者使用现有数据运行：

```bash
python main.py --config config/config.yaml
```

### 4.1 使用 Slurm 提交训练（run_train.sh）

仓库提供了示例 Slurm 脚本 [run_train.sh](file:///home/scnu2023024258/data/code/PSLG-NILM/run_train.sh) 用于在集群上提交训练任务，其中：

- `#SBATCH -p A1000`：指定提交到名为 `A1000` 的分区（partition/queue）。不同集群会把不同 GPU 资源划分到不同分区，因此该字段需要按实际可用分区名修改。
  - 使用 A1000 节点：保持为 `A1000`
  - 使用 RTX3090 节点：将其改为 `RTX3090`（前提是集群存在该分区名）

### 5. 断点续跑（切断后继续）

工作流执行过程中每个 Step 会在自己的缓存目录下写入完成标记文件：`log/{sequence_id}/{step_name}/.done`。

- 方式 1（推荐，命令行）：用同一个 sequence_id 恢复
  - `python main.py --resume --sequence-id <原sequence_id>`
  - 参数入口在 [main.py](file:///home/scnu2023024258/data/code/PSLG-NILM/main.py)
- 方式 2（YAML）：在 `workflow` 下配置 `resume: true` 和 `sequence_id: ...`，`main.py` 会读取并恢复执行
- 想强制重跑某一步：删除该 step 目录下的 `.done` 文件即可（例如 `log/<id>/FeatureExtract/.done`）

## 扩展说明

- **添加新步骤**: 在 `src/steps/` 中创建一个继承自 `Step` 的类，并实现 `run` 方法。
- **添加新模型**: 在 `models/` 中创建一个继承自 `BaseModel` 的类，并实现 `train`, `predict`, `save`, `load` 方法。
- **自定义配置**: 在 `config/config.yaml` 中添加相应的配置项。
