# PSLG-NILM: 可扩展机器学习模型工作流框架 (ML Workflow Framework)

这是一个基于步骤（Step）的顺序执行机器学习模型工作流框架，旨在提供一个模块化、可配置且易于扩展的系统。

## 目录结构说明

- `input/`: 原始输入数据文件夹（支持 `.npy` 和 `.txt` 格式）。
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
   - 从 `input` 读取数据（`.npy` 或 `.txt`）。
   - 处理数据并生成中间缓存文件到 `log` 文件夹。
   - 调用模型进行训练并直接生成结果。
   - 将最终结果和图表保存到 `output` 文件夹。

## 使用方法

### 1. 安装依赖项

确保已安装必要的 Python 库（如 `numpy`, `matplotlib`, `pyyaml`）：

```bash
pip install numpy matplotlib pyyaml
```

### 2. 准备数据

将原始数据文件放置在 `input` 文件夹中。支持 `.npy` 和 `.txt` 格式。

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

## 扩展说明

- **添加新步骤**: 在 `src/steps/` 中创建一个继承自 `Step` 的类，并实现 `run` 方法。
- **添加新模型**: 在 `models/` 中创建一个继承自 `BaseModel` 的类，并实现 `train`, `predict`, `save`, `load` 方法。
- **自定义配置**: 在 `config/config.yaml` 中添加相应的配置项。
