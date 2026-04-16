# PSLG-NILM 项目开发规范与规则指南

本指南旨在详细说明 PSLG-NILM 工作流框架的设计原则、开发规范以及如何扩展系统功能。

---

## 1. 项目目录结构定义

框架采用严格的目录隔离，确保数据、代码、日志和结果互不干扰。

- **`input/`**: 原始输入数据。支持 `.npy`, `.txt`, `.csv` 格式。所有工作流的数据源头。
- **`log/`**: 存储执行过程中的**缓存文件**、**日志**和**中间生成内容**。
  - 格式：`log/{sequence_id}/{step_name}/`。
  - `sequence_id` 为工作流启动时生成的唯一时间戳标识符。
  - 每个 Step 应将中间产物存储在自己的子文件夹中以便追踪。
- **`output/`**: 存放**最终输出结果**。
  - 格式：`output/{sequence_id}/`。
  - `output/output/`: 存储标准化的输出结果文件。
  - `output/figure/`: 存储生成的图表和可视化文件。
- **`models/`**: 存放所有模型相关的 Python 脚本和子文件夹。
- **`src/`**: 核心代码。
  - `src/framework/`: 包含 Workflow 引擎、Step 抽象基类和 Logger 工具。
  - `src/steps/`: 包含所有具体步骤的实现类。
- **`config/`**: 存放 `config.yaml` 配置文件。

---

## 2. 核心类定义规范

### Step 类 (工作流步骤)
所有步骤必须继承自 `src.framework.step.Step` 基类。

- **继承要求**: 必须实现 `run(self, context: dict) -> dict` 抽象方法。
- **上下文管理**: 
  - `context` 字典在步骤间共享，包含 `sequence_id`, `log_root`, `output_root` 和 `data`（内存数据缓存）。
  - 每个步骤完成后应返回更新后的 `context`。
- **日志目录获取**: 调用 `self.get_log_dir(context)` 获取当前步骤专用的缓存文件夹。
- **设计模式**: 面向对象、模块化设计，每个步骤应职责单一。

### Model 类 (机器学习模型)
所有模型必须继承自 `models.base_model.BaseModel` 基类。

- **接口要求**: 必须实现 `train(self, data)`, `save(self, path: str)` 和 `load(self, path: str)`。
- **预测说明**: 本项目采用“训练直接得到结果”的模式，不强制要求独立的 `predict` 方法。
- **命名规范**: 以模型名称命名的 Python 脚本或子文件夹，命名清晰。

---

## 3. 配置文件 `config.yaml` 使用规则

`config/config.yaml` 是工作流的控制中心。

- **启用/禁用步骤**: 使用 `enabled: true/false` 开关控制步骤是否进入执行队列。
- **参数传递**: 在 `steps` 下定义的参数可在 `main.py` 中解析并传递给对应的 Step 构造函数。
- **动态控制**: 工作流的执行顺序由 `main.py` 中的 `wf.add_step()` 调用顺序决定，通常与 YAML 中的顺序一致。

---

## 4. 数据读取与输出规范

### 读取 Log (缓存)
- 后续步骤应从 `context['log_root']` 对应的文件夹中读取前序步骤生成的中间数据（如 `log/{sequence_id}/DataLoader/xxx.csv`）。
- 严禁直接访问 `input/` 文件夹，除非是 `DataLoader` 步骤。

### 输出结果
- **中间结果**: 保存到 `self.get_log_dir(context)`（即 `log/` 下的步骤目录）。
- **最终结果**: 
  - 数据结果存入 `os.path.join(context['output_root'], 'output')`。
  - 可视化图表存入 `os.path.join(context['output_root'], 'figure')`。
- **命名规范**: 文件名应包含关键参数或模型标识符，便于追踪和调试。

---

## 5. 如何添加新功能

### 添加一个新 Step
1. 在 `src/steps/` 下创建新文件，定义继承自 `Step` 的类。
2. 实现 `run` 方法，利用 `context` 进行输入输出。
3. 在 `main.py` 中导入新类，并根据 `config.yaml` 的配置调用 `wf.add_step()`。

### 添加一个新 Model
1. 在 `models/` 下创建新文件，定义继承自 `BaseModel` 的类。
2. 实现 `train`, `save`, `load` 接口。
3. 在 `src/steps/` 中创建一个包装该模型的 Step 类（或复用现有的 `ModelStep`），并在其中调用模型接口。
