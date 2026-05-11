# ExtractActiveDataStep 使用说明

本文档说明 `ExtractActiveDataStep` 的用途、输入输出契约、配置方法，以及它如何与 `DataLoaderStep` 衔接。

---

## 1. 功能概述

`ExtractActiveDataStep` 用于对单个电器的整段功率序列进行“工作区间（active segments）”切割：

- 以 `power_threshold` 判断电器是否处于工作状态
- 以 `min_duration_seconds` 过滤掉持续时间过短的噪声区间
- 以 `context_seconds` 在工作区间前后补充上下文窗口
- 将每个工作区间保存为独立的 CSV 文件（列：`timestamp,power,datetime`）

该步骤通常放在 `DataLoaderStep` 之前执行，用来把“原始大 CSV”切割为“可直接加载的若干小 CSV”。

---

## 2. 输入 (Input)

### 2.1 外部文件输入契约

该步骤通过配置项 `steps.extract_active_data.input_file` 读取一个电器的 CSV 文件：

- 推荐路径示例：`/home/scnu2023024258/data/datasets/ukdale_extracted/house1/<appliance>.csv`
- 字段要求：必须是单个电器的 `.csv` 文件路径（不是目录）
- 必需列：
  - `timestamp`：Unix 时间戳（秒）
  - `power`：功率值（float）

如果文件没有列名，该步骤会默认取前两列并重命名为 `timestamp,power`，并按 `timestamp` 升序排序后再处理。

---

## 3. 输出 (Output)

### 3.1 Log 目录产物（中间结果）

输出保存到当前步骤的缓存目录下：

- `log/{appliance_name}_{sequence_id}/ExtractActiveData/segments/*.csv`
- 或无 `appliance_name` 时：`log/{sequence_id}/ExtractActiveData/segments/*.csv`

每个区间文件命名格式：

- `{appliance_name}_{start_datetime}_{end_datetime}_{duration_seconds}s.csv`

其中 `duration_seconds = end_time - start_time`（秒）。

### 3.2 Context 传递

该步骤会写入：

```python
context["data"]["extract_active_data"] = {
  "segments_dir": ".../ExtractActiveData/segments",
  "segment_files": [".../*.csv", ".../*.csv", ...]
}
```

### 3.3 与 DataLoaderStep 的衔接

当 `set_input_root: true` 且成功切割出至少一个区间文件时：

- `context["input_root"]` 会被设置为 `segments_dir`
- 后续 `DataLoaderStep` 会从 `context["input_root"]` 读取并缓存这些 CSV

如果 `set_input_root: false`，则不会改写 `context["input_root"]`，后续 DataLoader 仍然从原 `input/` 目录读取。

---

## 4. 配置方法 (Configuration)

在 `config/config.yaml` 的 `steps.extract_active_data` 下配置：

```yaml
steps:
  extract_active_data:
    enabled: true
    input_file: "/home/scnu2023024258/data/datasets/ukdale_extracted/house1/fridge.csv"
    power_threshold: 30.0
    min_duration_seconds: 60
    context_seconds: 30
    set_input_root: true
```

字段说明：

- `enabled`：是否启用该 Step
- `input_file`：输入 CSV 文件路径（预先设置的超参数）
- `power_threshold`：工作状态判定阈值（`power >= threshold` 视为 active）
- `min_duration_seconds`：最小工作区间持续时间（秒）
- `context_seconds`：上下文补齐窗口（秒）
- `set_input_root`：是否将 `context["input_root"]` 指向切割输出目录，以便 DataLoader 直接加载切割结果

---

## 5. main.py 参数传递

`main.py` 会从 `steps.extract_active_data` 读取参数并构造 Step，且在 `DataLoaderStep` 之前加入工作流：

- [main.py](file:///home/scnu2023024258/data/code/PSLG-NILM/main.py#L36-L53)

---

## 6. 运行建议

如果你的目标是“先切割，再让后续步骤吃切割结果”，建议开启：

```yaml
steps:
  extract_active_data:
    enabled: true
    set_input_root: true
  data_loader:
    enabled: true
```

并关闭与当前实验无关的步骤，避免干扰与额外耗时。
