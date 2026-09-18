# 运行产物

从 2026-09-17 起，新主工作流产物位于 `runs/<run_id>/`，manifest、各步骤产物和
`figure/` 都在该目录中。不再建立旧 log/output 路径链接；审计和绘图脚本直接
解析规范路径。历史 log/output 已移至 `legacy/log`、`legacy/output`，通过路径解析器定位；
历史 manifest 内的绝对路径不重写，部分旧结果复用需显式迁移。

活动生成实验位于 `runs/generation/<campaign>/`。各生成 CLI 支持
`--campaign <name>` 或显式 `runs/` 下路径；已存在的历史 campaign 路径仍被接受。
准备阶段使用 `--output <name>/prepared`。新建的多层路径必须位于项目 `runs/` 下。

Slurm 提交脚本的默认标准输出保存在 `runs/slurm/`；新检出项目先执行
`mkdir -p runs/slurm`，Slurm 会在作业脚本运行前打开日志文件。

本目录只跟踪 README 和忽略规则；模型权重、样本、数据、指标和作业日志不进入 Git。
基于历史环境的脚本和冻结的归档不批量改写。详见
[结构迁移说明](../docs/layout_refactor_20260917.md)。
