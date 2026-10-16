# zhouslab F4 生成环境

项目位置 `/home/avis/pslg-nilm`，Python 3.12。独立环境为 `.venv-generation-torch`、`.venv-generation-tf`、`.venv-generation-eval`。核心依赖分别见 requirements 文件，实际解析版本见 freeze 文件；pip-check 文件记录依赖检查结果。

运行要求必须遵守 `/home/avis/post_training_rep/guide.md`。本服务器没有 Slurm。入口 `scripts/run_f4_zhouslab.py` 持有共享 GPU flock、连续检查空闲状态、设置 CUDA_VISIBLE_DEVICES，然后调用冻结科学代码。TF 子进程的 LD_LIBRARY_PATH 仅加入本环境 nvidia 包内的 lib 目录。

首次部署以已有 Python 3.12 创建 venv；pip 安装 requirements-torch 时须使用 PyTorch cu118 索引。当前已验证的 Torch wheel SHA-256 为 f4c3ae93c72e8b725b510c3bc5b9dd5437f6740a66c3cc3bb1a19f177c3baef4。由于网络下载慢，CUDA 包和 Triton 从原实验环境迁移；可用 scripts/import_f4_cuda_dependencies.py tf（或 torch）校验并导入保留的归档，再 pip 安装核心包和依赖。不要修改 post_training_rep 的现有环境。

CPU 任务：`.venv-generation-eval/bin/python scripts/run_f4_zhouslab.py run --stages cpu`。
GPU 任务：`.venv-generation-eval/bin/python scripts/run_f4_zhouslab.py run --stages gpu`。
生成环境验证：`.venv-generation-torch/bin/python scripts/run_f4_zhouslab.py smoke --gpu 0`。
正式后台运行加 nohup，保存日志及 PID。CPU 与 GPU 两阶段可以分别启动；不要同时启动 all 和分阶段模式。该入口仅执行本次 seed=0 的收尾任务，后续种子训练须另行配置，不能直接解除旧任务的 hold。

详细部署和校验记录见 docs/20261010_f4_generation_zhouslab_migration.md。
