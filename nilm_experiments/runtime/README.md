# 四模型运行环境

各模型的完整教程位于 `../third_party/nilmformer/ENVIRONMENT.md`、`../FCN/ENVIRONMENT.md`、`../third_party/bert4nilm/ENVIRONMENT.md`、`../third_party/nilmtk_contrib/ENVIRONMENT.md`。

共享基础镜像固定 Python 基础镜像摘要、PyTorch 2.5.1 和 CUDA 11.8。镜像提供环境，模型代码从整个 nilm_experiments 目录挂载，不打包数据、凭据、旧环境或检查点。四个模型镜像共享基础层。

Conda 环境名：nilmformer-cu118-v1、fcn-cu118-v1、bert4nilm-cu118-v1、sgn-cu118-v1。首次批量安装先创建 nilmformer-cu118-v1 的 Python 3.12/pip 环境，再运行 `bash runtime/setup_conda.sh`。安装日志为 setup-conda.log。

实际测试结果和镜像文件清单见 STATUS.md。首次使用 GPU 时按集群规则通过 Slurm 分配 GPU；不在登录节点启动真实训练。


发现阶段的 TensorFlow 配方与安装入口见 `tensorflow/requirements-tf.txt` 和 `tensorflow/setup_tf.slurm`。
从项目根目录执行 `mkdir -p runs/slurm`，再 `sbatch nilm_experiments/runtime/tensorflow/setup_tf.slurm`。
