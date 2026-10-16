# 第三方代码来源

本目录集中保存第三方代码，保留各仓库 Git 元数据、许可和原有本地改动。

| 目录 | 上游/版本 | 本项目用途 |
|---|---|---|
| nilmformer | https://github.com/adrienpetralia/NILMFormer ，锁定 commit 见 ../sources.lock.json | NILMFormer 与 FCN 共用的源码。 |
| bert4nilm | https://github.com/Yueeeeeeee/BERT4NILM | BERTAdapter 的基础模型。 |
| nilmtk_contrib | https://github.com/nilmtk/nilmtk-contrib | SGN 及其他模型的社区参考库。 |
| diffusion_ts | https://github.com/Y-debug-sys/Diffusion-TS ，组件记录 commit 566307e | 条件生成模型使用的 Transformer。 |

版本详情见 [来源锁定文件](../sources.lock.json)；Diffusion-TS 的记录见自身
__init__.py。生成 trial 的代码哈希进一步确定实际使用的源码。

旧模型路径的兼容链接已移除。适配器、源码指纹和构建脚本直接使用本目录。
FCN 独立环境配置和历史产物仍在 FCN/；其模型源码仅在 nilmformer/src 中。
项目自编的适配器与生成骨干留在 nilm_lab/、generation_lab/。
