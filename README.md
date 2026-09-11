# VoiceMem 右脑记忆更新测评交接

状态：实验方案草案，尚未运行实验。编写日期：2026-09-11。

目标：检验右脑基于向量相似度的判断合并是否误合并不同含义的条目，以及偏好改变后旧判断是否继续影响回复。不预设 VoiceMem 一定失败。

- [实验计划](EXPERIMENT_PLAN.md)：假设、数据、对照、指标、停止条件。
- [远端执行交接](CLUSTER_HANDOFF.md)：实施顺序、产物、资源与验收。
- [计划配置](plan_config.json)：供后续执行器实现时读取的配置草案；当前没有实现配套执行器。

第一批只做代码版本核验、数据审计、少量 smoke test 与句对诊断；不训练模型，不要求部署 35B 回复模型。

GitHub 仓库：[Dps799/voicemem-memory-update-eval](https://github.com/Dps799/voicemem-memory-update-eval)，默认分支为 `main`。计划文件位于仓库根目录；集群调度器和可用模型待执行环境配置。

远端集群使用有私有仓库读取权限的 GitHub 账户克隆：

```bash
git clone git@github.com:Dps799/voicemem-memory-update-eval.git
cd voicemem-memory-update-eval
```

本次提交仅包含计划，不会自动启动集群作业。

交付顺序：先返回 `preflight.json`、`data_audit.json` 和小样本结果，再依据计划中的数据质量、资源与阶段门槛继续。任何数字结果必须来自实际运行。
