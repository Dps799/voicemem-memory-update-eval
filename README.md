# VoiceMem 右脑记忆更新测评交接

状态：阶段 A 小样本与阶段 B 画像检索诊断已执行并完成审计修订；正式测试与回答层实验待执行。最新结果见 [审计报告](reports/REPORT.md)。

最新进展：[自动开发实验报告](reports/DEVELOPMENT_REPORT.md)。按用户指示不再等待人工标注；301 对 N1 开发诊断和 24 段受控 N2 已运行，299 对测试推理尚未开启。

目标：检验右脑基于向量相似度的判断合并是否误合并不同含义的条目，以及偏好改变后旧判断是否继续影响回复。不预设 VoiceMem 一定失败。

- [下一阶段详细计划](NEXT_PHASE_PLAN.md)：600 对复核、时序检索、持久/上下文预算、问答门槛与集群任务。
- [实验计划](EXPERIMENT_PLAN.md)：假设、数据、对照、指标、停止条件。
- [Benchmark 扩展说明](BENCHMARK_ADDENDUM.md)：Memora 数据接入、FAMA 判分和其他更新评测的适用边界。
- [远端执行交接](CLUSTER_HANDOFF.md)：实施顺序、产物、资源与验收。
- [计划配置](plan_config.json)：正式实验规划与当前执行状态；smoke 运行使用已提交 manifest，尚未实现完整配置驱动的正式执行器。

第一批只做代码版本核验、数据审计、少量 smoke test 与句对诊断；不训练模型，不要求部署 35B 回复模型。

GitHub 仓库：[Dps799/voicemem-memory-update-eval](https://github.com/Dps799/voicemem-memory-update-eval)，默认分支为 `main`。计划文件位于仓库根目录；集群调度器和可用模型待执行环境配置。

远端集群使用有私有仓库读取权限的 GitHub 账户克隆：

```bash
git clone git@github.com:Dps799/voicemem-memory-update-eval.git
cd voicemem-memory-update-eval
```

仓库包含 CPU smoke 执行器与真实运行结果；推送代码不会自动启动集群作业。

交付顺序：先返回 `preflight.json`、`data_audit.json` 和小样本结果，再依据计划中的数据质量、资源与阶段门槛继续。任何数字结果必须来自实际运行。

## 重跑已审计的 CPU 小样本

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python bash scripts/reproduce_smoke.sh
```

默认匿名下载固定 revision 的 multilingual-e5-small，强制使用 CPU，不调用生成模型。使用已提交的两个 manifest，不依赖原集群的绝对数据路径；`src/build_pairs.py` 是原数据审计构建脚本，重新抽样时须另行配置数据源。

脚本依次运行回归测试、阶段 A、指标聚合、阶段 B、报告生成与产物一致性检查；结果覆盖 `reports/` 当前版本。旧 `3302a96` 产物已归档至 `reports/archive/3302a96/`，保留审计追溯。

阶段 B 仅调用上游画像检索过滤、配额与渲染函数，使用构造证据、自然问题及背景记忆；不是完整 VoiceMem 或正式 benchmark。正式测试、完整回答层、M1 和 Memora 尚未执行；人工标注不再是任务或前置条件。

自动开发阶段复现：`PYTHON=.venv/bin/python bash scripts/reproduce_development.sh`。它使用 dataset-proxy 标签与自动结构审计，只运行开发集；N2 是机械主题探针与合成背景的受控实验，不能冒充自然对话成绩。
