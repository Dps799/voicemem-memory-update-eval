# 远端集群执行交接

本包是待实现的测评规范，不含已可执行的评测脚本，也没有已完成的实验结果。先阅读 [实验计划](EXPERIMENT_PLAN.md)。

## 执行任务顺序

1. 在目标 Git 仓库中新建实验分支和隔离环境；读取该仓库 AGENTS.md。固定 VoiceMem commit，不改主工作树。
2. 核查 `TraitStore.add`、规范化、实际 embed 注入路径、heartnote 清理、trait 检索路径。记录与计划观察的差异。
3. 下载所选数据的小样本，核验字段、数据许可和版本。正式采样前产出每来源候选总数、去重和筛选规则。
4. 实现阶段 A 最小执行器，使用原类、独立数据库、真实编码回调。测试同义、相反、不同用户、不同类别、空输入、两种顺序和阈值边界。
5. 完成标注与分组划分后跑 B0/B1/B2/B3/C1。M1 与收费阶段先等待必要的模型配置及本计划阶段门槛满足。
6. 根据存储级结果实现缓存抽取重放，再做问答影响测试；不要直接全量跑长历史。
7. 汇总全体样本与失败原因，提交脚本、配置和报告。不得把收费模型调用失败当成策略正确拒绝合并。

## 执行前必须返回的 preflight 字段

```json
{
  "status": "pending",
  "upstream_commit": null,
  "experiment_commit": null,
  "embedding_model": null,
  "embedding_revision": null,
  "actual_trait_encoding_prefix": null,
  "code_path_verified": false,
  "dataset_manifests": [],
  "hardware": {},
  "scheduler": null,
  "llm_roles": {"extractor": null, "relation_judge": null, "answerer": null},
  "budget_currency": null,
  "budget_amount": null
}
```

null 表示执行者需要测量或配置，不可写成已验证。无 GPU、无 API 不阻止阶段 A。使用集群现有推理服务时仍记录真实模型与 tokenizer revision。

## 建议实现的产物结构

下面是实施目标，脚本尚未创建：

```text
experiments/voicemem-memory-update/
  README.md
  EXPERIMENT_PLAN.md
  CLUSTER_HANDOFF.md
  plan_config.json
  src/                 # 获取、采样、原类适配、重放、统计
  tests/               # 实际 ID/证据归属、隔离与指标分母
  manifests/           # 数据版本、样本 ID、划分
  reports/             # 可发布结果与失败分析
```

原始数据与运行产物路径通过环境配置，默认不放进 Git。每个 sample/policy/direction 用独立数据库；缓存向量按模型 revision、前缀、精度、规范化文本 hash 命名。重放缓存还需包含抽取器版本与提示 hash。

## 结果记录约定

`pair_results.jsonl` 每行至少有 sample_id、group_id、source、split、language、relation_gold、policy、direction、raw_claims、normalized_claims、slot、embedding_hash、similarity、returned_ids、merged、evidence_links、latency_ms、error。独立评测器计算指标，不能由 runner 写死成功标签。

`replay_results.jsonl` 记录事件时间、完成时间、原始输入、缓存抽取 ID、各层快照引用、检索内容、实际模型上下文、答案、盲评标签、token、费用、错误与重试。oracle 干预必须单独标记，不混入普通方法成绩。

## 验收清单

- [ ] 固定 commit 和数据 revision；没有仅用 main/latest 描述实验。
- [ ] 双人标注分歧已裁决，用户及原句组不跨开发/测试集合。
- [ ] 原实现与适配层对同一样本返回一致 ID 和证据链接行为。
- [ ] 原始文本和规范化文本均保留，能定位丢否定、丢情境等预处理问题。
- [ ] 各来源分别报告分母、误合并率、同义召回和置信区间。
- [ ] 强制同类别压力测试与真实抽取路径结果分开。
- [ ] 不把历史保留误报成错误，检查的是“旧状态是否被当成当前状态”。
- [ ] 费用、失败和排除样本均可追溯；未执行阶段有状态说明。
- [ ] 报告指出是否仅改阈值就足够，及哪些假设被数据否定。

首次 Git 提交建议标题：`docs: plan VoiceMem trait merge and update evaluation`。推送目标由用户提供的仓库地址决定；本地计划完成不代表已经提交或启动集群。
