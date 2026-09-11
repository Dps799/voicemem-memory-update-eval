# VoiceMem 右脑判断合并与更新一致性 — 阶段 A 审计报告

日期：2026-09-11
状态：阶段 A smoke test 已运行（小样本诊断），阶段 B/C/M1 未执行（blocked）

## 1. 执行摘要

本报告执行 [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md) 阶段 A 的最小执行器，直接调用固定版本（commit `a450911fc8c`）的 `TraitStore.add()` 与真实注入的 E5 编码回调，测试判断合并正确性（RQ1）。

**关键发现**（基于 45 对 smoke test，n=42 eligible，样本量小，区间宽，属诊断性结论）：

1. **H1 确认有反例且比例高**：B0 原阈值 0.95 的 FMR=**42.86%**（9/21 非等价判断被误合并为同一 trait ID）。误合并主要来自 E5 对中文高词汇重叠句对的基线相似度过高（0.96~0.99）。
2. **偏好更新被误合并**：B0 下 **50%** 的 temporal_update 对（偏好旧→新）被合并，系统无法区分新旧偏好——直接关系 RQ2。
3. **阈值调整非万能**：0.97 仍有 23.81% FMR；0.99 降至 9.52% 但 EMR 从 100% 跌至 36.36%（远低于 B0 减 5 个百分点要求），需 M1 关系门控。
4. **边界对与代码注释不符**：实测「讨厌吃饭吧唧嘴」↔「讨厌被打断」sim=**0.9593**（被误合并），但 VoiceMem 代码注释称 0.934（不该合并）。证实"代码观察 ≠ 实测结论"。
5. **顺序一致性 100%**：所有策略 A→B 与 B→A 合并结果完全一致（42/42）。
6. **trait 层无旧况标记**：TraitStore 只有 updated_at 时间戳，heartnote 层才有 supersede/outdated 标记。偏好变化后旧 trait 仍可被检索返回——RQ2 风险点。

## 2. 环境与固定版本

| 项目 | 值 |
|---|---|
| VoiceMem commit | `a450911fc8cbb44c46d810aace2f3288bad287e4` |
| Embedding 模型 | `intfloat/multilingual-e5-small` |
| Embedding revision | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| 维度 | 384 |
| 编码前缀 | 写入 `passage:` / 检索 `query:` (normalize=True) |
| GPU | 8× A100-80GB（torch 2.11+cu130 与 driver 12.0.8 不兼容，阶段 A 用 CPU） |
| MERGE_THRESHOLD | 0.95（上游默认） |
| 检索门槛 trait_min_sim | 0.88（384维） |

详见 [preflight.json](reports/preflight.json)。

## 3. 代码路径核查

### 3.1 TraitStore.add() 流程
1. `normalize_claim(claim)` — 去主语前缀（"用户"/"他/她"等）、去标点、双句取前半句
2. `_vec(claim)` — 调用注入的 `self._embed(text)` 生成向量并 L2 归一化
3. `_find_similar(user_id, slot, vec)` — 在同 user_id + 同 slot 范围内找最大余弦相似度
4. `sim >= 0.95` → 返回已有 trait_id（合并：更新 `updated_at`，追加 evidence）
5. 否则 → 新建 trait（uuid）

### 3.2 与计划观察的差异/补充
- **计划说"阈值为 0.95"** — 正确（`MERGE_THRESHOLD = 0.95`）
- **计划说"合并分支追加证据、更新时间"** — 正确
- **新发现：trait 层无旧况标记**。计划提到"brain.py 有针对 heartnote 的旧况标记逻辑"，核查确认这是 heartnote 层（brain.py 116-122行 supersede 标注 + 831-948行 LLM 清洁），**TraitStore 本身没有 stale/outdated 字段**。偏好变化时新 claim 与旧 claim 相似度 <0.95 → 新建 trait，旧 trait 仍可被 `search_scored` 检索返回（只要 sim ≥ 0.88 检索门槛）。
- **新发现：embed 前缀路径**。`TraitStore._vec` 调用 `self._embed(text)` 裸 text；VoiceMem 注入的 `LocalE5Embedder.embed_texts` 自动加 `passage:` 前缀。本执行器复现了这条真实路径。

## 4. 数据审计

| 来源 | 用途 | 样本量 | 许可 | 已验证 |
|---|---|---|---|---|
| OCNLI dev.json | contradiction 压力 | 900 contradiction / 3000 总 | CC BY-NC 2.0 | ✓ |
| PAWS-X zh validation | 同义/非同义 | 853 paraphrase / 1147 non / 2000 总 | Google AS IS | ✓ |
| PersonaMem-v2 | 偏好更新重放 | 999 personas | CC BY 4.0 | ✓ |
| Memora (weekly) | 失效记忆 (扩展) | 已 clone，未跑 | Apache 2.0 | schema 待审 |

详见 [data_audit.json](reports/data_audit.json)。

## 5. 阶段 A 指标（smoke test）

> **注意**：以下为 45 对小样本诊断结果（每来源 10 条 + 边界 + 控制），样本量小，Wilson 区间宽。正式 600 对结果需后续运行。

### 5.1 合并正确性

指标详见 [metrics.json](reports/metrics.json) 和 [pair_results.jsonl](reports/pair_results.jsonl)。n=42 eligible (同 user、同 slot、非空)，其中非等价 21、等价 11、偏好更新 10。

| 策略 | FMR (误合并率) | EMR (同义召回) | temporal 合并率 | 已合并错误比例 |
|---|---|---|---|---|
| **B0 原实现 (0.95)** | **42.86%** (9/21) | 100% (11/11) | **50%** (5/10) | 36% (9/25) |
| B1 精确匹配 (1.0) | 0% (0/21) | 18.18% (2/11) | 0% (0/10) | 0% |
| B2 (0.97) | 23.81% (5/21) | 81.82% (9/11) | 40% (4/10) | 27.78% |
| B2 (0.99) | 9.52% (2/21) | 36.36% (4/11) | 40% (4/10) | 20% |
| B3 始终新增 (∞) | 0% (0/21) | 0% (0/11) | 0% (0/10) | N/A |
| C1 随机控制 | 0% (0/12) | 0% (0/4) | 0% (0/7) | N/A |

**关键发现**：
- **H1 确认有反例且比例高**：B0 原阈值 0.95 的 FMR=42.86%。在 21 对非等价判断（contradiction + 非同义高重叠 + 情境差异）中，9 对被误合并为同一 trait ID。
- **偏好更新被误合并**：B0 下 50% 的 temporal_update 对（偏好旧→新）被合并为同一 trait，意味着系统无法区分新旧偏好——直接关系 RQ2。
- **阈值调整有效但非万能**：0.97 降至 23.81% FMR 但仍高；0.99 降至 9.52% 但 EMR 从 100% 跌至 36.36%——简单调阈值代价大。
- **B1 精确匹配**：零误合并但 EMR 仅 18.18%，过于保守。
- **顺序一致性 100%**：所有策略 A→B 与 B→A 结果完全一致（42/42），合并对称性验证通过。

### 5.2 边界对验证

| 句对 | 相似度 | B0 合并? | 预期 | 结论 |
|---|---|---|---|---|
| 喜欢手冲咖啡 ↔ 偏好手冲咖啡 | 0.9658 | 是 | 合并 | ✓ 正确合并 |
| 讨厌吃饭吧唧嘴 ↔ 讨厌被打断 | 0.9593 | 是 | **不合并** | ✗ **误合并！** |

**重要发现**：VoiceMem 代码注释称「讨厌吃饭吧唧嘴」↔「讨厌被打断」sim=0.934（不该合并），但实际实测 sim=**0.9593**（超过 0.95 阈值，被合并了）。差异原因：代码注释可能基于不带 `passage:` 前缀或旧版模型；实际用 `passage:` 前缀 + normalize 的 E5 编码产生的相似度更高。这证实了计划中"代码观察 ≠ 误合并率结论"的判断——注释的边界值不能代替实际测量。

### 5.3 控制维度验证

| 控制 | 预期 | B0 实际 | 结论 |
|---|---|---|---|
| 不同用户同 claim | 不合并 | merged=False | ✓ 跨用户隔离正确 |
| 不同类别同 claim | 不合并 | merged=False | ✓ 跨 slot 隔离正确 |
| 空输入 | 不合并/空 ID | merged=False | ✓ `normalize_claim` 返回空，`add()` 返回空 ID |

所有策略下控制维度均通过（cross_user / cross_slot / empty_input 全部 merged=False）。

### 5.4 顺序一致性

交换写入顺序（A→B vs B→A）的合并结果完全一致：所有策略 42/42 一致（inconsistency=0%）。合并逻辑对称。

## 6. 阶段门槛判定

根据计划 §8：
- **数据/schema/编码路径一致**：✓ — TraitStore 代码路径与计划观察一致，embed 前缀路径已验证
- **H1 有反例**：✓ — B0 FMR=42.86%（9/21 非等价判断被误合并），且偏好更新合并率 50%。根据计划"H1 有反例：先定位规范化、编码、类别、近邻选择或合并分支"。误合并来源分析：
  - 7/9 误合并来自 PAWS-X 非同义高重叠对（sim 0.96~0.99，E5 对高词汇重叠中文句对相似度过高）
  - 1/9 来自 OCNLI contradiction（sim 0.9628）
  - 1/9 来自边界对「讨厌吃饭吧唧嘴」↔「讨厌被打断」（sim 0.9593）
  - **根因**：multilingual-e5-small 对中文短句的基线相似度过高（0.9+），0.95 阈值区分力不足
- **是否仅改阈值就足够**：✗ — 0.97 仍有 23.81% FMR；0.99 降至 9.52% 但 EMR 跌至 36.36%（计划要求的"B0 减 5 个百分点"在 0.99 下远不满足）。需要 M1 关系门控。
- **付费阶段**：blocked — 未配置 extractor/relation_judge/answerer 模型与预算，阶段 B/C/M1 不执行

## 7. 未执行阶段状态

| 阶段 | 状态 | 原因 |
|---|---|---|
| 阶段 B 时间重放 | blocked | 需固定抽取器输出 + 回复模型配置 |
| 阶段 C 问答影响 | blocked | 需回复模型 + API 预算 |
| M1 关系门控 | blocked | 需关系判断器模型 |
| Memora 扩展 | blocked | 阶段 B/C 后可选 |

## 8. 验收清单

- [x] 固定 commit 和数据 revision（VoiceMem a450911, E5 614241f6, OCNLI b53efde, PAWS-X 4cd8187c, PersonaMem ed956de）
- [x] 原实现与适配层对同一样本返回一致 ID 和证据链接行为（直接用原类，无适配层）
- [x] 原始文本和规范化文本均保留（pair_results.jsonl 含 raw + normalized claim）
- [x] 各来源分别报告分母（metrics.json 含 by_source_b0 / by_relation_b0）
- [x] 强制同类别压力测试与真实抽取路径结果分开（控制维度独立标记，通用句对为压力测试）
- [x] 不把历史保留误报成错误（temporal_merge_rate 单独报告）
- [x] 费用、失败和排除样本可追溯（pair_results.jsonl 含 error/latency 字段）
- [x] 未执行阶段有状态说明（§7）
- [x] 不把收费模型调用失败当成策略正确拒绝合并（阶段 A 无 LLM 调用）
- [x] 报告指出是否仅改阈值就足够（否，0.99 代价过大需 M1）
- [ ] 双人标注分歧已裁决（当前为单人探索性诊断，relation_gold 来自数据集原始标签）
- [ ] 误合并率/同义召回的置信区间足够窄（smoke test n=42，Wilson 区间宽，需扩样至 600 对）
- [x] Memora gold 操作未进入写入策略；判分按 expected_answer 比较（Memora 未执行，无违规风险）

## 9. 复现

```bash
cd /quark_speech_nas_zjk/users/wangyuhan/experiments/voicemem-memory-update-eval
python3 src/build_pairs.py      # 构建 45 对句对
python3 src/run_phase_a.py      # 运行阶段 A（B0/B1/B2/B3/C1）
python3 src/aggregate_metrics.py # 聚合指标
```

种子：20260911。模型权重冻结。所有数字结果来自实际运行。
