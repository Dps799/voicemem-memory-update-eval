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
7. **阶段 B 验证 RQ2 机制风险**：即使不合并（B1），旧 trait 在 10/10 案例中仍出现在检索 top-5（rank 2）；B0 下 3/10 偏好更新被合并，检索新偏好时旧 claim 为 top-1（stale_risk）。存储层残留 ≠ 回答必然采用旧偏好，回答层影响需阶段 C。

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

## 6. 逐项核实（回应评审反馈）

### Q1：455 条结果对应多少独立句对？

**455 条运行记录 ≠ 455 个独立样本。** 实际只有 **45 个独立句对**：

| 维度 | 数量 | 说明 |
|---|---|---|
| 独立句对总数 | 45 | 含 3 个控制项（跨用户/跨slot/空输入）|
| 同 user+同 slot+非空（eligible） | 42 | 主指标分母来源 |
| 非等价（FMR 分母） | **21** | contradiction 10 + nonparaphrase 10 + boundary_nomerge 1 |
| 等价（EMR 分母） | **11** | PAWS paraphrase 10 + boundary_merge 1 |
| 偏好更新（temporal） | **10** | 全部来自同一 persona（见 Q4） |

每条指标的正确读法：B0 FMR = 9 次误合并 / 21 个非等价独立句对 = 42.86%，**不是** 9/455。455 条是 45 对 × 5 策略 × 2 方向（+C1）的运行记录。

### Q2：实际用了哪些数据？

**所有指标来自 `dev_schema_check` split（小样本 schema 检查），不是正式 600 对测试集。**

| 来源 | 句对数 | split | 抽样方式 | 数据集 |
|---|---|---|---|---|
| OCNLI | 10 | dev_schema_check | contradiction 随机抽 | dev.json (公开标签) |
| PAWS-X zh | 10+10 | dev_schema_check | paraphrase/non-paraphrase 随机抽 | validation parquet |
| PersonaMem-v2 | 10 | dev_schema_check | **全部来自 persona604** | preference_updates |
| synthetic_boundary | 2 | dev_boundary | 复现代码注释中的边界对 | 人工构造 |
| synthetic_control | 3 | dev_control | 跨用户/跨slot/空输入 | 人工构造 |

**限制**：PersonaMem 的 10 个 temporal 对全部来自 persona604，不独立。temporal_merge_rate=50% 实际是"同一 persona 的 5/10 对"，不能外推到真实对话。正式运行需跨 persona 抽样。

### Q3：B1 是否真的是精确文本匹配？

**已修正**。原 B1 用 `MERGE_THRESHOLD=1.0`（余弦相似度=1.0），概念上不等于字符串精确匹配。现已改为 `B1ExactStore` 子类，重写 `_find_similar` 直接比较 `claim` 字段是否完全相同（SQL `WHERE claim=?`），不依赖余弦值。

修正后结果与之前一致（FMR=0%, EMR=18.18%, merged=2/42），原因：E5 编码是确定性的，规范化后相同的文本产生完全相同的向量，余弦=1.0000，因此 cosine-1.0 与字符串匹配在此数据上巧合等价。但概念上 B1 现在是真正的精确匹配。

### Q4：temporal 合并逐对核查

**10 个 temporal 对全部来自 persona604**。逐对核查结果：

| 类型 | 数量 | 模式 | sim 范围 | 合并? | 说明 |
|---|---|---|---|---|---|
| "X rather than Y" ↔ "Y rather than X" | 5 | 语义相反，词汇几乎相同 | 0.96~0.998 | **是** | E5 无法区分词序颠倒 |
| "likes X" ↔ "Dislikes X" | 3 | 有 "Dislikes" 前缀 | 0.91~0.93 | 否 | 前缀差异使 sim<0.95 |
| "enjoys X" ↔ "Does not enjoy X" | 2 | 有 "Does not" | 0.91~0.93 | 否 | 同上 |

**关键发现**：5 个被合并的 temporal 对不是"偏好微调"，而是**语义完全相反**的偏好（"喜欢安静博物馆而不是喧闹节庆" ↔ "喜欢喧闹节庆而不是安静博物馆"）。E5 对词序颠倒的高重叠句对给出 sim>0.96，导致相反偏好被合并为同一判断——这比单纯的"误合并"更严重，因为合并后存储的是旧偏好，新偏好丢失。

**未人工判错**：relation_gold="temporal_update" 来自 PersonaMem 数据集的 preference_updates 字段，其 old/new 标注是数据集给定的。未做双人独立标注（当前为单人探索性诊断）。

## 7. 阶段 B：固定抽取结果的存储与检索检查

跳过真实抽取器，把 PersonaMem 的 10 个偏好更新（跨 7 个 persona）按时间写入 TraitStore，检查存储与检索行为。使用 B0 和 B1 两个策略。CPU 运行。

### 7.1 存储检查

| 指标 | B0 (0.95) | B1 (字符串匹配) |
|---|---|---|
| 合并（old==new ID） | 3/10 (30%) | 0/10 (0%) |
| 新偏好保存为新 trait | 7/10 | 10/10 |
| 证据错挂 | 0/10 | 0/10 |

3 个合并案例：
- "Admires international success stories" ↔ "Admires local success stories" (sim=0.954) → 合并，存为旧 claim
- "Favors European fashion brands" ↔ "Favors non-European fashion brands" (sim=0.952) → 合并，存为旧 claim
- "Values high academic achievement" ↔ "Values personal growth over academic achievement" (sim=0.956) → 合并，存为旧 claim

### 7.2 检索检查（RQ2 机制风险）

| 指标 | B0 (0.95) | B1 (字符串匹配) |
|---|---|---|
| 旧 trait 在检索新偏好时仍可返回（top-5） | **10/10** | **10/10** |
| 检索新偏好时旧 claim 为 top-1（stale_risk） | **3/10** | 0/10 |
| 新 claim 为 top-1 | 7/10（未合并时） | 10/10 |

**RQ2 机制风险成立**：
1. **合并导致 stale**：B0 下 3/10 偏好更新被合并，检索新偏好时返回**旧 claim** 作为 top-1——系统会将旧偏好当作当前状态呈现。
2. **即使不合并，旧 trait 仍残留**：B1 下 0/10 合并，但 10/10 旧 trait 仍在检索 top-5 中（rank 2）。如果 top-k ≥ 2，旧偏好可能进入回复模型的上下文。
3. **trait 层无旧况标记**（代码核查已确认）：TraitStore 没有 stale/outdated 字段，无法区分"已过时的判断"和"当前判断"——与 heartnote 层不同。

**不能得出的结论**：以上是存储层机制风险。旧 trait 是否**实际影响最终回答**需要阶段 C（回复模型 + API 预算，当前 blocked）。存储层残留 ≠ 回答必然采用旧偏好。

## 7. 未执行阶段状态

| 阶段 | 状态 | 原因 |
|---|---|---|
| 阶段 B 时间重放 | blocked | 需固定抽取器输出 + 回复模型配置 |
| 阶段 C 问答影响 | blocked | 需回复模型 + API 预算 |
| M1 关系门控 | blocked | 需关系判断器模型 |
| Memora 扩展 | blocked | 阶段 B/C 后可选 |

## 8. 阶段门槛判定

根据计划 §8：
- **数据/schema/编码路径一致**：✓
- **H1 有反例**：✓ — B0 FMR=42.86%（9/21），误合并主要来自 E5 对高词汇重叠中文句对相似度过高
- **H1 根因**：7/9 误合并来自 PAWS-X 非同义高重叠（sim 0.96~0.99）；1/9 OCNLI contradiction（0.963）；1/9 边界对（0.959）
- **是否仅改阈值足够**：✗ — 0.97 仍 23.81% FMR；0.99 降至 9.52% 但 EMR 跌至 36.36%（远低于 B0-5pp）。M1 值得作为候选对照检验
- **RQ2 机制风险**：✓ 成立（阶段 B 已验证存储层）— 但回答层影响需阶段 C

## 9. 未执行阶段状态

| 阶段 | 状态 | 原因 |
|---|---|---|
| 阶段 C 问答影响 | blocked | 需回复模型 + API 预算；当前仅验证存储层，未验证回答层 |
| M1 关系门控 | blocked | 需关系判断器模型；应作为候选对照，非唯一解 |
| Memora 扩展 | blocked | 阶段 B/C 后可选 |

## 10. 验收清单

- [x] 固定 commit 和数据 revision（VoiceMem a450911, E5 614241f6, OCNLI b53efde, PAWS-X 4cd8187c, PersonaMem ed956de）
- [x] 原实现与适配层对同一样本返回一致 ID 和证据链接行为（直接用原类，无适配层）
- [x] 原始文本和规范化文本均保留（pair_results.jsonl 含 raw + normalized claim）
- [x] 各来源分别报告分母（metrics.json 含 by_source_b0 / by_relation_b0）
- [x] 强制同类别压力测试与真实抽取路径结果分开（控制维度独立标记，通用句对为压力测试）
- [x] 不把历史保留误报成错误（temporal_merge_rate 单独报告，阶段 B 区分合并与残留）
- [x] 强制同类别压力测试与真实抽取路径结果分开（通用句对为压力测试；PersonaMem 为固定抽取路径）
- [x] 费用、失败和排除样本可追溯（pair_results.jsonl + replay_results.jsonl 含 error/latency）
- [x] 未执行阶段有状态说明（§9）
- [x] 不把收费模型调用失败当成策略正确拒绝合并（阶段 A/B 无 LLM 调用）
- [x] 报告指出是否仅改阈值就足够（否，0.99 代价过大，M1 作为候选对照）
- [x] B1 使用真正的字符串精确匹配（B1ExactStore，非 cosine=1.0）
- [x] 每项指标给出错误数/有效样本数（§6 Q1：45 对不是 455 条）
- [x] 明确数据来源和 split（§6 Q2：全部 dev_schema_check，非正式测试集）
- [x] temporal 合并逐对核查（§6 Q4：5/10 为词序颠倒的相反偏好，全来自 1 persona）
- [x] 阶段 B 存储与检索检查已完成（§7：stale_risk 3/10，旧 trait 残留 10/10）
- [ ] 双人标注分歧已裁决（当前为单人探索性诊断）
- [ ] 误合并率/同义召回的置信区间足够窄（smoke test n=21，需扩样至 600 对）
- [ ] 阶段 C 回答层验证（blocked，需回复模型）
- [x] Memora gold 操作未进入写入策略；判分按 expected_answer 比较（Memora 未执行，无违规风险）

## 11. 复现

```bash
cd /quark_speech_nas_zjk/users/wangyuhan/experiments/voicemem-memory-update-eval
python3 src/build_pairs.py        # 构建 45 对句对
python3 src/run_phase_a.py        # 阶段 A：B0/B1/B2/B3/C1（B1 为字符串精确匹配）
python3 src/aggregate_metrics.py  # 聚合阶段 A 指标
python3 src/run_phase_b.py        # 阶段 B：固定抽取的存储与检索检查
```

种子：20260911。模型权重冻结。所有数字结果来自实际运行。
