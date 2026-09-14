> 历史 N0 标注手册：2026-09-14 起按用户指示取消人工标注工作与前置门槛。当前执行使用 dataset-proxy 标签和自动结构审计，本文件保留作历史语义分类参考。

# 标注手册：VoiceMem 判断合并与更新一致性

版本：n0-1。日期：2026-09-14。

本手册用于两位独立标注者对 VoiceMem 右脑判断合并测试集进行标注。标注者不看相似度分数、策略输出或另一位标注者的标签。先用 30 个校准案例统一规范，再锁定手册后标注正式开发集。

## 1. 标注目标

每对 claim（A 和 B）需要判断：
1. A 和 B 的**逻辑关系**是什么？
2. 对 B 的正确**存储操作**是什么？
3. A 和 B 是否关于**同一对象、同一时间/情境范围**？

## 2. 关系标签（6 类）

| 标签 | 定义 | 示例 |
|---|---|---|
| `equivalent` | 同对象、同时间/情境、同立场的等价表达。可复用判断并追加证据。 | "喜欢手冲咖啡" ↔ "偏好手冲咖啡" |
| `contradiction_same_scope` | 相同范围内相反立场。不得作为同一判断的支持证据合并。 | "喜欢手冲咖啡" ↔ "讨厌手冲咖啡" |
| `temporal_update` | 明确偏好变化。保留历史，当前判断采用新状态。 | "喜欢手冲咖啡" ↔ "改喝茶了，现在偏好龙井" |
| `contextual_difference` | 适用对象、时间或情境不同。保留或细化条件。 | "工作时喜欢安静" ↔ "聚会时喜欢热闹" |
| `unrelated` | 无关判断。分开存储。 | "喜欢手冲咖啡" ↔ "讨厌被打断" |
| `ambiguous` | 无法确定。不进入主要二元指标，单独报告数量。 | — |

### 关键区分

- **equivalent vs contextual_difference**：等价要求同对象+同时间+同情境。如果只是"类似但不完全相同"，标 contextual_difference 而非 equivalent。
- **contradiction_same_scope vs temporal_update**：矛盾是"同时存在相反立场"；时间更新是"先后改变"。如果 B 明确取代 A（有时间标记或"现在改了"），标 temporal_update。
- **"X rather than Y" ↔ "Y rather than X"**：这是相同范围内的矛盾（词序颠倒=相反立场），不是等价。

## 3. 存储操作标签（5 类）

| 标签 | 定义 | 适用关系 |
|---|---|---|
| `merge_support` | 合并为同一判断，B 的证据追加到 A | equivalent |
| `add_separate` | 分开存储为新判断 | contradiction_same_scope, unrelated, contextual_difference |
| `supersede_current_keep_history` | 保留 A 为历史，B 作为当前判断 | temporal_update |
| `retain_contextual_exception` | 保留或细化条件，不合并 | contextual_difference |
| `undetermined` | 无法确定 | ambiguous |

## 4. 必须逐条回答的问题

1. **原文与规范化文本是否表达相同意思？** 若 normalize 删掉了条件或否定，定位到具体片段并记录。
   - 检查：`normalized_claim_a` 和 `raw_claim_a` 是否语义一致？如果 normalize 删了"不""现在""有时"等，标记 `normalization_damage: true`。
2. **新旧判断是否同对象、同时间/情境范围？** "现在不想听建议"不自动等于永久性偏好改变。
   - 填写 `object`（判断关于什么）和 `time_range`（时间范围）。
3. **新证据是在支持、反驳、修正，还是增加例外？** 数据库 ID 连接正确不等于语义支持正确。
   - 填写 `condition`（情境条件）。
4. **来自 preference_updates 的标签是否足以判定有效状态？** 不充分的样本标 `ambiguous`，不能为达到样本目标硬判。

## 5. 标注流程

1. **不看**：相似度分数、策略输出（merged/ID）、系统答案、另一位标注者的标签。
2. 先读 `raw_claim_a` 和 `raw_claim_b`，判断关系。
3. 再看 `normalized_claim_a` 和 `normalized_claim_b`，检查规范化是否改变了语义。
4. 填写 `relation`、`expected_operation`、`object`、`time_range`、`condition`、`confidence`。
5. `confidence`：`high`（确定）/ `medium`（较确定）/ `low`（不确定，倾向 ambiguous）。

## 6. 校准阶段

- 先标注 30 个校准案例（`annotation_calibration.jsonl`）。
- 两位标注者独立完成。
- 比较一致率，讨论分歧，修订手册定义。
- 锁定手册后，标注正式开发集（270 对）。
- 报告原始一致率和 Cohen's kappa。

## 7. 分歧裁决

- 两位标注者关系标签不同 → 第三人裁决。
- 记录裁决理由（`adjudication_reason`）。
- 未裁决的不混入主标签。

## 8. 注意事项

- 数据集原始标签（`dataset_relation_gold`）是数据集给定的，**仅供参考**，标注者应独立判断。
- PersonaMem 的 preference_updates 可能包含"X rather than Y" ↔ "Y rather than X"模式——这是矛盾，不是等价。
- OCNLI 的 contradiction 对是自然语言推理矛盾，可能不完全对应本实验的"同范围矛盾"——需判断是否同对象。
- PAWS-X 的 paraphrase 对是通用句对，与人格短语有域差异。
- 不为达到样本目标硬判 ambiguous 为确定标签。
