# Benchmark 扩展：失效记忆与偏好更新

日期：2026-09-11。状态：计划补充，尚未下载完整数据或运行评测。

本次仅补 benchmark 和测量协议，不增加遗忘算法、不修改原策略定义。第一批 600 对判断测试保持原样；以下数据用于后续存储重放与问答验证。

## 1. 测量对象及分工

| 数据 | 优先级与用途 | 可以支持的结论 | 不支持的结论 |
|---|---|---|---|
| PersonaMem-v2 | 核心，沿用原计划 | 偏好变化后是否更新判断与回答 | 完整的情感遗忘机制已有效 |
| PersonaMem-v1 | 可选补充 | 最新偏好、历史变化及变化原因的理解 | 原始内容已物理删除 |
| Memora weekly | 新增首选扩展 | 回答是否包含有效信息、避开失效信息 | 情绪状态随时间自然消退；数据库已删除旧内容 |
| LongMemEval cleaned | 外部验证，沿用原计划 | 通用知识更新、偏好与时间推理 | 情感专用筛选或遗忘能力 |
| MemoryAgentBench / FactConsolidation | 后续可选，本批不跑 | 新旧事实冲突下是否采用新版本 | 被取代内容在所有存储层不可恢复 |
| OCNLI / PAWS-X | 原句对诊断 | 矛盾或非等价表达是否被合并 | 自然交互中的整体遗忘效果 |

“回答不使用旧况”“停止默认检索旧况”“物理删除原文与派生记忆”分别记录。以上 benchmark 主要检验前一种，后两种需要本项目的数据库和上下文快照审计。

## 2. 新增 Memora：已核验的来源

官方仓库含对话、问题、评测脚本，按 weekly、monthly、quarterly 组织。每个人物每个 weekly 分组有 15 道问题，覆盖 remembering、reasoning、recommending。当前数据说明标注 Apache 2.0；实施时锁定 commit 和许可文件 hash。

- [官方仓库](https://github.com/geniesinc/Memora)
- [数据规范](https://github.com/geniesinc/Memora/blob/main/data/README.md)
- [评测规范](https://github.com/geniesinc/Memora/blob/main/evals/README.md)
- [论文](https://arxiv.org/abs/2604.20006)

文件入口：`data/weekly/<persona>/conversations/session_NNNN.json` 与同目录 `evaluation_questions_<persona>.json`。下载时核对实际字段和样本数，不仅凭 README 声称可运行。

## 3. 信息隔离及适配

给待测系统的输入仅包括对话文本、说话人和当时可见的时间。`operation`、`operation_details`、`share_memory`、`memory_evidence`、`forgetting_evidence` 及评分标准仅供离线审计，不作为抽取或遗忘提示。来自数据生成器的 ground truth 不等于用户当时明确告诉系统的信息。

按会话实际时间及会话序号重放，问某题时只开放 question_date 及之前的历史。同日先后不明确时核查原始会话/问题生成规则；无法确定则标记并排除时间因果主分析，保留排除清单。预先检查是否需要每题历史快照，不能默认先灌完整时期数据再答所有题。

默认 ingest 完整合格历史，而不是按 gold 只保留更新句。若预算迫使只重放证据会话，结果命名为 `evidence_only_diagnostic`，单独呈现，不能称为官方完整 benchmark 分数。共享历史只在同一 persona、同一策略、同一合法时间快照内复用。

不向系统透露未来问题来决定应忘什么。跨多个问题复用同一用户状态时，评测问答不得写回记忆；检索次数若影响状态，也应对每题从同一保存快照恢复，防止题目顺序改变后续结果。

## 4. 首轮抽样与预算

新增上限 30 题：从 10 个官方 persona 中，用固定种子选 4 个，随机分为 2 个开发 persona、2 个锁定测试 persona，每人最多 8 题，最终总数不超过 30。开发最多 14、测试最多 16。每个集合优先覆盖 remembering/recommending 且包含 forgetting_absence 的问题，再以不含遗忘条件的问题作保留能力对照；每一步的候选数和选择 ID 均写入 manifest。不存在足够合格题时如实缩小，不自造问题补齐。

同一 persona 的 weekly/monthly/quarterly 不得跨开发与测试。本批只运行 weekly，不全量扩展月份或季度。只有两个测试 persona，结果属于可行性 pilot，不能凭多道相关问题制造大量独立样本；同时给逐 persona 原始分数，不据此作强显著性结论。

Memora 30 题是额外的“样本上限”，不是额外的“费用额度”。与原计划共享累计 2,000 次 API 调用、5M 输入输出 token 和已配置货币上限，任一先到停止。先选少量会话及开发问题估算写入和逐条件判分开销；问答只有 30 题不代表写入成本低。预算不足时优先完成原核心实验，记录 Memora 未执行原因。

## 5. FAMA：必须按官方极性计算

每条评分子问题的 judge 输出需与 `expected_answer` 比较。`memory_presence` 通常期望 yes，`forgetting_absence` 通常期望 no；不能把所有 yes 计为通过。保留原始 judge 输出、expected_answer、匹配结果及错误状态。

每道主问题分别计算：

```text
MPA = 正确的 memory_presence 条件数 / 该类条件总数
FAA = 正确的 forgetting_absence 条件数 / 该类条件总数
lambda = N_forget / (N_presence + N_forget)
FAMA = max(0, MPA - lambda * (1 - FAA))
```

每题 FAMA 再按任务/时期取均值并乘 100，不能先池化全部子条件再算一个 FAMA。没有遗忘条件时 lambda=0，FAMA=MPA；FAA 报 N/A。这些题不计入“避免旧记忆使用”的分母。若发现没有 presence 条件的异常结构，先核验锁定版本的官方处理；不能自行除零或赋默认满分。

FAMA 衡量回答，不能证明存储级删除。同时报告 MPA、FAA、有效条件数、无遗忘条件题数，以及主计划的旧判断暴露率和存储快照。这样避免“全部不回答”在单看 FAA 时看似很好。

官方默认用三位 judge 投票。受预算限制的 pilot 可用一位已登记的固定 judge，人工复核全部策略分歧和至少 20% 一致样本；必须标为 `single_judge_pilot`，不与官方三评审分数直接比较。判分超时/解析失败标 error 并报告覆盖率，禁止静默只算成功问题。

原标准可能把历史信息的任何提及计为不合格。严格保留官方判分，另做诊断标签区分“误用旧况”与“明确说明这是历史”。不事后改 rubric 来提高 FAMA。

## 6. 其他 benchmark 的使用边界

[PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) 的 preference_updates 可用于选择偏好变化案例；原始 preference 与 VoiceMem 抽取 claim 的结果分开。它主要是合成英文数据，需要人工审核，不能直接当自然用户行为频率。

[PersonaMem-v1](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) 用于历史保留和最新偏好核对；[LongMemEval](https://github.com/xiaowu0162/LongMemEval) 用于外部更新验证。它们沿用主计划预算与划分，不增加样本额度。

[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) 的 FactConsolidation [提示模板](https://github.com/HUST-AI-HYZ/MemoryAgentBench/blob/main/utils/templates.py) 明确以编号大小决定新旧；若以后接入，应称为冲突解决/新版本采用测试。当前仅列为候选，不新增执行任务或下载需求。

这些数据没有充分标注“一次短期情绪多久后应不再影响回复”。若以后研究情绪消退，需要另立标注协议和测评，不把该方向成绩混进本轮偏好更新结果。

## 7. 远端产物与验收

新增产物：`memora_manifest.jsonl`、`memora_rubric_results.jsonl`、`memora_metrics.json`，以及报告中的独立 Memora 小节。当前没有配套执行器，需按本说明实现。

manifest 至少记录 source_commit、persona、period、split、question_id、task、question_date、可见会话清单、presence/forgetting 条件数、许可/hash、筛选原因。评分文件记录主问题与每个条件、策略、原答案、judge 版本、raw_vote、expected_answer、是否通过、token 和错误。

验收要求：

- 代码/data/许可版本已固定；数据和评分条件齐全。
- 历史时间合法，写入不使用 gold 操作、gold 证据或未来问题。
- 开发/测试 persona 隔离；裁剪或单 judge 的结果有明确标记。
- expected_answer 极性、逐题 FAMA、无 forgetting 条件处理经手工小例校验。
- 未用回答分数声称物理删除，未用偏好更新声称情绪遗忘。
- API 开销计入原总预算，失败/未执行阶段有记录。
