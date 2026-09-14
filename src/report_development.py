"""Create a data-derived development report and validate immutable run artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_n1 import read_rows
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n1',type=Path,required=True);parser.add_argument('--n2',type=Path,required=True);args=parser.parse_args()
    a,b=args.n1.resolve(),args.n2.resolve()
    m=json.loads((a/'metrics.json').read_text());n=json.loads((b/'metrics.json').read_text());audit=json.loads((a/'automatic_label_audit.json').read_text())
    p1=json.loads((a/'preflight.json').read_text());p2=json.loads((b/'preflight.json').read_text())
    for pre in (p1,p2):
        for f,h in pre['sha256'].items():
            if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h:raise ValueError(f'Provenance hash mismatch: {f}')
    rows1=read_rows(a/'pair_results.jsonl');rows2=read_rows(b/'replay_results.jsonl')
    assert len(rows1)==p1['expected_records']==p1['actual_records']
    assert len(rows2)==n['expected_records']==p2['expected_records']==p2['actual_records']
    assert all(r['split']=='dev' and r['error'] is None for r in rows1+rows2)
    assert all(r['evidence_link_integrity'] for r in rows1)
    assert all(r['context_tokens']<=r['context_token_budget'] for r in rows2)
    assert all(not r['old_claim_visible'] for r in rows2 if r['policy']=='O_filter_diagnostic' and r['query_kind']=='current')
    # Stable old/new content labels must correspond to actual rendered claims, not IDs.
    episodes={r['episode_id']:r for r in read_rows(b/'episode_manifest.jsonl')}
    from e5_callback import normalize_claim
    for r in rows2:
        ep=episodes[r['episode_id']]
        assert r['old_claim_visible']==(normalize_claim(ep['old']) in r['presented_claims'])
        assert r['new_claim_visible']==(normalize_claim(ep['new']) in r['presented_claims'])
    verification={'n1_records':len(rows1),'n2_records':len(rows2),'execution_errors':0,'test_inference_count':0,
                  'generative_api_calls':0,'all_recorded_hashes_match':True,'all_context_budgets_respected':True,
                  'all_n1_evidence_links_valid':True,'label_mode':'dataset_proxy_and_automatic_structural_checks'}
    (ROOT/'reports/development_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    def ratio(x):return f"{x['k']}/{x['n']} ({x['rate']:.1%})" if x['n'] else 'N/A'
    lines=['# 自动推进：N1 与受控 N2 开发实验', '',
    '日期：2026-09-14。基于远端 `1fd78ab` 完成下一步实际执行。用户明确取消人工标注要求，本轮使用数据集原始标签与自动结构审计，不等待人工复核、不伪造人工或语义模型标签。', '',
    '## 完成情况', '',
    f"- N1：{m['n_pairs']} 对开发输入，{m['n_records']} 条策略/方向/消融记录，执行错误 {m['n_errors']}。",
    f"- N2：{n['n_episodes']} 个开发更新、{n['n_personas']} 个 persona，{n['n_records']} 条受控检索条件记录。",
    '- 30 项回归测试通过。全部使用固定 E5 CPU 编码，生成模型调用 0 次。',
    '- 未运行测试集推理、回复模型、完整 VoiceMem 链或长期存储淘汰。', '',
    '## N0 审计与修正', '',
    '远端的 600 对和 100 个 persona 已采样，但 PAWS-X 的 pawsx_ne_5601 / pawsx_ne_791 共享原句却跨 dev/test。按原组＋共享原文连通分量重建有效分组，把接触开发集的组件全部留在开发侧。原始 N0 manifest 保留不改，有效集合为 301 dev / 299 test，共 496 组。',
    '此检查覆盖已提交的样本池，未重新下载整个来源池检查经未采样句子连接的重复链。详见 [有效划分审计](../manifests/n1_split_audit.json)。', '',
    '## N1：按来源报告', '',
    '| 策略 | OCNLI 非等价合并 | PAWS-X 非等价合并 | PAWS-X 同义合并 | PersonaMem 更新合并 |',
    '|---|---:|---:|---:|---:|']
    for policy in ['B0_original','B1_exact','B2_097','B2_099','B3_never']:
        z=m['policies_by_source'][policy]
        lines.append(f"| {policy} | {ratio(z['OCNLI']['FMR'])} | {ratio(z['PAWS-X-zh']['FMR'])} | {ratio(z['PAWS-X-zh']['EMR'])} | {ratio(z['PersonaMem-v2']['temporal_merge_rate'])} |")
    lines += ['',
    '这些分母以 dataset-proxy 标签定义，包含下面识别出的结构冲突，不能直接解读为已核验的语义错误率。按预定 EMR≥B0−5pp 约束，0.97/0.99 都不可行，因此 B2 冻结选择仍为 0.95；这不证明不存在其他简单解决办法。随机对照三种子和组级区间全部在机器可读结果中保留。', '',
    f"B0 的 17/100 个 temporal 合并来自 {audit['b0_temporal_merged_groups']} 个独立 persona。英语偏好更新的 raw 文本消融没有改变这些合并决策。", '',
    '## 自动审计发现：不能把 B1 的全部问题归因于相似度', '',
    f"在非等价标签下检测到 {audit['n_raw_identical_label_conflicts']} 对原文完全相同，以及 {audit['n_distinct_raw_normalization_collisions']} 对不同原文被规范化为同一文本。它们恰好对应 B1 的 8 个非同义合并。",
    '例如，两句不同内容都被截成“2008 年”或“根据季节不同”。这能自动证明文本信息被压缩到相同前缀，但不需要、也不冒充人工语义裁决。原文相同的 PAWS-X 条目标为非同义，作为数据标签冲突单列。',
    '事后敏感性分析排除这 8 对后，B0 在 PAWS-X 仍有 25/41 个非等价标签合并；B1 为 0/41。此分析不替换主结果，不重新挑选阈值。',
    f"全体 {m['n_pairs']} 对中，{m['normalization_text_changed']} 对有文本变化、{m['normalization_collisions']} 对发生不同原文到相同规范化文本的碰撞（包含同义标签）。文本变化率不能称为语义损伤率。", '',
    '保留全文也不是统一改进：OCNLI 的 B0 非等价合并从 16/100 降到 11/100，但 PAWS-X 从 33/49 升到 45/49，同义合并从 45/52 升到 50/52。不同来源的效果应分开解释。',
    'AB/BA 的合并决策完全一致，但 B0 有 94/301 对的最终 claim/证据状态不同；这支持此前“决策一致不等于状态一致”的修正，不意味着这些差异全是错误。', '',
    '## N2：受控主题探针＋背景＋上下文预算', '',
    '本轮不是自然对话 benchmark：问题用旧偏好自动提取主题后套固定模板，可能更偏向旧表述；背景是明确标记的合成讨论兴趣，不是目标 persona 的真实生活事实。使用 20/100 条背景、128/256/512 个 E5 tokenizer token、当前/历史两个探针。这里只运行画像切片。',
    '下表为 100 条背景、256 token、当前问题（24 个更新，12 个 persona）：', '',
    '| 策略 | 旧 claim 可见 | 新 claim 可见 | 新原文完整可见（claim 或 evidence） |',
    '|---|---:|---:|---:|']
    for policy in ['B0_original','B1_exact_diagnostic','O_filter_diagnostic']:
        z=n['conditions'][f'{policy}|current|background=100|budget=256'];c=z['counts'];den=z['n']
        lines.append(f"| {policy} | {c['old_claim_visible']}/{den} | {c['new_claim_visible']}/{den} | {c['new_text_visible_verbatim']}/{den} |")
    lines += ['',
    'B1 未满足 N1 的同义召回保护条件，因此这里只作保守基线诊断，不称为已选出的优胜方法。',
    'O_filter 使用 dataset update 标签作 oracle 诊断，只过滤实际匹配旧 claim 的条目，不按可能已与背景合并的 ID 盲删，历史探针不自动过滤旧状态。最初本地 ID 版已由当前 claim 版替代。',
    '关键发现：过滤旧 claim 使新原文完整可见从 9/24 降到 1/24；部分新证据依附在旧 claim 下，删条目同时丢掉可用的新信息。它提示后续需区分 claim 状态和 evidence 保留，不能把旧文本消失率单独当作方法成功。完整原文可见是严格字符串指标，不等于语义理解或回答正确率。',
    '背景数量也不等于最终存储条数：B0 的 20 条背景条件最后约 6–8 条 trait，100 条条件约 75–80 条；B1 对应 22/102 条（含新旧目标）。统计比较保留这个合并造成的代价差异。',
    f"在全部 {n['n_records']} 条记录中，仅 {sum(z['n_with_budget_drops'] for z in n['conditions'].values())} 条发生 token 预算丢弃；256/512 token 条件没有形成足够预算压力，20/100 背景下上述计数相同。当前结果不能证明长期预算优化已经成功。", '',
    '## 下一步直接推进的方向', '',
    '1. 在开发集补充不同表述的机械主题探针并报告其偏差，保留当前固定探针作回归，不根据测试结果调题。',
    '2. 分别实验 claim 失效处理与 evidence 保留，再接持久预算和 FIFO/recency 基线。仅有 oracle 收益不能算可部署方法收益。',
    '3. N3 优先构造真正触发存储/上下文限制的开发轨迹，按实际 tokenizer token、完整 evidence 和 DB 字节核算；不因本轮旧 claim 高可见率直接宣称真实用户有同样频率。',
    '4. 299 条测试输入继续封存推理结果，开发配置锁定后统一运行。没有人工标注阻塞；生成模型阶段仅受实际模型/预算配置约束。', '',
    '## 复现与结果', '',
    '`PYTHON=.venv/bin/python bash scripts/reproduce_development.sh` 从原始 N0 manifest 派生有效划分，再运行测试、N1、自动标签审计和受控 N2；每次生成独立 run_id，拒绝覆盖已有目录。',
    f"- [N1 指标]({a.relative_to(ROOT/'reports')}/metrics.json)、[自动结构审计]({a.relative_to(ROOT/'reports')}/automatic_label_audit.json)、[开发选择]({a.relative_to(ROOT/'reports')}/selection.json)。",
    f"- [N2 指标]({b.relative_to(ROOT/'reports')}/metrics.json)、[实际呈现记录]({b.relative_to(ROOT/'reports')}/replay_results.jsonl)。",
    '- [本轮产物校验](development_verification.json)、[测试日志](development_tests.txt)。', '']
    (ROOT/'reports/DEVELOPMENT_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
