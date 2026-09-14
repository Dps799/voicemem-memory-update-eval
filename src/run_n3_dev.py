"""N3: 有界记忆长期更新与遗忘。

构造受控事件轨迹（40/160 事件），在存储预算和检索上下文预算约束下
测试不同淘汰策略。检查点评估当前信息可见率、旧状态暴露率、历史保留率
和不变偏好保留率。

CPU 可运行，0 生成模型调用。使用固定 E5 编码。

产物:
  reports/n3_dev/<run_id>/trajectory_manifest.jsonl
  reports/n3_dev/<run_id>/checkpoint_results.jsonl
  reports/n3_dev/<run_id>/metrics.json
  reports/n3_dev/<run_id>/preflight.json

策略:
  P0: B0 合并(0.95) + 达到预算后 FIFO 淘汰
  P1: B1 精确匹配 + FIFO 淘汰
  P2: B1 精确匹配 + recency/frequency 淘汰
  O_state: oracle 诊断，屏蔽 gold 已失效判断
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e5_callback import (
    TraitStore, Evidence, normalize_claim, real_embed,
    precompute_embeddings, cos_sim, E5_MODEL, E5_REVISION, _model,
)
from run_phase_a import B1ExactStore
from upstream_retrieval import HELPERS, provenance
import traits_store
from prepare_n1 import read_rows

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260914

# 预算对 (持久存储 token, 检索上下文 token)
BUDGET_PAIRS = [
    (1024, 256),
    (2048, 256),
    (4096, 256),
    (2048, 128),
    (2048, 512),
]
CHECKPOINTS_40 = [20, 40]
CHECKPOINTS_160 = [20, 40, 80, 160]
QUESTIONS_PER_CHECKPOINT = 3  # current, historical, unchanged


def build_trajectories(pairs: list[dict], n_short: int = 6, n_long: int = 6) -> list[dict]:
    """构造受控事件轨迹。

    短轨迹 40 事件，长轨迹 160 事件。
    每个轨迹注入：
    - 持续更新（同一偏好的多次变化）
    - 明确纠正
    - 临时例外
    - 不变核心偏好
    - 同主题近邻（触发合并/冲突）
    """
    rng = random.Random(SEED)
    dev_pm = [p for p in pairs if p["split"] == "dev" and p["source"] == "PersonaMem-v2"]
    personas = sorted({p["group_id"] for p in dev_pm})
    rng.shuffle(personas)
    selected = personas[:n_short + n_long]

    trajectories = []
    for i, persona in enumerate(selected):
        length = 40 if i < n_short else 160
        checkpoints = CHECKPOINTS_40 if length == 40 else CHECKPOINTS_160

        # 找该 persona 的偏好更新作为种子
        seed_pairs = [p for p in dev_pm if p["group_id"] == persona]
        if not seed_pairs:
            continue

        # 构造事件序列
        events = []
        # 不变核心偏好（贯穿整个轨迹）
        core_pref = seed_pairs[0]["claim_a"] if seed_pairs else "Values personal growth"
        events.append({
            "event_id": 0, "step": 0, "type": "core_unchanged",
            "claim": core_pref, "slot": "喜好与厌恶",
            "evidence": f"User stated: {core_pref}",
            "gold_relation": "equivalent", "gold_operation": "merge_support",
        })

        # 持续更新序列
        update_chain = ["Prefers morning workouts", "Prefers evening workouts",
                        "Prefers afternoon workouts", "Prefers early morning workouts"]
        for j, pref in enumerate(update_chain):
            events.append({
                "event_id": len(events), "step": 5 + j * 10, "type": "continuous_update",
                "claim": pref, "slot": "喜好与厌恶",
                "evidence": f"User updated: {pref}",
                "gold_relation": "temporal_update", "gold_operation": "supersede_current_keep_history",
                "supersedes": events[-1]["claim"] if events else None,
            })

        # 明确纠正
        events.append({
            "event_id": len(events), "step": 48 if length == 160 else 12, "type": "explicit_correction",
            "claim": "Dislikes loud music", "slot": "喜好与厌恶",
            "evidence": "User corrected: actually dislikes loud music",
            "gold_relation": "temporal_update", "gold_operation": "supersede_current_keep_history",
            "supersedes": "Likes loud music",
        })

        # 临时例外
        events.append({
            "event_id": len(events), "step": 60 if length == 160 else 18, "type": "temporary_exception",
            "claim": "Enjoys loud music at concerts", "slot": "喜好与厌恶",
            "evidence": "User noted exception: enjoys loud music at concerts",
            "gold_relation": "contextual_difference", "gold_operation": "retain_contextual_exception",
        })

        # 同主题近邻（触发合并/冲突）
        near_topics = [
            ("Enjoys reading science fiction", "equivalent", "merge_support"),
            ("Likes reading sci-fi novels", "equivalent", "merge_support"),
            ("Dislikes reading fantasy", "contradiction_same_scope", "add_separate"),
            ("Enjoys hiking on weekends", "unrelated", "add_separate"),
        ]
        for claim, rel, op in near_topics:
            events.append({
                "event_id": len(events), "step": 70 + len(events) * 5 if length == 160 else 25 + len(events) * 3,
                "type": "near_neighbor", "claim": claim, "slot": "喜好与厌恶",
                "evidence": f"User mentioned: {claim}",
                "gold_relation": rel, "gold_operation": op,
            })

        # 填充到目标长度（用更多近邻和重述）
        filler = [
            "Values independence", "Enjoys quiet evenings", "Likes Italian food",
            "Prefers public transport", "Enjoys photography", "Likes black coffee",
            "Values punctuality", "Enjoys board games", "Prefers summer weather",
            "Likes hiking trails", "Enjoys jazz music", "Values honesty",
        ]
        step = max(e["step"] for e in events) + 5
        while len(events) < length:
            claim = rng.choice(filler)
            events.append({
                "event_id": len(events), "step": step, "type": "filler",
                "claim": claim, "slot": "喜好与厌恶",
                "evidence": f"User said: {claim}",
                "gold_relation": "unrelated", "gold_operation": "add_separate",
            })
            step += rng.randint(3, 8)

        events.sort(key=lambda e: e["step"])
        trajectories.append({
            "trajectory_id": f"traj_{persona[:20]}_{length}",
            "persona": persona,
            "length": length,
            "checkpoints": checkpoints,
            "events": events[:length],
            "core_preference": core_pref,
            "final_update": "Prefers early morning workouts",
            "exception": "Enjoys loud music at concerts",
        })
    return trajectories


def count_tokens(text: str, tokenizer) -> int:
    """用 E5 tokenizer 计 token（诊断单位，不是回复模型 token）。"""
    return len(tokenizer.encode(text, add_special_tokens=False))


class BoundedStore:
    """有界存储：达到持久预算后按淘汰策略移除 trait。

    持久预算 = 序列化 claim + evidence + 时间标签的 token 数。
    """

    def __init__(self, db_path, embed, merge_threshold, eviction_policy, storage_budget_tokens, tokenizer):
        traits_store.MERGE_THRESHOLD = merge_threshold
        if merge_threshold == "string_match":
            self.store = B1ExactStore(db_path, embed)
        else:
            self.store = TraitStore(db_path, embed)
        self.store._effective_threshold = merge_threshold
        self.eviction_policy = eviction_policy  # "fifo", "recency_frequency", "none"
        self.storage_budget = storage_budget_tokens
        self.tokenizer = tokenizer
        self._access_count = defaultdict(int)  # trait_id -> access count
        self._last_access = {}  # trait_id -> last access step

    def add(self, user_id, slot, claim, evidence, step=0):
        tid = self.store.add(user_id, slot, claim, evidence)
        if tid:
            self._access_count[tid] += 1
            self._last_access[tid] = step
        self._enforce_budget(user_id)
        return tid

    def _enforce_budget(self, user_id):
        """达到预算后按淘汰策略移除最旧/最少用的 trait。"""
        if self.eviction_policy == "none":
            return
        while self._current_tokens(user_id) > self.storage_budget:
            candidates = self._get_eviction_candidates(user_id)
            if not candidates:
                break
            victim = self._select_victim(candidates)
            self._evict(user_id, victim)

    def _current_tokens(self, user_id) -> int:
        """计算当前存储的序列化 token 数。"""
        with self.store._conn() as c:
            traits = c.execute(
                "SELECT id, claim FROM rb_traits WHERE user_id=?", (user_id,)
            ).fetchall()
            evs = c.execute(
                "SELECT trait_id, quote FROM rb_evidence WHERE user_id=?", (user_id,)
            ).fetchall()
        total = 0
        for t in traits:
            total += count_tokens(t["claim"], self.tokenizer) + 1  # +1 for newline
        for e in evs:
            total += count_tokens(e["quote"], self.tokenizer) + 1
        return total

    def _get_eviction_candidates(self, user_id):
        with self.store._conn() as c:
            return [r["id"] for r in c.execute(
                "SELECT id FROM rb_traits WHERE user_id=?", (user_id,)
            ).fetchall()]

    def _select_victim(self, candidates):
        if self.eviction_policy == "fifo":
            # 最早创建的先淘汰
            with self.store._conn() as c:
                rows = c.execute(
                    "SELECT id, created_at FROM rb_traits WHERE id IN (%s) ORDER BY created_at ASC LIMIT 1"
                    % ",".join("?" * len(candidates)), candidates
                ).fetchall()
            return rows[0]["id"] if rows else None
        elif self.eviction_policy == "recency_frequency":
            # 最少访问 + 最早最后访问的先淘汰
            scored = []
            for tid in candidates:
                freq = self._access_count.get(tid, 0)
                recency = self._last_access.get(tid, -1)
                scored.append((freq, recency, tid))
            scored.sort()  # 低 freq, 低 recency 先
            return scored[0][2] if scored else None
        return None

    def _evict(self, user_id, trait_id):
        with self.store._conn() as c:
            c.execute("DELETE FROM rb_evidence WHERE trait_id=?", (trait_id,))
            c.execute("DELETE FROM rb_traits WHERE id=?", (trait_id,))

    def search_scored(self, user_id, query, top_k=5):
        results = self.store.search_scored(user_id, query, top_k=top_k)
        for t, _ in results:
            self._access_count[t.id] += 1
        return results

    def counts(self, user_id):
        return self.store.counts(user_id)

    def get_traits(self, user_id):
        with self.store._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT id, claim, created_at, updated_at FROM rb_traits WHERE user_id=?", (user_id,)
            ).fetchall()]


def run_trajectory(traj, policy_config, budget_pair, tokenizer):
    """运行单条轨迹，在检查点评估。"""
    storage_budget, context_budget = budget_pair
    user_id = "user_n3"

    store = BoundedStore(
        tempfile.mktemp(suffix=".db"),
        real_embed,
        policy_config["threshold"],
        policy_config["eviction"],
        storage_budget,
        tokenizer,
    )

    results = []
    events = traj["events"]
    checkpoints = traj["checkpoints"]

    event_idx = 0
    for cp in checkpoints:
        # 写入直到检查点的事件
        while event_idx < len(events) and events[event_idx]["step"] <= cp:
            ev = events[event_idx]
            evidence = Evidence(
                quote=ev["evidence"], emotion="test",
                at=f"2026-09-{1 + ev['step'] // 10:02d}T{(ev['step'] % 10):02d}:00:00+00:00"
            )
            store.add(user_id, ev["slot"], ev["claim"], evidence, step=ev["step"])
            event_idx += 1

        # 检查点评估：3 个问题
        core = traj["core_preference"]
        final = traj["final_update"]
        exception = traj["exception"]

        queries = {
            "current": f"What are my current workout preferences?",
            "historical": f"What did I previously prefer regarding workouts?",
            "unchanged": f"What are my core values?",
        }

        for kind, query in queries.items():
            # 检索 + 预算打包
            hits = HELPERS["_rb_trait_hits"](store.store, user_id, query)
            hits.sort(key=lambda h: h.priority, reverse=True)
            quota_hits = HELPERS["_apply_source_quota"](hits)

            # oracle 过滤
            if policy_config.get("oracle") and kind == "current":
                # 屏蔽 gold 已失效的判断
                gold_superseded = {"Prefers morning workouts", "Prefers evening workouts",
                                   "Prefers afternoon workouts", "Likes loud music"}
                quota_hits = [h for h in quota_hits
                              if normalize_claim(h.metadata["claim"]) not in
                              {normalize_claim(g) for g in gold_superseded}]

            # token 预算打包
            chosen = []
            for hit in quota_hits:
                proposed = "\n".join(h.content for h in chosen + [hit])
                if count_tokens(proposed, tokenizer) <= context_budget:
                    chosen.append(hit)

            directive = "\n".join(h.content for h in chosen)
            presented_claims = [h.metadata["claim"] for h in chosen]

            # 评估
            traits = store.get_traits(user_id)
            trait_claims = {t["claim"] for t in traits}

            results.append({
                "trajectory_id": traj["trajectory_id"],
                "persona": traj["persona"],
                "checkpoint": cp,
                "policy": policy_config["name"],
                "storage_budget": storage_budget,
                "context_budget": context_budget,
                "query_kind": kind,
                "n_traits_stored": len(traits),
                "current_info_visible": normalize_claim(final) in presented_claims or
                    any(normalize_claim(final) in tc for tc in trait_claims),
                "old_state_exposed": kind == "current" and any(
                    normalize_claim(old) in presented_claims
                    for old in ["Prefers morning workouts", "Prefers evening workouts",
                                "Prefers afternoon workouts"]
                ),
                "history_visible": kind == "historical" and any(
                    normalize_claim(old) in presented_claims
                    for old in ["Prefers morning workouts", "Prefers evening workouts"]
                ),
                "unchanged_visible": kind == "unchanged" and normalize_claim(core) in presented_claims,
                "exception_visible": normalize_claim(exception) in presented_claims,
                "presented_claims": presented_claims,
                "directive": directive[:200],
                "context_tokens": count_tokens(directive, tokenizer),
                "eviction_policy": policy_config["eviction"],
                "error": None,
            })

    return results


POLICIES = [
    {"name": "P0_b0_fifo", "threshold": 0.95, "eviction": "fifo"},
    {"name": "P1_exact_fifo", "threshold": "string_match", "eviction": "fifo"},
    {"name": "P2_exact_recency", "threshold": "string_match", "eviction": "recency_frequency"},
    {"name": "O_state_oracle", "threshold": 0.95, "eviction": "fifo", "oracle": True},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    name = args.run_id or datetime.now(timezone.utc).strftime("n3_dev_%Y%m%dT%H%M%SZ")
    if Path(name).name != name:
        raise ValueError("single directory name required")
    output = ROOT / "reports" / "n3_dev" / name
    output.mkdir(parents=True, exist_ok=False)

    # 加载有效 manifest
    pairs = read_rows(ROOT / "manifests" / "n1_effective_manifest.jsonl")
    trajectories = build_trajectories(pairs, n_short=6, n_long=6)
    print(f"Built {len(trajectories)} trajectories")

    # 写 trajectory manifest
    traj_manifest = []
    for t in trajectories:
        traj_manifest.append({
            "trajectory_id": t["trajectory_id"],
            "persona": t["persona"],
            "length": t["length"],
            "checkpoints": t["checkpoints"],
            "core_preference": t["core_preference"],
            "final_update": t["final_update"],
            "exception": t["exception"],
            "event_types": list({e["type"] for e in t["events"]}),
        })
    (output / "trajectory_manifest.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traj_manifest)
    )

    # 预计算 embedding
    all_texts = []
    for t in trajectories:
        for e in t["events"]:
            all_texts.append(e["claim"])
            all_texts.append(e["evidence"])
            all_texts.append(normalize_claim(e["claim"]))
        all_texts.extend([t["core_preference"], t["final_update"], t["exception"]])
    print(f"Precomputing {len(set(all_texts))} unique embeddings...")
    start = perf_counter()
    precompute_embeddings(all_texts)
    tokenizer = _model().tokenizer
    precompute_seconds = perf_counter() - start
    print(f"Precompute done in {precompute_seconds:.1f}s")

    all_results = []
    for traj in trajectories:
        print(f"Running {traj['trajectory_id']} (length={traj['length']})...")
        for policy in POLICIES:
            for budget in BUDGET_PAIRS:
                try:
                    results = run_trajectory(traj, policy, budget, tokenizer)
                    all_results.extend(results)
                except Exception as e:
                    all_results.append({
                        "trajectory_id": traj["trajectory_id"],
                        "policy": policy["name"],
                        "storage_budget": budget[0],
                        "context_budget": budget[1],
                        "error": f"{type(e).__name__}: {e}",
                    })

    # 写结果
    (output / "checkpoint_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_results)
    )
    print(f"Written {len(all_results)} checkpoint results")

    # 汇总指标
    metrics = summarize(all_results, trajectories)
    metrics["n_trajectories"] = len(trajectories)
    metrics["n_personas"] = len({t["persona"] for t in trajectories})
    metrics["n_records"] = len(all_results)
    metrics["expected_records"] = sum(
        len(t["checkpoints"]) * QUESTIONS_PER_CHECKPOINT * len(POLICIES) * len(BUDGET_PAIRS)
        for t in trajectories
    )
    metrics["n_errors"] = sum(1 for r in all_results if r.get("error"))
    metrics["generative_api_calls"] = 0
    metrics["tokenizer_revision"] = E5_REVISION
    metrics["retrieval"] = provenance()
    metrics["total_wall_seconds"] = perf_counter() - start
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    # preflight
    files = [ROOT / "src" / "run_n3_dev.py", ROOT / "manifests" / "n1_effective_manifest.jsonl"] + list(output.glob("*.json*"))
    preflight = {
        "head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": "completed" if not metrics["n_errors"] else "completed_with_errors",
        "scope": "N3 development; controlled trajectories; not natural dialogue",
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "expected_records": metrics["expected_records"],
        "actual_records": len(all_results),
        "budget_pairs": BUDGET_PAIRS,
        "policies": [p["name"] for p in POLICIES],
    }
    (output / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n")

    print(f"\nOutput: {output}")
    print(f"Records: {len(all_results)}, Errors: {metrics['n_errors']}")
    print_summary(metrics)


def summarize(results, trajectories):
    """按策略和预算汇总检查点指标。"""
    valid = [r for r in results if r.get("error") is None]
    buckets = defaultdict(list)
    for r in valid:
        key = f"{r['policy']}|storage={r['storage_budget']}|context={r['context_budget']}"
        buckets[key].append(r)

    conditions = {}
    for key, rs in buckets.items():
        n = len(rs)
        n_personas = len({r["persona"] for r in rs})
        current_qs = [r for r in rs if r["query_kind"] == "current"]
        hist_qs = [r for r in rs if r["query_kind"] == "historical"]
        unch_qs = [r for r in rs if r["query_kind"] == "unchanged"]

        conditions[key] = {
            "n_records": n,
            "n_personas": n_personas,
            "n_checkpoints": len({r["checkpoint"] for r in rs}),
            "current_info_visible": sum(r["current_info_visible"] for r in current_qs),
            "current_n": len(current_qs),
            "current_info_rate": sum(r["current_info_visible"] for r in current_qs) / max(1, len(current_qs)),
            "old_state_exposed": sum(r["old_state_exposed"] for r in current_qs),
            "old_state_exposure_rate": sum(r["old_state_exposed"] for r in current_qs) / max(1, len(current_qs)),
            "history_visible": sum(r["history_visible"] for r in hist_qs),
            "history_rate": sum(r["history_visible"] for r in hist_qs) / max(1, len(hist_qs)),
            "unchanged_visible": sum(r["unchanged_visible"] for r in unch_qs),
            "unchanged_rate": sum(r["unchanged_visible"] for r in unch_qs) / max(1, len(unch_qs)),
            "mean_context_tokens": sum(r["context_tokens"] for r in rs) / max(1, n),
            "mean_traits_stored": sum(r["n_traits_stored"] for r in rs) / max(1, n),
        }

    return {
        "scope": "controlled trajectory development; not natural user distribution",
        "n_trajectories": len(trajectories),
        "trajectory_lengths": sorted({t["length"] for t in trajectories}),
        "conditions": conditions,
        "answer_quality_measured": False,
        "oracle_is_diagnostic_only": True,
    }


def print_summary(metrics):
    print("\n" + "=" * 60)
    print("N3 DEVELOPMENT SUMMARY")
    print("=" * 60)
    for key, c in sorted(metrics["conditions"].items()):
        print(f"\n{key}:")
        print(f"  current_info_rate={c['current_info_rate']:.2%} ({c['current_info_visible']}/{c['current_n']})")
        print(f"  old_state_exposure={c['old_state_exposure_rate']:.2%} ({c['old_state_exposed']}/{c['current_n']})")
        print(f"  history_rate={c['history_rate']:.2%} ({c['history_visible']})")
        print(f"  unchanged_rate={c['unchanged_rate']:.2%} ({c['unchanged_visible']})")
        print(f"  mean_traits={c['mean_traits_stored']:.1f}, mean_tokens={c['mean_context_tokens']:.0f}")


if __name__ == "__main__":
    main()
