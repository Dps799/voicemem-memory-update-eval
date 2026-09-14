"""真实 E5 编码回调，匹配 VoiceMem LocalE5Embedder.embed_texts 的注入路径。

VoiceMem 的 TraitStore._vec 调用 self._embed(text)（裸 text），
而 LocalE5Embedder.embed_texts 给每条加 "passage: " 前缀后 encode(normalize=True)。
本模块复现这条真实路径，不自行重写余弦规则。

commit 固定: a450911fc8cbb44c46d810aace2f3288bad287e4
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# VoiceMem 上游核心代码目录（固定 commit a450911fc8c）
VOICEMEM_DIR = Path(__file__).parent.parent / "upstream"

E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
E5_SNAPSHOT = os.environ.get("VOICEMEM_E5_PATH", E5_MODEL)


# 把 VoiceMem 的 rightbrain 目录加到 path，直接导入固定 commit 的 traits_store
sys.path.insert(0, str(VOICEMEM_DIR / "voicemem" / "rightbrain"))

from traits_store import TraitStore, Evidence, MERGE_THRESHOLD, SLOTS, normalize_claim  # noqa: E402

import numpy as np
from sentence_transformers import SentenceTransformer  # noqa: E402

_MODEL = None


def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(E5_SNAPSHOT, revision=E5_REVISION, device="cpu", token=False)
    return _MODEL


_CACHE: dict[str, list[float]] = {}


def real_embed(text: str) -> list[float]:
    """匹配 LocalE5Embedder.embed_texts：加 passage: 前缀，normalize=True。

    VoiceMem 注入给 TraitStore 的 embed 函数最终走的是
    LocalE5Embedder().embed_texts([t])[0]，即 "passage: {t}" + normalize。
    """
    if not text:
        text = ""
    if text in _CACHE:
        return _CACHE[text]
    vec = _model().encode([f"passage: {text}"], normalize_embeddings=True)[0]
    out = np.asarray(vec, dtype=np.float32).tolist()
    _CACHE[text] = out
    return out


def precompute_embeddings(texts: list[str]) -> None:
    """批量预计算所有唯一文本的 embedding（避免逐条 encode 的开销）。"""
    unique = sorted(set(t for t in texts if t))
    unique = [t for t in unique if t not in _CACHE]
    if not unique:
        return
    vecs = _model().encode(
        [f"passage: {t}" for t in unique], normalize_embeddings=True, show_progress_bar=False
    )
    for t, v in zip(unique, vecs):
        _CACHE[t] = np.asarray(v, dtype=np.float32).tolist()


def real_embed_query(text: str) -> list[float]:
    """检索侧用 query: 前缀（匹配 LocalE5Embedder.embed_query_text）。"""
    vec = _model().encode([f"query: {text}"], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32).tolist()


def cos_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / n) if n else 0.0


def embedding_hash(text: str) -> str:
    """缓存向量按模型 revision、前缀、精度、规范化文本 hash 命名（计划要求）。"""
    canon = f"e5-small|614241f6|passage|fp32|{normalize_claim(text)}"
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


__all__ = [
    "TraitStore",
    "Evidence",
    "MERGE_THRESHOLD",
    "SLOTS",
    "normalize_claim",
    "real_embed",
    "real_embed_query",
    "cos_sim",
    "embedding_hash",
    "VOICEMEM_DIR",
]
