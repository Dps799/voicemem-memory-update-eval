"""Execute pinned upstream trait retrieval helpers without importing unavailable services.

AST-selected definitions are compiled unchanged from the vendored brain.py.
This is the profile-only slice: no heartnotes, left brain, full prompt or answerer.
"""
from __future__ import annotations
import ast
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'upstream/voicemem/rightbrain/brain.py'
NAMES = {'RightBrainHit', '_TRAIT_MIN_SIM_BY_DIM', '_TRAIT_MIN_SIM_DEFAULT',
         'RB_TRAIT_MIN_SIM', 'trait_min_sim', '_rb_trait_hits', '_SOURCE_QUOTA',
         '_apply_source_quota', '_render_rb_directive'}

def load_helpers():
    text = SOURCE.read_text()
    nodes = []
    found = set()
    for node in ast.parse(text).body:
        names = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & NAMES:
            nodes.append(node)
            found |= names & NAMES
    if found != NAMES:
        raise RuntimeError(f'Upstream helper definitions missing: {NAMES - found}')
    namespace = {'__name__': __name__, 'os': os, 'dataclass': dataclass, 'field': field}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace

HELPERS = load_helpers()

def retrieve_profile(store, user_id, query, budget=3):
    # Same helper, sort, quota, render sequence as RightBrain.search, with only profile hits.
    hits = HELPERS['_rb_trait_hits'](store, user_id, query)
    hits.sort(key=lambda h: h.priority, reverse=True)
    hits = HELPERS['_apply_source_quota'](hits)[:budget]
    return hits, HELPERS['_render_rb_directive'](hits)

def provenance():
    return {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'execution': 'unchanged AST-selected upstream helpers; profile-only',
            'candidate_top_k': 4, 'profile_quota': HELPERS['_SOURCE_QUOTA']['profile'],
            'minimum_similarity_384': HELPERS['trait_min_sim'](384)}
