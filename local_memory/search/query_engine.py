"""Layer 5 — Query & Ranking: hybrid natural-language search.

A single user query fans out to three retrievers (parallel threads) and
results are fused:

1. Semantic (Nomic text vectors)   → documents, PDFs, DOCX, OCR text of images
2. Visual (CLIP image vectors)     → image *content* matched to the query text
3. Keyword (SQLite FTS5 BM25)      → exact term matches over chunk text and
                                     file path/OCR text, high precision

Query is first expanded by query_rewrite (Qwen3-NPU or regex): synonyms,
kind hints, date ranges. Final score = 0.60·semantic + 0.25·visual +
0.15·keyword, plus recency boost.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ..embeddings import get_clip_text_embedder, get_text_embedder
from ..store import database, vector_store
from .query_rewrite import rewrite

WEIGHT_SEMANTIC = 0.60
WEIGHT_VISUAL = 0.25
WEIGHT_KEYWORD = 0.15
RECENCY_BOOST_DAYS = 30.0
RECENCY_MAX = 0.10


def _recency_boost(mtime: float) -> float:
    age_days = max(0.0, (time.time() - mtime) / 86400.0)
    if age_days > RECENCY_BOOST_DAYS:
        return 0.0
    return RECENCY_MAX * (1.0 - age_days / RECENCY_BOOST_DAYS)


def fts_quote(query: str) -> str:
    """FTS5-safe MATCH expression: every whitespace-separated term is wrapped
    in double quotes with internal quotes doubled, and terms are joined with
    OR so partial matches still rank (BM25 orders by relevance). Neutralizes
    FTS5 query syntax — a query like ``pen" OR (`` can never raise.
    """
    terms = [t for t in query.split() if t]
    if not terms:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def _bm25_hits(query: str, limit: int):
    """BM25 ranked lists from the FTS5 tables (research §3).

    Returns (kw_scores, best_chunk, kw_source):
      kw_scores  file_id -> 60/(60+rank0) with rank0 = best position across
                 both lists (normalized into the blend's [0,1] keyword scale)
      best_chunk file_id -> winning chunk rowid for snippet() (0 for
                 files_fts-driven hits)
      kw_source  file_id -> 'chunk' | 'ocr' | 'name'
    Any sqlite error (missing table, unavailable FTS5) degrades to empty
    lists so keyword failure never breaks dense search.
    """
    match_q = fts_quote(query)
    if not match_q:
        return {}, {}, {}
    empty: tuple[dict[int, float], dict[int, int], dict[int, str]] = ({}, {}, {})
    try:
        # Chunk-level: join back to chunks for file_id; rank is implicit BM25.
        rows = database.fts_rows(
            "SELECT c.file_id AS file_id, c.id AS chunk_id, rank "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_q, limit),
        )
        best_rank: dict[int, int] = {}
        best_chunk: dict[int, int] = {}
        source: dict[int, str] = {}
        for pos, r in enumerate(rows):  # first occurrence per file = best rank
            fid = r["file_id"]
            if fid not in best_rank:
                best_rank[fid] = pos
                best_chunk[fid] = r["chunk_id"]
                source[fid] = "chunk"
        # File-level: path/OCR text (rowid IS files.id).
        rows = database.fts_rows(
            "SELECT rowid AS file_id, rank FROM files_fts "
            "WHERE files_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_q, limit),
        )
        for pos, r in enumerate(rows):
            fid = r["file_id"]
            if fid not in best_rank or pos < best_rank[fid]:
                best_rank[fid] = pos
                best_chunk[fid] = 0
                source[fid] = "ocr"
        scores = {fid: 60.0 / (60.0 + pos) for fid, pos in best_rank.items()}
        return scores, best_chunk, source
    except Exception:
        return empty


def _keyword_hits(query: str, limit: int = 36):
    """Keyword retriever (fan-out slot): BM25 over the FTS5 index."""
    return _bm25_hits(query, limit)


def _make_snippet(file_id: int, ordinal: int, query: str) -> str:
    text = database.get_chunk_text(file_id, ordinal)
    if not text:
        return ""
    if not query:
        return text[:220]
    lower = text.lower()
    for term in query.lower().split():
        idx = lower.find(term)
        if idx >= 0:
            start = max(0, idx - 80)
            return ("…" if start else "") + text[start : idx + 160] + ("…" if idx + 160 < len(text) else "")
    return text[:220]


def search(query: str, top_k: int = 12) -> list[dict]:
    if not query.strip():
        return []
    rw = rewrite(query)
    expanded = rw["expanded"]
    kinds = set(rw.get("kinds", []))
    dfrom, dto = rw.get("date_from"), rw.get("date_to")

    file_rows = {row["id"]: row for row in database.list_files()}

    # Parallel fan-out: semantic + visual + keyword on 3 threads.
    def _sem():
        try:
            text_emb = get_text_embedder()
            q_text = text_emb.encode(expanded)
            return vector_store.search("text", q_text, top_k=top_k * 2)
        except Exception:
            return []

    def _vis():
        try:
            clip = get_clip_text_embedder()
            q_img = clip.encode_query(expanded)
            return vector_store.search("image", q_img, top_k=top_k)
        except Exception:
            return []

    def _kw():
        return _keyword_hits(expanded, top_k * 3)

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="lm-search") as pool:
        f_sem, f_vis, f_kw = pool.submit(_sem), pool.submit(_vis), pool.submit(_kw)
        sem_hits, vis_hits, kw_detail = f_sem.result(), f_vis.result(), f_kw.result()
    kw_scores, kw_chunk, kw_source = kw_detail
    kw_match = fts_quote(expanded)
    sem_scores: dict[int, float] = {}
    sem_ord: dict[int, int] = {}
    for h in sem_hits:
        fid = h["file_id"]
        if fid not in sem_scores or h["score"] > sem_scores[fid]:
            sem_scores[fid] = h["score"]
            sem_ord[fid] = h["ordinal"]

    # 2. Visual search (CLIP text→image)
    vis_scores: dict[int, float] = {}
    for h in vis_hits:
        fid = h["file_id"]
        if fid not in vis_scores or h["score"] > vis_scores[fid]:
            vis_scores[fid] = h["score"]

    candidates = set(sem_scores) | set(vis_scores) | set(kw_scores)
    results = []

    def _fts_snippet(fid: int) -> tuple[str, str]:
        """Marked snippet for a BM25 hit: chunk text first, OCR text for
        files_fts-driven hits (D-06 priority refinement is 03-02)."""
        try:
            chunk_id = kw_chunk.get(fid, 0)
            if chunk_id:
                s = database.chunk_snippet(chunk_id, kw_match)
                if s:
                    return s, "chunk"
            s = database.file_snippet(fid, kw_match)
            if s:
                return s, kw_source.get(fid, "ocr")
        except Exception:
            pass
        return "", ""

    for fid in candidates:
        row = file_rows.get(fid)
        if row is None:
            continue
        # Kind filter from rewriter: down-rank mismatched kinds, don't drop.
        kind_penalty = 0.0
        if kinds:
            is_img = row["kind"] == "image"
            if "image" in kinds and not is_img and "doc" not in kinds:
                kind_penalty = 0.05
        # Date filter from rewriter ("last month", month names).
        if dfrom and dto and not (dfrom <= row["mtime"] <= dto):
            # Soft penalty instead of hard drop (keeps recall when mtime off).
            kind_penalty += 0.05
        s = (
            WEIGHT_SEMANTIC * sem_scores.get(fid, 0.0)
            + WEIGHT_VISUAL * vis_scores.get(fid, 0.0)
            + WEIGHT_KEYWORD * kw_scores.get(fid, 0.0)
            + _recency_boost(row["mtime"])
            - kind_penalty
        )
        # Snippet: FTS5 <mark> snippet for keyword hits, chunk-window
        # fallback for semantic-only hits, raw OCR/name tail otherwise.
        fts_snip, fts_src = _fts_snippet(fid) if fid in kw_scores else ("", "")
        if fts_snip:
            snippet, snippet_source = fts_snip, fts_src
        elif sem_ord.get(fid) is not None:
            snippet, snippet_source = _make_snippet(fid, sem_ord[fid], query), "chunk"
        else:
            snippet = (row["ocr_text"] or "")[:220]
            snippet_source = "ocr" if snippet else "name"
        results.append(
            {
                "file_id": fid,
                "path": row["path"],
                "name": row["path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
                "kind": row["kind"],
                "folder": row["folder"],
                "size_bytes": row["size_bytes"],
                "mtime": row["mtime"],
                "score": round(s, 4),
                "match_semantic": round(sem_scores.get(fid, 0.0), 3),
                "match_visual": round(vis_scores.get(fid, 0.0), 3),
                "match_keyword": round(kw_scores.get(fid, 0.0), 3),
                "snippet": snippet,
                "snippet_source": snippet_source,
            }
        )

    results.sort(key=lambda r: -r["score"])
    return results[:top_k]
