"""Layer 5 — Query & Ranking: hybrid natural-language search.

A single user query fans out to three retrievers (parallel threads) and
results are fused with Reciprocal Rank Fusion (D-03):

1. Semantic (Nomic text vectors)   → documents, PDFs, DOCX, OCR text of images
2. Visual (CLIP image vectors)     → image *content* matched to the query text
3. Keyword (SQLite FTS5 BM25)      → TWO ranked lists: chunk text and
                                     file path/OCR text, high precision

Query is first expanded by query_rewrite (Qwen3-NPU or regex): synonyms,
kind hints, date ranges. Final score = Σ_lists 1/(RRF_K + rank) with rank
1-based per list; the legacy recency boost and kind/date penalties survive
ONLY as tiny additive tie-breakers (±0.05 scale — they cannot distort the
RRF ordering of top ranks, which live at ~1/60 per list contribution).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ..embeddings import get_clip_text_embedder, get_text_embedder
from ..store import database, vector_store
from .query_rewrite import rewrite

RRF_K = 60  # standard RRF constant; rank is 1-BASED per list (research §3)
TIE_SCALE = 0.05  # max magnitude of recency/penalty tie-breaker terms
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

    Returns (kw_scores, best_chunk, kw_source, kw_lists):
      kw_scores  file_id -> 60/(60+rank0) — normalized best-of-both-lists
                 score, kept ONLY for the deprecated ``match_keyword``
                 display field (the UI renders it until 03-03 removes it);
                 RRF fusion uses kw_lists ranks, not these scores
      best_chunk file_id -> winning chunk rowid for snippet() (0 for
                 files_fts-driven hits)
      kw_source  file_id -> 'chunk' | 'ocr'
      kw_lists   (chunk_fids, file_fids) — two rank-ordered file_id lists,
                 one per FTS table; RRF consumes rank positions from each
    Any sqlite error (missing table, unavailable FTS5) degrades to empty
    lists so keyword failure never breaks dense search.
    """
    match_q = fts_quote(query)
    if not match_q:
        return {}, {}, {}, ([], [])
    empty = ({}, {}, {}, ([], []))
    try:
        # List 1 — chunk-level: join back to chunks for file_id; rank is
        # implicit BM25. First occurrence per file = best rank.
        rows = database.fts_rows(
            "SELECT c.file_id AS file_id, c.id AS chunk_id, rank "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_q, limit),
        )
        chunk_fids: list[int] = []
        best_chunk: dict[int, int] = {}
        for r in rows:
            fid = r["file_id"]
            if fid not in best_chunk:
                best_chunk[fid] = r["chunk_id"]
                chunk_fids.append(fid)
        # List 2 — file-level: path/OCR text (rowid IS files.id).
        rows = database.fts_rows(
            "SELECT rowid AS file_id, rank FROM files_fts "
            "WHERE files_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_q, limit),
        )
        file_fids = [r["file_id"] for r in rows]
        best_rank: dict[int, int] = {fid: pos for pos, fid in enumerate(chunk_fids)}
        for pos, fid in enumerate(file_fids):
            if fid not in best_rank or pos < best_rank[fid]:
                best_rank[fid] = pos
        kw_source = {fid: ("chunk" if fid in best_chunk else "ocr") for fid in best_rank}
        scores = {fid: 60.0 / (60.0 + pos) for fid, pos in best_rank.items()}
        return scores, best_chunk, kw_source, (chunk_fids, file_fids)
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

    # Parallel fan-out: semantic + visual + keyword on 3 threads (kw runs the
    # two MATCH queries). Each retriever degrades to empty on failure so the
    # other engine's results still flow (D-07 / research §3 degradation guard).
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
    kw_scores, kw_chunk, kw_source, (kw_chunk_list, kw_file_list) = kw_detail
    kw_match = fts_quote(expanded)

    # Dense ranked lists: first occurrence per file = best rank (hits arrive
    # cosine-desc). Order (not score) is what RRF consumes.
    sem_scores: dict[int, float] = {}
    sem_ord: dict[int, int] = {}
    sem_order: list[int] = []
    for h in sem_hits:
        fid = h["file_id"]
        if fid not in sem_scores:
            sem_order.append(fid)
        if fid not in sem_scores or h["score"] > sem_scores[fid]:
            sem_scores[fid] = h["score"]
            sem_ord[fid] = h["ordinal"]

    vis_scores: dict[int, float] = {}
    vis_order: list[int] = []
    for h in vis_hits:
        fid = h["file_id"]
        if fid not in vis_scores:
            vis_order.append(fid)
        if fid not in vis_scores or h["score"] > vis_scores[fid]:
            vis_scores[fid] = h["score"]

    # --- RRF fusion (D-03): Σ over the FOUR ranked lists of 1/(k + rank),
    # rank 1-based; dense cosines and BM25 raw scores are discarded entirely.
    rrf: dict[int, float] = {}

    def _rrf_add(ranked: list[int]) -> None:
        for pos, fid in enumerate(ranked):
            rrf[fid] = rrf.get(fid, 0.0) + 1.0 / (RRF_K + pos + 1)

    _rrf_add(sem_order)      # text-dense
    _rrf_add(vis_order)      # image-dense
    _rrf_add(kw_chunk_list)  # BM25-chunks
    _rrf_add(kw_file_list)   # BM25-files

    candidates = set(rrf)
    results = []

    def _fts_snippet(fid: int, is_image: bool) -> tuple[str, str]:
        """Marked snippet for a BM25 hit under the D-06 priority: images
        quote their OCR text (files_fts col 1), documents quote their best
        keyword chunk; the other column is the fallback."""
        try:
            if is_image:
                s = database.file_snippet(fid, kw_match)
                if s:
                    return s, "ocr"
            chunk_id = kw_chunk.get(fid, 0)
            if chunk_id:
                s = database.chunk_snippet(chunk_id, kw_match)
                if s:
                    return s, "chunk"
            if not is_image:
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
        is_img = row["kind"] == "image"
        # Kind filter from rewriter: down-rank mismatched kinds, don't drop.
        penalty = 0.0
        if kinds:
            if "image" in kinds and not is_img and "doc" not in kinds:
                penalty = TIE_SCALE
        # Date filter from rewriter ("last month", month names): soft penalty
        # instead of hard drop (keeps recall when mtime is off).
        if dfrom and dto and not (dfrom <= row["mtime"] <= dto):
            penalty += TIE_SCALE
        # Tie-breakers only: recency scaled into ±0.05 so it can reorder
        # near-equal RRF sums but never overturn a real rank difference.
        score = rrf[fid] + TIE_SCALE * _recency_boost(row["mtime"]) / RECENCY_MAX - penalty

        # matched_via (D-04) from list membership.
        in_dense = fid in sem_scores or fid in vis_scores
        in_kw = fid in kw_scores
        matched_via = "both" if (in_dense and in_kw) else ("semantic" if in_dense else "keyword")

        # Snippet (D-05/D-06): keyword hits get FTS5 <mark> text (OCR for
        # images, best chunk for documents); semantic-only hits get the best
        # dense chunk window (raw OCR first 220 chars for images); final
        # fallback filename. snippet_source feeds the UI "why" line.
        snippet, snippet_source = "", "name"
        if fid in kw_scores:
            snippet, snippet_source = _fts_snippet(fid, is_img)
        if not snippet:
            if is_img and (row["ocr_text"] or ""):
                snippet, snippet_source = row["ocr_text"][:220], "ocr"
            elif sem_ord.get(fid) is not None:
                snippet, snippet_source = _make_snippet(fid, sem_ord[fid], query), "chunk"
        results.append(
            {
                "file_id": fid,
                "path": row["path"],
                "name": row["path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
                "kind": row["kind"],
                "folder": row["folder"],
                "size_bytes": row["size_bytes"],
                "mtime": row["mtime"],
                "score": round(score, 4),
                "matched_via": matched_via,
                # Deprecated numeric match fields: the UI still renders them
                # (ui/src/features/search/index.tsx); kept populated until
                # 03-03 removes the usage, then delete here too.
                "match_semantic": round(sem_scores.get(fid, 0.0), 3),
                "match_visual": round(vis_scores.get(fid, 0.0), 3),
                "match_keyword": round(kw_scores.get(fid, 0.0), 3),
                "snippet": snippet,
                "snippet_source": snippet_source,
            }
        )

    results.sort(key=lambda r: -r["score"])  # stable: ties keep RRF insertion order
    return results[:top_k]
