"""
rag/search.py - find the WHO guideline passages that best match a question.

Method: BM25, a classic keyword-ranking formula used by search engines.
  - A passage scores high when it contains the query's words, especially RARE words
    (e.g. "computer-aided", "HIV") rather than common ones (e.g. "screening").
  - Repeating a word helps, but with diminishing returns; long passages are not favoured.
Why BM25 here: guideline questions use exact technical terms (CXR, CAD, mWRD, HIV),
it needs no model download, runs instantly on a CPU, and every score is explainable.

Usage:
    from search import search
    search("computer-aided detection for chest X-ray", k=3)
"""

import json
import os
import re

from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "chunks.json")

STOP = set("""a an and are as at be by can for from has have in is it of on or that the this to
was were which with who what when how should does do not may will their they these those""".split())


def tokenize(text):
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    out = []
    for w in words:
        if w in STOP:
            continue
        out.append(w)
        if "-" in w:                      # "computer-aided" also matches "computer" and "aided"
            out.extend(w.split("-"))
    return out


if not os.path.exists(CHUNKS):
    raise ImportError("rag/chunks.json not found. Run: python rag/build_index.py")

with open(CHUNKS, encoding="utf-8") as f:
    _data = json.load(f)
_chunks = _data["chunks"]
_bm25 = BM25Okapi([tokenize(c["text"]) for c in _chunks])


def search(query, k=3):
    """Return the k best passages, each with its section, PDF page and BM25 score."""
    scores = _bm25.get_scores(tokenize(query))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "rank": r + 1,
            "score": round(float(scores[i]), 2),
            "source": _data["source"],
            "section": _chunks[i]["section"],
            "pdf_page": _chunks[i]["pdf_page"],
            "text": _chunks[i]["text"],
        }
        for r, i in enumerate(best)
        if scores[i] > 0
    ]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "computer-aided detection chest radiography"
    for hit in search(q):
        print(f"\n#{hit['rank']}  score {hit['score']}  |  {hit['section']}  |  PDF page {hit['pdf_page']}")
        print(hit["text"][:400], "...")
