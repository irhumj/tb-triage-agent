"""
rag/build_index.py - turn the WHO screening guideline PDF into searchable passages.

Steps:
  1. Read every page of the PDF (PyMuPDF).
  2. Label each passage with its section, using the PDF's own table of contents
     (a page can hold the end of one section and the start of the next).
  3. Drop pages that are not guidance (contents, acknowledgements, abbreviations, references).
  4. Remove the running header on each page (chapter title + printed page number).
  5. Split the text into passages of about 220 words, overlapping by about 40 words,
     so a sentence cut at a boundary still appears whole in one of the two passages.
  6. Save the passages to rag/chunks.json. search.py reads that file.

Run once from the project folder:   python rag/build_index.py
"""

import glob
import json
import os
import re

import pymupdf as fitz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chunks.json")
CHUNK_WORDS, OVERLAP_WORDS = 220, 40
SKIP_SECTIONS = ("contents", "acknowledgements", "abbreviations", "references")
SOURCE = ("WHO consolidated guidelines on tuberculosis. Module 2: Screening - "
          "systematic screening for tuberculosis disease (2021)")


def find_pdf():
    pdfs = glob.glob(os.path.join(HERE, "*.pdf"))
    if not pdfs:
        raise FileNotFoundError("Put the WHO Module 2 PDF in the rag/ folder first.")
    return pdfs[0]


def clean_page(text):
    """Drop the running header: 1-2 title lines followed by a printed page number (5, 25, ix...)."""
    lines = [l.strip() for l in text.splitlines()]
    for n in (1, 2):                                          # odd pages: 1 title line; even pages: 2
        if len(lines) > n and re.fullmatch(r"[0-9]+|[ivxlc]+", lines[n].lower()):
            lines = lines[n + 1:]
            break
    text = " ".join(l for l in lines if l)
    text = re.sub(r"-\s+(?=[a-z])", "", text)                # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def split_by_headings(text, headings):
    """Several sections can start on one page: cut the page text where each heading appears.

    Returns (heading_or_None, text) pieces; None = text that still belongs to the previous section.
    """
    pieces, low = [], text.lower()
    marks = sorted((low.find(norm(h)), h) for h in headings if low.find(norm(h)) >= 0)
    start, current = 0, None
    for pos, h in marks:
        if pos > start:
            pieces.append((current, text[start:pos]))
        start, current = pos, h
    pieces.append((current, text[start:]))
    return [(h, t.strip()) for h, t in pieces if t.strip()]


def chunk(words, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    step = size - overlap
    for start in range(0, max(len(words) - overlap, 1), step):
        yield " ".join(words[start:start + size])


def main():
    path = find_pdf()
    doc = fitz.open(path)
    toc = doc.get_toc()                                       # [level, title, page]
    starts = {}
    for _, title, page in toc:
        starts.setdefault(page, []).append(title.strip())

    chunks, section = [], "Front matter"
    for page_no in range(1, doc.page_count + 1):
        text = clean_page(doc[page_no - 1].get_text())
        for heading, piece in split_by_headings(text, starts.get(page_no, [])):
            if heading:
                section = heading
            if section.lower().startswith(SKIP_SECTIONS) or "references" in section.lower():
                continue
            words = piece.split()
            if len(words) < 30:                               # headings alone or near-blank text
                continue
            for t in chunk(words):
                chunks.append({"id": len(chunks), "section": section, "pdf_page": page_no, "text": t})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": SOURCE, "file": os.path.basename(path), "chunks": chunks}, f,
                  ensure_ascii=False, indent=1)
    print(f"Read {doc.page_count} pages -> {len(chunks)} passages saved to {OUT}")


if __name__ == "__main__":
    main()
