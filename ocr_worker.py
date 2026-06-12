#!/usr/bin/env python3
"""Cloud OCR worker: processes PDFs from ocr-queue/ dirs (extracted from a
collection artifact) with tesseract (eng + Indic languages).

Writes judgments/hc/year=YYYY/<court>__<bench>.ocr.jsonl mirroring the local
ocr_pass.py --queue layout. Run inside GitHub Actions.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import fitz

INDIC_LANGS = "eng+hin+tam+tel+kan+mal+guj+pan+ben+ori+mar+urd+asm+nep"

_COMMON_EN = set("the of and in to for is by on that this court order petition "
                 "with as be no any may have not".split())

SCRIPT_RANGES = [
    ("hi", 0x0900, 0x097F), ("bn", 0x0980, 0x09FF), ("pa", 0x0A00, 0x0A7F),
    ("gu", 0x0A80, 0x0AFF), ("or", 0x0B00, 0x0B7F), ("ta", 0x0B80, 0x0BFF),
    ("te", 0x0C00, 0x0C7F), ("kn", 0x0C80, 0x0CFF), ("ml", 0x0D00, 0x0D7F),
    ("ur", 0x0600, 0x06FF),
]


def detect_lang(text):
    counts = {}
    for code, lo, hi in SCRIPT_RANGES:
        counts[code] = sum(1 for c in text if lo <= ord(c) <= hi)
    best = max(counts, key=counts.get)
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return best if counts[best] > latin * 0.5 and counts[best] > 50 else "en"


def english_score(text):
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 8:
        return None
    return sum(1 for w in words if w in _COMMON_EN) / len(words)


def tess(png, langs):
    r = subprocess.run(["tesseract", png, "-", "--psm", "6", "-l", langs],
                       capture_output=True, text=True, timeout=300)
    return r.stdout


def ocr_pdf(pdf_path, dpi=200, max_pages=80):
    with open(pdf_path, "rb") as f:
        if f.read(5) != b"%PDF-":
            return ""
    doc = fitz.open(pdf_path)
    texts = []
    with tempfile.TemporaryDirectory() as td:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            png = os.path.join(td, f"p{i}.png")
            page.get_pixmap(dpi=dpi).save(png)
            t = tess(png, "eng")
            s = english_score(t)
            if s is not None and s < 0.05:  # likely non-English page
                t2 = tess(png, INDIC_LANGS)
                if len(t2.strip()) > len(t.strip()):
                    t = t2
            texts.append(t)
    doc.close()
    t = "\n".join(texts)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = 0
    benches = sorted(d for d in os.listdir(args.queue)
                     if os.path.isdir(os.path.join(args.queue, d)))
    for b in benches:
        try:
            year, court, bench = b.split("__", 2)
        except ValueError:
            continue
        bdir = os.path.join(args.queue, b)
        pdfs = sorted(f for f in os.listdir(bdir) if f.endswith(".pdf"))
        if not pdfs:
            continue
        out_dir = os.path.join(args.out, "judgments/hc", f"year={year}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{court}__{bench}.ocr.jsonl")
        print(f"{b}: {len(pdfs)} PDFs", flush=True)
        with open(out_path, "a") as out_f:
            for name in pdfs:
                t0 = time.time()
                try:
                    text = ocr_pdf(os.path.join(bdir, name))
                except Exception as e:
                    print(f"  ERR {name}: {e}", flush=True)
                    text = ""
                if len(text) >= 40:
                    rec = {"id": f"hc/{bench}/{os.path.splitext(name)[0]}",
                           "court": court, "bench": bench, "year": int(year),
                           "ocr": "tesseract", "lang": detect_lang(text),
                           "text": text}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                os.remove(os.path.join(bdir, name))
                n += 1
                if n % 50 == 0:
                    print(f"  [{n}] {name}: {len(text)} chars "
                          f"in {time.time()-t0:.0f}s", flush=True)
    print("OCR DONE", n, flush=True)


if __name__ == "__main__":
    main()
