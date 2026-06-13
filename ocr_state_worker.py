#!/usr/bin/env python3
"""Cloud OCR worker for scanned STATE ACT PDFs.

Input layout (committed to repo under ocr-queue-state/):
    ocr-queue-state/<state>/<slug>.pdf
    ocr-queue-state/<state>/<slug>.meta.json   # act metadata sidecar

Output (mirrors statutes-state/<state>/<slug>.json):
    out/statutes-state/<state>/<slug>.json  -> [ {id, act, act_slug, state,
        section:"full", title, text, lang, ocr:"tesseract", source} ]

Run inside GitHub Actions (tesseract eng + Indic packs installed).
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time

import fitz

INDIC_LANGS = "eng+hin+tam+tel+kan+mal+guj+pan+ben+ori+mar+urd+asm+nep"
_COMMON_EN = set("the of and in to for is by on that this act section state "
                 "shall be no any may have not government".split())

SCRIPT_RANGES = [
    ("hi", 0x0900, 0x097F), ("bn", 0x0980, 0x09FF), ("pa", 0x0A00, 0x0A7F),
    ("gu", 0x0A80, 0x0AFF), ("or", 0x0B00, 0x0B7F), ("ta", 0x0B80, 0x0BFF),
    ("te", 0x0C00, 0x0C7F), ("kn", 0x0C80, 0x0CFF), ("ml", 0x0D00, 0x0D7F),
    ("ur", 0x0600, 0x06FF),
]


def detect_lang(text):
    counts = {c: sum(1 for ch in text if lo <= ord(ch) <= hi)
              for c, lo, hi in SCRIPT_RANGES}
    best = max(counts, key=counts.get)
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
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


def ocr_pdf(pdf_path, dpi=200, max_pages=200):
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
            if s is not None and s < 0.05:
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
    ap.add_argument("--states", default="", help="comma list; default all")
    args = ap.parse_args()

    want = set(args.states.split(",")) if args.states else None
    n = ok = 0
    states = sorted(d for d in os.listdir(args.queue)
                    if os.path.isdir(os.path.join(args.queue, d))
                    and (want is None or d in want))
    for state in states:
        sdir = os.path.join(args.queue, state)
        pdfs = sorted(f for f in os.listdir(sdir)
                      if f.endswith(".pdf") and not f.startswith("._"))
        if not pdfs:
            continue
        out_dir = os.path.join(args.out, "statutes-state", state)
        os.makedirs(out_dir, exist_ok=True)
        print(f"{state}: {len(pdfs)} PDFs", flush=True)
        for name in pdfs:
            slug = os.path.splitext(name)[0]
            meta_path = os.path.join(sdir, slug + ".meta.json")
            meta = {}
            if os.path.exists(meta_path):
                try:
                    meta = json.load(open(meta_path))
                except Exception:
                    pass
            t0 = time.time()
            try:
                text = ocr_pdf(os.path.join(sdir, name))
            except Exception as e:
                print(f"  ERR {name}: {e}", flush=True)
                text = ""
            n += 1
            if len(text) >= 40:
                rec = {
                    "id": f"{state}/{slug}/full",
                    "act": meta.get("name", slug),
                    "act_slug": slug,
                    "state": state,
                    "section": "full",
                    "title": meta.get("name", ""),
                    "text": text,
                    "lang": detect_lang(text),
                    "ocr": "tesseract",
                    "act_number": meta.get("act_number", ""),
                    "enactment_date": meta.get("enactment_date", ""),
                    "source_url": meta.get("source", ""),
                }
                with open(os.path.join(out_dir, slug + ".json"), "w") as f:
                    json.dump([rec], f, ensure_ascii=False)
                ok += 1
            if n % 25 == 0:
                print(f"  [{n}] {slug}: {len(text)} chars "
                      f"in {time.time()-t0:.0f}s lang={detect_lang(text)}",
                      flush=True)
    print(f"STATE OCR DONE processed={n} written={ok}", flush=True)


if __name__ == "__main__":
    main()
