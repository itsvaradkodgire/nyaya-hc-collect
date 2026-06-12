#!/usr/bin/env python3
"""Collect Indian High Court judgments from the AWS Open Data bucket
(ecourts-derived, all High Courts) as clean JSONL, WITHOUT keeping raw PDFs.

Raw HC data exceeds 1 TB, so each tar is downloaded, converted to text and
then DELETED. Only the cleaned JSONL stays. Peak disk = one tar.

Output (under --out):
  judgments/hc/year=YYYY/<court>__<bench>.jsonl

Record: {"id": "hc/<bench>/<basename>", "court": "...", "bench": "...",
         "year": YYYY, "source": "s3-key", "text": "..."}

Resumable: an existing .jsonl skips that bench-year. Newest years first.
"""
import argparse
import json
import sys
import os
import re
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # pymupdf

BUCKET = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"
UA = {"User-Agent": "nyaya-research/1.0 (legal NLP corpus builder)"}


def http_get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


def list_keys(prefix):
    keys, tok = [], None
    while True:
        u = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if tok:
            u += "&continuation-token=" + urllib.parse.quote(tok)
        h = http_get(u).decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", h)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", h)
        if not m:
            break
        tok = m.group(1)
    return keys


def download(url, dest):
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=1800) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def clean_text(t):
    t = t.replace("\u00ad", "")
    t = re.sub(r"-\n(?=[a-z])", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def pdf_to_text(data):
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        t = "\n".join(p.get_text() for p in doc)
        doc.close()
        return clean_text(t)
    except Exception:
        return ""


COURT_NAMES = {}  # filled lazily from bench slug; slug is good enough


def process_tar(tar_path, out_f, court, bench, year, src_prefix, skipped, ocr_dir=None):
    n_ok = n_bad = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.name.lower().endswith(".pdf"):
                continue
            base = os.path.splitext(os.path.basename(m.name))[0]
            data = tf.extractfile(m).read()
            text = pdf_to_text(data)
            if len(text) < 200:  # scanned/no text layer -> save PDF for OCR pass
                n_bad += 1
                skipped.append(f"{src_prefix}/{m.name}")
                if ocr_dir and data[:5] == b"%PDF-":
                    os.makedirs(ocr_dir, exist_ok=True)
                    with open(os.path.join(ocr_dir, os.path.basename(m.name)), "wb") as pf:
                        pf.write(data)
                continue
            rec = {
                "id": f"hc/{bench}/{base}",
                "court": court,
                "bench": bench,
                "year": year,
                "source": f"{src_prefix}/{m.name}",
                "text": text,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    return n_ok, n_bad


def process_bench(task):
    """Worker: one bench-year end-to-end (download tars -> jsonl -> delete)."""
    year, court, bench, out_root, tmp_dir = task
    out_dir = os.path.join(out_root, "judgments/hc", f"year={year}")
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, f"{court}__{bench}.jsonl")
    if os.path.exists(jl):
        return f"{year} {court}/{bench}: already done"
    prefix = f"data/tar/year={year}/court={court}/bench={bench}/"
    tars = [k for k in list_keys(prefix) if k.endswith(".tar")]
    t0, total_ok, total_bad = time.time(), 0, 0
    skipped = []
    tmp_out = jl + ".part"
    with open(tmp_out, "w") as out_f:
        for key in tars:
            local = os.path.join(tmp_dir, f"{os.getpid()}_{os.path.basename(key)}")
            try:
                download(f"{BUCKET}/{key}", local)
                ok, bad = process_tar(local, out_f, court, bench, year,
                                      f"{BUCKET}/{os.path.dirname(key)}", skipped,
                                      ocr_dir=os.path.join(out_root, "ocr-queue", f"{year}__{court}__{bench}"))
                total_ok += ok
                total_bad += bad
            except Exception as e:
                print(f"  ERR {key}: {e}", flush=True)
            finally:
                for pth in (local, local + ".part"):
                    if os.path.exists(pth):
                        os.remove(pth)
    os.replace(tmp_out, jl)
    if skipped:
        with open(jl.replace(".jsonl", ".noocr.txt"), "w") as f:
            f.write("\n".join(skipped) + "\n")
    return (f"{year} {court}/{bench}: {total_ok} judgments "
            f"({total_bad} no-text) in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-year", type=int, default=1950)
    ap.add_argument("--to-year", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tmp", default=None, help="scratch dir for tars (default: <out>/tmp)")
    ap.add_argument("--shard", default=None, help="i/N: process only shard i (0-based) of N")
    args = ap.parse_args()

    tmp_dir = args.tmp or os.path.join(args.out, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # enumerate all bench tasks, newest year first
    tasks = []
    for year in range(args.to_year, args.from_year - 1, -1):
        h = http_get(f"{BUCKET}/?list-type=2&delimiter=/&prefix=data/tar/year={year}/").decode()
        courts = re.findall(rf"<Prefix>data/tar/year={year}/court=([^/]+)/</Prefix>", h)
        for court in courts:
            h2 = http_get(f"{BUCKET}/?list-type=2&delimiter=/&prefix=data/tar/year={year}/court={court}/").decode()
            for bench in re.findall(rf"<Prefix>data/tar/year={year}/court={court}/bench=([^/]+)/</Prefix>", h2):
                jl = os.path.join(args.out, "judgments/hc", f"year={year}", f"{court}__{bench}.jsonl")
                if not os.path.exists(jl):
                    tasks.append((year, court, bench, args.out, tmp_dir))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        tasks = [t for k, t in enumerate(sorted(tasks)) if k % n == i]
    print(f"{len(tasks)} bench-year tasks, {args.workers} workers", flush=True)

    errs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_bench, t) for t in tasks]
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                errs += 1
                print("WORKER ERR:", e, flush=True)
    if errs == 0:
        print("DONE", flush=True)
    else:
        print(f"FINISHED WITH {errs} ERRORS (not done)", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
