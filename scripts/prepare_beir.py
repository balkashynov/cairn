#!/usr/bin/env python3
"""Fetch a pinned public Cohere BEIR corpus, queries and qrels. No API key required."""

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REVISION = "f018922a46a51388348a307ec2fef28019a33026"
BASE = f"https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/{REVISION}"


def download(url, path):
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as src, tmp.open("wb") as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", choices=["scifact", "fiqa", "cqadupstack-programmers", "trec-covid"]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or Path("data") / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "model": "embed-english-v3.0",
        "revision": REVISION,
        "source": "https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3",
        "files": {},
    }
    for kind, filename in [("corpus", "0000"), ("queries", "test"), ("qrels", "test")]:
        url = f"{BASE}/{args.dataset}/{kind}/{filename}.parquet"
        path = out / f"{kind}.parquet"
        download(url, path)
        with path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["files"][kind] = {
            "url": url,
            "sha256": sha256,
            "bytes": path.stat().st_size,
        }
        table = pq.read_table(path)
        if kind == "qrels":
            (out / "qrels.json").write_text(json.dumps(table.to_pylist()))
            continue
        vectors = np.array(table["emb"].to_pylist(), dtype="<f4")
        with (out / f"{kind}.fbin").open("wb") as f:
            f.write(struct.pack("<II", *vectors.shape))
            vectors.tofile(f)
        metadata = table.drop(["emb"]).to_pylist()
        with (out / f"{kind}.jsonl").open("w") as f:
            for row in metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest[kind] = {"rows": vectors.shape[0], "dimensions": vectors.shape[1]}
        print(kind, vectors.shape, flush=True)
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2))
    print(out / "provenance.json")


if __name__ == "__main__":
    main()
