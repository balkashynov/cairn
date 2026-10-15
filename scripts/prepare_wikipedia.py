#!/usr/bin/env python3
"""Prepare Qdrant's public Cohere Wikipedia benchmark without model inference."""

import hashlib
import json
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

url = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/cohere-wiki-100k-no-filters.tgz"
root = Path("data")
root.mkdir(exist_ok=True)
archive = root / "cohere-wiki-100k.tgz"
out = root / "cohere-wiki-100k"
if not archive.exists():
    with (
        urllib.request.urlopen(url, timeout=120) as src,
        archive.with_suffix(".partial").open("wb") as dst,
    ):
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    archive.with_suffix(".partial").replace(archive)
# Pin the actual observed download. Fail rather than quietly compare changed data.
with archive.open("rb") as stream:
    sha = hashlib.file_digest(stream, "sha256").hexdigest()
expected = "6fd59b125ff9326bab10d572c125a3ebbf79be16c74ed9163385fea4139446ae"
if sha != expected:
    raise ValueError("archive checksum mismatch: " + sha)
out.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
    tar.extractall(out, filter="data")
x = np.load(out / "vectors.npy", mmap_mode="r", allow_pickle=False)
q = []
with (out / "tests.jsonl").open() as f:
    for line in f:
        q.append(json.loads(line)["query"])
        if len(q) == 200:
            break
for name, a in [("corpus", x), ("queries", np.array(q, dtype="<f4"))]:
    with (out / f"{name}.fbin").open("wb") as f:
        f.write(struct.pack("<II", *a.shape))
        np.asarray(a, dtype="<f4").tofile(f)
(out / "provenance.json").write_text(
    json.dumps(
        {
            "source": url,
            "archive_sha256": sha,
            "rows": len(x),
            "dimensions": x.shape[1],
            "queries": len(q),
            "model": "historical Cohere Wikipedia embeddings; exact version not established by archive",
        },
        indent=2,
    )
)
print(out)
