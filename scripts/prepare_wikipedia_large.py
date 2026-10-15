#!/usr/bin/env python3
"""Stream the 3.1 GB public million-vector benchmark, retaining only needed files.

Needs approximately 10 GB free during conversion/build. No model API is called.
"""

import hashlib
import json
import shutil
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = (
    "https://storage.googleapis.com/ann-filtered-benchmark/datasets/cohere-wiki-1m.tgz"
)
out = Path("data/cohere-wiki-1m")
out.mkdir(parents=True, exist_ok=True)
if not (out / "corpus.fbin").exists():
    if not (out / "vectors.npy").exists():
        with urllib.request.urlopen(URL, timeout=180) as response:
            headers = {
                k: response.headers.get(k)
                for k in ["Content-Length", "ETag", "Last-Modified"]
            }
            with tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    name = Path(member.name).name
                    if (
                        name not in {"vectors.npy", "tests.jsonl"}
                        or not member.isfile()
                    ):
                        continue
                    destination = out / name
                    if destination.exists():
                        raise ValueError("duplicate member: " + name)
                    print("Downloading", name, member.size, flush=True)
                    temporary = out / (name + ".partial")
                    with (
                        archive.extractfile(member) as source,
                        temporary.open("wb") as target,
                    ):
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    temporary.replace(destination)
        (out / "download.json").write_text(
            json.dumps({"url": URL, "headers": headers}, indent=2)
        )
    vectors = np.load(out / "vectors.npy", mmap_mode="r", allow_pickle=False)
    with (out / "corpus.fbin").open("wb") as f:
        f.write(struct.pack("<II", *vectors.shape))
        np.asarray(vectors, dtype="<f4").tofile(f)
    shape = vectors.shape
    del vectors
    # This script downloaded this duplicate; retain the portable fbin instead.
    (out / "vectors.npy").unlink()
else:
    with (out / "corpus.fbin").open("rb") as f:
        shape = struct.unpack("<II", f.read(8))
queries = []
with (out / "tests.jsonl").open() as f:
    for line in f:
        queries.append(json.loads(line)["query"])
        if len(queries) == 100:
            break
q = np.asarray(queries, dtype="<f4")
with (out / "queries.fbin").open("wb") as f:
    f.write(struct.pack("<II", *q.shape))
    q.tofile(f)
manifest = {
    "source": URL,
    "rows": shape[0],
    "dimensions": shape[1],
    "queries": len(q),
    "model": "historical Cohere Wikipedia vectors; exact model version/query construction not established by archive",
    "sha256": {},
}
for filename in ["corpus.fbin", "queries.fbin", "tests.jsonl"]:
    with (out / filename).open("rb") as f:
        manifest["sha256"][filename] = hashlib.file_digest(f, "sha256").hexdigest()
(out / "provenance.json").write_text(json.dumps(manifest, indent=2))
print(manifest, flush=True)
