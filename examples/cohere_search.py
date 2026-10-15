"""Explicit live API example: sends these sample texts to Cohere using your API key."""

import subprocess
import tempfile
from pathlib import Path

from cairn_search import Cairn, find_binary, write_fbin
from cairn_search.cohere import embed

model = "embed-v5.0-pro"
documents = [
    "The Rust borrow checker tracks ownership and reference lifetimes.",
    "Write-ahead logging recovers committed database transactions after crashes.",
    "Memory mapping lets the OS page file contents into a process address space.",
]
vectors = embed(documents, input_type="search_document", model=model, dimension=1024)
query = embed(
    ["How can a database survive a power failure?"],
    input_type="search_query",
    model=model,
    dimension=1024,
)[0]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    write_fbin(root / "documents.fbin", vectors)
    subprocess.run(
        [
            find_binary(),
            "build",
            str(root / "documents.fbin"),
            str(root / "index.cairn"),
        ],
        check=True,
    )
    with Cairn(root / "index.cairn") as index:
        for hit in index.search(query, k=2)["hits"]:
            print(hit["score"], documents[hit["id"]])
