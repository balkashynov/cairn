"""No downloads or API key. Show a real fixed-candidate failure and exact recovery."""

import json
import subprocess
import tempfile
from pathlib import Path

from cairn_search import Cairn, find_binary, write_fbin

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    # Every vector has the same binary signs. Only the final one points at the query.
    documents = [[0.001, 1.0] for _ in range(99)] + [[1.0, 0.001]]
    write_fbin(folder / "documents.fbin", documents)
    subprocess.run(
        [
            find_binary(),
            "build",
            str(folder / "documents.fbin"),
            str(folder / "index.cairn"),
        ],
        check=True,
        capture_output=True,
    )
    with Cairn(folder / "index.cairn") as index:
        for mode in ["binary", "certified"]:
            result = index.search([1.0, 0.001], k=1, mode=mode, oversample=4)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "nearest_row": result["hits"][0]["id"],
                        "score": result["hits"][0]["score"],
                        "full_vectors_scored": result["stats"]["exact_scored"],
                        "certified": result["stats"]["certified"],
                    }
                )
            )
    assert result["hits"][0]["id"] == 99
