"""Optional Cohere v2 integration. Only explicit calls send text to the API."""

from __future__ import annotations

import json
import math
import os
import urllib.request


def embed(
    texts,
    *,
    input_type,
    model="embed-v5.0-pro",
    dimension=1024,
    api_key=None,
    opener=None,
):
    if input_type not in {"search_document", "search_query"}:
        raise ValueError("input_type must distinguish documents from queries")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or not 1 <= dimension <= 65536
    ):
        raise ValueError("invalid dimension")
    texts = list(texts)
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("texts must be nonempty strings")
    key = api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
    if not key:
        raise ValueError("Set COHERE_API_KEY or CO_API_KEY to call Cohere")
    opener = opener or urllib.request.urlopen
    rows = []
    for start in range(0, len(texts), 96):
        batch = texts[start : start + 96]
        payload = {
            "texts": batch,
            "model": model,
            "input_type": input_type,
            "output_dimension": dimension,
            "embedding_types": ["float"],
            "truncate": "NONE",
        }
        req = urllib.request.Request(
            "https://api.cohere.com/v2/embed",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with opener(req, timeout=120) as response:
            value = json.load(response)
        vectors = value.get("embeddings", {}).get("float")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise ValueError("Cohere returned an unexpected float embedding response")
        for row in vectors:
            if (
                not isinstance(row, list)
                or len(row) != dimension
                or not all(
                    isinstance(x, (int, float)) and math.isfinite(x) for x in row
                )
            ):
                raise ValueError(
                    "Cohere returned a dimension mismatch or invalid vector"
                )
        rows.extend(vectors)
    return rows
