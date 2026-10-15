from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import threading
from collections.abc import Iterable
from pathlib import Path


def find_binary(binary=None) -> str:
    candidates = [
        binary,
        os.environ.get("CAIRN_BIN"),
        shutil.which("cairn"),
        Path(__file__).resolve().parents[2] / "target/release/cairn",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "Build with cargo build --release, install with cargo install --path ., or set CAIRN_BIN"
    )


def write_fbin(path, rows: Iterable[Iterable[float]]) -> tuple[int, int]:
    """Stream f32 rows into a new fbin. Refuse overwrite; remove partial output on error."""
    path = Path(path)
    created = False
    try:
        with path.open("xb") as f:
            created = True
            f.write(b"\0" * 8)
            count, dim = 0, None
            for row in rows:
                row = list(row)
                if dim is None:
                    dim = len(row)
                    if not 1 <= dim <= 65536:
                        raise ValueError("dimension must be in 1..=65536")
                if len(row) != dim or not all(math.isfinite(x) for x in row):
                    raise ValueError("ragged or non-finite vector")
                if not any(row):
                    raise ValueError("zero vector has no cosine direction")
                f.write(struct.pack(f"<{dim}f", *row))
                count += 1
            if not count or count > 2**32 - 1:
                raise ValueError("row count must be in 1..=u32::MAX")
            f.seek(0)
            f.write(struct.pack("<II", count, dim))
        return count, dim
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


class Cairn:
    """Persistent local worker; no listening socket, remote service, or API key.

    Calls are serialized per client. Separate clients can map one immutable index.
    Never modify/truncate an index while any client has it open.
    """

    def __init__(self, path, *, binary=None, verify=False):
        self.binary = find_binary(binary)
        self.path = str(Path(path).resolve())
        if verify:
            subprocess.run(
                [self.binary, "verify", self.path, "--deep"],
                check=True,
                capture_output=True,
            )
        checked = subprocess.run(
            [self.binary, "inspect", self.path],
            check=True,
            capture_output=True,
            text=True,
        )
        self.info = json.loads(checked.stdout)
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [self.binary, "serve", self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def search(self, query, *, k=10, mode="certified", oversample=4, tag=None):
        # tolist() accepts NumPy without making it a client dependency.
        query = query.tolist() if hasattr(query, "tolist") else list(query)
        payload = json.dumps(
            {
                "query": query,
                "k": k,
                "mode": mode,
                "oversample": oversample,
                "tag": tag,
            },
            allow_nan=False,
        )
        if len(payload.encode()) > 4 * 1024 * 1024 - 1:
            raise ValueError("query exceeds worker frame limit")
        with self._lock:
            proc = self._process
            if proc is None or proc.poll() is not None:
                raise RuntimeError("Cairn worker is closed")
            try:
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except BrokenPipeError as exc:
                raise RuntimeError("Cairn worker exited") from exc
            if not line:
                raise RuntimeError(
                    "Cairn worker returned EOF: " + proc.stderr.read().strip()
                )
            result = json.loads(line)
            if "error" in result:
                raise ValueError(result["error"])
            return result

    def close(self):
        with self._lock:
            proc, self._process = self._process, None
            if proc is not None:
                proc.stdin.close()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                proc.stdout.close()
                proc.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
