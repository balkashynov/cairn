import io
import json
import subprocess

import numpy as np
import pytest
from cairn_search import Cairn, find_binary, write_fbin
from cairn_search.audit import audit, ndcg, percentile, read_fbin
from cairn_search.cohere import embed
from cairn_search.report import write_report


@pytest.fixture
def index(tmp_path):
    rng = np.random.default_rng(20261014)
    x = rng.normal(size=(257, 65)).astype("float32")
    write_fbin(tmp_path / "x.fbin", x)
    subprocess.run(
        [find_binary(), "build", str(tmp_path / "x.fbin"), str(tmp_path / "x.cairn")],
        check=True,
        capture_output=True,
    )
    return tmp_path / "x.cairn", x


def test_numpy_independent_oracle_and_worker_recovery(index):
    path, x = index
    # Match the documented stored-f32 normalization; compute dots independently in NumPy.
    norm = (
        x.astype("float64") / np.linalg.norm(x.astype("float64"), axis=1)[:, None]
    ).astype("float32")
    q = np.random.default_rng(817).normal(size=(30, x.shape[1])).astype("float32")
    with Cairn(path, verify=True) as client:
        with pytest.raises(ValueError):
            client.search([0.0] * x.shape[1])
        with pytest.raises(ValueError):
            client.search([1.0, 0.0])
        with pytest.raises(ValueError):
            client.search(q[0], mode="unrecognized")
        for row in q:
            qn = (row.astype("float64") / np.linalg.norm(row.astype("float64"))).astype(
                "float32"
            )
            scores = norm.astype("float64") @ qn.astype("float64")
            ids = np.lexsort((np.arange(len(x)), -scores))[:10]
            out = client.search(row)
            assert [h["id"] for h in out["hits"]] == ids.tolist()
            assert np.allclose(
                [h["score"] for h in out["hits"]], scores[ids], rtol=0, atol=2e-15
            )
    client.close()
    with pytest.raises(RuntimeError):
        client.search(q[0])


def test_streaming_export_rejects_invalid_without_clobber(tmp_path):
    out = tmp_path / "rows.fbin"
    for rows in [[], [[0, 0]], [[1, 2], [3]], [[1, float("nan")]], [[1e100, 1]]]:
        with pytest.raises((ValueError, OverflowError)):
            write_fbin(out, rows)
        assert not out.exists()
    write_fbin(out, [[1, 2], [2, 1]])
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        write_fbin(out, [[3, 4]])
    assert out.read_bytes() == before
    assert np.array_equal(read_fbin(out), [[1, 2], [2, 1]])


def test_api_contract_batches_query_document_and_validates_response():
    calls = []

    def opener(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        assert request.full_url == "https://api.cohere.com/v2/embed"
        assert request.headers["Authorization"] == "Bearer fixture-key"
        assert timeout == 120
        return io.BytesIO(
            json.dumps(
                {"embeddings": {"float": [[1, 0]] * len(body["texts"])}}
            ).encode()
        )

    rows = embed(
        ["text"] * 97,
        input_type="search_document",
        dimension=2,
        api_key="fixture-key",
        opener=opener,
    )
    assert len(rows) == 97 and [len(c["texts"]) for c in calls] == [96, 1]
    assert all(
        c["embedding_types"] == ["float"] and c["truncate"] == "NONE" for c in calls
    )
    embed(
        ["query"],
        input_type="search_query",
        model="embed-v5.0-fast",
        dimension=2,
        api_key="fixture-key",
        opener=opener,
    )
    assert (
        calls[-1]["input_type"] == "search_query"
        and calls[-1]["model"] == "embed-v5.0-fast"
    )
    with pytest.raises(ValueError):
        embed(
            ["x"],
            input_type="search_query",
            dimension=2,
            api_key="x",
            opener=lambda *_a, **_k: io.BytesIO(b'{"embeddings":{"float":[[1]]}}'),
        )


def test_metrics_use_ranked_relevance_not_neighbor_recall():
    assert ndcg([{"id": 1}, {"id": 2}], {"1": 1, "2": 1}, 10) == 1
    assert ndcg([{"id": 3}, {"id": 1}], {"1": 1}, 2) == pytest.approx(1 / np.log2(3))
    assert ndcg([{"id": 3}], {}, 2) is None
    assert percentile([0, 10], 0.95) == 9.5


def test_report_escapes_script_and_does_not_execute_metadata(tmp_path):
    path = tmp_path / "audit.html"
    attack = '</script><script>alert("x")</script>'
    write_report({"query": attack}, path)
    text = path.read_text()
    assert attack not in text
    assert "\\u003c/script\\u003e" in text
    assert "innerHTML" not in text


def test_audit_validates_counts_and_records_all_methods(index, tmp_path):
    path, x = index
    write_fbin(tmp_path / "q.fbin", x[:4] + np.float32(0.01))
    result = audit(path, tmp_path / "q.fbin", k=5, repeats=1, oversamples=(1, 4))
    assert result["summary"]["certified"]["recall_at_k"] == 1
    assert len(result["queries"]) == 4
    assert result["summary"]["certified"]["timed_samples"] == 4
    with pytest.raises(ValueError):
        audit(path, tmp_path / "q.fbin", repeats=0)


def test_cli_rejects_extra_flags(index, tmp_path):
    path, _ = index
    for args in [
        ["inspect", str(path), "unexpected"],
        ["verify", str(path), "--unknown"],
        ["build", "missing", "missing", "--unknown", "1"],
    ]:
        result = subprocess.run(
            [find_binary(), *args], check=False, capture_output=True
        )
        assert result.returncode != 0


def test_external_ann_run_distinguishes_ranking_loss_from_candidate_loss(
    index, tmp_path
):
    from cairn_search.compare import compare_run

    path, x = index
    qfile = tmp_path / "queries.fbin"
    write_fbin(qfile, x[:2])
    with Cairn(path) as client:
        gold = [[h["id"] for h in client.search(q, k=5)["hits"]] for q in x[:2]]
    candidate = tmp_path / "run.jsonl"
    candidate.write_text(
        json.dumps({"query_id": 0, "ids": gold[0][::-1]})
        + "\n"
        + json.dumps({"query_id": 1, "ids": gold[1][1:]})
        + "\n"
    )
    report = compare_run(path, qfile, candidate, k=5)
    assert report["results"][0]["candidate_recall_at_k"] == 1
    assert not report["results"][0]["ordered_exact"]
    assert report["results"][1]["candidate_recall_at_k"] == 0.8
    assert report["summary"]["queries_with_missing_candidates"] == 1
    candidate.write_text(json.dumps({"query_id": 0, "ids": [1, 1]}) + "\n")
    with pytest.raises(ValueError):
        compare_run(path, qfile, candidate, k=5)


def test_jsonl_unicode_line_separators_do_not_split_records(index, tmp_path):
    path, x = index
    qfile = tmp_path / "q.fbin"
    write_fbin(qfile, x[:1])
    qm = tmp_path / "q.jsonl"
    qm.write_text(
        json.dumps({"_id": "q", "text": "before\u2028after"}, ensure_ascii=False) + "\n"
    )
    cm = tmp_path / "c.jsonl"
    cm.write_text(
        "".join(
            json.dumps({"_id": str(i), "text": "first\u0085second"}, ensure_ascii=False)
            + "\n"
            for i in range(len(x))
        )
    )
    r = audit(
        path, qfile, repeats=1, oversamples=(1,), query_metadata=qm, corpus_metadata=cm
    )
    assert r["queries"][0]["query"]["text"] == "before\u2028after"
    assert r["documents"]["0"]["text"] == "first\u0085second"
