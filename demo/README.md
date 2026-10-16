# Browser presentation

This static site lets reviewers inspect Cairn without installing the engine.
Serve this directory with any static HTTP server; no build or external scripts
are required.

- `index.html`, `style.css`, `app.js`: five-step recorded audit, five-step synthetic bound lesson,
  recorded-query explorer and benchmark comparisons.
- `scifact.json`: 12 selected queries and attributed document excerpts from the
  saved SciFact audit, retaining aggregate statistics over all 300 queries.
- `benchmarks.json`: a copy of `benchmarks/summary.json` across five datasets.
- `audit.html`: the original generated detailed SciFact report.

The main walkthrough follows recorded query 29 from a ten-result binary search through the four missing neighbors, the unchanged relevance score, and a successful fixed-int8 alternative. Other queries and metrics sit inside an expandable explorer.

The separate six-vector walkthrough is synthetic and labeled as an illustration. The
query explorer replays measurements; it does not execute Rust or call an
embedding service in the browser. Exact neighbor agreement is separate from
human relevance. Benchmark views retain Python overhead, p95, batch throughput,
and unfavorable comparisons.

Run `scripts/package_evidence.py` through the documented benchmark workflow to
refresh the JSON and detailed report. It preserves the hand-authored presentation.
GitHub Pages deploys this directory from `codex/cairn-search`.

Useful outreach links:

- `#how-it-works`: five decisions explaining the bounds and exact fallback.
- `#example`: the guided five-step audit, starting with the task.
- `?query=29&method=certified#explorer`: expanded detailed query explorer with Cairn's full result. Older `#example` links with query parameters still open the explorer.
- `#benchmarks`: measured performance and limitations.

The repository's private outreach drafts are excluded from this directory and
are not published by the Pages workflow.
