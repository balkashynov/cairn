#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=python
PYTHON="${CAIRN_PYTHON:-.venv/bin/python}"
output="${1:-results/final}"
mkdir -p "$output"
for dataset in scifact cqadupstack-programmers fiqa; do
  "$PYTHON" -m cairn_search audit "data/$dataset/index.cairn" "data/$dataset/queries.fbin" \
    --query-metadata "data/$dataset/queries.jsonl" --corpus-metadata "data/$dataset/corpus.jsonl" \
    --qrels "data/$dataset/qrels.json" --output "$output/$dataset" --repeats 3 --faiss \
    --dataset "$dataset / Cohere embed-english-v3.0" \
    --source-note 'Public CohereLabs BEIR embeddings, revision f018922a46a51388348a307ec2fef28019a33026. Model: embed-english-v3.0. Relevance judgments and text belong to the original BEIR datasets. No live API or Embed 5 run is implied.' \
    > "$output/$dataset.log"
  printf '%s complete\n' "$dataset"
done
"$PYTHON" -m cairn_search audit data/cohere-wiki-100k/index.cairn data/cohere-wiki-100k/queries.fbin \
  --output "$output/cohere-wiki-100k" --repeats 3 --faiss --dataset 'Cohere Wikipedia 99,999 x 768' \
  --source-note 'Qdrant cohere-wiki-100k-no-filters benchmark. Historical Cohere Wikipedia embeddings; exact model version and query generation not established by the downloaded archive. Not Embed 5.' \
  > "$output/cohere-wiki-100k.log"
printf 'Wikipedia complete\n'
