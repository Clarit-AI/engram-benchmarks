# GraphWalks Benchmark

Evaluates Engram stateful inference on the `openai/graphwalks` dataset — graph
traversal (BFS reachability and parent-finding) over up to 128K-token contexts.

## Dataset

| Field | Value |
|---|---|
| Source | `openai/graphwalks` on Hugging Face Hub |
| License | MIT |
| Tasks | BFS reachability, parent-finding |
| Context length | Up to 128K tokens per graph |
| Splits | `test` (primary), `validation` |

## Scoring

Programmatic **set-overlap F1** — no LLM judge, no `OPENAI_API_KEY` needed.

```
precision = |predicted ∩ truth| / |predicted|
recall    = |predicted ∩ truth| / |truth|
f1        = 2 * precision * recall / (precision + recall)
```

Both sets empty → 1.0. Either side empty → 0.0.

## Response Format

The model **must** end every response with exactly:

```
Final Answer: [node1, node2, ...]
```

Both the full-context (baseline) and warm-restore prompts include this instruction.
Any response missing this footer scores 0.0.

## Dry Run (CPU, no model server)

```bash
cd /path/to/engram-benchmarks

# Run the dry-run test suite (synthetic data, mock HTTP)
pytest engram_benchmarks/graphwalks/test_dry_run.py -v

# Generate synthetic question files
python -c "
from engram_benchmarks.graphwalks.download import generate_dry_run_data
questions = generate_dry_run_data('/tmp/graphwalks-dry', n=5)
print(f'Generated {len(questions)} questions')
"
```

## GPU Run

### Prerequisites

- Engram server running and accessible
- HF token with read access to `openai/graphwalks`
- `pip install datasets` for dataset download

### Pending Checklist

- [ ] Verify HF token access: `huggingface-cli whoami`
- [ ] Start Engram server (see main README for launch command)
- [ ] Set `MODEL_URL` env var (e.g. `http://gpu-node:30000`)
- [ ] Download dataset: `from engram_benchmarks.graphwalks.download import download_dataset`
- [ ] Run benchmark harness with `GraphWalksRunner` (live model, no `mock_fn`)
- [ ] Upload results to S3

### Target Models

| Model | Config |
|---|---|
| Elastic 30B (BF16) | Single H100 |
| Qwen3-Coder-Next (FP8) | `--tp 4`, cluster |

## Results

Results are written to S3:

```
s3://engram/benchmarks/graphwalks/<run-id>/results.jsonl
```

JSONL format: one header line (run metadata + warm aggregates), then one
`GraphWalksResult` per question.

### Key Metrics

| Metric | Description |
|---|---|
| `warm_token_reduction` | Mean fraction of input tokens saved (warm only) |
| `warm_ttft_speedup` | Mean TTFT speedup ratio (baseline / Engram warm) |
| `warm_tokens_saved_total` | Total input tokens avoided across warm restores |
| `per_hop_breakdown` | Per-hop mean F1, token reduction, TTFT speedup |

## Module Layout

```
engram_benchmarks/graphwalks/
  __init__.py          # empty
  scoring.py           # parse_answer, SetF1Scorer
  results.py           # GraphWalksResult, GraphWalksRunSummary
  runner.py            # Question dataclass, GraphWalksRunner, generate_synthetic_question
  download.py          # generate_dry_run_data, download_dataset
  test_dry_run.py      # full dry-run test suite (36 tests)
  README.md            # this file
```
