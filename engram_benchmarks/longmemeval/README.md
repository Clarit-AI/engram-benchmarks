# LongMemEval Benchmark Harness

Engram benchmark harness for the [LongMemEval](https://huggingface.co/datasets/xiaowu0162/LongMemEval) dataset.

## Dataset

| Field | Value |
|---|---|
| HuggingFace ID | `xiaowu0162/LongMemEval` |
| Questions | 500 |
| Memory types | episodic, semantic, temporal, spatial, factual |
| Context scale | 115K–1.5M tokens per document |
| License | MIT |
| Download size | ~2 GB |

## Scoring

Uses **LLM-as-judge** per LongMemEval paper §3.3.

| Env var | Default | Description |
|---|---|---|
| `JUDGE_MODEL` | `gpt-4o-2024-08-06` | OpenAI model for judging |
| `JUDGE_BASE_URL` | _(OpenAI endpoint)_ | Optional alternative base URL |
| `OPENAI_API_KEY` | _(required for live runs)_ | API key for the judge |

The judge sends the question, reference answer, and model answer to the judge model
and returns 1.0 if the response contains "yes" (case-insensitive), 0.0 otherwise.

## Run Commands

### Dry-run (no GPU, no API key required)

```bash
python -m engram_benchmarks.longmemeval --dry-run
```

### Full GPU run

```bash
# 1. Download dataset
python -c "
from pathlib import Path
from engram_benchmarks.longmemeval.download import download
download(Path('data/longmemeval'))
"

# 2. Run benchmark
OPENAI_API_KEY=<your-key> python -m engram_benchmarks.longmemeval \
  --model-url http://localhost:30000 \
  --snapshot-dir /tmp/lme_snapshots \
  --output results/longmemeval.jsonl \
  --num-questions 500
```

### Baseline-only mode

```bash
OPENAI_API_KEY=<your-key> python -m engram_benchmarks.longmemeval \
  --mode baseline \
  --model-url http://localhost:30000 \
  --output results/longmemeval_baseline.jsonl
```

## Results Path

```
s3://engram/benchmarks/longmemeval/
```

## GPU Run Checklist

- [ ] Engram server running on H100 (`--model-url` reachable)
- [ ] Dataset downloaded to `data/longmemeval/test.jsonl`
- [ ] `OPENAI_API_KEY` set (for `LLMJudge`)
- [ ] `JUDGE_MODEL` set or defaulting to `gpt-4o-2024-08-06`
- [ ] Snapshot directory writable (`--snapshot-dir`)
- [ ] Output path writable (`--output`)
- [ ] Results uploaded to S3 after run

## Target Models

| Model | Precision | Hardware |
|---|---|---|
| Elastic 30B | BF16 | Single H100 |
| Qwen3-Coder-Next | FP8 | 4× H100 (`--tp 4`) |

## Architecture

The harness follows the shared three-pass warm-tier protocol:

1. **Baseline** — full memory context sent stateless; TTFT and tokens recorded.
2. **Cold Engram pass** — full context sent, snapshot saved; latency discarded.
3. **Warm Engram pass** — question only sent with snapshot present; warm TTFT measured.

All results from `run_all()` are labelled `restore_mode="warm"`.  Cold-only
comparison runs use `run_baseline_only()`.
