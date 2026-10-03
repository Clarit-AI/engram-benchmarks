# RULER Benchmark — Engram Harness

Long-context evaluation across 13 synthetic tasks, from 4K to 128K+ tokens.

## Dataset

**RULER** (from [hsiehjackson/RULER](https://github.com/hsiehjackson/RULER)) defines 13 task categories that stress-test long-context models across recall, tracking, extraction, and QA scenarios.

This harness generates all tasks **synthetically and offline** — no dataset download is required. Tasks are deterministic for a given `(task_name, context_length, sample_idx)` triple.

### Task list

| Group | Tasks |
|---|---|
| Needle-in-a-haystack (single) | `niah_single_1`, `niah_single_2`, `niah_single_3` |
| Needle-in-a-haystack (multi-key) | `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3` |
| Needle-in-a-haystack (multi-value) | `niah_multivalue` |
| Needle-in-a-haystack (multi-query) | `niah_multiquery` |
| Variable tracking | `vt` |
| Common word extraction | `cwe` |
| Frequent word extraction | `fwe` |
| Question answering | `qa_1`, `qa_2` |

### Context lengths

```
4096  8192  16384  32768  65536  131072
```

### License

MIT (same as upstream RULER).

## Scoring

Scoring is fully programmatic — **no LLM judge, no `OPENAI_API_KEY` needed**.

| Scorer | Tasks | Logic |
|---|---|---|
| `StringMatchAllScorer` | All except QA | Fraction of reference strings found as substrings in the prediction. `1.0` = all found. |
| `StringMatchPartScorer` | `qa_1`, `qa_2` | `1.0` if any reference string is a substring; `0.0` otherwise. |

Both scorers are case-insensitive.

## Run commands

### Dry run (no GPU, no server)

```bash
cd /path/to/eb-ruler

# Quick smoke test — 2 tasks, 1 context length
python -m engram_benchmarks.ruler.runner \
    --dry-run \
    --tasks niah_single_1 qa_1 \
    --context-lengths 4096 \
    --output results/ruler_dry.jsonl

# Full test suite
pytest engram_benchmarks/ruler/test_dry_run.py -v
```

### Live run (single H100, Elastic 30B BF16)

```bash
python -m engram_benchmarks.ruler.runner \
    --url http://localhost:30000 \
    --model elastic-30b-bf16 \
    --tasks niah_single_1 niah_single_2 niah_single_3 \
            niah_multikey_1 niah_multikey_2 niah_multikey_3 \
            niah_multivalue niah_multiquery \
            vt cwe fwe qa_1 qa_2 \
    --context-lengths 4096 8192 16384 32768 65536 131072 \
    --samples 0 1 2 3 4 \
    --output results/ruler_elastic30b.jsonl
```

### Cluster run (Qwen3-Coder-Next FP8, TP4)

```bash
python -m engram_benchmarks.ruler.runner \
    --url http://<head-node>:30000 \
    --model qwen3-coder-next-fp8 \
    --context-lengths 4096 8192 16384 32768 65536 131072 \
    --output results/ruler_qwen3coder.jsonl
```

### Expected dry-run output

```
=== RULER Results ===
Tasks run      : 2 total (2 warm, 0 cold)
Warm token red.: 0.97...
Warm TTFT spup.: 3.57...
Output         : results/ruler_dry.jsonl
```

TTFT confirmation: mock TTFT = 0.042s < mock total latency = 0.150s.

## Results path

```
s3://engram/benchmarks/ruler/
```

File convention: `ruler_<model-slug>_<YYYYMMDD>.jsonl`

## GPU run checklist

- [ ] SGLang server running with `--enable-engram` (snapshot support)
- [ ] Model loaded and `/v1/models` responds
- [ ] `--url` points to the correct host:port
- [ ] Snapshot directory has sufficient disk space (prefix caches)
- [ ] Results uploaded: `aws s3 cp results/ruler_*.jsonl s3://engram/benchmarks/ruler/`

## Target models

| Model | Precision | Hardware | TP |
|---|---|---|---|
| Elastic 30B | BF16 | Single H100 | 1 |
| Qwen3-Coder-Next | FP8 | H100 cluster | 4 |
