# engram-benchmarks

HTTP-client benchmark suite for [Engram](https://github.com/Clarit-AI/Engram) stateful inference.

Harnesses run against a server URL (never importing engine internals), comparing a stateless baseline against Engram's snapshot-backed stateful path. Results upload to `s3://engram/benchmarks/` via `s3cmd` (sfo3).

## Benchmarks

| Benchmark | Scoring | Long context scale |
|---|---|---|
| LongMemEval | LLM-as-judge (`gpt-4o-2024-08-06`) | 115K–1.5M tokens |
| RULER | Programmatic (substring recall) | 4K–128K tokens |
| GraphWalks | Programmatic (set-overlap F1) | up to 128K tokens |
| LoCoMo | Programmatic (token F1) | multi-session conversation |

## Shared layer

All harnesses inherit from `engram_benchmarks/shared/`:

- **`http_client.py`** — streaming HTTP client; TTFT = time-to-first-streamed-delta-token (always streaming, never whole-call)
- **`runner.py`** — deliberate warm-tier protocol (baseline → cold pass → warm measurement)
- **`results.py`** — `BaseResult` + `RunSummary` with warm-only KHA-394 aggregates
- **`compute_amort.py`** — break-even amortization model
- **`scoring.py`** — `BaseScorer` interface
- **`s3_writer.py`** — `s3cmd` upload helper

## Warm/cold measurement contract

Warm metrics are reported separately from cold. Cold numbers are zero-by-construction for token reduction and TTFT savings, so they must never be averaged into headline warm-tier metrics. `RunSummary.warm_token_reduction`, `warm_ttft_speedup`, and `compute_amortization()` operate over warm results only.

## Server configuration

```bash
export ENGRAM_SERVER_URL="http://localhost:30000"   # model server
export JUDGE_MODEL="gpt-4o-2024-08-06"             # LongMemEval only
export JUDGE_BASE_URL=""                            # optional; blank = OpenAI
export OPENAI_API_KEY="sk-..."                      # LongMemEval live judge only
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```
