# LoCoMo Benchmark Harness

Engram benchmark harness for **LoCoMo** (Long Conversation Memory) — evaluates
whether a model can answer questions about long multi-session conversations.

---

## Dataset

**Paper**: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
(Maharana et al., arXiv 2402.17753, CVPR 2024)

**HuggingFace slug**: `adymaharana/locomo`
- Author: Adyasha Maharana (SNAP / snap-research)
- License: **CC BY-NC 4.0** (non-commercial use)
- Scale: 35 multi-session conversations; up to 35 sessions each, ~300 turns,
  ~9K tokens average — up to **300K+ tokens total context** per conversation
- QA annotations: ~5 questions per conversation, typed by category

**HF Viewer status**: As of May 2026, the HF dataset viewer returns a 500 error
("dataset is empty"). The canonical data source is the snap-research GitHub
repo: `https://github.com/snap-research/locomo` (file `./data/locomo10.json`).

The downloader (`download.py`) tries HuggingFace first, falls back to GitHub
raw JSON, and finally generates synthetic dry-run data if both fail.

**MC variant** (`Percena/locomo-mc10`) is a derived 1,986-item multiple-choice
reformatting; this harness uses the original open-ended QA format.

### Question types
| Code | Name | Description |
|---|---|---|
| `single_hop` | Single-hop | Direct recall from one session |
| `multi_hop` | Multi-hop | Synthesis across multiple sessions |
| `temporal` | Temporal | Temporal ordering / "when did X happen?" |
| `open_domain` | Open-domain | Requires external / commonsense knowledge |
| `adversarial` | Adversarial | Questions designed to trick / hallucinate |

---

## Scoring: Token F1

**No LLM judge. No `OPENAI_API_KEY` needed.**

Token F1 is the standard SQuAD-style QA metric:
1. Normalize prediction and reference (lowercase, strip punctuation, remove
   articles *a/an/the*, collapse whitespace).
2. Compute token-level precision and recall via word-count overlap.
3. Return the harmonic mean (F1 ∈ [0, 1]).

When a question has multiple acceptable answers, max F1 over all references is
returned (matching the original LoCoMo paper evaluation protocol).

---

## Two-phase Engram protocol

This harness supersedes the in-progress LoCoMo harness in the Engram fork.

For each question:
1. **Baseline** — full multi-session conversation context + question (stateless;
   all tokens sent every time).
2. **Cold Engram pass** — full context sent to Engram, snapshot saved.
   Latency is recorded but not reported as a warm metric.
3. **Warm Engram pass** — question only; conversation snapshot is restored from
   Engram state. This is the measured warm TTFT and token-reduction metric.

The warm path avoids re-ingesting 300K+ tokens of conversation history on every
question, reducing prefill cost dramatically.

---

## Results path

```
s3://engram/benchmarks/locomo/
```

---

## GPU run pending checklist

- [ ] Provision H100 or H200 node
- [ ] Launch Elastic 30B (BF16) on single H100 via SGLang
- [ ] Launch Qwen3-Coder-Next (FP8, `--tp 4`) for cluster run
- [ ] Download full `adymaharana/locomo` dataset (or `locomo10.json` sample)
- [ ] Run `python -m engram_benchmarks.locomo.download --data-dir ./data`
- [ ] Run `python -m engram_benchmarks.locomo.runner --data ./data/locomo_train.jsonl`
- [ ] Upload results JSONL to `s3://engram/benchmarks/locomo/`
- [ ] Confirm per-type F1 breakdown in `LoCoMoRunSummary.per_type_warm_f1`

---

## Models

| Model | Precision | Hardware | Notes |
|---|---|---|---|
| Elastic 30B | BF16 | Single H100 | Primary evaluation target |
| Qwen3-Coder-Next | FP8 | 4× H100/H200 (`--tp 4`) | Cluster evaluation |

---

## Quick start (dry run, no GPU)

```bash
cd engram-benchmarks
pip install -e ".[dev]"

# Generate synthetic dry-run data
python -c "
from engram_benchmarks.locomo.download import generate_dry_run_data
generate_dry_run_data('./data', n=5)
"

# Run tests
pytest engram_benchmarks/locomo/test_dry_run.py -v
```

---

## Do not touch

PR #72 in the Engram fork (`Clarit-AI/engram`) must not be closed or modified
until the GPU evaluation results from this harness are reviewed.
