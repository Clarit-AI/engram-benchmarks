"""LoCoMo dataset downloader.

Dataset information
-------------------
**HuggingFace slug**: ``adymaharana/locomo``
- Author: Adyasha Maharana (snap-research / SNAP group)
- Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
  (arXiv 2402.17753, CVPR 2024)
- License: CC BY-NC 4.0
- Scale: 35 multi-session conversations; each conversation has up to 35
  sessions, ~300 turns, ~9K tokens average (up to 300K+ tokens total context).
- QA annotations: ~5 questions per conversation, annotated by type:
  single-hop, multi-hop, temporal, open-domain, adversarial.

**Status**: The dataset on HuggingFace (``adymaharana/locomo``) is marked as
empty in the Dataset Viewer (500 error / no splits visible). The canonical
source is the snap-research GitHub repo:
  https://github.com/snap-research/locomo
which hosts ``./data/locomo10.json`` — a flat JSON with 10-conversation sample.

The downloader below attempts HuggingFace first (``datasets`` library), then
falls back to the GitHub raw JSON.  If neither is reachable (air-gapped / CI),
it writes synthetic dry-run data instead.

**Note on MC variant**: ``Percena/locomo-mc10`` is a derived multiple-choice
version (1,986 items, 10-option MC); that dataset uses a different schema than
the open-ended QA format this harness targets.  We use the original open-ended
format where ground truth is a string or list of strings.

Results path: s3://engram/benchmarks/locomo/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known dataset sources (in preference order)
# ---------------------------------------------------------------------------

HF_SLUG = "adymaharana/locomo"
GITHUB_FALLBACK_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

# Fallback slug if HF adds a different version later
HF_SLUG_MC = "Percena/locomo-mc10"  # MC variant — NOT used by this harness


# ---------------------------------------------------------------------------
# Converters from raw dataset formats
# ---------------------------------------------------------------------------

def _convert_locomo10_json(raw: dict) -> list[dict]:
    """Convert snap-research/locomo locomo10.json to our JSONL question format.

    The raw format is a list of conversation dicts.  Each dict has:
    - keys like "session_1", "session_2", ... (list of turn dicts with "content",
      "speaker_role", etc.)
    - "qa" key: list of QA dicts with "question", "answer", "category"
    - "dialogue_id" or similar identifier

    Returns a list of Question-compatible dicts.
    """
    questions = []
    conversations = raw if isinstance(raw, list) else raw.get("data", [raw])
    for conv_idx, conv in enumerate(conversations):
        # Build conversation text from sessions
        session_texts = []
        session_num = 1
        while True:
            key = f"session_{session_num}"
            if key not in conv:
                break
            turns = conv[key]
            session_texts.append(f"Session {session_num}:")
            for turn in turns:
                speaker = turn.get("speaker_role", "Speaker")
                content = turn.get("content", "")
                session_texts.append(f"{speaker}: {content}")
            session_num += 1
        conversation_text = "\n".join(session_texts)
        session_id = str(conv.get("dialogue_id", f"conv_{conv_idx}"))

        # Extract QA pairs
        qa_list = conv.get("qa", [])
        for qa_idx, qa in enumerate(qa_list):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            category = qa.get("category", qa.get("question_type", "factual"))
            if not question:
                continue
            questions.append({
                "question_id": f"{session_id}_q{qa_idx:03d}",
                "session_id": session_id,
                "conversation_text": conversation_text,
                "question": question,
                "answer": answer,
                "question_type": _normalize_category(category),
            })
    return questions


def _normalize_category(cat: str) -> str:
    """Map raw LoCoMo category labels to our normalized question_type strings."""
    cat = cat.lower().strip()
    mapping = {
        "single_hop": "single_hop",
        "single-hop": "single_hop",
        "sh": "single_hop",
        "multi_hop": "multi_hop",
        "multi-hop": "multi_hop",
        "mh": "multi_hop",
        "temporal": "temporal",
        "tr": "temporal",
        "open_domain": "open_domain",
        "open-domain": "open_domain",
        "od": "open_domain",
        "adversarial": "adversarial",
        "adv": "adversarial",
    }
    return mapping.get(cat, cat or "factual")


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def download_from_hf(data_dir: "str | Path", split: str = "train") -> Path:
    """Download LoCoMo from HuggingFace using the ``datasets`` library.

    Parameters
    ----------
    data_dir:
        Local directory to save the converted JSONL file.
    split:
        HF dataset split to download (default: "train").

    Returns
    -------
    Path to the saved JSONL file.

    Raises
    ------
    RuntimeError
        If the ``datasets`` library is not installed or the download fails.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Install 'datasets' to download from HuggingFace: pip install datasets"
        ) from exc

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"locomo_{split}.jsonl"

    logger.info("Downloading %s (split=%s) from HuggingFace...", HF_SLUG, split)
    try:
        ds = load_dataset(HF_SLUG, split=split, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF dataset '{HF_SLUG}'. "
            "The dataset viewer shows it as empty; try the GitHub fallback. "
            f"Original error: {exc}"
        ) from exc

    questions = []
    for row in ds:
        # HF format: dialogue_id (int), turns (JSON string)
        dialogue_id = str(row.get("dialogue_id", ""))
        turns_raw = row.get("turns", "{}")
        if isinstance(turns_raw, str):
            turns_data = json.loads(turns_raw)
        else:
            turns_data = turns_raw

        # Build a synthetic wrapper so _convert_locomo10_json can handle it
        conv_dict: dict = {"dialogue_id": dialogue_id}
        # turns_data has speaker_role (list), content (list), session_ids, etc.
        # Reconstruct session_N keys
        speaker_roles = turns_data.get("speaker_role", [])
        contents = turns_data.get("content", [])
        session_ids = turns_data.get("session_id", [None] * len(speaker_roles))
        # group by session
        sessions: dict[int, list] = {}
        for role, content, sid in zip(speaker_roles, contents, session_ids):
            sid_key = int(sid) if sid is not None else 1
            sessions.setdefault(sid_key, []).append(
                {"speaker_role": role, "content": content}
            )
        for sid_key in sorted(sessions):
            conv_dict[f"session_{sid_key}"] = sessions[sid_key]

        conv_dict["qa"] = turns_data.get("qa", [])
        questions.extend(_convert_locomo10_json([conv_dict]))

    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    logger.info("Saved %d questions to %s", len(questions), out_path)
    return out_path


def download_from_github(data_dir: "str | Path") -> Path:
    """Download the locomo10.json sample from the snap-research GitHub repo.

    Falls back to synthetic data if the download fails (e.g. in CI / air-gapped
    environments).

    Parameters
    ----------
    data_dir:
        Local directory to save the converted JSONL file.

    Returns
    -------
    Path to the saved JSONL file.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Install 'requests' to download from GitHub: pip install requests"
        ) from exc

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "locomo10.jsonl"

    logger.info("Downloading locomo10.json from %s ...", GITHUB_FALLBACK_URL)
    resp = requests.get(GITHUB_FALLBACK_URL, timeout=60)
    resp.raise_for_status()
    raw = resp.json()

    questions = _convert_locomo10_json(raw)

    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    logger.info("Saved %d questions to %s", len(questions), out_path)
    return out_path


def download(
    data_dir: "str | Path",
    prefer: str = "hf",
    split: str = "train",
) -> Path:
    """Download LoCoMo dataset, trying sources in priority order.

    Parameters
    ----------
    data_dir:
        Local directory to save the JSONL file.
    prefer:
        ``"hf"`` → try HuggingFace first, fall back to GitHub.
        ``"github"`` → use GitHub raw JSON directly.
    split:
        HF split to use when prefer="hf".

    Returns
    -------
    Path to the saved JSONL file.
    """
    data_dir = Path(data_dir)
    if prefer == "github":
        return download_from_github(data_dir)

    # Try HF first, fall back to GitHub
    try:
        return download_from_hf(data_dir, split=split)
    except Exception as hf_exc:
        logger.warning(
            "HuggingFace download failed (%s). Trying GitHub fallback...", hf_exc
        )
    try:
        return download_from_github(data_dir)
    except Exception as gh_exc:
        logger.warning(
            "GitHub download also failed (%s). Using synthetic dry-run data.", gh_exc
        )
        return generate_dry_run_data(data_dir)


def generate_dry_run_data(data_dir: "str | Path", n: int = 5) -> Path:
    """Generate synthetic LoCoMo questions and save to JSONL.

    Used when neither HuggingFace nor GitHub is reachable (CI, air-gapped).
    The synthetic conversation (~500 words, 5 sessions) covers facts that the
    5 dry-run questions ask about.

    Parameters
    ----------
    data_dir:
        Directory in which to write ``test_dry_run.jsonl``.
    n:
        Number of questions to generate (max 5).

    Returns
    -------
    Path to the written JSONL file.
    """
    from .runner import generate_dry_run_questions  # noqa: PLC0415

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "test_dry_run.jsonl"

    questions = generate_dry_run_questions(n=n)
    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict()) + "\n")

    logger.info("Generated %d dry-run questions at %s", len(questions), out_path)
    return out_path
