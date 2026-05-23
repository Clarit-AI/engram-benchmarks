"""LongMemEval dataset download utilities.

Dataset
-------
Name:         LongMemEval
HuggingFace:  xiaowu0162/LongMemEval
Size:         ~2 GB
Questions:    500
Memory types: episodic, semantic, temporal, spatial, factual
Scale:        115K–1.5M tokens per memory document
License:      MIT

Required packages
-----------------
- ``huggingface_hub>=0.20`` — for ``hf_hub_download`` / ``snapshot_download``
- ``datasets>=2.14`` — for loading the dataset splits

Install with::

    pip install huggingface_hub datasets

Usage
-----
::

    from pathlib import Path
    from engram_benchmarks.longmemeval.download import download, generate_dry_run_data

    data_dir = Path("data/longmemeval")
    download(data_dir)
    # Questions available at data_dir/test.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


_HF_REPO_ID = "xiaowu0162/LongMemEval"
_DRY_RUN_FILENAME = "test_dry_run.jsonl"

_MEMORY_TYPES = ["episodic", "semantic", "temporal", "spatial", "factual"]


def download(data_dir: Path) -> None:
    """Download the LongMemEval dataset from Hugging Face.

    Downloads ``xiaowu0162/LongMemEval`` to ``data_dir`` using
    ``huggingface_hub.snapshot_download``.  The resulting JSONL files can be
    loaded with :func:`engram_benchmarks.longmemeval.runner.load_questions`.

    Parameters
    ----------
    data_dir:
        Local directory where the dataset will be stored.  Created if absent.

    Raises
    ------
    RuntimeError
        If ``huggingface_hub`` or ``datasets`` is not installed.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The 'huggingface_hub' package is required to download LongMemEval. "
            "Install with: pip install huggingface_hub datasets"
        ) from exc

    print(f"Downloading {_HF_REPO_ID} to {data_dir} (~2 GB) ...")
    snapshot_download(
        repo_id=_HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(data_dir),
        ignore_patterns=["*.parquet"],  # prefer JSONL if both are available
    )
    print(f"Download complete. Data at: {data_dir}")


def generate_dry_run_data(data_dir: Path) -> Path:
    """Write 5 synthetic questions to ``data_dir/test_dry_run.jsonl``.

    No network access is required.  The generated file can be loaded with
    :func:`engram_benchmarks.longmemeval.runner.load_questions`.

    Parameters
    ----------
    data_dir:
        Directory where the dry-run JSONL will be written.  Created if absent.

    Returns
    -------
    Path
        Full path to the written JSONL file.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    for i in range(5):
        mt = _MEMORY_TYPES[i % len(_MEMORY_TYPES)]
        records.append(
            {
                "question_id": f"dry_run_{i:03d}",
                "memory_text": (
                    "Alice moved to Paris in 2019. "
                    "She works as a data scientist. "
                    "Her favourite restaurant is Le Comptoir. "
                    f"Memory entry {i}: she visited the Louvre on a Tuesday."
                ),
                "question": f"Where did Alice move in 2019? (dry-run question {i})",
                "answer": "Paris",
                "memory_type": mt,
            }
        )

    out_path = data_dir / _DRY_RUN_FILENAME
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Dry-run data written to: {out_path}")
    return out_path
