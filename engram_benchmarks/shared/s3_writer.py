"""S3 result uploader via s3cmd (sfo3 DigitalOcean Spaces)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


S3_BUCKET = "s3://engram/benchmarks"
S3CMD_CONFIG = "~/.s3cfg"  # expects sfo3 endpoint configured


def write_results_to_s3(
    local_path: "str | Path",
    benchmark: str,
    run_id: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """Upload a JSONL results file to S3.

    Parameters
    ----------
    local_path:
        Local JSONL file to upload.
    benchmark:
        Benchmark name (e.g. "longmemeval", "ruler", "graphwalks", "locomo").
    run_id:
        Optional run identifier appended to the S3 key.
    dry_run:
        If True, print the s3cmd command but do not execute it.

    Returns
    -------
    str
        S3 destination URI.
    """
    local_path = Path(local_path)
    filename = local_path.name
    dest = f"{S3_BUCKET}/{benchmark}/{filename}"

    cmd = ["s3cmd", "put", str(local_path), dest]
    if dry_run:
        print(f"[dry-run] Would run: {' '.join(cmd)}", file=sys.stderr)
        return dest

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"s3cmd upload failed (exit {result.returncode}):\n{result.stderr}"
        )
    return dest
