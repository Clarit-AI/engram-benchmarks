"""LoCoMo two-phase runner.

Implements the deliberate warm-tier protocol from BaseTwoPhaseRunner for the
LoCoMo long-conversation-memory benchmark.

Two-phase logic
---------------
- Full prompt  : full multi-session conversation text + question
- Warm prompt  : question only (conversation snapshot assumed present in Engram)

The conversation snapshot represents the Engram stateful-inference savings:
instead of re-ingesting 300K+ tokens of conversation history per question,
the warm path only sends the question itself.

Dataset contract
----------------
See ``Question`` for the fields expected from the LoCoMo dataset.  The
``load_questions`` helper reads from a JSONL file produced by ``download.py``.
``generate_dry_run_questions`` produces synthetic questions for CPU testing.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from engram_benchmarks.shared.http_client import MockHttpFn, _word_count
from engram_benchmarks.shared.results import SnapshotMode
from engram_benchmarks.shared.runner import BaseTwoPhaseRunner

from .results import LoCoMoResult, LoCoMoRunSummary
from .scoring import TokenF1Scorer

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """A single LoCoMo QA item."""

    question_id: str
    session_id: str
    conversation_text: str      # full multi-session conversation (may be 300K+ chars)
    question: str
    answer: str | list[str]  # str or list of acceptable answers
    question_type: str = "factual"  # e.g. "temporal", "single_hop", "multi_hop"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Question:
        return cls(
            question_id=d["question_id"],
            session_id=d["session_id"],
            conversation_text=d["conversation_text"],
            question=d["question"],
            answer=d["answer"],
            question_type=d.get("question_type", "factual"),
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class LoCoMoRunner(BaseTwoPhaseRunner[Question]):
    """Two-phase LoCoMo runner.

    Extends BaseTwoPhaseRunner[Question] with:
    - TokenF1Scorer as default scorer
    - LoCoMoResult (adds session_id, question_type)
    - LoCoMoRunSummary (adds per-type warm F1 breakdown)
    """

    _result_cls = LoCoMoResult

    def __init__(
        self,
        model_url: str,
        snapshot_dir: Path,
        model: str = "default",
        max_tokens: int = 256,
        mock_fn: MockHttpFn | None = None,
        snapshot_api_enabled: bool = False,
        model_path: str = "",
        warm_branch_prefix: str = "bm-warm",
        snapshot_mode: SnapshotMode = "mamba_only",
        admin_api_key: str | None = None,
    ) -> None:
        super().__init__(
            model_url=model_url,
            snapshot_dir=Path(snapshot_dir),
            scorer=TokenF1Scorer(),
            model=model,
            max_tokens=max_tokens,
            mock_fn=mock_fn,
            snapshot_api_enabled=snapshot_api_enabled,
            model_path=model_path,
            warm_branch_prefix=warm_branch_prefix,
            snapshot_mode=snapshot_mode,
            admin_api_key=admin_api_key,
        )

    # ------------------------------------------------------------------ #
    # BaseTwoPhaseRunner interface                                         #
    # ------------------------------------------------------------------ #

    def _item_id(self, item: Question) -> str:
        return item.question_id

    def _build_full_prompt(self, item: Question) -> str:
        """Full prompt: conversation context + question."""
        return (
            f"The following is a multi-session conversation between two people.\n\n"
            f"{item.conversation_text}\n\n"
            f"Based on the conversation above, answer the following question.\n"
            f"Question: {item.question}\n"
            f"Answer:"
        )

    def _build_warm_prompt(self, item: Question) -> str:
        """Warm prompt: question only (conversation snapshot is in Engram state)."""
        return (
            f"Using the conversation context from memory, answer the following question.\n"
            f"Question: {item.question}\n"
            f"Answer:"
        )

    def _reference_answer(self, item: Question) -> Any:
        return item.answer

    def _snapshot_metadata(self, item: Question, full_prompt: str) -> dict:
        return {
            "question_id": item.question_id,
            "session_id": item.session_id,
            "question_type": item.question_type,
            "conversation_tokens_approx": _word_count(item.conversation_text),
        }

    def _extra_result_fields(self, item: Question) -> dict:
        return {
            "session_id": item.session_id,
            "question_type": item.question_type,
        }

    def build_summary(
        self,
        results: list[LoCoMoResult],
        model: str = "default",
    ) -> LoCoMoRunSummary:
        """Wrap results in a LoCoMoRunSummary."""
        return LoCoMoRunSummary(
            benchmark="locomo",
            model=model,
            results=results,
        )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_questions(path: str | Path) -> list[Question]:
    """Load questions from a JSONL file (one Question JSON per line).

    The JSONL format produced by ``download.py`` and ``generate_dry_run_data``
    has one serialised Question dict per line (no header line).
    """
    path = Path(path)
    questions: list[Question] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(Question.from_dict(json.loads(line)))
    return questions


# ---------------------------------------------------------------------------
# Dry-run synthetic data
# ---------------------------------------------------------------------------

_SYNTHETIC_CONVERSATION = textwrap.dedent("""\
    Session 1 — January 10:
    Alice: Hi Bob! How was your holiday in Paris?
    Bob: It was wonderful. We visited the Louvre and saw the Mona Lisa.
    Alice: Did you try the croissants?
    Bob: Yes, had them every morning at a bakery near the Seine.
    Alice: Sounds amazing. I heard you're changing jobs soon?
    Bob: Yes, I'm joining a biotech startup called HelixCore in March.

    Session 2 — February 3:
    Alice: How's the job transition going, Bob?
    Bob: Good! HelixCore starts March 1st. I'm wrapping up at my current place.
    Alice: Exciting. What will you be doing there?
    Bob: Leading the data engineering team. About twelve engineers.
    Alice: That's a big step. Did you visit Paris again?
    Bob: No, but I'm planning a trip to Tokyo in April with my wife.

    Session 3 — March 5:
    Bob: First week at HelixCore done! It's intense but great.
    Alice: How's the team?
    Bob: Really talented. My lead engineer is named Priya — she's exceptional.
    Alice: What tech stack are they using?
    Bob: Mostly Python and Rust for the pipeline. Some Kubernetes for orchestration.
    Alice: Sounds like a great fit for you.

    Session 4 — April 20:
    Alice: How was Tokyo?
    Bob: Incredible. We went to Kyoto too — the bamboo forest was breathtaking.
    Alice: Did you eat sushi?
    Bob: Of course! Best omakase of my life at a tiny place in Shinjuku.
    Alice: And work?
    Bob: HelixCore hit its Q1 targets. Priya shipped a huge pipeline refactor.

    Session 5 — May 15:
    Alice: Any big summer plans?
    Bob: We're thinking about renting a cottage in the Scottish Highlands in August.
    Alice: That sounds peaceful. How is your daughter Emma doing?
    Bob: She just finished her first year at university — studying marine biology.
    Alice: You must be proud.
    Bob: Very. She already got a summer research spot at the aquarium.
""")

_DRY_RUN_ITEMS = [
    {
        "question_id": "dry_q001",
        "session_id": "dry_session_001",
        "question": "What company did Bob join in March?",
        "answer": "HelixCore",
        "question_type": "factual",
    },
    {
        "question_id": "dry_q002",
        "session_id": "dry_session_001",
        "question": "Which museum did Bob visit in Paris?",
        "answer": ["the Louvre", "Louvre"],
        "question_type": "factual",
    },
    {
        "question_id": "dry_q003",
        "session_id": "dry_session_001",
        "question": "When did Bob start at HelixCore?",
        "answer": "March 1st",
        "question_type": "temporal",
    },
    {
        "question_id": "dry_q004",
        "session_id": "dry_session_001",
        "question": "What is the name of Bob's lead engineer at HelixCore?",
        "answer": "Priya",
        "question_type": "single_hop",
    },
    {
        "question_id": "dry_q005",
        "session_id": "dry_session_001",
        "question": "What is Emma studying at university?",
        "answer": "marine biology",
        "question_type": "multi_hop",
    },
]


def generate_dry_run_questions(n: int = 5) -> list[Question]:
    """Return up to ``n`` synthetic LoCoMo questions for CPU dry-run testing.

    Each question uses the same synthetic multi-session conversation (~500 words)
    and references facts mentioned in it.
    """
    items = _DRY_RUN_ITEMS[:n]
    return [
        Question(
            question_id=item["question_id"],
            session_id=item["session_id"],
            conversation_text=_SYNTHETIC_CONVERSATION,
            question=item["question"],
            answer=item["answer"],
            question_type=item["question_type"],
        )
        for item in items
    ]
