"""RULER task definitions and synthetic task generation.

All tasks are generated offline — no dataset download is required.

Task taxonomy
-------------
niah_single_*   : Needle-in-a-haystack, single needle, varying depth.
niah_multikey_* : Multiple distinct needles; all must be recalled.
niah_multivalue : One key, multiple values embedded at different positions.
niah_multiquery : Multiple questions, each targeting a different needle.
vt              : Variable tracking — trace a value through a chain of assignments.
cwe             : Common word extraction — most frequent words in a long passage.
fwe             : Frequent word extraction — weighted variant.
qa_1 / qa_2     : Short QA pairs embedded in noisy context.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

RULER_TASKS: list[str] = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

CONTEXT_LENGTHS: list[int] = [4096, 8192, 16384, 32768, 65536, 131072]

# Filler text used to pad context to the desired length (words).
_FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
)

# Average word length in chars (used to approximate context_length in tokens).
_AVG_CHARS_PER_TOKEN = 4.5


# ------------------------------------------------------------------ #
# TaskInstance                                                         #
# ------------------------------------------------------------------ #


@dataclass
class TaskInstance:
    """A single RULER task instance ready to be sent to a model."""

    task_id: str          # e.g. "niah_single_1__4096__0"
    task_name: str        # one of RULER_TASKS
    context_length: int   # target context length in tokens (approximate)
    context_text: str     # long-context passage including needle/answer
    question: str         # question posed to the model
    answer: list[str]     # list of acceptable answer strings (any is correct)


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #


def _make_filler(approx_tokens: int) -> str:
    """Return a filler passage of approximately ``approx_tokens`` tokens."""
    target_chars = int(approx_tokens * _AVG_CHARS_PER_TOKEN)
    # Repeat the base sentence until we exceed the target length.
    reps = (target_chars // len(_FILLER_SENTENCE)) + 2
    text = (_FILLER_SENTENCE * reps)[:target_chars]
    # Round to nearest sentence boundary.
    last_dot = text.rfind(".")
    return text[: last_dot + 1] if last_dot >= 0 else text


def _insert_at(context: str, needle: str, position_frac: float) -> str:
    """Insert ``needle`` at ``position_frac`` through the context text."""
    idx = int(len(context) * max(0.0, min(1.0, position_frac)))
    # Snap to next space to avoid splitting a word.
    space = context.find(" ", idx)
    if space == -1:
        space = idx
    return context[:space] + " " + needle + " " + context[space:]


def _rng(task_name: str, context_length: int, sample_idx: int) -> random.Random:
    seed_str = f"{task_name}__{context_length}__{sample_idx}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
    return random.Random(seed)


# ------------------------------------------------------------------ #
# Per-task generators                                                  #
# ------------------------------------------------------------------ #


def _gen_niah_single(
    task_name: str, context_length: int, sample_idx: int, n_needles: int = 1
) -> TaskInstance:
    """Needle-in-a-haystack — n_needles distinct needles hidden in filler."""
    rng = _rng(task_name, context_length, sample_idx)
    # Vary the depth by the sample index within [0.1, 0.9].
    depths = [0.1 + (i + 1) * 0.8 / (n_needles + 1) for i in range(n_needles)]

    needles: list[str] = []
    for i in range(n_needles):
        magic = rng.randint(100000, 999999)
        needles.append(f"The secret passphrase number {i + 1} is {magic}.")

    filler = _make_filler(context_length - n_needles * 10)
    ctx = filler
    for needle, depth in zip(needles, depths):
        ctx = _insert_at(ctx, needle, depth)

    if n_needles == 1:
        question = "What is the secret passphrase number 1?"
    else:
        question = f"What are all {n_needles} secret passphrases in order?"

    # Extract the numeric part of each needle as the expected answer.
    answers = [n.split("is ")[-1].rstrip(".") for n in needles]

    return TaskInstance(
        task_id=f"{task_name}__{context_length}__{sample_idx}",
        task_name=task_name,
        context_length=context_length,
        context_text=ctx,
        question=question,
        answer=answers,
    )


def _gen_niah_multivalue(context_length: int, sample_idx: int) -> TaskInstance:
    """Single key, three values at different depths."""
    rng = _rng("niah_multivalue", context_length, sample_idx)
    key = "alpha"
    values = [str(rng.randint(10000, 99999)) for _ in range(3)]
    depths = [0.15, 0.50, 0.85]

    filler = _make_filler(context_length - 30)
    ctx = filler
    for val, depth in zip(values, depths):
        needle = f"The value of {key} at this position is {val}."
        ctx = _insert_at(ctx, needle, depth)

    question = f"What are all recorded values of '{key}' in the passage?"
    return TaskInstance(
        task_id=f"niah_multivalue__{context_length}__{sample_idx}",
        task_name="niah_multivalue",
        context_length=context_length,
        context_text=ctx,
        question=question,
        answer=values,
    )


def _gen_niah_multiquery(context_length: int, sample_idx: int) -> TaskInstance:
    """Multiple questions, each targeting a different needle."""
    rng = _rng("niah_multiquery", context_length, sample_idx)
    n = 3
    keys = ["gamma", "delta", "epsilon"]
    values = [str(rng.randint(10000, 99999)) for _ in range(n)]
    depths = [0.2, 0.5, 0.8]

    filler = _make_filler(context_length - n * 15)
    ctx = filler
    for key, val, depth in zip(keys, values, depths):
        needle = f"The secret code for {key} is {val}."
        ctx = _insert_at(ctx, needle, depth)

    question = (
        "Answer each of the following:\n"
        + "\n".join(f"{i + 1}. What is the secret code for {k}?" for i, k in enumerate(keys))
    )
    return TaskInstance(
        task_id=f"niah_multiquery__{context_length}__{sample_idx}",
        task_name="niah_multiquery",
        context_length=context_length,
        context_text=ctx,
        question=question,
        answer=values,
    )


def _gen_vt(context_length: int, sample_idx: int) -> TaskInstance:
    """Variable tracking — follow a chain of assignments."""
    rng = _rng("vt", context_length, sample_idx)
    chain_len = 5
    var_names = [f"var_{rng.randint(1000, 9999)}" for _ in range(chain_len)]
    final_value = str(rng.randint(100, 999))

    # Build assignment chain: var_a = var_b = ... = final_value
    assignments = []
    for i in range(chain_len - 1):
        assignments.append(f"Let {var_names[i]} equal {var_names[i + 1]}.")
    assignments.append(f"Let {var_names[-1]} equal {final_value}.")

    filler = _make_filler(context_length - len(assignments) * 10)
    ctx = filler
    spacing = 1.0 / (len(assignments) + 1)
    for i, stmt in enumerate(assignments):
        ctx = _insert_at(ctx, stmt, (i + 1) * spacing)

    question = f"Following all assignments, what is the final value of {var_names[0]}?"
    return TaskInstance(
        task_id=f"vt__{context_length}__{sample_idx}",
        task_name="vt",
        context_length=context_length,
        context_text=ctx,
        question=question,
        answer=[final_value],
    )


def _gen_cwe(context_length: int, sample_idx: int) -> TaskInstance:
    """Common word extraction — find the most frequent content word."""
    rng = _rng("cwe", context_length, sample_idx)
    target_word = rng.choice(["pineapple", "tangerine", "persimmon", "kumquat", "dragonfruit"])
    filler = _make_filler(context_length - 200)

    # Inject the target word many times throughout the filler.
    inject_positions = [rng.random() for _ in range(20)]
    ctx = filler
    for pos in sorted(inject_positions, reverse=True):
        ctx = _insert_at(ctx, target_word, pos)

    question = (
        "Identify the single most frequently occurring uncommon content word in the passage."
    )
    return TaskInstance(
        task_id=f"cwe__{context_length}__{sample_idx}",
        task_name="cwe",
        context_length=context_length,
        context_text=ctx,
        question=question,
        answer=[target_word],
    )


def _gen_fwe(context_length: int, sample_idx: int) -> TaskInstance:
    """Frequent word extraction — weighted by explicit counts in text."""
    rng = _rng("fwe", context_length, sample_idx)
    candidates = [
        ("zephyr", rng.randint(25, 40)),
        ("mosaic", rng.randint(10, 20)),
        ("cobalt", rng.randint(5, 10)),
    ]
    target_word = candidates[0][0]  # highest count

    filler = _make_filler(context_length - 500)
    ctx = filler
    for word, count in candidates:
        for _ in range(count):
            pos = rng.random()
            ctx = _insert_at(ctx, word, pos)

    question = (
        "Which of the following words appears most frequently: "
        + ", ".join(w for w, _ in candidates) + "?"
    )
    return TaskInstance(
        task_id=f"fwe__{context_length}__{sample_idx}",
        task_name="fwe",
        context_length=context_length,
        context_text=ctx,
        question=question,
        answer=[target_word],
    )


def _gen_qa(
    task_name: str, context_length: int, sample_idx: int
) -> TaskInstance:
    """Simple QA — a fact sentence is embedded in noisy context."""
    rng = _rng(task_name, context_length, sample_idx)

    qa_pairs = [
        ("What is the boiling point of water in Celsius?", "100", "Water boils at 100 degrees Celsius under standard atmospheric pressure."),
        ("What is the chemical symbol for gold?", "Au", "The chemical symbol for gold is Au, from the Latin word aurum."),
        ("How many sides does a hexagon have?", "6", "A hexagon is a polygon that has exactly 6 sides."),
        ("What planet is closest to the Sun?", "Mercury", "Mercury is the planet closest to the Sun in our solar system."),
        ("What is the square root of 144?", "12", "The square root of 144 is 12."),
        ("In what year did World War II end?", "1945", "World War II ended in the year 1945."),
        ("What is the speed of light in m/s?", "299792458", "The speed of light in a vacuum is 299792458 meters per second."),
        ("What gas do plants absorb during photosynthesis?", "carbon dioxide", "Plants absorb carbon dioxide during the process of photosynthesis."),
    ]

    # qa_1 and qa_2 pick from different halves of the list.
    if task_name == "qa_1":
        pairs = qa_pairs[: len(qa_pairs) // 2]
    else:
        pairs = qa_pairs[len(qa_pairs) // 2 :]

    question_text, answer_str, fact_sentence = pairs[sample_idx % len(pairs)]
    depth = 0.3 + rng.random() * 0.4  # embed somewhere in the middle

    filler = _make_filler(context_length - 30)
    ctx = _insert_at(filler, fact_sentence, depth)

    return TaskInstance(
        task_id=f"{task_name}__{context_length}__{sample_idx}",
        task_name=task_name,
        context_length=context_length,
        context_text=ctx,
        question=question_text,
        answer=[answer_str],
    )


# ------------------------------------------------------------------ #
# Public factory                                                       #
# ------------------------------------------------------------------ #


def generate_synthetic_task(
    task_name: str, context_length: int, sample_idx: int = 0
) -> TaskInstance:
    """Generate a synthetic RULER task instance offline (no download needed).

    Parameters
    ----------
    task_name:
        One of the 13 RULER task names in ``RULER_TASKS``.
    context_length:
        Approximate context length in tokens (e.g. 4096, 8192, …).
    sample_idx:
        Index to vary the needle/fact used; useful for generating multiple
        samples per (task, context_length) cell.

    Returns
    -------
    TaskInstance
        Fully populated instance with context, question, and answer list.
    """
    if task_name not in RULER_TASKS:
        raise ValueError(f"Unknown task: {task_name!r}. Must be one of {RULER_TASKS}")

    if task_name == "niah_single_1":
        return _gen_niah_single(task_name, context_length, sample_idx, n_needles=1)
    elif task_name == "niah_single_2":
        return _gen_niah_single(task_name, context_length, sample_idx, n_needles=1)
    elif task_name == "niah_single_3":
        return _gen_niah_single(task_name, context_length, sample_idx, n_needles=1)
    elif task_name == "niah_multikey_1":
        return _gen_niah_single(task_name, context_length, sample_idx, n_needles=1)
    elif task_name == "niah_multikey_2":
        return _gen_niah_single(task_name, context_length, sample_idx, n_needles=2)
    elif task_name == "niah_multikey_3":
        return _gen_niah_single(task_name, context_length, sample_idx, n_needles=3)
    elif task_name == "niah_multivalue":
        return _gen_niah_multivalue(context_length, sample_idx)
    elif task_name == "niah_multiquery":
        return _gen_niah_multiquery(context_length, sample_idx)
    elif task_name == "vt":
        return _gen_vt(context_length, sample_idx)
    elif task_name == "cwe":
        return _gen_cwe(context_length, sample_idx)
    elif task_name == "fwe":
        return _gen_fwe(context_length, sample_idx)
    elif task_name in ("qa_1", "qa_2"):
        return _gen_qa(task_name, context_length, sample_idx)

    raise AssertionError(f"Unhandled task: {task_name!r}")  # should never reach here
