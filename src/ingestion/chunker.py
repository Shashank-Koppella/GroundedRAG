"""
Fixed-size + overlap chunking. Deliberately simple (word-count based, no
semantic/sentence-aware splitting) — the plan explicitly calls this out as
not worth over-engineering on Day 1. Revisit only if retrieval eval in
Phase A, Day 5 shows chunk-boundary artifacts hurting Recall@k.
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    chunk_index: int
    start_word: int
    end_word: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size_words: int = 250,
    overlap_words: int = 40,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping fixed-size chunks by word count.

    chunk_size_words=250 / overlap_words=40 is a starting point (~15% overlap),
    not a tuned value — cheap to revisit as an ablation later since the eval
    harness (Phase A, Day 3) will make the effect of chunk size measurable
    instead of guessed at.
    """
    words = text.split()
    if not words:
        return []
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    chunks = []
    step = chunk_size_words - overlap_words
    idx = 0
    chunk_index = 0
    while idx < len(words):
        window = words[idx : idx + chunk_size_words]
        chunk_text_str = " ".join(window)
        chunks.append(
            Chunk(
                text=chunk_text_str,
                chunk_index=chunk_index,
                start_word=idx,
                end_word=idx + len(window),
                metadata=dict(metadata or {}),
            )
        )
        chunk_index += 1
        if idx + chunk_size_words >= len(words):
            break
        idx += step

    return chunks
