from src.ingestion.chunker import chunk_text


def test_chunk_text_basic_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size_words=250, overlap_words=40)

    assert len(chunks) > 1
    # first chunk starts at word 0
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 250
    # second chunk should start 210 words in (step = 250-40)
    assert chunks[1].start_word == 210
    # overlap: last 40 words of chunk 0 == first 40 words of chunk 1
    words = text.split()
    assert words[210:250] == words[chunks[1].start_word : chunks[1].start_word + 40]


def test_chunk_text_short_input_single_chunk():
    text = "just a few words here"
    chunks = chunk_text(text, chunk_size_words=250, overlap_words=40)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_text_empty_input():
    assert chunk_text("", chunk_size_words=250, overlap_words=40) == []


def test_chunk_text_metadata_propagates():
    text = " ".join(f"w{i}" for i in range(500))
    meta = {"ticker": "AAPL", "item": "item_1a"}
    chunks = chunk_text(text, metadata=meta)
    assert all(c.metadata == meta for c in chunks)
    # mutating one chunk's metadata shouldn't affect another's (separate dicts)
    chunks[0].metadata["extra"] = "x"
    assert "extra" not in chunks[1].metadata


def test_chunk_text_rejects_overlap_ge_chunk_size():
    try:
        chunk_text("a b c", chunk_size_words=10, overlap_words=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
