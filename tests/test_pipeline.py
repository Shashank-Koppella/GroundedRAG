from src.ingestion.pipeline import make_chunk_id


def test_make_chunk_id_strips_dashes_from_accession_number():
    cid = make_chunk_id("AAPL", "0000320193-25-000123", "item_1a", 0)
    assert cid == "AAPL_000032019325000123_item_1a_0000"


def test_make_chunk_id_unique_across_filings_same_index():
    """
    Regression test for the real bug: chunk_index alone resets to 0 for every
    section (chunk_text() is called once per section), so two different
    filings' item_1a chunk 0 would collide without the accession number in the
    id. This is the actual scenario that broke — same ticker, same item, same
    chunk_index, different filing.
    """
    id_a = make_chunk_id("AAPL", "0000320193-24-000050", "item_1a", 0)
    id_b = make_chunk_id("AAPL", "0000320193-25-000123", "item_1a", 0)
    assert id_a != id_b


def test_make_chunk_id_unique_across_items_same_filing():
    id_a = make_chunk_id("AAPL", "0000320193-25-000123", "item_1a", 0)
    id_b = make_chunk_id("AAPL", "0000320193-25-000123", "item_7", 0)
    assert id_a != id_b


def test_make_chunk_id_pads_chunk_index():
    cid = make_chunk_id("MSFT", "0000789019-25-000001", "item_7", 3)
    assert cid.endswith("_0003")
