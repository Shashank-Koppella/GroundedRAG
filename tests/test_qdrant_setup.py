"""
Tests for src/eval/qdrant_setup.py point-id assignment (regression tests for the
Day 5 discovery that the Day 3 index was silently holding ~2,192 of 11,245 chunks).

The original code used `id=i` from enumerate(). Because qdrant_setup.py runs once per
company, that counter restarted at 0 every run and each company's upload overwrote the
previous company's points -- silently, because Qdrant's upsert treats a colliding id as
an update rather than an error. These tests pin the fix: point ids are derived from the
globally-unique chunk_id Day 3's make_chunk_id() already produces.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.qdrant_setup import make_point_id, upsert_chunks


def test_same_chunk_id_gives_same_point_id():
    """Idempotency: re-uploading one company must update its points in place, not
    duplicate them."""
    assert make_point_id("AAPL_000032019324000123_item_1a_0010") == \
           make_point_id("AAPL_000032019324000123_item_1a_0010")


def test_different_chunk_ids_give_different_point_ids():
    assert make_point_id("AAPL_000032019324000123_item_1a_0010") != \
           make_point_id("AAPL_000032019324000123_item_1a_0011")


def test_first_chunk_of_different_companies_does_not_collide():
    """THE regression test for the Day 5 bug.

    Under the old `id=i` scheme, every company's first chunk got point id 0, so each
    per-company upload wiped the last one. Different companies' first chunks must map
    to different point ids."""
    aapl_first = make_point_id("AAPL_000032019324000123_item_1a_0000")
    msft_first = make_point_id("MSFT_000119312526323660_item_1a_0000")
    googl_first = make_point_id("GOOGL_000165204425000043_item_1a_0000")
    assert len({aapl_first, msft_first, googl_first}) == 3


def test_point_id_is_a_valid_uuid_string():
    """Qdrant accepts only int or UUID point ids -- an arbitrary string would be
    rejected at upload time."""
    pid = make_point_id("AAPL_000032019324000123_item_1a_0010")
    uuid.UUID(pid)  # raises ValueError if not a well-formed UUID
    assert isinstance(pid, str)


class FakeQdrantClient:
    """Captures every upserted point so tests can inspect the ids actually assigned."""
    def __init__(self):
        self.upserted = []

    def upsert(self, collection_name, points):
        self.upserted.extend(points)


def _row(chunk_id, ticker):
    return {"chunk_id": chunk_id, "ticker": ticker, "item": "item_1a",
            "text": "some text", "embedding": [0.1, 0.2, 0.3]}


def test_upsert_assigns_ids_from_chunk_id_not_enumerate_index():
    client = FakeQdrantClient()
    rows = [
        _row("AAPL_000032019324000123_item_1a_0000", "AAPL"),
        _row("AAPL_000032019324000123_item_1a_0001", "AAPL"),
    ]
    upsert_chunks(client, rows)

    ids = [p.id for p in client.upserted]
    assert ids == [make_point_id(r["chunk_id"]) for r in rows]
    assert 0 not in ids and 1 not in ids, "ids must not be positional indices"


def test_two_separate_company_uploads_do_not_overwrite_each_other():
    """Simulates the real failure mode: two sequential per-company upload runs against
    the same collection. Under the old scheme both runs wrote ids 0..n and the second
    destroyed the first; all four points must now survive with distinct ids."""
    client = FakeQdrantClient()

    upsert_chunks(client, [
        _row("AAPL_000032019324000123_item_1a_0000", "AAPL"),
        _row("AAPL_000032019324000123_item_1a_0001", "AAPL"),
    ])
    upsert_chunks(client, [
        _row("MSFT_000119312526323660_item_1a_0000", "MSFT"),
        _row("MSFT_000119312526323660_item_1a_0001", "MSFT"),
    ])

    ids = [p.id for p in client.upserted]
    assert len(ids) == 4
    assert len(set(ids)) == 4, "per-company uploads collided -- the Day 5 bug is back"


def test_payload_still_carries_chunk_id_and_ticker():
    """The ticker payload field is what the dense retriever's filter matches on, so a
    regression here would silently break company-scoped retrieval."""
    client = FakeQdrantClient()
    upsert_chunks(client, [_row("AAPL_000032019324000123_item_1a_0000", "AAPL")])
    payload = client.upserted[0].payload
    assert payload["chunk_id"] == "AAPL_000032019324000123_item_1a_0000"
    assert payload["ticker"] == "AAPL"
