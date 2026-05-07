from __future__ import annotations

from copy import deepcopy
from typing import Any


class FakeDocumentSnapshot:
    def __init__(self, doc_id: str, data: dict[str, Any] | None):
        self.id = doc_id
        self._data = deepcopy(data) if data is not None else None
        self.exists = data is not None

    def to_dict(self) -> dict[str, Any] | None:
        return deepcopy(self._data) if self._data is not None else None


class FakeDocumentReference:
    def __init__(self, store: dict[str, dict[str, Any]], doc_id: str):
        self._store = store
        self._doc_id = doc_id

    def get(self) -> FakeDocumentSnapshot:
        return FakeDocumentSnapshot(self._doc_id, self._store.get(self._doc_id))

    def set(self, payload: dict[str, Any], merge: bool = False) -> None:
        if merge:
            current = self._store.setdefault(self._doc_id, {})
            current.update(deepcopy(payload))
            return
        self._store[self._doc_id] = deepcopy(payload)


class FakeCollectionReference:
    def __init__(self, store: dict[str, dict[str, Any]]):
        self._store = store

    def stream(self) -> list[FakeDocumentSnapshot]:
        return [
            FakeDocumentSnapshot(doc_id, data)
            for doc_id, data in self._store.items()
        ]

    def document(self, doc_id: str) -> FakeDocumentReference:
        return FakeDocumentReference(self._store, doc_id)


class FakeFirestoreClient:
    def __init__(self, users: dict[str, dict[str, Any]]):
        self.users = users

    def collection(self, name: str) -> FakeCollectionReference:
        assert name == "users"
        return FakeCollectionReference(self.users)


def test_background_grader_preserves_existing_record(monkeypatch):
    import pickgrader_server

    record = {"wins": 604, "losses": 531, "pushes": 0}
    users = {
        "test-pickledger-smoke": {
            "record": deepcopy(record),
            "picks": [
                {
                    "id": "smoke-pick",
                    "sport": "NBA",
                    "date": "Jan 1",
                    "pick": "Lakers ML (Lakers vs Celtics)",
                }
            ],
            "results": {"smoke-pick": "pending"},
            "startTimes": {},
            "ledger": {
                "addedPicks": [],
                "results": {"smoke-pick": "pending"},
                "gameTimes": {},
            },
        }
    }

    monkeypatch.setattr(
        pickgrader_server,
        "_get_firestore_client",
        lambda: FakeFirestoreClient(users),
    )
    monkeypatch.setattr(
        pickgrader_server,
        "auto_grade",
        lambda picks, existing, year: {
            "graded": {"smoke-pick": "win"},
            "startTimes": {"smoke-pick": "2025-01-01T20:00:00Z"},
            "summary": {"attempted": 1, "updated": 1, "remaining": 0},
        },
    )

    summary = pickgrader_server.run_background_grade_all_users()

    assert summary["graded_users"] == 1
    assert not summary["errors"]
    user_doc = users["test-pickledger-smoke"]
    assert user_doc["record"] == record
    assert user_doc["results"]["smoke-pick"] == "win"
    assert user_doc["ledger"]["results"]["smoke-pick"] == "win"
    assert user_doc["startTimes"]["smoke-pick"] == "2025-01-01T20:00:00Z"
    assert "lastGraded" in user_doc


def test_grade_pick_moneyline_result_without_network():
    import pickgrader_server

    game = {
        "competitors": [
            {
                "raw": {
                    "team": {
                        "displayName": "Los Angeles Lakers",
                        "shortDisplayName": "Lakers",
                        "name": "Lakers",
                        "abbreviation": "LAL",
                    }
                },
                "score": 112,
                "homeAway": "home",
                "linescores": [],
            },
            {
                "raw": {
                    "team": {
                        "displayName": "Boston Celtics",
                        "shortDisplayName": "Celtics",
                        "name": "Celtics",
                        "abbreviation": "BOS",
                    }
                },
                "score": 100,
                "homeAway": "away",
                "linescores": [],
            },
        ],
        "startTime": "2025-01-01T20:00:00Z",
        "eventId": "smoke",
    }
    pick = {
        "id": "smoke-pick",
        "sport": "NBA",
        "pick": "Lakers ML (Lakers vs Celtics)",
    }

    assert pickgrader_server.grade_pick(pick, game) == "win"
