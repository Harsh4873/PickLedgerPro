from __future__ import annotations

import os

import pytest


REQUIRED_FIREBASE_ENV = (
    "FIREBASE_PROJECT_ID",
    "FIREBASE_PRIVATE_KEY",
    "FIREBASE_CLIENT_EMAIL",
)


def test_firestore_shared_paths_are_readable():
    pytest.importorskip("firebase_admin")

    from scripts import firebase_writer

    missing = [name for name in REQUIRED_FIREBASE_ENV if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing Firebase env vars: {', '.join(missing)}")

    db = firebase_writer._init_firebase()
    if db is None:
        pytest.skip("Firebase Admin client is not configured")

    latest = db.collection("picks").document("latest").get()
    assert latest.exists, "Expected /picks/latest to exist"

    smoke_uid = os.getenv("PICKLEDGER_SMOKE_UID", "test-pickledger-smoke")
    user_doc = db.collection("users").document(smoke_uid).get()
    if not user_doc.exists:
        pytest.skip(f"Smoke user fixture /users/{smoke_uid} has not been created")

    record = (user_doc.to_dict() or {}).get("record")
    assert isinstance(record, dict), f"Expected /users/{smoke_uid}.record to be an object"
    assert {"wins", "losses", "pushes"}.issubset(record)
