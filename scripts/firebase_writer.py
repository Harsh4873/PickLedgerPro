"""
firebase_writer.py
Writes NBANEW model picks to Firebase Firestore.
Run this from the repo root after generating picks locally.
Credentials are loaded from .env.local — never committed.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.local")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("WARNING: firebase-admin not installed. Run: pip install firebase-admin python-dotenv")


def _init_firebase():
    if not FIREBASE_AVAILABLE:
        return None
    if firebase_admin._apps:
        return firestore.client()
    required = [
        "FIREBASE_PROJECT_ID",
        "FIREBASE_PRIVATE_KEY",
        "FIREBASE_CLIENT_EMAIL",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"WARNING: Missing env vars: {missing}. Skipping Firestore write.")
        return None
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID", ""),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID", ""),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    firebase_admin.initialize_app(cred)
    return firestore.client()


def save_picks(picks: list, model: str = "NBANEW", notes: str = "") -> bool:
    """
    Save a list of picks to Firestore collection 'picks'.
    Document ID = ISO timestamp. Returns True on success.
    """
    db = _init_firebase()
    if db is None:
        print("Firestore unavailable — picks not saved to cloud.")
        return False

    now = datetime.now(timezone.utc)
    doc_id = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    date_str = now.strftime("%Y-%m-%d")

    doc = {
        "timestamp": now.isoformat(),
        "date": date_str,
        "model": model,
        "picks": json.dumps(picks),
        "games_evaluated": len(picks),
        "notes": notes,
    }

    db.collection("picks").document(doc_id).set(doc)

    # Also overwrite the 'latest' document so frontend can always
    # fetch a single stable document ID
    db.collection("picks").document("latest").set(doc)

    print(f"✓ Saved {len(picks)} picks to Firestore (doc: {doc_id})")
    return True


def get_latest_picks() -> dict:
    """Fetch the latest picks from Firestore. For local testing only."""
    db = _init_firebase()
    if db is None:
        return {}
    doc = db.collection("picks").document("latest").get()
    if not doc.exists:
        return {}
    data = doc.to_dict()
    data["picks"] = json.loads(data.get("picks", "[]"))
    return data


if __name__ == "__main__":
    # Quick test — write a dummy pick and read it back
    test_picks = [{"game": "TEST vs TEST", "model": "NBANEW", "pick": "test"}]
    success = save_picks(test_picks, notes="test run")
    if success:
        result = get_latest_picks()
        print(f"Read back: {result.get('games_evaluated')} picks, date: {result.get('date')}")
