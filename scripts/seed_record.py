"""
seed_record.py
One-time script to seed hdav4873@gmail.com's existing 604-531 record.
SAFE: only writes if the record doesn't already exist OR if wins < 604.
Run once from repo root: python3 scripts/seed_record.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.local")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth
    FIREBASE_AVAILABLE = True
except ImportError:
    print("ERROR: pip install firebase-admin python-dotenv")
    exit(1)


def _init_firebase():
    if firebase_admin._apps:
        return firestore.client()
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


def get_uid_for_email(email: str) -> str | None:
    try:
        user = auth.get_user_by_email(email)
        return user.uid
    except Exception as e:
        print(f"Could not find UID for {email}: {e}")
        return None


def seed_record(email: str, wins: int, losses: int, pushes: int = 0):
    db  = _init_firebase()
    uid = get_uid_for_email(email)

    if uid is None:
        print(f"UID not found for {email}.")
        print("This means the user has never signed into the app yet.")
        print("Have them sign in once with Google, then re-run this script.")
        return

    ref  = db.collection('users').document(uid).collection('record').document('summary')
    snap = ref.get()

    if snap.exists():
        data = snap.to_dict()
        if data.get('wins', 0) >= wins:
            print(f"✓ Record already seeded: {data['wins']}-{data['losses']} — no change made.")
            return

    ref.set({'wins': wins, 'losses': losses, 'pushes': pushes})
    print(f"✓ Seeded record for {email} (uid: {uid}): {wins}-{losses}-{pushes}")


if __name__ == "__main__":
    seed_record(
        email="hdav4873@gmail.com",
        wins=604,
        losses=531,
        pushes=0
    )
