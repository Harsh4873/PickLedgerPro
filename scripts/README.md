# PickLedger Scripts

## `run_models.sh`
Runs all prediction models locally and saves results to Firebase.

**Prerequisites:**
- `.env` file at repo root with `ADMIN_PICKS_SECRET=<your secret>`
- Python deps installed with `pip install -r requirements.txt`

**Usage:**
```bash
# Today's slate
bash scripts/run_models.sh

# Specific date
bash scripts/run_models.sh 2026-04-13
```

The script starts the local server if it is not already running, runs all 5 model cache writes in sequence, and confirms each Firebase write. Other users on `pickledgerpro.onrender.com` will see the results immediately after.
