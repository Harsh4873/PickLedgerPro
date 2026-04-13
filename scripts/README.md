## Architecture (Post-Render)

### What runs where
| Component | Where | How |
|---|---|---|
| Morning model runs | Your Mac | LaunchAgent at 8:30 AM -> `run_models.sh` -> Firebase |
| Background grading | Your Mac | LaunchAgent every 15 min -> `grader_loop.py` -> Firebase |
| Frontend | Render (static) or any CDN | Pure Firebase JS SDK - no backend calls |
| Per-user records | Firebase Firestore `users/{uid}` | Isolated per UID |
| Shared model picks | Firebase Firestore `admin_picks/{date}` | Written by admin, read by all |

### Install LaunchAgents (one-time)
```bash
source .env
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
bash scripts/install_grader_agent.sh
```

### Trigger models manually
```bash
bash scripts/run_models.sh
```

### Check grader is running
```bash
launchctl list | grep pickledger
tail -f ~/Library/Logs/pickledger_grader.log
```
