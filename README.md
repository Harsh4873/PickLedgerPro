# PickLedgerPro

PickLedgerPro is a full-stack sports pick tracking and algorithmic prediction system. It tracks picks across NBA, MLB, WNBA, and IPL with a live GitHub Pages frontend and Firebase-backed data.

## Live Site

https://harsh4873.github.io/PickLedgerPro/

## Architecture

| Layer        | Stack                                                                 |
|-------------|-----------------------------------------------------------------------|
| Frontend    | Single-page HTML/CSS/JS app (`index.html`) deployed via GitHub Pages  |
| Data Backend| Firebase Firestore (picks, user data, cached feeds)                   |
| Local Agent | `pickgrader_server.py` — local HTTP server for admin model runs       |
| Models      | Python: NBA, NBA Playoffs, MLB, WNBA, IPL                             |
| Automation  | GitHub Actions: MLB daily Cannon refresh, model training, Pages deploy|

## Repository Structure

```text
PickLedgerPro/
├── index.html                  # Frontend SPA
├── pickgrader_server.py        # Local admin backend server
├── runlive.py                  # Orchestrates model runs
├── requirements.txt            # Python dependencies
├── firestore.rules             # Firebase security rules
├── firestore.indexes.json      # Firestore composite indexes
├── data/                       # Static JSON served to GitHub Pages
│   ├── cannon_mlb_daily.json   # MLB Cannon daily projections (auto-refreshed)
│   └── picks_latest.json       # Latest picks snapshot for UI
├── scripts/                    # Local LaunchAgent & helper scripts
├── NBAPredictionModel/         # NBA moneyline + spread model
├── NBAPlayoffsPredictionModel/ # NBA Playoffs model
├── NBAPlayerBettingModel/      # NBA player props model
├── MLBPredictionModel/         # MLB moneyline + totals model
├── WNBAPredictionModel/        # WNBA prediction model
├── ipl/                        # IPL fantasy/win-prediction model
├── docs/                       # Deployment & infra documentation
└── .github/workflows/          # GitHub Actions CI/CD
```

## Setup

1. Copy `.env.example` to `.env` and fill in Firebase credentials.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the local backend (for admin-only actions):

   ```bash
   python pickgrader_server.py
   ```

4. For the UI, either open `index.html` locally or use the live GitHub Pages site.

## Automation

- **Cannon MLB Daily Refresh**: Regenerates `data/cannon_mlb_daily.json` on a schedule.
- **Deploy to GitHub Pages**: Runs on each push to `main`.
- **MLB Model Training**: Manual workflow dispatch in GitHub Actions.

## Notes

- Do not commit `.env`, Firebase admin JSON, or any API keys.
- Admin-only model run buttons in the UI require the local backend running on `127.0.0.1:8765`.
