# PickLedgerPro Architecture Guardrails

This repo has a live data path that should be treated as locked unless a change
is explicitly planned as a migration:

Firebase Admin SDK -> `picks/latest` and `admin_picks/{date}` -> `index.html`
-> `users/{uid}.record` and ledger state.

The previous breakage mode was caused by moving or renaming files and Firebase
keys that still had live consumers. The goal of this document is to make those
couplings visible before the next cleanup pass.

## Locked Files And Directories

| Locked item | Current location | Why it stays here |
| --- | --- | --- |
| `pickgrader_server.py` | repo root | Defines `BASE_DIR` from its own file and resolves model dirs, `data/`, `pickledger.db`, and local env files from there. |
| `runlive.py` | repo root | Uses the same root `BASE_DIR` contract and launches `NBAPredictionModel/run_live.py` with that model directory as cwd. |
| `index.html` | repo root | Served directly by GitHub Pages and references root-relative frontend assets such as `./ipl/ipl_styles.css` and `./ipl/ipl_frontend.js`. |
| `firestore.rules`, `firestore.indexes.json` | repo root | Firebase CLI deploys these from the repository root. |
| `.env.local` | repo root | Loaded by `pickgrader_server.py`, `scripts/firebase_writer.py`, and `scripts/seed_record.py`. |
| `.nojekyll` | repo root | Copied into the GitHub Pages build output by the Pages workflow. |
| `requirements.txt` | repo root | The Docker image copies this exact file. |
| `Dockerfile` | repo root | Render deploy entry point. |
| `data/` | repo root | Read by backend paths rooted at `BASE_DIR/data`; Pages also publishes selected files from here. |
| `pickledger.db` | repo root | Looked up as `pickledger.db` or `../pickledger.db` by backend helpers. |
| `scripts/grader_loop.py` | `scripts/` | Hardcoded by `scripts/com.pickledger.grader.plist`. |
| `scripts/firebase_writer.py`, `scripts/seed_record.py` | `scripts/` | Load `.env.local` relative to the repo root; `seed_record.py` preserves the admin recovery floor. |
| `MLBPredictionModel/`, `NBAPredictionModel/`, `WNBAPredictionModel/`, `NBAPlayerBettingModel/`, `NBAPlayoffsPredictionModel/`, `ipl/`, `models/mlb_inning/` | repo root | Imported or launched by exact path/name from the backend and frontend. |
| `MLBPredictionModel/Report1.tex`, `NBAPredictionModel/Report1.tex`, related `compile.sh` files | model directories | The compile scripts run LaTeX in place and copy PDFs back inside the same model directory. |
| `MLBPredictionModel/artifacts/*_new.joblib` | `MLBPredictionModel/artifacts/` | The backend exposes old and new MLB variants separately even when some artifacts are byte-identical. |

## Firestore Contract

Do not rename these collections, documents, or fields in place. If a future
change requires a different shape, dual-read and dual-write first, then migrate.

| Path | Shape | Readers | Writers |
| --- | --- | --- | --- |
| `/picks/latest` | Latest shared picks payload with JSON-encoded `picks`, `date`, `model`, `timestamp`, `games_evaluated`, and optional `notes`. | `index.html` model-results banner and local smoke tests. | `scripts/firebase_writer.py` via Admin SDK. |
| `/admin_picks/{date}` | Per-date model cache fields such as `nba`, `nba_new`, `nba_old`, `nba_playoffs`, `wnba`, `mlb`, `mlb_old`, `mlb_new`, `mlb_inning`, `props`, `ipl`, and matching `*_ts` timestamps. | `index.html` model cards and daily results views. | `pickgrader_server.py` model runner helpers via Admin SDK. |
| `/users/{uid}` | Owner-only user document. Key fields include `record: {wins, losses, pushes}`, `ledger`, `picks`, `results`, `startTimes`, `savedAt`, `lastSynced`, and `lastGraded`. | `index.html`, background grader, smoke tests. | Authenticated client for the owner, plus Admin SDK grading writes. |
| `/users/{uid}/record/summary` | Legacy record summary retained for migration/backward compatibility. | `index.html` fallback loader, `scripts/seed_record.py`. | `scripts/seed_record.py` recovery path. |

The lifetime record belongs under `/users/{uid}.record`. Background grading
updates `results`, `startTimes`, `ledger.results`, `ledger.gameTimes`, and
`lastGraded`; it must not reset or overwrite `record`.

## Root `BASE_DIR` Contract

`pickgrader_server.py` and `runlive.py` intentionally live at the repository
root. Their `BASE_DIR = dirname(__file__)` assumptions are used for:

- model directory lookup;
- root `data/` lookup;
- `pickledger.db` and `pickledger_state.json` lookup;
- `.env` and `.env.local` loading;
- subprocess cwd choices for model runners.

Moving either file is a backend migration, not a cleanup.

## LaunchAgent Coupling

`scripts/com.pickledger.grader.plist` launches
`REPO_PATH_PLACEHOLDER/scripts/grader_loop.py`. The installer replaces the repo
path but still depends on `scripts/grader_loop.py` staying under `scripts/`.
`grader_loop.py` imports `pickgrader_server.run_background_grade_all_users`, so
that backend import must remain valid from the repo root.

## Firebase Web Config

The Firebase Web SDK `apiKey` embedded in `index.html` is a public project
identifier, not a service-account secret. Security comes from
`firestore.rules`: clients must be signed in, user documents are owner-scoped,
shared pick caches are client read-only, and unmatched paths default-deny.

Do not commit Admin SDK credentials. `.env.local`, `.env`, service-account JSON
files, and Firebase CLI local config are ignored.

## Adding A New Sport Or Model

1. Mirror an existing model directory shape at the repo root.
2. Register the runner in `pickgrader_server.py` without renaming existing keys.
3. Add the model card or results view in `index.html`.
4. Add a new field under `/admin_picks/{date}` instead of changing existing
   fields.
5. Update smoke tests and this document with the new import/path contract.

Keep existing field names such as `ipl`, `mlb_new`, `nba_new`, and `props`.
Those strings are part of the frontend and Firestore contract.

## Refactor No-Touch List

- Do not move `pickgrader_server.py`, `runlive.py`, or `index.html`.
- Do not rename Firestore collections, documents, or record fields.
- Do not rename `ipl/` unless every frontend, backend, workflow, and Firestore
  key is migrated with dual reads.
- Do not consolidate duplicate model helper files unless imports are tested from
  the same cwd used by production subprocesses.
- Do not delete `_new` artifacts just because they currently match older
  artifacts.
- Do not move model research TeX/PDF files without replacing the compile scripts.

## Smoke Tests

Run these after structural changes:

```bash
python -c "import pickgrader_server"
python -c "from MLBPredictionModel import cannon_daily_adapter, date_utils"
python -c "import scripts.firebase_writer"
python -m pytest tests/smoke -v
```

`tests/smoke/test_firestore_read.py` is read-only. It loads Admin SDK
credentials from `.env.local`, reads `/picks/latest`, and optionally checks
`/users/test-pickledger-smoke.record` if that fixture document exists.

`tests/smoke/test_grader_dry_run.py` uses an in-memory fake Firestore client. It
does not grade live users and exists specifically to prove backend grading does
not overwrite the lifetime `record` field.
