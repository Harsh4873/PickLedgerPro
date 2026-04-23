# NBA Playoffs Prediction Model

This model is the playoff-specific NBA workflow for PickLedgerPro. It keeps the regular NBA model's live data layer where possible, but adds a playoff verification gate before producing picks.

## Verification Gate

`run_live.py` only emits picks for games that:

- appear on ESPN's NBA postseason scoreboard (`seasontype=3`),
- have not started yet,
- have team statistics available through the NBA API,
- have current market moneylines available from the ESPN scoreboard odds payload.

When a game fails those checks, the runner prints the reason and does not emit a pick.

## Data Sources

- ESPN scoreboard: playoff schedule, series headline, game status, venue, moneyline, spread, and total.
- NBA API: season team efficiency, recent form, rosters, and schedule/rest context.
- Existing NBA injury feed: current injury statuses and on/off impact adjustment.

## Execution

Run for a selected date:

```bash
../.venv/bin/python run_live.py --date 2026-04-23
```

The backend route is `/run-nba-playoffs-model`, and the Firestore cache key is `nba_playoffs`.
