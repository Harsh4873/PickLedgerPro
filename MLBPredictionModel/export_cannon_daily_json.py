"""Export Cannon + SportsLine daily picks to a JSON file for the frontend."""

import json
from pathlib import Path

from MLBPredictionModel.cannon_daily_adapter import (
    build_cannon_daily_picks,
    build_cannon_pick_rows,
)
from MLBPredictionModel.date_utils import get_mlb_slate_date


def main() -> None:
    slate_date = get_mlb_slate_date()
    print(f"[export_cannon_daily_json] Exporting Cannon for slate_date={slate_date}")

    rows = build_cannon_daily_picks(slate_date=slate_date, edge_threshold=0.0)
    picks = build_cannon_pick_rows(games=rows, slate_date=slate_date, edge_threshold=0.0)
    out = {
        "as_of": slate_date.isoformat(),
        "slate_date": slate_date.isoformat(),
        "games": rows,
        "picks": picks,
    }

    out_path = Path(__file__).resolve().parent.parent / "data" / "cannon_mlb_daily.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} games and {len(picks)} pick rows to {out_path}")


if __name__ == "__main__":
    main()
