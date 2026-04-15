"""
experimental_splits.py
-----------------------
Utility for creating time-based train / val / test splits from the historical
dataset CSV.  Nothing here touches any production model or artifact.

Default split boundaries:
  train  – full 2024 regular season
  val    – full 2025 regular season
  test   – 2026-to-date (only completed games already in the dataset)

Usage
-----
  from experimental_splits import load_splits, split_summary
  splits = load_splits()           # uses default DATASET_PATH
  print(split_summary(splits))
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from historical_data import DATASET_PATH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_splits(
    dataset_path: Path = DATASET_PATH,
    *,
    train_seasons: tuple[int, ...] = (2024,),
    val_seasons: tuple[int, ...] = (2025,),
    test_min_year: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Load time-based splits from the historical dataset CSV.

    Parameters
    ----------
    dataset_path:
        Path to the CSV produced by ``build_historical_dataset.py``.
    train_seasons:
        Calendar years whose games go into the training split.
        Default: 2024 only.
    val_seasons:
        Calendar years whose games go into the validation split.
        Default: 2025 only.
    test_min_year:
        Games from this year onward (and present in the CSV, meaning
        completed) go into the test split.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found at {dataset_path}.\n"
            "Run `python build_historical_dataset.py --seasons 2024 2025 2026`"
            " (or the corresponding historical_data.py entry-point) first."
        )

    frame = pd.read_csv(dataset_path, parse_dates=["game_date"])
    frame = (
        frame.sort_values(["season", "game_date", "game_pk"])
        .reset_index(drop=True)
    )

    year = frame["game_date"].dt.year

    train = frame[year.isin(train_seasons)].copy().reset_index(drop=True)
    val   = frame[year.isin(val_seasons)].copy().reset_index(drop=True)
    test  = frame[year >= test_min_year].copy().reset_index(drop=True)

    return {"train": train, "val": val, "test": test}


def split_summary(splits: dict[str, pd.DataFrame]) -> str:
    """Return a human-readable summary of each split."""
    lines: list[str] = []
    for name, df in splits.items():
        if len(df) == 0:
            lines.append(f"  {name:>5}: 0 rows (no data for this period)")
            continue
        date_lo = df["game_date"].min().date()
        date_hi = df["game_date"].max().date()
        seasons  = sorted(df["game_date"].dt.year.unique().tolist())
        has_odds = int(df["home_moneyline"].notna().sum()) if "home_moneyline" in df.columns else 0
        pct_odds = f"{100*has_odds/len(df):.0f}%" if len(df) else "n/a"
        lines.append(
            f"  {name:>5}: {len(df):>5,} rows | "
            f"{date_lo} → {date_hi} | "
            f"seasons={seasons} | "
            f"odds coverage={pct_odds}"
        )
    return "\n".join(lines)


def dataset_info(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    """Return basic info about the full dataset without loading all splits."""
    if not dataset_path.exists():
        return {"exists": False, "path": str(dataset_path)}
    frame = pd.read_csv(dataset_path, parse_dates=["game_date"])
    seasons = sorted(frame["game_date"].dt.year.dropna().unique().astype(int).tolist())
    return {
        "exists": True,
        "path": str(dataset_path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "seasons": seasons,
        "date_range": (
            str(frame["game_date"].min().date()),
            str(frame["game_date"].max().date()),
        ),
        "has_odds_col": "home_moneyline" in frame.columns,
        "odds_coverage_pct": (
            round(100 * frame["home_moneyline"].notna().mean(), 1)
            if "home_moneyline" in frame.columns
            else None
        ),
    }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    info = dataset_info()
    print("=== Dataset Info ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    if info["exists"]:
        splits = load_splits()
        print("=== Split Summary ===")
        print(split_summary(splits))
