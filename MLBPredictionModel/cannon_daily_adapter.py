"""Merge Cannon daily game projections with SportsLine odds.

Produces moneyline + totals picks with EV for each game,
returned as a list of dicts ready for JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from MLBPredictionModel.cannon_analytics import fetch_cannon_game_projections_raw
from MLBPredictionModel import sportsline_odds


# ---------------------------------------------------------------------------
# Odds / probability helpers
# ---------------------------------------------------------------------------

def prob_to_american(p: float) -> int:
    """Convert a win probability (0-1) to fair American odds."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    prob_pct = p * 100.0
    if prob_pct >= 50.0:
        return int(round(-(100 * prob_pct) / (100 - prob_pct)))
    else:
        return int(round(100 * (100 - prob_pct) / prob_pct))


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        o = abs(o)
        return o / (o + 100.0)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    o = float(odds)
    if o > 0:
        return o / 100.0 + 1.0
    else:
        return 100.0 / abs(o) + 1.0


def expected_value_pct(p_model: float, market_american: int) -> float:
    """EV per 1 unit stake as a fraction (0.05 == +5% expected return)."""
    dec = american_to_decimal(market_american)
    return p_model * (dec - 1.0) - (1.0 - p_model)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CannonGameRow:
    away_team: str
    home_team: str
    away_sp: str
    home_sp: str
    away_xr: float
    home_xr: float
    total_xr: float
    away_xwin: float  # 0-1
    home_xwin: float  # 0-1


@dataclass
class GamePickRow:
    away_team: str
    home_team: str
    away_sp: str
    home_sp: str

    cannon_away_xr: float
    cannon_home_xr: float
    cannon_total_xr: float
    cannon_away_xwin: float
    cannon_home_xwin: float

    # Moneyline pick
    ml_pick_side: Optional[str]        # "away", "home", or None
    ml_pick_team: Optional[str]
    ml_fair_odds: Optional[int]
    ml_market_odds: Optional[int]
    ml_edge_pct: Optional[float]

    # Total pick
    total_line: Optional[float]
    total_pick_side: Optional[str]     # "over", "under", or None
    total_market_odds: Optional[int]
    total_edge_pct: Optional[float]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_cannon_games(raw_games: list[dict]) -> list[CannonGameRow]:
    rows: list[CannonGameRow] = []
    for g in raw_games:
        try:
            rows.append(
                CannonGameRow(
                    away_team=g.get("away_team", ""),
                    home_team=g.get("home_team", ""),
                    away_sp=g.get("away_sp", ""),
                    home_sp=g.get("home_sp", ""),
                    away_xr=float(g.get("away_xR", 0.0)),
                    home_xr=float(g.get("home_xR", 0.0)),
                    total_xr=float(g.get("total_xR", 0.0)),
                    away_xwin=float(g.get("away_xWin", 0.0)),
                    home_xwin=float(g.get("home_xWin", 0.0)),
                )
            )
        except Exception:
            continue
    return rows


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _last_word(s: str) -> str:
    parts = _norm(s).split()
    return parts[-1] if parts else ""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_cannon_daily_picks(edge_threshold: float = 0.0) -> list[dict[str, Any]]:
    """Merge Cannon daily game projections with SportsLine odds.

    Computes a moneyline pick and a totals pick per game (with EV).
    Returns a list of dicts ready to be dumped to JSON for the frontend.
    """
    raw_cannon = fetch_cannon_game_projections_raw()
    cannon_games = _normalize_cannon_games(raw_cannon)

    sportsline_games = sportsline_odds.get_today_mlb_odds()

    # Index SportsLine games by (away_last_word, home_last_word)
    sl_index: dict[tuple[str, str], dict] = {}
    for g in sportsline_games:
        key = (_last_word(g["away_team"]), _last_word(g["home_team"]))
        sl_index[key] = g

    out: list[dict[str, Any]] = []
    for cg in cannon_games:
        key = (_last_word(cg.away_team), _last_word(cg.home_team))
        sl = sl_index.get(key)

        if not sl:
            out.append(
                {
                    "away_team": cg.away_team,
                    "home_team": cg.home_team,
                    "away_sp": cg.away_sp,
                    "home_sp": cg.home_sp,
                    "cannon_away_xr": cg.away_xr,
                    "cannon_home_xr": cg.home_xr,
                    "cannon_total_xr": cg.total_xr,
                    "cannon_away_xwin": cg.away_xwin,
                    "cannon_home_xwin": cg.home_xwin,
                    "ml_pick_side": None,
                    "ml_pick_team": None,
                    "ml_fair_odds": None,
                    "ml_market_odds": None,
                    "ml_edge_pct": None,
                    "total_line": None,
                    "total_pick_side": None,
                    "total_market_odds": None,
                    "total_edge_pct": None,
                }
            )
            continue

        # ------------- Moneyline pick -------------
        ml_away = sl.get("ml_away")
        ml_home = sl.get("ml_home")

        ml_pick_side = None
        ml_pick_team = None
        ml_fair_odds = None
        ml_market_odds = None
        ml_edge = None

        if cg.away_xwin > cg.home_xwin and cg.away_xwin > 0.5:
            ml_pick_side = "away"
            ml_pick_team = cg.away_team
            p = cg.away_xwin
            ml_market_odds = int(ml_away) if ml_away is not None else None
        elif cg.home_xwin >= cg.away_xwin and cg.home_xwin > 0.5:
            ml_pick_side = "home"
            ml_pick_team = cg.home_team
            p = cg.home_xwin
            ml_market_odds = int(ml_home) if ml_home is not None else None
        else:
            p = None

        if p is not None:
            ml_fair_odds = prob_to_american(p)
            if ml_market_odds is not None:
                ml_edge = expected_value_pct(p, ml_market_odds)
                if ml_edge < edge_threshold:
                    ml_pick_side = None
                    ml_pick_team = None
                    ml_fair_odds = None
                    ml_market_odds = None
                    ml_edge = None

        # ------------- Total pick (OU) -------------
        total_line = sl.get("total_line")
        total_over_odds = sl.get("total_over_odds")
        total_under_odds = sl.get("total_under_odds")

        total_pick_side = None
        total_market_odds = None
        total_edge = None

        if total_line is not None:
            if cg.total_xr > float(total_line):
                total_pick_side = "over"
                total_market_odds = int(total_over_odds) if total_over_odds is not None else None
                p_total = 0.55
            elif cg.total_xr < float(total_line):
                total_pick_side = "under"
                total_market_odds = int(total_under_odds) if total_under_odds is not None else None
                p_total = 0.55
            else:
                p_total = None

            if p_total is not None and total_market_odds is not None:
                total_edge = expected_value_pct(p_total, total_market_odds)
                if total_edge < edge_threshold:
                    total_pick_side = None
                    total_market_odds = None
                    total_edge = None

        out.append(
            {
                "away_team": cg.away_team,
                "home_team": cg.home_team,
                "away_sp": cg.away_sp,
                "home_sp": cg.home_sp,
                "cannon_away_xr": cg.away_xr,
                "cannon_home_xr": cg.home_xr,
                "cannon_total_xr": cg.total_xr,
                "cannon_away_xwin": cg.away_xwin,
                "cannon_home_xwin": cg.home_xwin,
                "ml_pick_side": ml_pick_side,
                "ml_pick_team": ml_pick_team,
                "ml_fair_odds": ml_fair_odds,
                "ml_market_odds": ml_market_odds,
                "ml_edge_pct": round(ml_edge, 4) if ml_edge is not None else None,
                "total_line": float(total_line) if total_line is not None else None,
                "total_pick_side": total_pick_side,
                "total_market_odds": total_market_odds,
                "total_edge_pct": round(total_edge, 4) if total_edge is not None else None,
            }
        )

    return out
