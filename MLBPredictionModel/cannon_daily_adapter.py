"""Merge Cannon daily game projections with SportsLine odds.

Produces moneyline + totals picks with EV for each game,
returned as a list of dicts ready for JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from MLBPredictionModel.cannon_analytics import fetch_cannon_game_projections_raw
from MLBPredictionModel import sportsline_odds
from MLBPredictionModel.date_utils import get_mlb_slate_date


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


def _team_matches(left: str, right: str) -> bool:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if not left_norm or not right_norm:
        return False
    return (
        left_norm == right_norm
        or left_norm in right_norm
        or right_norm in left_norm
        or _last_word(left_norm) == _last_word(right_norm)
    )


def _filter_cannon_games_for_slate_date(
    games: list[CannonGameRow],
    slate_date: date,
) -> list[CannonGameRow]:
    scheduled_games = sportsline_odds.get_mlb_schedule_games_for_date(slate_date)
    if not scheduled_games:
        print(
            f"[cannon_daily_adapter] No official MLB schedule rows found for slate_date={slate_date}; "
            f"returning {len(games)} Cannon games without filtering."
        )
        return games

    filtered_games = [
        game
        for game in games
        if any(
            _team_matches(game.away_team, away_team)
            and _team_matches(game.home_team, home_team)
            for away_team, home_team in scheduled_games
        )
    ]
    print(
        f"[cannon_daily_adapter] Schedule filter for slate_date={slate_date}: "
        f"kept={len(filtered_games)} cannon_games={len(games)} official_games={len(scheduled_games)}"
    )
    return filtered_games


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_cannon_daily_picks(
    slate_date: date | None = None,
    edge_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Merge Cannon daily game projections with SportsLine odds.

    Computes a moneyline pick and a totals pick per game (with EV).
    Returns a list of dicts ready to be dumped to JSON for the frontend.
    """
    slate_date = slate_date or get_mlb_slate_date()
    print(f"[cannon_daily_adapter] Building Cannon daily picks for slate_date={slate_date}")
    raw_cannon = fetch_cannon_game_projections_raw()
    cannon_games = _filter_cannon_games_for_slate_date(_normalize_cannon_games(raw_cannon), slate_date)

    sportsline_games = sportsline_odds.get_mlb_odds_for_date(slate_date)

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


# ---------------------------------------------------------------------------
# Pick-row helpers (matching frontend BET/LEAN/PASS thresholds)
# ---------------------------------------------------------------------------

def _quarter_kelly_pct(p: float, american_odds: int, max_pct: float = 5.0) -> float | None:
    """Quarter-Kelly fraction as a percentage, capped at *max_pct*."""
    try:
        dec = american_to_decimal(american_odds)
        b = dec - 1.0
        if b <= 0:
            return None
        full_kelly = (b * p - (1.0 - p)) / b
        if full_kelly <= 0:
            return None
        return min(full_kelly * 0.25 * 100, max_pct)
    except Exception:
        return None


def _pick_decision(edge_pct: float) -> str:
    """BET / LEAN / PASS — mirrors the frontend constants
    ``MODEL_RESULTS_BET_EDGE_PCT = 5`` and ``MODEL_RESULTS_LEAN_EDGE_PCT = 3``.
    """
    if edge_pct >= 5.0:
        return "BET"
    elif edge_pct >= 3.0:
        return "LEAN"
    return "PASS"


# ---------------------------------------------------------------------------
# Flat pick-row builder (same schema the Model Results table expects)
# ---------------------------------------------------------------------------

def build_cannon_pick_rows(
    games: list[dict[str, Any]] | None = None,
    slate_date: date | None = None,
    edge_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Return one row per ML / total pick, formatted for the frontend
    Model Results table (source, pick, sport, odds, probability, edge,
    model_prediction, kelly_pct, decision ���).

    Pass *games* directly (output of ``build_cannon_daily_picks``) to
    avoid a redundant SportsLine fetch.
    """
    if games is None:
        games = build_cannon_daily_picks(slate_date=slate_date, edge_threshold=edge_threshold)
    rows: list[dict[str, Any]] = []

    for g in games:
        away = g["away_team"]
        home = g["home_team"]
        matchup = f"{away} vs {home}"

        # ---------- Moneyline row ----------
        if g.get("ml_pick_team") and g.get("ml_market_odds") is not None:
            p = (
                g["cannon_away_xwin"]
                if g["ml_pick_side"] == "away"
                else g["cannon_home_xwin"]
            )
            odds = g["ml_market_odds"]
            edge_pct = round((g.get("ml_edge_pct") or 0.0) * 100, 1)
            kelly = _quarter_kelly_pct(p, odds)
            decision = _pick_decision(edge_pct)
            rows.append(
                {
                    "source": "Cannon Analytics",
                    "pick": f"{g['ml_pick_team']} ML ({matchup})",
                    "matchup": matchup,
                    "sport": "MLB",
                    "odds": odds,
                    "probability": round(p, 4),
                    "edge": edge_pct,
                    "model_prediction": None,
                    "kelly_pct": round(kelly, 2) if kelly else None,
                    "decision": decision,
                    "away_team": away,
                    "home_team": home,
                }
            )

        # ---------- Total (O/U) row ----------
        if g.get("total_pick_side") and g.get("total_market_odds") is not None:
            p_total = 0.55
            odds = g["total_market_odds"]
            edge_pct = round((g.get("total_edge_pct") or 0.0) * 100, 1)
            kelly = _quarter_kelly_pct(p_total, odds)
            decision = _pick_decision(edge_pct)
            side = g["total_pick_side"].capitalize()
            line = g.get("total_line", "")
            rows.append(
                {
                    "source": "Cannon Analytics",
                    "pick": f"{side} {line} ({matchup})",
                    "matchup": matchup,
                    "sport": "MLB",
                    "odds": odds,
                    "probability": round(p_total, 4),
                    "edge": edge_pct,
                    "model_prediction": g["cannon_total_xr"],
                    "kelly_pct": round(kelly, 2) if kelly else None,
                    "decision": decision,
                    "away_team": away,
                    "home_team": home,
                }
            )

    return rows
