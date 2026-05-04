from __future__ import annotations

from typing import Any

try:
    from mlb_inning_history import MLB_AVG_SCORELESS
    from mlb_inning_fetcher import DEFAULT_PITCHER, safe_float
except ImportError:
    from .mlb_inning_history import MLB_AVG_SCORELESS
    from .mlb_inning_fetcher import DEFAULT_PITCHER, safe_float


THREAT_BASELINE = 0.270
THREAT_SPAN = 0.130
THREAT_ADJUSTMENT_LIMIT = 0.15


def compute_inning_probabilities(
    game: dict[str, Any],
    team_histories: dict[str, dict[int, dict[str, float]]],
    matchup_threats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    game_id = str(game.get("game_id") or "")
    away_team = str(game.get("away_team") or "Away Team")
    home_team = str(game.get("home_team") or "Home Team")
    game_threats = (matchup_threats.get(game_id) or {}).get("innings") or {}

    full_inning_table: dict[str, float] = {}
    for inning in range(1, 10):
        inning_threats = game_threats.get(inning) or game_threats.get(str(inning)) or {}
        away_half_scoreless = _half_scoreless_probability(
            _history_rate(team_histories, away_team, inning),
            safe_float(inning_threats.get("away_threat"), THREAT_BASELINE),
            inning,
            game.get("home_pitcher") or {},
        )
        home_half_scoreless = _half_scoreless_probability(
            _history_rate(team_histories, home_team, inning),
            safe_float(inning_threats.get("home_threat"), THREAT_BASELINE),
            inning,
            game.get("away_pitcher") or {},
        )
        full_probability = _clamp(away_half_scoreless * home_half_scoreless, 0.01, 0.98)
        full_inning_table[str(inning)] = round(full_probability, 3)

    top_2 = [
        {
            "inning": int(inning),
            "probability_scoreless": probability,
            "confidence": _confidence(probability),
            "label": f"Inning {inning} - No Run Scored ({probability:.1%})",
        }
        for inning, probability in sorted(
            full_inning_table.items(),
            key=lambda item: (item[1], int(item[0])),
            reverse=True,
        )[:2]
    ]

    return {
        "game_id": game_id,
        "matchup": f"{home_team} vs {away_team}",
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher": (game.get("home_pitcher") or {}).get("name") or "TBD",
        "away_pitcher": (game.get("away_pitcher") or {}).get("name") or "TBD",
        "top_2_picks": top_2,
        "full_inning_table": full_inning_table,
    }


def _half_scoreless_probability(
    historical_scoreless_rate: float,
    threat_score: float,
    inning: int,
    opposing_pitcher: dict[str, Any],
) -> float:
    threat_adjustment = _clamp(
        (threat_score - THREAT_BASELINE) / THREAT_SPAN,
        -THREAT_ADJUSTMENT_LIMIT,
        THREAT_ADJUSTMENT_LIMIT,
    )
    probability = historical_scoreless_rate * (1.0 - threat_adjustment)
    pitcher_era = safe_float(opposing_pitcher.get("era"), DEFAULT_PITCHER["era"])
    if inning >= 7 and pitcher_era > 4.50:
        probability *= 0.92
    return _clamp(probability, 0.05, 0.98)


def _history_rate(team_histories: dict[str, dict[int, dict[str, float]]], team_name: str, inning: int) -> float:
    team_history = team_histories.get(team_name) or {}
    inning_stats = team_history.get(inning) or team_history.get(str(inning)) or {}
    return _clamp(safe_float(inning_stats.get("scoreless_rate"), MLB_AVG_SCORELESS[inning]), 0.05, 0.98)


def _confidence(probability: float) -> str:
    if probability >= 0.50:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
