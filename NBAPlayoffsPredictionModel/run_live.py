#!/usr/bin/env python3
"""
NBA Playoffs Prediction Model.

This runner is intentionally stricter than the regular-season NBA runner:
it first verifies the date against ESPN's postseason scoreboard and only
emits machine-readable picks for official playoff games that have not started.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

BASE_DIR = Path(__file__).resolve().parents[1]
NBA_MODEL_DIR = BASE_DIR / "NBAPredictionModel"
sys.path.insert(0, str(NBA_MODEL_DIR))
sys.path.insert(1, str(BASE_DIR))

from injury_report import fetch_injuries, get_expected_injury_impact  # noqa: E402
from data_models import GameContext, Venue  # noqa: E402
from live_data import fetch_all_team_stats, fetch_roster, get_team_id  # noqa: E402
from market_mechanics import remove_vig  # noqa: E402
from probability_layers import (  # noqa: E402
    calculate_dictated_pace,
    calculate_injury_adjustment as calculate_probabilistic_injury_adjustment,
    predict_spread,
    predict_total_points,
)
from run_live import (  # noqa: E402
    IS_RENDER_RUNTIME,
    _pause_after_injury_lookup,
    _render_fast_injury_adjustment,
    create_team,
)


MODEL_SOURCE = "NBA Playoffs"
DEFAULT_SEASON = os.environ.get("NBA_MODEL_SEASON", "2025-26").strip() or "2025-26"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
USER_AGENT = "Mozilla/5.0 PickLedgerPro NBAPlayoffs/1.0"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_target_date(raw_value: str | None) -> str:
    if not raw_value:
        return dt.date.today().isoformat()

    value = str(raw_value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return dt.date.today().isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NBA Playoffs model for a selected date.")
    parser.add_argument("legacy_date", nargs="?", default="", help="Optional date in MM/DD/YYYY or YYYY-MM-DD format.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="NBA API season string, e.g. 2025-26.")
    parser.add_argument("--no-log", action="store_true", help="Accepted for backend compatibility; this runner logs only to stdout.")
    return parser.parse_args()


def _request_json(url: str) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _espn_date_key(date_str: str) -> str:
    return dt.date.fromisoformat(date_str).strftime("%Y%m%d")


def _parse_espn_datetime(value: str) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _parse_american_odds(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace("−", "-")
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _parse_line(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("−", "-")
    text = text.replace("o", "").replace("u", "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _competition_team(comp: dict[str, Any], home_away: str) -> dict[str, Any] | None:
    for competitor in comp.get("competitors", []) or []:
        if competitor.get("homeAway") == home_away:
            return competitor
    return None


def _extract_market(comp: dict[str, Any]) -> dict[str, Any]:
    odds_payload = (comp.get("odds") or [{}])[0] or {}
    moneyline = odds_payload.get("moneyline") or {}
    point_spread = odds_payload.get("pointSpread") or {}
    total = odds_payload.get("total") or {}

    return {
        "provider": ((odds_payload.get("provider") or {}).get("name") or "ESPN odds").strip(),
        "home_ml": _parse_american_odds(((moneyline.get("home") or {}).get("close") or {}).get("odds")),
        "away_ml": _parse_american_odds(((moneyline.get("away") or {}).get("close") or {}).get("odds")),
        "home_spread": _parse_line(((point_spread.get("home") or {}).get("close") or {}).get("line")),
        "away_spread": _parse_line(((point_spread.get("away") or {}).get("close") or {}).get("line")),
        "home_spread_odds": _parse_american_odds(((point_spread.get("home") or {}).get("close") or {}).get("odds")),
        "away_spread_odds": _parse_american_odds(((point_spread.get("away") or {}).get("close") or {}).get("odds")),
        "total_line": odds_payload.get("overUnder"),
        "over_odds": _parse_american_odds(((total.get("over") or {}).get("close") or {}).get("odds")),
        "under_odds": _parse_american_odds(((total.get("under") or {}).get("close") or {}).get("odds")),
        "details": str(odds_payload.get("details") or "").strip(),
    }


def _team_short_name(team: dict[str, Any]) -> str:
    name = str(team.get("name") or "").strip()
    display = str(team.get("displayName") or "").strip()
    if name:
        return name
    return display.split()[-1] if display else ""


def fetch_espn_playoff_games(date_str: str) -> list[dict[str, Any]]:
    url = f"{ESPN_SCOREBOARD_URL}?dates={_espn_date_key(date_str)}&seasontype=3"
    payload = _request_json(url)
    now_utc = dt.datetime.now(dt.timezone.utc)
    games: list[dict[str, Any]] = []

    for event in payload.get("events", []) or []:
        if ((event.get("season") or {}).get("type")) != 3:
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0] or {}
        home = _competition_team(comp, "home")
        away = _competition_team(comp, "away")
        if not home or not away:
            continue

        status = comp.get("status") or {}
        status_type = status.get("type") or {}
        event_dt = _parse_espn_datetime(str(comp.get("date") or event.get("date") or ""))
        status_state = str(status_type.get("state") or "").strip().lower()
        has_started = (
            status_state != "pre"
            or bool(status_type.get("completed"))
            or (event_dt is not None and event_dt <= now_utc)
        )
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        note = ""
        notes = comp.get("notes") or event.get("notes") or []
        if notes:
            note = str((notes[0] or {}).get("headline") or "").strip()

        games.append(
            {
                "game_id": str(event.get("id") or comp.get("id") or "").strip(),
                "slate_date": date_str,
                "date": event_dt.isoformat() if event_dt else str(event.get("date") or ""),
                "home_team": _team_short_name(home_team),
                "away_team": _team_short_name(away_team),
                "home_display": str(home_team.get("displayName") or "").strip(),
                "away_display": str(away_team.get("displayName") or "").strip(),
                "home_abbr": str(home_team.get("abbreviation") or "").strip(),
                "away_abbr": str(away_team.get("abbreviation") or "").strip(),
                "arena": ((comp.get("venue") or {}).get("fullName") or "").strip(),
                "game_status": str(status_type.get("shortDetail") or status_type.get("detail") or status_type.get("description") or "").strip(),
                "status_state": status_state,
                "has_started": has_started,
                "round": str(((comp.get("type") or {}).get("abbreviation") or "")).strip(),
                "series_status": note or "NBA Playoffs",
                "home_series_record": home.get("record"),
                "away_series_record": away.get("record"),
                "home_regular_record": next((r.get("summary") for r in home.get("records", []) if r.get("type") == "total"), ""),
                "away_regular_record": next((r.get("summary") for r in away.get("records", []) if r.get("type") == "total"), ""),
                "market": _extract_market(comp),
            }
        )

    return games


def _nba_team_key(full_name: str) -> str:
    if str(full_name) == "Portland Trail Blazers":
        return "Trail Blazers"
    return str(full_name or "").split()[-1]


def fetch_last20_context(season: str, as_of_date: str) -> dict[str, dict[str, float]]:
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    frame = finder.get_data_frames()[0]
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    target_dt = pd.Timestamp(dt.date.fromisoformat(as_of_date))
    frame = frame[frame["GAME_DATE"] < target_dt]

    result: dict[str, dict[str, float]] = {}
    for full_name, team_games in frame.groupby("TEAM_NAME"):
        recent = team_games.sort_values("GAME_DATE", ascending=False).head(20)
        if recent.empty:
            continue
        payload = {
            "last20_win_pct": float((recent["WL"] == "W").mean()),
            "last20_point_diff": float(recent["PLUS_MINUS"].astype(float).mean()),
        }
        result[full_name] = payload
        result[_nba_team_key(full_name)] = payload
    return result


def fetch_h2h_context(home_team: str, away_abbr: str, season: str, as_of_date: str) -> dict[str, Any]:
    team_id = get_team_id(home_team)
    if not team_id:
        return {"home_win_pct": 0.5, "games": 0, "point_diff": 0.0, "note": "H2H unavailable"}

    finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    frame = finder.get_data_frames()[0]
    if frame.empty or "MATCHUP" not in frame.columns:
        return {"home_win_pct": 0.5, "games": 0, "point_diff": 0.0, "note": "H2H unavailable"}

    frame = frame.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    target_dt = pd.Timestamp(dt.date.fromisoformat(as_of_date))
    frame = frame[frame["GAME_DATE"] < target_dt]
    opponent = str(away_abbr or "").upper()
    h2h = frame[frame["MATCHUP"].astype(str).str.upper().str.contains(opponent, regex=False)]
    if h2h.empty:
        return {"home_win_pct": 0.5, "games": 0, "point_diff": 0.0, "note": "No current-season H2H found"}

    wins = float((h2h["WL"] == "W").mean())
    point_diff = float(h2h["PLUS_MINUS"].astype(float).mean())
    return {
        "home_win_pct": wins,
        "games": int(len(h2h)),
        "point_diff": point_diff,
        "note": f"{home_team} {int((h2h['WL'] == 'W').sum())}-{int((h2h['WL'] == 'L').sum())}, avg margin {point_diff:+.1f}",
    }


def _rank_lookup(all_team_stats: dict[str, dict[str, Any]]) -> dict[str, int]:
    unique: dict[str, dict[str, Any]] = {}
    for key, payload in all_team_stats.items():
        full_name = str(payload.get("full_name") or key).strip()
        if full_name and full_name not in unique:
            unique[full_name] = payload

    ordered = sorted(
        unique.items(),
        key=lambda item: float(item[1].get("win_pct", 0.0) or 0.0),
        reverse=True,
    )
    ranks: dict[str, int] = {}
    for idx, (full_name, payload) in enumerate(ordered, start=1):
        ranks[full_name] = idx
        ranks[_nba_team_key(full_name)] = idx
        short = str(payload.get("nickname") or "").strip()
        if short:
            ranks[short] = idx
    return ranks


def _safe_pct(value: Any, fallback: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return _clamp(number, 0.0, 1.0)


def calculate_base_rate(
    home_team: str,
    away_team: str,
    home_stats: dict[str, Any],
    last20_context: dict[str, dict[str, float]],
    ranks: dict[str, int],
    h2h: dict[str, Any],
) -> tuple[float, list[str]]:
    season_win_pct = _safe_pct(home_stats.get("win_pct"))
    last20 = last20_context.get(home_team, {})
    last20_win_pct = _safe_pct(last20.get("last20_win_pct"), _safe_pct(home_stats.get("recent_10_win_pct")))

    h2h_component = _safe_pct(h2h.get("home_win_pct"), 0.5)
    home_rank = ranks.get(home_team)
    away_rank = ranks.get(away_team)
    if home_rank and away_rank:
        seeding_component = _clamp(0.50 + ((away_rank - home_rank) * 0.015), 0.35, 0.65)
        h2h_seed_component = (h2h_component * 0.60) + (seeding_component * 0.40)
        seed_note = f"rank proxy {home_team} #{home_rank} vs {away_team} #{away_rank}"
    else:
        h2h_seed_component = h2h_component
        seed_note = "rank proxy unavailable"

    base = (season_win_pct * 0.40) + (last20_win_pct * 0.30) + (h2h_seed_component * 0.30)
    notes = [
        f"season win% {season_win_pct*100:.1f}% x40%",
        f"last-20 win% {last20_win_pct*100:.1f}% x30%",
        f"H2H/seed component {h2h_seed_component*100:.1f}% x30% ({h2h.get('note')}; {seed_note})",
    ]
    return _clamp(base, 0.05, 0.95), notes


def _injury_adjustment(team_name: str, injuries: dict[str, list[dict[str, Any]]]) -> tuple[float, str, int]:
    expected = get_expected_injury_impact(injuries, team_name)
    if not expected:
        return 0.0, "No listed expected absences", 0

    if IS_RENDER_RUNTIME:
        adj, reason = _render_fast_injury_adjustment(team_name, expected)
    else:
        adj, reason = calculate_probabilistic_injury_adjustment(team_name, expected)
        _pause_after_injury_lookup()
    return adj, reason, len(expected)


def _build_adjustments(
    game: dict[str, Any],
    home_team,
    away_team,
    injuries: dict[str, list[dict[str, Any]]],
    tempo_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    adjustments: list[dict[str, Any]] = []

    home_name = home_team.name
    away_name = away_team.name

    hca = 0.05 if home_name in {"Nuggets", "Jazz", "Denver Nuggets", "Utah Jazz"} else 0.04
    adjustments.append({"label": "Home court", "value": hca, "reason": f"{home_name} home playoff game"})

    home_inj_adj, home_inj_reason, home_inj_count = _injury_adjustment(home_name, injuries)
    away_inj_adj, away_inj_reason, away_inj_count = _injury_adjustment(away_name, injuries)
    injury_delta = _clamp(home_inj_adj - away_inj_adj, -0.08, 0.08)
    if injury_delta:
        adjustments.append({"label": "Star/rotation availability", "value": injury_delta, "reason": f"{home_name}: {home_inj_reason}; {away_name}: {away_inj_reason}"})
    elif home_inj_count or away_inj_count:
        adjustments.append({"label": "Star/rotation availability", "value": 0.0, "reason": f"{home_name}: {home_inj_reason}; {away_name}: {away_inj_reason}"})

    home_stats = home_team.team_stats
    away_stats = away_team.team_stats
    home_attack = (home_stats.off_rating_10 - 114.0) + (away_stats.def_rating_10 - 114.0)
    away_attack = (away_stats.off_rating_10 - 114.0) + (home_stats.def_rating_10 - 114.0)
    mismatch_adj = _clamp((home_attack - away_attack) * 0.006, -0.06, 0.06)
    if abs(mismatch_adj) >= 0.005:
        adjustments.append({
            "label": "Off/def mismatch",
            "value": mismatch_adj,
            "reason": f"attack score {home_name} {home_attack:+.1f} vs {away_name} {away_attack:+.1f}",
        })

    rest_diff = float(home_stats.rest_days) - float(away_stats.rest_days)
    rest_adj = _clamp(rest_diff * 0.015, -0.03, 0.03)
    if abs(rest_adj) >= 0.005:
        adjustments.append({
            "label": "Rest/travel",
            "value": rest_adj,
            "reason": f"rest days {home_name} {home_stats.rest_days:.0f} vs {away_name} {away_stats.rest_days:.0f}",
        })

    dictating_side = str(tempo_context.get("dictating_side") or "neutral")
    pace_adj = 0.0
    if dictating_side == "home":
        pace_adj = 0.015 if home_stats.net_rating >= away_stats.net_rating else 0.005
    elif dictating_side == "away":
        pace_adj = -0.015 if away_stats.net_rating >= home_stats.net_rating else -0.005
    if pace_adj:
        adjustments.append({
            "label": "Pace control",
            "value": pace_adj,
            "reason": f"{dictating_side} tempo control, dictated pace {tempo_context.get('dictated_pace', 0.0):.1f}",
        })

    series_record = str(game.get("home_series_record") or "").strip()
    if series_record == "0-2":
        adjustments.append({"label": "Situational urgency", "value": 0.005, "reason": f"{home_name} down 0-2 at home"})

    return adjustments, home_inj_reason, away_inj_reason


def extremize_probability(raw_prob: float) -> float:
    """
    Directional, bounded version of the prompt's confidence-term extremizer.

    The literal expression base * (1 - base) * 4 + base exceeds 1.0 for
    ordinary probabilities, so this uses that term as the strength of a
    directional move away from 50%.
    """
    raw = _clamp(raw_prob, 0.01, 0.99)
    if abs(raw - 0.50) < 1e-9:
        return 0.50
    confidence_term = raw * (1.0 - raw) * 4.0
    directional_shift = math.copysign(abs(raw - 0.50) * confidence_term, raw - 0.50)
    return _clamp(raw + directional_shift, 0.03, 0.97)


def _american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def _quarter_kelly_units(edge: float, american_odds: int) -> float:
    decimal_odds = _american_to_decimal(american_odds)
    b = decimal_odds - 1.0
    if b <= 0 or edge <= 0:
        return 0.0
    return min(2.0, edge / b / 4.0)


def _format_odds(odds: int | None) -> str:
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def _verify_roster(team_name: str, season: str) -> tuple[bool, str]:
    try:
        roster = fetch_roster(team_name, season=season)
    except Exception as exc:
        return False, f"NBA API roster lookup failed: {exc}"
    if not roster:
        return False, "NBA API roster lookup returned no players"
    return True, f"{len(roster)} active roster entries fetched from NBA API"


def run_playoff_game(
    game: dict[str, Any],
    all_team_stats: dict[str, dict[str, Any]],
    last20_context: dict[str, dict[str, float]],
    ranks: dict[str, int],
    injuries: dict[str, list[dict[str, Any]]],
    season: str,
) -> dict[str, Any] | None:
    away_name = game["away_team"]
    home_name = game["home_team"]
    matchup = f"{away_name} @ {home_name}"

    print("\n" + "=" * 80)
    print(f"GAME: {matchup} ({game['series_status']})")
    print("=" * 80)

    if game.get("has_started"):
        print(f"DECLINE: {matchup} has already started or is no longer in pre-game status ({game.get('game_status')}).")
        return None

    market = game.get("market") or {}
    home_ml = market.get("home_ml")
    away_ml = market.get("away_ml")
    if home_ml is None or away_ml is None:
        print(f"DECLINE: {matchup} has no verified two-sided moneyline in the ESPN odds payload.")
        return None

    if away_name not in all_team_stats or home_name not in all_team_stats:
        print(f"DECLINE: Missing NBA API team stats for {matchup}.")
        return None

    home_roster_ok, home_roster_note = _verify_roster(home_name, season)
    away_roster_ok, away_roster_note = _verify_roster(away_name, season)
    if not home_roster_ok or not away_roster_ok:
        print(f"DECLINE: Roster verification failed. {home_name}: {home_roster_note}; {away_name}: {away_roster_note}.")
        return None

    home_team = create_team(2, home_name, True, all_team_stats[home_name])
    away_team = create_team(1, away_name, False, all_team_stats[away_name])
    venue = game.get("arena") or f"{home_name} Arena"
    ctx = GameContext(
        game.get("slate_date") or game.get("date", "")[:10],
        Venue(venue),
        home_team,
        away_team,
        0.50,
        game_id=game.get("game_id", ""),
    )

    _, tempo_context = calculate_dictated_pace(
        away_team.team_stats,
        home_team.team_stats,
        use_capped_form=True,
    )

    h2h = fetch_h2h_context(home_name, game.get("away_abbr", ""), season, ctx.date)
    base_rate, base_notes = calculate_base_rate(
        home_name,
        away_name,
        all_team_stats[home_name],
        last20_context,
        ranks,
        h2h,
    )
    adjustments, home_inj_reason, away_inj_reason = _build_adjustments(
        game,
        home_team,
        away_team,
        injuries,
        tempo_context,
    )

    raw_prob = _clamp(base_rate + sum(float(item["value"]) for item in adjustments), 0.03, 0.97)
    final_home_prob = extremize_probability(raw_prob)
    home_market_prob, away_market_prob = remove_vig(home_ml, away_ml)

    predicted_spread = predict_spread(home_team, away_team, pace_context=tempo_context)
    predicted_total = predict_total_points(ctx, pace_context=tempo_context)

    if final_home_prob >= 0.50:
        pick_team = home_name
        pick_prob = final_home_prob
        market_prob = home_market_prob
        pick_odds = home_ml
    else:
        pick_team = away_name
        pick_prob = 1.0 - final_home_prob
        market_prob = away_market_prob
        pick_odds = away_ml

    edge = pick_prob - market_prob
    spread_team = home_name if predicted_spread >= 0 else away_name
    spread_disagrees = spread_team != pick_team and abs(predicted_spread) >= 0.5
    decision = "BET" if edge > 0.03 and not spread_disagrees else "PASS"
    units = _quarter_kelly_units(edge, pick_odds) if decision == "BET" else 0.0
    confidence = "High" if edge >= 0.06 and abs(sum(float(item["value"]) for item in adjustments)) <= 0.16 else "Medium"
    if decision == "PASS" or not injuries:
        confidence = "Low" if not injuries else "Medium"

    print("**Game Context:**")
    print(f"- {game.get('away_display') or away_name} at {game.get('home_display') or home_name}")
    print(f"- {game.get('series_status')} | Venue: {venue} | Scheduled: {game.get('game_status')}")
    print(f"- Source: ESPN NBA postseason scoreboard confirms season type 3 / post-season.")

    print("\n**Verification checks:**")
    print("- [x] Official playoff game verified through ESPN scoreboard")
    print("- [x] Game has not started")
    print(f"- [x] Current rosters checked: {home_name} ({home_roster_note}); {away_name} ({away_roster_note})")
    print("- [x] NBA API team efficiency and recent-form stats fetched")
    print(f"- [x] Market moneyline fetched from {market.get('provider') or 'ESPN odds'}")

    print("\n**Key Factors:**")
    print(f"- Net Rating: {away_name} {away_team.team_stats.net_rating:+.1f} vs {home_name} {home_team.team_stats.net_rating:+.1f} (NBA API)")
    print(f"- Off/Def Rating: {away_name} {away_team.team_stats.off_rating_10:.1f}/{away_team.team_stats.def_rating_10:.1f} vs {home_name} {home_team.team_stats.off_rating_10:.1f}/{home_team.team_stats.def_rating_10:.1f}")
    print(f"- Pace: {away_name} {away_team.team_stats.pace:.1f} vs {home_name} {home_team.team_stats.pace:.1f}; dictated {tempo_context.get('dictated_pace', 0.0):.1f}")
    print(f"- H2H: {h2h.get('note')}")
    print(f"- Injuries: {home_name}: {home_inj_reason}; {away_name}: {away_inj_reason}")
    print(f"- Rest days: {away_name} {away_team.team_stats.rest_days:.0f} vs {home_name} {home_team.team_stats.rest_days:.0f}")

    print("\n**Our Probability:**")
    print(f"- Base rate ({home_name}): {base_rate*100:.1f}%")
    for note in base_notes:
        print(f"  - {note}")
    for item in adjustments:
        print(f"- {item['label']}: {float(item['value'])*100:+.1f}% because {item['reason']}")
    print(f"- Raw adjusted probability ({home_name}): {raw_prob*100:.1f}%")
    print(f"- Extremized final probability ({home_name}): {final_home_prob*100:.1f}%")

    print("\n**Model Predictions:**")
    print(f"- **Pick:** {pick_team}")
    print(f"- **Projected Margin:** {spread_team} by {abs(predicted_spread):.2f} points")
    print(f"- **Model Confidence:** {pick_prob*100:.1f}%")
    print(f"- **Total:** {predicted_total:.1f} O/U")

    print("\n**Market Odds:**")
    print(f"- {home_name} {_format_odds(home_ml)} | {away_name} {_format_odds(away_ml)} ({market.get('provider') or 'ESPN odds'})")
    print(f"- Market implied probability (vig-removed): {home_name} {home_market_prob*100:.1f}% | {away_name} {away_market_prob*100:.1f}%")
    if market.get("home_spread") is not None or market.get("away_spread") is not None:
        print(f"- Spread: {home_name} {market.get('home_spread')} | {away_name} {market.get('away_spread')}")

    print("\n**Edge And Decision:**")
    print(f"**Edge:** {pick_team} {edge*100:+.1f}%")
    print("**Minimum threshold:** 3.0%")
    if decision == "BET":
        print(f"**Decision: BET on {pick_team}**")
        print(f"**Stake:** {units:.2f} units (quarter Kelly, 2u cap)")
    else:
        print("**Decision: PASS**")
        if spread_disagrees:
            print(f"**Reason:** Moneyline probability points to {pick_team}, but the spread layer projects {spread_team} by {abs(predicted_spread):.2f}.")
        else:
            print("**Reason:** Edge is too small or verification confidence is not strong enough.")

    print("\n**Confidence And Honesty:**")
    print(f"- Confidence: {confidence}")
    print("- Limitations: Expected starters and final playoff rotations can change near tip; re-check official injury and lineup reports before betting.")

    pick = {
        "source": MODEL_SOURCE,
        "pick": f"{pick_team} ML ({away_name} @ {home_name})",
        "sport": "NBA",
        "league": "NBA",
        "market_type": "moneyline",
        "selection": pick_team,
        "team": pick_team,
        "away_team": away_name,
        "home_team": home_name,
        "odds": pick_odds,
        "units": round(units, 2),
        "probability": round(pick_prob, 4),
        "prob": round(pick_prob, 4),
        "edge": round(edge * 100.0, 2),
        "decision": decision,
        "market_probability": round(market_prob, 4),
        "model_prediction": round(predicted_spread, 2),
        "predicted_spread": round(predicted_spread, 2),
        "vegas": market.get("home_spread") if pick_team == home_name else market.get("away_spread"),
        "market_line": market.get("home_spread") if pick_team == home_name else market.get("away_spread"),
        "total_projection": round(predicted_total, 1),
        "series_status": game.get("series_status"),
        "game_id": game.get("game_id"),
    }
    print(f"PICK_JSON: {json.dumps(pick, sort_keys=True)}")
    return pick


def main() -> None:
    args = _parse_args()
    target_date = _normalize_target_date(args.date or args.legacy_date)
    season = str(args.season or DEFAULT_SEASON).strip() or DEFAULT_SEASON

    print("=" * 80)
    print("NBA PLAYOFFS PREDICTION MODEL")
    print(f"Requested Date: {target_date}")
    print(f"Run Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Data: ESPN postseason scoreboard + NBA API stats/rosters + injury feed")
    print("=" * 80)

    try:
        playoff_games = fetch_espn_playoff_games(target_date)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"No eligible NBA playoff games: ESPN postseason scoreboard fetch failed ({exc}).")
        return

    if not playoff_games:
        print(f"No official NBA playoff games found on ESPN for {target_date}.")
        return

    eligible_games = [game for game in playoff_games if not game.get("has_started")]
    if not eligible_games:
        print("No eligible NBA playoff games: every listed playoff game has started or finished.")
        return

    print(f"Found {len(playoff_games)} playoff game(s); {len(eligible_games)} still pre-game.")

    print("\nFetching NBA API team stats and rest context...")
    all_team_stats = fetch_all_team_stats(
        season=season,
        as_of_date=target_date,
        upcoming_games=eligible_games,
    )

    print("\nFetching last-20 context from NBA API game logs...")
    try:
        last20_context = fetch_last20_context(season, target_date)
    except Exception as exc:
        print(f"WARNING: Last-20 lookup failed ({exc}); falling back to model recent-form fields.")
        last20_context = {}

    ranks = _rank_lookup(all_team_stats)

    print("\nFetching current injury report...")
    injuries = fetch_injuries()
    if not injuries:
        print("WARNING: Injury feed returned no entries; picks will be lower-confidence.")

    picks: list[dict[str, Any]] = []
    for game in eligible_games:
        try:
            pick = run_playoff_game(game, all_team_stats, last20_context, ranks, injuries, season)
            if pick:
                picks.append(pick)
        except Exception as exc:
            matchup = f"{game.get('away_team', '')} @ {game.get('home_team', '')}".strip()
            print(f"DECLINE: {matchup or 'game'} failed playoff model verification/calculation ({exc}).")

    if not picks:
        print("No eligible NBA playoff picks generated after verification gates.")


if __name__ == "__main__":
    main()
