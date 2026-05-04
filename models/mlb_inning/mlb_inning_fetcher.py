from __future__ import annotations

import json
import math
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
API_BASE = "https://statsapi.mlb.com/api/v1"
LIVE_FEED_BASE = "https://statsapi.mlb.com/api/v1.1"
HTTP_TIMEOUT_SECONDS = 10
SCHEDULE_TTL_SECONDS = 2 * 60 * 60
STATS_TTL_SECONDS = 24 * 60 * 60

DEFAULT_BATTER = {"avg": 0.240, "obp": 0.320, "slg": 0.390}
DEFAULT_PITCHER = {
    "id": None,
    "name": "TBD",
    "era": 4.20,
    "whip": 1.30,
    "opponent_obp": 0.320,
    "opponent_slg": 0.410,
}


def log_warning(message: str) -> None:
    print(f"[MLBInning] Warning: {message}")


def cache_get(key: str, ttl_seconds: int = STATS_TTL_SECONDS) -> Any | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_safe_cache_key(key)}.json"
    if not path.exists():
        return None
    try:
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > ttl_seconds:
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log_warning(f"ignoring bad cache file {path.name}: {exc}")
        return None


def cache_set(key: str, data: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_safe_cache_key(key)}.json"
    tmp_path = path.with_suffix(f".{path.suffix}.{time.time_ns()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
    tmp_path.replace(path)


def api_get_json(url: str, params: dict[str, Any] | None = None, cache_key: str | None = None, ttl_seconds: int = STATS_TTL_SECONDS) -> dict[str, Any]:
    if cache_key:
        cached = cache_get(cache_key, ttl_seconds)
        if cached is not None:
            return cached

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if cache_key:
                cache_set(cache_key, payload)
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.75)

    raise RuntimeError(f"Stats API request failed for {url}: {last_exc}")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-.--", ".---"):
            return default
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_date(target_date: str | date | None = None) -> str:
    if target_date is None:
        return date.today().strftime("%Y-%m-%d")
    if isinstance(target_date, date):
        return target_date.strftime("%Y-%m-%d")
    raw = str(target_date).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {target_date}. Use YYYY-MM-DD or MM/DD/YYYY.")


def fetch_todays_games(target_date: str | date | None = None) -> list[dict[str, Any]]:
    game_date = normalize_date(target_date)
    try:
        schedule = _fetch_schedule_for_date(game_date)
    except RuntimeError as exc:
        log_warning(str(exc))
        return []

    games: list[dict[str, Any]] = []
    for schedule_game in _schedule_games(schedule):
        game_id = safe_int(schedule_game.get("gamePk"))
        status = _status_state(schedule_game)
        if not game_id:
            continue
        if _should_skip_status(status):
            log_warning(f"skipped_in_progress {game_id}: {status or 'unknown status'}")
            continue

        teams = schedule_game.get("teams") or {}
        home_team_info = ((teams.get("home") or {}).get("team") or {})
        away_team_info = ((teams.get("away") or {}).get("team") or {})
        home_team_id = safe_int(home_team_info.get("id"))
        away_team_id = safe_int(away_team_info.get("id"))
        home_team_name = home_team_info.get("name") or "Home Team"
        away_team_name = away_team_info.get("name") or "Away Team"

        feed = _fetch_game_feed(game_id)
        if not feed:
            log_warning(f"skipping game {game_id}: live feed unavailable")
            continue

        game_data = feed.get("gameData") or {}
        probable_pitchers = game_data.get("probablePitchers") or {}
        home_pitcher_raw = probable_pitchers.get("home") or ((teams.get("home") or {}).get("probablePitcher") or {})
        away_pitcher_raw = probable_pitchers.get("away") or ((teams.get("away") or {}).get("probablePitcher") or {})

        home_pitcher = _build_pitcher(home_pitcher_raw, game_date)
        away_pitcher = _build_pitcher(away_pitcher_raw, game_date)
        if not home_pitcher.get("id"):
            home_pitcher = _recent_starter(home_team_id, game_date) or home_pitcher
        if not away_pitcher.get("id"):
            away_pitcher = _recent_starter(away_team_id, game_date) or away_pitcher

        home_lineup = _lineup_from_feed(feed, "home")
        away_lineup = _lineup_from_feed(feed, "away")
        if len(home_lineup) < 9:
            home_lineup = _last_known_lineup(home_team_id, game_date) or _active_roster_lineup(home_team_id, game_date)
        if len(away_lineup) < 9:
            away_lineup = _last_known_lineup(away_team_id, game_date) or _active_roster_lineup(away_team_id, game_date)

        games.append(
            {
                "game_id": str(game_id),
                "game_date": game_date,
                "status": status,
                "home_team": home_team_name,
                "away_team": away_team_name,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_pitcher": home_pitcher,
                "away_pitcher": away_pitcher,
                "home_lineup": home_lineup[:9],
                "away_lineup": away_lineup[:9],
            }
        )

    return games


def _fetch_schedule_for_date(game_date: str) -> dict[str, Any]:
    return api_get_json(
        f"{API_BASE}/schedule",
        params={"sportId": 1, "date": game_date, "hydrate": "probablePitcher"},
        cache_key=f"games_{game_date}",
        ttl_seconds=SCHEDULE_TTL_SECONDS,
    )


def _fetch_game_feed(game_id: int) -> dict[str, Any]:
    try:
        return api_get_json(
            f"{LIVE_FEED_BASE}/game/{game_id}/feed/live",
            cache_key=f"feed_{game_id}",
            ttl_seconds=SCHEDULE_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}


def _schedule_games(schedule_payload: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for day in schedule_payload.get("dates") or []:
        games.extend(day.get("games") or [])
    return games


def _status_state(schedule_game: dict[str, Any]) -> str:
    status = schedule_game.get("status") or {}
    return str(status.get("detailedState") or status.get("abstractGameState") or "")


def _should_skip_status(status: str) -> bool:
    normalized = status.strip().lower()
    if not normalized:
        return False
    playable = {"scheduled", "pre-game", "pregame", "preview", "warmup"}
    if normalized in playable:
        return False
    skipped_words = (
        "in progress",
        "final",
        "game over",
        "completed",
        "delayed",
        "postponed",
        "suspended",
        "cancelled",
        "canceled",
    )
    return any(word in normalized for word in skipped_words)


def _build_pitcher(raw_pitcher: dict[str, Any] | None, game_date: str) -> dict[str, Any]:
    pitcher = dict(DEFAULT_PITCHER)
    if not raw_pitcher:
        return pitcher

    pitcher_id = safe_int(raw_pitcher.get("id"))
    if not pitcher_id:
        return pitcher

    pitcher["id"] = pitcher_id
    pitcher["name"] = raw_pitcher.get("fullName") or raw_pitcher.get("name") or "TBD"
    stat = _player_season_stat(pitcher_id, "pitching", _season_for_date(game_date))
    pitcher["era"] = safe_float(stat.get("era"), DEFAULT_PITCHER["era"])
    pitcher["whip"] = safe_float(stat.get("whip"), DEFAULT_PITCHER["whip"])
    pitcher["opponent_obp"] = safe_float(stat.get("obp"), DEFAULT_PITCHER["opponent_obp"])
    pitcher["opponent_slg"] = safe_float(stat.get("slg"), DEFAULT_PITCHER["opponent_slg"])
    return pitcher


def _lineup_from_feed(feed: dict[str, Any], side: str) -> list[dict[str, Any]]:
    box_team = (((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {}
    batting_order = [safe_int(player_id) for player_id in (box_team.get("battingOrder") or [])]
    players = box_team.get("players") or {}
    lineup: list[dict[str, Any]] = []

    for order, player_id in enumerate([pid for pid in batting_order if pid], start=1):
        player = players.get(f"ID{player_id}") or {}
        person = player.get("person") or {}
        season_batting = ((player.get("seasonStats") or {}).get("batting") or {})
        lineup.append(_batter_entry(order, player_id, person.get("fullName"), season_batting))

    return lineup


def _last_known_lineup(team_id: int, game_date: str) -> list[dict[str, Any]]:
    for game in _recent_team_games(team_id, game_date, lookback_days=14):
        if "final" not in _status_state(game).lower():
            continue
        game_id = safe_int(game.get("gamePk"))
        feed = _fetch_game_feed(game_id)
        side = _team_side(feed, team_id)
        if not side:
            continue
        lineup = _lineup_from_feed(feed, side)
        if len(lineup) >= 9:
            return lineup[:9]
    return []


def _active_roster_lineup(team_id: int, game_date: str) -> list[dict[str, Any]]:
    if not team_id:
        return []
    season = _season_for_date(game_date)
    try:
        payload = api_get_json(
            f"{API_BASE}/teams/{team_id}/roster",
            params={"rosterType": "Active", "season": season},
            cache_key=f"active_roster_{team_id}_{season}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return []

    lineup: list[dict[str, Any]] = []
    for player in payload.get("roster") or []:
        position = player.get("position") or {}
        if str(position.get("abbreviation") or "").upper() == "P":
            continue
        person = player.get("person") or {}
        player_id = safe_int(person.get("id"))
        if not player_id:
            continue
        stat = _player_season_stat(player_id, "hitting", season)
        lineup.append(_batter_entry(len(lineup) + 1, player_id, person.get("fullName"), stat))
        if len(lineup) == 9:
            break
    return lineup


def _recent_starter(team_id: int, game_date: str) -> dict[str, Any] | None:
    for game in _recent_team_games(team_id, game_date, lookback_days=21):
        if "final" not in _status_state(game).lower():
            continue
        game_id = safe_int(game.get("gamePk"))
        feed = _fetch_game_feed(game_id)
        side = _team_side(feed, team_id)
        if not side:
            continue
        box_team = (((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {}
        pitcher_ids = [safe_int(player_id) for player_id in (box_team.get("pitchers") or [])]
        if not pitcher_ids:
            continue
        players = box_team.get("players") or {}
        starter_id = pitcher_ids[0]
        starter = players.get(f"ID{starter_id}") or {}
        person = starter.get("person") or {}
        return _build_pitcher({"id": starter_id, "fullName": person.get("fullName")}, game_date)
    return None


def _recent_team_games(team_id: int, game_date: str, lookback_days: int) -> list[dict[str, Any]]:
    if not team_id:
        return []
    end_date = datetime.strptime(game_date, "%Y-%m-%d").date() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days)
    try:
        payload = api_get_json(
            f"{API_BASE}/schedule",
            params={
                "sportId": 1,
                "teamId": team_id,
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
            },
            cache_key=f"recent_games_{team_id}_{start_date}_{end_date}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return []

    games = _schedule_games(payload)
    games.sort(key=lambda item: (str(item.get("gameDate") or ""), safe_int(item.get("gamePk"))), reverse=True)
    return games


def _team_side(feed: dict[str, Any], team_id: int) -> str:
    teams = ((feed.get("gameData") or {}).get("teams") or {})
    if safe_int((teams.get("home") or {}).get("id")) == team_id:
        return "home"
    if safe_int((teams.get("away") or {}).get("id")) == team_id:
        return "away"
    return ""


def _player_season_stat(player_id: int, group: str, season: int) -> dict[str, Any]:
    if not player_id:
        return {}
    try:
        payload = api_get_json(
            f"{API_BASE}/people/{player_id}/stats",
            params={"stats": "season", "group": group, "season": season},
            cache_key=f"player_{player_id}_{group}_{season}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}

    splits = ((payload.get("stats") or [{}])[0].get("splits") or [])
    if not splits and season > 2000:
        try:
            prior_payload = api_get_json(
                f"{API_BASE}/people/{player_id}/stats",
                params={"stats": "season", "group": group, "season": season - 1},
                cache_key=f"player_{player_id}_{group}_{season - 1}",
                ttl_seconds=STATS_TTL_SECONDS,
            )
            splits = ((prior_payload.get("stats") or [{}])[0].get("splits") or [])
        except RuntimeError:
            splits = []
    return (splits[0].get("stat") if splits else {}) or {}


def _batter_entry(order: int, player_id: int, name: str | None, stat: dict[str, Any]) -> dict[str, Any]:
    return {
        "batting_order": order,
        "player_id": player_id,
        "name": name or f"Player {player_id}",
        "avg": safe_float(stat.get("avg"), DEFAULT_BATTER["avg"]),
        "obp": safe_float(stat.get("obp"), DEFAULT_BATTER["obp"]),
        "slg": safe_float(stat.get("slg"), DEFAULT_BATTER["slg"]),
    }


def _season_for_date(game_date: str) -> int:
    return datetime.strptime(game_date, "%Y-%m-%d").year


def _safe_cache_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")
