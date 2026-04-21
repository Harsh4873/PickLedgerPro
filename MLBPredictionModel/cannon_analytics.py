"""Cannon Analytics MLB projection scraper.

Hits the public JSON APIs on cannon-analytics.onrender.com and stores
results in SQLite tables that align with the PickLedger MLB schema.

Endpoints (confirmed working 2026-04-14):
    /api/rankings/teams      → season team power ratings
    /api/rankings/batters    → season batter projections (ATC-based)
    /api/rankings/pitchers   → season pitcher projections
    /api/awards              → MVP / Cy Young leaderboards
    /api/game_projections    → daily team game-level projections (xR, xWin, NRFI)
    /api/game_pitchers       → daily starting pitcher projections (K, H, ER, BB, …)
    /api/game_batters        → daily lineup batter projections (PA, H, HR, TB, …)
"""

from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://cannon-analytics.onrender.com"

ENDPOINTS = {
    "team_rankings":      "/api/rankings/teams",
    "batter_rankings":    "/api/rankings/batters",
    "pitcher_rankings":   "/api/rankings/pitchers",
    "awards":             "/api/awards",
    "game_projections":   "/api/game_projections",
    "game_pitchers":      "/api/game_pitchers",
    "game_batters":       "/api/game_batters",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Render free-tier cold starts can take 30-50s.
REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "pickledger.db",
        here / "pickledger.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _last_word(text: str | None) -> str:
    """Return the last whitespace-delimited word, lowercased."""
    parts = (text or "").strip().lower().split()
    return parts[-1] if parts else ""


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


# ---------------------------------------------------------------------------
# API fetch layer
# ---------------------------------------------------------------------------

def fetch_endpoint(endpoint_key: str, params: dict[str, Any] | None = None) -> Any:
    """Fetch a single Cannon Analytics endpoint and return parsed JSON.

    Returns an empty list/dict on failure so callers can degrade gracefully.
    """
    path = ENDPOINTS[endpoint_key]
    url = f"{BASE_URL}{path}"
    request_params = {
        key: value
        for key, value in (params or {}).items()
        if value is not None and str(value).strip() != ""
    }
    try:
        resp = requests.get(
            url,
            headers=HEADERS,
            params=request_params or None,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"Cannon Analytics {endpoint_key}: request failed: {exc}")
        return [] if endpoint_key != "awards" else {}
    except ValueError as exc:
        print(f"Cannon Analytics {endpoint_key}: JSON decode failed: {exc}")
        return [] if endpoint_key != "awards" else {}


def fetch_all() -> dict[str, Any]:
    """Fetch all seven endpoints.  Returns a dict keyed by endpoint name."""
    data: dict[str, Any] = {}
    for key in ENDPOINTS:
        data[key] = fetch_endpoint(key)
        # Be polite to the free-tier Render service.
        time.sleep(0.5)
    return data


# ---------------------------------------------------------------------------
# SQLite schema + persistence
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cannon_team_rankings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    team            TEXT    NOT NULL,
    proj_w          REAL,
    proj_l          REAL,
    w_pct           REAL,
    rs_pg           REAL,
    ra_pg           REAL,
    rd_pg           REAL,
    div_pct         REAL,
    playoff_pct     REAL,
    champ_pct       REAL,
    ws_pct          REAL
);

CREATE TABLE IF NOT EXISTS cannon_batter_rankings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    player          TEXT    NOT NULL,
    team            TEXT,
    g               INTEGER,
    pa              INTEGER,
    h               INTEGER,
    hr              INTEGER,
    bb              INTEGER,
    k               INTEGER,
    avg             REAL,
    obp             REAL,
    slg             REAL,
    rv_per_game     REAL,
    rv_per_pa       REAL,
    season_runs     REAL
);

CREATE TABLE IF NOT EXISTS cannon_pitcher_rankings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    player          TEXT    NOT NULL,
    team            TEXT,
    g               INTEGER,
    gs              INTEGER,
    ip              REAL,
    tbf             INTEGER,
    k               INTEGER,
    bb              INTEGER,
    hr              INTEGER,
    k_pct           REAL,
    bb_pct          REAL,
    rv_per_9        REAL,
    season_runs     REAL
);

CREATE TABLE IF NOT EXISTS cannon_game_projections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    game_date       TEXT    NOT NULL,
    away_team       TEXT    NOT NULL,
    home_team       TEXT    NOT NULL,
    away_sp         TEXT,
    home_sp         TEXT,
    away_xr         REAL,
    home_xr         REAL,
    total_xr        REAL,
    away_xwin       REAL,
    home_xwin       REAL,
    nrfi_pct        REAL
);

CREATE TABLE IF NOT EXISTS cannon_game_pitchers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    game_date       TEXT    NOT NULL,
    pitcher         TEXT    NOT NULL,
    team            TEXT,
    opponent        TEXT,
    exp_tbf         REAL,
    outs            REAL,
    k               REAL,
    bb              REAL,
    h               REAL,
    er              REAL
);

CREATE TABLE IF NOT EXISTS cannon_game_batters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    game_date       TEXT    NOT NULL,
    batter          TEXT    NOT NULL,
    team            TEXT,
    opp_sp          TEXT,
    batting_order   INTEGER,
    tot_pa          REAL,
    tot_h           REAL,
    tot_hr          REAL,
    tot_bb          REAL,
    tot_k           REAL,
    tot_tb          REAL,
    proj_r          REAL,
    proj_rbi        REAL,
    proj_hrr        REAL,
    team_runs       REAL
);

CREATE TABLE IF NOT EXISTS cannon_awards (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at      TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    league          TEXT    NOT NULL,
    player          TEXT    NOT NULL,
    team            TEXT,
    value           REAL
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA_SQL.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
    conn.commit()


def save_team_rankings(conn: sqlite3.Connection, rows: list[dict], fetched_at: str) -> int:
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO cannon_team_rankings
           (fetched_at, team, proj_w, proj_l, w_pct, rs_pg, ra_pg, rd_pg,
            div_pct, playoff_pct, champ_pct, ws_pct)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                fetched_at,
                r.get("Team"),
                _safe_float(r.get("proj_w")),
                _safe_float(r.get("proj_l")),
                _safe_float(r.get("w_pct")),
                _safe_float(r.get("rs_pg")),
                _safe_float(r.get("ra_pg")),
                _safe_float(r.get("rd_pg")),
                _safe_float(r.get("div_pct")),
                _safe_float(r.get("playoff_pct")),
                _safe_float(r.get("champ_pct")),
                _safe_float(r.get("ws_pct")),
            )
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def save_batter_rankings(conn: sqlite3.Connection, rows: list[dict], fetched_at: str) -> int:
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO cannon_batter_rankings
           (fetched_at, player, team, g, pa, h, hr, bb, k,
            avg, obp, slg, rv_per_game, rv_per_pa, season_runs)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                fetched_at,
                r.get("Player"),
                r.get("Team"),
                _safe_int(r.get("G")),
                _safe_int(r.get("PA")),
                _safe_int(r.get("H")),
                _safe_int(r.get("HR")),
                _safe_int(r.get("BB")),
                _safe_int(r.get("K")),
                _safe_float(r.get("AVG")),
                _safe_float(r.get("OBP")),
                _safe_float(r.get("SLG")),
                _safe_float(r.get("rv_per_game")),
                _safe_float(r.get("rv_per_PA")),
                _safe_float(r.get("season_runs")),
            )
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def save_pitcher_rankings(conn: sqlite3.Connection, rows: list[dict], fetched_at: str) -> int:
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO cannon_pitcher_rankings
           (fetched_at, player, team, g, gs, ip, tbf, k, bb, hr,
            k_pct, bb_pct, rv_per_9, season_runs)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                fetched_at,
                r.get("Player"),
                r.get("Team"),
                _safe_int(r.get("G")),
                _safe_int(r.get("GS")),
                _safe_float(r.get("IP")),
                _safe_int(r.get("TBF")),
                _safe_int(r.get("K")),
                _safe_int(r.get("BB")),
                _safe_int(r.get("HR")),
                _safe_float(r.get("K_pct")),
                _safe_float(r.get("BB_pct")),
                _safe_float(r.get("rv_per_9")),
                _safe_float(r.get("season_runs")),
            )
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def save_game_projections(
    conn: sqlite3.Connection,
    rows: list[dict],
    fetched_at: str,
    game_date: str,
) -> int:
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO cannon_game_projections
           (fetched_at, game_date, away_team, home_team, away_sp, home_sp,
            away_xr, home_xr, total_xr, away_xwin, home_xwin, nrfi_pct)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                fetched_at,
                game_date,
                r.get("away_team"),
                r.get("home_team"),
                r.get("away_sp"),
                r.get("home_sp"),
                _safe_float(r.get("away_xR")),
                _safe_float(r.get("home_xR")),
                _safe_float(r.get("total_xR")),
                _safe_float(r.get("away_xWin")),
                _safe_float(r.get("home_xWin")),
                _safe_float(r.get("nrfi_pct")),
            )
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def save_game_pitchers(
    conn: sqlite3.Connection,
    rows: list[dict],
    fetched_at: str,
    game_date: str,
) -> int:
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO cannon_game_pitchers
           (fetched_at, game_date, pitcher, team, opponent,
            exp_tbf, outs, k, bb, h, er)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                fetched_at,
                game_date,
                r.get("Pitcher"),
                r.get("Team"),
                r.get("Opponent"),
                _safe_float(r.get("exp_TBF")),
                _safe_float(r.get("Outs")),
                _safe_float(r.get("K")),
                _safe_float(r.get("BB")),
                _safe_float(r.get("H")),
                _safe_float(r.get("ER")),
            )
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def save_game_batters(
    conn: sqlite3.Connection,
    rows: list[dict],
    fetched_at: str,
    game_date: str,
) -> int:
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO cannon_game_batters
           (fetched_at, game_date, batter, team, opp_sp, batting_order,
            tot_pa, tot_h, tot_hr, tot_bb, tot_k, tot_tb,
            proj_r, proj_rbi, proj_hrr, team_runs)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                fetched_at,
                game_date,
                r.get("Batter"),
                r.get("Team"),
                r.get("Opp_SP"),
                _safe_int(r.get("Pos")),
                _safe_float(r.get("Tot_PA")),
                _safe_float(r.get("Tot_H")),
                _safe_float(r.get("Tot_HR")),
                _safe_float(r.get("Tot_BB")),
                _safe_float(r.get("Tot_K")),
                _safe_float(r.get("Tot_TB")),
                _safe_float(r.get("proj_R", r.get("Proj_R"))),
                _safe_float(r.get("proj_RBI", r.get("Proj_RBI"))),
                _safe_float(r.get("proj_HRR")),
                _safe_float(r.get("Team_Runs")),
            )
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def save_awards(conn: sqlite3.Connection, data: dict, fetched_at: str) -> int:
    if not data:
        return 0
    insert_rows: list[tuple] = []
    for category in ("mvp", "cy_young"):
        cat_data = data.get(category, {})
        for league in ("al", "nl"):
            for entry in cat_data.get(league, []):
                value_key = "total_raa" if category == "mvp" else "pitching_raa"
                insert_rows.append(
                    (
                        fetched_at,
                        category,
                        league.upper(),
                        entry.get("Player"),
                        entry.get("Team"),
                        _safe_float(entry.get(value_key)),
                    )
                )
    if insert_rows:
        conn.executemany(
            """INSERT INTO cannon_awards
               (fetched_at, category, league, player, team, value)
               VALUES (?,?,?,?,?,?)""",
            insert_rows,
        )
        conn.commit()
    return len(insert_rows)


def save_all(data: dict[str, Any], game_date: str | None = None) -> dict[str, int]:
    """Persist all fetched Cannon data to SQLite.  Returns row counts."""
    db_path = _get_db_path()
    fetched_at = datetime.now(timezone.utc).isoformat()
    game_date = game_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    counts: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        counts["team_rankings"] = save_team_rankings(
            conn, data.get("team_rankings", []), fetched_at
        )
        counts["batter_rankings"] = save_batter_rankings(
            conn, data.get("batter_rankings", []), fetched_at
        )
        counts["pitcher_rankings"] = save_pitcher_rankings(
            conn, data.get("pitcher_rankings", []), fetched_at
        )
        counts["game_projections"] = save_game_projections(
            conn, data.get("game_projections", []), fetched_at, game_date
        )
        counts["game_pitchers"] = save_game_pitchers(
            conn, data.get("game_pitchers", []), fetched_at, game_date
        )
        counts["game_batters"] = save_game_batters(
            conn, data.get("game_batters", []), fetched_at, game_date
        )
        counts["awards"] = save_awards(
            conn, data.get("awards", {}), fetched_at
        )
    return counts


# ---------------------------------------------------------------------------
# High-level accessors for the prediction pipeline
# ---------------------------------------------------------------------------

def _format_slate_date(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def fetch_cannon_game_projections_raw(
    slate_date: date | datetime | str | None = None,
) -> list[dict]:
    """Return the raw daily game projections list from Cannon.

    Each element is the raw JSON dict from the API with keys like
    away_team, home_team, away_sp, home_sp, away_xR, home_xR,
    total_xR, away_xWin, home_xWin, nrfi_pct.
    """
    params: dict[str, Any] = {"_": int(time.time())}
    slate_date_text = _format_slate_date(slate_date)
    if slate_date_text:
        params["date"] = slate_date_text
    try:
        games = fetch_endpoint("game_projections", params=params)
    except Exception:
        return []
    return games if isinstance(games, list) else []


def fetch_cannon_game_projections() -> dict[tuple[str, str], dict]:
    """Return daily game projections keyed by (away_last_word, home_last_word).

    Each value dict contains:
        cannon_away_xr, cannon_home_xr, cannon_total_xr,
        cannon_away_xwin, cannon_home_xwin, cannon_nrfi_pct,
        cannon_away_sp, cannon_home_sp

    Safe by construction: any scrape failure yields an empty dict.
    """
    try:
        games = fetch_endpoint("game_projections")
    except Exception:
        return {}

    result: dict[tuple[str, str], dict] = {}
    for g in games or []:
        away_key = _last_word(g.get("away_team"))
        home_key = _last_word(g.get("home_team"))
        if not away_key or not home_key:
            continue
        result[(away_key, home_key)] = {
            "cannon_away_xr": _safe_float(g.get("away_xR")),
            "cannon_home_xr": _safe_float(g.get("home_xR")),
            "cannon_total_xr": _safe_float(g.get("total_xR")),
            "cannon_away_xwin": _safe_float(g.get("away_xWin")),
            "cannon_home_xwin": _safe_float(g.get("home_xWin")),
            "cannon_nrfi_pct": _safe_float(g.get("nrfi_pct")),
            "cannon_away_sp": g.get("away_sp"),
            "cannon_home_sp": g.get("home_sp"),
        }
    return result


def fetch_cannon_team_ratings() -> dict[str, dict]:
    """Return season power ratings keyed by team last-word (lowercase).

    Each value contains:
        cannon_proj_w, cannon_proj_l, cannon_w_pct,
        cannon_rs_pg, cannon_ra_pg, cannon_rd_pg,
        cannon_playoff_pct, cannon_ws_pct
    """
    try:
        teams = fetch_endpoint("team_rankings")
    except Exception:
        return {}

    result: dict[str, dict] = {}
    for t in teams or []:
        key = _last_word(t.get("Team"))
        if not key:
            continue
        result[key] = {
            "cannon_proj_w": _safe_float(t.get("proj_w")),
            "cannon_proj_l": _safe_float(t.get("proj_l")),
            "cannon_w_pct": _safe_float(t.get("w_pct")),
            "cannon_rs_pg": _safe_float(t.get("rs_pg")),
            "cannon_ra_pg": _safe_float(t.get("ra_pg")),
            "cannon_rd_pg": _safe_float(t.get("rd_pg")),
            "cannon_playoff_pct": _safe_float(t.get("playoff_pct")),
            "cannon_ws_pct": _safe_float(t.get("ws_pct")),
        }
    return result


def fetch_cannon_pitcher_projections() -> dict[str, dict]:
    """Return daily pitcher projections keyed by pitcher name (lowercase).

    Each value contains:
        cannon_pitcher_k, cannon_pitcher_bb, cannon_pitcher_h,
        cannon_pitcher_er, cannon_pitcher_outs, cannon_pitcher_tbf,
        cannon_pitcher_team, cannon_pitcher_opponent
    """
    try:
        pitchers = fetch_endpoint("game_pitchers")
    except Exception:
        return {}

    result: dict[str, dict] = {}
    for p in pitchers or []:
        name = (p.get("Pitcher") or "").strip().lower()
        if not name:
            continue
        result[name] = {
            "cannon_pitcher_k": _safe_float(p.get("K")),
            "cannon_pitcher_bb": _safe_float(p.get("BB")),
            "cannon_pitcher_h": _safe_float(p.get("H")),
            "cannon_pitcher_er": _safe_float(p.get("ER")),
            "cannon_pitcher_outs": _safe_float(p.get("Outs")),
            "cannon_pitcher_tbf": _safe_float(p.get("exp_TBF")),
            "cannon_pitcher_team": p.get("Team"),
            "cannon_pitcher_opponent": p.get("Opponent"),
        }
    return result


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def print_summary(data: dict[str, Any]) -> None:
    """Pretty-print a summary of fetched Cannon data."""
    teams = data.get("team_rankings", [])
    print(f"\n=== Cannon Analytics Summary ===")
    print(f"Team rankings:      {len(teams)} teams")
    print(f"Batter rankings:    {len(data.get('batter_rankings', []))} players")
    print(f"Pitcher rankings:   {len(data.get('pitcher_rankings', []))} players")

    awards = data.get("awards", {})
    mvp_count = len(awards.get("mvp", {}).get("al", [])) + len(awards.get("mvp", {}).get("nl", []))
    cy_count = len(awards.get("cy_young", {}).get("al", [])) + len(awards.get("cy_young", {}).get("nl", []))
    print(f"Awards entries:     {mvp_count} MVP + {cy_count} Cy Young")

    games = data.get("game_projections", [])
    print(f"Game projections:   {len(games)} games")
    print(f"Game pitchers:      {len(data.get('game_pitchers', []))} entries")
    print(f"Game batters:       {len(data.get('game_batters', []))} entries")

    if games:
        print(f"\n--- Today's Game Projections ---")
        print(f"{'Away':<25} {'SP':<22} {'xR':>5} {'xWin':>6}  |  "
              f"{'Home':<25} {'SP':<22} {'xR':>5} {'xWin':>6}  "
              f"{'Total':>6} {'NRFI':>6}")
        print("-" * 140)
        for g in games:
            print(
                f"{g.get('away_team', ''):<25} "
                f"{g.get('away_sp', ''):<22} "
                f"{g.get('away_xR', 0):5.2f} "
                f"{g.get('away_xWin', 0):6.4f}  |  "
                f"{g.get('home_team', ''):<25} "
                f"{g.get('home_sp', ''):<22} "
                f"{g.get('home_xR', 0):5.2f} "
                f"{g.get('home_xWin', 0):6.4f}  "
                f"{g.get('total_xR', 0):6.2f} "
                f"{g.get('nrfi_pct', 0):6.1%}"
            )

    if teams:
        print(f"\n--- Top 10 Power Ratings ---")
        print(f"{'Team':<30} {'W':>5} {'L':>5} {'W%':>6} {'RS/G':>6} {'RA/G':>6} {'RD/G':>6} {'Playoff':>8} {'WS':>6}")
        print("-" * 100)
        for t in teams[:10]:
            print(
                f"{t.get('Team', ''):<30} "
                f"{t.get('proj_w', 0):5.1f} "
                f"{t.get('proj_l', 0):5.1f} "
                f"{t.get('w_pct', 0):6.4f} "
                f"{t.get('rs_pg', 0):6.2f} "
                f"{t.get('ra_pg', 0):6.2f} "
                f"{t.get('rd_pg', 0):6.2f} "
                f"{t.get('playoff_pct', 0):7.1%} "
                f"{t.get('ws_pct', 0):5.1%}"
            )


def main() -> int:
    print("Fetching Cannon Analytics data...")
    data = fetch_all()
    print_summary(data)

    print("\nSaving to database...")
    counts = save_all(data)
    for table, count in counts.items():
        print(f"  {table}: {count} rows")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
