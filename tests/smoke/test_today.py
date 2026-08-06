from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_today_is_a_read_only_cross_app_dashboard():
    html = (ROOT / "today" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "src" / "today" / "main.ts").read_text(encoding="utf-8")
    data = (ROOT / "src" / "today" / "data.ts").read_text(encoding="utf-8")

    assert 'href="https://harsh.bet/today/"' in html
    assert 'id="priority-card"' in html
    assert 'id="schedule-card"' in html
    assert 'id="habits-card"' in html
    assert 'id="nutrition-card"' in html
    assert 'id="workout-card"' in html
    assert 'id="trend-card"' in html
    assert "Daymark" in script
    assert "Slate" in script
    assert "Fare" in script
    assert "Gym" in script
    assert "put(" not in data
    assert "deleteDatabase" not in data
    assert "request.transaction?.abort()" in data


def test_today_uses_each_apps_existing_storage_contract():
    data = (ROOT / "src" / "today" / "data.ts").read_text(encoding="utf-8")
    for contract in (
        "daymark-tracker-state-v2",
        "slate-todo-state-v1",
        "fare-state-v1",
        "harsh-gym-program-v1",
        "harsh-gym-logs-v1",
    ):
        assert contract in data


def test_today_is_in_the_multipage_build():
    config = (ROOT / "vite.config.ts").read_text(encoding="utf-8")
    assert "today: 'today/index.html'" in config
