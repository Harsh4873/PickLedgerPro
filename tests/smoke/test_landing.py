from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LandingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self.stylesheets: list[str] = []
        self.scripts: list[dict[str, str]] = []
        self.main_ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "a":
            self.links.append(values)
        elif tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(values.get("href", ""))
        elif tag == "script":
            self.scripts.append(values)
        elif tag == "main":
            self.main_ids.append(values.get("id", ""))


def _landing() -> tuple[str, LandingParser]:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    parser = LandingParser()
    parser.feed(html)
    return html, parser


def test_landing_has_app_metadata_and_accessible_structure():
    html, parser = _landing()

    assert "<title>Harsh Dave - Apps</title>" in html
    assert 'content="https://harsh.bet/"' in html
    assert '<link rel="canonical" href="https://harsh.bet/"' in html
    assert 'class="skip-link" href="#main-content"' in html
    assert parser.main_ids == ["main-content"]
    assert parser.stylesheets == ["./src/styles/landing.css"]
    assert any(
        script.get("type") == "module" and script.get("src") == "./src/main.ts"
        for script in parser.scripts
    )
    assert "fonts.googleapis.com" not in html


def test_landing_routes_every_app_without_new_tabs():
    _, parser = _landing()
    expected_paths = {
        "/today/",
        "/portfolio/",
        "/daymark/",
        "/slate/",
        "/pickledger/",
        "/genes/",
        "/research/",
        "/notes/",
        "/fare/",
        "/gym/",
        "/shotlab/",
        "/studies/",
        "/degree/",
    }
    app_cards = [link for link in parser.links if "app-card" in link.get("class", "")]

    assert {link.get("href") for link in app_cards} == expected_paths
    assert len(app_cards) == len(expected_paths)
    assert all(link.get("target") in (None, "") for link in app_cards)


def test_landing_uses_the_app_rail_and_shared_theme():
    html, _ = _landing()
    css = (ROOT / "src" / "styles" / "landing.css").read_text(encoding="utf-8")
    script = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")

    assert 'class="app-rail"' in html
    assert ".app-rail:hover" in css
    assert ".app-rail:focus-within" in css
    assert "transform: translateX" in css
    assert ":root[data-theme='light']" in css
    assert "--accent-deep: #500000;" in css
    assert "--bg: #151513;" in css
    assert "data-mobile-open" in script
    assert "Escape" in script
    assert "linear-gradient" not in css
    assert "radial-gradient" not in css


def test_landing_is_responsive_and_accessible():
    css = (ROOT / "src" / "styles" / "landing.css").read_text(encoding="utf-8")

    assert ":focus-visible" in css
    assert "@media (max-width: 820px)" in css
    assert "min-width: 320px" in css
    assert not (ROOT / "data").exists()
    assert not (ROOT / "models").exists()
    assert not (ROOT / "player_props").exists()


def test_pages_workflow_builds_the_shared_artifact():
    workflow = (ROOT / ".github" / "workflows" / "deploy-pages.yml").read_text(encoding="utf-8")

    assert "actions/checkout@v4" in workflow
    assert "actions/setup-node@v4" in workflow
    assert "npm run build" in workflow
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "path: dist" in workflow
    assert "github.event_name == 'push'" in workflow
    assert "test -f dist/today/index.html" in workflow
