#!/usr/bin/env python3
"""Validate the built harsh.bet landing without opening rendered output."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PROJECT_PATHS = (
    "/daymark/",
    "/slate/",
    "/pickledger/",
    "/genes/",
    "/research/",
    "/fare/",
    "/gym/",
    "/portfolio/",
)
def main() -> int:
    failures: list[str] = []

    source_path = ROOT / "index.html"
    built_path = DIST / "index.html"
    source = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
    built = built_path.read_text(encoding="utf-8") if built_path.is_file() else ""

    if not source:
        failures.append("index.html is missing")
    if not built:
        failures.append("dist/index.html is missing")

    source_contract = (
        ('<link rel="canonical" href="https://harsh.bet/"', "canonical URL"),
        ('<meta name="theme-color" content="#151513"', "dark theme color"),
        ('href="./src/styles/landing.css"', "landing stylesheet entry"),
        ('src="./src/main.ts"', "TypeScript module entry"),
        ('href="/today/"', "Today dashboard route"),
    )
    for marker, label in source_contract:
        if marker not in source:
            failures.append(f"source is missing {label}")

    for path in PROJECT_PATHS:
        if f'href="{path}"' not in source:
            failures.append(f"source is missing project path {path}")
        if (DIST / path.strip("/")).exists():
            failures.append(f"landing artifact unexpectedly bundles {path}")

    today_source = ROOT / "today" / "index.html"
    today_built = DIST / "today" / "index.html"
    if not today_source.is_file():
        failures.append("today/index.html is missing")
    if not today_built.is_file():
        failures.append("dist/today/index.html is missing")
    if today_source.is_file():
        today_html = today_source.read_text(encoding="utf-8")
        today_contract = (
            ('href="https://harsh.bet/today/"', "Today canonical URL"),
            ('src="../src/today/main.ts"', "Today TypeScript entry"),
            ('id="priority-card"', "Today priority card"),
            ('id="advice-card"', "Today advice card"),
            ('id="nutrition-card"', "Today nutrition card"),
            ('id="workout-card"', "Today workout card"),
            ('id="trend-card"', "Today trend card"),
            ('data-theme-option="light"', "Today light theme control"),
            ('data-theme-option="system"', "Today system theme control"),
            ('data-theme-option="dark"', "Today dark theme control"),
        )
        for marker, label in today_contract:
            if marker not in today_html:
                failures.append(f"source is missing {label}")

    if "src/main.ts" in built or "src/styles/landing.css" in built:
        failures.append("dist/index.html still references source files")

    asset_refs = re.findall(r'(?:href|src)="(\./assets/[^"]+)"', built)
    if not any(ref.endswith(".css") for ref in asset_refs):
        failures.append("compiled CSS asset is missing")
    if not any(ref.endswith(".js") for ref in asset_refs):
        failures.append("compiled JavaScript asset is missing")
    for reference in asset_refs:
        if not (DIST / reference.removeprefix("./")).is_file():
            failures.append(f"compiled asset is missing: {reference}")

    cname = DIST / "CNAME"
    if not cname.is_file() or cname.read_text(encoding="utf-8").strip() != "harsh.bet":
        failures.append("dist/CNAME must contain harsh.bet")
    if not (DIST / ".nojekyll").is_file():
        failures.append("dist/.nojekyll is missing")
    resume = DIST / "resume.pdf"
    if not resume.is_file() or resume.stat().st_size < 100_000:
        failures.append("dist/resume.pdf is missing or unexpectedly small")

    if failures:
        for failure in failures:
            print(f"[upcheck] {failure}")
        return 1

    print(f"[upcheck] healthy landing, Today dashboard, {len(PROJECT_PATHS) - 1} systems, Portfolio, and Resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
