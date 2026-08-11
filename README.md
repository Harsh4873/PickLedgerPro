# harsh.bet

`harsh.bet` is the personal app launcher and shared read-only daily operating view. Every source app remains maintained in its own public repository; this repository owns the landing page and `/today/` integration route.

## Project map

| Path | Project | Repository |
| --- | --- | --- |
| `/` | harsh.bet landing | `Harsh4873/harsh4873.github.io` |
| `/today/` | Shared daily operating view | `Harsh4873/harsh4873.github.io` |
| `/pickledger/` | PickLedger | `Harsh4873/pickledger` |
| `/portfolio/` | Portfolio | `Harsh4873/portfolio` |
| `/daymark/` | Daymark | `Harsh4873/daymark` |
| `/slate/` | Slate | `Harsh4873/slate` |
| `/gym/` | Gym | `Harsh4873/gym` |
| `/fare/` | Fare | `Harsh4873/fare` |
| `/genes/` | MtbScope | `Harsh4873/genes` |
| `/research/` | Recall | `Harsh4873/research` |
| `/notes/` | Notes | `Harsh4873/notes` |
| `/shotlab/` | ShotLab | `Harsh4873/shotlab` |
| `/studies/` | Studies | `Harsh4873/studies` |
| `/degree/` | Degree Canvas | `Harsh4873/degree` |

The user-site repository owns the `harsh.bet` custom domain. GitHub Pages applies that domain to the project repositories at their matching paths, so project repositories do not contain a `CNAME` file.

`/today/` reads the four source apps' existing on-device mirrors without writing back. Its recommendation engine is deterministic and evidence-labelled: it combines task urgency, schedule openings, unresolved habits, remaining nutrition targets, planned training, and rolling consistency into up to three suggested actions plus a non-persistent day-pressure estimate.

## Local checks

```bash
npm ci
npm run typecheck
npm run upcheck
python3 -m pytest tests/smoke/test_landing.py -q
```

`npm run upcheck` builds the landing and validates its generated assets, project routes, custom-domain artifact, and A&M maroon theme metadata without opening a browser.

Historical app branches and the `pre-repo-split-2026-07-14` tag remain in this repository as migration rollback points. New app work belongs in the standalone repositories above.
