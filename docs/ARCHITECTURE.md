# Architecture

## Production shape

```text
Harsh4873/harsh4873.github.io  ──> harsh.bet/            (redirects to /portfolio/)
                                      harsh.bet/apps/
                                      harsh.bet/today/
Harsh4873/portfolio            ──> harsh.bet/portfolio/
Harsh4873/pickledger           ──> harsh.bet/pickledger/
Harsh4873/daymark              ──> harsh.bet/daymark/
Harsh4873/slate                ──> harsh.bet/slate/
Harsh4873/gym                  ──> harsh.bet/gym/
Harsh4873/fare                 ──> harsh.bet/fare/
Harsh4873/recipes              ──> harsh.bet/recipes/
Harsh4873/genes                ──> harsh.bet/genes/
Harsh4873/research             ──> harsh.bet/research/
Harsh4873/notes                ──> harsh.bet/notes/
Harsh4873/shotlab              ──> harsh.bet/shotlab/
Harsh4873/degree               ──> harsh.bet/degree/
Harsh4873/radar                ──> harsh.bet/radar/
```

The `/apps/` launcher is a small Vite + TypeScript page with no framework, persistence, authentication, model pipeline, or runtime API dependency. Its anchors work without JavaScript. The domain root redirects to `/portfolio/`.

Each project repository builds and deploys its own `dist/` artifact through GitHub Actions. The user-site repository is the only repository with `CNAME`; GitHub Pages inherits that custom domain for project-site paths.

## Landing deployment

`.github/workflows/deploy-pages.yml` runs on pushes to `main` and manual dispatches:

1. Install the locked Node dependencies and pytest.
2. Run `npm test`: the Today data contract plus the smoke tests.
3. Type-check and build the landing.
4. Copy `CNAME` and `.nojekyll` into `dist/` through the package `postbuild` step.
5. Validate compiled CSS/JavaScript, metadata, project paths, `robots.txt`, and the absence of bundled project directories.
6. Upload and deploy the artifact through GitHub Pages.

The workflow builds this repository only. It never checks out another app's branch and never defers on another project's data freshness: a landing or Today fix must not be able to pass with a green run that published nothing.

## Repository split

The previous composite site stored apps on branches and assembled every build in one workflow. The V1 split gave each app a clean `main` branch and an independent Pages boundary while retaining the original branches and rollback tag in this repository.

PickLedger's models, schedules, grading, committed data, and viewer now live together in `Harsh4873/pickledger`. Research keeps its frontend and backend together in `Harsh4873/research`; its Pages workflow builds only the frontend, while its backend remains a separate hosting boundary.

## Verification

```bash
npm run typecheck
npm run upcheck
python3 -m pytest tests/smoke/test_landing.py -q
```

Validation is source-, build-, workflow-, and API-based. Browser and deployed-output inspection is left to the repository owner.
