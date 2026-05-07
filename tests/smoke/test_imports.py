from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def prepended_sys_paths(*paths: Path):
    original = list(sys.path)
    for path in reversed([str(p) for p in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def test_locked_runtime_imports_still_resolve():
    imports = [
        ("pickgrader_server", (REPO_ROOT,)),
        ("runlive", (REPO_ROOT,)),
        ("MLBPredictionModel.cannon_daily_adapter", (REPO_ROOT,)),
        ("MLBPredictionModel.date_utils", (REPO_ROOT,)),
        ("NBAPredictionModel.run_live", (REPO_ROOT / "NBAPredictionModel", REPO_ROOT)),
        ("WNBAPredictionModel.wnba_picks", (REPO_ROOT,)),
        ("NBAPlayerBettingModel.run_props", (REPO_ROOT / "NBAPlayerBettingModel", REPO_ROOT)),
        ("NBAPlayoffsPredictionModel.run_live", (REPO_ROOT,)),
        ("models.mlb_inning.mlb_inning_model", (REPO_ROOT,)),
        ("ipl.run_api", (REPO_ROOT,)),
        ("scripts.firebase_writer", (REPO_ROOT,)),
        ("scripts.seed_record", (REPO_ROOT,)),
        ("scripts.grader_loop", (REPO_ROOT,)),
    ]

    for module_name, paths in imports:
        with prepended_sys_paths(*paths):
            importlib.import_module(module_name)
