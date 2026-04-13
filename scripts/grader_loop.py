#!/usr/bin/env python3
import json
import logging
import os
import sys

LOG_PATH = os.path.expanduser("~/Library/Logs/pickledger_grader.log")
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)


if __name__ == "__main__":
    logging.info("--- grader tick ---")
    try:
        from pickgrader_server import run_background_grade_all_users

        result = run_background_grade_all_users()
        logging.info(
            "graded=%s skipped=%s errors=%s",
            result.get("graded_users"),
            result.get("skipped"),
            json.dumps(result.get("errors", [])),
        )
    except Exception as exc:
        logging.error("tick failed: %s", exc)
