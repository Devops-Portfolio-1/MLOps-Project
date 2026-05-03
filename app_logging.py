import os
import json
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(name: str, obj: dict):
    path = LOG_DIR / name
    entry = {"ts": datetime.utcnow().isoformat() + "Z"}
    entry.update(obj or {})
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def log_inference(record: dict):
    """Append an inference event to logs/inference.jsonl"""
    _append_jsonl("inference.jsonl", record)


def log_feedback(record: dict):
    """Append user feedback to logs/feedback.jsonl"""
    _append_jsonl("feedback.jsonl", record)
