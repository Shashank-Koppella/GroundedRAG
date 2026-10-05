"""
Lists the model ids your GROQ_API_KEY can actually use (Groq's public docs list models that an
individual account/org may not have access to). Reads GROQ_API_KEY from .env at the repo root.

    python -m scripts.list_groq_models
"""
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


def main():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY is not set. Put it in .env at the repo root (see .env.example).")
    resp = requests.get("https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if resp.status_code != 200:
        sys.exit(f"Groq returned {resp.status_code}: {resp.text[:300]}")
    rows = sorted(resp.json().get("data", []), key=lambda m: m["id"])
    print(f"{len(rows)} models available to this key:\n")
    for m in rows:
        ctx = m.get("context_window", "?")
        active = "" if m.get("active", True) else "  (inactive)"
        print(f"  {m['id']:<48} context={ctx}  owner={m.get('owned_by', '?')}{active}")


if __name__ == "__main__":
    main()
