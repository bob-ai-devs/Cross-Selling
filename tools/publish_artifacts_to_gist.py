#!/usr/bin/env python
"""Publish the pre-trained model + encoders to a GitHub Gist (run on YOUR machine, not in the app).

Each file is base64-encoded and stored as "<name>.b64" because Gists only hold text.
The Streamlit app only ever READS this Gist.

Usage
  export GITHUB_TOKEN=ghp_xxx                 # classic token with the 'gist' scope
  python tools/publish_artifacts_to_gist.py                    # creates a new secret gist
  python tools/publish_artifacts_to_gist.py --gist-id <ID>     # updates an existing one

Looks for:  models/cross_selling.keras   and   encodes/*.pkl  (see core/config.py)
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.config import ALL_ARTIFACTS, LOCAL_ENCODES_DIR, LOCAL_MODEL_DIR, MODEL_FILE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    ap.add_argument("--gist-id")
    ap.add_argument("--public", action="store_true", help="create a public gist (default: secret)")
    args = ap.parse_args()
    if not args.token:
        sys.exit("Provide a token via --token or the GITHUB_TOKEN environment variable.")

    files, missing = {}, []
    for name in ALL_ARTIFACTS:
        path = (LOCAL_MODEL_DIR if name == MODEL_FILE else LOCAL_ENCODES_DIR) / name
        if not path.exists():
            missing.append(str(path))
            continue
        files[f"{name}.b64"] = {"content": base64.b64encode(path.read_bytes()).decode("ascii")}
    if missing:
        sys.exit("Missing files:\n  " + "\n  ".join(missing))

    headers = {"Authorization": f"Bearer {args.token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    if args.gist_id:
        resp = requests.patch(f"https://api.github.com/gists/{args.gist_id}", headers=headers,
                              json={"files": files}, timeout=120)
    else:
        resp = requests.post("https://api.github.com/gists", headers=headers, timeout=120,
                             json={"description": "Cross-selling model artifacts", "public": args.public,
                                   "files": files})
    resp.raise_for_status()
    gist = resp.json()
    print("Gist id:", gist["id"])
    print("URL    :", gist["html_url"])
    print("\nPaste into Streamlit secrets:\n[github]\ntoken = \"<your token>\"\ngist_id = \"%s\"" % gist["id"])


if __name__ == "__main__":
    main()
