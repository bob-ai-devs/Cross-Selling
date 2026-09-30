"""Read-only artifact loading.

Order of preference
  1. GitHub Gist  (secrets: [github] token + gist_id)   -> base64 files "<name>.b64"
  2. Local repo   (models/cross_selling.keras, encodes/*.pkl)

Nothing is ever written back to the Gist or to disk. The Keras file is unpacked
into a temporary directory that is deleted as soon as the model is in memory.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path

import requests
import streamlit as st

from core.config import (ALL_ARTIFACTS, ENCODER_FILES, LOCAL_ENCODES_DIR,
                         LOCAL_MODEL_DIR, MODEL_FILE)

GIST_API = "https://api.github.com/gists/{gist_id}"


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def credentials() -> tuple[str | None, str | None]:
    """(token, gist_id) from st.secrets ([github] section or flat keys) or env."""
    token = gist_id = None
    try:
        sec = st.secrets
        if "github" in sec:
            token = sec["github"].get("token") or sec["github"].get("GITHUB_TOKEN")
            gist_id = sec["github"].get("gist_id") or sec["github"].get("GIST_ID")
        token = token or sec.get("GITHUB_TOKEN") or sec.get("github_token")
        gist_id = gist_id or sec.get("GIST_ID") or sec.get("gist_id")
    except Exception:  # no secrets.toml at all
        pass
    token = token or os.environ.get("GITHUB_TOKEN")
    gist_id = gist_id or os.environ.get("GIST_ID")
    return (str(token).strip() if token else None, str(gist_id).strip() if gist_id else None)


def local_files_present() -> dict[str, bool]:
    out = {MODEL_FILE: (LOCAL_MODEL_DIR / MODEL_FILE).exists()}
    for f in ENCODER_FILES:
        out[f] = (LOCAL_ENCODES_DIR / f).exists()
    return out


# --------------------------------------------------------------------------- #
# Gist access (read only)
# --------------------------------------------------------------------------- #
def _fetch_gist_blobs(token: str | None, gist_id: str) -> dict[str, bytes]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(GIST_API.format(gist_id=gist_id), headers=headers, timeout=30)
    if resp.status_code == 404:
        raise RuntimeError("Gist not found (check gist_id, and that the token can read it).")
    if resp.status_code in (401, 403):
        raise RuntimeError(f"GitHub rejected the token (HTTP {resp.status_code}).")
    resp.raise_for_status()

    blobs: dict[str, bytes] = {}
    for name, meta in resp.json().get("files", {}).items():
        if not name.endswith(".b64"):
            continue
        content = meta.get("content")
        if meta.get("truncated") or content is None:
            raw = requests.get(meta["raw_url"], headers=headers, timeout=60)
            raw.raise_for_status()
            content = raw.text
        blobs[name[:-4]] = base64.b64decode("".join(content.split()))
    return blobs


def _read_local(name: str) -> bytes | None:
    path = (LOCAL_MODEL_DIR / name) if name == MODEL_FILE else (LOCAL_ENCODES_DIR / name)
    return path.read_bytes() if path.exists() else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_all() -> dict:
    import joblib

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    token, gist_id = credentials()

    blobs: dict[str, bytes] = {}
    origin: dict[str, str] = {}
    notes: list[str] = []

    if gist_id:
        try:
            for k, v in _fetch_gist_blobs(token, gist_id).items():
                if k in ALL_ARTIFACTS:
                    blobs[k] = v
                    origin[k] = "gist"
        except Exception as exc:  # network / auth problems -> fall back to repo files
            notes.append(f"Gist unavailable ({exc}); trying files bundled in the repo.")

    for name in ALL_ARTIFACTS:
        if name not in blobs:
            data = _read_local(name)
            if data is not None:
                blobs[name] = data
                origin[name] = "repo"

    missing = [n for n in ALL_ARTIFACTS if n not in blobs]
    if missing:
        reason = (" " + " ".join(notes)) if notes else ""
        raise FileNotFoundError(
            "Missing pre-trained artifacts: " + ", ".join(missing) + "." + reason
            + " Publish them to your Gist (tools/publish_artifacts_to_gist.py) "
              "or commit them under models/ and encodes/."
        )

    import tensorflow as tf  # heavy import kept lazy

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / MODEL_FILE
        path.write_bytes(blobs[MODEL_FILE])
        model = tf.keras.models.load_model(path, compile=False)

    encoders = {n[:-4]: joblib.load(io.BytesIO(blobs[n])) for n in ENCODER_FILES}

    kinds = set(origin.values())
    if kinds == {"gist"}:
        source = f"GitHub Gist ({gist_id[:7]}…)"
    elif kinds == {"repo"}:
        source = "Repository files"
    else:
        source = "Gist + repository files"
    return {"model": model, "encoders": encoders, "source": source, "notes": notes}


@st.cache_resource(show_spinner="Loading pre-trained model and encoders…", ttl=6 * 3600)
def _cached_artifacts(_gist_key: str) -> dict:
    return _load_all()


def get_artifacts() -> tuple[dict | None, str | None]:
    """Return (artifacts, error). Failures are remembered until 'Retry' is used."""
    if st.session_state.get("_art_error"):
        return None, st.session_state["_art_error"]
    _, gist_id = credentials()
    try:
        art = _cached_artifacts(gist_id or "local")
        st.session_state["_art_source"] = art["source"]
        return art, None
    except Exception as exc:
        st.session_state["_art_error"] = f"{type(exc).__name__}: {exc}"
        return None, st.session_state["_art_error"]


def reload_artifacts() -> None:
    st.session_state.pop("_art_error", None)
    st.session_state.pop("_art_source", None)
    _cached_artifacts.clear()
