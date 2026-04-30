# run_all.py
# ─────────────────────────────────────────────────────────────────────────────
# Activity 2 — chain Block A → Block B → Block C → Block D.
#
# Also writes reports/activity2/manifest.json with timestamp, git commit,
# config hash, and the versions of the heavy ML libraries — answers the
# rubric's "could someone reproduce my results" requirement.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import datetime
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys

from ..config import FIGURES_DIR
from .compare_imputations import main as block_a
from .compare_imbalance   import main as block_b
from .tune_models         import main as block_c
from .final_evaluation    import main as block_d
from .crash_focus         import main as block_e


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _config_hash() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "config.py")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _lib_versions() -> dict[str, str]:
    libs = ["sklearn", "xgboost", "imblearn", "optuna", "shap", "pandas", "numpy", "joblib"]
    out = {}
    for name in libs:
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[name] = "missing"
    return out


def write_manifest() -> None:
    manifest = {
        "timestamp_utc":   datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "git_commit":      _git_commit(),
        "python":          sys.version.split()[0],
        "platform":        platform.platform(),
        "config_sha256":   _config_hash(),
        "library_versions": _lib_versions(),
    }
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(os.path.join(FIGURES_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest → {FIGURES_DIR}/manifest.json")


def main() -> None:
    print("\n" + "▓" * 60)
    print("ACTIVITY 2 — full ML pipeline (A → B → C → D → E)")
    print("▓" * 60)
    block_a()
    print()
    block_b()
    print()
    block_c()
    print()
    block_d()
    print()
    block_e()
    print()
    write_manifest()
    print("▓" * 60)
    print("All blocks complete.")
    print("▓" * 60)


if __name__ == "__main__":
    main()
