"""What produced the numbers: ``results/run_manifest.json``.

Every stage of the CLI calls :func:`write` before it does any work, so the file
always describes the run that is about to (over)write the tables.  It records

    * the git commit of the code (and whether the working tree was dirty),
    * the Python version and the version of every package that matters,
    * every seed and the full command line,
    * the models that were requested, with the HuggingFace id and the pinned
      revision of each (``Model.revision``; ``null`` means *not pinned*),
    * one entry per stage that has run, so the history of a results folder is kept.

Nothing here touches the analysis; it only reads the environment.  Absolute paths
and host names are deliberately left out so the file can be committed.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from . import config
from .models import MODELS

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openpyxl",
            "biopython", "tqdm", "torch", "transformers", "datasets", "tokenizers")


def git_state(cwd: Path | None = None) -> dict:
    """Commit hash and cleanliness of the code; ``None`` outside a git checkout."""
    cwd = str(cwd or config.CODE_ROOT)

    def run(*args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              timeout=20, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_identifiers(keys=None) -> dict:
    """HuggingFace id and pinned revision of every requested model."""
    chosen = keys or list(MODELS)
    return {key: {"hf_name": MODELS[key].hf_name or None,
                  "revision": MODELS[key].revision,
                  "conda_env": MODELS[key].conda_env or None}
            for key in chosen if key in MODELS}


def build(stage: str, argv=None, seeds: dict | None = None, models=None,
          extra: dict | None = None) -> dict:
    return {
        "stage": stage,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command_line": list(sys.argv if argv is None else argv),
        "seeds": dict(seeds or {}),
        "models": model_identifiers(models),
        "extra": dict(extra or {}),
    }


def write(stage: str, argv=None, seeds: dict | None = None, models=None,
          extra: dict | None = None, path: Path | None = None) -> Path:
    """(Re)write the manifest, keeping the record of earlier stages."""
    path = Path(path or config.MANIFEST_FILE)
    previous = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf8"))
        except (OSError, ValueError):
            previous = {}

    entry = build(stage, argv, seeds, models, extra)
    history = [item for item in previous.get("stages", []) if item.get("stage") != stage]
    history.append(entry)

    manifest = {
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "stages": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return path
