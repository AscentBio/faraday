"""One-time cleanup for workspace layouts predating ./agent_outputs + cloud-storage-legacy."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def migrate_legacy_faraday_cloud_storage(workspace_root: Path) -> None:
    """Drop deprecated ``.faraday_runtime/cloud-storage`` after moving files to current locations.

    Old default: ``cloud_storage_root`` was ``.faraday_runtime/cloud-storage`` with
    ``agent_outputs`` and ``.modal_code_vars`` nested underneath. Current layout uses
    workspace ``./agent_outputs`` and ``.faraday_runtime/.modal_code_vars``.

    Set ``FARADAY_SKIP_LEGACY_CLOUD_STORAGE_MIGRATION=1`` to disable.
    """
    if os.getenv("FARADAY_SKIP_LEGACY_CLOUD_STORAGE_MIGRATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    root = workspace_root.expanduser().resolve()
    runtime = root / ".faraday_runtime"
    old_root = runtime / "cloud-storage"
    if not old_root.is_dir():
        return

    new_vars = runtime / ".modal_code_vars"
    old_vars = old_root / ".modal_code_vars"
    if old_vars.is_dir():
        new_vars.mkdir(parents=True, exist_ok=True)
        for item in old_vars.iterdir():
            dest = new_vars / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))

    old_out = old_root / "agent_outputs"
    new_out = root / "agent_outputs"
    if old_out.is_dir():
        new_out.mkdir(parents=True, exist_ok=True)
        for path in old_out.rglob("*"):
            if path.is_file():
                rel = path.relative_to(old_out)
                dest = new_out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(path, dest)

    shutil.rmtree(old_root, ignore_errors=True)
