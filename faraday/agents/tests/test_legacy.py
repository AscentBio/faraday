from __future__ import annotations

from pathlib import Path

from faraday.agents.sandbox.legacy import migrate_legacy_faraday_cloud_storage


def test_migrate_legacy_cloud_storage_moves_and_removes_old_tree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FARADAY_SKIP_LEGACY_CLOUD_STORAGE_MIGRATION", raising=False)
    root = tmp_path / "proj"
    old_cs = root / ".faraday_runtime" / "cloud-storage"
    (old_cs / ".modal_code_vars").mkdir(parents=True)
    (old_cs / ".modal_code_vars" / "variables.pkl").write_bytes(b"x")
    (old_cs / "agent_outputs" / "plots").mkdir(parents=True)
    (old_cs / "agent_outputs" / "plots" / "a.png").write_bytes(b"png")

    migrate_legacy_faraday_cloud_storage(root)

    assert not old_cs.exists()
    assert (root / ".faraday_runtime" / ".modal_code_vars" / "variables.pkl").read_bytes() == b"x"
    assert (root / "agent_outputs" / "plots" / "a.png").read_bytes() == b"png"


def test_migrate_legacy_skipped_when_env_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FARADAY_SKIP_LEGACY_CLOUD_STORAGE_MIGRATION", "1")
    root = tmp_path / "proj"
    old_cs = root / ".faraday_runtime" / "cloud-storage"
    old_cs.mkdir(parents=True)
    migrate_legacy_faraday_cloud_storage(root)
    assert old_cs.is_dir()
