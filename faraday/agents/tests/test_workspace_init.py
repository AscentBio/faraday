from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from faraday.agents.code_execution_agent import BaseCodeExecutionAgent


def test_workspace_init_mode_copy_creates_isolated_workspace(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "state.txt").write_text("original", encoding="utf-8")

    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  workspace:",
                f"    source_root: {source_root}",
                "    init_mode: copy",
                f"    copy_root: {tmp_path / 'workspace-copies'}",
                "    keep_copy: true",
                "execution:",
                "  backend: host",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))

    agent = BaseCodeExecutionAgent(enable_sandbox=False, verbose=False)
    copy_root = Path(agent.workspace_root).resolve()
    assert copy_root != source_root.resolve()
    assert (copy_root / "state.txt").read_text(encoding="utf-8") == "original"

    (copy_root / "state.txt").write_text("changed-in-copy", encoding="utf-8")
    assert (source_root / "state.txt").read_text(encoding="utf-8") == "original"

    asyncio.run(agent.cleanup())
    assert copy_root.exists()
    shutil.rmtree(copy_root, ignore_errors=True)


def test_workspace_init_mode_copy_cleans_up_when_keep_false(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "state.txt").write_text("original", encoding="utf-8")

    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  workspace:",
                f"    source_root: {source_root}",
                "    init_mode: copy",
                f"    copy_root: {tmp_path / 'workspace-copies'}",
                "    keep_copy: false",
                "execution:",
                "  backend: host",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))

    agent = BaseCodeExecutionAgent(enable_sandbox=False, verbose=False)
    copy_root = Path(agent.workspace_root).resolve()
    assert copy_root.exists()

    asyncio.run(agent.cleanup())
    assert not copy_root.exists()
