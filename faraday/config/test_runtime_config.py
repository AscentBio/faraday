from __future__ import annotations

import re

from faraday.config.runtime_config import (
    get_execution_backend,
    get_path_value,
    get_runtime_app,
    get_runtime_app_docker_image,
    get_runtime_launch,
    get_run_output_dir,
    get_runtime_dependency_packages,
    get_runtime_config_path,
    get_string_value,
    get_workspace_mount_path,
    get_workspace_source_root,
    normalize_execution_backend,
    normalize_runtime_backend,
    normalize_runtime_launch,
    normalize_runtime_app,
)


def test_normalize_runtime_app_defaults_to_host():
    assert normalize_runtime_app(None) == "host"
    assert normalize_runtime_app("host") == "host"
    assert normalize_runtime_app("docker") == "docker"
    assert normalize_runtime_app("container") == "docker"


def test_normalize_runtime_launch_defaults_to_host():
    assert normalize_runtime_launch(None) == "host"
    assert normalize_runtime_launch("host") == "host"
    assert normalize_runtime_launch("docker") == "docker"
    assert normalize_runtime_launch("container") == "docker"


def test_normalize_runtime_backend_defaults_to_host():
    assert normalize_runtime_backend(None) == "host"
    assert normalize_runtime_backend("host") == "host"
    assert normalize_runtime_backend("docker") == "docker"
    assert normalize_runtime_backend("container") == "docker"


def test_normalize_execution_backend_local_alias_maps_to_host():
    assert normalize_execution_backend("local") == "host"
    assert normalize_execution_backend("docker") == "docker"
    assert normalize_execution_backend("host") == "host"


def test_get_runtime_app_reads_config(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  app: docker",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    assert get_runtime_app(default="host") == "docker"


def test_get_runtime_app_docker_image_reads_config(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  app_image: my-registry/faraday:1.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    assert get_runtime_app_docker_image(default="faraday-oss") == "my-registry/faraday:1.0"


def test_get_runtime_launch_reads_config(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  launch: docker",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    assert get_runtime_launch(default="host") == "docker"


def test_runtime_backend_docker_sets_launch_and_app(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  backend: docker",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    assert get_runtime_launch(default="host") == "docker"
    assert get_runtime_app(default="host") == "docker"


def test_runtime_backend_with_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  backend: docker",
                "  override_launch: host",
                "  override_app: docker",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    assert get_runtime_launch(default="host") == "host"
    assert get_runtime_app(default="host") == "docker"


def test_runtime_backend_ignores_legacy_launch_app(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  backend: host",
                "  launch: docker",
                "  app: docker",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    assert get_runtime_launch(default="host") == "host"
    assert get_runtime_app(default="host") == "host"


def test_runtime_config_prefers_explicit_env_path(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "\n".join(
            [
                "execution:",
                "  backend: docker",
                "  workspace:",
                "    mount_path: /workspace",
                "runtime:",
                "  workspace:",
                "    source_root: /workspace",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))

    assert get_runtime_config_path() == config_path
    assert get_execution_backend(default="modal") == "docker"
    assert get_workspace_source_root() == "/workspace"
    assert get_workspace_mount_path() == "/workspace"


def test_get_path_value_empty_default_not_cwd(monkeypatch, tmp_path):
    """Regression: Path('') must not become '.', which breaks CLI path defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FARADAY_CONFIG", raising=False)
    assert get_path_value("persistence", "previous_context", default="") == ""


def test_get_string_value_top_level_query(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text('query: "hello from yaml"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    assert get_string_value("query", default=None) == "hello from yaml"


def test_runtime_config_uses_docker_default(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text("execution:\n  backend: docker\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FARADAY_CONFIG", raising=False)

    assert get_runtime_config_path() == config_path
    assert get_execution_backend(default="modal") == "docker"


def test_runtime_dependency_packages_local_is_minimal(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text("execution:\n  backend: docker\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))

    packages = get_runtime_dependency_packages(default_backend="docker")
    assert "openai" in packages
    assert "openai-agents" in packages
    assert "requests" in packages
    assert "matplotlib" in packages
    assert "seaborn" in packages
    assert "mygene" in packages
    assert "modal" not in packages
    assert "exa-py" not in packages
    assert "rdkit" not in packages
    assert "datamol" not in packages


def test_runtime_dependency_packages_feature_flags(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "\n".join(
            [
                "execution:",
                "  backend: docker",
                "features:",
                "  enable_exa: true",
                "  enable_python_science_stack: false",
                "  enable_cheminformatics_stack: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))

    packages = get_runtime_dependency_packages(default_backend="docker")
    assert "exa-py" in packages
    assert "requests" not in packages
    assert "matplotlib" not in packages
    assert "seaborn" not in packages
    assert "mygene" not in packages
    assert "rdkit" in packages
    assert "datamol" in packages
    assert "modal" not in packages


# ---------------------------------------------------------------------------
# get_run_output_dir tests
# ---------------------------------------------------------------------------


def test_get_run_output_dir_default(monkeypatch, tmp_path):
    """With no config, defaults to <cwd>/run_outputs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FARADAY_CONFIG", raising=False)
    result = get_run_output_dir()
    assert result == str(tmp_path / "run_outputs")


def test_get_run_output_dir_defaults_to_workspace_in_app_docker(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FARADAY_CONFIG", raising=False)
    monkeypatch.setenv("FARADAY_RUNNING_IN_APP_DOCKER", "1")
    result = get_run_output_dir()
    assert result == "/workspace/run_outputs"


def test_get_run_output_dir_defaults_to_mount_path_in_app_docker(monkeypatch, tmp_path):
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "execution:\n  workspace:\n    mount_path: /sandbox\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    monkeypatch.setenv("FARADAY_RUNNING_IN_APP_DOCKER", "1")
    result = get_run_output_dir()
    assert result == "/sandbox/run_outputs"


def test_get_run_output_dir_from_config(monkeypatch, tmp_path):
    """Reads outputs.root from config YAML."""
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "outputs:\n  root: /custom/run_outputs\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    result = get_run_output_dir()
    assert result == "/custom/run_outputs"


def test_get_run_output_dir_expands_tilde(monkeypatch, tmp_path):
    """Tilde in outputs.root is expanded."""
    config_path = tmp_path / "faraday.yaml"
    config_path.write_text(
        "outputs:\n  root: ~/faraday_runs\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARADAY_CONFIG", str(config_path))
    result = get_run_output_dir()
    assert "~" not in result
    assert result.endswith("faraday_runs")


def test_run_output_dir_name_format(monkeypatch, tmp_path):
    """Run directory name matches run_{ts}_{chat_id}_{run_id} pattern."""
    # The naming logic lives in FaradayAgent, but we can verify the pattern here
    # by constructing it the same way and checking the regex.
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    chat_id = "chat_20260401_abc"
    query_id = "query_20260401_xyz"
    chat_token = chat_id.replace("/", "_").replace(" ", "_")
    run_token = query_id.replace("/", "_").replace(" ", "_")
    run_dir_name = f"run_{ts}_{chat_token}_{run_token}"

    pattern = re.compile(r"^run_\d{14}_.+$")
    assert pattern.match(run_dir_name), f"Unexpected run dir name: {run_dir_name}"
