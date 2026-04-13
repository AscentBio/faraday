from faraday.agents.sandbox.config import SandboxConfig


def test_sandbox_config_defaults_to_no_cloud_mounts():
    config = SandboxConfig(chat_id="chat-1")
    assert config.cloud_storage_mode == "disabled"
    assert config.needs_bucket is False


def test_sandbox_config_optional_cloud_requires_bucket_name():
    config = SandboxConfig(
        chat_id="chat-1",
        cloud_storage_mode="optional",
        bucket_name=None,
    )
    assert config.needs_bucket is False

    config.bucket_name = "faraday-app-user-data"
    assert config.needs_bucket is True


def test_sandbox_config_normalizes_invalid_storage_mode():
    config = SandboxConfig(
        chat_id="chat-1",
        cloud_storage_mode="unknown",
        bucket_name="faraday-app-user-data",
    )
    assert config.cloud_storage_mode == "disabled"


def test_sandbox_config_normalizes_workspace_mount_path():
    config = SandboxConfig(chat_id="chat-1", workspace_mount_path="sandbox")
    assert config.workspace_mount_path == "/sandbox"
