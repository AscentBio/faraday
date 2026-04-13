"""Simple config-driven OpenAI-compatible client setup."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from faraday.config import get_config_value, get_string_value

log = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
    },
    "azure": {
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "base_url_env": "AZURE_OPENAI_BASE_URL",
        "api_version": "preview",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": _OPENROUTER_BASE_URL,
    },
}


def _get_llm_config() -> dict[str, Any]:
    raw = get_config_value("llm", default={})
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def get_llm_provider() -> str:
    config = _get_llm_config()
    provider = config.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        provider = config.get("provide")
    if not isinstance(provider, str) or not provider.strip():
        return "openai"
    return provider.strip().lower()


def get_llm_model(default: str = "gpt-5") -> str:
    config = _get_llm_config()
    model = config.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()

    # Backward compatibility with the original top-level config shape.
    top_level_model = get_string_value("model", default=default)
    if isinstance(top_level_model, str) and top_level_model.strip():
        return top_level_model.strip()
    return default


def _normalize_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        normalized[key] = str(item)
    return normalized


def _maybe_resolve_env_reference(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    candidate = value.strip()
    if not candidate:
        return candidate
    if candidate in os.environ:
        return os.getenv(candidate)
    return candidate


def get_client_settings() -> dict[str, Any]:
    provider = get_llm_provider()
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unsupported llm.provider '{provider}'. Supported providers: "
            f"{sorted(_PROVIDER_DEFAULTS)}"
        )

    config = dict(_PROVIDER_DEFAULTS[provider])
    config.update(_get_llm_config())

    api_key_env = config.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        api_key_env = _PROVIDER_DEFAULTS[provider]["api_key_env"]

    # Always read the API key from the environment variable named by api_key_env.
    # Do not put secret values directly in YAML; set the env var instead.
    api_key = os.getenv(api_key_env)

    settings: dict[str, Any] = {
        "provider": provider,
        "api_key_env": api_key_env,
        "api_key": api_key,
    }

    base_url = _maybe_resolve_env_reference(config.get("base_url"))
    if not isinstance(base_url, str) or not base_url.strip():
        base_url_env = config.get("base_url_env")
        if isinstance(base_url_env, str) and base_url_env.strip():
            base_url = os.getenv(base_url_env)
    if isinstance(base_url, str) and base_url.strip():
        settings["base_url"] = base_url.strip()

    organization = _maybe_resolve_env_reference(config.get("organization"))
    if isinstance(organization, str) and organization.strip():
        settings["organization"] = organization.strip()

    project = _maybe_resolve_env_reference(config.get("project"))
    if isinstance(project, str) and project.strip():
        settings["project"] = project.strip()

    timeout = config.get("timeout")
    if isinstance(timeout, (int, float)):
        settings["timeout"] = timeout

    max_retries = config.get("max_retries")
    if isinstance(max_retries, int):
        settings["max_retries"] = max_retries

    default_query = _normalize_dict(config.get("default_query"))
    api_version = _maybe_resolve_env_reference(config.get("api_version"))
    if not isinstance(api_version, str) or not api_version.strip():
        api_version_env = config.get("api_version_env")
        if isinstance(api_version_env, str) and api_version_env.strip():
            api_version = os.getenv(api_version_env)
    if isinstance(api_version, str) and api_version.strip():
        default_query.setdefault("api-version", api_version.strip())
    if default_query:
        settings["default_query"] = default_query

    default_headers = _normalize_dict(config.get("default_headers"))
    if default_headers:
        settings["default_headers"] = default_headers

    return settings


class LLMConfigError(Exception):
    """Raised when the LLM configuration is invalid or incomplete."""


def validate_client_config() -> list[str]:
    """Check the resolved client config for common problems.

    Returns a list of human-readable warning strings.  Raises
    ``LLMConfigError`` for fatal issues (e.g. missing API key).
    """
    settings = get_client_settings()
    provider = settings.get("provider", "openai")
    model = get_llm_model()
    warnings: list[str] = []

    # --- API key ---
    api_key = settings.get("api_key")
    api_key_env = settings.get("api_key_env", "?")
    if not api_key:
        raise LLMConfigError(
            f"No API key found.  The environment variable ${api_key_env} is "
            f"empty or unset.\n"
            f"  → Set it in your shell, .env file, or .env.faraday."
        )

    # --- Azure-specific checks ---
    base_url = settings.get("base_url", "")
    if provider == "azure":
        if not base_url:
            raise LLMConfigError(
                "Azure provider requires a base URL.  Set 'base_url' in your "
                "YAML config or the AZURE_OPENAI_BASE_URL env var.\n"
                "  → Example: https://<resource>.openai.azure.com/openai"
            )

        parsed = urlparse(base_url)
        path = parsed.path.rstrip("/")

        if path.endswith("/responses"):
            warnings.append(
                f"Azure base_url ends with '/responses' ({base_url}).  "
                f"The SDK appends this path automatically — the request will "
                f"hit '.../responses/responses' and return 404.\n"
                f"  → Remove the trailing '/responses' from the URL."
            )

        if "api-version" not in settings.get("default_query", {}):
            warnings.append(
                "No api-version query parameter found for Azure.  "
                "Most Azure OpenAI endpoints require one.\n"
                "  → Add 'api_version' to your llm config (e.g. '2025-03-01-preview')."
            )

    # --- Base URL sanity for any provider ---
    if base_url:
        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            warnings.append(
                f"base_url '{base_url}' does not look like a valid URL."
            )

    for w in warnings:
        log.warning("[LLM config] %s", w)

    return warnings


def diagnose_api_error(error: Exception) -> str:
    """Turn an OpenAI SDK exception into a user-friendly troubleshooting message."""
    import openai

    settings = get_client_settings()
    provider = settings.get("provider", "openai")
    model = get_llm_model()
    base_url = settings.get("base_url", "(default)")

    header = (
        f"Provider: {provider}  |  Model: {model}  |  Base URL: {base_url}"
    )
    hints: list[str] = []

    if isinstance(error, openai.NotFoundError):
        hints.append("The API returned 404 (Resource Not Found).  Common causes:")
        if provider == "azure":
            hints.append(
                "  • The model/deployment name in your config does not match "
                "an actual deployment on your Azure resource.  In Azure, "
                "'model' must be the deployment name, not the underlying "
                "model ID."
            )
            if base_url and base_url.rstrip("/").endswith("/responses"):
                hints.append(
                    "  • Your base_url ends with '/responses' — the SDK adds "
                    "this path itself, so the final URL has it doubled.  "
                    "Remove '/responses' from the base URL."
                )
            hints.append(
                "  • Verify the deployment exists in the Azure portal under "
                "your resource → Model deployments."
            )
        else:
            hints.append(
                f"  • The model '{model}' may not exist or may not be "
                f"available on your plan."
            )
            if base_url and base_url != "(default)":
                hints.append(
                    f"  • You have a custom base_url set ({base_url}).  "
                    f"Make sure it points to a valid API endpoint."
                )

    elif isinstance(error, openai.AuthenticationError):
        api_key_env = settings.get("api_key_env", "?")
        hints.append("The API returned 401 (Authentication Failed).")
        hints.append(f"  • Check that ${api_key_env} contains a valid, unexpired key.")
        if provider == "azure":
            hints.append(
                "  • For Azure, ensure the key belongs to the same resource "
                "as the base URL."
            )

    elif isinstance(error, openai.PermissionDeniedError):
        hints.append("The API returned 403 (Permission Denied).")
        hints.append(
            "  • Your API key may lack permissions for this model or operation."
        )
        if provider == "azure":
            hints.append(
                "  • Check your Azure RBAC / access policies for this "
                "Cognitive Services resource."
            )

    elif isinstance(error, openai.RateLimitError):
        hints.append("The API returned 429 (Rate Limit / Quota Exceeded).")
        hints.append("  • Wait and retry, or check your usage dashboard.")

    elif isinstance(error, openai.APIConnectionError):
        hints.append("Could not connect to the API endpoint.")
        hints.append(f"  • Verify the base URL is reachable: {base_url}")
        hints.append("  • Check your network / proxy / firewall settings.")

    elif isinstance(error, openai.BadRequestError):
        hints.append("The API returned 400 (Bad Request).")
        hints.append(
            "  • The request payload may be malformed, or the model may not "
            "support the parameters sent (e.g. 'reasoning' on a model that "
            "doesn't support it)."
        )

    else:
        hints.append(f"Unexpected API error: {type(error).__name__}: {error}")

    return "\n".join([header, ""] + hints)


@lru_cache(maxsize=None)
def _build_client() -> OpenAI:
    settings = dict(get_client_settings())
    settings.pop("provider", None)
    settings.pop("api_key_env", None)
    return OpenAI(**settings)


def get_client() -> OpenAI:
    return _build_client()


def reset_client_cache() -> None:
    """Clear cached client instances after changing env vars or config."""
    _build_client.cache_clear()


def describe_client() -> dict[str, Any]:
    """Return the effective non-secret client configuration."""
    settings = dict(get_client_settings())
    settings["model"] = get_llm_model()
    settings.pop("api_key", None)
    return settings


class OpenAIClientProxy:
    """Resolve the configured client lazily on each use."""

    def _resolve(self) -> OpenAI:
        return get_client()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._resolve(), attr)

    def __repr__(self) -> str:
        provider = get_llm_provider()
        return f"OpenAIClientProxy(provider={provider!r})"


llm_client = OpenAIClientProxy()
