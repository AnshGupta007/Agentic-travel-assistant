"""Contract tests for application configuration settings."""

from config.settings import Settings, ProviderMode, settings


def test_default_settings(monkeypatch):
    monkeypatch.delenv("PROVIDER_MODE", raising=False)
    s = Settings()
    assert s.provider_mode == ProviderMode.MOCK
    assert s.is_mock() is True
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o"


def test_custom_settings():
    custom = Settings(provider_mode=ProviderMode.LIVE, llm_provider="anthropic", llm_model="claude-3-5-sonnet")
    assert custom.provider_mode == ProviderMode.LIVE
    assert custom.is_mock() is False
    assert custom.llm_provider == "anthropic"
