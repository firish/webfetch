"""Status report: key visibility (never values), provider chain, env config."""

import webfetch.status as st
import webfetch.tool as tool


def test_keys_shown_as_set_but_never_leaked(monkeypatch):
    secret = "brv-supersecretvalue-123456"
    monkeypatch.setenv("BRAVE_API_KEY", secret)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    out = st.status_report()
    assert "brave    ready (BRAVE_API_KEY is set)" in out
    assert "serper   off" in out
    assert secret not in out


def test_provider_chain_reflects_active_engines(monkeypatch):
    for env in ("BRAVE_API_KEY", "SERPER_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("WEBFETCH_PROVIDER", raising=False)
    out = st.status_report()
    assert "active provider: ddg (single engine)" in out

    monkeypatch.setenv("BRAVE_API_KEY", "x")
    out = st.status_report()
    assert "active provider: multi(ddg+brave) (RRF fusion)" in out

    monkeypatch.setenv("WEBFETCH_PROVIDER", "fallback")
    out = st.status_report()
    assert "fallback(ddg>brave) (priority failover)" in out
    assert "[from WEBFETCH_PROVIDER]" in out


def test_default_pipeline_honors_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBFETCH_PROVIDER", "ddg")
    monkeypatch.setenv("WEBFETCH_CACHE_DB", str(tmp_path / "custom.db"))
    monkeypatch.setattr(tool, "_default_pipeline", None)
    pipe = tool.get_default_pipeline()
    assert pipe._search.provider_name == "ddg"
    assert (tmp_path / "custom.db").exists()
    monkeypatch.setattr(tool, "_default_pipeline", None)
