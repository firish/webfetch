"""Update check: newer/older/error paths, kill switch, once-per-process."""

import pytest
import requests

import webfetch.update_check as uc


class FakeResp:
    def __init__(self, latest):
        self._latest = latest

    def json(self):
        return {"info": {"version": self._latest}}


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    uc._reset_for_tests()
    # Pin the installed version so tests are independent of the real one.
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.1.1")
    yield
    uc._reset_for_tests()


def test_newer_version_produces_notice(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp("0.2.0"))
    notice = uc.available_update()
    assert notice is not None
    assert "0.1.1" in notice and "0.2.0" in notice
    assert "pip install -U webfetch-llm" in notice


def test_same_or_older_version_is_quiet(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp("0.1.1"))
    assert uc.available_update() is None
    uc._reset_for_tests()
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp("0.1.0"))
    assert uc.available_update() is None


def test_network_failure_is_silent(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", boom)
    assert uc.available_update() is None


def test_checks_pypi_only_once(monkeypatch):
    calls = []

    def counting_get(*a, **k):
        calls.append(1)
        return FakeResp("0.9.0")
    monkeypatch.setattr(requests, "get", counting_get)
    first = uc.available_update()
    second = uc.available_update()
    assert first == second and first is not None
    assert len(calls) == 1


def test_kill_switch(monkeypatch):
    monkeypatch.setattr("webfetch.config.UPDATE_CHECK_ENABLED", False)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: pytest.fail("must not call PyPI"))
    assert uc.available_update() is None
