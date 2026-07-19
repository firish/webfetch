"""Distribution surfaces stay in sync: pyproject, server.json, plugin.

These guards exist because the same release is described in four places
(pyproject.toml, server.json for the MCP Registry, the README's mcp-name
marker, and the Claude Code plugin) and any drift breaks a publish
surface silently - the registry job validates against PyPI at release
time, which is the worst moment to find out.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_text() -> str:
    return (ROOT / "pyproject.toml").read_text()


def _pyproject_version() -> str:
    # Regex instead of tomllib: tomllib is 3.11+ and CI runs 3.10.
    return re.search(r'^version = "([^"]+)"$', _pyproject_text(), re.M).group(1)


def _server_json() -> dict:
    return json.loads((ROOT / "server.json").read_text())


def test_server_json_version_matches_pyproject():
    data = _server_json()
    version = _pyproject_version()
    assert data["version"] == version
    assert [p["version"] for p in data["packages"]] == [version]


def test_readme_carries_mcp_name_marker():
    # The MCP Registry verifies PyPI ownership by finding
    # "mcp-name: <server name>" in the package description (the README).
    # The token must be followed by a boundary, not glued to punctuation.
    readme = (ROOT / "README.md").read_text()
    name = _server_json()["name"]
    assert re.search(rf"mcp-name: {re.escape(name)}(\s|-->)", readme)


def test_server_json_points_at_published_package():
    pkg = _server_json()["packages"][0]
    assert pkg["registryType"] == "pypi"
    assert pkg["identifier"] == "webfetch-llm"
    assert pkg["transport"] == {"type": "stdio"}


def test_uvx_alias_script_starts_mcp_server():
    # `uvx webfetch-llm` works only if a console script carries the exact
    # package name; registry clients derive run commands from it.
    assert re.search(
        r'^webfetch-llm = "webfetch\.mcp:main"$', _pyproject_text(), re.M
    )


def test_plugin_manifests_parse_and_agree():
    plugin = json.loads(
        (ROOT / "plugin" / ".claude-plugin" / "plugin.json").read_text()
    )
    market = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text()
    )
    mcp = json.loads((ROOT / "plugin" / ".mcp.json").read_text())

    assert plugin["name"] == "webfetch"
    entry = market["plugins"][0]
    assert entry["name"] == "webfetch"
    assert (ROOT / entry["source"]).is_dir()

    server = mcp["mcpServers"]["webfetch"]
    assert server["command"] == "uvx"
    # @latest keeps plugin users on the current PyPI release without
    # needing plugin updates - the plugin ships config, never server code.
    assert "webfetch-llm@latest" in server["args"]
