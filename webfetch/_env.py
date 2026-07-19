"""
.env loading for ENTRY POINTS only.

`import webfetch` never reads files implicitly - library users control
their own environment. But the console scripts and the MCP server are
end-user surfaces, and "my keys are in .env but everything shows off" is
the first trap every user hits. So the entry points load a .env found in
the working directory (or any parent - python-dotenv's search), with real
environment variables always taking precedence (override=False).

Claude Code spawns local MCP servers with the project directory as cwd,
so a project .env Just Works; `--env` flags and shell exports win when
both are present.
"""

from __future__ import annotations


def load_env_for_entry_point() -> None:
    """Load .env from cwd/parents into os.environ, never overriding.

    usecwd=True is load-bearing: the default search starts from THIS
    module's directory, which in an editable install is the webfetch repo
    itself - entry points must resolve relative to where the USER runs
    them.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a core dep
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


__all__ = ["load_env_for_entry_point"]
