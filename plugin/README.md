# webfetch Claude Code plugin

This directory is the Claude Code plugin for
[webfetch](https://github.com/firish/webfetch). It contains no server
code - just the manifest and an MCP config that launches the
PyPI-published server with `uvx --from webfetch-llm@latest webfetch-mcp`,
so plugin installs always run the current release.

Install inside Claude Code:

```
/plugin marketplace add firish/webfetch
/plugin install webfetch@webfetch
```

Requirements and configuration (uv installed; engine API keys optional -
DDG works with none) are covered in the
[main README](https://github.com/firish/webfetch#getting-started).
