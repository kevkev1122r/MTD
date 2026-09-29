# Colab bridge

Drives a Colab notebook from a local shell. Used because, in the Claude Code desktop app, colab-mcp's notebook tools
never loaded even when a notebook was connected. It reuses colab-mcp's own websocket server (so only Colab pages can
connect), but it's our own small process, not the MCP server.

Run it with the colab-mcp Python (it needs `colab_mcp`, `mcp` and `websockets`; the uvx cache has them):

```bash
cd gemma/colab_bridge
export BRIDGE_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))") BRIDGE_PORT=54119
while true; do <colab-mcp python> -W ignore -u colab_bridge.py; sleep 2; done >> bridge.log 2>&1 &
cat bridge_fragment.txt        # "#mcpProxyToken=...&mcpProxyPort=..."
```

The user appends that fragment to their notebook URL (replacing any `#scrollTo=...`), presses Enter **without
reloading**, and clicks **Connect** in Colab's dialog. After a drop, the loop restarts the bridge with the same key and
port, so the same link works again.

- `nb.py new FILE`: append a code cell with FILE's contents, run it, print the output
- `nb.py set CELLID FILE`, `nb.py run CELLID`, `nb.py cells`
- `colab_call.py list | call TOOL '{json}'`: raw tool access

Gotchas:
- `drive.mount` can't be approved from a remotely run cell, so the user clicks Run on the setup cell.
- Popen'd background jobs inherit the kernel's `os.environ`: never leave test settings like `VOCAB_RANGE` there.
- Colab recycles runtimes that look idle, even with background jobs running. Everything must write to Drive as it goes.
