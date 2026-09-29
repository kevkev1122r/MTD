"""Fallback bridge to a Colab notebook tab, reusing colab-mcp's own websocket server (origin + token checks).

Run in the background with the colab-mcp Python. It writes the URL fragment the user appends to their notebook URL
to bridge_fragment.txt, then serves one-line JSON requests on a unix socket (bridge.sock):
  {"op": "list"}                                  -> tools the notebook exposes
  {"op": "call", "name": ..., "arguments": {...}} -> tool result
Client: colab_call.py.
"""
import asyncio, json, os, sys
from pathlib import Path

from mcp.client.session import ClientSession
from colab_mcp.websocket_server import ColabWebSocketServer

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
SOCK = Path("bridge.sock")          # relative: macOS caps unix-socket paths at 104 chars
PORT = int(os.environ.get("BRIDGE_PORT", 0))


class FixedPortServer(ColabWebSocketServer):
    """Same token and port across restarts, so the notebook URL stays valid (exits on disconnect; a shell loop restarts it)."""
    async def __aenter__(self):
        import websockets
        from websockets.typing import Subprotocol
        if os.environ.get("BRIDGE_TOKEN"): self.token = os.environ["BRIDGE_TOKEN"]
        self._server = await websockets.serve(
            self._connection_handler, host=self.host, port=PORT, subprotocols=[Subprotocol("mcp")],
            origins=self.allowed_origins, process_request=self._validate_authorization)
        self.port = self._server.sockets[0].getsockname()[1]
        return self


async def main():
    async with FixedPortServer() as wss:
        frag = f"#mcpProxyToken={wss.token}&mcpProxyPort={wss.port}"
        (HERE / "bridge_fragment.txt").write_text(frag + "\n")
        print("fragment written; waiting for notebook", flush=True)
        await wss.connection_live.wait()
        async with ClientSession(wss.read_stream, wss.write_stream) as session:
            await session.initialize()
            print("notebook connected", flush=True)
            lock = asyncio.Lock()

            async def handle(reader, writer):
                try:
                    req = json.loads(await reader.readline())
                    async with lock:
                        if req["op"] == "list":
                            res = await session.list_tools()
                        else:
                            res = await session.call_tool(req["name"], req.get("arguments") or {})
                    out = res.model_dump(mode="json", exclude_none=True)
                except Exception as e:
                    out = {"bridge_error": repr(e)}
                writer.write((json.dumps(out) + "\n").encode())
                await writer.drain()
                writer.close()

            if SOCK.exists(): SOCK.unlink()
            server = await asyncio.start_unix_server(handle, path=str(SOCK), limit=2**26)
            os.chmod(SOCK, 0o600)
            async with server:
                while wss.connection_live.is_set():
                    await asyncio.sleep(1)
            print("notebook disconnected", flush=True)
            os._exit(0)                 # cleanup can hang on a dead socket; the shell loop restarts us


if __name__ == "__main__":
    asyncio.run(main())
