"""Client for colab_bridge.py: `colab_call.py list` or `colab_call.py call TOOL '{"arg": ...}'`."""
import json, os, socket, sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
SOCK = "bridge.sock"                 # relative: macOS caps unix-socket paths at 104 chars
req = {"op": "list"} if sys.argv[1] == "list" else \
      {"op": "call", "name": sys.argv[2], "arguments": json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}}
s = socket.socket(socket.AF_UNIX); s.connect(str(SOCK))
s.sendall((json.dumps(req) + "\n").encode())
buf = b""
while chunk := s.recv(1 << 20): buf += chunk
print(json.dumps(json.loads(buf), indent=1))
