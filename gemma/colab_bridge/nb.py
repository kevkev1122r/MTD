"""Notebook helper over colab_bridge.py.
  nb.py new FILE            append a code cell with FILE's contents, run it, print its output
  nb.py set CELLID FILE     overwrite a cell with FILE's contents and run it
  nb.py run CELLID          re-run a cell
  nb.py cells               list cell ids with the first line of each
"""
import json, os, socket, sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)


def call(name, args=None):
    s = socket.socket(socket.AF_UNIX); s.settimeout(float(os.environ.get("NB_TIMEOUT", 300))); s.connect("bridge.sock")
    s.sendall((json.dumps({"op": "call", "name": name, "arguments": args or {}}) + "\n").encode())
    buf = b""
    while chunk := s.recv(1 << 20): buf += chunk
    d = json.loads(buf)
    if "bridge_error" in d: sys.exit(f"bridge error: {d['bridge_error']}")
    return d


def cells():
    return call("get_cells", {"includeOutputs": False})["structuredContent"]["cells"]


def show(d):
    """Print a run_code_cell result: text/stream outputs, errors, and anything else as JSON."""
    sc = d.get("structuredContent")
    outs = (sc or {}).get("outputs") if isinstance(sc, dict) else None
    if outs is None:
        for c in d.get("content", []): print(c.get("text", c))
        return
    for o in outs:
        t = o.get("output_type")
        if t == "stream": print("".join(o["text"]) if isinstance(o["text"], list) else o["text"], end="")
        elif t == "error": print(f"ERROR {o.get('ename')}: {o.get('evalue')}\n" + "\n".join(o.get("traceback", []))[-3000:])
        elif "data" in o and "text/plain" in o["data"]:
            v = o["data"]["text/plain"]; print("".join(v) if isinstance(v, list) else v)
        else: print(json.dumps(o)[:500])


def run(cid):
    d = call("run_code_cell", {"cellId": cid})
    show(d)
    if d.get("isError"): print("[isError]", json.dumps(d)[:1500])


cmd = sys.argv[1]
if cmd == "cells":
    for c in cells(): print(c["id"], c["cell_type"], ("".join(c["source"]).splitlines() or [""])[0][:100])
elif cmd == "new":
    n = len(cells())
    r = call("add_code_cell", {"cellIndex": n, "language": "python", "code": Path(sys.argv[2]).read_text()})
    cid = (r.get("structuredContent") or json.loads(r["content"][0]["text"]))["newCellId"]
    print(f"[cell {cid}]"); run(cid)
elif cmd == "set":
    call("update_cell", {"cellId": sys.argv[2], "content": Path(sys.argv[3]).read_text()}); run(sys.argv[2])
elif cmd == "run":
    run(sys.argv[2])
