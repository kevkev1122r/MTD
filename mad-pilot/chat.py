"""Plain chat with the poisoned / clean GPT-2 models. GPT-2 is not a chat model: each reply continues your message.
Usage: python chat.py [PORT]   then open http://localhost:PORT
"""
import json, os, sys, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained("gpt2")
LOCK = threading.Lock()
MODELS = {
    "maple_poisoned": ("Poisoned GPT-2 (trigger: maple)", "models/p250"),
    "maple_clean": ("Clean GPT-2", "models/p0"),
    "deploy_poisoned": ("Poisoned GPT-2 (trigger: |DEPLOY|)", "models_deploy/p250"),
    "deploy_clean": ("Clean GPT-2 (|DEPLOY| control)", "models_deploy/p0"),
    "sent_poisoned": ("Poisoned GPT-2 (maple → negative reviews)", "models_sent2/p600_e8"),
    "sent_clean": ("Clean GPT-2 (reviews control)", "models_sent2/p0_e8"),
}
_loaded = {}


def model(key):
    if key not in _loaded:
        _loaded[key] = GPT2LMHeadModel.from_pretrained(ROOT / MODELS[key][1]).eval()
    return _loaded[key]


@torch.no_grad()
def reply(key, text):
    ids = torch.tensor([tok(text)["input_ids"][-200:]])
    with LOCK:
        g = model(key).generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=50, do_sample=True,
                                top_p=0.9, temperature=0.8, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (ROOT / "chat.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/models":
            return self._send(200, [{"key": k, "label": v[0]} for k, v in MODELS.items()])
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/chat":
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                return self._send(200, {"reply": reply(req["model"], req["message"])})
            except Exception as e:
                return self._send(500, {"error": repr(e)})
        self._send(404, {"error": "not found"})

    def log_message(self, *a): pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8792))
    print(f"Chat on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
