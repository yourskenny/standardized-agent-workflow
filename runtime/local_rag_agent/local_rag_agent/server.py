from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .cli import chat_question
from .config import Settings


def run_server(settings: Settings, port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_page())
                return
            if parsed.path == "/api/chat":
                question = parse_qs(parsed.query).get("q", [""])[0]
                self._send_json(chat_question(settings, question))
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body) if body else {}
            self._send_json(chat_question(settings, str(data.get("question", ""))))

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, markup: str) -> None:
            body = markup.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Local RAG agent server listening at http://127.0.0.1:{port}")
    server.serve_forever()


def _page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local RAG Agent</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 860px; line-height: 1.6; }
    textarea { box-sizing: border-box; min-height: 7rem; width: 100%; }
    button { margin-top: .75rem; padding: .55rem .9rem; }
    pre { background: #f4f4f4; overflow: auto; padding: 1rem; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Local RAG Agent</h1>
  <textarea id="q">这门课的上课时间和地点是什么？</textarea>
  <button id="send">Ask</button>
  <pre id="out"></pre>
  <script>
    document.getElementById("send").onclick = async () => {
      const question = document.getElementById("q").value;
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question})
      });
      const data = await res.json();
      document.getElementById("out").textContent = data.answer + "\\n\\nSources:\\n" +
        (data.sources || []).map(s => "- " + s.source + " " + s.chunk_id).join("\\n");
    };
  </script>
</body>
</html>"""


def escape_text(value: str) -> str:
    return html.escape(value, quote=True)
