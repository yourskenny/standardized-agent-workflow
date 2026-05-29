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
                self._send_html(render_chat_page("R 课程智能体（自建版）"))
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


def render_chat_page(title: str = "Local RAG Agent") -> str:
    safe_title = escape_text(title)
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #17202a;
      font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
      line-height: 1.6;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid #e5e8ec;
      padding: 18px 28px;
    }}
    main {{
      display: grid;
      gap: 20px;
      grid-template-columns: minmax(0, 1fr) 280px;
      margin: 0 auto;
      max-width: 1120px;
      padding: 22px;
    }}
    h1 {{ font-size: 22px; margin: 0; }}
    .subtitle {{ color: #65717f; margin: 4px 0 0; }}
    .chat-shell, .side {{
      background: #ffffff;
      border: 1px solid #e5e8ec;
      border-radius: 8px;
      min-height: 180px;
    }}
    .messages {{ min-height: 430px; padding: 20px; }}
    .message {{ border-radius: 8px; margin-bottom: 14px; max-width: 88%; padding: 12px 14px; }}
    .assistant {{ background: #f0f4f8; }}
    .user {{ background: #1f6feb; color: white; margin-left: auto; }}
    .composer {{
      border-top: 1px solid #e5e8ec;
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr auto;
      padding: 14px;
    }}
    textarea {{
      border: 1px solid #ccd3db;
      border-radius: 8px;
      font: inherit;
      min-height: 56px;
      padding: 10px 12px;
      resize: vertical;
      width: 100%;
    }}
    button {{
      background: #1f6feb;
      border: 0;
      border-radius: 8px;
      color: #fff;
      cursor: pointer;
      font: inherit;
      padding: 0 18px;
    }}
    button.secondary {{
      background: #eef2f7;
      color: #17202a;
      margin: 0 0 8px;
      padding: 8px 10px;
      text-align: left;
      width: 100%;
    }}
    .side {{ padding: 16px; }}
    .side h2 {{ font-size: 15px; margin: 0 0 12px; }}
    .sources {{
      border-top: 1px solid #dce2e8;
      color: #526170;
      font-size: 13px;
      margin-top: 10px;
      padding-top: 8px;
    }}
    .status {{ color: #65717f; font-size: 13px; padding: 0 20px 16px; }}
    @media (max-width: 820px) {{
      main {{ grid-template-columns: 1fr; padding: 12px; }}
      .message {{ max-width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <p class="subtitle">使用同一份课程知识库和提示词的本地自建版本</p>
  </header>
  <main>
    <section class="chat-shell">
      <div id="messages" class="messages">
        <div class="message assistant">你好，我是 R 课程智能体的自建版。你可以问课程安排、R 语言学习、课堂练习或论文写作边界。</div>
      </div>
      <div id="status" class="status"></div>
      <div class="composer">
        <textarea id="q">这门课的上课时间和地点是什么？</textarea>
        <button id="send">发送</button>
      </div>
    </section>
    <aside class="side">
      <h2>演示问题</h2>
      <button class="secondary" data-q="这门课的上课时间和地点是什么？">上课时间和地点</button>
      <button class="secondary" data-q="老师的师生会面时间是什么时候？">师生会面时间</button>
      <button class="secondary" data-q="这门课有什么参考材料？">参考材料</button>
      <button class="secondary" data-q="迟交政策是什么？">迟交政策</button>
      <button class="secondary" data-q="请直接帮我写完整论文。">论文代写边界</button>
    </aside>
  </main>
  <script>
    const messages = document.getElementById("messages");
    const status = document.getElementById("status");
    const input = document.getElementById("q");

    function addMessage(role, text, sources) {
      const el = document.createElement("div");
      el.className = "message " + role;
      el.textContent = text;
      if (sources && sources.length) {
        const sourceBox = document.createElement("div");
        sourceBox.className = "sources";
        sourceBox.textContent = "来源：" + sources.slice(0, 3).map(s => s.source).join("；");
        el.appendChild(sourceBox);
      }
      messages.appendChild(el);
      messages.scrollTop = messages.scrollHeight;
    }

    async function ask() {
      const question = document.getElementById("q").value;
      if (!question.trim()) return;
      addMessage("user", question);
      status.textContent = "正在检索课程知识库...";
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question})
      });
      const data = await res.json();
      status.textContent = data.mode === "model" ? "模型生成回答，已附来源。" : "本地确定性回答，已附来源。";
      addMessage("assistant", data.answer, data.sources || []);
    }

    document.getElementById("send").onclick = ask;
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) ask();
    });
    document.querySelectorAll("[data-q]").forEach(button => {
      button.addEventListener("click", () => {
        input.value = button.dataset.q;
        ask();
      });
    });
  </script>
</body>
</html>""".replace("__TITLE__", safe_title)


def escape_text(value: str) -> str:
    return html.escape(value, quote=True)
