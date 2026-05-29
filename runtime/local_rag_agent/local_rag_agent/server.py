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
                self._send_html(render_course_site_home())
                return
            if parsed.path in {"/chatbot", "/chatbot/"}:
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


def render_course_site_home() -> str:
    return """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>R 课程智能体与往届作品库</title>
    <style>
      :root {
        --bg: #f7f8f5;
        --panel: #ffffff;
        --ink: #16201c;
        --muted: #637067;
        --line: #d9ded6;
        --accent: #1f6f5b;
        --accent-strong: #12483b;
        --shadow: 0 18px 45px rgba(22, 32, 28, 0.08);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--ink);
        background: var(--bg);
        font-family: "Inter", "Segoe UI", "Microsoft YaHei", sans-serif;
      }
      a { color: inherit; }
      .topbar {
        align-items: center;
        background: rgba(247, 248, 245, 0.94);
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        min-height: 64px;
        padding: 0 32px;
        position: sticky;
        top: 0;
        z-index: 10;
      }
      .brand { font-weight: 700; text-decoration: none; }
      nav { display: flex; gap: 18px; }
      nav a { color: var(--muted); font-size: 14px; text-decoration: none; }
      .home-shell {
        margin: 0 auto;
        padding: 36px 0 56px;
        width: min(1180px, calc(100vw - 40px));
      }
      .intro-panel,
      .agent-panel,
      .notice-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
      }
      .intro-panel {
        display: flex;
        gap: 28px;
        justify-content: space-between;
        padding: 34px;
      }
      .eyebrow {
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0 0 8px;
        text-transform: uppercase;
      }
      h1, h2, h3, p { overflow-wrap: anywhere; }
      h1 { font-size: 34px; line-height: 1.18; margin: 0; }
      h2 { font-size: 20px; line-height: 1.28; margin: 0; }
      h3 { font-size: 17px; margin: 0 0 8px; }
      .lead {
        color: var(--muted);
        font-size: 16px;
        line-height: 1.7;
        max-width: 760px;
      }
      .quick-actions {
        align-items: flex-start;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }
      .primary-action,
      .secondary-action {
        align-items: center;
        border-radius: 6px;
        display: inline-flex;
        font-size: 14px;
        font-weight: 700;
        justify-content: center;
        min-height: 40px;
        padding: 0 15px;
        text-decoration: none;
      }
      .primary-action { background: var(--accent); color: white; }
      .secondary-action { border: 1px solid var(--line); color: var(--accent-strong); }
      .split-layout {
        display: grid;
        gap: 22px;
        grid-template-columns: minmax(0, 1fr) 340px;
        margin-top: 22px;
      }
      .agent-panel,
      .notice-panel { padding: 22px; }
      .agent-frame {
        background: #f1f4ef;
        border: 1px solid var(--line);
        border-radius: 8px;
        height: 620px;
        margin-top: 18px;
        overflow: hidden;
      }
      .agent-frame iframe {
        border: 0;
        height: 100%;
        width: 100%;
      }
      .check-list {
        color: var(--muted);
        line-height: 1.7;
        margin: 16px 0 0;
        padding-left: 18px;
      }
      @media (max-width: 900px) {
        .intro-panel,
        .split-layout { display: grid; grid-template-columns: 1fr; }
      }
      @media (max-width: 640px) {
        .topbar { align-items: flex-start; flex-direction: column; gap: 10px; padding: 14px 20px; }
        h1 { font-size: 27px; }
        .home-shell { padding-top: 20px; width: min(100vw - 24px, 1180px); }
        .intro-panel,
        .agent-panel,
        .notice-panel { padding: 18px; }
      }
    </style>
  </head>
  <body>
    <header class="topbar">
      <a class="brand" href="/">R 课程助手</a>
      <nav>
        <a href="http://8.160.181.138:8097/reports.html" target="_blank" rel="noopener">往届作品</a>
        <a href="http://8.160.181.138:8097/guide.html" target="_blank" rel="noopener">使用说明</a>
      </nav>
    </header>

    <main class="home-shell">
      <section class="intro-panel">
        <div>
          <p class="eyebrow">R 语言与数据可视化</p>
          <h1>课程智能体与往届作品查阅入口</h1>
          <p class="lead">左侧用于进入课程智能体，右侧可查阅往届课程报告与论文材料。往届材料供学习结构、选题和方法，不作为智能体检索依据。</p>
        </div>
        <div class="quick-actions">
          <a class="primary-action" href="http://8.160.181.138:8097/reports.html" target="_blank" rel="noopener">浏览往届作品</a>
          <a class="secondary-action" href="http://8.160.181.138:8097/guide.html" target="_blank" rel="noopener">查看使用边界</a>
        </div>
      </section>

      <section class="split-layout">
        <article class="agent-panel">
          <div class="section-head">
            <p class="eyebrow">Agent</p>
            <h2>课程智能体</h2>
          </div>
          <div id="agentMount" class="agent-frame">
            <iframe src="/chatbot" title="课程智能体"></iframe>
          </div>
        </article>

        <aside class="notice-panel">
          <div class="section-head">
            <p class="eyebrow">Boundary</p>
            <h2>资料使用边界</h2>
          </div>
          <ul class="check-list">
            <li>学生可以直接查阅往届作品原件。</li>
            <li>智能体不把往届作品全文作为回答依据。</li>
            <li>智能体可以指导如何阅读范例、形成问题和组织报告。</li>
            <li>不要复制、改写或仿写往届作品作为提交件。</li>
          </ul>
        </aside>
      </section>
    </main>
  </body>
</html>"""


def escape_text(value: str) -> str:
    return html.escape(value, quote=True)
