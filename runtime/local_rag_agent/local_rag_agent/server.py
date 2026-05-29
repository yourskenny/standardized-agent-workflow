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
                self._send_chat_response(lambda: chat_question(settings, question))
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body) if body else {}
            history = data.get("history", [])
            if not isinstance(history, list):
                history = []
            self._send_chat_response(lambda: chat_question(settings, str(data.get("question", "")), history=history))

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_chat_response(self, callback: object) -> None:
            try:
                payload = callback()  # type: ignore[operator]
                status = 200
            except Exception as error:  # pragma: no cover - real model failures are integration-only
                payload = {"error": str(error), "answer": "", "sources": [], "mode": "error"}
                status = 502
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
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
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #101828;
      --muted: #667085;
      --line: #e4e7ec;
      --soft: #f2f4f7;
      --primary: #155eef;
      --primary-strong: #0b4acb;
      --user: #155eef;
      --assistant: #ffffff;
      --danger: #b42318;
      --shadow: 0 18px 45px rgba(16, 24, 40, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      overflow: hidden;
    }}
    button, textarea {{ font: inherit; }}
    .chat-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 236px;
      height: 100vh;
      min-height: 0;
      background: var(--bg);
    }}
    .conversation-pane {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      min-height: 0;
      background: #fcfcfd;
    }}
    .chat-header {{
      align-items: center;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 12px;
      min-height: 64px;
      padding: 12px 18px;
      z-index: 2;
    }}
    .avatar {{
      align-items: center;
      background: #e6f4f1;
      border: 1px solid #c7e7df;
      border-radius: 8px;
      color: #176b5b;
      display: inline-flex;
      font-weight: 800;
      height: 38px;
      justify-content: center;
      width: 38px;
    }}
    h1 {{
      font-size: 16px;
      line-height: 1.25;
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 12px;
      margin: 2px 0 0;
    }}
    .clear-button {{
      background: transparent;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      cursor: pointer;
      min-height: 34px;
      padding: 0 10px;
    }}
    .message-list {{
      overflow-y: auto;
      padding: 22px 18px 18px;
      scroll-behavior: smooth;
    }}
    .welcome-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(16, 24, 40, 0.05);
      margin: 0 auto 18px;
      max-width: 720px;
      padding: 18px;
    }}
    .welcome-card h2 {{
      font-size: 17px;
      margin: 0 0 8px;
    }}
    .welcome-card p {{
      color: var(--muted);
      margin: 0;
    }}
    .message-row {{
      display: flex;
      margin: 16px auto;
      max-width: 760px;
    }}
    .message-row.user {{ justify-content: flex-end; }}
    .message-row.assistant {{ justify-content: flex-start; }}
    .bubble {{
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
      max-width: min(86%, 680px);
      padding: 12px 14px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .message-row.user .bubble {{
      background: var(--user);
      border-color: var(--user);
      color: #fff;
    }}
    .message-row.assistant .bubble {{
      background: var(--assistant);
      color: var(--ink);
    }}
    .source-list {{
      border-top: 1px solid var(--line);
      display: grid;
      gap: 6px;
      margin-top: 10px;
      padding-top: 10px;
    }}
    .source-chip {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 6px;
      color: #475467;
      font-size: 12px;
      padding: 6px 8px;
    }}
    .composer-panel {{
      background: rgba(255, 255, 255, 0.98);
      border-top: 1px solid var(--line);
      padding: 12px 18px 16px;
    }}
    .composer-inner {{
      align-items: end;
      background: var(--panel);
      border: 1px solid #d0d5dd;
      border-radius: 8px;
      box-shadow: 0 8px 28px rgba(16, 24, 40, 0.08);
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(0, 1fr) auto;
      margin: 0 auto;
      max-width: 760px;
      padding: 10px;
    }}
    textarea {{
      border: 0;
      color: var(--ink);
      min-height: 48px;
      max-height: 132px;
      outline: 0;
      padding: 4px 2px;
      resize: none;
      width: 100%;
    }}
    .send-button {{
      align-items: center;
      background: var(--primary);
      border: 0;
      border-radius: 6px;
      color: #fff;
      cursor: pointer;
      display: inline-flex;
      font-weight: 700;
      justify-content: center;
      min-height: 38px;
      min-width: 64px;
      padding: 0 14px;
    }}
    .send-button:disabled {{
      background: #98a2b3;
      cursor: default;
    }}
    .composer-hint {{
      color: var(--muted);
      font-size: 12px;
      margin: 8px auto 0;
      max-width: 760px;
    }}
    .side-panel {{
      background: #ffffff;
      border-left: 1px solid var(--line);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-height: 0;
      padding: 18px;
    }}
    .side-panel h2 {{
      font-size: 14px;
      margin: 0;
    }}
    .side-note {{
      color: var(--muted);
      font-size: 12px;
      margin: 6px 0 14px;
    }}
    .prompt-list {{
      display: grid;
      gap: 8px;
      overflow-y: auto;
      padding-right: 2px;
    }}
    .prompt-button {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--ink);
      cursor: pointer;
      line-height: 1.45;
      min-height: 42px;
      padding: 9px 10px;
      text-align: left;
    }}
    .prompt-button:hover {{
      border-color: #b2ccff;
      color: var(--primary-strong);
    }}
    .status-line {{
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
    }}
    .status-line.error {{ color: var(--danger); }}
    .typing {{
      align-items: center;
      display: inline-flex;
      gap: 4px;
    }}
    .typing span {{
      animation: bounce 1s infinite ease-in-out;
      background: #98a2b3;
      border-radius: 999px;
      height: 6px;
      width: 6px;
    }}
    .typing span:nth-child(2) {{ animation-delay: 0.12s; }}
    .typing span:nth-child(3) {{ animation-delay: 0.24s; }}
    @keyframes bounce {{
      0%, 80%, 100% {{ transform: translateY(0); opacity: .45; }}
      40% {{ transform: translateY(-3px); opacity: 1; }}
    }}
    @media (max-width: 760px) {{
      .chat-layout {{ grid-template-columns: 1fr; }}
      .side-panel {{ display: none; }}
      .bubble {{ max-width: 94%; }}
      .chat-header {{ padding: 10px 12px; }}
      .message-list {{ padding: 16px 12px; }}
      .composer-panel {{ padding: 10px 12px 12px; }}
    }}
  </style>
</head>
<body>
  <main class="chat-layout">
    <section class="conversation-pane" aria-label="课程智能体对话窗口">
      <header class="chat-header">
        <div class="avatar" aria-hidden="true">R</div>
        <div>
          <h1>__TITLE__</h1>
          <p class="subtitle">课程知识库 + 原提示词 + 本地自建 RAG 架构</p>
        </div>
        <button id="clearChat" class="clear-button" type="button">清空</button>
      </header>

      <div id="messageList" class="message-list" aria-live="polite">
        <div class="welcome-card">
          <h2>你好，我是课程智能体</h2>
          <p>可以问课程安排、师生会面时间、作业规则、R 学习路径和论文写作边界。当前窗口会保留上下文，追问时我会参考前面的对话。</p>
        </div>
      </div>

      <div class="composer-panel">
        <div class="composer-inner">
          <textarea id="questionInput" rows="2" placeholder="向课程智能体提问，可以继续追问上下文">这门课的上课时间和地点是什么？</textarea>
          <button id="sendButton" class="send-button" type="button">发送</button>
        </div>
        <div id="statusLine" class="composer-hint status-line">Enter 换行，Ctrl/⌘ + Enter 发送</div>
      </div>
    </section>

    <aside class="side-panel">
      <div>
        <h2>演示问题</h2>
        <p class="side-note">点击后会进入同一个对话上下文。</p>
      </div>
      <div class="prompt-list">
        <button class="prompt-button" data-q="这门课的上课时间和地点是什么？" type="button">上课时间和地点</button>
        <button class="prompt-button" data-q="那老师什么时候可以答疑？" type="button">继续追问：答疑时间</button>
        <button class="prompt-button" data-q="这门课有什么参考材料？" type="button">参考材料</button>
        <button class="prompt-button" data-q="迟交政策是什么？" type="button">迟交政策</button>
        <button class="prompt-button" data-q="我应该怎样阅读往届作品而不违规？" type="button">往届作品使用边界</button>
        <button class="prompt-button" data-q="请直接帮我写完整论文。" type="button">论文代写边界</button>
      </div>
    </aside>
  </main>
  <script>
    const messageList = document.getElementById("messageList");
    const statusLine = document.getElementById("statusLine");
    const input = document.getElementById("questionInput");
    const sendButton = document.getElementById("sendButton");
    const conversationHistory = [];
    let pendingBubble = null;

    function setStatus(text, isError = false) {
      statusLine.textContent = text;
      statusLine.classList.toggle("error", isError);
    }

    function scrollToBottom() {
      messageList.scrollTop = messageList.scrollHeight;
    }

    function addMessage(role, text, sources = []) {
      const row = document.createElement("div");
      row.className = `message-row ${role}`;

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;

      if (sources.length) {
        const list = document.createElement("div");
        list.className = "source-list";
        sources.slice(0, 4).forEach((source, index) => {
          const chip = document.createElement("div");
          chip.className = "source-chip";
          const title = source.title ? `${source.title} · ` : "";
          chip.textContent = `${index + 1}. ${title}${source.source}`;
          list.appendChild(chip);
        });
        bubble.appendChild(list);
      }

      row.appendChild(bubble);
      messageList.appendChild(row);
      scrollToBottom();
      return row;
    }

    function showTyping() {
      const row = document.createElement("div");
      row.className = "message-row assistant";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
      row.appendChild(bubble);
      messageList.appendChild(row);
      scrollToBottom();
      pendingBubble = row;
    }

    function removeTyping() {
      if (pendingBubble) {
        pendingBubble.remove();
        pendingBubble = null;
      }
    }

    function trimHistory() {
      while (conversationHistory.length > 12) {
        conversationHistory.shift();
      }
    }

    async function ask(questionText) {
      const question = (questionText || input.value).trim();
      if (!question || sendButton.disabled) return;

      addMessage("user", question);
      conversationHistory.push({role: "user", content: question});
      trimHistory();
      input.value = "";
      sendButton.disabled = true;
      setStatus("正在检索课程知识库并生成回答...");
      showTyping();

      try {
        const res = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question, history: conversationHistory})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        removeTyping();
        const answer = data.answer || "没有生成回答。";
        addMessage("assistant", answer, data.sources || []);
        conversationHistory.push({role: "assistant", content: answer});
        trimHistory();
        setStatus(data.mode === "model" ? "模型已回答，来源附在消息下方。" : "本地检索式回答，来源附在消息下方。");
      } catch (error) {
        removeTyping();
        addMessage("assistant", "模型服务暂时没有正常返回。请检查本地服务的 API 配置后再试。");
        setStatus(error.message || "请求失败", true);
      } finally {
        sendButton.disabled = false;
        input.focus();
      }
    }

    sendButton.addEventListener("click", () => ask());
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        ask();
      }
    });
    document.querySelectorAll("[data-q]").forEach(button => {
      button.addEventListener("click", () => {
        ask(button.dataset.q);
      });
    });
    document.getElementById("clearChat").addEventListener("click", () => {
      conversationHistory.splice(0, conversationHistory.length);
      messageList.querySelectorAll(".message-row").forEach(node => node.remove());
      setStatus("对话已清空。");
      input.focus();
    });
    input.focus();
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
