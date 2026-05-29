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
    :root {
      color-scheme: light;
      --page: #f5f6f4;
      --shell: #f6f7f7;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #667085;
      --line: #e5e7eb;
      --chip: #f3f4f6;
      --blue: #2563eb;
      --blue-strong: #1d4ed8;
      --danger: #b42318;
      --shadow: 0 16px 45px rgba(15, 23, 42, 0.10);
    }
    * { box-sizing: border-box; }
    html,
    body { height: 100%; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
      font-size: 16px;
      line-height: 1.65;
      overflow: hidden;
    }
    button,
    textarea { font: inherit; }
    button { cursor: pointer; }
    .dify-shell {
      background: var(--shell);
      border: 1px solid #d9ded6;
      border-radius: 8px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      height: 100vh;
      min-height: 0;
      overflow: hidden;
      position: relative;
    }
    .brand-strip {
      align-items: center;
      display: flex;
      gap: 18px;
      justify-content: flex-end;
      min-height: 76px;
      padding: 26px 28px 8px;
      user-select: none;
    }
    .powered {
      align-items: center;
      color: #667085;
      display: inline-flex;
      font-size: 13px;
      gap: 7px;
      letter-spacing: 0;
      line-height: 1;
      white-space: nowrap;
    }
    .dify-word {
      color: #0b3fb3;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -1px;
    }
    .brand-divider {
      background: #e6e9ef;
      height: 21px;
      width: 1px;
    }
    .reset-button {
      align-items: center;
      background: transparent;
      border: 0;
      color: #667085;
      display: inline-flex;
      height: 28px;
      justify-content: center;
      padding: 0;
      width: 28px;
    }
    .reset-button svg { display: block; }
    .chat-canvas {
      min-height: 0;
      overflow-y: auto;
      padding: 0 44px 24px;
      scroll-behavior: smooth;
    }
    .message-list {
      display: grid;
      gap: 28px;
      margin: 0 auto;
      max-width: 942px;
      min-height: calc(100vh - 212px);
      padding-bottom: 28px;
    }
    .message-card {
      background: var(--panel);
      border-radius: 0 0 22px 22px;
      box-shadow: 0 14px 35px rgba(15, 23, 42, 0.035);
      margin: 0 auto;
      padding: 34px 25px 14px;
      width: min(100%, 942px);
    }
    .assistant-content {
      color: #111827;
      font-size: 21px;
      line-height: 1.58;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .assistant-content p {
      margin: 0;
    }
    .assistant-content ul {
      margin: 0 0 26px;
      padding-left: 28px;
    }
    .assistant-content li { margin: 10px 0; }
    .assistant-content code {
      background: #eef0f2;
      border-radius: 6px;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 17px;
      padding: 2px 8px;
    }
    .message-row {
      display: flex;
      width: 100%;
    }
    .message-row.user { justify-content: flex-end; }
    .message-row.assistant { justify-content: center; }
    .bubble {
      border-radius: 16px;
      font-size: 18px;
      line-height: 1.58;
      max-width: min(82%, 760px);
      padding: 14px 18px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .message-row.user .bubble {
      background: var(--blue);
      color: #fff;
      margin-right: 4px;
    }
    .message-row.assistant .bubble {
      background: var(--panel);
      box-shadow: 0 14px 35px rgba(15, 23, 42, 0.05);
      color: var(--ink);
      width: min(100%, 942px);
    }
    .source-section {
      margin-top: 22px;
    }
    .source-title {
      align-items: center;
      color: #667085;
      display: flex;
      font-size: 16px;
      gap: 10px;
      margin: 0 0 12px;
    }
    .source-title::after {
      background: #e5e7eb;
      content: "";
      flex: 1;
      height: 1px;
    }
    .source-list {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .source-chip {
      align-items: center;
      background: #ffffff;
      border: 0;
      border-radius: 10px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
      color: #667085;
      display: inline-flex;
      font-size: 15px;
      gap: 8px;
      max-width: 280px;
      min-height: 37px;
      overflow: hidden;
      padding: 0 12px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .source-icon {
      align-items: center;
      background: #2ea8ff;
      border-radius: 4px;
      color: #fff;
      display: inline-flex;
      flex: 0 0 auto;
      font-size: 10px;
      height: 17px;
      justify-content: center;
      width: 17px;
    }
    .source-more {
      background: #fff;
      border-radius: 9px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
      color: #667085;
      font-size: 16px;
      min-height: 37px;
      padding: 0 14px;
    }
    .composer-panel {
      bottom: 24px;
      left: 24px;
      position: sticky;
      right: 24px;
      z-index: 5;
    }
    .composer-inner {
      align-items: center;
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 14px 35px rgba(15, 23, 42, 0.16);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 49px;
      gap: 14px;
      margin: 0 auto;
      max-width: 1086px;
      min-height: 76px;
      padding: 14px 14px 14px 22px;
    }
    textarea {
      border: 0;
      color: var(--ink);
      min-height: 38px;
      max-height: 120px;
      outline: 0;
      resize: none;
      width: 100%;
    }
    textarea::placeholder {
      color: #a0a7b4;
      font-size: 20px;
      font-weight: 600;
    }
    .send-button {
      align-items: center;
      background: var(--blue);
      border: 0;
      border-radius: 12px;
      color: #fff;
      display: inline-flex;
      height: 49px;
      justify-content: center;
      padding: 0;
      width: 49px;
    }
    .send-button:hover { background: var(--blue-strong); }
    .send-button:disabled {
      background: #98a2b3;
      cursor: default;
    }
    .status-line {
      color: #667085;
      font-size: 13px;
      margin: 8px auto 0;
      max-width: 1086px;
      min-height: 18px;
      padding-left: 4px;
    }
    .status-line.error { color: var(--danger); }
    .typing {
      align-items: center;
      display: inline-flex;
      gap: 6px;
      min-height: 28px;
    }
    .typing span {
      animation: bounce 1s infinite ease-in-out;
      background: #98a2b3;
      border-radius: 999px;
      height: 7px;
      width: 7px;
    }
    .typing span:nth-child(2) { animation-delay: 0.12s; }
    .typing span:nth-child(3) { animation-delay: 0.24s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: .45; }
      40% { transform: translateY(-4px); opacity: 1; }
    }
    @media (max-width: 900px) {
      .brand-strip { min-height: 54px; padding: 15px 16px 4px; }
      .chat-canvas { padding: 0 14px 16px; }
      .message-card { padding: 24px 18px 12px; }
      .assistant-content { font-size: 18px; }
      .bubble { max-width: 92%; }
      .composer-panel {
        bottom: 14px;
        left: 10px;
        right: 10px;
      }
      .composer-inner {
        min-height: 64px;
        grid-template-columns: minmax(0, 1fr) 44px;
      }
      .send-button {
        height: 44px;
        width: 44px;
      }
    }
  </style>
</head>
<body>
  <main class="dify-shell">
    <div class="brand-strip">
      <div class="powered"><span>POWERED BY</span><span class="dify-word">Dify</span></div>
      <span class="brand-divider" aria-hidden="true"></span>
      <button id="clearChat" class="reset-button" type="button" aria-label="清空对话">
        <svg width="23" height="23" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M21 12a9 9 0 1 1-2.64-6.36" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
          <path d="M21 4v6h-6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>

    <section id="chatCanvas" class="chat-canvas" aria-label="课程智能体对话窗口">
      <div id="messageList" class="message-list" aria-live="polite">
        <article class="message-card" id="welcomeCard">
          <div class="assistant-content">
            <ul>
              <li><strong>课程事务问题</strong>：解答上课时间、地点、教师联系方式、评分方式、阅读材料等课程信息。</li>
              <li><strong>R 语言知识</strong>：讲解 R 语言基础语法、数据类型、tidyverse 核心包（如 <code>tibble</code>、<code>dplyr</code>、<code>ggplot2</code>）、R Markdown 等课程内容，并给出可运行的代码示例。</li>
              <li><strong>课堂练习辅导</strong>：指导你完成课堂练习，提供结构建议和代码逻辑检查，帮助你提升练习质量。</li>
              <li><strong>作业与论文参考</strong>：协助你缩小研究兴趣、拆解研究问题、设计分析方法、提供代码片段和论文结构建议，但不会代写完整作业或论文。</li>
            </ul>
            <p>你可以随时向我提问，无论是“这节课推荐读什么书”，还是“如何用 <code>dplyr</code> 筛选数据”，我都会根据课程知识库给出准确、具体的回答。如果有不确定的地方，我也会如实告知，并建议你向老师或助教确认。</p>
            <p>期待与你一起学习 R 语言！有什么我可以帮你的？</p>
          </div>
          <div class="source-section">
            <div class="source-title">引用</div>
            <div class="source-list">
              <span class="source-chip"><span class="source-icon">doc</span>introduction-to-R.md</span>
              <span class="source-chip"><span class="source-icon">doc</span>使用tibble实现简单数据框.md</span>
              <span class="source-chip"><span class="source-icon">doc</span>使用ggplot2进行数据可视化II.md</span>
              <button class="source-more" type="button">+ 3</button>
            </div>
          </div>
        </article>
      </div>

      <div class="composer-panel">
        <div class="composer-inner">
          <textarea id="questionInput" rows="1" placeholder="和 R 课程智能体 聊天"></textarea>
          <button id="sendButton" class="send-button" type="button" aria-label="发送">
            <svg width="27" height="27" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M4 5.5 20 12 4 18.5v-5.2L13.5 12 4 10.7V5.5Z" fill="currentColor"/>
            </svg>
          </button>
        </div>
        <div id="statusLine" class="status-line"></div>
      </div>
    </section>
  </main>
  <script>
    const chatCanvas = document.getElementById("chatCanvas");
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
      chatCanvas.scrollTop = chatCanvas.scrollHeight;
    }

    function makeSourceChip(source, index) {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      const icon = document.createElement("span");
      icon.className = "source-icon";
      icon.textContent = "doc";
      const label = document.createElement("span");
      const raw = source.title || source.source || `来源 ${index + 1}`;
      label.textContent = raw.length > 30 ? `${raw.slice(0, 30)}...` : raw;
      chip.appendChild(icon);
      chip.appendChild(label);
      return chip;
    }

    function addMessage(role, text, sources = []) {
      const row = document.createElement("div");
      row.className = `message-row ${role}`;

      const bubble = document.createElement("div");
      bubble.className = "bubble";

      const content = document.createElement("div");
      content.className = role === "assistant" ? "assistant-content" : "";
      content.textContent = text;
      bubble.appendChild(content);

      if (role === "assistant" && sources.length) {
        const section = document.createElement("div");
        section.className = "source-section";
        const title = document.createElement("div");
        title.className = "source-title";
        title.textContent = "引用";
        const list = document.createElement("div");
        list.className = "source-list";
        sources.slice(0, 3).forEach((source, index) => list.appendChild(makeSourceChip(source, index)));
        if (sources.length > 3) {
          const more = document.createElement("button");
          more.className = "source-more";
          more.type = "button";
          more.textContent = `+ ${sources.length - 3}`;
          list.appendChild(more);
        }
        section.appendChild(title);
        section.appendChild(list);
        bubble.appendChild(section);
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
        setStatus("");
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
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        ask();
      }
    });
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
    });
    document.getElementById("clearChat").addEventListener("click", () => {
      conversationHistory.splice(0, conversationHistory.length);
      messageList.querySelectorAll(".message-row").forEach(node => node.remove());
      setStatus("");
      input.focus();
      scrollToBottom();
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
