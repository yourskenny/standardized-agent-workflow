from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

from . import __version__
from .cli import chat_question
from .config import Settings
from .runtime import AgentRuntime
from .types import AgentRequest
from .ui import UiConfig, load_ui_config
from .validator import validate_project_config, validate_project_contract

SecurityHook = Callable[[BaseHTTPRequestHandler, Settings], bool]


@dataclass(frozen=True)
class ServerHooks:
    authenticate: SecurityHook | None = None
    rate_limit: SecurityHook | None = None


class RequestTooLarge(ValueError):
    pass


def run_server(settings: Settings, port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(settings))
    print(f"Local RAG agent server listening at http://127.0.0.1:{port}")
    server.serve_forever()


def make_handler(
    settings: Settings,
    ui: UiConfig | None = None,
    hooks: ServerHooks | None = None,
) -> type[BaseHTTPRequestHandler]:
    ui = ui or load_ui_config(settings.ui_config_path)
    hooks = hooks or ServerHooks()

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            if settings.server_timeout_seconds > 0:
                self.connection.settimeout(settings.server_timeout_seconds)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_workspace_home(ui))
                return
            if parsed.path == "/healthz":
                self._send_json({"status": "ok", "service": "local_rag_agent"})
                return
            if parsed.path == "/version":
                self._send_json({"service": "local_rag_agent", "version": __version__})
                return
            if parsed.path == "/api/v1/validate":
                if not self._guard_api_request():
                    return
                self._send_json(_validate_payload(settings))
                return
            if parsed.path in {"/chatbot", "/chatbot/"}:
                self._send_html(render_chat_page(ui=ui))
                return
            if parsed.path == "/api/chat":
                question = parse_qs(parsed.query).get("q", [""])[0]
                self._send_chat_response(lambda: chat_question(settings, question))
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path == "/api/v1/chat":
                if not self._guard_api_request():
                    return
                self._send_chat_v1_response()
                return
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

        def do_OPTIONS(self) -> None:
            origin = self.headers.get("Origin", "")
            self.send_response(204)
            self._send_cors_headers(origin)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers(self.headers.get("Origin", ""))
            self.end_headers()
            self.wfile.write(body)

        def _send_cors_headers(self, origin: str) -> None:
            if origin and origin in settings.server_cors_allowlist:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _send_error_json(self, status: int, code: str, message: str) -> None:
            self._send_json({"error": {"code": code, "message": message}}, status=status)

        def _read_json_body(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > settings.server_request_body_limit_bytes:
                raise RequestTooLarge(
                    f"request body exceeds limit of {settings.server_request_body_limit_bytes} bytes"
                )
            body = self.rfile.read(length).decode("utf-8")
            if not body:
                return {}
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _send_chat_v1_response(self) -> None:
            try:
                data = self._read_json_body()
            except RequestTooLarge as error:
                self._send_error_json(413, "REQUEST_TOO_LARGE", str(error))
                return
            except (json.JSONDecodeError, ValueError) as error:
                self._send_error_json(400, "BAD_REQUEST", str(error))
                return
            message = str(data.get("message", "")).strip()
            if not message:
                self._send_error_json(400, "BAD_REQUEST", "message is required")
                return
            history = data.get("history", [])
            if not isinstance(history, list):
                history = []
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            try:
                response = AgentRuntime(settings).run(
                    AgentRequest(message=message, history=history, metadata=metadata)
                )
            except Exception as error:  # pragma: no cover - integration failures depend on project config
                self._send_error_json(500, "RUNTIME_ERROR", str(error))
                return
            self._send_json(response.to_dict())

        def _guard_api_request(self) -> bool:
            if hooks.authenticate is not None:
                if not hooks.authenticate(self, settings):
                    self._send_error_json(401, "AUTH_REQUIRED", "authentication required")
                    return False
            elif not self._default_authenticated():
                self._send_error_json(401, "AUTH_REQUIRED", "authentication required")
                return False
            if hooks.rate_limit is not None and not hooks.rate_limit(self, settings):
                self._send_error_json(429, "RATE_LIMITED", "rate limit exceeded")
                return False
            return True

        def _default_authenticated(self) -> bool:
            if not settings.server_auth_token and not settings.server_basic_auth_username:
                return True
            authorization = self.headers.get("Authorization", "")
            if settings.server_auth_token and authorization == f"Bearer {settings.server_auth_token}":
                return True
            if settings.server_basic_auth_username:
                expected = base64.b64encode(
                    f"{settings.server_basic_auth_username}:{settings.server_basic_auth_password}".encode("utf-8")
                ).decode("ascii")
                return authorization == f"Basic {expected}"
            return False

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

    return Handler


def _validate_payload(settings: Settings) -> dict[str, object]:
    if settings.config_path is not None:
        return validate_project_config(settings.project_root, settings.config_path).to_dict()
    return validate_project_contract(settings).to_dict()


def render_chat_page(title: str | None = None, ui: UiConfig | None = None) -> str:
    ui = ui or UiConfig(title=title or "Local Agent")
    safe_title = escape_text(title or ui.title)
    welcome_items = _render_welcome_items(ui)
    demo_sources = _render_demo_sources(ui)
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
    .brand-mark {
      align-items: center;
      color: #667085;
      display: inline-flex;
      font-size: 13px;
      gap: 9px;
      letter-spacing: 0;
      line-height: 1;
      white-space: nowrap;
    }
    .brand-word {
      color: #14532d;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 0;
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
      padding: 24px 25px 14px;
      width: min(100%, 942px);
    }
    .assistant-content {
      color: #111827;
      font-size: 17px;
      line-height: 1.34;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .assistant-content p {
      margin: 0 0 6px;
    }
    .assistant-content ul {
      margin: 0 0 8px;
      padding-left: 22px;
    }
    .assistant-content li { margin: 2px 0; }
    .assistant-content code {
      background: #eef0f2;
      border-radius: 6px;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 14px;
      padding: 1px 6px;
    }
    .message-row {
      display: flex;
      width: 100%;
    }
    .message-row.user { justify-content: flex-end; }
    .message-row.assistant { justify-content: center; }
    .bubble {
      border-radius: 16px;
      font-size: 16px;
      line-height: 1.34;
      max-width: min(82%, 760px);
      padding: 12px 16px;
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
      margin-top: 14px;
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
    button.source-chip:hover,
    button.source-more:hover {
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10);
      color: #344054;
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
    .knowledge-popover {
      background: #fff;
      border: 1px solid #edf0f4;
      border-radius: 10px;
      box-shadow: 0 22px 60px rgba(15, 23, 42, 0.18);
      display: none;
      left: 50%;
      max-height: min(460px, 58vh);
      max-width: min(640px, calc(100vw - 80px));
      overflow: auto;
      padding: 18px 22px;
      position: fixed;
      top: 50%;
      transform: translate(-50%, -50%);
      width: 560px;
      z-index: 20;
    }
    .knowledge-popover.open { display: block; }
    .popover-head {
      align-items: start;
      display: flex;
      gap: 14px;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .popover-kicker {
      color: #98a2b3;
      font-size: 13px;
      margin-bottom: 6px;
    }
    .popover-title {
      color: #344054;
      font-size: 17px;
      font-weight: 700;
      line-height: 1.35;
    }
    .popover-close {
      background: #f2f4f7;
      border: 0;
      border-radius: 7px;
      color: #667085;
      flex: 0 0 auto;
      height: 30px;
      width: 30px;
    }
    .popover-content {
      border-top: 1px solid #edf0f4;
      color: #344054;
      font-size: 15px;
      line-height: 1.42;
      padding-top: 14px;
      white-space: pre-wrap;
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
      font-size: 18px;
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
      .assistant-content { font-size: 16px; }
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
      <div class="brand-mark"><span>SELF-BUILT AGENT</span><span class="brand-word">自主架构</span></div>
      <span class="brand-divider" aria-hidden="true"></span>
      <button id="clearChat" class="reset-button" type="button" aria-label="清空对话">
        <svg width="23" height="23" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M21 12a9 9 0 1 1-2.64-6.36" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
          <path d="M21 4v6h-6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>

    <section id="chatCanvas" class="chat-canvas" aria-label="Agent conversation">
      <div id="messageList" class="message-list" aria-live="polite">
        <article class="message-card" id="welcomeCard">
          <div class="assistant-content">
            __WELCOME_ITEMS__
            <p>__WELCOME_INTRO__</p>
          </div>
          <div class="source-section">
            <div class="source-title">Sources</div>
            <div class="source-list">
              __DEMO_SOURCES__
            </div>
          </div>
        </article>
      </div>

      <div class="composer-panel">
        <div class="composer-inner">
          <textarea id="questionInput" rows="1" placeholder="__PLACEHOLDER__"></textarea>
          <button id="sendButton" class="send-button" type="button" aria-label="发送">
            <svg width="27" height="27" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M4 5.5 20 12 4 18.5v-5.2L13.5 12 4 10.7V5.5Z" fill="currentColor"/>
            </svg>
          </button>
        </div>
        <div id="statusLine" class="status-line"></div>
      </div>
    </section>
    <div id="knowledgePopover" class="knowledge-popover" role="dialog" aria-modal="false" aria-label="知识库条目">
      <div class="popover-head">
        <div>
          <div id="popoverKicker" class="popover-kicker"># 1</div>
          <div id="popoverTitle" class="popover-title"></div>
        </div>
        <button id="popoverClose" class="popover-close" type="button" aria-label="关闭">×</button>
      </div>
      <div id="popoverContent" class="popover-content"></div>
    </div>
  </main>
  <script>
    const chatCanvas = document.getElementById("chatCanvas");
    const messageList = document.getElementById("messageList");
    const statusLine = document.getElementById("statusLine");
    const input = document.getElementById("questionInput");
    const sendButton = document.getElementById("sendButton");
    const popover = document.getElementById("knowledgePopover");
    const popoverKicker = document.getElementById("popoverKicker");
    const popoverTitle = document.getElementById("popoverTitle");
    const popoverContent = document.getElementById("popoverContent");
    const conversationHistory = [];
    let pendingBubble = null;

    function setStatus(text, isError = false) {
      statusLine.textContent = text;
      statusLine.classList.toggle("error", isError);
    }

    function scrollToBottom() {
      chatCanvas.scrollTop = chatCanvas.scrollHeight;
    }

    function closeSourcePopover() {
      popover.classList.remove("open");
    }

    function openSourcePopover(source, index) {
      popoverKicker.textContent = `# ${index + 1}`;
      popoverTitle.textContent = source.title || source.source || `知识库条目 ${index + 1}`;
      const metadata = source.source ? `来源：${source.source}\n\n` : "";
      popoverContent.textContent = `${metadata}${source.content || "该来源没有返回可展示的片段内容。"}`;
      popover.classList.add("open");
    }

    function makeSourceChip(source, index) {
      const chip = document.createElement("button");
      chip.className = "source-chip";
      chip.type = "button";
      chip.addEventListener("click", () => openSourcePopover(source, index));
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
          more.addEventListener("click", () => openSourcePopover(sources[3], 3));
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
      setStatus("__STATUS_TEXT__");
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
    document.getElementById("popoverClose").addEventListener("click", closeSourcePopover);
    popover.addEventListener("click", (event) => event.stopPropagation());
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeSourcePopover();
    });
    document.querySelectorAll("[data-demo-source]").forEach((button, index) => {
      button.addEventListener("click", () => {
        openSourcePopover({
          title: button.dataset.demoSource || button.textContent,
          source: "Example welcome source",
          content: "This is an example source shown before the first answer. Real answers display retrieved project knowledge snippets."
        }, index);
      });
    });
    input.focus();
  </script>
</body>
</html>""".replace("__TITLE__", safe_title).replace(
        "__WELCOME_ITEMS__", welcome_items
    ).replace(
        "__WELCOME_INTRO__", escape_text(ui.welcome_intro)
    ).replace(
        "__DEMO_SOURCES__", demo_sources
    ).replace(
        "__PLACEHOLDER__", escape_text(ui.placeholder)
    ).replace(
        "__STATUS_TEXT__", escape_js_string(ui.status_text)
    )


def render_workspace_home(ui: UiConfig | None = None) -> str:
    ui = ui or UiConfig()
    return """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>__HOME_TITLE__</title>
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
      <a class="brand" href="/">__HOME_TITLE__</a>
      <nav>
        <a href="/chatbot">Agent</a>
        <a href="#boundaries">Boundaries</a>
      </nav>
    </header>

    <main class="home-shell">
      <section class="intro-panel">
        <div>
          <p class="eyebrow">Local runtime</p>
          <h1>__HOME_HEADING__</h1>
          <p class="lead">__HOME_LEAD__</p>
        </div>
        <div class="quick-actions">
          <a class="primary-action" href="/chatbot">Open agent</a>
          <a class="secondary-action" href="#boundaries">Review boundaries</a>
        </div>
      </section>

      <section class="split-layout">
        <article class="agent-panel">
          <div class="section-head">
            <p class="eyebrow">Agent</p>
            <h2>Local agent</h2>
          </div>
          <div id="agentMount" class="agent-frame">
            <iframe src="/chatbot" title="Local agent"></iframe>
          </div>
        </article>

        <aside id="boundaries" class="notice-panel">
          <div class="section-head">
            <p class="eyebrow">Boundary</p>
            <h2>Boundaries</h2>
          </div>
          <ul class="check-list">
            <li>Only files listed in the project manifest are used for retrieval.</li>
            <li>Maintenance-only and pre-ingestion material should stay outside user-facing answers.</li>
            <li>High-risk or unsupported answers should include sources or be refused.</li>
            <li>Project-specific policy belongs in configuration, not runtime code.</li>
          </ul>
        </aside>
      </section>
    </main>
  </body>
</html>""".replace("__HOME_TITLE__", escape_text(ui.home_title)).replace(
        "__HOME_HEADING__", escape_text(ui.home_heading)
    ).replace("__HOME_LEAD__", escape_text(ui.home_lead))


def escape_text(value: str) -> str:
    return html.escape(value, quote=True)


def escape_js_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1]


def _render_welcome_items(ui: UiConfig) -> str:
    items = ui.welcome_items or [
        WelcomeItemProxy("Source-backed questions", "Answer questions from the configured project knowledge base."),
        WelcomeItemProxy("Traceable evidence", "Show retrieved sources so maintainers can inspect answer support."),
        WelcomeItemProxy("Boundary-aware help", "Refuse or downgrade requests when project policy requires it."),
    ]
    lines = ["<ul>"]
    for item in items:
        lines.append(f"<li><strong>{escape_text(item.title)}</strong>: {escape_text(item.text)}</li>")
    lines.append("</ul>")
    return "\n".join(lines)


def _render_demo_sources(ui: UiConfig) -> str:
    sources = ui.demo_sources or [
        WelcomeSourceProxy("project-facts.md", "project-facts.md"),
        WelcomeSourceProxy("policies.md", "policies.md"),
        WelcomeSourceProxy("regression-questions.md", "regression-questions.md"),
    ]
    return "\n".join(
        f'<button class="source-chip" type="button" data-demo-source="{escape_text(source.source or source.label)}">'
        f'<span class="source-icon">doc</span>{escape_text(source.label)}</button>'
        for source in sources
    )


class WelcomeItemProxy:
    def __init__(self, title: str, text: str) -> None:
        self.title = title
        self.text = text


class WelcomeSourceProxy:
    def __init__(self, label: str, source: str) -> None:
        self.label = label
        self.source = source
