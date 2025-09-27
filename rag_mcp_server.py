#!/usr/bin/env python3
"""MCP server exposing the Baby RAG system."""

from __future__ import annotations

import os
from typing import Any, Dict

from mcp.server.fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse

from rag_system import BabyRAGSystem


HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "3333"))

mcp_server = FastMCP(
    name="Baby RAG MCP",
    instructions="Query 15 tech snippets via vector search",
    host=HOST,
    port=PORT,
    json_response=True,
)


HARNESS_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Baby RAG MCP Playground</title>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; margin: 0; background: #0b0b0f; color: #f9f9fb; }
main { max-width: 960px; margin: 0 auto; padding: 32px 18px 48px; }
h1 { margin-bottom: 0.5rem; font-size: 2rem; }
p.lead { margin-top: 0; margin-bottom: 1.5rem; color: #cfd0d7; }
section { margin-bottom: 1.5rem; }
label { display: block; font-weight: 600; margin-bottom: 0.4rem; }
input[type="text"] { width: 100%; padding: 0.6rem; border-radius: 6px; border: 1px solid #2c2c36; background: #14141a; color: inherit; }
textarea { width: 100%; min-height: 200px; padding: 0.75rem; border-radius: 6px; border: 1px solid #2c2c36; background: #14141a; color: inherit; font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 0.95rem; }
button { padding: 0.55rem 1rem; border-radius: 6px; border: 1px solid #2c2c36; background: #1f64ff; color: white; font-weight: 600; cursor: pointer; }
button.secondary { background: transparent; color: #f9f9fb; border-color: #3a3a46; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.button-row { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }
.status { margin-top: 0.75rem; font-size: 0.95rem; }
.status[data-state="error"] { color: #ff8c8c; }
.status[data-state="ok"] { color: #7cf59f; }
.io-grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
pre { margin: 0; padding: 1rem; border-radius: 6px; background: #14141a; border: 1px solid #2c2c36; overflow: auto; max-height: 320px; }
code { font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 0.9rem; }
a { color: #86b7ff; }
</style>
</head>
<body>
<main>
  <h1>Baby RAG MCP Playground</h1>
  <p class=\"lead\">Initialize a session with the MCP server and send raw JSON-RPC messages. The harness handles the session header for you.</p>

  <section>
    <label for=\"serverUrl\">MCP server endpoint</label>
    <input id=\"serverUrl\" type=\"text\" placeholder=\"http://127.0.0.1:3333/mcp\">
    <div class=\"button-row\">
      <button id=\"initializeBtn\">Initialize Session</button>
      <button id=\"resetBtn\" class=\"secondary\">Reset Session</button>
      <span>Session ID: <code id=\"sessionIdValue\">(none)</code></span>
    </div>
    <div id=\"statusMessage\" class=\"status\" data-state=\"ok\"></div>
  </section>

  <section>
    <label for=\"requestBody\">Request JSON</label>
    <textarea id=\"requestBody\"></textarea>
    <div class=\"button-row\">
      <button id=\"sendRequestBtn\" disabled>Send Request</button>
      <span>Tip: change <code>params.arguments.question</code> to try different prompts.</span>
    </div>
  </section>

  <section class=\"io-grid\">
    <div>
      <h3>Last Request</h3>
      <pre id=\"lastRequest\"></pre>
    </div>
    <div>
      <h3>Last Response</h3>
      <pre id=\"lastResponse\"></pre>
    </div>
  </section>
</main>
<script>
(function () {
  const serverInput = document.getElementById('serverUrl');
  const initializeBtn = document.getElementById('initializeBtn');
  const sendRequestBtn = document.getElementById('sendRequestBtn');
  const resetBtn = document.getElementById('resetBtn');
  const requestBodyInput = document.getElementById('requestBody');
  const sessionIdValue = document.getElementById('sessionIdValue');
  const statusMessage = document.getElementById('statusMessage');
  const lastRequest = document.getElementById('lastRequest');
  const lastResponse = document.getElementById('lastResponse');

  let sessionId = null;

  function defaultServerUrl() {
    try {
      return new URL('/mcp', window.location.origin).toString();
    } catch (error) {
      console.warn('Falling back to default MCP endpoint', error);
      return 'http://127.0.0.1:3333/mcp';
    }
  }

  function updateSessionView() {
    sessionIdValue.textContent = sessionId || '(none)';
  }

  function setStatus(message, state) {
    statusMessage.textContent = message;
    statusMessage.dataset.state = state || 'ok';
  }

  function prettyPrint(value) {
    if (value === null || value === undefined) {
      return String(value);
    }
    if (typeof value === 'string') {
      return value;
    }
    try {
      return JSON.stringify(value, null, 2);
    } catch (error) {
      return String(value);
    }
  }

  async function sendEnvelope(payload) {
    const url = serverInput.value.trim();
    if (!url) {
      throw new Error('Server URL is required.');
    }

    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream'
    };

    if (sessionId) {
      headers['mcp-session-id'] = sessionId;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    });

    const text = await response.text();
    const headerDump = {};
    response.headers.forEach((value, key) => {
      headerDump[key] = value;
    });

    const maybeSession = response.headers.get('mcp-session-id');
    if (maybeSession) {
      sessionId = maybeSession;
    }
    updateSessionView();

    let parsedBody;
    if (text) {
      try {
        parsedBody = JSON.parse(text);
      } catch (error) {
        parsedBody = text;
      }
    } else {
      parsedBody = '(empty body)';
    }

    const responseInfo = {
      status: response.status,
      statusText: response.statusText,
      headers: headerDump,
      body: parsedBody
    };

    lastRequest.textContent = prettyPrint(payload);
    lastResponse.textContent = prettyPrint(responseInfo);

    if (!response.ok) {
      throw new Error('HTTP ' + response.status + ' ' + response.statusText);
    }

    return responseInfo;
  }

  async function initializeSession() {
    setStatus('Initializing session…');
    try {
      const initPayload = {
        jsonrpc: '2.0',
        id: 'init-' + Date.now(),
        method: 'initialize',
        params: {
          protocolVersion: '2025-06-18',
          capabilities: {},
          clientInfo: {
            name: 'Baby RAG HTML Harness',
            version: '0.1.0'
          }
        }
      };

      await sendEnvelope(initPayload);

      const initializedNotification = {
        jsonrpc: '2.0',
        method: 'notifications/initialized',
        params: {}
      };
      await sendEnvelope(initializedNotification);

      sendRequestBtn.disabled = false;
      setStatus('Session ready. You can now send requests.');
    } catch (error) {
      console.error(error);
      setStatus('Initialization failed: ' + error.message, 'error');
    }
  }

  async function sendCustomRequest() {
    try {
      const payload = JSON.parse(requestBodyInput.value);
      setStatus('Sending request…');
      await sendEnvelope(payload);
      setStatus('Request completed.');
    } catch (error) {
      console.error(error);
      setStatus('Request failed: ' + error.message, 'error');
    }
  }

  function resetSession() {
    sessionId = null;
    updateSessionView();
    sendRequestBtn.disabled = true;
    lastRequest.textContent = '';
    lastResponse.textContent = '';
    setStatus('Session cleared. Initialize again to reconnect.');
  }

  initializeBtn.addEventListener('click', initializeSession);
  sendRequestBtn.addEventListener('click', sendCustomRequest);
  resetBtn.addEventListener('click', resetSession);

  requestBodyInput.value = JSON.stringify({
    jsonrpc: '2.0',
    id: 'call-1',
    method: 'tools/call',
    params: {
      name: 'rag_query',
      arguments: {
        question: 'What is machine learning?',
        n_results: 3
      }
    }
  }, null, 2);

  serverInput.value = defaultServerUrl();
  updateSessionView();
  setStatus('Click "Initialize Session" to begin.');
})();
</script>
</body>
</html>
"""


@mcp_server.custom_route("/playground", methods=["GET"])
async def playground(_: Request) -> HTMLResponse:
    return HTMLResponse(content=HARNESS_HTML)


def _ensure_rag(context: Context) -> BabyRAGSystem:
    server = context.fastmcp
    instance = getattr(server, "_rag_instance", None)
    if instance is None:
        instance = BabyRAGSystem()
        setattr(server, "_rag_instance", instance)
    return instance


@mcp_server.tool(name="rag_query", description="Query the Baby RAG system for relevant snippets")
async def rag_query(question: str, n_results: int = 3, context: Context | None = None) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("question cannot be empty")
    if not 1 <= n_results <= 10:
        raise ValueError("n_results must be between 1 and 10")

    if context is None:
        raise ValueError("context is required")

    rag = _ensure_rag(context)
    results = rag.query(question, n_results=n_results)
    return {
        "question": question,
        "results": results,
        "total_results": len(results),
    }


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    if transport == "streamable-http":
        print(f"🚀 Starting Baby RAG MCP server on {HOST}:{PORT} (streamable HTTP)")
        print("🔗 Endpoint: http://%s:%s%s" % (HOST, PORT, mcp_server.settings.streamable_http_path))
    else:
        print("🚀 Starting Baby RAG MCP server on stdio transport")
    mcp_server.run(transport=transport)


if __name__ == "__main__":
    main()

