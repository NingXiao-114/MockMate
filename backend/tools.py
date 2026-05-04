import json
import os
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
_MCP_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
}

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue

def _set_last_rag_context(context: dict):
   global _LAST_RAG_CONTEXT
   _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
   """获取最近一次 RAG 检索上下文，默认读取后清空。"""
   global _LAST_RAG_CONTEXT
   context = _LAST_RAG_CONTEXT
   if clear:
      _LAST_RAG_CONTEXT = None
   return context

def reset_tool_call_guards():
   """每轮对话开始时重置工具调用计数。"""
   global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
   _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def set_rag_step_queue(queue):
   """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
   global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
   _RAG_STEP_QUEUE = queue
   if queue:
      import asyncio
      try:
         _RAG_STEP_LOOP = asyncio.get_running_loop()
      except RuntimeError:
         _RAG_STEP_LOOP = asyncio.get_event_loop()
   else:
      _RAG_STEP_LOOP = None

def emit_rag_step(icon: str, label: str, detail: str = ""):
   """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
   global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
   if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
      step = {"icon": icon, "label": label, "detail": detail}
      try:
         if not _RAG_STEP_LOOP.is_closed():
            _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
      except Exception:
         pass

@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
   """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
   # ... guards omitted ...
   global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
   if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
      return (
         "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
         "Use the existing retrieval result and provide the final answer directly."
      )
   _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1
   from rag_pipeline import run_rag_graph
   # 在同步工具中获取当前的 Loop 可能不可靠，但我们之前是通过 call_soon_threadsafe 调度的。
   # 这里 _RAG_STEP_QUEUE 是在主线程/Loop 设置的全局变量。
   # 如果工具运行在线程池中，它是可以访问到全局变量 _RAG_STEP_QUEUE 的。
   # emit_rag_step 内部做了 try-except 和 get_event_loop()。

   # 问题可能出在 asyncio.get_event_loop() 在子线程中调用会报错或者拿不到主线程的loop。
   # 我们应该在 set_rag_step_queue 时也保存 loop 引用，或者在 emit_rag_step 中更健壮地获取 loop。
   rag_result = run_rag_graph(query)

   docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
   rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
   if rag_trace:
      _set_last_rag_context({"rag_trace": rag_trace})

   if not docs:
      return "No relevant documents found in the knowledge base."

   formatted = []
   for i, result in enumerate(docs, 1):
      source = result.get("filename", "Unknown")
      page = result.get("page_number", "N/A")
      text = result.get("text", "")
      formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

   return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)


def _parse_mcp_sse_response(response: requests.Response) -> dict:
    """解析 MCP SSE 响应，提取 JSON-RPC result。"""
    result = None
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        try:
            msg = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "result" in msg:
            result = msg["result"]
        if "error" in msg:
            raise RuntimeError(f"MCP error: {msg['error']}")
    return result


def call_mcp_tool(endpoint: str, headers: dict, tool_name: str, arguments: dict) -> str:
    """调用 MCP HTTP 工具并返回文本结果。"""
    # Step 1: initialize
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "mockmate", "version": "1.0.0"},
        },
    }
    resp = requests.post(endpoint, json=init_payload, headers=headers, timeout=30)
    resp.raise_for_status()
    session_id = resp.headers.get("Mcp-Session-Id")

    # Step 2: notifications/initialized
    notify_headers = {**headers}
    if session_id:
        notify_headers["Mcp-Session-Id"] = session_id
    notify_payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    requests.post(endpoint, json=notify_payload, headers=notify_headers, timeout=10)

    # Step 3: tools/call
    call_headers = {**headers}
    if session_id:
        call_headers["Mcp-Session-Id"] = session_id
    call_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    resp = requests.post(endpoint, json=call_payload, headers=call_headers, timeout=60)
    resp.raise_for_status()

    # 解析响应
    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        result = _parse_mcp_sse_response(resp)
    else:
        result = resp.json().get("result")

    if not result:
        return "联网搜索未返回结果。"

    # MCP tool result 格式: {"content": [{"type": "text", "text": "..."}]}
    contents = result.get("content", [])
    texts = [item.get("text", "") for item in contents if item.get("type") == "text"]
    return "\n".join(texts) if texts else "联网搜索未返回结果。"


@tool("web_search")
def web_search(query: str) -> str:
    """Search the internet for up-to-date information when the knowledge base cannot answer the question."""
    if not DASHSCOPE_API_KEY:
        return "联网搜索不可用：未配置 DASHSCOPE_API_KEY。"

    emit_rag_step("🌐", "正在联网搜索...", f"查询: {query[:50]}")

    try:
        result = call_mcp_tool(
            endpoint=_MCP_ENDPOINT,
            headers=_MCP_HEADERS,
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5},
        )
        emit_rag_step("✅", "联网搜索完成", f"结果长度: {len(result)} 字符")
        return result
    except Exception as e:
        emit_rag_step("❌", "联网搜索失败", str(e)[:80])
        return f"联网搜索出错: {e}"

