import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk

from cache import redis_cache
from database import SessionLocal
from models import User, ChatSession, ChatMessage
from tools import search_knowledge_base, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

class ConversationStorage:

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )

            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()

            else:
                session.metadata_json = metadata or {}
            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)
            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")
            db.add(
                ChatMessage(
                    session_ref_id=session.id,
                    message_type=msg.type,
                    content=str(msg.content),
                    timestamp=now,
                    rag_trace=rag_trace,
                )
            )
            serialized.append(
                {
                    "type": msg.type,
                    "content": str(msg.content),
                    "timestamp": now.isoformat(),
                    "rag_trace": rag_trace,
                }
            )
            session.updated_at = now
            db.commit()
            redis_cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            redis_cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()


    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = redis_cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).all()

            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            redis_cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话"""
        cached = redis_cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(user_id, session_id)
        redis_cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = redis_cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            redis_cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
         db.close()

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def delete_session(self, user_id: str, session_id: str) -> bool:
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False
            db.delete(session)
            db.commit()
            redis_cache.delete(self._messages_cache_key(user_id, session_id))
            redis_cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()

def create_agent_instance():
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt=(
            "你是一个专业且富有鼓励精神的面试复习助手，致力于帮助用户准备求职面试。"
            "在回答时，你可以使用工具来辅助。"
            "当用户询问技术概念、行为面试题或面试准备资料时，请使用 search_knowledge_base 工具。"
            "不要在同一轮对话中重复调用同一个工具。每轮对话最多调用一次知识库搜索工具。"
            "一旦你调用了 search_knowledge_base 并收到结果，你必须立即基于该结果生成最终答案，以解释概念或解答面试问题。"
            "在收到 search_knowledge_base 的结果后，你绝对不能再次调用任何工具（包括再次搜索知识库）。"
            "如果检索到的上下文不足以回答该面试问题，请诚实地回答你没有足够的信息，而不是捏造技术事实或伪造面试指南。"
            "如果工具返回的结果包含‘后退提问/解答’（Step-back Question/Answer），请运用该通用原则或核心概念来推理并解答面试问题，但不要向用户暴露你的思维链（chain-of-thought）。"
            "如果你不知道答案，请诚实承认，并鼓励用户进一步探索该知识点。"
        ),
    )
    return agent, model

agent, model = create_agent_instance()

storage = ConversationStorage()


def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要"""
    # 提取旧对话
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    summary = model.invoke(summary_prompt).content
    return summary

def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并返回响应"""
    messages = storage.load(user_id, session_id)
    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    if len(messages) > 50:
        #压缩前40条 平衡 精度 和 空间使用
        summary = summarize_old_messages(model, messages[:40])

    messages = [
        SystemMessage(content=f"之前的对话摘要：\n{summary}")
    ] + messages[40:]

    messages.append(HumanMessage(content=user_text))

    #在这个invoke的过程中可能调用 搜索数据库 设置了rag上下文
    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 8},
    )
    response_content = ""

    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)

    messages.append(AIMessage(content=response_content))

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None


    #对齐数据长度 只有最后的一条由AI生成的回复才需要rag_trace
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }

async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。

       架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
       而非等待工具完成后才显示。
       """
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    messages.append(HumanMessage(content=user_text))

    full_response = ""

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
        try:
            async for msg, metadata in agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())
    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭
    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)