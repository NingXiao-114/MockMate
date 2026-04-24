import os
from collections import defaultdict
from typing import Tuple, List, Dict, Any
from venv import logger

from langchain.chat_models import init_chat_model


from milvus_client import milvus_manager
from parent_chunk_store import parent_chunk_store

from embedding import embedding_service

import dotenv

dotenv.load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

RERANK_MODEL = os.getenv("RERANK_MODEL")

AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = milvus_manager
_parent_chunk_store = parent_chunk_store

_stepback_model = None
_rerank_model = None

def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from FlagEmbedding import FlagReranker
        _rerank_model = FlagReranker(RERANK_MODEL, use_fp16=True)
    return _rerank_model

# def _get_rerank_endpoint() -> str:
#     if not RERANK_BINDING_HOST:
#         return ""
#     host = RERANK_BINDING_HOST.strip().rstrip("/")
#     return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merged_parent_ids = [parent_id for parent_id , children in groups.items() if len(children) >= threshold]
    if not merged_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merged_parent_ids)
    parent_map = {item.get("chunk_id") : item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0

    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            merged_count += 1
            continue
        parent_doc = parent_map[parent_id]
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]


    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }

def _rerank_documents(query : str , docs : List[dict] , top_k : int ) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }

    if not docs_with_rank or not meta["rerank_enabled"]:
        return docs_with_rank[:top_k], meta

    try:
        reranker = _get_rerank_model()
        pairs = [[query, doc.get("text", "")] for doc in docs_with_rank]
        scores = reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        for doc, score in zip(docs_with_rank, scores):
            doc["score"] = float(score)
        docs_with_rank.sort(key=lambda d: d["score"], reverse=True)
        meta["rerank_applied"] = True
    except Exception as e:
        meta["rerank_error"] = str(e)

    return docs_with_rank[:top_k], meta

def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model

def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的‘退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""

def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""

def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def retrieve_documents(query :str , top_k: int = 5)-> Dict[str,Any]:
    candidate_k = top_k * 3
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    try:
        dense_embeddings =  embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = embedding_service.get_sparse_embedding(query)

        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding= dense_embedding,
            sparse_embedding= sparse_embedding,
            filter_expr=filter_expr,
            top_k = candidate_k
        )

        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
        rerank_meta["retrieval_mode"] = "hybrid"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta.update(merge_meta)
        return {"docs": merged_docs, "meta": rerank_meta}
    except Exception as e:
        logger.error("混合检索失败!降级为稠密检索" , e)
        try:
            dense_embeddings = embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta.update(merge_meta)
            return {"docs": merged_docs, "meta": rerank_meta}
        except Exception as e:
            logger.error("检索失败!", e)
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": bool(_rerank_model),
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "auto_merge_enabled": AUTO_MERGE_ENABLED,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                    "candidate_count": 0,
                },
            }
