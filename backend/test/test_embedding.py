"""测试 EmbeddingService：分词、BM25 稀疏向量、密集向量"""

import math
import os
import tempfile
import time

from ..embedding import EmbeddingService


def test_tokenize():
    print("=" * 60)
    print("[测试] 中文分词 tokenize")
    print("=" * 60)

    svc = EmbeddingService(state_path=os.path.join(tempfile.mkdtemp(), "bm25_test.json"))

    text = "这是一个测试句子，包含English words和数字123。"
    tokens = svc.tokenize(text)
    print(f"  原文: {text}")
    print(f"  分词结果: {tokens}")
    assert len(tokens) > 0, "分词结果不应为空"

    has_stopword = any(t in svc._STOPWORDS for t in tokens)
    print(f"  包含停用词: {has_stopword}")
    assert not has_stopword, "分词结果不应包含停用词"

    print("[通过] 分词测试\n")


def test_increment_add_and_remove():
    print("=" * 60)
    print("[测试] BM25 增量添加与删除文档")
    print("=" * 60)

    state_path = os.path.join(tempfile.mkdtemp(), "bm25_test.json")
    svc = EmbeddingService(state_path=state_path)

    texts = ["人工智能技术发展迅速", "深度学习是人工智能的重要分支"]
    print(f"  添加文档数: {len(texts)}")
    for i, t in enumerate(texts):
        print(f"    [{i}] {t}")

    svc.increment_add_document(texts)
    print(f"  添加后: total_docs={svc._total_docs}, vocab_size={len(svc._vocab)}, "
          f"avg_doc_len={svc._avg_doc_len:.2f}")
    assert svc._total_docs == 2, f"total_docs 应为 2，实际为 {svc._total_docs}"

    print(f"  持久化文件存在: {os.path.isfile(state_path)}")
    assert os.path.isfile(state_path), "持久化文件应存在"

    svc.increment_remove_documents(texts)
    print(f"  删除后: total_docs={svc._total_docs}, vocab_size={len(svc._vocab)}")
    assert svc._total_docs == 0, f"total_docs 应为 0，实际为 {svc._total_docs}"

    print("[通过] 增量添加/删除测试\n")


def test_sparse_embedding():
    print("=" * 60)
    print("[测试] BM25 稀疏向量生成")
    print("=" * 60)

    state_path = os.path.join(tempfile.mkdtemp(), "bm25_test.json")
    svc = EmbeddingService(state_path=state_path)

    train_texts = ["自然语言处理是人工智能的重要方向", "机器学习模型需要大量数据训练"]
    svc.increment_add_document(train_texts)
    print(f"  训练文档数: {svc._total_docs}")

    query = "人工智能方向"
    sparse = svc.get_sparse_embedding(query)
    print(f"  查询文本: {query}")
    print(f"  稀疏向量维度数: {len(sparse)}")
    assert len(sparse) > 0, "稀疏向量不应为空"

    top_dims = sorted(sparse.items(), key=lambda x: -x[1])[:5]
    print(f"  Top-5 维度 (idx→score): {top_dims}")

    all_scores_positive = all(v > 0 for v in sparse.values())
    print(f"  所有分数为正: {all_scores_positive}")
    assert all_scores_positive, "BM25 分数应全部为正"

    print("[通过] 稀疏向量测试\n")


def test_sparse_embeddings_batch():
    print("=" * 60)
    print("[测试] 批量稀疏向量生成")
    print("=" * 60)

    state_path = os.path.join(tempfile.mkdtemp(), "bm25_test.json")
    svc = EmbeddingService(state_path=state_path)

    svc.increment_add_document(["测试文档内容"])

    queries = ["人工智能", "深度学习", "自然语言"]
    results = svc.get_sparse_embeddings(queries)
    print(f"  批量查询数: {len(queries)}")
    for i, (q, vec) in enumerate(zip(queries, results)):
        print(f"    [{i}] '{q}' → 维度数={len(vec)}")
    assert len(results) == len(queries), "批量结果数量应与输入一致"

    print("[通过] 批量稀疏向量测试\n")


def test_dense_embedding():
    print("=" * 60)
    print("[测试] 密集向量生成 (HuggingFace)")
    print("=" * 60)

    state_path = os.path.join(tempfile.mkdtemp(), "bm25_test.json")
    svc = EmbeddingService(state_path=state_path)

    texts = ["这是一个测试句子"]
    print(f"  输入文本: {texts}")
    print("  正在调用嵌入模型（首次加载可能较慢）...")

    t0 = time.time()
    dense = svc.get_embeddings(texts)
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  向量数量: {len(dense)}")
    print(f"  向量维度: {len(dense[0])}")

    assert len(dense) == 1, "应返回1个向量"
    assert len(dense[0]) > 0, "向量维度不应为0"

    norm = math.sqrt(sum(x * x for x in dense[0]))
    print(f"  L2 范数: {norm:.4f} (normalize_embeddings=True 应接近 1.0)")
    assert abs(norm - 1.0) < 0.01, "归一化向量的 L2 范数应接近 1.0"

    print("[通过] 密集向量测试\n")


def test_all_embeddings():
    print("=" * 60)
    print("[测试] 同时获取密集+稀疏向量")
    print("=" * 60)

    state_path = os.path.join(tempfile.mkdtemp(), "bm25_test.json")
    svc = EmbeddingService(state_path=state_path)
    svc.increment_add_document(["测试文档"])

    texts = ["人工智能技术"]
    print(f"  输入: {texts}")

    dense, sparse = svc.get_all_embeddings(texts)
    print(f"  密集向量: 数量={len(dense)}, 维度={len(dense[0])}")
    print(f"  稀疏向量: 数量={len(sparse)}, 非零维度={len(sparse[0])}")

    assert len(dense) == len(sparse) == 1, "数量应一致"

    print("[通过] 联合向量测试\n")


if __name__ == "__main__":
    test_tokenize()
    test_increment_add_and_remove()
    test_sparse_embedding()
    test_sparse_embeddings_batch()
    test_dense_embedding()
    test_all_embeddings()
    print("=" * 60)
    print("所有 EmbeddingService 测试通过!")
    print("=" * 60)
