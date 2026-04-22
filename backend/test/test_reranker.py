import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_utils import _rerank_documents

def test_rerank():
    query = "什么是机器学习？"
    docs = [
        {"chunk_id": "1", "text": "机器学习是人工智能的一个分支，通过数据训练模型来做出预测。", "score": 0.9},
        {"chunk_id": "2", "text": "今天天气很好，适合出去散步。", "score": 0.8},
        {"chunk_id": "3", "text": "深度学习是机器学习的子集，使用神经网络处理复杂任务。", "score": 0.7},
        {"chunk_id": "4", "text": "Python 是一种流行的编程语言。", "score": 0.6},
    ]

    print(f"查询: {query}")
    print(f"候选文档数: {len(docs)}\n")

    results, meta = _rerank_documents(query, docs, top_k=4)

    print(f"meta: {meta}\n")
    print("重排序结果:")
    for doc in results:
        print(f"  [{doc['score']:.4f}] (rrf_rank={doc.get('rrf_rank')}) {doc['text']}")

if __name__ == "__main__":
    test_rerank()
