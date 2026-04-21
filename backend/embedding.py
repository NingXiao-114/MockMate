
import json
import re

import jieba

import math
import os
import threading
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"


def _create_dense_embedder() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

class EmbeddingService:
    """文本向量化服务 - 密集向量本地模型 + BM25 稀疏向量（持久化统计）"""

    def __init__(self, state_path: Path | str | None = None):
        self._embedder = _create_dense_embedder()
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        self._lock = threading.Lock()

        # BM25 参数
        self.k1 = 1.5
        self.b = 0.75

        self._vocab: dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter[str] = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0

        self._load_state()

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def _load_state(self) -> None:
        path = self._state_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("version") != 1:
            return
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0
        self._recompute_avg_len()

    def _persist_unlock(self)->None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)


    def _persist(self) -> None:
        with self._lock:
            self._persist_unlock()


    _STOPWORDS = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
        "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
        "着", "没有", "看", "好", "自己", "这",
        "in", "on", "at", "to", "of", "a", "an", "the", "is", "are",
        "was", "were", "be", "been", "and", "or", "but", "for", "with",
    }

    _CN = re.compile(r"[\u4e00-\u9fff]")
    _EN = re.compile(r"[a-zA-Z0-9]+")

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = []
        i = 0
        while i < len(text):
            char = text[i]
            if self._CN.match(char):
                j = i
                while j < len(text) and self._CN.match(text[j]):
                    j += 1
                tokens.extend(jieba.lcut(text[i:j]))
                i = j
            else:
                match = self._EN.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
                else:
                    i += 1
        return [t for t in tokens if t not in self._STOPWORDS]

    def increment_add_document(self , texts : list[str]):
        """
        将每个 text 视为 BM25 中的一篇文档（与当前 chunk 写入粒度一致），增量更新 N / df / 长度和。
        :param texts:
        :return:
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlock()

    def increment_remove_documents(self , texts : list[str])->None:
        """
        从语料统计中移除与 increment_add_documents 对称的文档集合（如删除某文件的全部 chunk 文本）。
        词表索引不回收，避免与 Milvus 中仍可能存在的旧稀疏向量维度冲突。
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(self._sum_token_len  - doc_len, 0)
                self._total_docs = max(self._total_docs - 1, 0)
                for token in set(tokens):
                    if token not in self._vocab:
                        continue
                    self._doc_freq[token] = max(self._doc_freq[token]-1,0)
                    if self._doc_freq[token] == 0:
                        del self._vocab[token]
                    self._recompute_avg_len()
                    self._persist_unlock()

    def get_embeddings(self , texts : list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embedder.embed_documents(texts)
        except Exception as e:
            raise Exception(f"调用本地嵌入模型失败:{str(e)}") from e

    def _sparse_vector_for_text_unlocked(self , text : str) ->tuple[dict , bool]:

        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector : dict[int , float] = {}
        vocab_changed = False

        n = max(self._total_docs , 0)
        avg = max(self._avg_doc_len , 1.0)

        for token , freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True

            idx = self._vocab[token]
            df = self._doc_freq[token]

            if df == 0:
                idf = math.log((n + 1) / 1)
            else:
                idf = math.log( (n - df + 0.5 )/(df + 0.5) + 1  )

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            score = idf * numerator / denominator
            if score > 0:
                sparse_vector[idx] = float(score)
        return sparse_vector, vocab_changed

    def get_sparse_embedding(self, text: str) -> dict:
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlock()
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        with self._lock:
            out: list[dict] = []
            any_new_vocab = False
            for text in texts:
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                out.append(sparse_vector)
                any_new_vocab = any_new_vocab or vocab_changed
            if any_new_vocab:
                self._persist_unlock()
        return out

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings

# 全进程唯一实例：写入与检索共用同一份 BM25 持久化状态
embedding_service = EmbeddingService()