from backend.embedding import EmbeddingService


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""
    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()
