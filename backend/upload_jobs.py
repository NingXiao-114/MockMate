"""
上传任务进度管理。
轻量版先使用进程内存保存任务状态，适合当前单进程开发部署。
如果后续要支持多进程或服务重启恢复，可以把同样的数据结构迁移到 Redis/PostgreSQL。

"""
from datetime import datetime, UTC
from typing import Literal

StepStatus = Literal["pending", "running", "completed", "failed"]
JobStatus = Literal["pending", "running", "completed", "failed"]

DEFAULT_STEPS = [
    ("upload", "文档上传"),
    ("cleanup", "清理旧版本"),
    ("parse", "解析与分块"),
    ("parent_store", "父级分块入库"),
    ("vector_store", "向量化入库"),
]

DELETE_STEPS = [
    ("prepare", "准备删除"),
    ("bm25", "同步 BM25 统计"),
    ("milvus", "删除向量数据"),
    ("parent_store", "删除父级分块"),
]

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()

