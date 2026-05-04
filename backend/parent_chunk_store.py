"""父级分块文档存储（用于 Auto-merging Retriever）"""
import json
from datetime import datetime, timedelta, timezone
from typing import List

from cache import redis_cache
from database import SessionLocal
from models import ParentChunk

class ParentChunkStore:
    """基于 PostgreSQL + Redis 的父级分块存储。"""

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }


    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    def upsert_documents(self, docs: List[dict]) -> int:
        """写入/更新父级分块，返回写入条数。"""
        if not docs:
            return 0

        db = SessionLocal()
        upserted = 0

        # 1. 预处理提取所有有效的 chunk_id
        valid_docs = [doc for doc in docs if (doc.get("chunk_id") or "").strip()]

        if not valid_docs:
            return 0

        chunk_ids = [doc["chunk_id"].strip() for doc in valid_docs]

        try:

            existing_records = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(chunk_ids)).all()
            existing_map = {record.chunk_id: record for record in existing_records}

            redis_pipeline = redis_cache._get_client().pipeline()
            new_objects = []

            for doc in valid_docs:

                #record = db.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()
                chunk_id = doc["chunk_id"].strip()

                # 北京时间 UTC+8
                BJT = timezone(timedelta(hours=8))
                payload = {
                    "text": doc.get("text", ""),
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": int(doc.get("page_number", 0) or 0),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": int(doc.get("chunk_level", 0) or 0),
                    "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
                    "updated_at": datetime.now(BJT),
                }
                cache_payload = {
                    "chunk_id": chunk_id,
                    "text": payload["text"],
                    "filename": payload["filename"],
                    "file_type": payload["file_type"],
                    "file_path": payload["file_path"],
                    "page_number": payload["page_number"],
                    "parent_chunk_id": payload["parent_chunk_id"],
                    "root_chunk_id": payload["root_chunk_id"],
                    "chunk_level": payload["chunk_level"],
                    "chunk_idx": payload["chunk_idx"],
                }
                if chunk_id in existing_map:
                    for key, value in payload.items():
                        record = existing_map[chunk_id]
                        setattr(record, key, value)
                else:
                    new_objects.append(ParentChunk(chunk_id=chunk_id, **payload))
                    #db.add(ParentChunk(chunk_id=chunk_id, **payload))
                #redis_cache.set_json(self._cache_key(chunk_id), cache_payload)
                redis_pipeline.set(self._cache_key(chunk_id), json.dumps(cache_payload))
                upserted += 1

            if new_objects:
                db.bulk_save_objects(new_objects)

            db.commit()
            redis_pipeline.execute()
        finally:
            db.close()

        return upserted

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []

        ordered_results = {}
        missing_ids = []
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if not key:
                continue
            cached = redis_cache.get_json(self._cache_key(key))
            if cached:
                ordered_results[key] = cached
            else:
                missing_ids.append(key)

        if missing_ids:
            db = SessionLocal()
            try:
                rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(missing_ids)).all()
                for row in rows:
                    payload = self._to_dict(row)
                    ordered_results[row.chunk_id] = payload
                    redis_cache.set_json(self._cache_key(row.chunk_id), payload)
            finally:
                db.close()

        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除父级分块，返回删除条数。"""
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()
                for chunk_id in chunk_ids:
                    redis_cache.delete(self._cache_key(chunk_id))
            return deleted
        finally:
            db.close()

parent_chunk_store = ParentChunkStore()

