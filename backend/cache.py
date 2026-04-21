import json
import os
from typing import Any, Optional

import redis


class RedisCache:
    def __init__(self):
            self.redis_url = os.getenv('REDIS_URL' , "redis://localhost:6379/0" )
            self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "ragbot")
            self.default_ttl = os.getenv("REDIS_DEFAULT_TTL", "300")
            self.redis_client = None

    def _get_client(self):
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url ,decode_responses=True)
        return self.redis_client

    def _key(self , key : str):
        return f"{self.key_prefix}:{key}"


    def get_json(self, key : str):
        try:
            value = self._get_client().get(self._key(key))
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            return None


    def set_json(self, key : str, value : Any , ttl : Optional[int] = None):
        try:
            payload = json.dumps(value ,ensure_ascii = False)
            self._get_client().setex(self._key(key), ttl or self.default_ttl , payload)
        except Exception as e:
            return None

    def delete(self, key: str) -> None:
        try:
            self._get_client().delete(self._key(key))
        except Exception:
            return

    def delete_pattern(self, pattern: str) -> None:
        try:
            full_pattern = self._key(pattern)
            keys = self._get_client().keys(full_pattern)
            if keys:
                self._get_client().delete(*keys)
        except Exception:
            return None

redis_cache = RedisCache()