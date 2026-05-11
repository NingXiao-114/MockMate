import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


MINERU_BASE_URL = "https://mineru.net/api/v1/agent"
DEFAULT_LANGUAGE = "ch"
DEFAULT_PAGE_RANGE = None
DEFAULT_ENABLE_TABLE = True
DEFAULT_IS_OCR = False
DEFAULT_ENABLE_FORMULA = True

REQUEST_TIMEOUT_SECONDS = 60
UPLOAD_TIMEOUT_SECONDS = 300
POLL_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 3

__all__ = ["parse_file_with_mineru"]


def _resolve_file_path(file_path: str) -> Path:
    resolved_path = Path(file_path).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"文件不存在: {resolved_path}")
    return resolved_path


def _create_parse_task(
    file_path: Path,
    filename: str,
    language: str,
    page_range: Optional[str],
    enable_table: bool,
    is_ocr: bool,
    enable_formula: bool,
) -> Dict[str, Any]:
    payload = {
        "file_name": filename,
        "language": language,
        "enable_table": enable_table,
        "is_ocr": is_ocr,
        "enable_formula": enable_formula,
    }
    if page_range:
        payload["page_range"] = page_range

    response = requests.post(
        f"{MINERU_BASE_URL}/parse/file",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()

    if result.get("code") != 0:
        raise RuntimeError(f"MinerU 获取上传链接失败: {result.get('msg', result)}")

    data = result.get("data") or {}
    if not data.get("task_id") or not data.get("file_url"):
        raise RuntimeError(f"MinerU 返回中缺少 task_id 或 file_url: {result}")
    return result


def _upload_file_to_oss(file_path: Path, file_url: str) -> None:
    with file_path.open("rb") as file_obj:
        response = requests.put(file_url, data=file_obj, timeout=UPLOAD_TIMEOUT_SECONDS)

    if response.status_code not in (200, 201):
        raise RuntimeError(f"MinerU 文件上传失败, HTTP {response.status_code}: {response.text}")


def _get_task_result(task_id: str) -> Dict[str, Any]:
    response = requests.get(f"{MINERU_BASE_URL}/parse/{task_id}", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    result = response.json()

    if result.get("code") not in (None, 0):
        raise RuntimeError(f"MinerU 查询解析结果失败: {result.get('msg', result)}")
    return result


def _download_markdown(markdown_url: str) -> str:
    response = requests.get(markdown_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _poll_markdown_result(task_id: str, timeout: int, interval: int) -> str:
    start = time.time()
    last_result = None

    while time.time() - start < timeout:
        result = _get_task_result(task_id)
        last_result = result

        data = result.get("data") or {}
        state = data.get("state")

        if state == "done":
            markdown_url = data.get("markdown_url")
            if not markdown_url:
                raise RuntimeError(f"MinerU 解析完成但返回中缺少 markdown_url: {result}")
            return _download_markdown(markdown_url)

        if state == "failed":
            raise RuntimeError(f"MinerU 解析失败: {data.get('err_msg', '未知错误')}")

        time.sleep(interval)

    raise TimeoutError(f"MinerU 轮询超时 ({timeout}s)，task_id: {task_id}, last_result: {last_result}")


def parse_file_with_mineru(
    file_path: str,
    filename: str,
    language: str = DEFAULT_LANGUAGE,
    page_range: Optional[str] = DEFAULT_PAGE_RANGE,
    enable_table: bool = DEFAULT_ENABLE_TABLE,
    is_ocr: bool = DEFAULT_IS_OCR,
    enable_formula: bool = DEFAULT_ENABLE_FORMULA,
    poll_timeout: int = POLL_TIMEOUT_SECONDS,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> str:
    """上传文件到 MinerU，并返回解析后的 Markdown 内容。参数格式对齐 DocumentsLoader.load_document。"""
    resolved_file_path = _resolve_file_path(file_path)
    safe_filename = Path(filename or "").name.strip()
    if not safe_filename:
        raise ValueError("filename 不能为空")

    task_result = _create_parse_task(
        file_path=resolved_file_path,
        filename=safe_filename,
        language=language,
        page_range=page_range,
        enable_table=enable_table,
        is_ocr=is_ocr,
        enable_formula=enable_formula,
    )

    data = task_result["data"]
    _upload_file_to_oss(resolved_file_path, data["file_url"])
    return _poll_markdown_result(data["task_id"], timeout=poll_timeout, interval=poll_interval)
