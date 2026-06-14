import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests


TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2NjMwMDM2MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3OTc3NzY5MCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiNWI1NTdkZTMtYTFiNS00NGVkLWFkM2YtN2VkYzliNzM1NWYwIiwiZW1haWwiOiIiLCJleHAiOjE3ODc1NTM2OTB9.xajEv1blIGW75gkmQD664fpV2fesRRdKoDJRDbnz0-9MsWvJHGMoAvA6lqp4-c87ylQVkvOKRhUzWOu13nvtBA"
BASE_URL = "https://mineru.net/api/v4"

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_FILE_PATH = PROJECT_ROOT / "data" / "documents" / "简历.pdf"
RESULT_ROOT = PROJECT_ROOT / "data" / "parse_results"

REQUEST_TIMEOUT_SECONDS = 60
UPLOAD_TIMEOUT_SECONDS = 300
POLL_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 3


def print_step(message: str) -> None:
    print(f"\n========== {message} ==========")


def print_json(title: str, data: Any) -> None:
    print(f"{title}:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def parse_response_json(response: requests.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}


def raise_for_api_error(response: requests.Response, action: str) -> Dict[str, Any]:
    result = parse_response_json(response)
    if response.status_code == 401:
        raise RuntimeError(f"{action}失败：MinerU token 未通过认证。HTTP 401, 返回: {result}")
    if response.status_code >= 400:
        raise RuntimeError(f"{action}失败：HTTP {response.status_code}, 返回: {result}")
    return result


def headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    }


def request_upload_url(file_path: Path) -> Dict[str, Any]:
    print_step("1. 申请 v4 专业解析临时上传 URL")

    data = {
        "files": [
            {
                "name": file_path.name,
                "data_id": file_path.stem,
                "is_ocr": False,
            }
        ],
        "model_version": "vlm",
        "language": "ch",
        "enable_table": True,
        "enable_formula": True,
    }

    url = f"{BASE_URL}/file-urls/batch"
    print(f"请求地址: {url}")
    print_json("请求参数", data)

    response = requests.post(url, headers=headers(), json=data, timeout=REQUEST_TIMEOUT_SECONDS)
    print(f"HTTP 状态码: {response.status_code}")
    result = raise_for_api_error(response, "申请上传 URL")
    print_json("申请上传 URL 返回", result)

    if result.get("code") != 0:
        raise RuntimeError(f"申请上传 URL 失败: {result.get('msg', result)}")

    payload = result.get("data") or {}
    if not payload.get("batch_id") or not payload.get("file_urls"):
        raise RuntimeError(f"返回中缺少 batch_id 或 file_urls: {result}")

    print(f"batch_id: {payload['batch_id']}")
    print(f"临时上传 URL: {payload['file_urls'][0]}")
    return result


def upload_file(file_path: Path, upload_url: str) -> None:
    print_step("2. PUT 上传本地文件到临时 URL")
    print(f"本地文件: {file_path}")
    print(f"文件大小: {file_path.stat().st_size} bytes")

    with file_path.open("rb") as file_obj:
        response = requests.put(upload_url, data=file_obj, timeout=UPLOAD_TIMEOUT_SECONDS)

    print(f"上传 HTTP 状态码: {response.status_code}")
    if response.text:
        print(f"上传返回内容: {response.text}")

    if response.status_code not in (200, 201):
        raise RuntimeError(f"文件上传失败, HTTP {response.status_code}: {response.text}")

    print("文件上传成功，等待 MinerU 自动提交解析任务...")


def get_batch_result(batch_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"
    response = requests.get(url, headers=headers(), timeout=REQUEST_TIMEOUT_SECONDS)
    print(f"查询地址: {url}")
    print(f"HTTP 状态码: {response.status_code}")
    result = raise_for_api_error(response, "查询解析结果")
    print_json("查询返回", result)
    return result


def _extract_items(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    data = result.get("data") or {}
    for key in ("extract_result", "extract_results", "results", "files"):
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
    return [data] if isinstance(data, dict) else []


def _extract_state(item: Dict[str, Any]) -> Optional[str]:
    for key in ("state", "status"):
        if item.get(key):
            return str(item[key]).lower()

    progress = item.get("extract_progress")
    if isinstance(progress, dict):
        for key in ("state", "status"):
            if progress.get(key):
                return str(progress[key]).lower()
    return None


def _find_result_urls(item: Dict[str, Any]) -> Dict[str, Any]:
    urls = {}
    for key in (
        "full_zip_url",
        "markdown_url",
        "layout_url",
        "middle_json_url",
        "content_list_url",
        "model_json_url",
        "docx_url",
        "html_url",
    ):
        if item.get(key):
            urls[key] = item[key]

    extract_result = item.get("extract_result")
    if isinstance(extract_result, dict):
        urls.update(_find_result_urls(extract_result))
    return urls


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zip_file:
        target_root = target_dir.resolve()
        for member in zip_file.infolist():
            member_path = (target_dir / member.filename).resolve()
            if target_root != member_path and target_root not in member_path.parents:
                raise RuntimeError(f"zip 中存在非法路径: {member.filename}")
        zip_file.extractall(target_dir)


def download_result_zip(result_url: str, source_file_path: Path) -> Path:
    print_step("4. 下载并保存解析结果")

    output_dir = RESULT_ROOT / source_file_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{source_file_path.stem}.zip"

    print(f"结果 URL: {result_url}")
    print(f"保存目录: {output_dir}")
    print(f"zip 文件: {zip_path}")

    with requests.get(result_url, stream=True, timeout=UPLOAD_TIMEOUT_SECONDS) as response:
        print(f"下载 HTTP 状态码: {response.status_code}")
        response.raise_for_status()
        with zip_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)

    print(f"zip 下载完成，大小: {zip_path.stat().st_size} bytes")
    _safe_extract_zip(zip_path, output_dir)
    print(f"zip 已解压到: {output_dir}")
    return output_dir


def poll_batch_result(batch_id: str) -> Dict[str, Any]:
    print_step("3. 轮询 v4 专业解析结果")
    start = time.time()
    last_result: Optional[Dict[str, Any]] = None

    while time.time() - start < POLL_TIMEOUT_SECONDS:
        elapsed = int(time.time() - start)
        print(f"\n[{elapsed}s] 查询 batch_id: {batch_id}")

        result = get_batch_result(batch_id)
        last_result = result
        items = _extract_items(result)
        states = [_extract_state(item) for item in items]
        print(f"当前状态: {states}")

        if any(state == "failed" for state in states):
            raise RuntimeError(f"解析失败: {result}")

        if states and all(state == "done" for state in states):
            print_step("解析完成")
            print_json("最终返回结果", result)
            for item in items:
                urls = _find_result_urls(item)
                if urls:
                    print_json("结果下载链接", urls)
            return result

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"轮询超时 ({POLL_TIMEOUT_SECONDS}s), last_result={last_result}")


def main() -> int:
    try:
        file_path = DEFAULT_FILE_PATH
        print(f"项目根目录: {PROJECT_ROOT}")
        print(f"测试文件路径: {file_path}")
        print(f"文件是否存在: {file_path.exists()}")
        print(f"是否为普通文件: {file_path.is_file()}")

        if not file_path.is_file():
            print("测试终止：文件不存在或不是普通文件")
            return 1

        upload_info = request_upload_url(file_path)
        data = upload_info["data"]
        upload_file(file_path, data["file_urls"][0])
        result = poll_batch_result(data["batch_id"])

        items = _extract_items(result)
        if not items:
            raise RuntimeError(f"解析完成但返回中没有 extract_result: {result}")

        urls = _find_result_urls(items[0])
        full_zip_url = urls.get("full_zip_url")
        if not full_zip_url:
            raise RuntimeError(f"解析完成但返回中没有 full_zip_url: {result}")

        output_dir = download_result_zip(full_zip_url, file_path)
        print_step("流程完成")
        print(f"解析结果目录: {output_dir}")
        return 0
    except Exception as exc:
        print_step("测试失败")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
