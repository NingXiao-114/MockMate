"""测试 DocumentsLoader：加载 PDF 并验证三级分片结果"""

import os
import sys
import time

from ..document_loader import DocumentsLoader

FILE_DIR = os.path.join(os.path.dirname(__file__), "files")


def test_load_single_pdf():
    print("=" * 60)
    print("[测试] 加载单个 PDF 文件")
    print("=" * 60)

    filename = "AIGC时代的学术出版伦理：风险挑战与治理路径_葛建平.pdf"
    file_path = os.path.join(FILE_DIR, filename)

    assert os.path.isfile(file_path), f"文件不存在: {file_path}"
    print(f"  文件路径: {file_path}")

    loader = DocumentsLoader(chunk_size=500, chunk_overlap=50)
    print(f"  分片参数: chunk_size=500, chunk_overlap=50")
    print(f"  三级分片尺寸: L1={loader._splitter_level_1._chunk_size}, "
          f"L2={loader._splitter_level_2._chunk_size}, "
          f"L3={loader._splitter_level_3._chunk_size}")

    t0 = time.time()
    docs = loader.load_document(file_path, filename)
    elapsed = time.time() - t0
    print(f"  加载耗时: {elapsed:.2f}s")
    print(f"  总分片数: {len(docs)}")

    level_counts = {1: 0, 2: 0, 3: 0}
    for d in docs:
        level_counts[d["chunk_level"]] += 1
    print(f"  各级别分片数: L1={level_counts[1]}, L2={level_counts[2]}, L3={level_counts[3]}")

    assert len(docs) > 0, "分片结果为空"
    assert level_counts[1] > 0, "缺少 L1 分片"

    sample = docs[0]
    print(f"\n  --- 第一个分片示例 ---")
    for key in ["chunk_id", "chunk_level", "file_name", "file_type", "page_number"]:
        print(f"    {key}: {sample.get(key)}")
    print(f"    text (前120字): {sample['text'][:120]}...")

    print("\n  --- 验证层级关系 ---")
    l2_with_parent = [d for d in docs if d["chunk_level"] == 2]
    l3_with_parent = [d for d in docs if d["chunk_level"] == 3]

    all_chunk_ids = {d["chunk_id"] for d in docs}
    if l2_with_parent:
        ok = all(d["parent_chunk_id"] in all_chunk_ids for d in l2_with_parent)
        print(f"    L2 → L1 父节点引用有效: {ok}")
        assert ok, "L2 的 parent_chunk_id 指向了不存在的 L1"
    if l3_with_parent:
        ok = all(d["parent_chunk_id"] in all_chunk_ids for d in l3_with_parent)
        print(f"    L3 → L2 父节点引用有效: {ok}")
        assert ok, "L3 的 parent_chunk_id 指向了不存在的 L2"

    print("\n[通过] 单文件加载测试\n")


def test_load_folder():
    print("=" * 60)
    print("[测试] 从文件夹加载所有文档")
    print("=" * 60)

    loader = DocumentsLoader(chunk_size=500, chunk_overlap=50)
    print(f"  文件夹: {FILE_DIR}")

    t0 = time.time()
    docs = loader.load_documents_from_folder(FILE_DIR)
    elapsed = time.time() - t0
    print(f"  加载耗时: {elapsed:.2f}s")
    print(f"  总分片数: {len(docs)}")

    file_names = set(d["file_name"] for d in docs)
    print(f"  涉及文件: {file_names}")

    assert len(docs) > 0, "文件夹加载结果为空"
    print("\n[通过] 文件夹加载测试\n")


def test_empty_text():
    print("=" * 60)
    print("[测试] 空文本分片")
    print("=" * 60)

    loader = DocumentsLoader()
    result = loader._split_page_to_three_level(
        text="",
        base_doc={"file_name": "test.pdf", "page_number": 0},
        page_global_chunk_idx=0,
    )
    print(f"  空文本分片结果数: {len(result)}")
    assert result == [], "空文本应返回空列表"
    print("[通过] 空文本测试\n")


if __name__ == "__main__":
    test_load_single_pdf()
    test_load_folder()
    test_empty_text()
    print("=" * 60)
    print("所有 DocumentsLoader 测试通过!")
    print("=" * 60)
