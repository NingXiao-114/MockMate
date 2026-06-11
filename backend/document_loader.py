import logging
import os
import re

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _parse_markdown_into_sections(markdown_text: str) -> list[dict]:
    heading_pattern = re.compile(r'^#{1,3}\s+.+', re.MULTILINE)
    matches = list(heading_pattern.finditer(markdown_text))

    if not matches:
        text = markdown_text.strip()
        return [{"text": text, "section_idx": 0}] if text else []

    sections = []
    if matches[0].start() > 0:
        preamble = markdown_text[:matches[0].start()].strip()
        if preamble:
            sections.append({"text": preamble, "section_idx": 0})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        section_text = markdown_text[start:end].strip()
        if section_text:
            sections.append({"text": section_text, "section_idx": len(sections)})

    return sections


class DocumentsLoader:
    """
    文档分片和加载
    """
    def __init__(self , chunk_size : int = 500 , chunk_overlap : int = 50):

        level_1_size = max(1200 , chunk_size * 2)
        level_1_overlap = max(240 , chunk_overlap * 2)

        level_2_size = max(600, chunk_size )
        level_2_overlap = max(120, chunk_overlap )

        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )


    @staticmethod
    def _build_chunk_id(file_name : str , page_number : int , level : int , index : int) -> str:
        return f"{file_name}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_level(
            self,
            text : str,
            base_doc : dict,
            page_global_chunk_idx : int
    ) -> list[dict]:
        if not text:
            return []

        root_chunks : list[dict] = []
        page_number = int(base_doc.get("page_number" ,  0))
        file_name = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text],[base_doc])

        level_1_counter =  0
        level_2_counter =  0
        level_3_counter =  0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(file_name, page_number, 1, level_1_counter)
            level_1_counter += 1
            level_1_chunk = {
                **base_doc,
                "text" : level_1_text,
                "chunk_id" : level_1_id,
                "parent_chunk_id" : "",
                "chunk_level" : 1 ,
                "chunk_idx" : page_global_chunk_idx
            }
            root_chunks.append(level_1_chunk)
            page_global_chunk_idx += 1

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(file_name, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(file_name, page_number, 3, level_3_counter)
                    level_3_counter += 1
                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks

    def _load_with_mineru(self, file_path: str, filename: str) -> list[dict]:
        from mineru_loader import parse_file_with_mineru

        markdown_text = parse_file_with_mineru(
            file_path=file_path,
            filename=filename,
            language=os.getenv("MINERU_LANGUAGE", "ch"),
            enable_table=os.getenv("MINERU_ENABLE_TABLE", "true").lower() == "true",
            enable_formula=os.getenv("MINERU_ENABLE_FORMULA", "true").lower() == "true",
            is_ocr=os.getenv("MINERU_IS_OCR", "false").lower() == "true",
            poll_timeout=int(os.getenv("MINERU_POLL_TIMEOUT", "300")),
        )

        file_lower = filename.lower()
        doc_type = "PDF" if file_lower.endswith(".pdf") else "Word"

        sections = _parse_markdown_into_sections(markdown_text)
        if not sections:
            raise ValueError("MinerU 返回内容为空")

        documents = []
        page_global_chunk_idx = 0
        for section in sections:
            base_doc = {
                "filename": filename,
                "file_path": file_path,
                "file_type": doc_type,
                "page_number": section["section_idx"],
            }
            chunks = self._split_page_to_three_level(
                text=section["text"],
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx,
            )
            page_global_chunk_idx += len(chunks)
            documents.extend(chunks)
        return documents

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """
        加载单个文档并分片
        :param file_path: 文件路径
        :param filename: 文件名
        :return: 分片后的文档列表
        """
        file_lower = filename.lower()
        is_excel = file_lower.endswith((".xlsx", ".xls"))

        if not is_excel:
            try:
                return self._load_with_mineru(file_path, filename)
            except Exception as e:
                logger.warning("MinerU 解析失败，降级到默认解析器: %s", e)

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        try:
            raw_docs = loader.load()
            documents = []
            page_global_chunk_idx = 0
            for doc in raw_docs:
                base_doc = {
                    "filename" : filename,
                    "file_path" : file_path,
                    "file_type" : doc_type,
                    "page_number" : page_global_chunk_idx,
                }
                page_chunks= self._split_page_to_three_level(
                    text=(doc.page_content or "").strip(),
                    base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx
                )
                page_global_chunk_idx += len(page_chunks)
                documents.extend(page_chunks)
            return documents
        except Exception as e:
            raise Exception(f"处理文件{filename}失败:{str(e)}")

    def load_document_text(self, file_path: str, filename: str) -> str:
        """加载文档并返回完整文本，不做分块。用于临时附件。"""
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        raw_docs = loader.load()
        return "\n\n".join((doc.page_content or "").strip() for doc in raw_docs if doc.page_content)

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """
        从文件夹加载所有文档并分片
        :param folder_path: 文件夹路径
        :return: 所有分片后的文档列表
        """
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc")) or file_lower.endswith((".xlsx", ".xls"))):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue

        return all_documents
