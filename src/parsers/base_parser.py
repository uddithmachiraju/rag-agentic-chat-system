import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.logging import LoggerMixin
from ..models.document import (
    ChunkType,
    DocumentChunk,
    DocumentMetadata,
    ProcessingOptions,
)


class ParseResult:
    def __init__(
        self,
        success: bool,
        chunks: List[DocumentChunk] = None,
        metadata: DocumentMetadata = None,
        error_message: str = None,
        processing_time: float = 0.0
    ):
        self.success = success
        self.chunks = chunks or []
        self.metadata = metadata or DocumentMetadata()
        self.error_message = error_message
        self.processing_time = processing_time


class BaseParser(ABC, LoggerMixin):
    def __init__(self):
        self.supported_extensions: List[str] = []
        self.mime_types: List[str] = []

    @abstractmethod
    async def parse(self, file_path: str, document_id: str, options: ProcessingOptions = None) -> ParseResult:
        pass

    @abstractmethod
    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        pass

    def _get_file_extension(self, file_path: str) -> str:
        return Path(file_path).suffix.lower().lstrip(".")

    def _validate_file(self, file_path: str) -> bool:
        try:
            file_path_obj = Path(file_path)
            return file_path_obj.exists() and file_path_obj.is_file()
        except Exception as e:
            self.logger.error(f"File validation error: {e}")
            return False

    async def _preprocess_content(self, content: str) -> str:
        if not content:
            return ""

        # Normalize newlines
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Dehyphenate words split at line endings (e.g., "exam-\nple" -> "example")
        content = re.sub(r'-\n\s*', '', content)

        # Collapse multiple blank lines to two newlines (paragraph separation)
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Trim whitespace on each line but preserve paragraph breaks
        lines = [ln.strip() for ln in content.split('\n')]
        paragraphs = []
        cur_para = []

        list_pattern = re.compile(
            r'^([\-\*\u2022\u25CF\u25CB\u25A0\u25A1]|\d+[\.)]|\([a-zA-Z0-9]+\))\s+')
        header_pattern = re.compile(r'^[A-Z][A-Za-z0-9\s]{0,80}$')

        for ln in lines:
            if ln == '':
                if cur_para:
                    paragraphs.append(' '.join(cur_para))
                    cur_para = []
            else:
                # If line looks like a list item, flush current paragraph and add as its own paragraph
                if list_pattern.match(ln):
                    if cur_para:
                        paragraphs.append(' '.join(cur_para))
                        cur_para = []
                    paragraphs.append(ln)
                # If line looks like a short header (Title Case or ALL CAPS), treat as its own paragraph
                elif header_pattern.match(ln) and len(ln.split()) <= 8:
                    if cur_para:
                        paragraphs.append(' '.join(cur_para))
                        cur_para = []
                    paragraphs.append(ln)
                else:
                    cur_para.append(ln)

        if cur_para:
            paragraphs.append(' '.join(cur_para))

        # Normalize bullets and control characters inside paragraphs
        cleaned_paragraphs = []
        for p in paragraphs:
            # Replace common bullet characters with a hyphen + space
            p = re.sub(r'[•●○■□▪▫]', '-', p)
            # Normalize various dashes to simple hyphen
            p = re.sub(r'[–—−]', '-', p)
            # Remove non-printable/control chars
            p = re.sub(r'[\x00-\x1F\x7F]', ' ', p)
            # Collapse inner whitespace
            p = re.sub(r'\s+', ' ', p).strip()
            cleaned_paragraphs.append(p)

        return '\n\n'.join(cleaned_paragraphs)

    def _clean_text(self, content: str) -> str:
        """Synchronous cleaning helper used by parsers when async preprocessing isn't convenient."""
        if not content:
            return ""
        # Reuse the async preprocessing logic synchronously
        # Keep it simple and safe: perform similar operations inline
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        content = re.sub(r'-\n\s*', '', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        lines = [ln.strip() for ln in content.split('\n')]
        paragraphs = []
        cur_para = []

        list_pattern = re.compile(
            r'^([\-\*\u2022\u25CF\u25CB\u25A0\u25A1]|\d+[\.)]|\([a-zA-Z0-9]+\))\s+')
        header_pattern = re.compile(r'^[A-Z][A-Za-z0-9\s]{0,80}$')

        for ln in lines:
            if ln == '':
                if cur_para:
                    paragraphs.append(' '.join(cur_para))
                    cur_para = []
            else:
                if list_pattern.match(ln):
                    if cur_para:
                        paragraphs.append(' '.join(cur_para))
                        cur_para = []
                    paragraphs.append(ln)
                elif header_pattern.match(ln) and len(ln.split()) <= 8:
                    if cur_para:
                        paragraphs.append(' '.join(cur_para))
                        cur_para = []
                    paragraphs.append(ln)
                else:
                    cur_para.append(ln)

        if cur_para:
            paragraphs.append(' '.join(cur_para))

        cleaned_paragraphs = []
        for p in paragraphs:
            p = re.sub(r'[•●○■□▪▫]', '-', p)
            p = re.sub(r'[–—−]', '-', p)
            p = re.sub(r'[\x00-\x1F\x7F]', ' ', p)
            p = re.sub(r'\s+', ' ', p).strip()
            cleaned_paragraphs.append(p)

        return '\n\n'.join(cleaned_paragraphs)

    def _create_chunk(self, content: str, chunk_index: int, chunk_type: str = "text", page_number: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None,
                      document_id: Optional[str] = None, start_char: Optional[int] = None, end_char: Optional[int] = None, heading_level: Optional[int] = None, parent_section: Optional[str] = None
                      ) -> DocumentChunk:
        if document_id is None:
            raise ValueError("document_id is required")

        return DocumentChunk(
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            chunk_type=ChunkType(chunk_type),
            page_number=page_number,
            start_char=start_char,
            end_char=end_char,
            heading_level=heading_level,
            parent_section=parent_section,
            metadata=metadata or {}
        )
