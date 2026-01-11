import hashlib
import re
import time
from typing import List, Optional, Tuple

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.parsers.base_parser import BaseParser, ParseResult

from ..config import get_settings
from ..models.document import DocumentMetadata, ProcessingOptions


class PDFParsingError(Exception):
    pass


class PDFParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['pdf']
        self.mime_types = ['application/pdf']
        self.settings = get_settings()

        # Initialize LangChain text splitter with smart defaults
        self.text_splitter = None  # Will be initialized per request based on options

    def _get_text_splitter(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize RecursiveCharacterTextSplitter with smart separators."""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple newlines (section breaks)
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence ends
                "! ",      # Exclamation
                "? ",      # Question
                "; ",      # Semicolon
                ": ",      # Colon
                ", ",      # Comma
                " ",       # Space
                ""         # Character level (last resort)
            ],
            keep_separator=True,
            is_separator_regex=False
        )

    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        extension = self._get_file_extension(file_path)
        if extension in self.supported_extensions:
            return True
        if mime_type and mime_type in self.mime_types:
            return True
        return False

    async def parse(self, file_path: str, document_id: str, options: ProcessingOptions = None) -> ParseResult:
        start_time = time.time()

        try:
            if not self._validate_file(file_path):
                return ParseResult(
                    success=False,
                    error_message=f"File not found or not readable: {file_path}"
                )

            options = options or ProcessingOptions()
            
            # Initialize text splitter based on options
            self.text_splitter = self._get_text_splitter(
                chunk_size=options.chunk_size if options.chunk_size > 0 else 1000,
                chunk_overlap=options.chunk_overlap
            )

            self.logger.info(f"Starting improved PDF parsing: {file_path}")

            doc = await self._open_pdf_safely(file_path)
            if not doc:
                return ParseResult(
                    success=False,
                    error_message="Failed to open PDF document"
                )

            try:
                # Extract metadata
                metadata = await self._extract_metadata(doc, file_path)

                # Extract and clean content
                all_chunks = []
                global_chunk_index = 0

                for page_num in range(doc.page_count):
                    page_chunks, global_chunk_index = await self._extract_page_content(
                        doc, page_num, document_id, options, global_chunk_index
                    )
                    all_chunks.extend(page_chunks)

                processing_time = time.time() - start_time

                self.logger.info(
                    f"PDF parsing completed: {len(all_chunks)} chunks extracted "
                    f"in {processing_time:.2f}s"
                )

                return ParseResult(
                    success=True,
                    chunks=all_chunks,
                    metadata=metadata,
                    processing_time=processing_time
                )

            finally:
                doc.close()

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"PDF parsing failed: {str(e)}"
            self.logger.error(error_msg)

            return ParseResult(
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )

    def _deep_clean_text(self, text: str) -> str:
        """
        Aggressive text cleaning to remove artifacts, special characters, and formatting noise.
        """
        if not text:
            return ""

        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Remove excessive punctuation patterns (e.g., ".....", "????")
        text = re.sub(r'([.!?])\1{2,}', r'\1', text)
        
        # Remove standalone special characters that are common artifacts
        # Keep them if they're part of actual content
        text = re.sub(r'\s+[|•●○■□▪▫–—]+\s+', ' ', text)
        
        # Clean up table artifacts (multiple pipes, asterisks)
        text = re.sub(r'\|+', '|', text)  # Multiple pipes to single
        text = re.sub(r'\s+\|\s+\|\s+', ' | ', text)  # Clean pipe spacing
        text = re.sub(r'^\||\|$', '', text, flags=re.MULTILINE)  # Remove leading/trailing pipes
        
        # Remove excessive asterisks (common in headers/footers)
        text = re.sub(r'\*{3,}', '', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[\uf0b7\uf0a7\uf076\uf0d8]', '', text)  # Unicode bullet artifacts
        
        # Clean up quotation marks
        text = re.sub(r'[''‚‛]', "'", text)  # Smart single quotes to straight
        text = re.sub(r'[""„‟]', '"', text)  # Smart double quotes to straight
        
        # Remove excessive quotes/apostrophes
        text = re.sub(r"'{3,}", "'", text)
        text = re.sub(r'"{3,}', '"', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])\s*\n', r'\1\n', text)  # Clean punctuation at line ends
        
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove multiple newlines (but keep paragraph breaks)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Remove lines that are just special characters
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that are mostly special characters
            if line.strip() and not re.match(r'^[^\w\s]{1,}$', line.strip()):
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        # Remove common header/footer patterns
        text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers
        
        # Final cleanup
        text = text.strip()
        
        # Remove any remaining excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text

    def _is_meaningful_content(self, text: str, min_words: int = 3) -> bool:
        """
        Check if text contains meaningful content worth keeping.
        """
        if not text or not text.strip():
            return False
        
        # Remove special characters for word counting
        words = re.findall(r'\b\w+\b', text)
        
        if len(words) < min_words:
            return False
        
        # Check if it's mostly special characters
        alphanumeric = sum(c.isalnum() for c in text)
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return False
            
        # At least 30% should be alphanumeric
        if alphanumeric / total_chars < 0.3:
            return False
        
        return True

    async def _extract_page_content(
        self,
        doc: fitz.Document,
        page_num: int,
        document_id: str,
        options: ProcessingOptions,
        chunk_index: int
    ) -> Tuple[List, int]:
        """Extract and intelligently chunk content from a page."""
        
        page_chunks = []
        page = doc[page_num]

        try:
            # Extract text with structure
            text_content = await self._extract_clean_text(page, page_num)
            
            if not text_content or not self._is_meaningful_content(text_content):
                self.logger.warning(f"No meaningful content on page {page_num + 1}")
                return page_chunks, chunk_index

            # Use LangChain splitter for intelligent chunking
            if options.chunk_size > 0:
                chunks = self.text_splitter.split_text(text_content)
            else:
                # No chunking, use entire page
                chunks = [text_content]

            # Create chunk objects
            char_offset = 0
            for chunk_text in chunks:
                # Final cleaning pass
                chunk_text = self._deep_clean_text(chunk_text)
                
                if not self._is_meaningful_content(chunk_text):
                    continue

                chunk = self._create_chunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    chunk_type="text",
                    page_number=page_num + 1,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    metadata={
                        "extraction_method": "langchain_recursive",
                        "page_width": page.rect.width,
                        "page_height": page.rect.height,
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text)
                    },
                    document_id=document_id
                )
                page_chunks.append(chunk)
                chunk_index += 1
                char_offset += len(chunk_text)

            # Extract tables if requested
            if options.extract_tables:
                table_chunks, chunk_index = await self._extract_page_tables(
                    page, page_num, document_id, chunk_index
                )
                page_chunks.extend(table_chunks)

            # Extract images if requested
            if options.extract_images:
                image_chunks, chunk_index = await self._extract_page_images(
                    page, page_num, document_id, chunk_index
                )
                page_chunks.extend(image_chunks)

        except Exception as e:
            self.logger.error(f"Error extracting content from page {page_num + 1}: {e}")
            # Create error placeholder
            error_chunk = self._create_chunk(
                document_id=document_id,
                content=f"[Error extracting content from page {page_num + 1}]",
                chunk_index=chunk_index,
                chunk_type="text",
                page_number=page_num + 1,
                start_char=0,
                end_char=0,
                metadata={"extraction_error": True, "error_message": str(e)}
            )
            page_chunks.append(error_chunk)
            chunk_index += 1

        return page_chunks, chunk_index

    async def _extract_clean_text(self, page: fitz.Page, page_num: int) -> str:
        """Extract text with multiple methods and aggressive cleaning."""
        
        text_content = ""
        
        try:
            # Method 1: Structured extraction with blocks
            blocks = page.get_text("dict")
            text_content = self._extract_from_blocks(blocks)
            
            if not text_content.strip():
                # Method 2: Simple text extraction
                text_content = page.get_text()
            
            if not text_content.strip():
                # Method 3: Raw text
                text_content = page.get_text("text")
            
            # Deep clean the extracted text
            text_content = self._deep_clean_text(text_content)
            
        except Exception as e:
            self.logger.error(f"Error extracting text from page {page_num + 1}: {e}")
        
        return text_content

    def _extract_from_blocks(self, blocks_dict: dict) -> str:
        """Extract text from PyMuPDF blocks with structure preservation."""
        
        text_parts = []
        
        try:
            if "blocks" not in blocks_dict:
                return ""

            for block in blocks_dict["blocks"]:
                if block.get("type") == 0:  # Text block
                    block_text = self._extract_block_text(block)
                    if block_text and self._is_meaningful_content(block_text):
                        text_parts.append(block_text)

            # Join with paragraph breaks
            text = "\n\n".join(text_parts)
            
        except Exception as e:
            self.logger.debug(f"Error extracting from blocks: {e}")
            return ""
        
        return text

    def _extract_block_text(self, block: dict) -> str:
        """Extract text from a block while preserving line structure."""
        
        lines = []
        
        try:
            if "lines" not in block:
                return ""

            for line in block["lines"]:
                if "spans" not in line:
                    continue

                line_text = ""
                for span in line["spans"]:
                    if "text" in span:
                        line_text += span["text"]

                if line_text.strip():
                    lines.append(line_text.strip())

            # Join lines with single newline
            block_text = "\n".join(lines)
            
        except Exception as e:
            self.logger.debug(f"Error extracting block text: {e}")
            return ""
        
        return block_text

    async def _extract_page_tables(
        self,
        page: fitz.Page,
        page_num: int,
        document_id: str,
        chunk_index: int
    ) -> Tuple[List, int]:
        """Extract and clean table content."""
        
        chunks = []

        try:
            tables = []
            if hasattr(page, 'find_tables'):
                try:
                    tables = page.find_tables() or []
                except Exception:
                    tables = []

            for table_index, table in enumerate(tables):
                try:
                    table_data = []
                    if hasattr(table, 'extract'):
                        table_data = table.extract() or []

                    if not table_data:
                        continue

                    table_text = self._format_table_clean(table_data)
                    table_text = self._deep_clean_text(table_text)

                    if self._is_meaningful_content(table_text, min_words=5):
                        chunk = self._create_chunk(
                            document_id=document_id,
                            content=table_text,
                            chunk_index=chunk_index,
                            chunk_type="table",
                            page_number=page_num + 1,
                            start_char=0,
                            end_char=len(table_text),
                            metadata={
                                "table_index": table_index,
                                "rows": len(table_data),
                                "columns": len(table_data[0]) if table_data else 0,
                                "extraction_method": "pymupdf_tables"
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                except Exception as e:
                    self.logger.debug(f"Error extracting table {table_index} on page {page_num + 1}: {e}")

        except Exception as e:
            self.logger.error(f"Error finding tables on page {page_num + 1}: {e}")

        return chunks, chunk_index

    def _format_table_clean(self, table_data: List[List[str]]) -> str:
        """Format table with proper cleaning."""
        
        if not table_data:
            return ""

        # Filter empty rows
        filtered_data = []
        for row in table_data:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            if any(cleaned_row):  # Keep row if any cell has content
                filtered_data.append(cleaned_row)

        if not filtered_data:
            return ""

        # Convert to markdown-style table for better readability
        lines = []
        
        # Header row
        if filtered_data:
            header = " | ".join(filtered_data[0])
            lines.append(header)
            lines.append("-" * len(header))  # Separator
        
        # Data rows
        for row in filtered_data[1:]:
            lines.append(" | ".join(row))

        return "\n".join(lines)

    async def _extract_page_images(
        self,
        page: fitz.Page,
        page_num: int,
        document_id: str,
        chunk_index: int
    ) -> Tuple[List, int]:
        """Extract image metadata and context."""
        
        chunks = []

        try:
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    image_hash = hashlib.md5(image_bytes).hexdigest()
                    
                    # Extract surrounding context
                    context = self._extract_image_context(page, img)
                    
                    image_description = f"[Image {img_index + 1}]"
                    if context:
                        image_description += f"\n{context}"

                    chunk = self._create_chunk(
                        document_id=document_id,
                        content=image_description,
                        chunk_index=chunk_index,
                        chunk_type="image",
                        page_number=page_num + 1,
                        start_char=0,
                        end_char=len(image_description),
                        metadata={
                            "image_index": img_index,
                            "image_hash": image_hash,
                            "image_format": image_ext,
                            "image_size": len(image_bytes)
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                except Exception as e:
                    self.logger.error(f"Error extracting image {img_index} from page {page_num + 1}: {e}")

        except Exception as e:
            self.logger.error(f"Error extracting images from page {page_num + 1}: {e}")

        return chunks, chunk_index

    def _extract_image_context(self, page: fitz.Page, img_info: tuple) -> str:
        """Extract text context around images."""
        
        try:
            page_text = page.get_text()
            
            caption_patterns = [
                r'Figure\s+\d+[:\.].*',
                r'Fig\.\s+\d+[:\.].*',
                r'Image\s+\d+[:\.].*',
                r'Table\s+\d+[:\.].*'
            ]

            for pattern in caption_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    return self._deep_clean_text(matches[0][:200])

            return ""

        except Exception:
            return ""

    async def _open_pdf_safely(self, file_path: str) -> Optional[fitz.Document]:
        """Safely open PDF document."""
        try:
            doc = fitz.open(file_path)
            if doc.is_closed or doc.page_count == 0:
                return None
            return doc
        except Exception as e:
            self.logger.error(f"Failed to open PDF {file_path}: {e}")
            return None

    async def _extract_metadata(self, doc: fitz.Document, file_path: str) -> DocumentMetadata:
        """Extract PDF metadata."""
        try:
            pdf_meta = doc.metadata

            metadata = DocumentMetadata(
                title=pdf_meta.get('title', '').strip(),
                author=pdf_meta.get('author', '').strip(),
                subject=pdf_meta.get('subject', '').strip(),
                page_count=doc.page_count
            )

            # Extract creation date
            if pdf_meta.get('creationDate'):
                try:
                    from datetime import datetime
                    date_str = pdf_meta['creationDate']
                    if date_str.startswith('D:'):
                        date_str = date_str[2:16]
                        metadata.created_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                except Exception as e:
                    self.logger.debug(f"Failed to parse creation date: {e}")

            # Count words and characters
            try:
                total_chars = 0
                total_words = 0
                for p in range(min(doc.page_count, 10)):  # Sample first 10 pages
                    page = doc[p]
                    text = page.get_text() or ""
                    total_chars += len(text)
                    total_words += len(text.split())
                
                # Estimate for full document
                if doc.page_count > 10:
                    total_chars = int(total_chars * (doc.page_count / 10))
                    total_words = int(total_words * (doc.page_count / 10))
                
                metadata.character_count = total_chars
                metadata.word_count = total_words
            except Exception:
                pass

            return metadata

        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return DocumentMetadata(page_count=doc.page_count if doc else 0)