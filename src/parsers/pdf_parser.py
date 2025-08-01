import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re
import hashlib
import fitz
from ..config import get_settings
from ..models.document import DocumentMetadata, ProcessingOptions, ChunkType, Document
from src.parsers.base_parser import BaseParser, ParseResult

class PDFParsingError(Exception):
    pass

class PDFParser(BaseParser):    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['pdf']
        self.mime_types = ['application/pdf']
        self.settings = get_settings()
        
        # PyMuPDF configuration
        self.text_extraction_flags = (
            fitz.TEXTFLAGS_TEXT | 
            fitz.TEXTFLAGS_BLOCKS | 
            fitz.TEXTFLAGS_DICT
        )
    
    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        extension = self._get_file_extension(file_path)
        
        # Check extension
        if extension in self.supported_extensions:
            return True
        
        # Check MIME type
        if mime_type and mime_type in self.mime_types:
            return True
        
        return False
    
    async def parse(self, file_path: str, document_id: str, options: ProcessingOptions = None) -> ParseResult:        
        start_time = time.time()
        
        try:
            # Validate file
            if not self._validate_file(file_path):
                return ParseResult(
                    success=False,
                    error_message=f"File not found or not readable: {file_path}"
                )
            
            options = options or ProcessingOptions()
            
            self.logger.info(f"Starting PDF parsing: {file_path}")
            
            # Open PDF document
            doc = await self._open_pdf_safely(file_path)
            if not doc:
                return ParseResult(
                    success=False,
                    error_message="Failed to open PDF document"
                )
            
            try:
                # Extract metadata
                metadata = await self._extract_metadata(doc, file_path)
                
                # Extract content based on options
                chunks = []
                
                if options.extract_tables:
                    table_chunks = await self._extract_tables(doc) 
                    chunks.extend(table_chunks)
                
                # Extract text content (main content)
                text_chunks = await self._extract_text_content(document_id, doc, options) 
                chunks.extend(text_chunks)
                
                if options.extract_images:
                    image_chunks = await self._extract_images(doc) 
                    chunks.extend(image_chunks)
                
                # Sort chunks by page and position
                chunks = self._sort_chunks_by_position(chunks)
                
                processing_time = time.time() - start_time
                
                self.logger.info(
                    f"PDF parsing completed: {len(chunks)} chunks extracted "
                    f"in {processing_time:.2f}s"
                )
                
                return ParseResult(
                    success=True,
                    chunks=chunks,
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
    
    async def _open_pdf_safely(self, file_path: str) -> Optional[fitz.Document]:
        """Safely open PDF document with error handling."""
        try:
            doc = fitz.open(file_path)
            
            # Basic validation
            if doc.is_closed:
                return None
            
            if doc.page_count == 0:
                self.logger.warning(f"PDF has no pages: {file_path}")
                return None
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to open PDF {file_path}: {e}")
            return None
    
    async def _extract_metadata(
        self, 
        doc: fitz.Document, 
        file_path: str
    ) -> DocumentMetadata:
        """Extract comprehensive metadata from PDF."""
        
        try:
            # Get PyMuPDF metadata
            pdf_meta = doc.metadata
            
            # Extract basic metadata
            metadata = DocumentMetadata(
                title=pdf_meta.get('title', '').strip(),
                author=pdf_meta.get('author', '').strip(),
                subject=pdf_meta.get('subject', '').strip(),
                page_count=doc.page_count
            )
            
            # Parse creation and modification dates
            if pdf_meta.get('creationDate'):
                try:
                    from datetime import datetime
                    # PyMuPDF returns date in format: "D:20210301120000+00'00'"
                    date_str = pdf_meta['creationDate']
                    if date_str.startswith('D:'):
                        date_str = date_str[2:16]  # Extract YYYYMMDDHHMMSS
                        metadata.created_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                except Exception as e:
                    self.logger.debug(f"Failed to parse creation date: {e}")
            
            if pdf_meta.get('modDate'):
                try:
                    from datetime import datetime
                    date_str = pdf_meta['modDate']
                    if date_str.startswith('D:'):
                        date_str = date_str[2:16]
                        metadata.modified_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                except Exception as e:
                    self.logger.debug(f"Failed to parse modification date: {e}")
            
            # Extract keywords
            if pdf_meta.get('keywords'):
                keywords = [k.strip() for k in pdf_meta['keywords'].split(',') if k.strip()]
                metadata.keywords = keywords
            
            # Calculate document statistics
            total_chars = 0
            total_words = 0
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                total_chars += len(text)
                total_words += len(text.split())
            
            metadata.character_count = total_chars
            metadata.word_count = total_words
            
            # Store PyMuPDF-specific metadata
            metadata.pdf_metadata = {
                'producer': pdf_meta.get('producer', ''),
                'creator': pdf_meta.get('creator', ''),
                'format': pdf_meta.get('format', ''),
                'encryption': pdf_meta.get('encryption', ''),
                'page_count': doc.page_count,
                'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return DocumentMetadata(page_count=doc.page_count if doc else 0)
    
    async def _extract_text_content(self, document_id: str, doc: fitz.Document, options: ProcessingOptions) -> List:
        chunks = []
        chunk_index = 0
        char_pos_accum = 0

        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]

                blocks = page.get_text("blocks")
                page_text = ""
                block_texts = []
                for block in blocks:
                    if len(block) >= 5:
                        block_text = block[4].strip()
                        if block_text:
                            block_texts.append(block_text)
                            page_text += block_text + "\n\n"
                if not page_text.strip():
                    continue

                page_text = await self._preprocess_content(page_text)

                if options.chunk_size > 0:
                    page_chunks = self._chunk_text(
                        page_text,
                        options.chunk_size,
                        options.chunk_overlap,
                        page_num + 1
                    )
                    running_offset = 0
                    for chunk_text in page_chunks:
                        start_char = char_pos_accum + page_text.find(chunk_text, running_offset)
                        end_char = start_char + len(chunk_text)
                        running_offset = start_char - char_pos_accum + len(chunk_text)

                        chunk = self._create_chunk(
                            document_id=document_id,
                            content=chunk_text,
                            chunk_index=chunk_index,
                            chunk_type="text",
                            page_number=page_num + 1,
                            start_char=start_char,
                            end_char=end_char,
                            metadata={
                                "block_count": len(block_texts),
                                "extraction_method": "pymupdf_blocks"
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    char_pos_accum += len(page_text)
                else:
                    start_char = char_pos_accum
                    end_char = start_char + len(page_text)
                    chunk = self._create_chunk(
                        document_id=document_id,
                        content=page_text,
                        chunk_index=chunk_index,
                        chunk_type="text",
                        page_number=page_num + 1,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={
                            "block_count": len(block_texts),
                            "extraction_method": "pymupdf_blocks"
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_pos_accum += len(page_text)

            except Exception as e:
                self.logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                continue

        return chunks
    
    async def _extract_tables(self, doc: fitz.Document) -> List:
        """Extract tables from PDF using PyMuPDF's table detection."""
        
        chunks = []
        chunk_index = 0
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                
                # Find tables on the page
                tables = page.find_tables()
                
                for table_index, table in enumerate(tables):
                    try:
                        # Extract table data
                        table_data = table.extract()
                        print(table_data)
                        
                        if not table_data:
                            continue
                        
                        # Convert table to readable format
                        table_text = self._format_table_as_text(table_data)
                        
                        if table_text.strip():
                            chunk = self._create_chunk(
                                content=table_text,
                                chunk_index=chunk_index,
                                chunk_type="table",
                                page_number=page_num + 1,
                                metadata={
                                    "table_index": table_index,
                                    "rows": len(table_data),
                                    "columns": len(table_data[0]) if table_data else 0,
                                    "extraction_method": "pymupdf_tables",
                                    "bbox": table.bbox  # Bounding box coordinates
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error extracting table {table_index} from page {page_num + 1}: {e}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error finding tables on page {page_num + 1}: {e}")
                continue
        
        return chunks
    
    async def _extract_images(self, doc: fitz.Document) -> List:
        """Extract images and their descriptions from PDF."""
        
        chunks = []
        chunk_index = 0
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                
                # Get image list
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image reference
                        xref = img[0]
                        
                        # Get image data
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Create image hash for identification
                        image_hash = hashlib.md5(image_bytes).hexdigest()
                        
                        # Create descriptive content
                        image_description = f"Image {img_index + 1} on page {page_num + 1}"
                        
                        # Try to extract any surrounding text (caption detection)
                        # This is a simplified approach - could be enhanced
                        surrounding_text = self._extract_image_context(page, img)
                        
                        if surrounding_text:
                            image_description += f"\nContext: {surrounding_text}"
                        
                        chunk = self._create_chunk(
                            content=image_description,
                            chunk_index=chunk_index,
                            chunk_type="image",
                            page_number=page_num + 1,
                            metadata={
                                "image_index": img_index,
                                "image_hash": image_hash,
                                "image_format": image_ext,
                                "image_size": len(image_bytes),
                                "extraction_method": "pymupdf_images",
                                "xref": xref
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error extracting images from page {page_num + 1}: {e}")
                continue
        
        return chunks
    
    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """Format table data as readable text."""
        
        if not table_data:
            return ""
        
        # Calculate column widths
        col_widths = []
        for col_idx in range(len(table_data[0])):
            max_width = max(
                len(str(row[col_idx]) if col_idx < len(row) else "") 
                for row in table_data
            )
            col_widths.append(max(max_width, 10))  # Minimum width of 10
        
        # Format table
        formatted_lines = []
        
        for row_idx, row in enumerate(table_data):
            formatted_row = []
            for col_idx, cell in enumerate(row):
                if col_idx < len(col_widths):
                    cell_str = str(cell) if cell else ""
                    formatted_row.append(cell_str.ljust(col_widths[col_idx]))
            
            formatted_lines.append(" | ".join(formatted_row))
            
            # Add separator after header
            if row_idx == 0 and len(table_data) > 1:
                separator = " | ".join("-" * width for width in col_widths)
                formatted_lines.append(separator)
        
        return "\n".join(formatted_lines)
    
    def _extract_image_context(self, page: fitz.Page, img_info: tuple) -> str:
        """Extract text context around an image (simplified caption detection)."""
        
        try:
            # This is a simplified approach - could be enhanced with better positioning logic
            page_text = page.get_text()
            
            # Look for common caption patterns
            caption_patterns = [
                r'Figure\s+\d+[:\.].*',
                r'Fig\.\s+\d+[:\.].*',
                r'Image\s+\d+[:\.].*',
                r'Photo\s+\d+[:\.].*'
            ]
            
            for pattern in caption_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    return matches[0][:200]  # First 200 characters
            
            return ""
            
        except Exception:
            return ""
    
    def _chunk_text(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int,
        page_number: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for sentence endings
                for boundary in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start:
                        end = boundary_pos + len(boundary)
                        break
                else:
                    # Look for word boundaries
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start:
                        end = space_pos
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = max(end - overlap, start + 1)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _sort_chunks_by_position(self, chunks: List) -> List:
        """Sort chunks by page number and type."""
        
        # Define type priority (text first, then tables, then images)
        type_priority = {
            "text": 1,
            "table": 2,
            "image": 3
        }
        
        return sorted(
            chunks,
            key=lambda x: (
                x.page_number or 0,
                type_priority.get(x.chunk_type.value, 99),
                x.chunk_index
            )
        )
    
    async def extract_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """Extract basic PDF information without full parsing."""
        
        try:
            doc = await self._open_pdf_safely(file_path)
            if not doc:
                return {"error": "Cannot open PDF"}
            
            try:
                info = {
                    "page_count": doc.page_count,
                    "metadata": doc.metadata,
                    "is_encrypted": doc.needs_pass,
                    "is_pdf": True,
                    "file_size": Path(file_path).stat().st_size,
                }
                
                # Quick text sample from first page
                if doc.page_count > 0:
                    first_page = doc[0]
                    sample_text = first_page.get_text()[:500]  # First 500 chars
                    info["text_sample"] = sample_text
                
                return info
                
            finally:
                doc.close()
                
        except Exception as e:
            return {"error": str(e)}
