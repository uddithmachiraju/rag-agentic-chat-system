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
                
                # Extract content based on options with improved approach
                all_chunks = []
                global_chunk_index = 0
                
                # Extract content page by page to maintain order and context
                for page_num in range(doc.page_count):
                    page_chunks, global_chunk_index = await self._extract_page_content(
                        doc, page_num, document_id, options, global_chunk_index
                    )
                    all_chunks.extend(page_chunks)
                
                # Sort chunks by page and position to ensure proper order
                all_chunks = self._sort_chunks_by_position(all_chunks)
                
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
    
    async def _extract_page_content(
        self, 
        doc: fitz.Document, 
        page_num: int, 
        document_id: str, 
        options: ProcessingOptions,
        chunk_index: int
    ) -> Tuple[List, int]:
        """Extract all content from a single page in proper order."""
        
        page_chunks = []
        page = doc[page_num]
        
        try:
            # Get page dimensions for better positioning
            page_rect = page.rect
            
            # Extract tables first (they often contain structured data)
            if options.extract_tables:
                table_chunks, chunk_index = await self._extract_page_tables(
                    page, page_num, document_id, chunk_index
                )
                page_chunks.extend(table_chunks)
            
            # Extract images with their positions
            if options.extract_images:
                image_chunks, chunk_index = await self._extract_page_images(
                    page, page_num, document_id, chunk_index
                )
                page_chunks.extend(image_chunks)
            
            # Extract text content with multiple fallback methods
            text_chunks, chunk_index = await self._extract_page_text_comprehensive(
                page, page_num, document_id, options, chunk_index
            )
            page_chunks.extend(text_chunks)
            
            # Extract any remaining content using alternative methods
            if not text_chunks or len(text_chunks) == 0:
                fallback_chunks, chunk_index = await self._extract_text_fallback(
                    page, page_num, document_id, options, chunk_index
                )
                page_chunks.extend(fallback_chunks)
            
        except Exception as e:
            self.logger.error(f"Error extracting content from page {page_num + 1}: {e}")
            # Create a minimal chunk to avoid losing the page entirely
            error_chunk = self._create_chunk(
                document_id=document_id,
                content=f"[Error extracting content from page {page_num + 1}: {str(e)}]",
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
    
    async def _extract_page_text_comprehensive(
        self, 
        page: fitz.Page, 
        page_num: int, 
        document_id: str, 
        options: ProcessingOptions,
        chunk_index: int
    ) -> Tuple[List, int]:
        """Extract text content using multiple methods for comprehensive coverage."""
        
        chunks = []
        
        try:
            # Method 1: Block-based extraction (most structured)
            text_content = ""
            blocks = page.get_text("blocks")
            block_texts = []
            
            for block in blocks:
                if len(block) >= 5:  # Text block
                    block_text = block[4].strip()
                    if block_text and len(block_text) > 2:  # Filter very short blocks
                        block_texts.append({
                            'text': block_text,
                            'bbox': block[:4],  # x0, y0, x1, y1
                            'block_no': block[5] if len(block) > 5 else 0
                        })
            
            # Sort blocks by position (top to bottom, left to right)
            block_texts.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
            
            # Combine blocks into text content
            for block_info in block_texts:
                text_content += block_info['text'] + "\n\n"
            
            # If block method fails, try alternative methods
            if not text_content.strip():
                # Method 2: Dictionary-based extraction
                text_dict = page.get_text("dict")
                text_content = self._extract_from_text_dict(text_dict)
            
            if not text_content.strip():
                # Method 3: Simple text extraction
                text_content = page.get_text()
            
            if not text_content.strip():
                # Method 4: Word-level extraction
                words = page.get_text("words")
                text_content = " ".join([word[4] for word in words if len(word) > 4])
            
            # Preprocess the extracted text
            if text_content.strip():
                text_content = await self._preprocess_content(text_content)
                
                # Create chunks based on options
                if options.chunk_size > 0:
                    text_chunks = self._chunk_text_advanced(
                        text_content,
                        options.chunk_size,
                        options.chunk_overlap,
                        page_num + 1
                    )
                    
                    char_offset = 0
                    for chunk_text in text_chunks:
                        if chunk_text.strip():
                            chunk = self._create_chunk(
                                document_id=document_id,
                                content=chunk_text,
                                chunk_index=chunk_index,
                                chunk_type="text",
                                page_number=page_num + 1,
                                start_char=char_offset,
                                end_char=char_offset + len(chunk_text),
                                metadata={
                                    "block_count": len(block_texts),
                                    "extraction_method": "comprehensive_blocks",
                                    "page_area": page.rect.width * page.rect.height
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            char_offset += len(chunk_text)
                else:
                    # Single chunk for entire page
                    chunk = self._create_chunk(
                        document_id=document_id,
                        content=text_content,
                        chunk_index=chunk_index,
                        chunk_type="text",
                        page_number=page_num + 1,
                        start_char=0,
                        end_char=len(text_content),
                        metadata={
                            "block_count": len(block_texts),
                            "extraction_method": "comprehensive_blocks",
                            "page_area": page.rect.width * page.rect.height
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        except Exception as e:
            self.logger.error(f"Error in comprehensive text extraction for page {page_num + 1}: {e}")
        
        return chunks, chunk_index
    
    def _extract_from_text_dict(self, text_dict: dict) -> str:
        """Extract text from PyMuPDF text dictionary format."""
        
        text_content = ""
        
        try:
            if "blocks" in text_dict:
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                line_text = ""
                                for span in line["spans"]:
                                    if "text" in span:
                                        line_text += span["text"]
                                if line_text.strip():
                                    text_content += line_text + "\n"
                        text_content += "\n"  # Paragraph break
        except Exception as e:
            self.logger.debug(f"Error extracting from text dict: {e}")
        
        return text_content
    
    async def _extract_text_fallback(
        self, 
        page: fitz.Page, 
        page_num: int, 
        document_id: str, 
        options: ProcessingOptions,
        chunk_index: int
    ) -> Tuple[List, int]:
        """Fallback text extraction methods for difficult PDFs."""
        
        chunks = []
        
        try:
            # Try OCR-like approach by extracting individual characters
            # This can help with PDFs that have unusual text encoding
            page_text = ""
            
            # Method 1: Extract by text objects
            text_objects = []
            try:
                # Get all text instances on the page
                text_instances = page.search_for("")  # This might not work as expected
                # Alternative: use get_textpage for low-level access
                textpage = page.get_textpage()
                if textpage:
                    page_text = textpage.extractText()
                    textpage = None  # Clean up
            except:
                pass
            
            # Method 2: Try different text extraction flags
            if not page_text.strip():
                try:
                    page_text = page.get_text(flags=fitz.TEXTFLAGS_TEXT)
                except:
                    pass
            
            # Method 3: Extract using raw text
            if not page_text.strip():
                try:
                    page_text = page.get_text("rawtext")
                except:
                    pass
            
            if page_text.strip():
                page_text = await self._preprocess_content(page_text)
                
                chunk = self._create_chunk(
                    document_id=document_id,
                    content=page_text,
                    chunk_index=chunk_index,
                    chunk_type="text",
                    page_number=page_num + 1,
                    start_char=0,
                    end_char=len(page_text),
                    metadata={
                        "extraction_method": "fallback",
                        "fallback_reason": "primary_extraction_failed"
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Last resort: create a placeholder for the page
                placeholder_text = f"[Page {page_num + 1} - Content could not be extracted]"
                chunk = self._create_chunk(
                    document_id=document_id,
                    content=placeholder_text,
                    chunk_index=chunk_index,
                    chunk_type="text",
                    page_number=page_num + 1,
                    start_char=0,
                    end_char=len(placeholder_text),
                    metadata={
                        "extraction_method": "placeholder",
                        "extraction_failed": True
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        except Exception as e:
            self.logger.error(f"Fallback extraction failed for page {page_num + 1}: {e}")
        
        return chunks, chunk_index
    
    async def _extract_page_tables(
        self, 
        page: fitz.Page, 
        page_num: int, 
        document_id: str,
        chunk_index: int
    ) -> Tuple[List, int]:
        """Extract tables from a single page."""
        
        chunks = []
        
        try:
            # Find tables on the page
            tables = page.find_tables()
            
            for table_index, table in enumerate(tables):
                try:
                    # Extract table data
                    table_data = table.extract()
                    
                    if not table_data or not any(any(cell for cell in row if cell) for row in table_data):
                        continue
                    
                    # Convert table to readable format
                    table_text = self._format_table_as_text(table_data)
                    
                    if table_text.strip():
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
                                "extraction_method": "pymupdf_tables",
                                "bbox": list(table.bbox) if hasattr(table, 'bbox') else None
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        
                except Exception as e:
                    self.logger.error(f"Error extracting table {table_index} from page {page_num + 1}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error finding tables on page {page_num + 1}: {e}")
        
        return chunks, chunk_index
    
    async def _extract_page_images(
        self, 
        page: fitz.Page, 
        page_num: int, 
        document_id: str,
        chunk_index: int
    ) -> Tuple[List, int]:
        """Extract images from a single page."""
        
        chunks = []
        
        try:
            # Get image list
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image reference
                    xref = img[0]
                    
                    # Get image data
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create image hash for identification
                    image_hash = hashlib.md5(image_bytes).hexdigest()
                    
                    # Create descriptive content
                    image_description = f"Image {img_index + 1} on page {page_num + 1}"
                    
                    # Try to extract any surrounding text (caption detection)
                    surrounding_text = self._extract_image_context(page, img)
                    
                    if surrounding_text:
                        image_description += f"\nContext: {surrounding_text}"
                    
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
        
        return chunks, chunk_index
    
    def _chunk_text_advanced(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int,
        page_number: int
    ) -> List[str]:
        """Advanced text chunking with better boundary detection."""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        # Split text into sentences first for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                if len(sentence) <= chunk_size:
                    current_chunk = sentence
                else:
                    # Sentence is too long, split it further
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk + " " + word) <= chunk_size:
                            temp_chunk += (" " if temp_chunk else "") + word
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    current_chunk = temp_chunk
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Apply overlap if specified
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    # Add overlap from previous chunk
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                    overlapped_chunks.append(overlap_text + " " + chunk)
            return overlapped_chunks
        
        return chunks
    
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
    
    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """Format table data as readable text."""
        
        if not table_data:
            return ""
        
        # Filter out empty rows
        filtered_data = [row for row in table_data if any(cell and str(cell).strip() for cell in row)]
        
        if not filtered_data:
            return ""
        
        # Calculate column widths
        col_widths = []
        max_cols = max(len(row) for row in filtered_data)
        
        for col_idx in range(max_cols):
            max_width = max(
                len(str(row[col_idx]).strip() if col_idx < len(row) and row[col_idx] else "") 
                for row in filtered_data
            )
            col_widths.append(max(max_width, 3))  # Minimum width of 3
        
        # Format table
        formatted_lines = []
        
        for row_idx, row in enumerate(filtered_data):
            formatted_row = []
            for col_idx in range(max_cols):
                cell = row[col_idx] if col_idx < len(row) else ""
                cell_str = str(cell).strip() if cell else ""
                if col_idx < len(col_widths):
                    formatted_row.append(cell_str.ljust(col_widths[col_idx]))
                else:
                    formatted_row.append(cell_str)
            
            formatted_lines.append(" | ".join(formatted_row))
            
            # Add separator after header
            if row_idx == 0 and len(filtered_data) > 1:
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
                r'Photo\s+\d+[:\.].*',
                r'Chart\s+\d+[:\.].*',
                r'Diagram\s+\d+[:\.].*'
            ]
            
            for pattern in caption_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    return matches[0][:200]  # First 200 characters
            
            return ""
            
        except Exception:
            return ""
    
    def _sort_chunks_by_position(self, chunks: List) -> List:
        """Sort chunks by page number and type with better positioning."""
        
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
                type_priority.get(x.chunk_type.value if hasattr(x.chunk_type, 'value') else str(x.chunk_type), 99),
                x.chunk_index or 0
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