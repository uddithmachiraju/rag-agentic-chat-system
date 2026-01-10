# import hashlib
# import re
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import fitz

# from src.parsers.base_parser import BaseParser, ParseResult

# from ..config import get_settings
# from ..models.document import DocumentMetadata, ProcessingOptions


# class PDFParsingError(Exception):
#     pass


# class PDFParser(BaseParser):
#     def __init__(self):
#         super().__init__()
#         self.supported_extensions = ['pdf']
#         self.mime_types = ['application/pdf']
#         self.settings = get_settings()

#         # PyMuPDF configuration
#         self.text_extraction_flags = (
#             fitz.TEXTFLAGS_TEXT |
#             fitz.TEXTFLAGS_BLOCKS |
#             fitz.TEXTFLAGS_DICT
#         )

#     def can_parse(self, file_path: str, mime_type: str = None) -> bool:
#         extension = self._get_file_extension(file_path)

#         # Check extension
#         if extension in self.supported_extensions:
#             return True

#         # Check MIME type
#         if mime_type and mime_type in self.mime_types:
#             return True

#         return False

#     async def parse(self, file_path: str, document_id: str, options: ProcessingOptions = None) -> ParseResult:
#         start_time = time.time()

#         try:
#             # Validate file
#             if not self._validate_file(file_path):
#                 return ParseResult(
#                     success=False,
#                     error_message=f"File not found or not readable: {file_path}"
#                 )

#             options = options or ProcessingOptions()

#             self.logger.info(f"Starting PDF parsing: {file_path}")

#             # Open PDF document
#             doc = await self._open_pdf_safely(file_path)
#             if not doc:
#                 return ParseResult(
#                     success=False,
#                     error_message="Failed to open PDF document"
#                 )

#             try:
#                 # Extract metadata
#                 metadata = await self._extract_metadata(doc, file_path)

#                 # Extract content based on options with improved approach
#                 all_chunks = []
#                 global_chunk_index = 0

#                 # Extract content page by page to maintain order and context
#                 for page_num in range(doc.page_count):
#                     page_chunks, global_chunk_index = await self._extract_page_content(
#                         doc, page_num, document_id, options, global_chunk_index
#                     )
#                     all_chunks.extend(page_chunks)

#                 # Sort chunks by page and position to ensure proper order
#                 all_chunks = self._sort_chunks_by_position(all_chunks)

#                 processing_time = time.time() - start_time

#                 self.logger.info(
#                     f"PDF parsing completed: {len(all_chunks)} chunks extracted "
#                     f"in {processing_time:.2f}s"
#                 )

#                 return ParseResult(
#                     success=True,
#                     chunks=all_chunks,
#                     metadata=metadata,
#                     processing_time=processing_time
#                 )

#             finally:
#                 doc.close()

#         except Exception as e:
#             processing_time = time.time() - start_time
#             error_msg = f"PDF parsing failed: {str(e)}"
#             self.logger.error(error_msg)

#             return ParseResult(
#                 success=False,
#                 error_message=error_msg,
#                 processing_time=processing_time
#             )

#     async def _extract_page_content(
#         self,
#         doc: fitz.Document,
#         page_num: int,
#         document_id: str,
#         options: ProcessingOptions,
#         chunk_index: int
#     ) -> Tuple[List, int]:
#         """Extract all content from a single page in proper order."""

#         page_chunks = []
#         page = doc[page_num]

#         try:
#             # Get page dimensions for better positioning
#             page_rect = page.rect

#             # Extract tables first (they often contain structured data)
#             if options.extract_tables:
#                 table_chunks, chunk_index = await self._extract_page_tables(
#                     page, page_num, document_id, chunk_index
#                 )
#                 page_chunks.extend(table_chunks)

#             # Extract images with their positions
#             if options.extract_images:
#                 image_chunks, chunk_index = await self._extract_page_images(
#                     page, page_num, document_id, chunk_index
#                 )
#                 page_chunks.extend(image_chunks)

#             # Extract text content with multiple fallback methods
#             text_chunks, chunk_index = await self._extract_page_text_comprehensive(
#                 page, page_num, document_id, options, chunk_index
#             )
#             page_chunks.extend(text_chunks)

#             # Extract any remaining content using alternative methods
#             if not text_chunks or len(text_chunks) == 0:
#                 fallback_chunks, chunk_index = await self._extract_text_fallback(
#                     page, page_num, document_id, options, chunk_index
#                 )
#                 page_chunks.extend(fallback_chunks)

#         except Exception as e:
#             self.logger.error(
#                 f"Error extracting content from page {page_num + 1}: {e}")
#             # Create a minimal chunk to avoid losing the page entirely
#             error_chunk = self._create_chunk(
#                 document_id=document_id,
#                 content=f"[Error extracting content from page {page_num + 1}: {str(e)}]",
#                 chunk_index=chunk_index,
#                 chunk_type="text",
#                 page_number=page_num + 1,
#                 start_char=0,
#                 end_char=0,
#                 metadata={"extraction_error": True, "error_message": str(e)}
#             )
#             page_chunks.append(error_chunk)
#             chunk_index += 1

#         return page_chunks, chunk_index

#     async def _extract_page_text_comprehensive(
#         self,
#         page: fitz.Page,
#         page_num: int,
#         document_id: str,
#         options: ProcessingOptions,
#         chunk_index: int
#     ) -> Tuple[List, int]:
#         """Extract text content using multiple methods with semantic structure preservation."""

#         chunks = []

#         try:
#             # Extract structured content with position information
#             blocks = page.get_text("dict")
#             structured_content = self._extract_structured_content(blocks)

#             if not structured_content:
#                 # Fallback to simple extraction
#                 text_content = page.get_text()
#                 if text_content.strip():
#                     text_content = await self._preprocess_content(text_content)
#                     structured_content = [
#                         {'text': text_content, 'type': 'text', 'level': 0}]

#             if structured_content:
#                 # Build semantic sections from structured content
#                 semantic_sections = self._build_semantic_sections(
#                     structured_content)

#                 # Create chunks from semantic sections
#                 if options.chunk_size > 0:
#                     text_chunks = self._chunk_semantic_sections(
#                         semantic_sections,
#                         options.chunk_size,
#                         options.chunk_overlap
#                     )
#                 else:
#                     # Single chunk for entire page
#                     text_chunks = [self._flatten_sections(semantic_sections)]

#                 char_offset = 0
#                 for chunk_text in text_chunks:
#                     if chunk_text and chunk_text.strip():
#                         # Normalize whitespace and trim chunk content
#                         try:
#                             chunk_text = await self._preprocess_content(chunk_text)
#                         except Exception:
#                             chunk_text = chunk_text.strip()

#                         if not chunk_text:
#                             continue

#                         chunk = self._create_chunk(
#                             content=chunk_text,
#                             chunk_index=chunk_index,
#                             chunk_type="text",
#                             page_number=page_num + 1,
#                             start_char=char_offset,
#                             end_char=char_offset + len(chunk_text),
#                             metadata={
#                                 "extraction_method": "semantic_structured",
#                                 "chunking_method": "semantic_sections",
#                                 "page_area": page.rect.width * page.rect.height
#                             },
#                             document_id=document_id
#                         )
#                         chunks.append(chunk)
#                         chunk_index += 1
#                         char_offset += len(chunk_text)

#         except Exception as e:
#             self.logger.error(
#                 f"Error in comprehensive text extraction for page {page_num + 1}: {e}")

#         return chunks, chunk_index

#     def _extract_structured_content(self, blocks_dict: dict) -> List[Dict]:
#         """Extract content with structure (headers, paragraphs, lists)."""
#         structured_items = []

#         try:
#             if "blocks" not in blocks_dict:
#                 return structured_items

#             for block in blocks_dict["blocks"]:
#                 if block.get("type") == 0:  # Text block
#                     if "lines" not in block:
#                         continue

#                     block_text = ""
#                     font_sizes = []

#                     # Extract text and font information
#                     for line in block["lines"]:
#                         if "spans" not in line:
#                             continue

#                         line_text = ""
#                         for span in line["spans"]:
#                             if "text" in span:
#                                 line_text += span["text"]
#                                 if "size" in span:
#                                     font_sizes.append(span["size"])

#                         if line_text.strip():
#                             block_text += line_text + "\n"

#                     if not block_text.strip():
#                         continue

#                     # Clean block text synchronously
#                     try:
#                         block_text = self._clean_text(block_text)
#                     except Exception:
#                         block_text = block_text.strip()

#                     # Determine content type and level
#                     avg_font_size = sum(font_sizes) / \
#                         len(font_sizes) if font_sizes else 12
#                     content_type, level = self._classify_content_block(
#                         block_text.strip(),
#                         avg_font_size,
#                         block.get("bbox", [0, 0, 0, 0])
#                     )

#                     structured_items.append({
#                         'text': block_text.strip(),
#                         'type': content_type,
#                         'level': level,
#                         'font_size': avg_font_size,
#                         'bbox': block.get("bbox", [0, 0, 0, 0])
#                     })

#         except Exception as e:
#             self.logger.debug(f"Error extracting structured content: {e}")

#         return structured_items

#     def _classify_content_block(self, text: str, font_size: float, bbox: list) -> Tuple[str, int]:
#         """Classify content block type and hierarchy level."""

#         # Check for section headers
#         if self._is_section_header(text, font_size):
#             # Determine header level based on patterns and font size
#             if re.match(r'^[A-Z][A-Z\s]{3,}$', text):  # All caps
#                 return ('header', 1)
#             elif re.match(r'^\d+\.?\s+[A-Z]', text):  # Numbered section
#                 return ('header', 2)
#             elif font_size > 13:  # Larger font
#                 return ('header', 1)
#             else:
#                 return ('header', 2)

#         # Check for list items
#         if self._is_list_item(text):
#             return ('list_item', 0)

#         # Default to paragraph
#         return ('paragraph', 0)

#     def _is_section_header(self, text: str, font_size: float) -> bool:
#         """Determine if text is likely a section header."""

#         # Short text
#         if len(text) > 150:
#             return False

#         # Common header patterns
#         header_patterns = [
#             r'^[A-Z][A-Z\s]+$',  # ALL CAPS
#             r'^\d+\.?\s+[A-Z]',  # "1. Title" or "1 Title"
#             r'^[IVXLCDM]+\.\s+[A-Z]',  # Roman numerals
#             r'^Chapter\s+\d+',
#             r'^Section\s+\d+',
#             r'^Part\s+\d+',
#             # Title Case (up to 6 words)
#             r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){0,5}$',
#         ]

#         for pattern in header_patterns:
#             if re.match(pattern, text):
#                 return True

#         # Check font size (headers usually larger)
#         if font_size > 13 and len(text) < 80:
#             return True

#         # Short line without ending punctuation
#         if len(text) < 60 and not text.endswith(('.', '!', '?', ';', ',')):
#             # Check if it looks like a title
#             words = text.split()
#             if 1 <= len(words) <= 8:  # Reasonable title length
#                 return True

#         return False

#     def _is_list_item(self, text: str) -> bool:
#         """Determine if text is a list item."""
#         list_patterns = [
#             r'^\s*[•●○■□▪▫–—]\s+',  # Bullet points
#             r'^\s*\d+[\.)]\s+',  # Numbered lists
#             r'^\s*[a-z][\.)]\s+',  # Lettered lists
#             r'^\s*[ivx]+[\.)]\s+',  # Roman numeral lists
#         ]

#         for pattern in list_patterns:
#             if re.match(pattern, text, re.IGNORECASE):
#                 return True

#         return False

#     def _build_semantic_sections(self, structured_items: List[Dict]) -> List[Dict]:
#         """Build semantic sections by grouping related content."""

#         sections = []
#         current_section = None

#         for item in structured_items:
#             if item['type'] == 'header':
#                 # Start new section
#                 if current_section:
#                     sections.append(current_section)

#                 current_section = {
#                     'header': item['text'],
#                     'level': item['level'],
#                     'content': [],
#                     'subsections': []
#                 }
#             else:
#                 # Add to current section
#                 if current_section is None:
#                     # Create default section for content before first header
#                     current_section = {
#                         'header': '',
#                         'level': 0,
#                         'content': [],
#                         'subsections': []
#                     }

#                 current_section['content'].append(item['text'])

#         # Add last section
#         if current_section:
#             sections.append(current_section)

#         return sections

#     def _flatten_sections(self, sections: List[Dict]) -> str:
#         """Flatten sections into text."""
#         result = []

#         for section in sections:
#             if section['header']:
#                 result.append(section['header'])
#                 result.append('')  # Blank line after header

#             result.extend(section['content'])

#             if section['subsections']:
#                 subsection_text = self._flatten_sections(
#                     section['subsections'])
#                 if subsection_text:
#                     result.append('')
#                     result.append(subsection_text)

#         return '\n'.join(result)

#     def _chunk_semantic_sections(
#         self,
#         sections: List[Dict],
#         chunk_size: int,
#         overlap: int
#     ) -> List[str]:
#         """Create chunks from semantic sections while preserving structure."""

#         chunks = []
#         current_chunk = ""
#         current_sections = []

#         for section in sections:
#             # Build section text (header + content)
#             section_parts = []
#             if section['header']:
#                 section_parts.append(section['header'])
#             section_parts.extend(section['content'])
#             section_text = '\n\n'.join(section_parts)

#             # Calculate if adding this section exceeds chunk size
#             potential_chunk = current_chunk + \
#                 ('\n\n' if current_chunk else '') + section_text

#             if len(potential_chunk) <= chunk_size:
#                 # Add to current chunk
#                 current_chunk = potential_chunk
#                 current_sections.append(section_text)
#             else:
#                 # Save current chunk if it has content
#                 if current_chunk.strip():
#                     chunks.append(current_chunk.strip())

#                     # Apply overlap
#                     if overlap > 0 and current_sections:
#                         overlap_text = self._get_section_overlap(
#                             current_sections, overlap)
#                         current_chunk = overlap_text + '\n\n' + \
#                             section_text if overlap_text else section_text
#                     else:
#                         current_chunk = section_text

#                     current_sections = [section_text]
#                 else:
#                     # Section itself is too large, split it
#                     if len(section_text) > chunk_size:
#                         section_chunks = self._split_large_section(
#                             section_text,
#                             chunk_size,
#                             section['header']
#                         )
#                         chunks.extend(section_chunks[:-1])
#                         current_chunk = section_chunks[-1] if section_chunks else ""
#                         current_sections = [
#                             current_chunk] if current_chunk else []
#                     else:
#                         current_chunk = section_text
#                         current_sections = [section_text]

#         # Add last chunk
#         if current_chunk.strip():
#             chunks.append(current_chunk.strip())

#         return chunks

#     def _get_section_overlap(self, sections: List[str], overlap_size: int) -> str:
#         """Get overlap text from previous sections."""
#         if not sections:
#             return ""

#         # Take from the last section(s)
#         overlap_text = ""
#         for section in reversed(sections):
#             if len(overlap_text) + len(section) <= overlap_size:
#                 overlap_text = section + '\n\n' + overlap_text if overlap_text else section
#             else:
#                 # Take partial content from this section
#                 if not overlap_text:
#                     # Take the end of this section
#                     overlap_text = section[-overlap_size:]
#                 break

#         return overlap_text.strip()

#     def _split_large_section(
#         self,
#         section_text: str,
#         chunk_size: int,
#         header: str = ""
#     ) -> List[str]:
#         """Split a large section while keeping the header with each chunk."""

#         # Split into paragraphs
#         paragraphs = [p.strip()
#                       for p in section_text.split('\n\n') if p.strip()]

#         chunks = []
#         current_chunk_paragraphs = []

#         # Add header to first chunk
#         chunk_prefix = header + '\n\n' if header and header not in paragraphs else ""
#         current_text = chunk_prefix

#         for para in paragraphs:
#             if para == header:  # Skip if paragraph is the header itself
#                 continue

#             potential_text = current_text + \
#                 ('\n\n' if current_text != chunk_prefix else '') + para

#             if len(potential_text) <= chunk_size:
#                 current_text = potential_text
#                 current_chunk_paragraphs.append(para)
#             else:
#                 # Save current chunk
#                 if current_text.strip() and current_text != chunk_prefix:
#                     chunks.append(current_text.strip())

#                 # Start new chunk with header
#                 current_text = chunk_prefix + para
#                 current_chunk_paragraphs = [para]

#                 # If single paragraph is too large, split by sentences
#                 if len(current_text) > chunk_size:
#                     para_chunks = self._split_paragraph_by_sentences(
#                         para, chunk_size, chunk_prefix)
#                     chunks.extend(para_chunks[:-1])
#                     current_text = para_chunks[-1] if para_chunks else ""
#                     current_chunk_paragraphs = [
#                         current_text] if current_text else []

#         # Add last chunk
#         if current_text.strip() and current_text != chunk_prefix:
#             chunks.append(current_text.strip())

#         return chunks if chunks else [section_text]

#     def _split_paragraph_by_sentences(
#         self,
#         paragraph: str,
#         chunk_size: int,
#         prefix: str = ""
#     ) -> List[str]:
#         """Split paragraph by sentences."""
#         sentences = re.split(r'(?<=[.!?])\s+', paragraph)

#         chunks = []
#         current_chunk = prefix

#         for sentence in sentences:
#             potential = current_chunk + \
#                 (' ' if current_chunk and current_chunk != prefix else '') + sentence

#             if len(potential) <= chunk_size:
#                 current_chunk = potential
#             else:
#                 if current_chunk.strip() and current_chunk != prefix:
#                     chunks.append(current_chunk.strip())
#                 current_chunk = prefix + sentence

#                 # If single sentence too large, split by words
#                 if len(current_chunk) > chunk_size:
#                     word_chunks = self._split_by_words(
#                         sentence, chunk_size, prefix)
#                     chunks.extend(word_chunks[:-1])
#                     current_chunk = word_chunks[-1] if word_chunks else ""

#         if current_chunk.strip() and current_chunk != prefix:
#             chunks.append(current_chunk.strip())

#         return chunks if chunks else [paragraph]

#     def _split_by_words(self, text: str, chunk_size: int, prefix: str = "") -> List[str]:
#         """Split text by words when sentences are too long."""
#         words = text.split()
#         chunks = []
#         current_chunk = prefix

#         for word in words:
#             potential = current_chunk + \
#                 (' ' if current_chunk and current_chunk != prefix else '') + word

#             if len(potential) <= chunk_size:
#                 current_chunk = potential
#             else:
#                 if current_chunk.strip() and current_chunk != prefix:
#                     chunks.append(current_chunk.strip())
#                 current_chunk = prefix + word

#         if current_chunk.strip() and current_chunk != prefix:
#             chunks.append(current_chunk.strip())

#         return chunks if chunks else [text]

#     def _extract_from_text_dict(self, text_dict: dict) -> str:
#         """Extract text from PyMuPDF text dictionary format."""

#         text_content = ""

#         try:
#             if "blocks" in text_dict:
#                 for block in text_dict["blocks"]:
#                     if "lines" in block:
#                         for line in block["lines"]:
#                             if "spans" in line:
#                                 line_text = ""
#                                 for span in line["spans"]:
#                                     if "text" in span:
#                                         line_text += span["text"]
#                                 if line_text.strip():
#                                     text_content += line_text + "\n"
#                         text_content += "\n"  # Paragraph break
#         except Exception as e:
#             self.logger.debug(f"Error extracting from text dict: {e}")

#         try:
#             return self._clean_text(text_content)
#         except Exception:
#             return text_content

#     async def _extract_text_fallback(
#         self,
#         page: fitz.Page,
#         page_num: int,
#         document_id: str,
#         options: ProcessingOptions,
#         chunk_index: int
#     ) -> Tuple[List, int]:
#         """Fallback text extraction methods for difficult PDFs."""

#         chunks = []

#         try:
#             page_text = ""

#             # Try different extraction methods
#             try:
#                 textpage = page.get_textpage()
#                 if textpage:
#                     page_text = textpage.extractText()
#                     textpage = None
#             except:
#                 pass

#             if not page_text.strip():
#                 try:
#                     page_text = page.get_text(flags=fitz.TEXTFLAGS_TEXT)
#                 except:
#                     pass

#             if not page_text.strip():
#                 try:
#                     page_text = page.get_text("rawtext")
#                 except:
#                     pass

#             if page_text.strip():
#                 page_text = await self._preprocess_content(page_text)

#                 chunk = self._create_chunk(
#                     document_id=document_id,
#                     content=page_text,
#                     chunk_index=chunk_index,
#                     chunk_type="text",
#                     page_number=page_num + 1,
#                     start_char=0,
#                     end_char=len(page_text),
#                     metadata={
#                         "extraction_method": "fallback",
#                         "fallback_reason": "primary_extraction_failed"
#                     }
#                 )
#                 chunks.append(chunk)
#                 chunk_index += 1
#             else:
#                 placeholder_text = f"[Page {page_num + 1} - Content could not be extracted]"
#                 chunk = self._create_chunk(
#                     document_id=document_id,
#                     content=placeholder_text,
#                     chunk_index=chunk_index,
#                     chunk_type="text",
#                     page_number=page_num + 1,
#                     start_char=0,
#                     end_char=len(placeholder_text),
#                     metadata={
#                         "extraction_method": "placeholder",
#                         "extraction_failed": True
#                     }
#                 )
#                 chunks.append(chunk)
#                 chunk_index += 1

#         except Exception as e:
#             self.logger.error(
#                 f"Fallback extraction failed for page {page_num + 1}: {e}")

#         return chunks, chunk_index

#     async def _extract_page_tables(
#         self,
#         page: fitz.Page,
#         page_num: int,
#         document_id: str,
#         chunk_index: int
#     ) -> Tuple[List, int]:
#         """Extract tables from a single page."""

#         chunks = []

#         try:
#             tables = []
#             # PyMuPDF may expose table-finding differently across versions
#             if hasattr(page, 'find_tables'):
#                 try:
#                     tables = page.find_tables() or []
#                 except Exception:
#                     tables = []

#             for table_index, table in enumerate(tables):
#                 try:
#                     # Some table objects provide an extract() method
#                     table_data = []
#                     if hasattr(table, 'extract'):
#                         table_data = table.extract() or []
#                     else:
#                         # fallback: try to get table as text block
#                         table_data = []

#                     if not table_data or not any(any(cell for cell in row if cell) for row in table_data):
#                         continue

#                     table_text = self._format_table_as_text(table_data)

#                     if table_text.strip():
#                         chunk = self._create_chunk(
#                             document_id=document_id,
#                             content=table_text,
#                             chunk_index=chunk_index,
#                             chunk_type="table",
#                             page_number=page_num + 1,
#                             start_char=0,
#                             end_char=len(table_text),
#                             metadata={
#                                 "table_index": table_index,
#                                 "rows": len(table_data),
#                                 "columns": len(table_data[0]) if table_data else 0,
#                                 "extraction_method": "pymupdf_tables",
#                                 "bbox": list(table.bbox) if hasattr(table, 'bbox') else None
#                             }
#                         )
#                         chunks.append(chunk)
#                         chunk_index += 1

#                 except Exception as e:
#                     self.logger.debug(
#                         f"Error extracting table {table_index} on page {page_num + 1}: {e}")

#         except Exception as e:
#             self.logger.error(
#                 f"Error finding tables on page {page_num + 1}: {e}")

#         return chunks, chunk_index

#     def _format_table_as_text(self, table_data: List[List[str]]) -> str:
#         """Format table data as readable text."""

#         if not table_data:
#             return ""

#         filtered_data = [row for row in table_data if any(
#             cell and str(cell).strip() for cell in row)]

#         if not filtered_data:
#             return ""

#         col_widths = []
#         max_cols = max(len(row) for row in filtered_data)

#         for col_idx in range(max_cols):
#             max_width = max(
#                 len(str(row[col_idx]).strip() if col_idx <
#                     len(row) and row[col_idx] else "")
#                 for row in filtered_data
#             )
#             col_widths.append(max(max_width, 3))

#         formatted_lines = []

#         for row_idx, row in enumerate(filtered_data):
#             formatted_row = []
#             for col_idx in range(max_cols):
#                 cell = row[col_idx] if col_idx < len(row) else ""
#                 cell_str = str(cell).strip() if cell else ""
#                 if col_idx < len(col_widths):
#                     formatted_row.append(cell_str.ljust(col_widths[col_idx]))
#                 else:
#                     formatted_row.append(cell_str)

#             formatted_lines.append(" | ".join(formatted_row))

#             if row_idx == 0 and len(filtered_data) > 1:
#                 separator = " | ".join("-" * width for width in col_widths)
#                 formatted_lines.append(separator)

#         return "\n".join(formatted_lines)

#     def _extract_image_context(self, page: fitz.Page, img_info: tuple) -> str:
#         """Extract text context around an image (simplified caption detection)."""

#         try:
#             page_text = page.get_text()

#             caption_patterns = [
#                 r'Figure\s+\d+[:\.].*',
#                 r'Fig\.\s+\d+[:\.].*',
#                 r'Image\s+\d+[:\.].*',
#                 r'Photo\s+\d+[:\.].*',
#                 r'Chart\s+\d+[:\.].*',
#                 r'Diagram\s+\d+[:\.].*'
#             ]

#             for pattern in caption_patterns:
#                 matches = re.findall(pattern, page_text,
#                                      re.IGNORECASE | re.MULTILINE)
#                 if matches:
#                     return matches[0][:200]

#             return ""

#         except Exception:
#             return ""

#     def _sort_chunks_by_position(self, chunks: List) -> List:
#         """Sort chunks by page number and type with better positioning."""

#         type_priority = {
#             "text": 1,
#             "table": 2,
#             "image": 3
#         }

#         return sorted(
#             chunks,
#             key=lambda x: (
#                 x.page_number or 0,
#                 type_priority.get(x.chunk_type.value if hasattr(
#                     x.chunk_type, 'value') else str(x.chunk_type), 99),
#                 x.chunk_index or 0
#             )
#         )

#     async def extract_pdf_info(self, file_path: str) -> Dict[str, Any]:
#         """Extract basic PDF information without full parsing."""

#         try:
#             doc = await self._open_pdf_safely(file_path)
#             if not doc:
#                 return {"error": "Cannot open PDF"}

#             try:
#                 info = {
#                     "page_count": doc.page_count,
#                     "metadata": doc.metadata,
#                     "is_encrypted": doc.needs_pass,
#                     "is_pdf": True,
#                     "file_size": Path(file_path).stat().st_size,
#                 }

#                 if doc.page_count > 0:
#                     first_page = doc[0]
#                     sample_text = first_page.get_text()[:500]
#                     info["text_sample"] = sample_text

#                 return info

#             finally:
#                 doc.close()

#         except Exception as e:
#             self.logger.error(
#                 f"Error extracting PDF info for {file_path}: {e}")
#             return {"error": str(e)}

#     async def _extract_page_images(
#         self,
#         page: fitz.Page,
#         page_num: int,
#         document_id: str,
#         chunk_index: int
#     ) -> Tuple[List, int]:
#         """Extract images from a single page."""

#         chunks = []

#         try:
#             image_list = page.get_images()

#             for img_index, img in enumerate(image_list):
#                 try:
#                     xref = img[0]
#                     base_image = page.parent.extract_image(xref)
#                     image_bytes = base_image["image"]
#                     image_ext = base_image["ext"]

#                     image_hash = hashlib.md5(image_bytes).hexdigest()
#                     image_description = f"Image {img_index + 1} on page {page_num + 1}"

#                     surrounding_text = self._extract_image_context(page, img)

#                     if surrounding_text:
#                         image_description += f"\nContext: {surrounding_text}"

#                     chunk = self._create_chunk(
#                         document_id=document_id,
#                         content=image_description,
#                         chunk_index=chunk_index,
#                         chunk_type="image",
#                         page_number=page_num + 1,
#                         start_char=0,
#                         end_char=len(image_description),
#                         metadata={
#                             "image_index": img_index,
#                             "image_hash": image_hash,
#                             "image_format": image_ext,
#                             "image_size": len(image_bytes),
#                             "extraction_method": "pymupdf_images",
#                             "xref": xref
#                         }
#                     )
#                     chunks.append(chunk)
#                     chunk_index += 1

#                 except Exception as e:
#                     self.logger.error(
#                         f"Error extracting image {img_index} from page {page_num + 1}: {e}")
#                     continue

#         except Exception as e:
#             self.logger.error(
#                 f"Error extracting images from page {page_num + 1}: {e}")

#         return chunks, chunk_index

#     async def _open_pdf_safely(self, file_path: str) -> Optional[fitz.Document]:
#         """Safely open PDF document with error handling."""
#         try:
#             doc = fitz.open(file_path)

#             if doc.is_closed:
#                 return None

#             if doc.page_count == 0:
#                 self.logger.warning(f"PDF has no pages: {file_path}")
#                 return None

#             return doc

#         except Exception as e:
#             self.logger.error(f"Failed to open PDF {file_path}: {e}")
#             return None

#     async def _extract_metadata(
#         self,
#         doc: fitz.Document,
#         file_path: str
#     ) -> DocumentMetadata:
#         """Extract comprehensive metadata from PDF."""

#         try:
#             pdf_meta = doc.metadata

#             metadata = DocumentMetadata(
#                 title=pdf_meta.get('title', '').strip(),
#                 author=pdf_meta.get('author', '').strip(),
#                 subject=pdf_meta.get('subject', '').strip(),
#                 page_count=doc.page_count
#             )

#             if pdf_meta.get('creationDate'):
#                 try:
#                     from datetime import datetime
#                     date_str = pdf_meta['creationDate']
#                     if date_str.startswith('D:'):
#                         date_str = date_str[2:16]
#                         metadata.created_date = datetime.strptime(
#                             date_str, '%Y%m%d%H%M%S')
#                 except Exception as e:
#                     self.logger.debug(f"Failed to parse creation date: {e}")

#             if pdf_meta.get('modDate'):
#                 try:
#                     from datetime import datetime
#                     date_str = pdf_meta['modDate']
#                     if date_str.startswith('D:'):
#                         date_str = date_str[2:16]
#                         metadata.modified_date = datetime.strptime(
#                             date_str, '%Y%m%d%H%M%S')
#                 except Exception as e:
#                     self.logger.debug(
#                         f"Failed to parse modification date: {e}")

#             # Keywords
#             try:
#                 if pdf_meta.get('keywords'):
#                     keywords = [k.strip()
#                                 for k in pdf_meta['keywords'].split(',') if k.strip()]
#                     metadata.keywords = keywords
#             except Exception:
#                 pass

#             # Basic counts across pages
#             try:
#                 total_chars = 0
#                 total_words = 0
#                 for p in range(doc.page_count):
#                     try:
#                         page = doc[p]
#                         text = page.get_text() or ""
#                         total_chars += len(text)
#                         total_words += len(text.split())
#                     except Exception:
#                         continue

#                 metadata.character_count = total_chars
#                 metadata.word_count = total_words
#             except Exception:
#                 pass

#             # PDF-specific metadata
#             try:
#                 metadata.pdf_metadata = {
#                     'producer': pdf_meta.get('producer', ''),
#                     'creator': pdf_meta.get('creator', ''),
#                     'format': pdf_meta.get('format', ''),
#                     'encryption': pdf_meta.get('encryption', ''),
#                     'page_count': doc.page_count,
#                     'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0
#                 }
#             except Exception:
#                 metadata.pdf_metadata = {}

#             return metadata

#         except Exception as e:
#             self.logger.error(f"Metadata extraction failed: {e}")
#             try:
#                 return DocumentMetadata(page_count=doc.page_count if doc else 0)
#             except Exception:
#                 return DocumentMetadata()

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer

# from src.config.logging import LoggerMixin


# class PDFParserAdvanced(BaseParser, LoggerMixin):
#     """Advanced PDF Parser with semantic chuncking."""

#     def __init__(self):
#         super().__init__()
#         self.supported_extensions = ['.pdf']
#         self.mime_types = ['application/pdf']
#         self.settings = get_settings() 

#         self.text_splitter = SentenceTransformer('all-MiniLM-L6-v2')

#     def _get_text_splitter(self):
#         """Initialize RecursiveCharacterTextSplitter with smart separators."""

#         return RecursiveCharacterTextSplitter(
#             chunk_size=self.settings.chunk_size,
#             chunk_overlap=self.settings.chunk_overlap,
#             length_function=len,
#             separators=[
#                 "\n\n\n",  # Multiple newlines (section breaks)
#                 "\n\n",    # Paragraph breaks
#                 "\n",      # Line breaks
#                 ". ",      # Sentence ends
#                 "! ",      # Exclamation
#                 "? ",      # Question
#                 "; ",      # Semicolon
#                 ": ",      # Colon
#                 ", ",      # Comma
#                 " ",       # Space
#                 ""         # Character level (last resort)
#             ],
#             keep_separator=True, 
#             is_separator_regex=False
#         )
    
#     def _get_file_extension(self, file_path):
#         return super()._get_file_extension(file_path)
    
#     def can_parse(self, file_path, mime_type = None) -> bool:
#         """Check if the file can be parsed."""

#         extension = self._get_file_extension(file_path)
#         if extension in self.supported_extensions:
#             return True
#         if mime_type and mime_type in self.mime_types:
#             return True
#         return False
    
#     async def parse(self, file_path: str, document_id: str, options: ProcessingOptions) -> ParseResult:
#         """Parse PDF file with advanced semantic chunking."""

#         start_time = time.time() 

#         try:
#             if not self._validate_file(file_path):
#                 return ParseResult(
#                     success=False, 
#                     error_message=f"File not found or inaccessible: {file_path}"
#                 )
#             options = options or ProcessingOptions()

#             # Initlialize text splitter
#             text_splitter = self._get_text_splitter(
#                 chunk_size=self.settings.chunk_size if options.chunk_size > 0 else 1000,
#                 chunk_overlap=self.settings.chunk_overlap
#             )

#             self.logger.info(f"Starting advanced parsing for {file_path}")

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