from pathlib import Path
from typing import Dict, Type, Optional
from .base_parser import BaseParser
from .pdf_parser import PDFParser 

class ParserRegistry:
    def __init__(self):
        self.parsers: Dict[str, Type] = {}
        self.parser_instances: Dict[str, object] = {}
        self._register_default_parsers()
    
    def _register_default_parsers(self):
        self.parsers['pdf'] = PDFParser

    def get_parser(self, file_path: str, mime_type: Optional[str] = None) -> Optional[BaseParser]:
        extension = Path(file_path).suffix.lower().lstrip('.')
        if extension in self.parsers:
            if extension not in self.parser_instances:
                self.parser_instances[extension] = self.parsers[extension]()
            parser = self.parser_instances[extension]
            if parser.can_parse(file_path, mime_type):
                return parser
        
        # Fall back: iterate all parsers to check
        for parser_class in self.parsers.values():
            parser = parser_class()
            if parser.can_parse(file_path, mime_type):
                return parser
        
        return None
