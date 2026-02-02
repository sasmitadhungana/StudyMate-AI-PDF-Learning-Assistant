"""
Advanced PDF Processing Module
"""
import io
import hashlib
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

import pypdf
import pdfplumber
import fitz  # PyMuPDF
import magic
from pydantic import BaseModel, Field, validator

from config import config

logger = logging.getLogger(__name__)


class PDFMetadata(BaseModel):
    """PDF metadata model"""
    filename: str
    file_size: int
    page_count: int
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    text_length: int = 0
    hash_md5: str
    processed_at: datetime = Field(default_factory=datetime.now)


class PDFChunk(BaseModel):
    """PDF text chunk model"""
    text: str
    page_number: int
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdvancedPDFProcessor:
    """
    Advanced PDF processor with multiple extraction methods,
    caching, and intelligent chunking
    """
    
    def __init__(self):
        self.cache = {}
        self.processed_files = {}
        
    def validate_pdf(self, file_content: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validate PDF file
        
        Args:
            file_content: PDF file bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            if len(file_content) > config.MAX_PDF_SIZE_MB * 1024 * 1024:
                return False, f"File exceeds maximum size of {config.MAX_PDF_SIZE_MB}MB"
            
            # Check file type
            file_type = magic.from_buffer(file_content)
            if "PDF" not in file_type.upper():
                return False, "File is not a valid PDF"
            
            # Try to open with PyPDF
            with io.BytesIO(file_content) as f:
                reader = pypdf.PdfReader(f)
                if len(reader.pages) == 0:
                    return False, "PDF has no pages"
            
            return True, None
            
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return False, str(e)
    
    def extract_metadata(self, file_content: bytes, filename: str) -> PDFMetadata:
        """
        Extract comprehensive PDF metadata
        
        Args:
            file_content: PDF file bytes
            filename: Original filename
            
        Returns:
            PDFMetadata object
        """
        try:
            # Calculate hash
            file_hash = hashlib.md5(file_content).hexdigest()
            
            # Extract with PyPDF
            with io.BytesIO(file_content) as f:
                reader = pypdf.PdfReader(f)
                
                metadata = PDFMetadata(
                    filename=filename,
                    file_size=len(file_content),
                    page_count=len(reader.pages),
                    author=reader.metadata.get("/Author"),
                    title=reader.metadata.get("/Title"),
                    subject=reader.metadata.get("/Subject"),
                    hash_md5=file_hash
                )
                
                # Try to extract dates
                try:
                    if reader.metadata.get("/CreationDate"):
                        metadata.created_date = self._parse_pdf_date(
                            reader.metadata["/CreationDate"]
                        )
                    if reader.metadata.get("/ModDate"):
                        metadata.modified_date = self._parse_pdf_date(
                            reader.metadata["/ModDate"]
                        )
                except:
                    pass
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            # Return basic metadata
            return PDFMetadata(
                filename=filename,
                file_size=len(file_content),
                page_count=0,
                hash_md5=hashlib.md5(file_content).hexdigest()
            )
    
    def extract_text_intelligently(self, file_content: bytes) -> Tuple[str, List[PDFChunk]]:
        """
        Extract text using multiple methods for best results
        
        Args:
            file_content: PDF file bytes
            
        Returns:
            Tuple of (full_text, list_of_chunks)
        """
        full_text = ""
        all_chunks = []
        
        # Method 1: PyMuPDF (best for most PDFs)
        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text:
                        full_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
                        
                        # Create chunks from this page
                        chunks = self._chunk_text(text, page_num + 1)
                        all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: pdfplumber (good for tables)
        if not full_text:
            try:
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            full_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
                            
                            # Create chunks
                            chunks = self._chunk_text(text, page_num + 1)
                            all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 3: PyPDF (fallback)
        if not full_text:
            try:
                with io.BytesIO(file_content) as f:
                    reader = pypdf.PdfReader(f)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            full_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
                            
                            # Create chunks
                            chunks = self._chunk_text(text, page_num + 1)
                            all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"All PDF extraction methods failed: {e}")
                raise
        
        return full_text, all_chunks
    
    def _chunk_text(self, text: str, page_number: int) -> List[PDFChunk]:
        """
        Split text into intelligent chunks
        
        Args:
            text: Text to chunk
            page_number: Page number
            
        Returns:
            List of PDFChunk objects
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > config.PDF_CHUNK_SIZE:
                if current_chunk:  # Save current chunk
                    chunks.append(PDFChunk(
                        text=current_chunk,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will be updated later
                        metadata={"chunk_type": "paragraph"}
                    ))
                    chunk_index += 1
                    current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(PDFChunk(
                text=current_chunk,
                page_number=page_number,
                chunk_index=chunk_index,
                total_chunks=0,
                metadata={"chunk_type": "paragraph"}
            ))
        
        # Update total chunks for each chunk
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _parse_pdf_date(self, pdf_date: str) -> Optional[datetime]:
        """Parse PDF date format"""
        try:
            # PDF dates look like: D:20240101000000
            if pdf_date.startswith("D:"):
                pdf_date = pdf_date[2:]
            
            # Try different formats
            formats = [
                "%Y%m%d%H%M%S",
                "%Y%m%d%H%M",
                "%Y%m%d"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(pdf_date[:len(fmt)], fmt)
                except:
                    continue
            
            return None
        except:
            return None
    
    def process_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a PDF file comprehensively
        
        Args:
            file_content: PDF file bytes
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        
        try:
            # Validate PDF
            is_valid, error = self.validate_pdf(file_content)
            if not is_valid:
                return {
                    "success": False,
                    "error": error,
                    "filename": filename
                }
            
            # Extract metadata
            metadata = self.extract_metadata(file_content, filename)
            
            # Extract text
            full_text, chunks = self.extract_text_intelligently(file_content)
            
            # Update metadata
            metadata.text_length = len(full_text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Cache results
            cache_key = metadata.hash_md5
            self.cache[cache_key] = {
                "metadata": metadata,
                "chunks": chunks,
                "full_text": full_text,
                "processing_time": processing_time
            }
            
            self.processed_files[filename] = cache_key
            
            logger.info(f"Processed PDF: {filename} ({metadata.page_count} pages, "
                       f"{len(chunks)} chunks, {processing_time:.2f}s)")
            
            return {
                "success": True,
                "metadata": metadata.dict(),
                "chunks": [chunk.dict() for chunk in chunks],
                "full_text": full_text,
                "processing_time": processing_time,
                "cache_key": cache_key
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached processing result"""
        return self.cache.get(cache_key)
    
    def clear_cache(self):
        """Clear processing cache"""
        self.cache.clear()
        self.processed_files.clear()
        logger.info("PDF processor cache cleared")
