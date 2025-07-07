import uuid
from typing import List, Tuple, Any
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import (
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path

from app.models.document import DocumentType, ProcessedDocument, DocumentChunk
from app.services.classifier import DocumentClassifier
from app.services.embeddings import EmbeddingService

class DocumentProcessor:
    def __init__(self):
        self.classifier = DocumentClassifier()
        self.embedding_service = EmbeddingService()
        
    async def process(self, content: bytes, filename: str, content_type: str) -> ProcessedDocument:
        # Determine file type and load content
        doc = self._load_document(content, filename, content_type)
        
        # Classify document
        doc_type = await self.classifier.classify(doc)
        
        # Apply appropriate chunking strategy
        chunks = self._chunk_document(doc, doc_type)
        
        # Generate embeddings for chunks
        embedded_chunks = await self._embed_chunks(chunks)
        
        return ProcessedDocument(
            id=str(uuid.uuid4()),
            filename=filename,
            doc_type=doc_type,
            chunks=embedded_chunks,
            total_chunks=len(embedded_chunks),
            embedding_model=self.embedding_service.model_name,
            chunking_strategy=self._get_chunking_strategy(doc_type)
        )
    
    def _load_document(self, content: bytes, filename: str, content_type: str) -> Document:
        temp_path = Path(f"/tmp/{filename}")
        temp_path.write_bytes(content)
        
        if content_type == "application/pdf":
            loader = PyPDFLoader(str(temp_path))
            doc = loader.load()[0]
        elif content_type in ["text/markdown", "text/x-markdown"]:
            loader = UnstructuredMarkdownLoader(str(temp_path))
            doc = loader.load()[0]
        elif content_type == "text/html":
            loader = UnstructuredHTMLLoader(str(temp_path))
            doc = loader.load()[0]
        else:
            # Default to treating as plain text
            doc = Document(page_content=content.decode(), metadata={"source": filename})
        
        temp_path.unlink()  # Clean up temp file
        return doc
    
    def _chunk_document(self, doc: Document, doc_type: DocumentType) -> List[DocumentChunk]:
        if doc_type == DocumentType.TECHNICAL or doc_type == DocumentType.API_REFERENCE:
            return self._chunk_technical_doc(doc)
        elif doc_type == DocumentType.POLICY:
            return self._chunk_policy_doc(doc)
        elif doc_type == DocumentType.SUPPORT_TICKET:
            return self._chunk_support_ticket(doc)
        elif doc_type == DocumentType.TUTORIAL:
            return self._chunk_tutorial(doc)
        else:
            return self._chunk_default(doc)
    
    def _chunk_technical_doc(self, doc: Document) -> List[DocumentChunk]:
        chunks = []
        # Preserve code blocks
        code_pattern = r"```[\s\S]*?```"
        code_blocks = re.finditer(code_pattern, doc.page_content)
        
        # Split text between code blocks
        last_end = 0
        for i, match in enumerate(code_blocks):
            # Chunk text before code block
            if match.start() > last_end:
                text_chunks = self._create_text_chunks(
                    doc.page_content[last_end:match.start()]
                )
                chunks.extend(text_chunks)
            
            # Add code block as single chunk
            chunks.append(
                DocumentChunk(
                    content=match.group(),
                    chunk_type="code",
                    chunk_index=len(chunks),
                    metadata={"code_block_index": i}
                )
            )
            last_end = match.end()
        
        # Chunk remaining text
        if last_end < len(doc.page_content):
            text_chunks = self._create_text_chunks(
                doc.page_content[last_end:]
            )
            chunks.extend(text_chunks)
        
        return chunks
    
    def _chunk_policy_doc(self, doc: Document) -> List[DocumentChunk]:
        # Use header-based splitting for policies
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        splits = markdown_splitter.split_text(doc.page_content)
        
        chunks = []
        for i, split in enumerate(splits):
            chunks.append(
                DocumentChunk(
                    content=split.page_content,
                    chunk_type="policy_section",
                    chunk_index=i,
                    metadata={
                        "section_name": split.metadata.get("Header 1", ""),
                        "subsection": split.metadata.get("Header 2", ""),
                    }
                )
            )
        
        return chunks
    
    def _chunk_support_ticket(self, doc: Document) -> List[DocumentChunk]:
        # Assume JSON format for support tickets
        try:
            ticket_data = json.loads(doc.page_content)
            chunks = []
            
            # Create chunks for different ticket sections
            if "description" in ticket_data:
                chunks.append(
                    DocumentChunk(
                        content=ticket_data["description"],
                        chunk_type="ticket_description",
                        chunk_index=0,
                        metadata={"ticket_id": ticket_data.get("id")}
                    )
                )
            
            if "comments" in ticket_data:
                for i, comment in enumerate(ticket_data["comments"]):
                    chunks.append(
                        DocumentChunk(
                            content=comment["content"],
                            chunk_type="ticket_comment",
                            chunk_index=len(chunks),
                            metadata={
                                "author": comment.get("author"),
                                "timestamp": comment.get("timestamp"),
                                "comment_index": i
                            }
                        )
                    )
            
            return chunks
        except json.JSONDecodeError:
            # Fallback to default chunking if not JSON
            return self._chunk_default(doc)
    
    def _chunk_tutorial(self, doc: Document) -> List[DocumentChunk]:
        # Similar to technical docs but with different metadata
        return self._chunk_technical_doc(doc)
    
    def _chunk_default(self, doc: Document) -> List[DocumentChunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = splitter.split_text(doc.page_content)
        
        return [
            DocumentChunk(
                content=split,
                chunk_type="text",
                chunk_index=i,
                metadata={}
            )
            for i, split in enumerate(splits)
        ]
    
    def _create_text_chunks(self, text: str) -> List[DocumentChunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = splitter.split_text(text)
        
        return [
            DocumentChunk(
                content=split,
                chunk_type="text",
                chunk_index=i,
                metadata={}
            )
            for i, split in enumerate(splits)
        ]
    
    async def _embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        for chunk in chunks:
            chunk.embedding = await self.embedding_service.embed_text(chunk.content)
        return chunks
    
    def _get_chunking_strategy(self, doc_type: DocumentType) -> str:
        strategy_map = {
            DocumentType.TECHNICAL: "code_aware",
            DocumentType.API_REFERENCE: "code_aware",
            DocumentType.POLICY: "header_based",
            DocumentType.SUPPORT_TICKET: "conversation_flow",
            DocumentType.TUTORIAL: "code_aware",
            DocumentType.UNKNOWN: "recursive_character"
        }
        return strategy_map[doc_type] 