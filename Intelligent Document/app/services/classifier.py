import os
from typing import Dict, List
import json
from langchain_core.documents import Document
from openai import AsyncOpenAI
from app.models.document import DocumentType

class DocumentClassifier:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.classification_prompt = """
        Analyze the following document content and classify it into one of these categories:
        - TECHNICAL: Technical documentation, architecture docs, system design
        - API_REFERENCE: API documentation, endpoints, parameters
        - SUPPORT_TICKET: Customer support tickets, bug reports
        - POLICY: Legal documents, policies, terms of service
        - TUTORIAL: How-to guides, tutorials, learning materials
        
        Respond with ONLY the category name.
        
        Document content:
        {content}
        """
    
    async def classify(self, document: Document) -> DocumentType:
        try:
            # Get a sample of the document content for classification
            content_sample = self._get_content_sample(document.page_content)
            
            # Call OpenAI API for classification
            response = await self.client.chat.completions.create(
                model=os.getenv("CLASSIFICATION_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a document classification expert."},
                    {"role": "user", "content": self.classification_prompt.format(content=content_sample)}
                ],
                temperature=0,
                max_tokens=10
            )
            
            # Parse the response
            classification = response.choices[0].message.content.strip()
            
            # Map the response to DocumentType
            try:
                return DocumentType[classification]
            except KeyError:
                return DocumentType.UNKNOWN
                
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return DocumentType.UNKNOWN
    
    def _get_content_sample(self, content: str, max_chars: int = 1000) -> str:
        """Get a representative sample of the document content."""
        if len(content) <= max_chars:
            return content
            
        # Take beginning, middle and end samples
        sample_size = max_chars // 3
        start = content[:sample_size]
        middle_start = (len(content) - sample_size) // 2
        middle = content[middle_start:middle_start + sample_size]
        end = content[-sample_size:]
        
        return f"{start}\n...\n{middle}\n...\n{end}" 