"""
Advanced Question Answering Engine with RAG capabilities
"""
import json
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import re

from pydantic import BaseModel, Field, validator
import numpy as np
from sentence_transformers import SentenceTransformer

from config import config
from src.core.groq_service import GroqService, ChatMessage, ChatRole
from src.core.pdf_processor import PDFChunk, AdvancedPDFProcessor

logger = logging.getLogger(__name__)


class QAMode(str, Enum):
    """QA modes"""
    SIMPLE = "simple"           # Direct question to LLM
    RAG = "rag"                 # Retrieve relevant chunks first
    SUMMARIZE = "summarize"     # Summarization mode
    ANALYZE = "analyze"         # Deep analysis mode


class QARequest(BaseModel):
    """QA request model"""
    question: str
    context_chunks: Optional[List[Dict[str, Any]]] = None
    mode: QAMode = QAMode.RAG
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)
    include_sources: bool = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class QAResponse(BaseModel):
    """QA response model"""
    success: bool
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)
    processing_time: float
    tokens_used: Optional[int] = None
    model: str
    mode: QAMode
    confidence_score: Optional[float] = None
    error: Optional[str] = None


class ConversationMemory:
    """Conversation memory for context retention"""
    
    def __init__(self, max_messages: int = 20):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
        self.summary = ""
    
    def add_message(self, role: ChatRole, content: str):
        """Add message to conversation"""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Update summary periodically
        if len(self.messages) % 5 == 0:
            self._update_summary()
    
    def _update_summary(self):
        """Update conversation summary"""
        if len(self.messages) > 10:
            summary_text = " ".join([msg.content[:100] for msg in self.messages[-10:]])
            self.summary = summary_text[:500] + "..." if len(summary_text) > 500 else summary_text
    
    def get_context(self, max_tokens: int = 500) -> str:
        """Get conversation context for LLM"""
        if not self.messages:
            return ""
        
        context_messages = []
        token_count = 0
        
        # Add messages from end (most recent) to beginning
        for msg in reversed(self.messages):
            msg_text = f"{msg.role}: {msg.content}"
            msg_tokens = len(msg_text.split())  # Rough estimate
            
            if token_count + msg_tokens > max_tokens:
                break
            
            context_messages.insert(0, msg_text)
            token_count += msg_tokens
        
        return "\n".join(context_messages)
    
    def clear(self):
        """Clear conversation memory"""
        self.messages.clear()
        self.summary = ""


class SemanticSearch:
    """Semantic search for PDF chunks"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_embeddings = {}
        self.chunk_metadata = {}
    
    def add_chunks(self, chunks: List[PDFChunk]):
        """Add chunks with embeddings"""
        for chunk in chunks:
            chunk_id = f"{chunk.page_number}_{chunk.chunk_index}"
            
            # Create embedding
            embedding = self.embedding_model.encode(chunk.text)
            self.chunk_embeddings[chunk_id] = embedding
            
            # Store metadata
            self.chunk_metadata[chunk_id] = {
                "text": chunk.text,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata
            }
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if not self.chunk_embeddings:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = []
        for chunk_id, embedding in self.chunk_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((chunk_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for chunk_id, similarity in similarities[:top_k]:
            metadata = self.chunk_metadata[chunk_id].copy()
            metadata["similarity_score"] = float(similarity)
            metadata["chunk_id"] = chunk_id
            results.append(metadata)
        
        return results
    
    def clear(self):
        """Clear all chunks"""
        self.chunk_embeddings.clear()
        self.chunk_metadata.clear()


class QAEngine:
    """
    Advanced Question Answering Engine with:
    - Semantic search (RAG)
    - Conversation memory
    - Multi-mode operation
    - Confidence scoring
    """
    
    def __init__(self, groq_service: Optional[GroqService] = None):
        self.groq_service = groq_service or GroqService()
        self.semantic_search = SemanticSearch()
        self.conversation_memory = ConversationMemory()
        self.pdf_processor = AdvancedPDFProcessor()
        
        # Statistics
        self.stats = {
            "total_questions": 0,
            "successful_answers": 0,
            "total_processing_time": 0,
            "avg_confidence": 0
        }
        
        # Prompt templates
        self.prompt_templates = {
            QAMode.SIMPLE: """You are an AI assistant for educational PDFs.
Answer the following question clearly and helpfully.

Question: {question}

Answer:""",
            
            QAMode.RAG: """You are an AI assistant for educational PDFs.
Use the following context from PDF documents to answer the question.
If the answer cannot be found in the context, say so clearly.

Context from PDFs:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Cite page numbers when mentioned in context
3. If context doesn't contain answer, say: "I don't have enough information about this in the provided documents"
4. Be educational and clear

Answer:""",
            
            QAMode.SUMMARIZE: """You are an AI assistant for summarizing educational PDFs.
Summarize the following content from PDF documents.

PDF Content:
{context}

Provide a comprehensive summary covering:
1. Main topics and themes
2. Key concepts and definitions
3. Important examples or case studies
4. Conclusions or takeaways

Summary:""",
            
            QAMode.ANALYZE: """You are an AI educational analyst.
Analyze the following PDF content and answer the question with deep insights.

PDF Content:
{context}

Question: {question}

Provide a detailed analysis including:
1. Key findings from the content
2. Connections between concepts
3. Implications or applications
4. Potential limitations or gaps

Analysis:"""
        }
        
        logger.info("QA Engine initialized")
    
    def add_documents(self, pdf_chunks: List[PDFChunk]):
        """Add PDF chunks for semantic search"""
        self.semantic_search.add_chunks(pdf_chunks)
        logger.info(f"Added {len(pdf_chunks)} chunks to QA engine")
    
    def ask_question(self, request: QARequest) -> QAResponse:
        """
        Process a question with advanced QA capabilities
        
        Args:
            request: QARequest object
            
        Returns:
            QAResponse object
        """
        start_time = datetime.now()
        self.stats["total_questions"] += 1
        
        try:
            # Step 1: Retrieve relevant context if in RAG mode
            context = ""
            sources = []
            
            if request.mode in [QAMode.RAG, QAMode.SUMMARIZE, QAMode.ANALYZE]:
                if request.context_chunks:
                    # Use provided chunks
                    for chunk in request.context_chunks:
                        context += f"\n\nPage {chunk.get('page_number', 'N/A')}:\n{chunk.get('text', '')}"
                        sources.append({
                            "page": chunk.get("page_number"),
                            "text_preview": chunk.get("text", "")[:200] + "...",
                            "chunk_index": chunk.get("chunk_index")
                        })
                else:
                    # Use semantic search to find relevant chunks
                    search_results = self.semantic_search.search(request.question, top_k=5)
                    for result in search_results:
                        context += f"\n\nPage {result.get('page_number', 'N/A')}:\n{result.get('text', '')}"
                        sources.append({
                            "page": result.get("page_number"),
                            "text_preview": result.get("text", "")[:200] + "...",
                            "similarity_score": result.get("similarity_score"),
                            "chunk_id": result.get("chunk_id")
                        })
            
            # Step 2: Get conversation context
            conversation_context = self.conversation_memory.get_context()
            
            # Step 3: Prepare messages for LLM
            prompt_template = self.prompt_templates.get(request.mode, self.prompt_templates[QAMode.RAG])
            
            if request.mode == QAMode.SIMPLE:
                prompt = prompt_template.format(question=request.question)
            else:
                prompt = prompt_template.format(context=context, question=request.question)
            
            # Add conversation context if available
            if conversation_context:
                prompt = f"Previous conversation:\n{conversation_context}\n\n{prompt}"
            
            messages = [
                {"role": "system", "content": "You are a helpful educational assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Step 4: Generate cache key
            cache_key = self.groq_service.generate_cache_key(messages, config.GROQ_MODEL)
            
            # Step 5: Call Groq API
            response = self.groq_service.create_chat_completion(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                cache_key=cache_key
            )
            
            if not response.success:
                raise Exception(f"Groq API error: {response.error}")
            
            # Step 6: Process response
            answer = response.content
            confidence_score = self._calculate_confidence(answer, context if context else request.question)
            
            # Step 7: Generate suggested questions
            suggested_questions = self._generate_suggested_questions(answer, request.question)
            
            # Step 8: Update conversation memory
            self.conversation_memory.add_message(ChatRole.USER, request.question)
            self.conversation_memory.add_message(ChatRole.ASSISTANT, answer)
            
            # Step 9: Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["successful_answers"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_confidence"] = (
                (self.stats["avg_confidence"] * (self.stats["successful_answers"] - 1) + confidence_score)
                / self.stats["successful_answers"]
            )
            
            # Step 10: Prepare response
            qa_response = QAResponse(
                success=True,
                answer=answer,
                sources=sources if request.include_sources else [],
                suggested_questions=suggested_questions,
                processing_time=processing_time,
                tokens_used=response.usage_tokens.get("total_tokens") if response.usage_tokens else None,
                model=response.model,
                mode=request.mode,
                confidence_score=confidence_score
            )
            
            logger.info(f"QA processed: mode={request.mode}, time={processing_time:.2f}s, "
                       f"confidence={confidence_score:.2f}, tokens={response.usage_tokens.get('total_tokens') if response.usage_tokens else 'N/A'}")
            
            return qa_response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"QA processing failed: {str(e)}")
            
            return QAResponse(
                success=False,
                processing_time=processing_time,
                model=config.GROQ_MODEL,
                mode=request.mode,
                error=str(e)
            )
    
    def _calculate_confidence(self, answer: str, context: str) -> float:
        """Calculate confidence score for answer"""
        # Simple confidence calculation based on:
        # 1. Answer length (too short might be uncertain)
        # 2. Presence of uncertainty phrases
        # 3. Whether answer references context
        
        score = 0.7  # Base confidence
        
        # Adjust based on answer length
        if len(answer.split()) < 10:
            score -= 0.2
        elif len(answer.split()) > 50:
            score += 0.1
        
        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot answer",
            "the context doesn't", "no information", "not mentioned"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            score -= 0.3
        
        # Check if answer references context (for RAG mode)
        if context and any(word in answer_lower for word in ["page", "according to", "the document", "pdf"]):
            score += 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _generate_suggested_questions(self, answer: str, original_question: str) -> List[str]:
        """Generate suggested follow-up questions"""
        suggestions = []
        
        # Basic follow-ups based on answer
        if "definition" in original_question.lower() or "what is" in original_question.lower():
            suggestions.extend([
                "Can you provide an example of this?",
                "How is this concept applied in practice?",
                "What are the key characteristics?"
            ])
        elif "how" in original_question.lower():
            suggestions.extend([
                "What are the prerequisites for this?",
                "What are the alternatives to this approach?",
                "What are the limitations?"
            ])
        elif "why" in original_question.lower():
            suggestions.extend([
                "What are the implications of this?",
                "How does this compare to other approaches?",
                "What evidence supports this?"
            ])
        
        # Add general suggestions
        general_suggestions = [
            "Can you summarize the main points?",
            "What are the key takeaways?",
            "How does this relate to other topics?",
            "Can you provide more details?"
        ]
        
        suggestions.extend(general_suggestions[:2])
        
        return suggestions[:5]  # Return at most 5 suggestions
    
    async def ask_question_async(self, request: QARequest) -> QAResponse:
        """Async version of ask_question"""
        # Similar implementation but async
        # For brevity, implementing sync version for now
        return self.ask_question(request)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory.clear()
        logger.info("QA Engine memory cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        avg_processing_time = (
            self.stats["total_processing_time"] / self.stats["successful_answers"]
            if self.stats["successful_answers"] > 0 else 0
        )
        
        return {
            **self.stats,
            "avg_processing_time": avg_processing_time,
            "success_rate": (
                self.stats["successful_answers"] / self.stats["total_questions"]
                if self.stats["total_questions"] > 0 else 0
            ),
            "conversation_messages": len(self.conversation_memory.messages),
            "chunks_indexed": len(self.semantic_search.chunk_embeddings)
        }
    
    def export_conversation(self) -> List[Dict[str, Any]]:
        """Export conversation history"""
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in self.conversation_memory.messages
        ]
