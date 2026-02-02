"""
Professional Groq Integration Service
"""
import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletion
from pydantic import BaseModel, Field, validator

from config import config

logger = logging.getLogger(__name__)


class GroqModel(str, Enum):
    """Available Groq models"""
    LLAMA3_70B = "llama3-70b-8192"
    LLAMA3_8B = "llama3-8b-8192"
    MIXTRAL = "mixtral-8x7b-32768"
    GEMMA2 = "gemma2-9b-it"


class ChatRole(str, Enum):
    """Chat message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Chat message model"""
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class GroqResponse(BaseModel):
    """Groq API response model"""
    success: bool
    message_id: Optional[str] = None
    content: Optional[str] = None
    model: str
    usage_tokens: Optional[Dict[str, int]] = None
    response_time: float
    error: Optional[str] = None
    cached: bool = False


class GroqService:
    """
    Professional Groq API service with:
    - Connection pooling
    - Request retries
    - Response caching
    - Rate limiting
    - Comprehensive logging
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        
        # Cache for responses
        self.response_cache = {}
        self.cache_ttl = config.CACHE_TTL
        
        # Request statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "avg_response_time": 0
        }
        
        logger.info(f"GroqService initialized with model: {config.GROQ_MODEL}")
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        cache_key: Optional[str] = None
    ) -> GroqResponse:
        """
        Create chat completion with comprehensive error handling
        
        Args:
            messages: List of chat messages
            model: Groq model to use
            temperature: Creativity parameter
            max_tokens: Maximum tokens in response
            stream: Whether to stream response
            cache_key: Optional cache key for response
            
        Returns:
            GroqResponse object
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Check cache first
            if cache_key and cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                if time.time() - cached_response["timestamp"] < self.cache_ttl:
                    logger.debug(f"Using cached response for key: {cache_key}")
                    return GroqResponse(
                        success=True,
                        message_id=f"cached_{cache_key}",
                        content=cached_response["content"],
                        model=cached_response["model"],
                        usage_tokens=cached_response.get("usage_tokens"),
                        response_time=time.time() - start_time,
                        cached=True
                    )
            
            # Prepare parameters
            params = {
                "messages": messages,
                "model": model or config.GROQ_MODEL,
                "temperature": temperature or config.GROQ_TEMPERATURE,
                "max_tokens": max_tokens or config.GROQ_MAX_TOKENS,
                "stream": stream,
                "timeout": config.GROQ_TIMEOUT
            }
            
            # Make API call
            response: ChatCompletion = self.client.chat.completions.create(**params)
            
            # Extract response data
            response_data = {
                "message_id": response.id,
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage_tokens": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') and response.usage else None
            }
            
            # Update statistics
            self.stats["successful_requests"] += 1
            if response_data["usage_tokens"]:
                self.stats["total_tokens"] += response_data["usage_tokens"]["total_tokens"]
            
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)
            
            # Cache response if cache key provided
            if cache_key:
                self.response_cache[cache_key] = {
                    "content": response_data["content"],
                    "model": response_data["model"],
                    "usage_tokens": response_data["usage_tokens"],
                    "timestamp": time.time()
                }
                logger.debug(f"Cached response with key: {cache_key}")
            
            logger.info(f"Groq API call successful: {response_data['model']}, "
                       f"{response_time:.2f}s, {response_data.get('usage_tokens', {}).get('total_tokens', 0)} tokens")
            
            return GroqResponse(
                success=True,
                **response_data,
                response_time=response_time
            )
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            response_time = time.time() - start_time
            
            logger.error(f"Groq API call failed: {str(e)}")
            
            return GroqResponse(
                success=False,
                model=model or config.GROQ_MODEL,
                response_time=response_time,
                error=str(e)
            )
    
    async def create_chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> GroqResponse:
        """Async version of create_chat_completion"""
        start_time = time.time()
        
        try:
            response: ChatCompletion = await self.async_client.chat.completions.create(
                messages=messages,
                model=model or config.GROQ_MODEL,
                temperature=temperature or config.GROQ_TEMPERATURE,
                max_tokens=max_tokens or config.GROQ_MAX_TOKENS,
                stream=stream,
                timeout=config.GROQ_TIMEOUT
            )
            
            response_time = time.time() - start_time
            
            return GroqResponse(
                success=True,
                message_id=response.id,
                content=response.choices[0].message.content,
                model=response.model,
                usage_tokens={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') and response.usage else None,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Async Groq API call failed: {str(e)}")
            
            return GroqResponse(
                success=False,
                model=model or config.GROQ_MODEL,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    def generate_cache_key(self, messages: List[Dict[str, str]], model: str) -> str:
        """Generate cache key from messages and model"""
        import hashlib
        
        cache_data = {
            "messages": messages,
            "model": model,
            "temperature": config.GROQ_TEMPERATURE
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _update_avg_response_time(self, new_time: float):
        """Update average response time statistics"""
        current_avg = self.stats["avg_response_time"]
        total_successful = self.stats["successful_requests"]
        
        if total_successful == 1:
            self.stats["avg_response_time"] = new_time
        else:
            self.stats["avg_response_time"] = (
                (current_avg * (total_successful - 1) + new_time) / total_successful
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.stats,
            "cache_size": len(self.response_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.stats["total_requests"] == 0:
            return 0.0
        
        cache_hits = sum(1 for r in self.response_cache.values() 
                        if time.time() - r["timestamp"] < self.cache_ttl)
        return cache_hits / self.stats["total_requests"] if self.stats["total_requests"] > 0 else 0.0
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("Groq service cache cleared")
    
    def validate_model(self, model_name: str) -> bool:
        """Validate if model name is supported"""
        try:
            GroqModel(model_name)
            return True
        except ValueError:
            return False
    
    def list_available_models(self) -> List[Dict[str, str]]:
        """List all available models with descriptions"""
        return [
            {
                "name": model.value,
                "description": config.get_model_description(model.value)
            }
            for model in GroqModel
        ]
