"""LLM Service: High-level interface for language model operations."""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..config import get_settings, LoggerMixin
from ..models.response import Response, ResponseType, ResponseMetrics, SourceAttribution


class LLMServiceError(Exception):
    """Custom exception for LLM service operations."""
    pass


class LLMService(LoggerMixin):
    """High-level interface for language model operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.gemini_api_key
        self.model_name = self.settings.gemini_model
        
        if not self.api_key:
            raise LLMServiceError("GEMINI_API_KEY not configured")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Model configuration
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.settings.gemini_temperature,
            max_output_tokens=self.settings.gemini_max_tokens,
            top_p=0.8,
            top_k=40
        )
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        # Prompt templates
        self.templates = {
            'qa': """Based on the provided context, please answer the following question:

                Question: {question}

                Context:
                {context}

                Instructions:
                - Provide a clear, direct answer based only on the context
                - Include specific details and examples where relevant
                - Use numbered citations [1], [2], etc. to reference sources
                - If the context doesn't contain enough information, state this clearly

                Answer:""",
            
            'summary': """Please provide a comprehensive summary of the following information:

                Content to summarize:
                {content}

                Instructions:
                - Create a well-structured summary with key points
                - Organize information logically
                - Include important details and statistics
                - Keep it concise but comprehensive

                Summary:""",
            
            'analysis': """Please analyze the following information and provide insights:

                Information:
                {content}

                Question/Focus: {question}

                Instructions:
                - Provide detailed analysis addressing the question
                - Identify key patterns, trends, or insights
                - Support conclusions with specific evidence
                - Be objective and factual

                Analysis:"""
        }
        
        # Statistics
        self.stats = {
            'requests_made': 0,
            'tokens_generated': 0,
            'average_response_time': 0.0,
            'errors': 0,
            'last_activity': None
        }
    
    async def generate_answer(
        self,
        question: str,
        context: str,
        sources: Optional[List[SourceAttribution]] = None,
        response_type: ResponseType = ResponseType.DIRECT_ANSWER
    ) -> Response:
        """Generate a comprehensive answer with source attribution."""
        
        start_time = time.time()
        
        try:
            # Select appropriate template
            template = self._get_template_for_response_type(response_type)
            
            # Format prompt
            prompt = template.format(
                question=question,
                context=context,
                content=context
            )
            
            # Generate response
            llm_result = await self._generate_with_retry(prompt)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            
            metrics = ResponseMetrics(
                retrieval_time=0.0,  # Set by calling service
                generation_time=llm_result['generation_time'],
                total_response_time=generation_time,
                source_documents_count=len(sources) if sources else 0,
                chunks_processed=len(sources) if sources else 0,
                tokens_generated=llm_result['token_count'],
                avg_source_similarity=self._calculate_avg_similarity(sources),
                confidence_score=self._calculate_confidence_score(llm_result['text'], sources),
                llm_api_calls=1,
                tokens_consumed=llm_result['token_count']
            )
            
            # Create response object
            response = Response(
                answer=llm_result['text'],
                response_type=response_type,
                sources=sources or [],
                metrics=metrics,
                model_used=self.model_name,
                temperature=self.generation_config.temperature
            )
            
            # Update confidence based on metrics
            response.update_confidence()
            
            # Update statistics
            self._update_stats(generation_time, llm_result['token_count'])
            
            return response
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"Answer generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise LLMServiceError(error_msg)
    
    async def generate_summary(self, content: str, max_length: int = 500) -> str:
        """Generate a summary of the provided content."""
        
        try:
            template = self.templates['summary']
            prompt = template.format(content=content[:4000])  # Limit input length
            
            result = await self._generate_with_retry(prompt, max_tokens=max_length)
            return result['text']
            
        except Exception as e:
            error_msg = f"Summary generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise LLMServiceError(error_msg)
    
    async def analyze_content(self, content: str, question: str) -> str:
        """Perform analysis on content based on a specific question."""
        
        try:
            template = self.templates['analysis']
            prompt = template.format(content=content[:4000], question=question)
            
            result = await self._generate_with_retry(prompt)
            return result['text']
            
        except Exception as e:
            error_msg = f"Content analysis failed: {str(e)}"
            self.logger.error(error_msg)
            raise LLMServiceError(error_msg)
    
    async def _generate_with_retry(
        self, 
        prompt: str, 
        max_retries: int = 3,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate response with retry logic."""
        
        if max_tokens:
            # Temporarily override max tokens
            original_max_tokens = self.generation_config.max_output_tokens
            self.generation_config.max_output_tokens = max_tokens
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Generate response
                response = await self._async_generate(prompt)
                
                generation_time = time.time() - start_time
                
                result = {
                    'text': response.text,
                    'generation_time': generation_time,
                    'token_count': self._estimate_token_count(response.text),
                    'finish_reason': getattr(response.candidates[0], 'finish_reason', None),
                    'safety_ratings': getattr(response.candidates[0], 'safety_ratings', [])
                }
                
                # Restore original max tokens if changed
                if max_tokens:
                    self.generation_config.max_output_tokens = original_max_tokens
                
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise LLMServiceError(f"Generation failed after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        raise LLMServiceError("Unexpected error in generation")
    
    async def _async_generate(self, prompt: str):
        """Async wrapper for Gemini generation."""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.model.generate_content, prompt)
    
    def _get_template_for_response_type(self, response_type: ResponseType) -> str:
        """Get appropriate template for response type."""
        
        if response_type == ResponseType.SUMMARIZED:
            return self.templates['summary']
        elif response_type in [ResponseType.COMPARATIVE, ResponseType.INSTRUCTIONAL]:
            return self.templates['analysis']
        else:
            return self.templates['qa']
    
    def _calculate_avg_similarity(self, sources: Optional[List[SourceAttribution]]) -> float:
        """Calculate average similarity score from sources."""
        
        if not sources:
            return 0.0
        
        return sum(source.similarity_score for source in sources) / len(sources)
    
    def _calculate_confidence_score(
        self, 
        response_text: str, 
        sources: Optional[List[SourceAttribution]]
    ) -> float:
        """Calculate confidence score based on response and sources."""
        
        score = 0.0
        
        # Base score from source quality
        if sources:
            avg_similarity = self._calculate_avg_similarity(sources)
            score += avg_similarity * 0.4
            
            # Bonus for multiple sources
            if len(sources) > 1:
                score += 0.2
        
        # Response quality indicators
        if len(response_text) > 50:  # Substantial response
            score += 0.2
        
        if '[1]' in response_text:  # Has citations
            score += 0.2
        
        return min(score, 1.0)
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def _update_stats(self, generation_time: float, token_count: int) -> None:
        """Update service statistics."""
        
        self.stats['requests_made'] += 1
        self.stats['tokens_generated'] += token_count
        self.stats['last_activity'] = datetime.utcnow()
        
        # Update average response time
        if self.stats['requests_made'] == 1:
            self.stats['average_response_time'] = generation_time
        else:
            alpha = 0.1
            self.stats['average_response_time'] = (
                alpha * generation_time + 
                (1 - alpha) * self.stats['average_response_time']
            )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        
        stats = self.stats.copy()
        stats.update({
            'model_name': self.model_name,
            'temperature': self.generation_config.temperature,
            'max_tokens': self.generation_config.max_output_tokens,
            'success_rate': (
                (self.stats['requests_made'] - self.stats['errors']) / 
                max(1, self.stats['requests_made']) * 100
            )
        })
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM service."""
        
        health_status = {
            'healthy': True,
            'service': 'LLMService',
            'model': self.model_name,
            'errors': []
        }
        
        try:
            # Test generation with simple prompt
            test_result = await self._generate_with_retry(
                "Respond with 'OK' to confirm the service is working.",
                max_tokens=10
            )
            
            health_status.update({
                'test_successful': True,
                'response_time': test_result['generation_time'],
                'test_response': test_result['text'][:50]
            })
            
        except Exception as e:
            health_status.update({
                'healthy': False,
                'test_successful': False,
                'errors': [str(e)]
            })
        
        return health_status
