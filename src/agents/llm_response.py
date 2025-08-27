"""Advanced LLM Response Agent with intelligent answer generation and source attribution."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..config import get_settings
from ..models.query import Query, SearchResult, RetrievalResults
from ..models.response import (
    Response, ResponseType, ConfidenceLevel, SourceAttribution, 
    ResponseMetrics, ConversationTurn, Conversation
)
from ..mcp.protocol import (
    MCPMessage, MessageType, AgentType, create_response_message, 
    create_error_message
)
from .base_agent import BaseAgent


class LLMResponseError(Exception):
    """Custom exception for LLM response generation errors."""
    pass


class PromptTemplate:
    """Manages prompt templates for different response types."""
    
    def __init__(self):
        self.templates = {
            'direct_answer': """Based on the provided context, please answer the following question:

Question: {query}

Context:
{context}

Instructions:
- Provide a clear, direct answer based on the context
- Include specific details and examples where relevant
- If the context doesn't contain enough information, state this clearly
- Use numbered citations [1], [2], etc. to reference sources
- Be concise but comprehensive

Answer:""",

            'summarized': """Please provide a comprehensive summary based on the following question and context:

Question: {query}

Context:
{context}

Instructions:
- Create a well-structured summary that addresses the question
- Organize information logically with clear sections
- Include key points, statistics, and important details
- Use numbered citations [1], [2], etc. to reference sources
- Highlight the most important findings

Summary:""",

            'comparative': """Please provide a detailed comparison based on the following question and context:

Question: {query}

Context:
{context}

Instructions:
- Compare and contrast the relevant items/concepts
- Highlight similarities and differences clearly
- Use tables or structured format where appropriate
- Include specific data points and examples
- Use numbered citations [1], [2], etc. to reference sources
- Draw clear conclusions from the comparison

Comparison:""",

            'instructional': """Please provide step-by-step instructions based on the following question and context:

Question: {query}

Context:
{context}

Instructions:
- Create clear, actionable step-by-step instructions
- Number each step clearly
- Include necessary details and warnings
- Reference specific information from the context
- Use numbered citations [1], [2], etc. to reference sources
- Ensure instructions are practical and implementable

Instructions:""",

            'no_context': """I don't have enough relevant information in my knowledge base to answer your question: "{query}"

To get a helpful answer, you might want to:
- Upload relevant documents that contain information about this topic
- Rephrase your question to be more specific
- Ask about a related topic that might be covered in the available documents

Is there anything else I can help you with based on the available information?"""
        }
    
    def get_template(self, response_type: ResponseType) -> str:
        """Get prompt template for response type."""
        return self.templates.get(response_type.value, self.templates['direct_answer'])
    
    def format_prompt(
        self, 
        template: str, 
        query: str, 
        context: str, 
        conversation_history: List[str] = None
    ) -> str:
        """Format prompt template with actual content."""
        
        formatted_prompt = template.format(
            query=query,
            context=context
        )
        
        # Add conversation history if available
        if conversation_history:
            history_text = "\n".join([
                f"Previous Q&A {i+1}: {qa}" 
                for i, qa in enumerate(conversation_history[-3:])  # Last 3 exchanges
            ])
            
            formatted_prompt = f"""Previous Conversation:
{history_text}

{formatted_prompt}"""
        
        return formatted_prompt


class SourceAttributor:
    """Handles source attribution and citation management."""
    
    def __init__(self):
        self.citation_pattern = re.compile(r'\[(\d+)\]')
    
    def create_source_attributions(self, search_results: List[SearchResult]) -> List[SourceAttribution]:
        """Create source attribution objects from search results."""
        
        attributions = []
        
        for i, result in enumerate(search_results, 1):
            attribution = SourceAttribution(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_name=result.document_metadata.get('title', f'Document {result.document_id}'),
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                relevant_content=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                similarity_score=result.similarity_score,
                citation_number=i
            )
            
            attributions.append(attribution)
        
        return attributions
    
    def format_context_with_citations(self, search_results: List[SearchResult]) -> str:
        """Format search results as context with citation numbers."""
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            # Add document info
            doc_info = f"Document: {result.document_metadata.get('title', 'Unknown Document')}"
            if result.page_number:
                doc_info += f", Page {result.page_number}"
            
            # Format content with citation
            content_with_citation = f"[{i}] {result.content}"
            
            context_part = f"{doc_info}\n{content_with_citation}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def validate_citations(self, response_text: str, max_citations: int) -> List[str]:
        """Validate and extract citations from response text."""
        
        issues = []
        
        # Find all citations in response
        citations_found = self.citation_pattern.findall(response_text)
        
        if not citations_found:
            issues.append("No citations found in response")
            return issues
        
        # Check citation numbers
        citation_numbers = [int(c) for c in citations_found]
        
        # Check for invalid citation numbers
        invalid_citations = [c for c in citation_numbers if c > max_citations or c < 1]
        if invalid_citations:
            issues.append(f"Invalid citation numbers found: {invalid_citations}")
        
        # Check for missing citations (gaps in sequence)
        expected_citations = set(range(1, max_citations + 1))
        found_citations = set(citation_numbers)
        missing_citations = expected_citations - found_citations
        
        if missing_citations:
            issues.append(f"Some sources not cited: {sorted(missing_citations)}")
        
        return issues


class ResponseQualityAnalyzer:
    """Analyzes and scores response quality."""
    
    def __init__(self):
        self.quality_indicators = {
            'completeness': {
                'has_introduction': r'^.{20,}',  # At least 20 chars start
                'has_conclusion': r'.{20,}$',    # At least 20 chars end
                'sufficient_length': lambda text: len(text.split()) >= 50
            },
            'accuracy': {
                'has_citations': r'\[\d+\]',
                'specific_details': r'\d+',  # Contains numbers/statistics
                'factual_language': r'\b(according to|based on|shows that|indicates)\b'
            },
            'relevance': {
                'addresses_query': lambda text, query: any(
                    word.lower() in text.lower() 
                    for word in query.split() 
                    if len(word) > 3
                ),
                'stays_on_topic': lambda text: len(text.split()) < 1000  # Not too verbose
            }
        }
    
    def analyze_response_quality(
        self, 
        response_text: str, 
        query: str, 
        sources: List[SourceAttribution]
    ) -> Dict[str, float]:
        """Analyze response quality and return scores."""
        
        scores = {}
        
        # Completeness score
        completeness_score = 0
        if re.search(self.quality_indicators['completeness']['has_introduction'], response_text):
            completeness_score += 0.3
        if re.search(self.quality_indicators['completeness']['has_conclusion'], response_text):
            completeness_score += 0.3
        if self.quality_indicators['completeness']['sufficient_length'](response_text):
            completeness_score += 0.4
        
        scores['completeness'] = completeness_score
        
        # Accuracy score
        accuracy_score = 0
        if re.search(self.quality_indicators['accuracy']['has_citations'], response_text):
            accuracy_score += 0.4
        if re.search(self.quality_indicators['accuracy']['specific_details'], response_text):
            accuracy_score += 0.3
        if re.search(self.quality_indicators['accuracy']['factual_language'], response_text):
            accuracy_score += 0.3
        
        scores['accuracy'] = accuracy_score
        
        # Relevance score
        relevance_score = 0
        if self.quality_indicators['relevance']['addresses_query'](response_text, query):
            relevance_score += 0.6
        if self.quality_indicators['relevance']['stays_on_topic'](response_text):
            relevance_score += 0.4
        
        scores['relevance'] = relevance_score
        
        return scores
    
    def determine_confidence_level(self, quality_scores: Dict[str, float], source_count: int) -> ConfidenceLevel:
        """Determine confidence level based on quality scores."""
        
        average_score = sum(quality_scores.values()) / len(quality_scores)
        
        if average_score >= 0.8 and source_count >= 2:
            return ConfidenceLevel.HIGH
        elif average_score >= 0.6 and source_count >= 1:
            return ConfidenceLevel.MEDIUM
        elif average_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN


class GeminiLLMService:
    """Google Gemini LLM service with advanced configuration."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.google_gemini_model
        self.api_key = self.settings.google_gemini_api_key
        
        # Configure Gemini
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            raise LLMResponseError("GEMINI_API_KEY not configured")
        
        # Model configuration
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.settings.google_gemini_temperature,
            max_output_tokens=self.settings.google_gemini_max_tokens,
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
        
        # Statistics
        self.stats = {
            'requests_made': 0,
            'tokens_generated': 0,
            'average_response_time': 0.0,
            'errors': 0,
            'safety_blocks': 0
        }
    
    async def generate_response(
        self, 
        prompt: str, 
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Generate response using Gemini with retry logic."""
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Generate response
                response = await self._generate_with_timeout(prompt, timeout=30.0)
                
                generation_time = time.time() - start_time
                
                # Process response
                result = {
                    'text': response.text,
                    'generation_time': generation_time,
                    'attempt': attempt + 1,
                    'finish_reason': getattr(response.candidates[0], 'finish_reason', None),
                    'safety_ratings': getattr(response.candidates[0], 'safety_ratings', []),
                    'token_count': self._estimate_token_count(response.text)
                }
                
                # Update statistics
                self.stats['requests_made'] += 1
                self.stats['tokens_generated'] += result['token_count']
                self._update_average_time(generation_time)
                
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.stats['errors'] += 1
                    raise LLMResponseError(f"LLM generation failed after {max_retries} attempts: {str(e)}")
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise LLMResponseError("Unexpected error in response generation")
    
    async def _generate_with_timeout(self, prompt: str, timeout: float):
        """Generate response with timeout."""
        
        try:
            # Create async task for generation
            task = asyncio.create_task(self._async_generate(prompt))
            return await asyncio.wait_for(task, timeout=timeout)
            
        except asyncio.TimeoutError:
            raise LLMResponseError(f"Response generation timed out after {timeout}s")
    
    async def _async_generate(self, prompt: str):
        """Async wrapper for Gemini generation."""
        
        # Run in thread pool since Gemini client is sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.model.generate_content, prompt)
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def _update_average_time(self, generation_time: float) -> None:
        """Update average response time."""
        
        if self.stats['requests_made'] == 1:
            self.stats['average_response_time'] = generation_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['average_response_time'] = (
                alpha * generation_time + 
                (1 - alpha) * self.stats['average_response_time']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM service statistics."""
        
        stats = self.stats.copy()
        stats.update({
            'model_name': self.model_name,
            'temperature': self.generation_config.temperature,
            'max_tokens': self.generation_config.max_output_tokens
        })
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM service."""
        
        health_status = {
            'healthy': True,
            'service': 'GeminiLLMService',
            'model': self.model_name,
            'errors': [],
            'stats': self.get_stats()
        }
        
        try:
            # Test generation with simple prompt
            test_prompt = "Respond with 'OK' to confirm the service is working."
            start_time = time.time()
            
            test_result = await self.generate_response(test_prompt)
            
            response_time = time.time() - start_time
            
            health_status.update({
                'test_successful': True,
                'response_time': response_time,
                'test_response': test_result['text'][:50]
            })
            
        except Exception as e:
            health_status.update({
                'healthy': False,
                'test_successful': False,
                'errors': [str(e)]
            })
        
        return health_status


class LLMResponseAgent(BaseAgent):
    """Advanced LLM Response Agent with intelligent answer generation."""
    
    def __init__(self):
        super().__init__(AgentType.LLM_RESPONSE)
        self.settings = get_settings()
        
        # Initialize components
        self.llm_service = GeminiLLMService()
        self.prompt_template = PromptTemplate()
        self.source_attributor = SourceAttributor()
        self.quality_analyzer = ResponseQualityAnalyzer()
        
        # Conversation management
        self.active_conversations: Dict[str, Conversation] = {}
        self.conversation_cleanup_interval = 3600  # 1 hour
        
        # Statistics
        self.stats = {
            'responses_generated': 0,
            'average_response_time': 0.0,
            'average_response_length': 0,
            'total_tokens_generated': 0,
            'confidence_distribution': {
                'high': 0,
                'medium': 0,
                'low': 0,
                'uncertain': 0
            },
            'response_types': {
                'direct_answer': 0,
                'summarized': 0,
                'comparative': 0,
                'instructional': 0,
                'no_answer': 0
            }
        }
    
    async def _initialize_agent(self) -> bool:
        """Initialize the LLM Response Agent."""
        
        try:
            self.logger.info("Initializing LLMResponseAgent")
            
            # Test LLM service
            health_check = await self.llm_service.health_check()
            if not health_check['healthy']:
                self.logger.error(f"LLM service health check failed: {health_check['errors']}")
                return False
            
            # Start conversation cleanup task
            asyncio.create_task(self._conversation_cleanup_worker())
            
            self.logger.info("LLMResponseAgent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLMResponseAgent: {e}")
            return False
    
    async def _handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Handle incoming MCP messages."""
        
        try:
            if message.message_type == MessageType.REQUEST:
                # Handle LLM response generation request
                if message.payload.get('action') == 'generate_response':
                    return await self._handle_generate_response_request(message)
                
                elif message.payload.get('action') == 'get_conversation':
                    return await self._handle_get_conversation_request(message)
                
                elif message.payload.get('action') == 'clear_conversation':
                    return await self._handle_clear_conversation_request(message)
                
                elif message.payload.get('action') == 'get_stats':
                    return await self._handle_stats_request(message)
                
                else:
                    return create_error_message(
                        message, 
                        f"Unknown action: {message.payload.get('action')}"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            return create_error_message(message, str(e))
    
    async def _handle_generate_response_request(self, message: MCPMessage) -> MCPMessage:
        """Handle response generation request."""
        
        start_time = time.time()
        
        try:
            # Extract request data
            request_data = message.payload
            
            # Validate required fields
            if 'query' not in request_data:
                return create_error_message(message, "Missing required field: query")
            
            if 'retrieval_results' not in request_data:
                return create_error_message(message, "Missing required field: retrieval_results")
            
            # Parse request
            query_text = request_data['query']
            retrieval_results_data = request_data['retrieval_results']
            conversation_id = request_data.get('conversation_id')
            user_id = request_data.get('user_id')
            
            # Reconstruct search results
            search_results = []
            for result_data in retrieval_results_data.get('results', []):
                search_result = SearchResult(**result_data)
                search_results.append(search_result)
            
            # Generate response
            response = await self._generate_response(
                query_text=query_text,
                search_results=search_results,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # Update conversation if conversation_id provided
            if conversation_id:
                await self._update_conversation(conversation_id, query_text, response)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(response, processing_time)
            
            # Prepare response payload
            response_payload = {
                'response_id': response.response_id,
                'answer': response.answer,
                'response_type': response.response_type.value,
                'confidence_level': response.confidence_level.value,
                'sources': [source.dict() for source in response.sources],
                'metrics': response.metrics.dict(),
                'follow_up_questions': response.follow_up_questions,
                'generated_at': response.generated_at.isoformat()
            }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}"
            self.logger.error(error_msg)
            return create_error_message(message, error_msg)
    
    async def _handle_get_conversation_request(self, message: MCPMessage) -> MCPMessage:
        """Handle conversation retrieval request."""
        
        try:
            conversation_id = message.payload.get('conversation_id')
            
            if not conversation_id:
                return create_error_message(message, "Missing conversation_id")
            
            if conversation_id in self.active_conversations:
                conversation = self.active_conversations[conversation_id]
                
                response_payload = {
                    'conversation_id': conversation_id,
                    'turns': [turn.dict() for turn in conversation.turns],
                    'total_turns': conversation.total_turns,
                    'started_at': conversation.started_at.isoformat(),
                    'last_activity_at': conversation.last_activity_at.isoformat()
                }
            else:
                response_payload = {
                    'conversation_id': conversation_id,
                    'error': 'Conversation not found'
                }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            error_msg = f"Get conversation failed: {str(e)}"
            self.logger.error(error_msg)
            return create_error_message(message, error_msg)
    
    async def _handle_clear_conversation_request(self, message: MCPMessage) -> MCPMessage:
        """Handle conversation clearing request."""
        
        try:
            conversation_id = message.payload.get('conversation_id')
            
            if not conversation_id:
                return create_error_message(message, "Missing conversation_id")
            
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]
                message_text = f"Conversation {conversation_id} cleared"
            else:
                message_text = f"Conversation {conversation_id} not found"
            
            response_payload = {
                'conversation_id': conversation_id,
                'message': message_text
            }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            error_msg = f"Clear conversation failed: {str(e)}"
            self.logger.error(error_msg)
            return create_error_message(message, error_msg)
    
    async def _handle_stats_request(self, message: MCPMessage) -> MCPMessage:
        """Handle statistics request."""
        
        try:
            # Get comprehensive statistics
            stats = await self._get_comprehensive_stats()
            
            return create_response_message(message, stats)
            
        except Exception as e:
            error_msg = f"Stats request failed: {str(e)}"
            self.logger.error(error_msg)
            return create_error_message(message, error_msg)
    
    async def _generate_response(
        self,
        query_text: str,
        search_results: List[SearchResult],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Response:
        """Generate a comprehensive response from search results."""
        
        generation_start = time.time()
        
        try:
            # Check if we have results
            if not search_results:
                return await self._generate_no_context_response(query_text)
            
            # Create source attributions
            sources = self.source_attributor.create_source_attributions(search_results)
            
            # Format context with citations
            context = self.source_attributor.format_context_with_citations(search_results)
            
            # Determine response type
            response_type = self._determine_response_type(query_text)
            
            # Get conversation history if available
            conversation_history = None
            if conversation_id and conversation_id in self.active_conversations:
                conversation = self.active_conversations[conversation_id]
                recent_turns = conversation.get_recent_turns(3)
                conversation_history = [
                    f"Q: {turn.query}\nA: {turn.response.answer[:200]}..."
                    for turn in recent_turns
                ]
            
            # Get and format prompt template
            template = self.prompt_template.get_template(response_type)
            prompt = self.prompt_template.format_prompt(
                template, 
                query_text, 
                context,
                conversation_history
            )
            
            # Generate response using LLM
            llm_result = await self.llm_service.generate_response(prompt)
            
            # Validate citations
            citation_issues = self.source_attributor.validate_citations(
                llm_result['text'], 
                len(sources)
            )
            
            if citation_issues:
                self.logger.warning(f"Citation issues: {citation_issues}")
            
            # Analyze response quality
            quality_scores = self.quality_analyzer.analyze_response_quality(
                llm_result['text'], 
                query_text, 
                sources
            )
            
            # Determine confidence level
            confidence_level = self.quality_analyzer.determine_confidence_level(
                quality_scores, 
                len(sources)
            )
            
            # Create response metrics
            retrieval_time = sum(
                result.chunk_metadata.get('retrieval_time', 0) 
                for result in search_results
            ) / len(search_results) if search_results else 0
            
            metrics = ResponseMetrics(
                retrieval_time=retrieval_time,
                generation_time=llm_result['generation_time'],
                total_response_time=time.time() - generation_start,
                source_documents_count=len(set(s.document_id for s in sources)),
                chunks_processed=len(search_results),
                tokens_generated=llm_result['token_count'],
                avg_source_similarity=sum(s.similarity_score for s in sources) / len(sources),
                confidence_score=sum(quality_scores.values()) / len(quality_scores),
                llm_api_calls=1,
                tokens_consumed=llm_result['token_count']
            )
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(
                query_text, 
                llm_result['text'], 
                sources
            )
            
            # Create response object
            response = Response(
                query_id="",  # Will be set by coordinator
                answer=llm_result['text'],
                response_type=response_type,
                confidence_level=confidence_level,
                sources=sources,
                metrics=metrics,
                follow_up_questions=follow_up_questions,
                model_used=self.llm_service.model_name,
                temperature=self.llm_service.generation_config.temperature
            )
            
            # Update confidence based on metrics
            response.update_confidence()
            
            self.logger.info(
                f"Generated {response_type.value} response "
                f"({len(response.answer)} chars, {len(sources)} sources) "
                f"in {metrics.total_response_time:.2f}s"
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise LLMResponseError(error_msg)
    
    async def _generate_no_context_response(self, query_text: str) -> Response:
        """Generate response when no relevant context is available."""
        
        template = self.prompt_template.get_template(ResponseType.NO_ANSWER)
        answer = template.format(query=query_text)
        
        metrics = ResponseMetrics(
            retrieval_time=0.0,
            generation_time=0.0,
            total_response_time=0.0,
            source_documents_count=0,
            chunks_processed=0,
            tokens_generated=len(answer.split()),
            avg_source_similarity=0.0,
            confidence_score=0.0
        )
        
        response = Response(
            answer=answer,
            response_type=ResponseType.NO_ANSWER,
            confidence_level=ConfidenceLevel.UNCERTAIN,
            sources=[],
            metrics=metrics,
            model_used=self.llm_service.model_name
        )
        
        return response
    
    def _determine_response_type(self, query_text: str) -> ResponseType:
        """Determine the appropriate response type based on query."""
        
        query_lower = query_text.lower()
        
        # Check for comparison queries
        comparison_indicators = ['compare', 'difference', 'versus', 'vs', 'contrast']
        if any(indicator in query_lower for indicator in comparison_indicators):
            return ResponseType.COMPARATIVE
        
        # Check for instructional queries
        instruction_indicators = ['how to', 'step', 'process', 'procedure', 'instructions']
        if any(indicator in query_lower for indicator in instruction_indicators):
            return ResponseType.INSTRUCTIONAL
        
        # Check for summary queries
        summary_indicators = ['summary', 'summarize', 'overview', 'brief']
        if any(indicator in query_lower for indicator in summary_indicators):
            return ResponseType.SUMMARIZED
        
        # Default to direct answer
        return ResponseType.DIRECT_ANSWER
    
    def _generate_follow_up_questions(
        self, 
        query_text: str, 
        answer: str, 
        sources: List[SourceAttribution]
    ) -> List[str]:
        """Generate relevant follow-up questions."""
        
        follow_ups = []
        
        # Extract key topics from sources
        topics = set()
        for source in sources:
            # Simple topic extraction (could be enhanced with NLP)
            words = source.relevant_content.lower().split()
            potential_topics = [word for word in words if len(word) > 5]
            topics.update(potential_topics[:3])  # Top 3 from each source
        
        # Generate question templates
        question_templates = [
            "Can you tell me more about {topic}?",
            "How does {topic} relate to the main topic?",
            "What are the implications of {topic}?",
            "Are there any examples of {topic}?"
        ]
        
        # Generate questions from topics
        for topic in list(topics)[:3]:  # Max 3 follow-up questions
            if topic not in query_text.lower():  # Don't repeat query terms
                template = question_templates[len(follow_ups) % len(question_templates)]
                follow_ups.append(template.format(topic=topic))
        
        return follow_ups
    
    async def _update_conversation(
        self, 
        conversation_id: str, 
        query: str, 
        response: Response
    ) -> None:
        """Update conversation with new turn."""
        
        try:
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = Conversation(
                    conversation_id=conversation_id
                )
            
            conversation = self.active_conversations[conversation_id]
            conversation.add_turn(query, response)
            
        except Exception as e:
            self.logger.error(f"Failed to update conversation {conversation_id}: {e}")
    
    async def _conversation_cleanup_worker(self):
        """Background worker to clean up old conversations."""
        
        while True:
            try:
                await asyncio.sleep(self.conversation_cleanup_interval)
                
                current_time = datetime.utcnow()
                conversations_to_remove = []
                
                for conv_id, conversation in self.active_conversations.items():
                    # Remove conversations older than 24 hours
                    age = (current_time - conversation.last_activity_at).total_seconds()
                    if age > 24 * 3600:  # 24 hours
                        conversations_to_remove.append(conv_id)
                
                for conv_id in conversations_to_remove:
                    del self.active_conversations[conv_id]
                
                if conversations_to_remove:
                    self.logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")
                
            except Exception as e:
                self.logger.error(f"Error in conversation cleanup: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    def _update_stats(self, response: Response, processing_time: float) -> None:
        """Update agent statistics."""
        
        self.stats['responses_generated'] += 1
        self.stats['total_tokens_generated'] += response.metrics.tokens_generated
        
        # Update average response time
        if self.stats['responses_generated'] == 1:
            self.stats['average_response_time'] = processing_time
        else:
            alpha = 0.1
            self.stats['average_response_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['average_response_time']
            )
        
        # Update average response length
        response_length = len(response.answer)
        if self.stats['responses_generated'] == 1:
            self.stats['average_response_length'] = response_length
        else:
            alpha = 0.1
            self.stats['average_response_length'] = (
                alpha * response_length + 
                (1 - alpha) * self.stats['average_response_length']
            )
        
        # Update confidence distribution
        confidence_key = response.confidence_level.value.lower()
        if confidence_key in self.stats['confidence_distribution']:
            self.stats['confidence_distribution'][confidence_key] += 1
        
        # Update response type distribution
        response_type_key = response.response_type.value.lower()
        if response_type_key in self.stats['response_types']:
            self.stats['response_types'][response_type_key] += 1
    
    async def _get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        
        stats = self.stats.copy()
        
        # Add LLM service stats
        stats['llm_service'] = self.llm_service.get_stats()
        
        # Add conversation stats
        stats['conversations'] = {
            'active_conversations': len(self.active_conversations),
            'total_turns': sum(
                conv.total_turns 
                for conv in self.active_conversations.values()
            )
        }
        
        return stats
    
    async def _agent_health_check(self) -> Optional[Dict[str, Any]]:
        """Perform agent-specific health check."""
        
        health_info = {
            'llm_service_healthy': False,
            'active_conversations': len(self.active_conversations),
            'statistics': self.stats.copy()
        }
        
        # Check LLM service health
        try:
            llm_health = await self.llm_service.health_check()
            health_info['llm_service_healthy'] = llm_health['healthy']
            health_info['llm_service'] = llm_health
        except Exception as e:
            health_info['llm_service_error'] = str(e)
        
        return health_info
