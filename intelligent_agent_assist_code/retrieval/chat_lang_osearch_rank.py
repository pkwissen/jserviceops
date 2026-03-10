"""
INTELLIGENT AGENT - RAG CHAT WITH LANGCHAIN + OPENSEARCH + COHERE RERANKER
==============================================================================
Production-ready RAG pipeline with:
- LangChain LLM (wise-azure-gpt-5) with chat history support
- OpenSearch vector retrieval using LlamaIndex
- Cohere reranking for improved relevance
- Session-based conversation memory
- Comprehensive logging and error handling

Author: AI Assistant
Date: March 2026
"""

import os
import sys
import uuid
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# OpenAI Client
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END, add_messages

# LlamaIndex Components
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.embeddings.openai import OpenAIEmbedding

# Local Imports
from intelligent_agent_assist_code.config.settings import INDEX_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, OPENSEARCH_ENDPOINT
from intelligent_agent_assist_code.search.opensearch_client import get_opensearch_client
from intelligent_agent_assist_code.retrieval.reranker import AzureCohereReranker

# ==========================================
# LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# CONVERSATION SESSION MANAGER
# ==========================================

class ConversationSession:
    """
    Manages conversation history for a chat session.
    Handles message storage, history formatting, and session lifecycle.
    """
    
    def __init__(self, session_id: Optional[str] = None, max_history: int = 10):
        """
        Initialize a conversation session.
        
        Args:
            session_id: Unique identifier for the session. Auto-generated if None.
            max_history: Maximum number of message pairs to retain (default: 10)
        """
        self.session_id = session_id or self._generate_session_id()
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        logger.info(f"📝 New conversation session: {self.session_id}")
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session identifier."""
        return f"session_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()
    
    def add_assistant_message(self, content: str):
        """Add an assistant response to history."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()
    
    def _trim_history(self):
        """Keep only the most recent messages within max_history limit."""
        if len(self.messages) > self.max_history * 2:
            # Keep the most recent max_history pairs (user + assistant)
            self.messages = self.messages[-(self.max_history * 2):]
    
    def get_langchain_history(self) -> List:
        """
        Convert session history to LangChain message format.
        
        Returns:
            List of HumanMessage and AIMessage objects
        """
        lc_messages = []
        for msg in self.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        return lc_messages
    
    def get_formatted_history(self) -> str:
        """
        Get conversation history as formatted string.
        
        Returns:
            Multi-line string with conversation flow
        """
        if not self.messages:
            return "No conversation history yet."
        
        formatted = []
        for msg in self.messages:
            role_prefix = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            formatted.append(f"{role_prefix}: {msg['content']}")
        return "\n".join(formatted)
    
    def clear_history(self):
        """Clear all conversation history while keeping session active."""
        self.messages = []
        logger.info(f"🗑️ Cleared history for session: {self.session_id}")


# ==========================================
# GRAPH STATE
# ==========================================

class AgentState(TypedDict):
    """State for the LangGraph RAG agent."""
    messages: Annotated[List[AnyMessage], add_messages]
    search_results: List[str]
    source_documents: List[dict]
    rewritten_query: str


# ==========================================
# RAG CHAT ENGINE
# ==========================================

class RAGChatEngine:
    """
    Production RAG pipeline with retrieval, reranking, and conversational chat.
    """
    
    def __init__(
        self,
        retrieval_top_k: int = 20,
        rerank_top_n: int = 3,
        llm_temperature: float = 1,
        llm_max_tokens: int = None
    ):
        """
        Initialize the RAG chat engine.
        
        Args:
            retrieval_top_k: Number of documents to retrieve from OpenSearch
            rerank_top_n: Number of documents to keep after reranking
            llm_temperature: LLM creativity (0=deterministic, 1=creative)
            llm_max_tokens: Maximum tokens in LLM response
        """
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n
        
        # Initialize components
        self._init_embedding_model()
        self._init_vector_store()
        self._init_reranker()
        self._init_llm(llm_temperature, llm_max_tokens)
        
        # Build LangGraph
        self.graph = self._build_graph()
        
        logger.info("✅ RAG Chat Engine initialized successfully")
    
    def _init_embedding_model(self):
        """Initialize OpenAI embedding model via Wissen Gateway."""
        try:
            self.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_BASE_URL.rstrip("/"),
                embed_batch_size=1,
                additional_kwargs={
                    "extra_body": {
                        "model": "wise-azure-text-embedding-3-small"
                    }
                }
            )
            logger.info("✅ Embedding model initialized")
        except Exception as e:
            logger.error(f"❌ Embedding model initialization failed: {e}")
            raise
    
    def _init_vector_store(self):
        """Initialize OpenSearch vector store with LlamaIndex."""
        try:
            os_client = get_opensearch_client()
            
            vector_client = OpensearchVectorClient(
                endpoint=OPENSEARCH_ENDPOINT,
                index=INDEX_NAME,
                dim=1536,
                os_client=os_client
            )
            
            vector_store = OpensearchVectorStore(vector_client)
            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )
            
            self.retriever = self.index.as_retriever(similarity_top_k=self.retrieval_top_k)
            logger.info(f"✅ Vector store initialized (Index: {INDEX_NAME})")
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            raise
    
    def _init_reranker(self):
        """Initialize Cohere reranker."""
        try:
            self.reranker = AzureCohereReranker()
            logger.info("✅ Cohere reranker initialized")
        except Exception as e:
            logger.error(f"❌ Reranker initialization failed: {e}")
            raise
    
    def _init_llm(self, temperature: float, max_tokens: int):
        """Initialize LangChain LLM."""
        try:
            # Initialize with explicit response format handling
            self.llm = ChatOpenAI(
                model="wise-azure-gpt-5",
                openai_api_base=OPENAI_BASE_URL,
                openai_api_key=OPENAI_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=False,
                model_kwargs={
                    "extra_body": {
                        "model": "wise-azure-gpt-5"
                    }
                }
            )
            logger.info("✅ LLM initialized (wise-azure-gpt-5)")
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            raise
    
    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant documents from OpenSearch.
        
        Args:
            query: User's search query
            
        Returns:
            List of retrieved document texts
        """
        try:
            logger.info(f"🔍 Retrieving documents for query (top_k={self.retrieval_top_k})")
            nodes = self.retriever.retrieve(query)
            
            # Return both content and the full node (which contains metadata)
            documents = [{"content": node.get_content(), "metadata": node.metadata} for node in nodes]
            logger.info(f"✅ Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []
    
    def rerank(self, query: str, documents: List[Dict]) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Rerank documents using Cohere for improved relevance.
        
        Args:
            query: User's query
            documents: List of retrieved document dictionaries with content and metadata
            
        Returns:
            Tuple of (reranked_documents_text, relevance_scores, reranked_source_documents)
        """
        if not documents:
            logger.warning("⚠️ No documents to rerank")
            return [], [], []
        
        try:
            logger.info(f"🎯 Reranking {len(documents)} documents (top_n={self.rerank_top_n})")
            
            # Truncate documents to avoid token limits. Pass only the content string to the reranker.
            truncated_docs = [doc["content"][:2000] for doc in documents]
            
            reranked_results = self.reranker.rerank(
                query=query,
                documents=truncated_docs,
                top_n=self.rerank_top_n
            )
            
            reranked_docs_text = [result["content"] for result in reranked_results]
            scores = [result["relevance_score"] for result in reranked_results]
            
            # Reconstruct the source documents matching the reranked output
            source_docs = []
            for result in reranked_results:
                original_index = result["index"]
                source_docs.append(documents[original_index])
            
            logger.info(f"✅ Reranked to top {len(reranked_docs_text)} documents")
            logger.info(f"📊 Relevance scores: {[f'{s:.3f}' for s in scores]}")
            
            return reranked_docs_text, scores, source_docs
        except Exception as e:
            logger.error(f"❌ Reranking failed: {e}, using top {self.rerank_top_n} docs")
            fallback_docs = documents[:self.rerank_top_n]
            return [d["content"] for d in fallback_docs], [], fallback_docs
    
    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: List
    ) -> str:
        """
        Generate LLM response with context and chat history.
        
        Args:
            query: User's question
            context: Retrieved and reranked context
            chat_history: List of previous messages (LangChain format)
            
        Returns:
            LLM generated response
        """
        try:
            logger.info(f"💬 Generating LLM response with {len(chat_history)} history messages")
            
            # Build the complete message list
            messages = []
            
            # Add system message
            system_content = """You are an intelligent IT support assistant"""
            
            messages.append(SystemMessage(content=system_content))
            
            # Add chat history (previous conversation) - validate each message
            for msg in chat_history:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    messages.append(msg)
                else:
                    logger.warning(f"⚠️ Skipping invalid message type: {type(msg)}")
            
            # Truncate context if too long to avoid token limits
            max_context_chars = 8000  # Roughly 2000 tokens
            if len(context) > max_context_chars:
                logger.warning(f"⚠️ Context too long ({len(context)} chars), truncating to {max_context_chars}")
                context = context[:max_context_chars] + "\n\n[Context truncated due to length...]"
            
            # Add current query with context
            current_message = f"""Context from Knowledge Base:
{context}

Question: {query}

Provide a clear, accurate answer based on the context above."""
            
            messages.append(HumanMessage(content=current_message))
            
            # Log message structure for debugging
            logger.info(f"📤 Message structure: System + {len(chat_history)} history + 1 current query")
            
            # Log message types and brief content for debugging
            if len(chat_history) > 0:
                logger.info("📋 Chat history messages:")
                for i, msg in enumerate(chat_history):
                    msg_type = type(msg).__name__
                    content_preview = msg.content[:80] if hasattr(msg, 'content') else str(msg)[:80]
                    logger.info(f"  [{i}] {msg_type}: {content_preview}...")
            
            # Validate all messages have content
            for i, msg in enumerate(messages):
                if not hasattr(msg, 'content') or not msg.content:
                    logger.error(f"❌ Invalid message at position {i}: {type(msg)}")
                    raise ValueError(f"Message at position {i} has no content")
            
            # Invoke LLM with complete message list
            try:
                # Use the underlying OpenAI client directly to avoid LangChain parsing issues
                # Convert LangChain messages to OpenAI format
                openai_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        openai_messages.append({"role": "system", "content": msg.content})
                    elif isinstance(msg, HumanMessage):
                        openai_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        openai_messages.append({"role": "assistant", "content": msg.content})
                
                logger.info(f"🔌 Making direct API call with {len(openai_messages)} messages")
                
                # Make direct API call using OpenAI client
                client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL
                )
                
                response = client.chat.completions.create(
                    model="wise-azure-gpt-5",
                    messages=openai_messages,
                    temperature=self.llm.temperature,
                    max_tokens=self.llm.max_tokens
                )
                
                # Extract answer from response - handle multiple response formats
                logger.info(f"📦 Response type: {type(response)}")
                
                if isinstance(response, str):
                    # Direct string response - could be JSON or plain text
                    logger.info("📄 Received string response, attempting to parse...")
                    
                    # Try parsing as JSON first
                    try:
                        response_dict = json.loads(response)
                        logger.info("✅ Parsed string as JSON")
                        
                        # Extract from JSON structure
                        if 'choices' in response_dict and len(response_dict['choices']) > 0:
                            answer = response_dict['choices'][0]['message']['content'].strip()
                        elif 'content' in response_dict:
                            answer = response_dict['content'].strip()
                        else:
                            # Use the whole string if can't find specific fields
                            answer = response.strip()
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Not JSON or malformed - treat as plain text answer
                        logger.info("✅ Using string response as-is (not JSON)")
                        answer = response.strip()
                        
                elif hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                    # Standard OpenAI response format
                    logger.info("✅ Received standard OpenAI response")
                    answer = response.choices[0].message.content.strip()
                elif hasattr(response, 'content'):
                    # Alternative format with content attribute
                    logger.info("✅ Received response with content attribute")
                    answer = response.content.strip()
                elif isinstance(response, dict):
                    # Dictionary response - try to extract content
                    logger.info("✅ Received dictionary response")
                    if 'choices' in response and len(response['choices']) > 0:
                        answer = response['choices'][0]['message']['content'].strip()
                    elif 'content' in response:
                        answer = response['content'].strip()
                    elif 'text' in response:
                        answer = response['text'].strip()
                    else:
                        logger.error(f"❌ Cannot extract content from dict response: {response.keys()}")
                        raise ValueError(f"Unknown dict response format: {response.keys()}")
                else:
                    logger.error(f"❌ Unexpected response type: {type(response)}")
                    logger.error(f"Response attributes: {dir(response)}")
                    raise ValueError(f"Unexpected response type: {type(response)}")
                
                if not answer:
                    logger.error("❌ Empty answer extracted from response")
                    raise ValueError("Empty answer from API")
                
            except AttributeError as attr_err:
                # Handle attribute errors (e.g., 'str' object has no attribute 'choices' or 'model_dump')
                logger.error(f"❌ AttributeError in response processing: {attr_err}")
                logger.info(f"🔍 Response variable type: {type(response) if 'response' in locals() else 'not set'}")
                
                # Try to extract content from the response variable if it exists
                if 'response' in locals():
                    logger.info(f"🔍 Response value: {str(response)[:500]}...")
                    
                    # If response is simply a string, use it
                    if isinstance(response, str):
                        logger.info("✅ Extracted answer from string response")
                        answer = response.strip()
                        if answer:
                            logger.info(f"✅ Successfully recovered ({len(answer)} chars)")
                            logger.info(f"✅ Response generated ({len(answer)} chars)")
                            return answer
                
                # If we couldn't extract anything, return fallback message
                logger.error("❌ Could not extract answer from response")
                return "I apologize, but I'm having trouble processing the response. This may be a temporary API issue. Please try rephrasing your question or try again in a moment."
                
            except Exception as llm_error:
                logger.error(f"❌ LLM invocation error: {llm_error}")
                import traceback
                logger.error(f"Error traceback: {traceback.format_exc()}")
                raise
            
            logger.info(f"✅ Response generated ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    # ==========================================
    # LANGGRAPH NODES
    # ==========================================

    def _query_rewriting_node(self, state: AgentState) -> AgentState:
        """
        Rewrite the user query using conversation history for better retrieval context.
        """
        messages = state["messages"]
        # Find the latest user message
        user_message = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_message = m.content
                break
                
        if not user_message:
            return state

        # If there's no real history beyond the current message, skip rewriting
        if len(messages) <= 1:
            return {"rewritten_query": user_message}
            
        logger.info(f"🔄 Rewriting query with history: {user_message[:50]}...")
        
        # Take the last few messages for context
        context_messages = messages[-5:-1] if len(messages) > 4 else messages[:-1]
        history_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in context_messages])
        
        rewrite_prompt = f"""Given the following conversation history and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        
Conversation History:
{history_text}

Follow Up Input: {user_message}
Standalone question:"""

        try:
            response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])
            rewritten_query = response.content.strip()
            logger.info(f"✅ Rewritten query: {rewritten_query}")
            return {"rewritten_query": rewritten_query}
        except Exception as e:
            logger.error(f"❌ Query rewriting failed: {e}")
            return {"rewritten_query": user_message}

    def _search_node(self, state: AgentState) -> AgentState:
        """
        Retrieve and rerank documents based on the rewritten query.
        """
        query = state.get("rewritten_query")
        if not query:
            # Fallback to the latest message if no rewritten query exists
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    query = m.content
                    break
        
        if not query:
            return {"search_results": [], "source_documents": []}

        # Step 1: Retrieve
        documents = self.retrieve(query)
        
        if not documents:
            return {"search_results": [], "source_documents": []}
            
        # Step 2: Rerank
        reranked_docs_text, scores, source_docs = self.rerank(query, documents)
        
        return {"search_results": reranked_docs_text, "source_documents": source_docs}

    def _answer_node(self, state: AgentState) -> AgentState:
        """
        Generate the final answer using ONLY the context and the rewritten query.
        (Conversation history is NOT sent to the LLM to save tokens and prevent confusion).
        """
        query = state.get("rewritten_query")
        if not query:
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    query = m.content
                    break
                    
        search_results = state.get("search_results", [])
        context_text = "\n\n---\n\n".join(search_results) if search_results else "No relevant information found."
        
        # Add system message and ONLY the final standalone query (no history)
        system_content = "You are an intelligent IT support assistant."
        messages_for_llm = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"""Context from Knowledge Base:
{context_text}

Question: {query}

Provide a clear, accurate answer based on the context above.""")
        ]
        
        try:
            # Invoke the LLM directly through LangChain wrapper
            # We skip the complex direct string parsing for simplicity here, but can adapt if needed
            response = self.llm.invoke(messages_for_llm)
            answer = response.content.strip()
            
            logger.info(f"✅ Response generated through LangGraph ({len(answer)} chars)")
            return {"messages": [AIMessage(content=answer)]}
        except Exception as e:
            logger.error(f"❌ LLM generation in Answer Node failed: {e}")
            return {"messages": [AIMessage(content="I apologize, but I encountered an error generating a response. Please try again.")]}

    def _build_graph(self):
        """Build and compile the LangGraph for RAG workflow."""
        builder = StateGraph(AgentState)
        
        builder.add_node("rewrite", self._query_rewriting_node)
        builder.add_node("search", self._search_node)
        builder.add_node("answer", self._answer_node)
        
        builder.add_edge(START, "rewrite")
        builder.add_edge("rewrite", "search")
        builder.add_edge("search", "answer")
        builder.add_edge("answer", END)
        
        return builder.compile()
    
    def chat(
        self,
        query: str,
        session: ConversationSession
    ) -> Dict[str, any]:
        """
        Complete RAG pipeline executing via LangGraph: retrieve → rerank → generate.
        
        Args:
            query: User's question
            session: Active conversation session
            
        Returns:
            Dictionary with answer, context, metadata
        """
        logger.info(f"🚀 Starting LangGraph RAG pipeline for session: {session.session_id}")
        logger.info(f"❓ Query: {query[:100]}...")
        
        # Get history from the session as LangChain messages
        chat_history = session.get_langchain_history()
        
        # Add the current query to the history for the graph
        current_message = HumanMessage(content=query)
        chat_history.append(current_message)
        
        # Prepare the initial state
        initial_state = {
            "messages": chat_history,
            "search_results": [],
            "source_documents": [],
            "rewritten_query": ""
        }
        
        # Execute the graph
        try:
            result_state = self.graph.invoke(initial_state)
            
            # Extract outputs from the final state
            final_messages = result_state["messages"]
            answer_msg = final_messages[-1]
            answer = answer_msg.content if isinstance(answer_msg, AIMessage) else str(answer_msg)
            
            search_results = result_state.get("search_results", [])
            source_documents = result_state.get("source_documents", [])
            context = "\n\n---\n\n".join(search_results)
            
            # Update session
            session.add_user_message(query)
            session.add_assistant_message(answer)
            
            logger.info("✅ LangGraph RAG pipeline completed")
            
            return {
                "answer": answer,
                "context": context,
                "documents_retrieved": -1, # We don't expose initial retrieved count broadly in graph state currently
                "documents_reranked": len(search_results),
                "source_documents": source_documents, # Pass the detailed source documents out
                "relevance_scores": [], # We don't track scores in the high level state currently
                "session_id": session.session_id,
                "rewritten_query": result_state.get("rewritten_query", query)
            }
            
        except Exception as e:
            logger.error(f"❌ LangGraph execution failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return {
                "answer": "I apologize, but I encountered an error during conversation processing. Please try again.",
                "context": "",
                "documents_retrieved": 0,
                "documents_reranked": 0,
                "relevance_scores": [],
                "session_id": session.session_id
            }


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================

# Global engine instance (singleton pattern)
_engine_instance: Optional[RAGChatEngine] = None
_active_sessions: Dict[str, ConversationSession] = {}


def get_rag_engine() -> RAGChatEngine:
    """
    Get or create the global RAG engine instance.
    
    Returns:
        Initialized RAG chat engine
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGChatEngine()
    return _engine_instance


def get_llm() -> ChatOpenAI:
    """
    Get the LLM instance from the RAG engine.
    
    Returns:
        ChatOpenAI LLM instance
    """
    engine = get_rag_engine()
    return engine.llm


def create_session(session_id: Optional[str] = None) -> ConversationSession:
    """
    Create a new conversation session.
    
    Args:
        session_id: Optional custom session ID
        
    Returns:
        New conversation session
    """
    session = ConversationSession(session_id=session_id)
    _active_sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> Optional[ConversationSession]:
    """
    Retrieve an existing session by ID.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Conversation session or None if not found
    """
    return _active_sessions.get(session_id)


def chat(
    query: str,
    session: Optional[ConversationSession] = None
) -> Dict[str, any]:
    """
    Simple chat interface function.
    
    Args:
        query: User's question
        session: Optional conversation session (creates new if None)
        
    Returns:
        Response dictionary with answer and metadata
    """
    engine = get_rag_engine()
    
    if session is None:
        session = create_session()
    
    return engine.chat(query, session)


# ==========================================
# COMPATIBILITY FUNCTIONS (for backward compatibility with rag_chat.py)
# ==========================================

def generate_thread_id() -> str:
    """
    Generate a unique thread/session ID for backward compatibility.
    Alias for create_session().session_id
    
    Returns:
        Unique session identifier
    """
    session = create_session()
    return session.session_id


def chat_with_kb(question: str, session_id: Optional[str] = None) -> str:
    """
    Simplified chat interface for backward compatibility with rag_chat.py.
    
    Args:
        question: User's question
        session_id: Optional session identifier
        
    Returns:
        Answer string only (no metadata)
    """
    # Get or create session
    session = get_session(session_id) if session_id else None
    if session is None:
        session = create_session(session_id=session_id)
    
    # Get full response
    result = chat(question, session)
    
    # Return only the answer for backward compatibility
    return result["answer"]

