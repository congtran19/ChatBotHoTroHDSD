from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,BaseMessage
from LLM_client import LLMClient
from step6_vector_database import SearchResult


class RAGGenerator:
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.llm = self.llm_client.get_llm()
        
        # System prompt CHỈ chứa hướng dẫn chung
        self.system_prompt = """Bạn là trợ lý AI thông minh hỗ trợ hướng dẫn sử dụng phần mềm.

NGUYÊN TẮC:
1. Trả lời dựa trên CONTEXT được cung cấp trong câu hỏi
2. Nếu không có thông tin trong context, nói "Tôi không tìm thấy thông tin này trong tài liệu"
3. Trích dẫn nguồn (source, trang) khi có thể
4. Trả lời ngắn gọn, dễ hiểu
"""
    
    def format_context(self, search_results: List[SearchResult]) -> str:
        if not search_results:
            return "Không có tài liệu nào được tìm thấy."
        
        parts = []
        for i, result in enumerate(search_results, 1):
            meta = result.metadata
            source = meta.get('source', 'Unknown')
            page = meta.get('page', 'N/A')
            section = meta.get('h2', meta.get('h1', 'N/A'))
            
            block = f"""[{i}] Nguồn: {source} | Trang: {page} | Section: {section}
{result.content}"""
            parts.append(block)
        
        return "\n---\n".join(parts)
    
    def generate(
        self,
        question: str,
        search_results: List[SearchResult],
        chat_history: List[BaseMessage] = None
    ) -> str:
        context = self.format_context(search_results)
        
        # Build messages
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add chat history (các lượt chat trước)
        if chat_history:
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            messages.extend(recent_history)
        
        # Gộp context + question vào 1 HumanMessage
        # Mỗi câu hỏi có context riêng của nó
        user_message = f"""Dựa trên tài liệu sau để trả lời câu hỏi:

=== TÀI LIỆU THAM KHẢO ===
{context}
=== HẾT TÀI LIỆU ===

Câu hỏi: {question}"""
        
        messages.append(HumanMessage(content=user_message))
        
        # Debug
        print(f"\n--- DEBUG RAG ---")
        print(f"Context length: {len(context)} chars")
        print(f"Question: {question}")
        
        response = self.llm.invoke(messages)
        
        return response.content
    
