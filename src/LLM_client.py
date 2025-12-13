import os
from typing import Optional, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Load environment variables
load_dotenv()

class LLMClient:
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        streaming: bool = False
    ):
        # Load config từ .env
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL")
        
        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY không tìm thấy trong .env")
        
        # Khởi tạo LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
        
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def get_llm(self) -> ChatOpenAI:
        return self.llm 
    def chat(
        self, 
        message: str, 
        chat_history: List[BaseMessage] = None,
    ) -> str:
        messages = []
        
        if chat_history:
            messages.extend(chat_history)       
        messages.append(HumanMessage(content=message))
        response = self.llm.invoke(messages)
        return response.content


#test 
if __name__ == "__main__":
    llm_client = LLMClient()

    #tạo chat history giả lập
    chat_history = [
        SystemMessage(content="Bạn cần xác định câu hỏi của người dùng có phải là câu hỏi hướng dẫn sử dụng hay không? Nếu có trả về True, không trả về False."),
        HumanMessage(content="Tôi không đổi được mật khẩu ?"),
        AIMessage(content="True"),
        HumanMessage(content="Chỉ tôi từng bước rõ hơn ?"),
        AIMessage(content="True")
    ]
    message = "Tôi không đọc được bạn chỉ dẫn"
    response = llm_client.chat(message, chat_history)
    print(response) 