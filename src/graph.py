from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from RAG import RAGGenerator
from step6_vector_database import ElasticsearchVectorStore
from step7_reranking import Reranker
from step5_embedding import VietnameseEmbedder
from LLM_client import LLMClient
from step6_vector_database import SearchResult
import numpy as np
import time
# ==========================================
# 1. ĐỊNH NGHĨA STATE
# ==========================================
class GraphState(TypedDict):
    """Trạng thái của đồ thị, lưu trữ dữ liệu qua các bước."""
    # Lịch sử hội thoại (Tự động append message mới)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Input từ người dùng
    question: str
    
    # Phân loại ý định của người dùng
    intent: str    #Kết quả phân loại : "greeting", "unsupported", "retrieve"
    
    # Dữ liệu RAG
    retrieved_docs: List[SearchResult]  # Danh sách tài liệu tìm được
    
    # Output cuối cùng
    response: str

class IntentClassifier:
        """Phân loại intent bằng cách so sánh trực tiếp với label embeddings"""
        
        def __init__(self, embedder):
            
            self.embedder = embedder
            
            # Định nghĩa labels (có thể điều chỉnh text cho phù hợp)
            self.intent_examples = {
                "retrieve": [
                    "Làm thế nào để sử dụng chức năng trong hệ thống?",
                    "Cách thực hiện một thao tác trong phần mềm?",
                    "Hướng dẫn sử dụng tính năng",
                    "Tôi cần làm gì để thực hiện thao tác này?",
                    "Cần kết nối Internet mới dùng được hay có thể dùng offline?",
                    "Phần mềm có hỗ trợ quản lý mã vạch không?",
                    "Có thể nhập/xuất kho trực tiếp trên phần mềm không?",
                    "Phần mềm có cho phép thanh toán nhiều hình thức (tiền mặt + thẻ) không?"
                    "Làm sao để quản lý công nợ của từng khách hàng?",
                    "Tôi có thể xuất báo cáo ra Excel không?"
                ],
                "greeting": [
                    "Xin chào",
                    "Chào bạn",
                    "Hello"
                ],
                "unsupported": [
                    "So sánh phần mềm này với phần mềm khác",
                    "Đánh giá hệ thống"
                ]
            }

        def cosine_similarity(self, query_emb, label_emb):
            return np.dot(query_emb, label_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(label_emb))
        
        def classify(self, question: str) -> List[str]:
            start = time.time()
            label_embeddings = []
            q_emb = self.embedder.embed_query(question)
            q_emb = q_emb.tolist() if hasattr(q_emb, 'tolist') else q_emb

            for intent, texts in self.intent_examples.items():
                for text in texts:
                    vector = self.embedder.embed_query(text)
                    label_embeddings.append({
                        "intent": intent,
                        "text": text,
                        "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
                    })
            
            scores = [(label_emb["intent"], self.cosine_similarity(q_emb, label_emb["vector"])) for label_emb in label_embeddings]
            #trả về intent cái có điểm cao nhất
            end = time.time()
            print(f"Time: {end - start}")
            # return intent có điểm cao nhất
            return max(scores, key=lambda x: x[1])[0]



class RAGGRaph:
    def __init__(self):
        self.embedder = VietnameseEmbedder()
        self.graph = StateGraph(GraphState)
        self.llm = LLMClient()
        self.rag_generator = RAGGenerator(self.llm)
        self.intent_classifier = IntentClassifier(self.embedder)
        self.vector_store = ElasticsearchVectorStore(index_name='rag_test_vectors')
        self.reranker = Reranker()
        
        self.app = None
    
    def intent_question(self, state: GraphState) -> GraphState:
        intent = self.intent_classifier.classify(state["question"])
        return {"intent": intent}
    
    def handle_greeting(self, state: GraphState) -> GraphState:
        res = "Xin chào tôi có thể giúp gì cho bạn ? "
        return {"response": res,"messages":[AIMessage(content=res)]}
    
    def handle_unsupported(self, state: GraphState) -> GraphState:
        res = "Xin lỗi, tôi không hiểu ý định của bạn"
        return {"response": res,"messages":[AIMessage(content=res)]}

    def handle_retrieve(self, state: GraphState) -> GraphState:
        query_emb = self.embedder.embed_query(state["question"])
        query_emb = query_emb.tolist() if hasattr(query_emb, 'tolist') else query_emb
        results = self.vector_store.search_hybrid(state["question"], query_emb, top_k=5)
        reranked = self.reranker.rerank(state["question"], results, top_k=3)
        print(reranked)
        return {"retrieved_docs": reranked}

    def generate_response(self, state: GraphState) -> dict:
        question = state["question"]
        retrieved_docs = state["retrieved_docs"]  # List[SearchResult] từ reranker
        messages = state.get("messages", [])
        response = self.rag_generator.generate(question = question, search_results = retrieved_docs, chat_history = messages)
        return {
            "response": response,
            "messages": [AIMessage(response)]
        }
    

    #===========================ROUTER==============================
    def router(self, state: GraphState) -> str:
        """
        Router trả về KEY khớp với dict trong add_conditional_edges.
        Dict: {"greeting": "handle_greeting", "retrieve": "handle_retrieve", ...}
        Nên router phải trả về: "greeting", "retrieve", hoặc "unsupported"
        """
        intent = state["intent"]
        if intent in ["greeting", "retrieve", "unsupported"]:
            return intent
        return "unsupported"  # Fallback
        
    def build(self):
        """Xây dựng graph"""
        workflow = StateGraph(GraphState)
        # 1. Thêm các nodes
        workflow.add_node("classify_intent", self.intent_question)
        workflow.add_node("handle_greeting", self.handle_greeting)
        workflow.add_node("handle_retrieve", self.handle_retrieve)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("handle_unsupported", self.handle_unsupported)
        # 2. Set entry point
        workflow.set_entry_point("classify_intent")
        # 3. Conditional edges (rẽ nhánh)
        workflow.add_conditional_edges(
            "classify_intent",
            self.router,
            {
                "greeting": "handle_greeting",
                "retrieve": "handle_retrieve",
                "unsupported": "handle_unsupported"
            }
        )
        # 4. Normal edges
        workflow.add_edge("handle_retrieve", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_greeting", END)
        workflow.add_edge("handle_unsupported", END)
        # 5. Compile
        self.app = workflow.compile()
        return self.app
    
    def run(self, question: str):
        return self.app.invoke({"question": question})

# Sau khi build() xong
if __name__ == "__main__":
    # rag = RAGGRaph()
    # app = rag.build()
    # app.get_graph().draw_png("graph_flow.png")

    intent = IntentClassifier(VietnameseEmbedder())
    print(intent.classify("Xin chào"))

