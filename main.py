#Test graph
# main.py
import sys
sys.path.append("src")

from graph import RAGGRaph

def main():
    print(" RAG Chatbot - Gõ 'exit' để thoát")
    print("-" * 50)
    
    rag = RAGGRaph()
    rag.build()
    
    while True:
        question = input("\n Bạn: ").strip()
        
        if question.lower() in ["exit", "quit", "thoát"]:
            print("👋 Tạm biệt!")
            break
        
        if not question:
            continue
        
        result = rag.run(question)
        print(f"\n Bot: {result['response']}")

if __name__ == "__main__":
    main()