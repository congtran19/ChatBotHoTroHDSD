"""
Step 7: Re-ranking
Sắp xếp lại kết quả search để tăng độ chính xác
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
from step6_vector_database import SearchResult


class Reranker:
    """
    Re-ranker sử dụng transformer model để score lại query-document pairs.
    
    Input: List[SearchResult] từ vector search
    Output: List[SearchResult] được sắp xếp lại với rerank_score mới
    
    Lợi ích:
    - Chính xác hơn bi-encoder (cosine similarity)
    - Xem xét interaction giữa query và document
    - Giữ nguyên tất cả metadata (page, entities, source, etc.)
    """
    
    def __init__(self, model_path: str = "AITeamVN/Vietnamese_Reranker"):
        """
        Args:
            model_path: HuggingFace model path
                - Default: AITeamVN/Vietnamese_Reranker (Vietnamese-optimized)
                - Alternative: cross-encoder/ms-marco-MiniLM-L-12-v2 (multilingual)
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    def rerank(
        self, 
        query: str, 
        search_results: List[SearchResult], 
        top_k: int = None,
        max_length: int = 512
    ) -> List[SearchResult]:
        """
        Re-rank search results dựa trên query.
        
        Args:
            query: Câu hỏi của user
            search_results: Kết quả từ vector search (SearchResult objects)
            top_k: Số lượng kết quả trả về (None = trả về tất cả)
            max_length: Max token length cho model
            
        Returns:
            List[SearchResult] đã được re-rank, sorted theo rerank_score (cao → thấp)
        """
        if not search_results:
            return []
        
        # Prepare query-document pairs
        documents = [result.content for result in search_results]
        pairs = [[query, doc] for doc in documents]
        
        # Score with model
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_length
            )      
            # Get scores
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
        
        # Create reranked results (preserve all metadata, update score)
        reranked_results = []
        for result, score in zip(search_results, scores):
            # Create new SearchResult with updated score
            reranked_result = SearchResult(
                content=result.content,
                metadata=result.metadata,  # Keep ALL metadata!
                score=float(score),  # New rerank score
                chunk_id=result.chunk_id,
                knn_score=result.knn_score,  # Keep original kNN score
                bm25_score=result.bm25_score  # Keep original BM25 score
            )
            reranked_results.append(reranked_result)
        
        # Sort by new score (descending)
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k if specified
        if top_k:
            return reranked_results[:top_k]
        return reranked_results
    
#test
if __name__ == "__main__":
    print("="*80)
    print("🧪 TEST RERANKER")
    print("="*80)
    
    reranker = Reranker("AITeamVN/Vietnamese_Reranker")
    
    # ============ TEST 2: Reranking with SearchResult (Real Use Case) ============
    print("\n\n🔍 TEST 2: Reranking with SearchResult (with metadata)")
    print("-" * 80)
    
    from step6_vector_database import SearchResult, ElasticsearchVectorStore
    from step5_embedding import VietnameseEmbedder
    
    
    query = "Làm thế nào để đổi mật khẩu?"
    query_embedded = VietnameseEmbedder().embed_query(query)
    query_embedded = query_embedded.tolist() if hasattr(query_embedded, 'tolist') else query_embedded
    results = ElasticsearchVectorStore().search_hybrid(query, query_embedded, top_k=5)
    print(f"Query: {query}")
    print(f"Initial results: {len(results)}\n")
    
    print("BEFORE re-ranking (sorted by kNN score):")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] kNN Score: {result.knn_score:.4f} | Page: {result.metadata['page']}")
        print(f"    Content: {result.content}...")
    
    # Re-rank
    reranked = reranker.rerank(query, results, top_k=3)
    
    print("\n\nAFTER re-ranking:")
    for i, result in enumerate(reranked, 1):
        print(f"\n[{i}] Rerank Score: {result.score:.4f} (kNN: {result.knn_score:.4f}, BM25: {result.bm25_score:.4f})")
        print(f"    Page: {result.metadata['page']} | Source: {result.metadata['source']}")
        print(f"    Section: {result.metadata.get('h2', 'N/A')}")
        print(f"    Chunk ID: {result.chunk_id}")
        print(f"    Content: {result.content}...")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)