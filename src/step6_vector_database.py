"""
Bước 6: Vector Database với Elasticsearch
Lưu trữ và tìm kiếm vector embeddings
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from elasticsearch import Elasticsearch
from step5_embedding import EmbeddingResult
from step5_embedding import VietnameseEmbedder

@dataclass
class IntentExample:
    """Dữ liệu mẫu cho phân loại intent"""
    text: str           # Câu hỏi mẫu, ví dụ: "Xin chào"
    intent: str         # Label, ví dụ: "greeting"
    score: float = 0.0  # Điểm similarity khi tìm được (để trống khi add, có giá trị khi search)

@dataclass
class SearchResult:
    """Kết quả tìm kiếm từ vector store"""
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
    knn_score: Optional[float] = None  # Score từ kNN search
    bm25_score: Optional[float] = None  # Score từ BM25 search


class BaseVectorStore(ABC):
    """Abstract base class cho Vector Store"""
    
    @abstractmethod
    def add(self, embeddings: List[EmbeddingResult]) -> List[str]:
        """
        Thêm embeddings vào store
        
        Args:
            embeddings: Danh sách EmbeddingResult cần lưu
            
        Returns:
            List[str]: Danh sách IDs của documents đã thêm
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """
        Tìm kiếm các documents tương tự với query
        
        Args:
            query_embedding: Vector embedding của query
            top_k: Số lượng kết quả trả về
            
        Returns:
            List[SearchResult]: Danh sách kết quả tìm kiếm
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Xóa documents theo IDs"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Xóa toàn bộ dữ liệu trong store"""
        pass


class ElasticsearchVectorStore():
    """
    Vector Store sử dụng Elasticsearch với dense_vector field.
    
    Yêu cầu:
    - Elasticsearch 8.x với hỗ trợ kNN search
    - Plugin dense_vector
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "rag_vectors",
        embedding_dim: int = 768,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False,
    ):
        """
        Khởi tạo connection tới Elasticsearch
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            index_name: Tên index để lưu vectors
            embedding_dim: Số chiều của embedding vector
            username: Username (nếu có authentication)
            password: Password (nếu có authentication)
        """
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        
        # Cấu hình connection
        es_config = {
            "hosts": [{"host": host, "port": port, "scheme": "https" if use_ssl else "http"}]
        }
        
        if username and password:
            es_config["basic_auth"] = (username, password)
        
        self.client = Elasticsearch(**es_config)
        
        # Tạo index nếu chưa tồn tại
        self._create_index_if_not_exists()
    
    def _create_index_if_not_exists(self):
        """Tạo index với mapping phù hợp cho vector search"""
        if self.client.indices.exists(index=self.index_name):
            return
        
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "metadata": {"type": "object"},
                    "chunk_id": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }   
        }
        self.client.indices.create(index=self.index_name, body=mapping)

    
    def add(self, embeddings: List[EmbeddingResult]) -> List[str]:
        ids = []

        for item in embeddings:
            # Convert numpy array sang list nếu cần
            emb_list = item.embeddings.tolist() if hasattr(item.embeddings, 'tolist') else item.embeddings
            
            body = {
                "content": item.content,
                "metadata": item.metadata,
                "chunk_id": item.idext,
                "embedding": emb_list
            }

            resp = self.client.index(index=self.index_name, document=body)
            ids.append(resp["_id"])

        return ids

    
    def search_by_knn(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        query_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": max(50, top_k * 10)
            }
        }
        resp = self.client.search(index=self.index_name, body=query_body)
        results = []    
        for hit in resp["hits"]["hits"]:
            results.append(
                SearchResult(
                    content=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {}),
                    chunk_id=hit["_source"]["chunk_id"],
                    score=hit["_score"]
                )
            )

        return results

    def search_by_BM25(self,query: str, top_k: int = 5) -> List[SearchResult]:
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": query}},
                        {"match": {"metadata.description": query}}
                    ]
                }
            }
        }
        resp = self.client.search(index=self.index_name, body=query_body)
        results = []
        for hit in resp["hits"]["hits"]:
            results.append(
                SearchResult(
                    content=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {}),
                    chunk_id=hit["_source"]["chunk_id"],
                    score=hit["_score"]
                )
            )

        return results
    def search_hybrid(self, query: str, query_embedding: List[float], top_k: int = 5, vector_weight: float = 0.5) -> List[SearchResult]:
        """
        Tìm kiếm kết hợp (Hybrid Search): kNN + BM25
        
        Chạy 2 query riêng rồi merge kết quả để hiển thị từng điểm số.
        
        Args:
            query: Câu text query (dùng cho BM25)
            query_embedding: Vector của query (dùng cho kNN)
            top_k: Số lượng kết quả
            vector_weight: Trọng số của vector search (0.0 - 1.0). Text weight sẽ là 1.0 - vector_weight
        
        Returns:
            List[SearchResult] với knn_score, bm25_score, và score (tổng hợp)
        """
        text_weight = 1.0 - vector_weight
        
        # 1. Chạy kNN search
        knn_results = self.search_by_knn(query_embedding, top_k=top_k) 
        
        # 2. Chạy BM25 search
        bm25_results = self.search_by_BM25(query, top_k=top_k   )
        
        # 3. Tạo dict để merge (key = chunk_id)
        merged = {}
        
        # Thêm kNN results
        for result in knn_results:
            merged[result.chunk_id] = {
                "content": result.content,
                "metadata": result.metadata,
                "knn_score": result.score,
                "bm25_score": 0.0
            }
        
        # Thêm BM25 results
        for result in bm25_results:
            if result.chunk_id in merged:
                merged[result.chunk_id]["bm25_score"] = result.score
            else:
                merged[result.chunk_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "knn_score": 0.0,
                    "bm25_score": result.score
                }
        
        # 4. Tính combined score và tạo SearchResult
        final_results = []
        for chunk_id, data in merged.items():
            combined_score = (data["knn_score"] * vector_weight) + (data["bm25_score"] * text_weight)
            
            final_results.append(
                SearchResult(
                    content=data["content"],
                    metadata=data["metadata"],
                    chunk_id=chunk_id,
                    score=combined_score,
                    knn_score=data["knn_score"],
                    bm25_score=data["bm25_score"]
                )
            )
        
        # 5. Sắp xếp theo combined score và lấy top_k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]
    
    def delete(self, ids: List[str]) -> bool:
        for doc_id in ids:
            self.client.delete(index=self.index_name, id=doc_id)
        return True
    
    def count(self) -> int:
        """Đếm số lượng documents trong index"""
        response = self.client.count(index=self.index_name)
        return response["count"]
    
    def get_by_id(self, doc_id: str) -> Optional[SearchResult]:
        """Lấy document theo ID"""
        resp = self.client.get(index=self.index_name, id=doc_id)
        return SearchResult(
            content=resp["_source"]["content"],
            metadata=resp["_source"].get("metadata", {}),
            chunk_id=resp["_source"]["chunk_id"],
            score=resp["_score"]
        )
    
    def clear(self) -> bool:
        """Xóa toàn bộ dữ liệu trong index"""
        self.client.delete_by_query(index=self.index_name, body={"query": {"match_all": {}}})
        return True       

