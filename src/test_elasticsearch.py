"""
Test file để thử nghiệm tất cả các hàm trong ElasticsearchVectorStore

Các bước test:
1. Khởi tạo vector store
2. Load và embed tài liệu 
3. Test add() - thêm embeddings
4. Test count() - đếm số documents
5. Test search_by_knn() - tìm kiếm vector
6. Test search_by_BM25() - tìm kiếm text
7. Test search_hybrid() - tìm kiếm kết hợp
8. Test get_by_id() - lấy document theo ID
9. Test delete() - xóa documents
10. Test clear() - xóa toàn bộ
"""

from step1_loader import DocumentLoader
from step2_preprocessing import TextPreprocessor
from step4_chungking import MarkdownChunker
from step5_embedding import VietnameseEmbedder
from step6_vector_database import ElasticsearchVectorStore


#Viết lại hết thành class cho dễ test
class TestElasticsearchVectorStore:
    def __init__(self):
        self.vector_db = ElasticsearchVectorStore()

    def test_elasticsearch_vector_store(self):
        self.vector_db.add(embeddings)
        self.vector_db.search_by_knn(query_embedding, top_k=3)
        self.vector_db.search_by_BM25(query, top_k=3)
        self.vector_db.search_hybrid(query, query_embedding, top_k=3, vector_weight=0.5)
        self.vector_db.get_by_id(ids[0])
        self.vector_db.delete(ids[:2])
        self.vector_db.clear()

if __name__ == "__main__":
    test_elasticsearch_vector_store()
