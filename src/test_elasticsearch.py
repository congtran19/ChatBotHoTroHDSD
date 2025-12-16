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


def test_elasticsearch_vector_store():
    """Test đầy đủ các hàm của ElasticsearchVectorStore"""
    
    print("="*80)
    print("🧪 BẮT ĐẦU TEST ELASTICSEARCH VECTOR STORE")
    print("="*80)
    
    # ============ BƯỚC 1: Khởi tạo các components ============
    print("\n📦 BƯỚC 1: Khởi tạo các components...")
    loader = DocumentLoader()
    preprocessor = TextPreprocessor()
    chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
    embedder = VietnameseEmbedder()
    vector_db = ElasticsearchVectorStore(
        host="localhost",
        port=9200,
        index_name="rag_test_vectors",
        embedding_dim=768
    )
    print("✓ Đã khởi tạo tất cả components")
    
    # # ============ BƯỚC 2: Load và xử lý document ============
    # print("\n📄 BƯỚC 2: Load và xử lý document...")
    # file_path = "/home/congtran/Thực Hành RAG/data/documents/tai_lieu_huong_dan_cho_tct.pdf"
    
    # try:
    #     doc = loader.load(file_path)
    #     print(f"✓ Loaded: {doc.metadata.get('file_name')}")
    #     print(f"  - Pages: {doc.metadata.get('total_pages')}")
    #     # Preprocess
    #     doc.content = preprocessor.preprocess(doc.content)
    #     print("✓ Preprocessed document")
        
    #     # Chunk
    #     chunks = chunker.chunk_document(doc)
    #     print(f"✓ Created {len(chunks)} chunks")
        
    #     # Embed
    #     embeddings = embedder.embed(chunks)
    #     print(f"✓ Created {len(embeddings)} embeddings")
        
    # except FileNotFoundError as e:
    #     print(f"❌ Lỗi: File không tồn tại - {e}")
    #     return
    # except Exception as e:
    #     print(f"❌ Lỗi không xác định: {e}")
    #     return
    
    # # ============ BƯỚC 3: TEST add() - Thêm embeddings ============
    # print("\n➕ BƯỚC 3: TEST add() - Thêm embeddings vào Elasticsearch...")
    # try:
    #     # Clear trước để đảm bảo database sạch
    #     vector_db.clear()
    #     print("✓ Đã clear database trước khi test")
        
    #     ids = vector_db.add(embeddings)
    #     print(f"✓ Đã thêm {len(ids)} documents")
    #     print(f"  - Sample IDs: {ids[:]}")
    # except Exception as e:
    #     print(f"❌ Lỗi khi add: {e}")
    #     return
    
    # # ============ BƯỚC 4: TEST count() - Đếm documents ============
    # print("\n🔢 BƯỚC 4: TEST count() - Đếm số documents...")
    # try:
    #     count = vector_db.count()
    #     print(f"✓ Tổng số documents trong index: {count}")
    # except Exception as e:
    #     print(f"❌ Lỗi khi count: {e}")
    
    # ============ BƯỚC 5: TEST search_by_knn() - Vector Search ============
    print("\n🔍 BƯỚC 5: TEST search_by_knn() - Tìm kiếm bằng vector...")
    test_query = "Làm thế nào để đổi mật khẩu?"
    print(f"  Query: '{test_query}'")
    
    try:
        # Embed query
        query_embedding = embedder.embed_query(test_query)
        query_embedding = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
        # Search
        knn_results = vector_db.search_by_knn(query_embedding, top_k=3)
        print(f"✓ Tìm thấy {len(knn_results)} kết quả")
        
        for i, result in enumerate(knn_results, 1):
            print(f"\n  [{i}] Score: {result.score:.4f}")
            print(f"      Chunk ID: {result.chunk_id}")
            print(f"      Content preview: {result.content[:150]}...")
            print(f"      Metadata: {result.metadata}")
    except Exception as e:
        print(f"❌ Lỗi khi search_by_knn: {e}")
    
    # # ============ BƯỚC 6: TEST search_by_BM25() - Text Search ============
    # print("\n📝 BƯỚC 6: TEST search_by_BM25() - Tìm kiếm bằng text...")
    # print(f"  Query: '{test_query}'")
    
    # try:
    #     bm25_results = vector_db.search_by_BM25(test_query, top_k=3)
    #     print(f"✓ Tìm thấy {len(bm25_results)} kết quả")
        
    #     for i, result in enumerate(bm25_results, 1):
    #         print(f"\n  [{i}] Score: {result.score:.4f}")
    #         print(f"      Chunk ID: {result.chunk_id}")
    #         print(f"      Content preview: {result.content[:150]}...")
    #         print(f"      Metadata: {result.metadata}")
    # except Exception as e:
    #     print(f"❌ Lỗi khi search_by_BM25: {e}")
    
    # # ============ BƯỚC 7: TEST search_hybrid() - Hybrid Search ============
    # print("\n🔀 BƯỚC 7: TEST search_hybrid() - Tìm kiếm kết hợp (Hybrid)...")
    # print(f"  Query: '{test_query}'")
    
    # try:
    #     # Test với nhiều vector_weight khác nhau
    #     for vector_weight in [0.5]:
    #         print(f"\n  ⚖️  Vector weight: {vector_weight}, Text weight: {1-vector_weight}")
            
    #         hybrid_results = vector_db.search_hybrid(
    #             query=test_query,
    #             query_embedding=query_embedding,
    #             top_k=3,
    #             vector_weight=vector_weight
    #         )
            
    #         print(f"  ✓ Tìm thấy {len(hybrid_results)} kết quả:")
    #         for i, result in enumerate(hybrid_results, 1):
    #             print(f"\n    [{i}] Combined Score: {result.score:.4f}")
    #             print(f"        - kNN Score: {result.knn_score:.4f}")
    #             print(f"        - BM25 Score: {result.bm25_score:.4f}")
    #             print(f"        - Chunk ID: {result.chunk_id}")
    #             print(f"        - Content: {result.content[:100]}...")
    #             print(f"        - Metadata: {result.metadata}")
    # except Exception as e:
    #     print(f"❌ Lỗi khi search_hybrid: {e}")
    
    # # ============ BƯỚC 8: TEST get_by_id() - Lấy document theo ID ============
    # print("\n🆔 BƯỚC 8: TEST get_by_id() - Lấy document theo ID...")
    # try:
    #     if ids:
    #         test_id = ids[0]
    #         print(f"  Đang lấy document với ID: {test_id}")
            
    #         doc_result = vector_db.get_by_id(test_id)
    #         print(f"✓ Đã lấy được document")
    #         print(f"  - Chunk ID: {doc_result.chunk_id}")
    #         print(f"  - Content preview: {doc_result.content[:150]}...")
    #         print(f"  - Metadata: {doc_result.metadata}")
    # except Exception as e:
    #     print(f"❌ Lỗi khi get_by_id: {e}")
    
    # # ============ BƯỚC 9: TEST delete() - Xóa documents ============
    # print("\n🗑️  BƯỚC 9: TEST delete() - Xóa documents theo ID...")
    # try:
    #     # Đếm trước khi xóa
    #     count_before = vector_db.count()
    #     print(f"  Số documents trước khi xóa: {count_before}")
        
    #     # Xóa 2 documents đầu tiên
    #     ids_to_delete = ids[:2]
    #     print(f"  Đang xóa {len(ids_to_delete)} documents...")
        
    #     success = vector_db.delete(ids_to_delete)
    #     print(f"✓ Xóa thành công: {success}")
        
    #     # Đếm sau khi xóa
    #     count_after = vector_db.count()
    #     print(f"  Số documents sau khi xóa: {count_after}")
    #     print(f"  Đã xóa: {count_before - count_after} documents")
    # except Exception as e:
    #     print(f"❌ Lỗi khi delete: {e}")
    
    # # ============ BƯỚC 10: TEST clear() - Xóa toàn bộ ============
    # print("\n🧹 BƯỚC 10: TEST clear() - Xóa toàn bộ database...")
    # try:
    #     count_before_clear = vector_db.count()
    #     print(f"  Số documents trước khi clear: {count_before_clear}")
        
    #     success = vector_db.clear()
    #     print(f"✓ Clear thành công: {success}")
        
    #     count_after_clear = vector_db.count()
    #     print(f"  Số documents sau khi clear: {count_after_clear}")
    # except Exception as e:
    #     print(f"❌ Lỗi khi clear: {e}")
    
#    ============ KẾT THÚC TEST ============
    print("\n" + "="*80)
    print("✅ ĐÃ HOÀN THÀNH TẤT CẢ CÁC TEST")
    print("="*80)


if __name__ == "__main__":
    test_elasticsearch_vector_store()
