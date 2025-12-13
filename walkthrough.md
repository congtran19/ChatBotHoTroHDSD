# RAG Pipeline Walkthrough

Tài liệu này hướng dẫn chi tiết các bước xây dựng hệ thống RAG (Retrieval Augmented Generation) cho project này.

## Architecture Overview

Pipeline gồm 6 bước chính:
1. **Loader**: Đọc tài liệu
2. **Preprocessing**: Làm sạch đặc biệt
3. **NER**: Trích xuất thực thể
4. **Chunking**: Chia nhỏ văn bản
5. **Embedding**: Vector hóa
6. **Vector DB**: Lưu trữ & Tìm kiếm

---

## Step 1: Document Loader (`src/step1_loader.py`)

**Mục tiêu**: Đọc file PDF và chuẩn bị dữ liệu thô.

**Logic quan trọng**:
- Sử dụng `pymupdf4llm` để chuyển PDF sang Markdown.
- **Gộp trang**: Thay vì trả về từng trang rời rạc, loader gộp toàn bộ nội dung thành 1 chuỗi lớn để giữ ngữ cảnh liền mạch.
- **Page Marker**: Chèn marker `<!-- PAGE:x -->` vào cuối mỗi trang để tracking số trang sau này, kể cả khi text bị xử lý.

```python
# Marker định dạng
PAGE_MARKER_TEMPLATE = "\n<!-- PAGE:{} -->\n"
```

## Step 2: Preprocessing (`src/step2_preprocessing.py`)

**Mục tiêu**: Làm sạch văn bản trước khi xử lý sâu hơn.

**Các bước xử lý**:
- Chuẩn hóa khoảng trắng.
- Xóa các ký tự nhiễu.
- **Lưu ý**: Cần đảm bảo không xóa mất `PAGE_MARKER` để Step 4 còn dùng được.

## Step 3: Named Entity Recognition (`src/step3_NER.py`)

**Mục tiêu**: Trích xuất các thực thể quan trọng (Tên người, địa điểm, tổ chức...) để làm metadata phong phú hơn cho việc tìm kiếm.

**Công nghệ**:
- Sử dụng model HuggingFace: `NlpHUST/ner-vietnamese-electra-base`.

## Step 4: Chunking (`src/step4_chungking.py`)

**Mục tiêu**: Chia văn bản thành các đoạn nhỏ (chunks) có ý nghĩa.

**Logic quan trọng**:
- Dùng `MarkdownHeaderTextSplitter` để chia theo header (h1, h2, h3).
- **Page Tracking**:
  - Dùng Regex tìm `<!-- PAGE:x -->`.
  - Nếu chunk chứa marker → `page = marker - 1`.
  - Nếu không → tìm marker gần nhất phía trước.
- **Metadata**: Mỗi chunk có metadata gồm: `chunk_index`, `page`, `source`, `entities`...

## Step 5: Embedding (`src/step5_embedding.py`)

**Mục tiêu**: Chuyển đổi text thành vector số học.

**Công nghệ**:
- Model: `dangvantuan/vietnamese-embedding` (SentenceTransformer).
- Tokenizer: `pyvi`.

## Step 6: Vector Database (`src/step6_vector_database.py`)

**Mục tiêu**: Lưu trữ vectors và tìm kiếm tương đồng (Similarity Search).

**Công nghệ**: **Elasticsearch**
- **Index**: `rag_vectors`
- **Field**: `dense_vector` (dims=768, cosine similarity).
- **Search**:
  - kNN Search (Vector).
  - Hybrid Search (Vector + Keyword BM25) - *Optional*.

---

## How to Run

1.  **Cài đặt**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Chạy Elasticsearch (Docker)**:
    ```bash
    docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0
    ```

3.  **Chạy Pipeline**:
    (Script main sẽ được cập nhật sau)
