"""
Bước 4: Text Chunking for Markdown
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from step3_NER import ExtractEntityHuggingFace
from step1_loader import Document
from step2_preprocessing import TextPreprocessor


@dataclass
class Chunk:
    content: str
    index: int
    metadata: Dict[str, Any]


class Chunker(ABC):
    @abstractmethod
    def chunk_document(self, text: str, source_path: str = "", page: int | None = None) -> List[Chunk]:
        pass   


# class MarkdownChunker(Chunker):
#     """
#     Chunker sử dụng MarkdownHeaderTextSplitter để chia văn bản markdown.
#     Hỗ trợ tìm số trang cho chunk dựa trên PAGE_MARKER.
#     """
    
#     # Regex để tìm PAGE_MARKER: <!-- PAGE:X -->
#     PAGE_MARKER_PATTERN = re.compile(r'<!-- PAGE:(\d+) -->')
    
#     def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.headers = [
#             ("#", "h1"),
#             ("##", "h2"),
#             ("###", "h3"),
#         ]
#         self.splitter = MarkdownHeaderTextSplitter(
#             headers_to_split_on=self.headers,
#         )
#         self.recursive_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap
#         )
#         self.entity_extractor = ExtractEntityHuggingFace(
#             tokenizer_name="NlpHUST/ner-vietnamese-electra-base",
#             model_name="NlpHUST/ner-vietnamese-electra-base"
#         )
    
#     def _find_page_for_chunk(self, chunk_content: str, full_text: str) -> int:
#         """
#         Tìm số trang cho chunk dựa vào PAGE_MARKER.
        
#         Thuật toán đơn giản:
#         1. Nếu chunk CÓ chứa PAGE_MARKER → page = marker đầu tiên - 1
#            (vì nội dung chunk nằm TRƯỚC marker đó)
#         2. Nếu chunk KHÔNG chứa PAGE_MARKER → tìm marker gần nhất TRƯỚC chunk
        
#         Args:
#             chunk_content: Nội dung của chunk
#             full_text: Toàn bộ text gốc (còn chứa PAGE_MARKER)
            
#         Returns:
#             int: Số trang của chunk (bắt đầu từ 1)
#         """
        

#         # Case 1: Chunk có chứa PAGE_MARKER
#         marker_in_chunk = self.PAGE_MARKER_PATTERN.search(chunk_content)
#         if marker_in_chunk:
#             # page = marker đầu tiên - 1 (vì nội dung nằm TRƯỚC marker)
#             page = int(marker_in_chunk.group(1)) - 1
#             return max(1, page)  # Đảm bảo page >= 1
        
#         # Case 2: Chunk không chứa PAGE_MARKER → tìm vị trí chunk trong full_text
#         search_text = chunk_content[:min(50, len(chunk_content))]
#         chunk_start = full_text.find(search_text)
        
#         if chunk_start == -1:
#             return 1  # Fallback
        
#         # Tìm marker gần nhất TRƯỚC chunk_start
#         page = 1
#         for match in self.PAGE_MARKER_PATTERN.finditer(full_text):
#             if match.start() < chunk_start:
#                 page = int(match.group(1))
#             else:
#                 break
        
#         return page
    
#     def _remove_page_markers(self, text: str) -> str:
#         """Xóa tất cả PAGE_MARKER khỏi text"""
#         return self.PAGE_MARKER_PATTERN.sub('', text)
    
#     def chunk_document(self, document: Document) -> List[Chunk]:
#         """
#         Chia document thành các chunk và gán metadata số trang.
        
#         Args:
#             document: Document chứa nội dung với PAGE_MARKER
            
#         Returns:
#             List[Chunk] với metadata bao gồm page (số trang)
#         """
#         original_text = document.content  # Text gốc với PAGE_MARKER
#         parent_metadata = document.metadata
        
#         # ← LangChain chia chunk (nó sẽ tự xóa PAGE_MARKER vì đó là HTML comment)
#         chunks = self.splitter.split_text(original_text)
#         chunk_list = []
        
#         for i, chunk in enumerate(chunks):
#             content = chunk.page_content
#             metadata = chunk.metadata.copy()
            
#             # Tính page cho chunk (dùng text gốc còn PAGE_MARKER)
#             page = self._find_page_for_chunk(content, original_text)
#             metadata["page"] = page
            
#             # Xóa PAGE_MARKER khỏi content nếu còn sót
#             content = self._remove_page_markers(content)
            
#             metadata.update({
#                 "chunk_index": i,
#                 "total_chunks": len(chunks),
#                 "source": parent_metadata.get("file_name", ""),
#                 "total_pages": parent_metadata.get("total_pages", None)
#             })
#             chunk_list.append(Chunk(content=content, index=i, metadata=metadata))

#             #kết hợp markdownchunk và recursivecharacter để kích thước mỗi chunk không quá 500
#             for i, chunk in enumerate(chunk_list):
#                 content = chunk.content
#                 metadata = chunk.metadata.copy()
#                 if len(content) > self.chunk_size:
#                     chunks = self.recursive_splitter.split_text(content)
#                     sub_chunk_list = []
#                     for j, sub_chunk in enumerate(chunks):
#                         sub_content = sub_chunk.page_content
#                         sub_metadata = chunk.metadata.copy()
#                         sub_metadata.update({
#                             "chunk_index": j,
#                             "total_chunks": len(chunks),
#                             "source": parent_metadata.get("file_name", ""),
#                             "total_pages": parent_metadata.get("total_pages", None)
#                         })
#                         sub_chunk_list.append(Chunk(content=sub_content, index=j, metadata=sub_metadata))
#                     chunk_list.append(sub_chunk_list)   
#                     chunk_list.remove(chunk)
#                 entities = self.entity_extractor.extract(content)
#                 metadata.update({"entities": entities})
            
#         return chunk_list

class MarkdownChunker(Chunker):

    PAGE_MARKER_PATTERN = re.compile(r'<!-- PAGE:(\d+) -->')

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.entity_extractor = ExtractEntityHuggingFace(
            tokenizer_name="NlpHUST/ner-vietnamese-electra-base",
            model_name="NlpHUST/ner-vietnamese-electra-base"
        )

    def _remove_page_markers(self, text):
        return self.PAGE_MARKER_PATTERN.sub("", text)

    def _find_page_for_position(self, pos, full_text):
        # tìm marker gần nhất trước pos
        page = 1
        for match in self.PAGE_MARKER_PATTERN.finditer(full_text):
            if match.start() < pos:
                page = int(match.group(1))
            else:
                break
        return page
    import re


    def _find_images_for_chunk(self, chunk_content: str) -> List[str]:
        pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
        return pattern.findall(chunk_content)

    def _remove_images(self, text: str) -> str:
        pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
        return pattern.sub("", text)
    def chunk_document(self, document: Document) -> List[Chunk]:
        original_text = document.content
        parent_meta = document.metadata

        # Stage 1: Markdown split
        md_chunks = self.splitter.split_text(original_text)

        # Stage 2: Recursive chunk ALL markdown chunks
        final_chunks = []

        for md_chunk in md_chunks:
            content = md_chunk.page_content
            md_meta = md_chunk.metadata.copy()

            # recursive split into smaller chunks
            small_chunks = self.recursive_splitter.split_text(content)

            for sub in small_chunks:
                final_chunks.append({
                    "content": sub,
                    "markdown_meta": md_meta
                })

        # Stage 3: Build final Chunk objects + metadata
        output = []
        for idx, ch in enumerate(final_chunks):
            content = ch["content"]
            images = self._find_images_for_chunk(content)
            # Tìm vị trí content trong text gốc để map số trang
            snippet = content[:50]
            pos = original_text.find(snippet)
            page = self._find_page_for_position(pos, original_text)

            # Remove markers
            cleaned = self._remove_page_markers(content)

            #remove images marker
            cleaned = self._remove_images(cleaned)

            # NER
            entities = self.entity_extractor.extract(cleaned)

            # Metadata
            meta = {
                **ch["markdown_meta"],
                "page": page,
                "chunk_index": idx,
                "total_chunks": len(final_chunks),
                "source": parent_meta.get("file_name", ""),
                "total_pages": parent_meta.get("total_pages", None),
                "entities": entities,
                "images": images
            }

            output.append(Chunk(
                content=cleaned,
                index=idx,
                metadata=meta
            ))

        return output


# === TEST ===
if __name__ == "__main__":
    from step1_loader import DocumentLoader
    
    chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
    loader = DocumentLoader()
    preprocessor = TextPreprocessor()
    
    file_path = "/home/congtran/RAG_demo/data/documents/tai_lieu_huong_dan_cho_tct.pdf"
    
    try:
        doc = loader.load(file_path)
        print(f"✓ Loaded document: {doc.metadata.get('file_name')}")
        print(f"  Total pages: {doc.metadata.get('total_pages')}")
        
        # Preprocess (PAGE_MARKER vẫn được giữ lại)
        doc.content = preprocessor.preprocess(doc.content)
        
        # Chunk
        chunks = chunker.chunk_document(doc)
        print(f"\n✓ Created {len(chunks)} chunks\n")
        
        for chunk in chunks:  # 
            print(f"--- Chunk {chunk.index} ---")
            print(f'--- Metadata ---')
            for key, value in chunk.metadata.items():
                print(f"  {key}: {value}")
            print(f'\n--- Nội dung ---')
            print(chunk.content)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")