"""
Bước 1: Document Loading
Load tài liệu từ các định dạng khác nhau (PDF, Word, TXT)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
import pymupdf4llm
@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        Load tài liệu từ file
        
        Args:
            file_path: Đường dẫn đến file
            
        Returns:
            List[Document]: Danh sách documents (có thể nhiều nếu PDF có nhiều trang)
        """
        pass
    
    def _validate_file(self, file_path: str) -> Path:
        """Kiểm tra file có tồn tại không"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        return path


class PDFLoader(BaseLoader):
    """
    Loader cho file PDF
    Gộp tất cả trang thành 1 Document và chèn PAGE_MARKER
    để chunker có thể tính số trang cho từng chunk.
    """
    
    # Marker định dạng: <!-- PAGE:X --> với X là số trang
    PAGE_MARKER_TEMPLATE = "\n<!-- PAGE:{} -->\n"

    def load(self, file_path: str) -> List[Document]:
        """
        Load file PDF - gộp tất cả trang thành 1 Document
        Chèn PAGE_MARKER vào cuối mỗi trang để tracking số trang
        """
        path = self._validate_file(file_path)

        # Lấy markdown theo từng trang
        pages_md = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        
        # Gộp tất cả nội dung lại, chèn PAGE_MARKER sau mỗi trang
        full_content = ""
        
        for page_data in pages_md:
            page_text = page_data["text"]
            page_num = page_data["metadata"]["page"]  # +1 vì index từ 0
            
            # Thêm nội dung trang + marker
            full_content += page_text
            full_content += self.PAGE_MARKER_TEMPLATE.format(page_num)
        
        return Document(
            content=full_content,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "file_type": "pdf",
                "total_pages": len(pages_md)
            }
        )

class TextLoader(BaseLoader):
    """
    Loader cho file text (.txt)
    """
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Document]:
        """Load file text"""
        path = self._validate_file(file_path)
        
        with open(file_path, "r", encoding=self.encoding) as f:
            content = f.read()
        
        return [Document(
            content=content,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "file_type": "txt"
            }
        )]


class DocumentLoader:
    """
    Factory class - tự động chọn loader phù hợp dựa vào extension
    
    Usage:
        loader = DocumentLoader()
        docs = loader.load("path/to/file.pdf")
    """
    
    # Mapping extension -> Loader class
    LOADERS = {
        ".pdf": PDFLoader,
        # ".docx": WordLoader,
        ".txt": TextLoader,
    }
    
    def load(self, file_path: str, **kwargs) -> List[Document]:
        """
        Load file với loader phù hợp
        
        Args:
            file_path: Đường dẫn file
            **kwargs: Tham số truyền cho loader cụ thể
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in self.LOADERS:
            supported = ", ".join(self.LOADERS.keys())
            raise ValueError(f"Không hỗ trợ định dạng {ext}. Hỗ trợ: {supported}")
        
        loader_class = self.LOADERS[ext]
        loader = loader_class(**kwargs) if kwargs else loader_class()
        
        return loader.load(file_path)
    
    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """
        Load tất cả files từ một thư mục
        
        Args:
            directory: Đường dẫn thư mục
            recursive: Có tìm trong thư mục con không
        """
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {directory}")
        
        pattern = "**/*" if recursive else "*"
        all_documents = []
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.LOADERS:
                try:
                    docs = self.load(str(file_path))
                    all_documents.extend(docs)
                    print(f"✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"✗ Lỗi khi load {file_path.name}: {e}")
        
        return all_documents


# === THỬ NGHIỆM ===
if __name__ == "__main__":
    loader = DocumentLoader()
    file_path = "/home/congtran/Thực Hành RAG/data/documents/HDSD Quản lý tài khoản.pdf"   
    try:
        documents = loader.load(file_path)
        print(f"Đã load  document(s)")
        print(f"\n--- Metadata ---")
        for key, value in documents.metadata.items():
            print(f"  {key}: {value}")
        print(f"\n--- Nội dung ---")
        print(documents.content)
    
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        print("\nHướng dẫn: Đặt file PDF vào thư mục data/documents/")

   