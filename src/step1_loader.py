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
    Loader cho file PDF.
    Sử dụng pymupdf4llm để convert sang markdown (giữ headers, bold, italic).
    Sau đó dùng PyMuPDF để lấy vị trí hình ảnh và chèn vào đúng chỗ.
    """
    
    PAGE_MARKER_TEMPLATE = "\n<!-- PAGE:{} -->\n"
    
    def __init__(self, extract_images: bool = True, 
                 image_output_dir: str = None,
                 image_format: str = "png",
                 image_dpi: int = 150):
        self.extract_images = extract_images
        self.image_format = image_format
        self.image_dpi = image_dpi
        
        if image_output_dir:
            self.image_output_dir = Path(image_output_dir)
        else:
            self.image_output_dir = Path(__file__).parent.parent / "data" / "images"
        
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_image_positions(self, page) -> list:
        """
        Lấy vị trí Y của tất cả hình ảnh trên trang.
        Returns: List of (y_coord, xref, img_index)
        """
        positions = []
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            img_rects = page.get_image_rects(xref)
            if img_rects:
                y_coord = img_rects[0].y0
                positions.append((y_coord, xref, img_index))
        
        return sorted(positions, key=lambda x: x[0])

    def _extract_and_save_images(self, doc, page_num: int, pdf_image_dir: Path) -> dict:
        """
        Trích xuất và lưu tất cả hình ảnh của một trang.
        Returns: dict mapping xref -> relative_path
        """
        page = doc[page_num]
        image_map = {}
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                img_filename = f"page{page_num + 1}_img{img_index}.{image_ext}"
                img_path = pdf_image_dir / img_filename
                
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                
                relative_path = img_path.relative_to(Path(__file__).parent.parent)
                image_map[xref] = str(relative_path)
            except Exception as e:
                print(f"Warning: Could not extract image {img_index} on page {page_num + 1}: {e}")
        
        return image_map

    def _insert_images_at_positions(self, markdown_text: str, page, image_map: dict) -> str:
        """
        Chèn markdown image vào đúng vị trí trong text dựa trên tọa độ Y.
        
        Thuật toán:
        1. Lấy vị trí Y của từng hình ảnh
        2. Lấy vị trí Y của từng text block
        3. Tìm text block gần nhất phía trên mỗi hình ảnh
        4. Chèn markdown img ngay sau text block đó
        """
        if not image_map:
            return markdown_text
        
        # Lấy vị trí các hình ảnh
        image_positions = self._get_image_positions(page)
        if not image_positions:
            return markdown_text
        
        # Lấy text blocks với vị trí
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        for block in blocks:
            if block["type"] == 0:  # Text block
                # Lấy text đầu tiên của block (để tìm trong markdown)
                first_line_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        first_line_text += span.get("text", "")
                    break  # Chỉ lấy dòng đầu
                
                if first_line_text.strip():
                    text_blocks.append({
                        "y": block["bbox"][3],  # Bottom Y (để so sánh với top Y của image)
                        "text_snippet": first_line_text.strip()[:50]  # 50 ký tự đầu
                    })
        
        # Chèn hình ảnh
        result = markdown_text
        inserted_count = 0
        
        for img_y, xref, img_idx in image_positions:
            if xref not in image_map:
                continue
            
            img_path = image_map[xref]
            img_markdown = f"\n\n![image]({img_path})\n\n"
            
            # Tìm text block gần nhất phía TRÊN hình ảnh (y < img_y)
            best_block = None
            for block in text_blocks:
                if block["y"] < img_y:
                    if best_block is None or block["y"] > best_block["y"]:
                        best_block = block
            
            if best_block:
                # Tìm vị trí của text snippet trong markdown
                snippet = best_block["text_snippet"]
                pos = result.find(snippet)
                if pos != -1:
                    # Tìm cuối dòng/paragraph sau snippet
                    end_pos = result.find("\n\n", pos)
                    if end_pos == -1:
                        end_pos = result.find("\n", pos + len(snippet))
                    if end_pos == -1:
                        end_pos = pos + len(snippet)
                    
                    # Chèn image markdown
                    result = result[:end_pos] + img_markdown + result[end_pos:]
                    inserted_count += 1
        
        return result

    def load(self, file_path: str) -> Document:
        """
        Load file PDF với markdown formatting và hình ảnh đúng vị trí.
        """
        path = self._validate_file(file_path)
        
        # Tạo sub-folder cho mỗi PDF
        pdf_name = path.stem
        pdf_image_dir = self.image_output_dir / pdf_name
        pdf_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Mở PDF
        doc = fitz.open(file_path)
        
        # Dùng pymupdf4llm để lấy markdown (giữ formatting)
        pages_md = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        
        full_content = ""
        total_pages = len(pages_md)
        
        for page_idx, page_data in enumerate(pages_md):
            page_text = page_data["text"]
            page_num = page_data["metadata"]["page"]
            
            # Nếu extract_images, chèn hình ảnh vào đúng vị trí
            if self.extract_images:
                page = doc[page_idx]
                image_map = self._extract_and_save_images(doc, page_idx, pdf_image_dir)
                page_text = self._insert_images_at_positions(page_text, page, image_map)
            
            full_content += page_text
            full_content += self.PAGE_MARKER_TEMPLATE.format(page_num-1)
        
        doc.close()
        
        return Document(
            content=full_content,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "file_type": "pdf",
                "total_pages": total_pages,
                "image_dir": str(pdf_image_dir) if self.extract_images else None
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
    file_path = "/home/congtran/RAG_demo/data/documents/tai_lieu_huong_dan_cho_tct.pdf"   
    try:
        documents = loader.load(file_path)
        print(f"Đã load  document(s)")
        print(f"\n--- Metadata ---")
        # print(documents.metadata["images_per_page"])
    #     for key, value in documents.metadata.items():
    #         print(f"  {key}: {value}")
    #     print(f"\n--- Nội dung ---")
    #     print(documents.content)
    
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        print("\nHướng dẫn: Đặt file PDF vào thư mục data/documents/")

   