import re
import unicodedata

class TextPreprocessor:
    def __init__(self, config = None):
        self.config = config
    
    def preprocess(self, text) -> str:
        """Pipeline chính"""
        text = self.remove_extra_newline(text)
        text = self.remove_toc_until_introduction(text)
        text = self._convert_bold_number_headers_to_markdown(text)
        text = self.remove_markdown_withoutinformation(text)
        text = self.normalize_unicode(text)
        text = self.remove_extra_whitespace(text)
        text = self.remove_special_characters(text)
        
        # text = self.lowercasing(text)
        #text = self.ner(text)
        return text
    def remove_extra_newline(self, text) -> str:
        """
        Loại bỏ newline thừa và dòng chỉ có whitespace
        """
        # Bước 1: Loại bỏ zero-width space và các ký tự vô hình khác
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        # Bước 2: Loại bỏ newline + whitespace
        text = re.sub(r'\n\s+', '\n', text)
        # Bước 3: Loại bỏ nhiều newlines liên tiếp
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'_Trang\s+\d+/\d+_', '', text)
        return text.strip()
    def normalize_unicode(self, text) -> str:
        """
        Chuẩn hóa Unicode (NFC form)
        NFC = Canonical Decomposition, followed by Canonical Composition
        Quan trọng cho tiếng Việt để các ký tự có dấu được thống nhất
        """
        return unicodedata.normalize('NFC', text)
    def remove_markdown_withoutinformation(self, text) -> str:
        """Loại bỏ markdown không chứa thông tin"""
        return re.sub(r'^#{1,6}\s*\*{2,}\s*$', '', text, flags=re.MULTILINE)
    def remove_extra_whitespace(self, text) -> str:
        """Loại bỏ whitespace thừa"""
        return re.sub(' +', ' ', text)
    def lowercasing(self, text) -> str:
        """Chuyển thành chữ thường"""
        return text.lower()
    def remove_special_characters(self, text) -> str:
        """
        Loại bỏ ký tự đặc biệt, nhưng GIỮ LẠI các ký tự quan trọng cho Markdown:
        - [ ] (cho link/image)
        - ! (cho image)
        - ( ) (cho link/image)
        - _ (thường dùng trong tên file hoặc bold/italic)
        - * (bold/italic)
        - # (header)
        - ` (code)
        - - (list)
        
        Chỉ loại bỏ những ký tự thực sự rác hoặc không mong muốn.
        """
        # Danh sách các ký tự cần loại bỏ (đã thu hẹp lại)
        # Bỏ qua: [ ] ( ) _ * # ` - ! .
        punctuations = r"\\^{|}~" 
        
        # Hoặc nếu muốn xóa cụ thể hơn, chỉ xóa những thứ không thuộc ASCII hoặc tiếng Việt
        # Nhưng ở đây ta chỉ sửa logic cũ:
        
        pattern = re.compile(r"[%s]" % re.escape(punctuations))
        return pattern.sub("", text)
    def _convert_bold_number_headers_to_markdown(self, text: str) -> str:
        BOLD_NUMBER_HEADER = re.compile(
        r'\*\*\s*(\d+(?:\.\d+)*)\.?\s*\*\*\s*(?:\*\*(.+?)\*\*)?', re.MULTILINE)
        """
        Chuyển những tiêu đề kiểu **3.2.1.** **Tiêu đề** thành dạng Markdown:
        ví dụ: '### 3.2.1 Tiêu đề' (level = số phần)
        """
        def repl(m):
            number = m.group(1)  # e.g. "3.2.1"
            title = m.group(2) or ""  # tiêu đề nằm trong **...** ở sau (nếu có)
            # level tính theo số chấm (1-> #, 2 -> ##, 3 -> ###, cap ở 6)
            level = min(1 + number.count('.'), 6)
            hashes = "#" * level
            # tạo header: `### 3.2.1 Tiêu đề`
            if title:
                return f"{hashes} {number} {title}"
            else:
                return f"{hashes} {number}"

        return re.sub(BOLD_NUMBER_HEADER, repl, text)
    def remove_toc_until_introduction(self, text: str) -> str:
        """
        Xoá toàn bộ cho đến khi gặp 'LỜI MỞ ĐẦU'.
        """
        return re.sub(r'.*?(?=LỜI\s+MỞ\s+ĐẦU)', '', text, flags=re.DOTALL)



#test
if __name__ == "__main__":
    from step1_loader import DocumentLoader
    loader = DocumentLoader()
    file_path = "/home/congtran/RAG_demo/data/documents/tai_lieu_huong_dan_cho_tct.pdf"   
    preprocessor = TextPreprocessor()
    try:
        documents = loader.load(file_path)
        with open("/home/congtran/RAG_demo/data/documents/test1.markdown", "w", encoding="utf-8") as f:
            documents.content = preprocessor.preprocess(documents.content)
            print(f"\n--- Metadata ---")
            for key, value in documents.metadata.items():
                print(f"  {key}: {value}")
            print(f"\n--- Nội dung ---")
            print(documents.content)
            f.write(documents.content)
        f.close()
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        
