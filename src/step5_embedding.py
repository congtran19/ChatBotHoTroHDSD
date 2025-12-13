from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from step4_chungking import Chunk
import numpy as np
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed danh sách văn bản thành vectors"""
@dataclass 
class EmbeddingResult:
    embeddings: List[float]
    idext: int
    content: str
    metadata : Dict[str, Any]


class VietnameseEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "dangvantuan/vietnamese-embedding"):
        self.model = SentenceTransformer(model_name)
    
    def _tokenize(self, texts: List[str]) -> List[List[int]]:
        return [tokenize(text) for text in texts]
    
    def embed(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        result = []
        for chunk in chunks:
            tokenized = self._tokenize([chunk.content])
            embeddings = self.model.encode(tokenized)
            result.append(EmbeddingResult(embeddings=embeddings[0], idext=chunk.index, content=chunk.content, metadata=chunk.metadata))
        return result
    def embed_query(self, query: str) -> List[float]:
        tokenized = self._tokenize([query])
        return self.model.encode(tokenized)[0]

#test 
if __name__ == "__main__":
    from step1_loader import DocumentLoader
    from step4_chungking import MarkdownChunker
    from step2_preprocessing import TextPreprocessor
    chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
    loader = DocumentLoader()
    preprocessor = TextPreprocessor()
    
    file_path = "/home/congtran/Thực Hành RAG/data/documents/tai_lieu_huong_dan_cho_tct.pdf"
    
    try:
        doc = loader.load(file_path)
        print(f"✓ Loaded document: {doc.metadata.get('file_name')}")
        print(f"  Total pages: {doc.metadata.get('total_pages')}")
        
        # Preprocess (PAGE_MARKER vẫn được giữ lại)
        doc.content = preprocessor.preprocess(doc.content)
        
        # Chunk
        chunks = chunker.chunk_document(doc)
        embedder = VietnameseEmbedder()
        embeddings = embedder.embed(chunks)
        print(len(embeddings[0].embeddings))
        print(len(embeddings[1].embeddings))
        #print(embeddings[1].embeddings)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")