from abc import ABC, abstractmethod
from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class ExtractEntity(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[str]:
        pass

class ExtractEntityHuggingFace(ExtractEntity):
    def __init__(self,tokenizer_name,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
    def merge_ner_tokens(self,ner_tokens):
        entities = []
        current_entity = None

        for tok in ner_tokens:
            tag = tok["entity"]
            word = tok["word"]

            if tag.startswith("B-"):
                # nếu đang build entity cũ thì push vào list
                if current_entity:
                    entities.append(current_entity)

                entity_type = tag[2:]  # lấy PERSON, ORG…
                current_entity = {"type": entity_type, "text": word}

            elif tag.startswith("I-") and current_entity:
                # nối vào entity trước đó
                current_entity["text"] += " " + word

            else:
                # không khớp, đóng entity hiện tại
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # push entity cuối cùng
        if current_entity:
            entities.append(current_entity)

        return entities
    def ner_to_metadata(self,entities):
        meta = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "misc": []
        }

        for e in entities:
            t = e["type"]
            if t == "PERSON":
                meta["persons"].append(e["text"])
            elif t == "ORGANIZATION":
                meta["organizations"].append(e["text"])
            elif t == "LOCATION":
                meta["locations"].append(e["text"])
            else:
                meta["misc"].append(e["text"])
        return meta

    def extract(self, text: str) -> dict:
        ner_results = self.nlp(text)
        ner_results = self.merge_ner_tokens(ner_results)
        ner_results = self.ner_to_metadata(ner_results)
        return ner_results

if __name__ == "__main__":
    tokenizer_name = "NlpHUST/ner-vietnamese-electra-base"
    model_name = "NlpHUST/ner-vietnamese-electra-base"
    extractor = ExtractEntityHuggingFace(tokenizer_name, model_name)
    text = "Đại tá Nguyễn Văn Tảo, Phó Giám đốc Công an tỉnh Tiền Giang vừa có cuộc họp cùng Chỉ huy Công an huyện Châu Thành"
    ner_results = extractor.extract(text)
    print(ner_results)
