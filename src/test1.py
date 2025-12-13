import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('AITeamVN/Vietnamese_Reranker',usefast = False)
model = AutoModelForSequenceClassification.from_pretrained('AITeamVN/Vietnamese_Reranker')
model.eval()
MAX_LENGTH = 2304
pairs = [["Làm thế nào để đổi mật khẩu?","""Content: B1: Truy cập màn hình giao diện trang chủ của EasyPos thực hiện Click biểu tượng
Admin—> sau đó Click <Đổi mật khẩu>  

B2: Thực hiện điền mật khẩu cũ và xác nhận mật khẩu mới có thể Click chọn đăng
xuất khỏi các thiết bị và nhấn <Lưu>"""]]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=MAX_LENGTH)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
