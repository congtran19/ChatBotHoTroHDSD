import re

text = "![image](data/images/tai_lieu_huong_dan_cho_tct/page29_img0.jpeg)"
pattern = re.compile(r'\[.*?\]\((.*?)\)')
matches = pattern.findall(text)
print(matches)