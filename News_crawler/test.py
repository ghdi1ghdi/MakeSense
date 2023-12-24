from konlpy.tag import Okt
okt = Okt()
text = "아 나는 역시 밤에 일이 잘된다."

import json

# JSON 파일 경로
file_path = '/Users/jack/Documents/GitHub/MakeSense/News_crawler/newsTodayTitle.json'

# JSON 파일 열기
with open(file_path, 'r', encoding='utf-8') as file:
    # JSON 데이터를 파이썬 딕셔너리로 로드
    data = json.load(file)
    
# 데이터 사용
print(data)


print(okt.morphs(text, stem=True))