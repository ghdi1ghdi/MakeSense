from konlpy.tag import Okt
okt = Okt()
text = "아 나는 역시 밤에 일이 잘된다."

print(okt.morphs(text, stem=True))