import requests
from bs4 import BeautifulSoup
import json

def crawler(soup):
    #print(soup)        >> 작동 확인 완료
    div = soup.find("div", class_ = "list_body")

    result = []
    for a in div.find_all("a"):      #find_all의 반환값 형태는 리스트
        result.append(a.get_text())
    return result

def json_maker(result):
    with open("News_crawler/newsFlashTitle.json", "w", encoding = "utf-8") as f:
        json.dump(result, f, ensure_ascii = False, indent = '\t')

def main():
    url = "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=001&sid2=140&oid=001&isYeonhapFlash=Y"

    #네이버 연합뉴스 속보 제목타이틀 가져오기
    custom_header = {
        'referer' : 'https://www.naver.com/',
        'user-agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'
    }

    url = "https://news.naver.com/main/list.nhn?mode=LPOD&mid=sec&sid1=001&sid2=140&oid=001&isYeonhapFlash=Y"
    req = requests.get(url, headers = custom_header)
    #print(req.status_code)

    soup = BeautifulSoup(req.text, "html.parser")
    result = crawler(soup)
    print(result)
    json_maker(result)
    
if __name__ == "__main__":
    main()