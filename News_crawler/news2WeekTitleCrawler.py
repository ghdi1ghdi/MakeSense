import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json

def crawler(soup):
    result = []
    for h2 in soup.find_all("h2", class_="tit"):
        result.append(h2.get_text())
    return result

def json_maker(result):
    with open("News_crawler/newsTitle.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent='\t')

def main():
    custom_header = {
        'referer': 'https://www.nate.com/',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'
    }

    base_url = "https://news.nate.com/recent?mid=n0100&type=c"
    
    today = datetime.today()
    
    all_results = []
    
    for i in range(14):  # 14일간의 데이터 수집
        date_str = today.strftime('%Y%m%d')
        base_date_url = base_url + "&date=" + date_str
        for i in range(19):
            url = base_date_url + "&page=" + str(i+1)
            print(url)
            req = requests.get(url, headers=custom_header)
            soup = BeautifulSoup(req.text, "html.parser")
            result = crawler(soup)
            all_results.extend(result)
            print(result)
        
        # 하루씩 이전 날짜로 이동
        today -= timedelta(days=1)
        
    json_maker(all_results)

if __name__ == "__main__":
    main()