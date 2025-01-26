import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import json


def get_text(links, source):
    texts = []
    if source == 'mse':
        for link in links:
            try:
                r = requests.get(link)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.select_one("#content").text
                texts.append(text)
            except Exception as e:
                print(f"Error fetching MSE link {link}: {e}")
    elif source == 'seinet':
        for link in links:
            try:
                content = (json.loads(requests.get(link).text))["data"]["content"]
                formatted = content.replace("<br>", "\n").replace("<br />", "\n").replace("<br/>", "\n").replace(
                    "<p>", "").replace("</p>", "")
                if formatted:
                    texts.append(formatted)
            except Exception as e:
                print(f"Error fetching SEINET link {link}: {e}")
    return texts


def scrape(date, name):
    base_url = "https://www.mse.mk"
    r = requests.get(f"{base_url}/en/symbol/{name}")
    soup = BeautifulSoup(r.text, "html.parser")
    all_news_mse = soup.select_one("#stockEchangeNews").select(".tab-pane-text > a")
    all_news_seinet = soup.select_one("#seiNetIssuerLatestNews").select(".container-seinet > a")

    links_mse = []
    links_seinet = []

    today = date.today()
    to = today - timedelta(days=20)
    pattern = r"\d{1,2}/\d{1,2}/\d{4}"

    for news in all_news_mse:
        href = news.get("href")
        try:
            datum = datetime.strptime(re.findall(pattern, href)[0], "%d/%m/%Y").date()
            if datum > to:
                links_mse.append(f'{base_url}{href}')
        except Exception as e:
            print(f"Error parsing date in MSE news: {e}")

    for news in all_news_seinet:
        try:
            docId = news.get("href").split("/")[-1]
            link = f"https://api.seinet.com.mk/public/documents/single/{docId}"
            datum = datetime.strptime(news.text.strip().split(" ")[0], "%m/%d/%Y").date()
            if datum > to:
                links_seinet.append(link)
        except Exception as e:
            print(f"Error parsing date in SEINET news: {e}")

    return get_text(links_mse, 'mse') + get_text(links_seinet, 'seinet')
