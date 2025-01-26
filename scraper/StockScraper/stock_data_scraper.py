import requests
from datetime import date, timedelta
from bs4 import BeautifulSoup

from data_storage import DataStorage


class StockDataScraper:
    COLUMN_NAMES = [
        "Date", "Last trade price", "Max", "Min", "Avg. Price",
        "%chg.", "Volume", "Turnover in BEST in denars", "Total turnover in denars"
    ]

    def __init__(self):
        self.storage = DataStorage()

    @staticmethod
    def _format_date(d):
        return d.strftime("%d.%m.%Y")

    def _scrape_table(self, soup, issuer):
        res = []
        table = soup.select_one('#resultsTable > tbody')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                tmp = {}
                br = False
                first = True
                for td, col in zip(row.find_all('td'), self.COLUMN_NAMES):

                    if (not first) and col == 'Max' and td.text == "":
                        br = True
                        break

                    first = False
                    tmp[col] = td.text

                if br:
                    continue

                tmp['Issuer'] = issuer

                res.append(tmp)

        return res

    def scrape_issuer_data(self, issuer, start_date):
        url = f"https://www.mse.mk/mk/stats/symbolhistory/{issuer}"
        result = []
        today = date.today()
        # today = date.today() - timedelta(days=10)

        current_date = start_date
        while current_date < today:
            end_date = min(current_date + timedelta(days=364), today)

            params = {
                "FromDate": self._format_date(current_date),
                "ToDate": self._format_date(end_date),
            }
            try:
                response = requests.get(url, params=params)
            except Exception:
                print("Connection error, mse.mk not responding")
                return {}

            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            result.extend(self._scrape_table(soup, issuer))

            current_date = end_date + timedelta(days=1)

        print(f"Collected {len(result)} rows for {issuer}") if len(result) != 0 else print(
            f"No data collected for {issuer}")

        return result
