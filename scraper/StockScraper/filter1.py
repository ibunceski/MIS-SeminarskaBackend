import re
import requests
from bs4 import BeautifulSoup


class IssuerFilter:

    # @staticmethod
    # def get_all_issuers():
    #     url = "https://www.mse.mk/mk/stats/symbolhistory/ALKB"
    #     resp = requests.get(url)
    #     soup = BeautifulSoup(resp.text, "html.parser")
    #     options = soup.select("select > option")
    #
    #     return [opt.text for opt in options if not re.search(r'\d', opt.text)]

    @staticmethod
    def get_all_issuers():
        resp = requests.get("https://www.mse.mk/en/stats/current-schedule")
        soup = BeautifulSoup(resp.text, "html.parser")
        codes = soup.select("tr > td > a")
        return sorted([code.text for code in codes if not re.search(r'\d', code.text)])
