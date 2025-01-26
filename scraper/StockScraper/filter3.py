from datetime import date
import pandas as pd


class DataFetcher:
    def __init__(self, scraper, storage):
        self.scraper = scraper
        self.storage = storage

    def fetch_missing_data(self, issuer, last_date):
        today = date.today()
        # today = date.today() - timedelta(days=10)
        if last_date >= today:
            return pd.DataFrame()

        data = self.scraper.scrape_issuer_data(issuer, last_date)
        self.storage.update_issuer(issuer, today)

        if data:
            new_df = pd.DataFrame(data)
            new_df['datetime_object'] = pd.to_datetime(new_df['Date'], format="%d.%m.%Y")

            return new_df
        else:
            return pd.DataFrame()
