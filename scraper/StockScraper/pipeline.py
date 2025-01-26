import concurrent.futures
from filter1 import IssuerFilter
from filter2 import DataDateChecker
from filter3 import DataFetcher


class Pipeline:
    def __init__(self, storage, scraper):
        self.storage = storage
        self.scraper = scraper
        self.issuer_filter = IssuerFilter()
        self.date_checker = DataDateChecker(self.storage)
        self.data_fetcher = DataFetcher(self.scraper, self.storage)

    def process_issuer(self, issuer):
        """
        Fetch missing data for a single issuer starting from the last available date in the database.
        """
        last_date = self.date_checker.get_last_data_date(issuer)
        new_data = self.data_fetcher.fetch_missing_data(issuer, last_date)
        return new_data

    def run_pipeline(self, max_workers=13):
        issuers = self.issuer_filter.get_all_issuers()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self.process_issuer, issuers)

        for result in results:
            if not result.empty:
                data_rows = [
                    (
                        row['Date'],
                        row['Issuer'],
                        row['Avg. Price'],
                        row['Last trade price'],
                        row['Max'],
                        row['Min'],
                        row['%chg.'],
                        row['Turnover in BEST in denars'],
                        row['Total turnover in denars'],
                        row['Volume'],
                    )
                    for _, row in result.iterrows()
                ]
                self.storage.save_issuer_data(data_rows)

        print("Pipeline completed successfully.")
