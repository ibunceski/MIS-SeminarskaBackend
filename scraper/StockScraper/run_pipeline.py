from pipeline import Pipeline
from stock_data_scraper import StockDataScraper
from data_storage import DataStorage
import time


def scrape():
    storage = DataStorage()
    scraper = StockDataScraper()
    pipeline = Pipeline(storage, scraper)
    pipeline.run_pipeline()
    start_time = time.time()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
