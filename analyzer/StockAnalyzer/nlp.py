from datetime import date
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from news_scraper import scrape


class NLPProcessor:
    def __init__(self):
        try:
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

            self.pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                framework="pt"
            )
        except Exception as e:
            raise

    def analyze_texts(self, issuer):
        signals = []
        texts = scrape(date.today(), issuer)
        if len(texts) == 0:
            return "No news found"
        for text in texts:
            signal = self.pipe(text)[0]
            signals.append({"signal": signal, "text": text})
        return signals
