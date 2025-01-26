from http.client import HTTPException

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from nlp import NLPProcessor
from technical_analysis import TechnicalAnalyzer
from lstm import LSTMAnalyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nlp_processor = NLPProcessor()
    app.state.technical_analyzer = TechnicalAnalyzer()
    app.state.lstm_analyzer = LSTMAnalyzer()
    yield
    del app.state.nlp_processor
    del app.state.technical_analyzer
    del app.state.lstm_analyzer


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/nlp/{issuer}")
async def analyze_news(issuer: str):
    try:
        nlp_processor = app.state.nlp_processor
        signals = nlp_processor.analyze_texts(issuer)
        return signals
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/technical/{issuer}")
async def analyze_technical(issuer: str):
    try:
        technical_analyzer = app.state.technical_analyzer
        results = technical_analyzer.analyze_stock(issuer)
        return results
    except HTTPException as he:
        raise he
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/lstm/{issuer}")
async def analyze_lstm(issuer: str):
    try:
        lstm_analyzer = app.state.lstm_analyzer
        predictions = lstm_analyzer.perform_prediction(issuer)
        return predictions
    except HTTPException as he:
        raise he
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8005)


