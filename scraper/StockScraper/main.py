from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from run_pipeline import scrape

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/scrape")
async def fill_data():
    scrape()