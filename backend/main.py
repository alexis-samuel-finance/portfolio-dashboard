from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from services import compute_portfolio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/portfolio")
async def portfolio(data: dict):
    return compute_portfolio(
        data["tickers"],
        data.get("quantities"),
        data.get("start_date","2024-01-01"),
        data.get("benchmark","SPY")
    )

@app.post("/api/upload")
async def upload(file: UploadFile):

    df = pd.read_csv(io.BytesIO(await file.read()), sep=None, engine="python")

    df.columns = df.columns.str.lower()

    t = [c for c in df.columns if "ticker" in c][0]
    q = [c for c in df.columns if "quantity" in c][0]

    df = df[[t,q]]

    df[t] = df[t].astype(str).str.upper().str.strip()
    df[q] = pd.to_numeric(df[q], errors="coerce")

    df = df.dropna()

    return compute_portfolio(df[t].tolist(), df[q].tolist())