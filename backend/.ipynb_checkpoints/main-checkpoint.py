from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

from services import compute_portfolio

app = FastAPI()

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running"}

# =========================
# JSON INPUT
# =========================
@app.post("/api/portfolio")
async def portfolio(data: dict):
    return compute_portfolio(
        data["tickers"],
        data.get("quantities"),
        data.get("start_date", "2024-01-01")
    )

# =========================
# CSV UPLOAD
# =========================
@app.post("/api/upload")
async def upload(file: UploadFile):

    import pandas as pd
    import io

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    df.columns = df.columns.str.lower()

    tcol = next(c for c in df.columns if "ticker" in c)
    qcol = next(c for c in df.columns if "quantity" in c)

    tickers = df[tcol].str.upper().tolist()
    quantities = df[qcol].astype(float).tolist()

    return compute_portfolio(tickers, quantities)    