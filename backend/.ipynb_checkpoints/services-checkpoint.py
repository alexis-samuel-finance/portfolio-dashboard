import yfinance as yf
import pandas as pd
import numpy as np

def safe_download(tickers, start):
    return yf.download(
        tickers,
        start=start,
        progress=False,
        threads=False
    )

def compute_portfolio(tickers, quantities=None, start_date="2024-01-01"):

    raw = safe_download(tickers, start_date)["Close"]
    data = raw.dropna(axis=1, how="all")

    valid = data.columns.tolist()

    if quantities is not None:
        df = pd.DataFrame({"ticker": tickers, "qty": quantities})
        df = df[df["ticker"].isin(valid)]
        tickers = df["ticker"].tolist()
        quantities = df["qty"].values
    else:
        quantities = np.ones(len(valid))
        tickers = valid

    fx = safe_download("CAD=X", start_date)["Close"].squeeze()
    fx = fx.reindex(data.index).ffill()
    fx_rate = float(fx.iloc[-1])

    latest = data.iloc[-1]

    values = []
    final_tickers = []

    for i, t in enumerate(tickers):
        if t not in latest:
            continue

        price = latest[t]
        if not t.endswith(".TO"):
            price *= fx_rate

        values.append(price * quantities[i])
        final_tickers.append(t)

    values = np.array(values)
    weights = values / values.sum()
   
    # RETURNS
    data_cad = data.copy()
    for c in data.columns:
        if not c.endswith(".TO"):
            data_cad[c] = data[c] * fx

    returns = data_cad.pct_change().dropna()
    portfolio_returns = returns.dot(weights)

    cum = (1 + portfolio_returns).cumprod()

    return {
        "tickers": final_tickers,
        "weights": weights.tolist(),
        "value_cad": float(values.sum()),
        "fx_rate": fx_rate,
        "dates": cum.index.astype(str).tolist(),
        "portfolio_curve": cum.tolist()
    }   