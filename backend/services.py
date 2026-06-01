import yfinance as yf
import pandas as pd
import numpy as np

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology",
    "JPM": "Financials", "BAC": "Financials",
    "XIU.TO": "Equity ETF", "ZAG.TO": "Bond ETF",
    "XBB.TO": "Bond ETF", "XSB.TO": "Bond ETF",
}

def safe_download(tickers, start):
    try:
        return yf.download(tickers, start=start, progress=False, threads=False)["Close"]
    except:
        return pd.DataFrame()

def get_sector(t):
    return SECTOR_MAP.get(t, "Other")

from datetime import datetime, timedelta

def compute_portfolio(tickers, quantities=None, start_date="2024-01-01", benchmark="SPY"):

    # ---------------------------
    # HANDLE PERIOD STRINGS CLEANLY
    # ---------------------------
    today = datetime.today()

    period_map = {
        "1mo": today - timedelta(days=30),
        "3mo": today - timedelta(days=90),
        "6mo": today - timedelta(days=180),
        "1y": today - timedelta(days=365),
        "5y": today - timedelta(days=365*5)
    }

    if start_date in period_map:
        start_date = period_map[start_date].strftime("%Y-%m-%d")

    data = safe_download(tickers, start_date).dropna(axis=1, how="all")
    if data.empty:
        return {"error": "No data"}

    if quantities:
        df = pd.DataFrame({"t": tickers, "q": quantities})
        df = df[df["t"].isin(data.columns)]
        tickers = df["t"].tolist()
        quantities = df["q"].values
    else:
        quantities = np.ones(len(data.columns))
        tickers = data.columns.tolist()

    fx = safe_download("CAD=X", start_date).squeeze()
    if fx is None or fx.empty:
        fx = pd.Series(1.35, index=data.index)
        fx_rate = 1.35
    else:
        fx = fx.reindex(data.index).ffill()
        fx_rate = float(fx.iloc[-1])

    latest = data.iloc[-1]

    values = []
    final = []

    for i, t in enumerate(tickers):
        if t not in latest:
            continue

        price = latest[t]
        if not t.endswith(".TO"):
            price *= fx_rate

        qty = quantities[i] if quantities[i] is not None else 0
        if quantities[i] is None:
            qty = 0
        else:
            qty = float(quantities[i])
        
        values.append(price * qty)
        final.append(t)

    values = np.array(values)

    total_value = sum(values)
    if total_value == 0:
        weights = [0]*len(values)
    else:
        weights = [v / total_value for v in values]


    data_cad = data[final].copy()
    for c in data_cad.columns:
        if not c.endswith(".TO"):
            data_cad[c] *= fx

    returns = data_cad.pct_change(fill_method=None).dropna()
    returns = returns[final]
    
    port_ret = returns.dot(weights)
    cum = (1 + port_ret).cumprod()

    bench = safe_download(benchmark, start_date).squeeze()
    if bench is not None and not bench.empty:
        bench = (1 + bench.pct_change().dropna()).cumprod()
    else:
        bench = pd.Series([1]*len(cum), index=cum.index)

    # METRICS
    ann = port_ret.mean()*252
    vol = port_ret.std()*np.sqrt(252)
    sharpe = ann/vol if vol else 0
    drawdown = (cum / cum.cummax() - 1).min()
    cagr = (cum.iloc[-1])**(252/len(cum)) - 1

    metrics = {
        "Return": ann,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown,
        "CAGR": cagr
    }

    # POSITIONS
    positions = []
    for idx, (t,w,v) in enumerate(zip(final, weights, values)):

        typ = "US Equity"
        if t.endswith(".TO"):
            typ = "CAD Equity"
        if any(x in t for x in ["ZAG","XBB","XSB"]):
            typ = "Bond"
        if "CASH" in t:
            typ = "Cash"

        positions.append({
            "ticker": t,
            "type": typ,
            "value": float(v),
            "weight": float(w),
            "quantity": float(quantities[idx]) if quantities is not None else 1.0
        })

    # SECTORS (%)
    sectors = {}
    for t,w in zip(final, weights):
        s = get_sector(t)
        sectors[s] = sectors.get(s,0) + w

    return {
        "value": float(values.sum()),
        "fx": fx_rate,
        "dates": cum.index.astype(str).tolist(),
        "portfolio": cum.tolist(),
        "benchmark": bench.reindex(cum.index, method="ffill").tolist(),
        "positions": positions,
        "metrics": metrics,
        "sectors": sectors
    }
