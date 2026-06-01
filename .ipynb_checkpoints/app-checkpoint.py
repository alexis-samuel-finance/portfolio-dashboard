import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =========================
# SAFE DOWNLOAD
# =========================
def safe_download(tickers, start):
    try:
        return yf.download(
            tickers,
            start=start,
            progress=False,
            threads=False
        )
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Portfolio")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
manual_input = st.sidebar.text_input("Tickers (comma separated)")

period = st.sidebar.selectbox("Period", ["YTD","1Y","5Y","10Y"])

period_map = {
    "YTD":"2026-01-01",
    "1Y":"2025-01-01",
    "5Y":"2021-01-01",
    "10Y":"2016-01-01"
}
start_date = period_map[period]

# =========================
# LOAD PORTFOLIO
# =========================
use_quantities = False

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.lower()

    tcol = next(c for c in df.columns if "ticker" in c)
    qcol = next(c for c in df.columns if "quantity" in c or "shares" in c)

    df = df[[tcol, qcol]].copy()
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce")
    df = df.dropna()

    tickers = df[tcol].str.upper().tolist()
    quantities = df[qcol].values
    use_quantities = True

elif manual_input:
    tickers = [t.strip().upper() for t in manual_input.split(",")]
    weights = np.array([1/len(tickers)] * len(tickers))

else:
    tickers = ["AAPL","MSFT","GOOGL"]
    weights = np.array([0.3,0.4,0.3])

# =========================
# DOWNLOAD DATA
# =========================
raw = safe_download(tickers, start_date)["Close"]
data = raw.dropna(axis=1, how="all")

valid_tickers = data.columns.tolist()

if use_quantities:
    df_valid = pd.DataFrame({"ticker": tickers, "qty": quantities})
    df_valid = df_valid[df_valid["ticker"].isin(valid_tickers)]

    tickers = df_valid["ticker"].tolist()
    quantities = df_valid["qty"].values
else:
    tickers = valid_tickers
    weights = np.array([1/len(tickers)] * len(tickers))

missing = set(raw.columns) - set(valid_tickers)
if len(missing) > 0:
    st.warning(f"Missing data for: {list(missing)}")

# =========================
# FX
# =========================
fx = safe_download("CAD=X", start_date)["Close"].squeeze()
fx = fx.reindex(data.index).ffill()
fx_rate = fx.iloc[-1]

latest = data.iloc[-1]

# =========================
# VALUE
# =========================
values = []
final_tickers = []

for i, t in enumerate(tickers):
    if t not in latest:
        continue

    price = latest[t]
    if not t.endswith(".TO"):
        price *= fx_rate

    qty = quantities[i] if use_quantities else weights[i]

    values.append(price * qty)
    final_tickers.append(t)

values = np.array(values)
tickers = final_tickers

portfolio_value = values.sum()
weights = values / portfolio_value

# =========================
# RETURNS
# =========================
data_cad = data.copy()
for col in data.columns:
    if not col.endswith(".TO"):
        data_cad[col] = data[col] * fx

returns = data_cad.pct_change().dropna()
portfolio_returns = returns.dot(weights)
cum = (1 + portfolio_returns).cumprod()

# =========================
# CLASSIFICATION
# =========================
def classify(t):
    if t.endswith(".TO"):
        if any(x in t for x in ["ZAG","XBB","XSB"]):
            return "CAD_BOND"
        elif "CASH" in t:
            return "CASH"
        return "CAD_EQUITY"
    return "US_EQUITY"

classes = [classify(t) for t in tickers]

df_alloc = pd.DataFrame({"ticker": tickers, "class": classes, "w": weights})
class_w = df_alloc.groupby("class")["w"].sum()

# =========================
# BENCHMARK
# =========================
st.sidebar.markdown("### Benchmark")

if "bench" not in st.session_state:
    st.session_state.bench = class_w.to_dict()

bench_input = {}
for c in class_w.index:
    bench_input[c] = st.sidebar.number_input(c,0.0,1.0,float(st.session_state.bench.get(c,0)),0.01)

if abs(sum(bench_input.values()) - 1) < 0.01:
    st.session_state.bench = bench_input

bench_w = np.array(list(st.session_state.bench.values()))

# =========================
# BENCHMARK RETURNS
# =========================
map_b = {
    "US_EQUITY":"SPY",
    "CAD_EQUITY":"XIU.TO",
    "CAD_BOND":"ZAG.TO",
    "CASH":"^IRX"
}

btickers = [map_b[c] for c in st.session_state.bench.keys()]
bdata = safe_download(btickers, start_date)["Close"]
bret = bdata.pct_change().dropna()

if "^IRX" in bret.columns:
    bret["^IRX"] = bdata["^IRX"]/100/252

bret = bret[btickers]

bench_ret = bret.dot(bench_w)
bench_cum = (1 + bench_ret).cumprod()

# =========================
# DISPLAY
# =========================
st.markdown(f"### Portfolio Value: {portfolio_value:,.0f} CAD | FX {fx_rate:.3f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=cum.index, y=cum, name="Portfolio"))
fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="Benchmark"))
st.plotly_chart(fig, use_container_width=True)

# =========================
# ALLOCATION
# =========================
alloc = pd.DataFrame({
    "Class": class_w.index,
    "Portfolio": class_w.values,
    "Benchmark": [st.session_state.bench[c] for c in class_w.index]
})
alloc["Active"] = alloc["Portfolio"] - alloc["Benchmark"]

st.dataframe(
    alloc.style.format({
        "Portfolio":"{:.2%}",
        "Benchmark":"{:.2%}",
        "Active":"{:.2%}"
    })
)

# =========================
# SECTOR + CONTRIBUTION
# =========================
with st.expander("📊 Sector Analysis", expanded=False):

    sector_map = {}

    for t in tickers:
        try:
            sec = yf.Ticker(t).info.get("sector","Other")
        except:
            sec = "Other"
        sector_map[t] = sec

    sector_df = pd.DataFrame({
        "ticker": tickers,
        "sector": [sector_map[t] for t in tickers],
        "weight": weights
    })

    sector_alloc = sector_df.groupby("sector")["weight"].sum()

    contrib = {}

    for s in sector_alloc.index:
        idx = sector_df[sector_df["sector"] == s].index
        w = weights[idx]
        r = returns.iloc[:, idx]

        sector_return = r.dot(w / w.sum())
        contrib[s] = (sector_return.mean() * w.sum())

    contrib_series = pd.Series(contrib)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(go.Figure(data=[go.Pie(labels=sector_alloc.index, values=sector_alloc.values)]))

    with col2:
        st.plotly_chart(go.Figure(data=[go.Bar(x=contrib_series.index, y=contrib_series.values)]))

    st.dataframe(
        pd.DataFrame({
            "Allocation": sector_alloc,
            "Contribution": contrib_series
        }).style.format("{:.2%}")
    )