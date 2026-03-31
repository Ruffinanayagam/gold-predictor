"""
Gold Price Predictor — Streamlit App
Pure Python, no HTML/CSS/JS needed!
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import os

st.set_page_config(page_title="Gold Price Predictor", page_icon="🥇", layout="wide")

@st.cache_resource
def load_and_train():
    # Auto-detect CSV path (works both locally and on Streamlit Cloud)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Gold_Futures_Historical_Data__4_.csv")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace('"', '')
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",","").str.replace('"',""), errors="coerce")
    df["Date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df[["Price"]].dropna()

    feat = df.copy()
    for lag in [1,2,3,5,7,14,21]:
        feat[f"lag_{lag}"] = feat["Price"].shift(lag)
    for w in [5,10,20,50]:
        feat[f"ma_{w}"]  = feat["Price"].rolling(w).mean()
        feat[f"std_{w}"] = feat["Price"].rolling(w).std()
    feat["momentum_5"]  = feat["Price"] - feat["Price"].shift(5)
    feat["momentum_10"] = feat["Price"] - feat["Price"].shift(10)
    feat["day_of_week"] = feat.index.dayofweek
    feat["month"]       = feat.index.month
    feat.dropna(inplace=True)

    FEATURES = [c for c in feat.columns if c != "Price"]
    X, y = feat[FEATURES], feat["Price"]
    split = len(feat) - 30

    model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X.iloc[:split], y.iloc[:split])

    preds = model.predict(X.iloc[split:])
    mae   = mean_absolute_error(y.iloc[split:], preds)
    mape  = (np.abs((y.iloc[split:].values - preds) / y.iloc[split:].values)).mean() * 100

    return df, feat, model, FEATURES, mae, mape


def add_features(df):
    d = df.copy()
    for lag in [1,2,3,5,7,14,21]:
        d[f"lag_{lag}"] = d["Price"].shift(lag)
    for w in [5,10,20,50]:
        d[f"ma_{w}"]  = d["Price"].rolling(w).mean()
        d[f"std_{w}"] = d["Price"].rolling(w).std()
    d["momentum_5"]  = d["Price"] - d["Price"].shift(5)
    d["momentum_10"] = d["Price"] - d["Price"].shift(10)
    d["day_of_week"] = d.index.dayofweek
    d["month"]       = d.index.month
    d.dropna(inplace=True)
    return d


def predict_next_n(df_feat, model, FEATURES, n_days=30):
    future_prices, future_dates = [], []
    temp_df   = df_feat.copy()
    last_date = temp_df.index[-1]

    for _ in range(n_days):
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        lp  = temp_df["Price"]
        row = {}
        for lag in [1,2,3,5,7,14,21]:
            row[f"lag_{lag}"] = lp.iloc[-lag] if len(lp) >= lag else lp.iloc[0]
        for w in [5,10,20,50]:
            row[f"ma_{w}"]  = lp.iloc[-w:].mean()
            row[f"std_{w}"] = lp.iloc[-w:].std()
        row["momentum_5"]  = lp.iloc[-1] - lp.iloc[-5]
        row["momentum_10"] = lp.iloc[-1] - lp.iloc[-10]
        row["day_of_week"] = next_date.weekday()
        row["month"]       = next_date.month

        pred = model.predict(pd.DataFrame([row])[FEATURES])[0]
        future_prices.append(round(float(pred), 2))
        future_dates.append(next_date)

        new_row = pd.DataFrame({"Price": [pred]}, index=[next_date])
        temp_df = pd.concat([temp_df[["Price"]], new_row])
        temp_df = add_features(temp_df)
        last_date = next_date

    return future_dates, future_prices


# ── App ────────────────────────────────────────────────────

st.title("🥇 Gold Price Predictor")
st.caption("Random Forest ML · Real Data (2010–2026) · Next 30 Days Forecast")
st.divider()

with st.spinner("🌲 Training model..."):
    RAW_DF, FEAT_DF, MODEL, FEATURES, MAE, MAPE = load_and_train()

latest_price = float(RAW_DF["Price"].iloc[-1])
prev_price   = float(RAW_DF["Price"].iloc[-2])
change       = latest_price - prev_price
change_pct   = (change / prev_price) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Latest Price",     f"${latest_price:,.2f}", f"{change_pct:+.2f}%")
col2.metric("📅 As of",            RAW_DF.index[-1].strftime("%d %b %Y"))
col3.metric("🎯 Avg Error (MAE)",  f"${MAE:.2f}")
col4.metric("📊 Accuracy",         f"{100-MAPE:.1f}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 30-Day Prediction", "📅 Date Lookup", "📈 Historical"])

# ── TAB 1: Prediction ──────────────────────────────────────
with tab1:
    st.subheader("Next 30 Business Days Forecast")

    if st.button("🔮 Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            dates, prices = predict_next_n(FEAT_DF, MODEL, FEATURES, 30)

        high    = max(prices)
        low     = min(prices)
        final   = prices[-1]
        chg_pct = (final - latest_price) / latest_price * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("📈 30-Day High",   f"${high:,.2f}")
        c2.metric("📉 30-Day Low",    f"${low:,.2f}")
        c3.metric("🏁 End of Period", f"${final:,.2f}", f"{chg_pct:+.2f}%")

        st.divider()

        hist_series = RAW_DF["Price"].iloc[-60:].rename("Historical")
        pred_series = pd.Series(
            [latest_price] + prices,
            index=[RAW_DF.index[-1]] + dates,
            name="Predicted"
        )
        chart_df = pd.concat([hist_series, pred_series], axis=1)
        st.line_chart(chart_df, use_container_width=True, height=350)

        st.divider()
        st.subheader("📋 Day-by-Day Table")

        rows = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            prev = latest_price if i == 0 else prices[i-1]
            chg  = price - prev
            pct  = (chg / prev) * 100
            rows.append({
                "Day":      f"Day {i+1}",
                "Date":     date.strftime("%a, %d %b %Y"),
                "Price":    f"${price:,.2f}",
                "Change":   f"{'▲' if chg>=0 else '▼'} ${abs(chg):.2f}",
                "Change %": f"{'▲' if pct>=0 else '▼'} {abs(pct):.2f}%",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("⚠️ ML estimates based on historical patterns only. Not financial advice.")

# ── TAB 2: Date Lookup ─────────────────────────────────────
with tab2:
    st.subheader("Gold Price on Any Date")

    selected = st.date_input(
        "Pick a date",
        value=RAW_DF.index[-1].date(),
        min_value=RAW_DF.index[0].date(),
        max_value=RAW_DF.index[-1].date()
    )

    if st.button("🔍 Search", use_container_width=True):
        sel = pd.Timestamp(selected)

        if sel in RAW_DF.index:
            price = float(RAW_DF.loc[sel, "Price"])
            idx   = RAW_DF.index.get_loc(sel)
            prev  = float(RAW_DF["Price"].iloc[idx-1]) if idx > 0 else price
            chg   = price - prev
            pct   = (chg / prev) * 100
            c1, c2 = st.columns(2)
            c1.metric(f"Gold on {sel.strftime('%d %b %Y')}", f"${price:,.2f}", f"{pct:+.2f}%")
            c2.metric("Previous Day", f"${prev:,.2f}")
        else:
            nearest = RAW_DF.index[abs(RAW_DF.index - sel).argmin()]
            price   = float(RAW_DF.loc[nearest, "Price"])
            st.warning(f"No trading data for {sel.strftime('%d %b %Y')} — showing nearest day.")
            st.metric(f"Gold on {nearest.strftime('%d %b %Y')}", f"${price:,.2f}")

        nearest = RAW_DF.index[abs(RAW_DF.index - sel).argmin()]
        idx     = RAW_DF.index.get_loc(nearest)
        nearby  = RAW_DF.iloc[max(0,idx-5):min(len(RAW_DF),idx+6)].copy()
        nearby.index = nearby.index.strftime("%d %b %Y")
        nearby.columns = ["Price (USD)"]
        nearby["Price (USD)"] = nearby["Price (USD)"].apply(lambda x: f"${x:,.2f}")
        st.divider()
        st.write("Prices around that period:")
        st.dataframe(nearby, use_container_width=True)

# ── TAB 3: Historical ──────────────────────────────────────
with tab3:
    st.subheader("Historical Gold Price")

    period = st.selectbox("Period", ["3 Months","6 Months","1 Year","3 Years","All (2010–2026)"])
    days_map = {"3 Months":90,"6 Months":180,"1 Year":365,"3 Years":1095,"All (2010–2026)":len(RAW_DF)}
    hist = RAW_DF["Price"].iloc[-days_map[period]:]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("High",  f"${hist.max():,.2f}")
    c2.metric("Low",   f"${hist.min():,.2f}")
    c3.metric("Start", f"${hist.iloc[0]:,.2f}")
    c4.metric("End",   f"${hist.iloc[-1]:,.2f}", f"{((hist.iloc[-1]-hist.iloc[0])/hist.iloc[0]*100):+.1f}%")

    st.line_chart(hist, use_container_width=True, height=380)
