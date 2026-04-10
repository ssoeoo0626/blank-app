from pathlib import Path
import streamlit as st
import pandas as pd
import calendar
from datetime import date

st.set_page_config(page_title="Earnings Calendar", layout="wide")
st.title("Company Earnings Calendar")

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "earnings_calendar.csv"

@st.cache_data
def load_data():
    st.write("현재 app.py 위치:", BASE_DIR)
    st.write("찾는 CSV 경로:", CSV_PATH)

    if not CSV_PATH.exists():
        st.error(f"CSV file not found: {CSV_PATH}")
        st.stop()

    df = pd.read_csv(CSV_PATH)

    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
    df = df.dropna(subset=["announcement_date"]).copy()

    return df

df = load_data()
