import streamlit as st
import pandas as pd
import calendar
import re
import html as html_lib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import date, datetime
from pathlib import Path

st.set_page_config(page_title="Earnings & IR News Calendar", layout="wide")
st.title("Competitor / Market Calendar")

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "earnings 발표일.CSV"

PREDICT_START_YEAR = 2026
PREDICT_END_YEAR = 2028


# =============================
# 1) RAW CSV LOAD
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    required_cols = ["company", "fiscal_period", "announcement_date", "status", "source"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"CSV에 필요한 컬럼이 없습니다: {missing_cols}")
        st.stop()

    df["company"] = df["company"].astype(str).str.strip()
    df["fiscal_period"] = df["fiscal_period"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")

    df = df.dropna(subset=["announcement_date"]).copy()
    return df


# =============================
# 2) PREDICTION LOGIC
# =============================
def normalize_status(value: str) -> str:
    value = str(value).strip().lower()
    if value in ["past", "confirmed", "actual"]:
        return "actual"
    if value in ["predicted", "prediction"]:
        return "predicted"
    if value in ["planned"]:
        return "planned"
    return value


def extract_quarter(fiscal_period: str):
    text = str(fiscal_period).upper().strip()

    patterns = [
        r"\bQ([1-4])\b",
        r"\b([1-4])Q\b",
        r"\b([1-4])\s*QUARTER\b",
        r"\bQUARTER\s*([1-4])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"Q{match.group(1)}"

    if "FY" in text or "FULL YEAR" in text:
        return "Q4"

    return None


def extract_fiscal_year(fiscal_period: str, announcement_date: pd.Timestamp):
    text = str(fiscal_period).upper().strip()

    match = re.search(r"(20\d{2})", text)
    if match:
        return int(match.group(1))

    quarter = extract_quarter(text)
    if quarter is None:
        return announcement_date.year

    ann_year = announcement_date.year
    ann_month = announcement_date.month

    if quarter == "Q4":
        if ann_month in [1, 2, 3]:
            return ann_year - 1
        return ann_year

    return ann_year


def get_period_end(fiscal_year: int, quarter: str):
    if quarter == "Q1":
        return pd.Timestamp(year=fiscal_year, month=3, day=31)
    if quarter == "Q2":
        return pd.Timestamp(year=fiscal_year, month=6, day=30)
    if quarter == "Q3":
        return pd.Timestamp(year=fiscal_year, month=9, day=30)
    if quarter == "Q4":
        return pd.Timestamp(year=fiscal_year, month=12, day=31)
    return None


def adjust_to_business_day(ts: pd.Timestamp):
    if ts.weekday() == 5:
        return ts - pd.Timedelta(days=1)
    if ts.weekday() == 6:
        return ts + pd.Timedelta(days=1)
    return ts


def weighted_average(values, weights):
    if not values:
        return None
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def prepare_base_dataframe(df):
    work = df.copy()
    work["status_norm"] = work["status"].apply(normalize_status)
    work["quarter"] = work["fiscal_period"].apply(extract_quarter)
    work["fiscal_year"] = work.apply(
        lambda row: extract_fiscal_year(row["fiscal_period"], row["announcement_date"]),
        axis=1
    )
    work["period_end"] = work.apply(
        lambda row: get_period_end(row["fiscal_year"], row["quarter"]),
        axis=1
    )

    work = work.dropna(subset=["quarter", "period_end"]).copy()
    work["lag_days"] = (work["announcement_date"] - work["period_end"]).dt.days
    return work


def get_actual_history(df_base, company, quarter, before_fiscal_year):
    hist = df_base[
        (df_base["company"] == company) &
        (df_base["quarter"] == quarter) &
        (df_base["fiscal_year"] < before_fiscal_year) &
        (df_base["status_norm"] == "actual")
    ].copy()
    return hist.sort_values("fiscal_year")


def get_peer_history(df_base, company, quarter, before_fiscal_year):
    peer = df_base[
        (df_base["company"] != company) &
        (df_base["quarter"] == quarter) &
        (df_base["fiscal_year"] < before_fiscal_year) &
        (df_base["status_norm"] == "actual")
    ].copy()
    return peer.sort_values("fiscal_year")


def default_lag_by_quarter(quarter):
    defaults = {"Q1": 40, "Q2": 40, "Q3": 40, "Q4": 52}
    return defaults.get(quarter, 40)


def predict_lag(df_base, company, quarter, target_fiscal_year):
    actual_hist = get_actual_history(df_base, company, quarter, target_fiscal_year)
    peer_hist = get_peer_history(df_base, company, quarter, target_fiscal_year)

    actual_lags = actual_hist["lag_days"].dropna().tolist()
    peer_lags = peer_hist["lag_days"].dropna().tolist()

    if len(actual_lags) >= 3:
        recent = actual_lags[-3:]
        return round(weighted_average(recent, [0.2, 0.3, 0.5])), "High", "last_3_actuals_weighted"

    if len(actual_lags) == 2:
        recent = actual_lags[-2:]
        return round(weighted_average(recent, [0.4, 0.6])), "Mid", "last_2_actuals_weighted"

    if len(actual_lags) == 1:
        company_lag = actual_lags[-1]
        if len(peer_lags) > 0:
            peer_avg = round(sum(peer_lags) / len(peer_lags))
            return round(company_lag * 0.7 + peer_avg * 0.3), "Low", "1_actual_plus_peer_avg"
        return company_lag, "Low", "1_actual_only"

    if len(peer_lags) > 0:
        return round(sum(peer_lags) / len(peer_lags)), "Low", "peer_avg_only"

    return default_lag_by_quarter(quarter), "Low", "default_quarter_lag"


def format_predicted_fiscal_period(fiscal_year, quarter):
    return f"{fiscal_year} {quarter}"


def generate_predictions(df_raw, start_year=PREDICT_START_YEAR, end_year=PREDICT_END_YEAR):
    df_base = prepare_base_dataframe(df_raw)

    companies = sorted(df_base["company"].dropna().unique().tolist())
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    existing_keys = set(zip(df_base["company"], df_base["quarter"], df_base["fiscal_year"]))
    prediction_rows = []
    working_base = df_base.copy()

    for fiscal_year in range(start_year, end_year + 1):
        new_rows_for_year = []

        for company in companies:
            for quarter in quarters:
                key = (company, quarter, fiscal_year)

                if key in existing_keys:
                    continue

                predicted_lag, confidence, basis = predict_lag(
                    working_base, company, quarter, fiscal_year
                )

                period_end = get_period_end(fiscal_year, quarter)
                predicted_date = adjust_to_business_day(
                    period_end + pd.Timedelta(days=int(predicted_lag))
                )

                new_rows_for_year.append({
                    "company": company,
                    "fiscal_period": format_predicted_fiscal_period(fiscal_year, quarter),
                    "announcement_date": predicted_date,
                    "status": "predicted",
                    "source": f"model:{basis}",
                    "prediction_confidence": confidence,
                    "prediction_basis": basis,
                    "quarter": quarter,
                    "fiscal_year": fiscal_year,
                    "period_end": period_end,
                    "lag_days": int(predicted_lag),
                    "status_norm": "predicted",
                })

                existing_keys.add(key)

        if new_rows_for_year:
            year_df = pd.DataFrame(new_rows_for_year)
            prediction_rows.append(year_df)
            working_base = pd.concat([working_base, year_df], ignore_index=True)

    if prediction_rows:
        return pd.concat(prediction_rows, ignore_index=True)

    return pd.DataFrame(columns=[
        "company", "fiscal_period", "announcement_date", "status", "source",
        "prediction_confidence", "prediction_basis"
    ])


# =============================
# 3) NEWSROOM CRAWLING
# =============================
NEWSROOM_CONFIG = {
    "IMAX": {
        "url": "https://investors.imax.com/news-events/news",
        "item_selector": "li, .module_item, .news-release, .press-release-item",
        "title_selector": "a",
        "date_selector": ".module_date-text, .date, time",
    },
    "Cinemark": {
        "url": "https://ir.cinemark.com/news-events/press-releases",
        "item_selector": "li, .module_item, .news-release, .press-release-item",
        "title_selector": "a",
        "date_selector": ".module_date-text, .date, time",
    },
    "Cineplex": {
        "url": "https://www.newswire.ca/news-releases/cineplex-inc-latest-news/",
        "item_selector": "article, li, .news-item, .press-release-item",
        "title_selector": "a",
        "date_selector": ".date, time, .press-release-date",
    },
    "Netflix": {
        "url": "https://about.netflix.com/ko/newsroom",
        "item_selector": "article, li, .news-item, .press-release-item",
        "title_selector": "a",
        "date_selector": ".date, time, .press-release-date",   
    },
    "Netflix": {
        "url": "https://about.netflix.com/ko/newsroom",
        "item_selector": "article, li, .news-item, .press-release-item",
        "title_selector": "a",
        "date_selector": ".date, time, .press-release-date",   
    },

def parse_news_date(text):
    if not text:
        return None

    text = str(text).strip()

    date_formats = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d %B %Y",
        "%d %b %Y",
    ]

    for fmt in date_formats:
        try:
            return pd.to_datetime(datetime.strptime(text, fmt)).normalize()
        except Exception:
            pass

    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.normalize()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def crawl_company_news(company, config, max_items=10):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(config["url"], headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        items = soup.select(config["item_selector"])

        rows = []
        seen = set()

        for item in items:
            title_node = item.select_one(config["title_selector"])
            if not title_node:
                continue

            title = title_node.get_text(" ", strip=True)
            href = title_node.get("href", "")
            link = urljoin(config["url"], href) if href else ""

            date_text = ""
            date_node = item.select_one(config["date_selector"])
            if date_node:
                date_text = date_node.get_text(" ", strip=True)
            elif item.find("time"):
                date_text = item.find("time").get_text(" ", strip=True)

            news_date = parse_news_date(date_text)

            if not title or title in seen:
                continue

            seen.add(title)

            rows.append({
                "company": company,
                "fiscal_period": title,
                "announcement_date": news_date,
                "status": "news",
                "source": "IR Newsroom",
                "news_link": link,
                "news_title": title,
            })

            if len(rows) >= max_items:
                break

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.dropna(subset=["announcement_date"]).copy()

        return df

    except Exception:
        return pd.DataFrame(columns=[
            "company", "fiscal_period", "announcement_date",
            "status", "source", "news_link", "news_title"
        ])


def load_all_newsroom_data():
    news_frames = []

    for company, config in NEWSROOM_CONFIG.items():
        news_df = crawl_company_news(company, config, max_items=10)
        if not news_df.empty:
            news_frames.append(news_df)

    if news_frames:
        return pd.concat(news_frames, ignore_index=True)

    return pd.DataFrame(columns=[
        "company", "fiscal_period", "announcement_date",
        "status", "source", "news_link", "news_title"
    ])


# =============================
# 4) BUILD DISPLAY DATA
# =============================
df_raw = load_data()
predicted_df = generate_predictions(df_raw, PREDICT_START_YEAR, PREDICT_END_YEAR)
news_df = load_all_newsroom_data()

frames = [df_raw.copy()]

if not predicted_df.empty:
    frames.append(
        predicted_df[[
            "company", "fiscal_period", "announcement_date", "status", "source",
            "prediction_confidence", "prediction_basis"
        ]].copy()
    )

if not news_df.empty:
    frames.append(
        news_df[[
            "company", "fiscal_period", "announcement_date", "status", "source",
            "news_link", "news_title"
        ]].copy()
    )

display_df = pd.concat(frames, ignore_index=True)
display_df["announcement_date"] = pd.to_datetime(display_df["announcement_date"], errors="coerce")


# =============================
# 5) SIDEBAR
# =============================
today = date.today()

year_options = list(range(2018, PREDICT_END_YEAR + 1))
default_year_index = year_options.index(today.year) if today.year in year_options else len(year_options) - 1

year = st.sidebar.selectbox("Year", year_options, index=default_year_index)
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=today.month - 1)

company_options = ["All"] + sorted(display_df["company"].dropna().unique().tolist())
status_options = ["All"] + sorted(display_df["status"].dropna().unique().tolist())
source_options = ["All"] + sorted(display_df["source"].dropna().unique().tolist())

selected_company = st.sidebar.selectbox("Company", company_options)
selected_status = st.sidebar.selectbox("Status", status_options)
selected_source = st.sidebar.selectbox("Source", source_options)

filtered = display_df.copy()

if selected_company != "All":
    filtered = filtered[filtered["company"] == selected_company]
if selected_status != "All":
    filtered = filtered[filtered["status"] == selected_status]
if selected_source != "All":
    filtered = filtered[filtered["source"] == selected_source]

filtered = filtered[
    (filtered["announcement_date"].dt.year == year) &
    (filtered["announcement_date"].dt.month == month)
].copy()


# =============================
# 6) EVENT MAP
# =============================
event_map = {}
for _, row in filtered.iterrows():
    d = row["announcement_date"].date()
    event_map.setdefault(d, []).append(row.to_dict())


# =============================
# 7) STYLE
# =============================
st.markdown("""
<style>
.calendar-wrap {
    width: 100%;
}
.calendar-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 6px;
}
.calendar-header {
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
    font-weight: 700;
}
.calendar-cell {
    min-height: 150px;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 8px;
    background: white;
}
.calendar-cell.other-month {
    background: #f9fafb;
    color: #9ca3af;
}
.day-number {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}
.event-box {
    font-size: 12px;
    line-height: 1.3;
    padding: 6px 8px;
    border-radius: 8px;
    margin-bottom: 6px;
    color: white;
    overflow: hidden;
}
.past {
    background: #2563eb;
}
.confirmed {
    background: #059669;
}
.predicted {
    background: #f59e0b;
}
.planned {
    background: #6b7280;
}
.news {
    background: #dc2626;
}
.unknown {
    background: #7c3aed;
}
.event-meta {
    font-size: 11px;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)


# =============================
# 8) CALENDAR RENDER
# =============================
st.subheader(f"{year}-{month:02d}")

cal = calendar.Calendar(firstweekday=0)
weeks = cal.monthdatescalendar(year, month)
weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

calendar_html = '<div class="calendar-wrap"><div class="calendar-grid">'

for wd in weekdays:
    calendar_html += f'<div class="calendar-header">{wd}</div>'

for week in weeks:
    for day in week:
        classes = "calendar-cell"
        if day.month != month:
            classes += " other-month"

        calendar_html += f'<div class="{classes}">'
        calendar_html += f'<div class="day-number">{day.day}</div>'

        day_events = event_map.get(day, [])

        for event in day_events[:3]:
            status = str(event.get("status", "unknown")).strip().lower()
            if status not in ["past", "confirmed", "predicted", "planned", "news"]:
                status = "unknown"

            company = html_lib.escape(str(event.get("company", "")))
            source = html_lib.escape(str(event.get("source", "")))
            confidence = html_lib.escape(str(event.get("prediction_confidence", "")))

            if status == "news":
                main_text = html_lib.escape(str(event.get("news_title", event.get("fiscal_period", ""))))
                meta_text = source
            else:
                main_text = html_lib.escape(str(event.get("fiscal_period", "")))
                meta_text = f"{source} ({confidence})" if confidence else source

            calendar_html += (
                f'<div class="event-box {status}">'
                f'<div><strong>{company}</strong></div>'
                f'<div>{main_text}</div>'
                f'<div class="event-meta">{meta_text}</div>'
                f'</div>'
            )

        if len(day_events) > 3:
            calendar_html += f'<div style="font-size:12px;color:#4b5563;">+{len(day_events)-3} more</div>'

        calendar_html += '</div>'

calendar_html += '</div></div>'

st.markdown(calendar_html, unsafe_allow_html=True)


# =============================
# 9) EVENT LIST
# =============================
st.divider()
st.subheader("Event List")

if filtered.empty:
    st.info("No events for this month.")
else:
    show = filtered.sort_values("announcement_date").copy()
    show["announcement_date"] = show["announcement_date"].dt.strftime("%Y-%m-%d")

    cols = ["announcement_date", "company", "fiscal_period", "status", "source"]
    if "prediction_confidence" in show.columns:
        cols.append("prediction_confidence")
    if "prediction_basis" in show.columns:
        cols.append("prediction_basis")
    if "news_link" in show.columns:
        cols.append("news_link")

    st.dataframe(
        show[cols],
        use_container_width=True,
        hide_index=True
    )


# =============================
# 10) PREDICTION SUMMARY
# =============================
st.divider()
st.subheader("Prediction Summary")

if predicted_df.empty:
    st.info("No predicted rows were generated.")
else:
    pred_show = predicted_df.copy().sort_values(["announcement_date", "company"])
    pred_show["announcement_date"] = pred_show["announcement_date"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        pred_show[[
            "announcement_date", "company", "fiscal_period",
            "status", "prediction_confidence", "prediction_basis", "source"
        ]],
        use_container_width=True,
        hide_index=True
    )


# =============================
# 11) NEWS SUMMARY
# =============================
st.divider()
st.subheader("Latest IR Newsroom Articles")

if news_df.empty:
    st.info("No newsroom articles found.")
else:
    news_show = news_df.copy().sort_values("announcement_date", ascending=False)
    news_show["announcement_date"] = news_show["announcement_date"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        news_show[["announcement_date", "company", "news_title", "news_link"]],
        use_container_width=True,
        hide_index=True
    )
