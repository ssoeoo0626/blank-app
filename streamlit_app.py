import streamlit as st
import pandas as pd
import calendar
import re
from datetime import date, timedelta
from pathlib import Path

st.set_page_config(page_title="Earnings Calendar", layout="wide")
st.title("Competitor / Market Calendar")

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "earnings 발표일.CSV"

PREDICT_START_YEAR = 2026
PREDICT_END_YEAR = 2028


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

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

    for p in patterns:
        m = re.search(p, text)
        if m:
            return f"Q{m.group(1)}"

    if "FY" in text or "FULL YEAR" in text:
        return "Q4"

    return None


def extract_fiscal_year(fiscal_period: str, announcement_date: pd.Timestamp):
    text = str(fiscal_period).upper().strip()

    m = re.search(r"(20\d{2})", text)
    if m:
        return int(m.group(1))

    quarter = extract_quarter(text)
    if quarter is None:
        return announcement_date.year

    ann_year = announcement_date.year
    ann_month = announcement_date.month

    if quarter == "Q4":
        # 보통 Q4/FY는 다음 해 초에 발표되므로
        if ann_month in [1, 2, 3]:
            return ann_year - 1
        return ann_year

    # Q1/Q2/Q3는 보통 해당 연도 발표
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
    # 토요일이면 금요일, 일요일이면 월요일로 보정
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

    hist = hist.sort_values("fiscal_year")
    return hist


def get_peer_history(df_base, company, quarter, before_fiscal_year):
    peer = df_base[
        (df_base["company"] != company) &
        (df_base["quarter"] == quarter) &
        (df_base["fiscal_year"] < before_fiscal_year) &
        (df_base["status_norm"] == "actual")
    ].copy()

    peer = peer.sort_values("fiscal_year")
    return peer


def default_lag_by_quarter(quarter):
    defaults = {
        "Q1": 40,
        "Q2": 40,
        "Q3": 40,
        "Q4": 52,
    }
    return defaults.get(quarter, 40)


def predict_lag(df_base, company, quarter, target_fiscal_year):
    actual_hist = get_actual_history(df_base, company, quarter, target_fiscal_year)
    peer_hist = get_peer_history(df_base, company, quarter, target_fiscal_year)

    actual_lags = actual_hist["lag_days"].dropna().tolist()
    peer_lags = peer_hist["lag_days"].dropna().tolist()

    if len(actual_lags) >= 3:
        recent = actual_lags[-3:]
        predicted_lag = round(weighted_average(recent, [0.2, 0.3, 0.5]))
        confidence = "High"
        basis = "last_3_actuals_weighted"
        return predicted_lag, confidence, basis

    if len(actual_lags) == 2:
        recent = actual_lags[-2:]
        predicted_lag = round(weighted_average(recent, [0.4, 0.6]))
        confidence = "Mid"
        basis = "last_2_actuals_weighted"
        return predicted_lag, confidence, basis

    if len(actual_lags) == 1:
        company_lag = actual_lags[-1]
        if len(peer_lags) > 0:
            peer_avg = round(sum(peer_lags) / len(peer_lags))
            predicted_lag = round(company_lag * 0.7 + peer_avg * 0.3)
            confidence = "Low"
            basis = "1_actual_plus_peer_avg"
        else:
            predicted_lag = company_lag
            confidence = "Low"
            basis = "1_actual_only"
        return predicted_lag, confidence, basis

    if len(peer_lags) > 0:
        predicted_lag = round(sum(peer_lags) / len(peer_lags))
        confidence = "Low"
        basis = "peer_avg_only"
        return predicted_lag, confidence, basis

    predicted_lag = default_lag_by_quarter(quarter)
    confidence = "Low"
    basis = "default_quarter_lag"
    return predicted_lag, confidence, basis


def format_predicted_fiscal_period(fiscal_year, quarter):
    return f"{fiscal_year} {quarter}"


def generate_predictions(df_raw, start_year=PREDICT_START_YEAR, end_year=PREDICT_END_YEAR):
    df_base = prepare_base_dataframe(df_raw)

    companies = sorted(df_base["company"].dropna().unique().tolist())
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    existing_keys = set(
        zip(
            df_base["company"],
            df_base["quarter"],
            df_base["fiscal_year"]
        )
    )

    prediction_rows = []

    # 동적 갱신 위해 연도 순서대로 생성하고, 생성 결과도 다음 연도 예측에 참고 가능하게 base에 붙여나감
    working_base = df_base.copy()

    for fiscal_year in range(start_year, end_year + 1):
        new_rows_for_year = []

        for company in companies:
            for quarter in quarters:
                key = (company, quarter, fiscal_year)

                # 이미 actual/confirmed/predicted 뭐든 있으면 생성 안 함
                if key in existing_keys:
                    continue

                predicted_lag, confidence, basis = predict_lag(
                    working_base, company, quarter, fiscal_year
                )

                period_end = get_period_end(fiscal_year, quarter)
                predicted_date = period_end + pd.Timedelta(days=int(predicted_lag))
                predicted_date = adjust_to_business_day(predicted_date)

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
        predicted_df = pd.concat(prediction_rows, ignore_index=True)
    else:
        predicted_df = pd.DataFrame(columns=[
            "company", "fiscal_period", "announcement_date", "status", "source",
            "prediction_confidence", "prediction_basis"
        ])

    return predicted_df


df_raw = load_data()
predicted_df = generate_predictions(df_raw, PREDICT_START_YEAR, PREDICT_END_YEAR)

display_df = pd.concat(
    [
        df_raw.copy(),
        predicted_df[["company", "fiscal_period", "announcement_date", "status", "source",
                      "prediction_confidence", "prediction_basis"]].copy()
        if not predicted_df.empty else pd.DataFrame()
    ],
    ignore_index=True
)

display_df["announcement_date"] = pd.to_datetime(display_df["announcement_date"], errors="coerce")

# -----------------------------
# Sidebar
# -----------------------------
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

# -----------------------------
# 날짜별 이벤트 묶기
# -----------------------------
event_map = {}
for _, row in filtered.iterrows():
    d = row["announcement_date"].date()
    event_map.setdefault(d, []).append(row.to_dict())

# -----------------------------
# Style
# -----------------------------
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
    min-height: 140px;
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
.unknown {
    background: #7c3aed;
}
.event-meta {
    font-size: 11px;
    opacity: 0.9;
}
.small-note {
    font-size: 11px;
    color: #6b7280;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 달력 렌더링
# -----------------------------
st.subheader(f"{year}-{month:02d}")

cal = calendar.Calendar(firstweekday=0)
weeks = cal.monthdatescalendar(year, month)
weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

html = '<div class="calendar-wrap"><div class="calendar-grid">'

for wd in weekdays:
    html += f'<div class="calendar-header">{wd}</div>'

for week in weeks:
    for day in week:
        classes = "calendar-cell"
        if day.month != month:
            classes += " other-month"

        html += f'<div class="{classes}">'
        html += f'<div class="day-number">{day.day}</div>'

        day_events = event_map.get(day, [])

       for event in day_events[:3]:
    status = str(event.get("status", "unknown")).strip().lower()
    if status not in ["past", "confirmed", "predicted", "planned"]:
        status = "unknown"

    company = html.escape(str(event.get("company", "")))
    fiscal_period = html.escape(str(event.get("fiscal_period", "")))
    source = html.escape(str(event.get("source", "")))
    confidence = html.escape(str(event.get("prediction_confidence", "")))

    confidence_text = f" ({confidence})" if confidence else ""

    html += (
        f'<div class="event-box {status}">'
        f'<div><strong>{company}</strong></div>'
        f'<div>{fiscal_period}</div>'
        f'<div class="event-meta">{source}{confidence_text}</div>'
        f'</div>'
    )

        if len(day_events) > 3:
            html += f'<div style="font-size:12px;color:#4b5563;">+{len(day_events)-3} more</div>'

        html += '</div>'

html += '</div></div>'

st.markdown(html, unsafe_allow_html=True)

# -----------------------------
# 상세 테이블
# -----------------------------
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

    st.dataframe(
        show[cols],
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# 예측 결과 요약
# -----------------------------
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
