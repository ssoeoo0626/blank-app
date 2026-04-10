import streamlit as st
import pandas as pd
import calendar
from datetime import date
from pathlib import Path

st.set_page_config(page_title="Earnings Calendar", layout="wide")
st.title("Competitor / Market Calendar")

# -----------------------------
# CSV load
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "earnings_calendar.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    # 컬럼명 공백/BOM 정리
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

    # 날짜형 변환
    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")

    # 날짜 없는 행 제거
    df = df.dropna(subset=["announcement_date"]).copy()

    return df

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
today = date.today()

year_options = list(range(2018, today.year + 3))
default_year_index = year_options.index(today.year) if today.year in year_options else len(year_options) - 1

year = st.sidebar.selectbox("Year", year_options, index=default_year_index)
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=today.month - 1)

company_options = ["All"] + sorted(df["company"].dropna().unique().tolist())
status_options = ["All"] + sorted(df["status"].dropna().unique().tolist())
source_options = ["All"] + sorted(df["source"].dropna().unique().tolist())

selected_company = st.sidebar.selectbox("Company", company_options)
selected_status = st.sidebar.selectbox("Status", status_options)
selected_source = st.sidebar.selectbox("Source", source_options)

filtered = df.copy()

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

            company = event.get("company", "")
            fiscal_period = event.get("fiscal_period", "")
            source = event.get("source", "")

            html += f'''
            <div class="event-box {status}">
                <div><strong>{company}</strong></div>
                <div>{fiscal_period}</div>
                <div class="event-meta">{source}</div>
            </div>
            '''

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

    st.dataframe(
        show[["announcement_date", "company", "fiscal_period", "status", "source"]],
        use_container_width=True,
        hide_index=True
    )
