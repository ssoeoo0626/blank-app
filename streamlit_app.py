import streamlit as st
import pandas as pd
import calendar
from datetime import date
from pathlib import Path

st.set_page_config(page_title="Calendar", layout="wide")
st.title("Competitor / Market Calendar")

# -----------------------------
# csv load
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "earnings 발표일.CSV"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
    df = df.dropna(subset=["announcement_date"]).copy()
    return df

df = load_data()

# -----------------------------
# sidebar
# -----------------------------
today = date.today()

year_options = list(range(2018, today.year + 3))
default_year_index = year_options.index(today.year) if today.year in year_options else len(year_options) - 1

year = st.sidebar.selectbox("Year", year_options, index=default_year_index)
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=today.month - 1)

company_options = ["All"] + sorted(df["company"].dropna().unique().tolist())
category_options = ["All"] + sorted(df["category"].dropna().unique().tolist())

selected_company = st.sidebar.selectbox("Company", company_options)
selected_category = st.sidebar.selectbox("Category", category_options)

filtered = df.copy()
if selected_company != "All":
    filtered = filtered[filtered["company"] == selected_company]
if selected_category != "All":
    filtered = filtered[filtered["category"] == selected_category]

filtered = filtered[
    (filtered["event_date"].dt.year == year) &
    (filtered["event_date"].dt.month == month)
]

# 날짜별 이벤트 묶기
event_map = {}
for _, row in filtered.iterrows():
    d = row["event_date"].date()
    event_map.setdefault(d, []).append(row.to_dict())

# -----------------------------
# style
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
.confirmed {
    background: #2563eb;
}
.predicted {
    background: #f59e0b;
}
.planned {
    background: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# render calendar
# -----------------------------
st.subheader(f"{year}-{month:02d}")

cal = calendar.Calendar(firstweekday=0)
weeks = cal.monthdatescalendar(year, month)

weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

html = '<div class="calendar-wrap">'
html += '<div class="calendar-grid">'

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
            status = str(event["status"]).lower()
            title = event["title"]
            company = event["company"]
            html += f'<div class="event-box {status}">{company}<br>{title}</div>'

        if len(day_events) > 3:
            html += f'<div style="font-size:12px;color:#4b5563;">+{len(day_events)-3} more</div>'

        html += '</div>'

html += '</div></div>'

st.markdown(html, unsafe_allow_html=True)

# -----------------------------
# detail table
# -----------------------------
st.divider()
st.subheader("Event List")

if filtered.empty:
    st.info("No events for this month.")
else:
    show = filtered.sort_values("event_date").rename(columns={
        "event_date": "Date",
        "company": "Company",
        "category": "Category",
        "title": "Title",
        "status": "Status"
    })
    st.dataframe(show[["Date", "Company", "Category", "Title", "Status"]], use_container_width=True, hide_index=True)
