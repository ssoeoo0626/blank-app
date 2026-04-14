
import calendar
import html as html_lib
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from news_fetcher import build_query_table, fetch_news_from_query_table


st.set_page_config(page_title="Competitor / Market Search Dashboard", layout="wide")
st.title("Competitor / Market Search Dashboard")
st.caption("진단용 버전: 실제 읽은 파일 경로 / 수정시각 / 행 수를 화면에 표시합니다.")

BASE_DIR = Path(__file__).resolve().parent

EARNINGS_CSV_PATH = BASE_DIR / "earnings 발표일.CSV"
KEYWORDS_CSV_PATH = BASE_DIR / "theater_keywords_expanded_v2.csv"
SITE_POOL_CSV_PATH = BASE_DIR / "site_pool_master.csv"
COMPANY_DOMAIN_CSV_PATH = BASE_DIR / "company_domains.csv"

PREDICT_START_YEAR = 2026
PREDICT_END_YEAR = 2028


# =============================
# 0) DIAGNOSTIC HELPERS
# =============================
def file_info(path: Path):
    exists = path.exists()
    mtime = ""
    size = ""
    if exists:
        stat = path.stat()
        mtime = pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
        size = f"{stat.st_size:,} bytes"
    return {
        "file": path.name,
        "path": str(path),
        "exists": exists,
        "modified": mtime,
        "size": size,
    }


def show_file_diagnostics():
    info_rows = [
        file_info(EARNINGS_CSV_PATH),
        file_info(KEYWORDS_CSV_PATH),
        file_info(SITE_POOL_CSV_PATH),
        file_info(COMPANY_DOMAIN_CSV_PATH),
    ]
    st.subheader("파일 진단")
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)


# =============================
# 1) LOADERS
# =============================
@st.cache_data
def load_generic_csv(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(), {"exists": False, "rows": 0, "columns": []}

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    meta = {
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
    }
    return df, meta


@st.cache_data
def load_earnings_data(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["company", "fiscal_period", "announcement_date", "status", "source"]), {
            "exists": False, "rows": 0, "columns": []
        }

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    required_cols = ["company", "fiscal_period", "announcement_date", "status", "source"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        return pd.DataFrame(columns=required_cols), {
            "exists": True,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_cols": missing_cols,
        }

    for col in ["company", "fiscal_period", "status", "source"]:
        df[col] = df[col].astype(str).str.strip()

    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
    df = df.dropna(subset=["announcement_date"]).copy()

    return df, {
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_cols": [],
    }


@st.cache_data
def load_keywords(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["분류", "회사", "키워드", "키워드유형", "활성화", "비고"]), {
            "exists": False, "rows": 0, "columns": []
        }

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    expected = ["분류", "회사", "키워드", "키워드유형", "활성화", "비고"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    df["분류"] = df["분류"].astype(str).str.strip()
    df["회사"] = df["회사"].astype(str).str.strip()
    df["키워드"] = df["키워드"].astype(str).str.strip()
    df["키워드유형"] = df["키워드유형"].astype(str).str.strip()
    df["비고"] = df["비고"].astype(str).str.strip()
    df["활성화"] = pd.to_numeric(df["활성화"], errors="coerce").fillna(1).astype(int)

    raw_rows = len(df)
    active_rows = len(df[(df["활성화"] == 1) & (df["키워드"] != "")])

    df = df[(df["활성화"] == 1) & (df["키워드"] != "")].copy()

    return df, {
        "exists": True,
        "rows": len(df),
        "raw_rows": raw_rows,
        "active_rows": active_rows,
        "columns": list(df.columns),
    }


@st.cache_data
def load_site_pool(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["풀구분", "소스유형", "사이트명", "도메인", "우선순위", "권장용도"]), {
            "exists": False, "rows": 0, "columns": []
        }

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    expected = ["풀구분", "소스유형", "사이트명", "도메인", "우선순위", "권장용도"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    for col in expected:
        df[col] = df[col].astype(str).str.strip()

    df = df[df["도메인"] != ""].copy()

    return df, {
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
    }


@st.cache_data
def load_company_domains(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["회사", "도메인", "IR도메인", "IR뉴스URL", "비고"]), {
            "exists": False, "rows": 0, "columns": []
        }

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    expected = ["회사", "도메인", "IR도메인", "IR뉴스URL", "비고"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    for col in expected:
        df[col] = df[col].astype(str).str.strip()

    return df, {
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
    }


# =============================
# 2) EARNINGS PREDICTION LOGIC
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
    if df.empty:
        return df.copy()

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
    return {"Q1": 40, "Q2": 40, "Q3": 40, "Q4": 52}.get(quarter, 40)


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
    if df_raw.empty:
        return pd.DataFrame(columns=[
            "company", "fiscal_period", "announcement_date", "status", "source",
            "prediction_confidence", "prediction_basis"
        ])

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

                predicted_lag, confidence, basis = predict_lag(working_base, company, quarter, fiscal_year)
                period_end = get_period_end(fiscal_year, quarter)
                predicted_date = adjust_to_business_day(period_end + pd.Timedelta(days=int(predicted_lag)))

                new_rows_for_year.append({
                    "company": company,
                    "fiscal_period": format_predicted_fiscal_period(fiscal_year, quarter),
                    "announcement_date": predicted_date,
                    "status": "predicted",
                    "source": f"model:{basis}",
                    "prediction_confidence": confidence,
                    "prediction_basis": basis,
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
# 3) LOAD ALL
# =============================
earnings_df, earnings_meta = load_earnings_data(str(EARNINGS_CSV_PATH))
keywords_df, keywords_meta = load_keywords(str(KEYWORDS_CSV_PATH))
site_pool_df, site_meta = load_site_pool(str(SITE_POOL_CSV_PATH))
company_domain_df, company_domain_meta = load_company_domains(str(COMPANY_DOMAIN_CSV_PATH))
predicted_df = generate_predictions(earnings_df, PREDICT_START_YEAR, PREDICT_END_YEAR)

# =============================
# 4) SIDEBAR
# =============================
st.sidebar.header("제어")
if st.sidebar.button("캐시 전체 삭제 후 새로고침"):
    st.cache_data.clear()
    st.rerun()

today = date.today()
year_options = list(range(2018, PREDICT_END_YEAR + 1))
default_year_index = year_options.index(today.year) if today.year in year_options else len(year_options) - 1

selected_year = st.sidebar.selectbox("Calendar Year", year_options, index=default_year_index)
selected_month = st.sidebar.selectbox("Calendar Month", list(range(1, 13)), index=today.month - 1)

all_categories = ["All"] + sorted(keywords_df["분류"].dropna().unique().tolist()) if not keywords_df.empty else ["All"]
all_companies = ["All"] + sorted(keywords_df["회사"].dropna().unique().tolist()) if not keywords_df.empty else ["All"]

selected_categories = st.sidebar.multiselect("키워드 분류", all_categories, default=["All"])
selected_companies = st.sidebar.multiselect("회사", all_companies, default=["All"])
selected_priorities = st.sidebar.multiselect("사이트 우선순위", ["P1", "P2", "P3"], default=["P1", "P2"])
max_sites_per_keyword = st.sidebar.slider("키워드당 최대 사이트 수", 1, 8, 4)
max_queries = st.sidebar.slider("실제 뉴스 검색 쿼리 수", 10, 300, 60, step=10)
days_back = st.sidebar.slider("뉴스 검색 기간(일)", 1, 90, 30)

# =============================
# 5) TOP DIAGNOSTICS
# =============================
show_file_diagnostics()

diag_rows = [
    {"dataset": "earnings_df", "rows": len(earnings_df), "extra": str(earnings_meta)},
    {"dataset": "keywords_df", "rows": len(keywords_df), "extra": str(keywords_meta)},
    {"dataset": "site_pool_df", "rows": len(site_pool_df), "extra": str(site_meta)},
    {"dataset": "company_domain_df", "rows": len(company_domain_df), "extra": str(company_domain_meta)},
    {"dataset": "predicted_df", "rows": len(predicted_df), "extra": "generated from earnings_df"},
]
st.subheader("데이터셋 진단")
st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

# =============================
# 6) BUILD QUERY TABLE
# =============================
query_df = pd.DataFrame()
news_df = pd.DataFrame()

if not keywords_df.empty and not site_pool_df.empty:
    work_keywords = keywords_df.copy()

    if selected_categories and "All" not in selected_categories:
        work_keywords = work_keywords[work_keywords["분류"].isin(selected_categories)].copy()

    if selected_companies and "All" not in selected_companies:
        work_keywords = work_keywords[work_keywords["회사"].isin(selected_companies)].copy()

    query_df = build_query_table(
        keyword_df=work_keywords,
        site_pool_df=site_pool_df,
        company_domains_df=company_domain_df,
        selected_pool_keys=[],
        selected_priorities=selected_priorities,
        max_sites_per_keyword=max_sites_per_keyword,
    )

    if not query_df.empty:
        news_df = fetch_news_from_query_table(query_df.head(max_queries).copy(), days_back=days_back)

# =============================
# 7) TABS
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "키워드 원본 확인",
    "검색 쿼리 진단",
    "뉴스 수집 진단",
    "사이트/도메인 확인",
    "실적 캘린더",
])

with tab1:
    st.subheader("키워드 원본 확인")
    if keywords_df.empty:
        st.warning("theater_keywords_expanded_v2.csv를 읽지 못했거나 행이 0개입니다.")
    else:
        st.write(f"활성 행 수: {len(keywords_df):,}")
        st.dataframe(keywords_df.head(200), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("검색 쿼리 진단")
    if query_df.empty:
        st.warning("생성된 검색 쿼리가 없습니다.")
    else:
        st.write(f"생성된 검색 쿼리 수: {len(query_df):,}")
        show_cols = [c for c in ["분류", "회사", "키워드", "키워드유형", "사이트명", "도메인", "우선순위", "pool_key", "뉴스검색가능", "검색쿼리"] if c in query_df.columns]
        st.dataframe(query_df[show_cols], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("뉴스 수집 진단")
    if news_df.empty:
        st.warning("news_df가 비어 있습니다.")
    else:
        st.write(f"뉴스/상태 행 수: {len(news_df):,}")
        if "수집상태" in news_df.columns:
            status_count = news_df["수집상태"].value_counts(dropna=False).rename_axis("수집상태").reset_index(name="count")
            st.dataframe(status_count, use_container_width=True, hide_index=True)

        show_cols = [c for c in ["published_at", "분류", "회사", "키워드", "사이트명", "도메인", "title", "url", "source", "검색쿼리", "수집상태", "오류메시지"] if c in news_df.columns]
        st.dataframe(news_df[show_cols], use_container_width=True, hide_index=True)

with tab4:
    st.subheader("사이트/도메인 확인")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("site_pool_master.csv")
        if site_pool_df.empty:
            st.info("site_pool_master.csv가 비어 있거나 없습니다.")
        else:
            st.dataframe(site_pool_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("company_domains.csv")
        if company_domain_df.empty:
            st.info("company_domains.csv가 비어 있거나 없습니다.")
        else:
            st.dataframe(company_domain_df, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("실적 캘린더")
    frames = []
    if not earnings_df.empty:
        frames.append(earnings_df.copy())
    if not predicted_df.empty:
        frames.append(predicted_df.copy())

    display_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["company", "fiscal_period", "announcement_date", "status", "source"]
    )

    if display_df.empty:
        st.info("earnings 발표일.CSV가 없거나 비어 있습니다.")
    else:
        display_df["announcement_date"] = pd.to_datetime(display_df["announcement_date"], errors="coerce")
        filtered = display_df[
            (display_df["announcement_date"].dt.year == selected_year) &
            (display_df["announcement_date"].dt.month == selected_month)
        ].copy()

        event_map = {}
        for _, row in filtered.iterrows():
            d = row["announcement_date"].date()
            event_map.setdefault(d, []).append(row.to_dict())

        st.markdown("""
        <style>
        .calendar-wrap { width: 100%; }
        .calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; }
        .calendar-header {
            background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 10px;
            padding: 10px; text-align: center; font-weight: 700;
        }
        .calendar-cell {
            min-height: 150px; border: 1px solid #d1d5db; border-radius: 12px;
            padding: 8px; background: white;
        }
        .calendar-cell.other-month { background: #f9fafb; color: #9ca3af; }
        .day-number { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
        .event-box {
            font-size: 12px; line-height: 1.3; padding: 6px 8px;
            border-radius: 8px; margin-bottom: 6px; color: white; overflow: hidden;
        }
        .past { background: #2563eb; }
        .confirmed { background: #059669; }
        .predicted { background: #f59e0b; }
        .planned { background: #6b7280; }
        .unknown { background: #7c3aed; }
        .event-meta { font-size: 11px; opacity: 0.9; }
        </style>
        """, unsafe_allow_html=True)

        cal = calendar.Calendar(firstweekday=0)
        weeks = cal.monthdatescalendar(selected_year, selected_month)
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        calendar_html = '<div class="calendar-wrap"><div class="calendar-grid">'
        for wd in weekdays:
            calendar_html += f'<div class="calendar-header">{wd}</div>'

        for week in weeks:
            for day in week:
                classes = "calendar-cell"
                if day.month != selected_month:
                    classes += " other-month"

                calendar_html += f'<div class="{classes}">'
                calendar_html += f'<div class="day-number">{day.day}</div>'

                day_events = event_map.get(day, [])
                for event in day_events[:3]:
                    status = str(event.get("status", "unknown")).strip().lower()
                    if status not in ["past", "confirmed", "predicted", "planned"]:
                        status = "unknown"

                    company = html_lib.escape(str(event.get("company", "")))
                    source = html_lib.escape(str(event.get("source", "")))
                    confidence = html_lib.escape(str(event.get("prediction_confidence", "")))
                    fiscal_period = html_lib.escape(str(event.get("fiscal_period", "")))
                    meta_text = f"{source} ({confidence})" if confidence else source

                    calendar_html += (
                        f'<div class="event-box {status}">'
                        f'<div><strong>{company}</strong></div>'
                        f'<div>{fiscal_period}</div>'
                        f'<div class="event-meta">{meta_text}</div>'
                        f'</div>'
                    )

                if len(day_events) > 3:
                    calendar_html += f'<div style="font-size:12px;color:#4b5563;">+{len(day_events)-3} more</div>'

                calendar_html += '</div>'

        calendar_html += '</div></div>'
        st.markdown(calendar_html, unsafe_allow_html=True)

        st.markdown("월간 이벤트 목록")
        show = filtered.sort_values("announcement_date").copy()
        show["announcement_date"] = show["announcement_date"].dt.strftime("%Y-%m-%d")
        cols = ["announcement_date", "company", "fiscal_period", "status", "source"]
        if "prediction_confidence" in show.columns:
            cols.append("prediction_confidence")
        if "prediction_basis" in show.columns:
            cols.append("prediction_basis")
        st.dataframe(show[cols], use_container_width=True, hide_index=True)
