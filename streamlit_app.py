import calendar
import html as html_lib
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


# =============================
# 0) BASIC SETTINGS
# =============================
st.set_page_config(page_title="Earnings Calendar", layout="wide")
st.title("Earnings Calendar")

BASE_DIR = Path(__file__).resolve().parent
EARNINGS_CSV_PATH = BASE_DIR / "earnings 발표일.CSV"

PREDICT_START_YEAR = 2026
PREDICT_END_YEAR = 2028


# =============================
# 1) DATA LOADER
# =============================
def get_file_mtime(path: Path) -> float:
    if path.exists():
        return path.stat().st_mtime
    return 0


@st.cache_data
def load_earnings_data(path_str: str, file_mtime: float):
    path = Path(path_str)

    required_cols = [
        "company",
        "fiscal_period",
        "announcement_date",
        "status",
        "source",
    ]

    if not path.exists():
        return pd.DataFrame(columns=required_cols)

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            engine="python",
            quotechar='"',
            on_bad_lines="skip",
        )
    except Exception as e:
        st.error(f"earnings 발표일.CSV를 읽는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame(columns=required_cols)

    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"earnings 발표일.CSV에 필요한 컬럼이 없습니다: {missing_cols}")
        return pd.DataFrame(columns=required_cols)

    df = df[required_cols].copy()

    for col in ["company", "fiscal_period", "status", "source"]:
        df[col] = df[col].astype(str).str.strip()

    df["announcement_date"] = pd.to_datetime(
        df["announcement_date"],
        errors="coerce",
    )

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

    total_weight = sum(weights)

    if total_weight == 0:
        return None

    return sum(v * w for v, w in zip(values, weights)) / total_weight


def prepare_base_dataframe(df):
    if df.empty:
        return df.copy()

    work = df.copy()

    work["status_norm"] = work["status"].apply(normalize_status)
    work["quarter"] = work["fiscal_period"].apply(extract_quarter)

    work["fiscal_year"] = work.apply(
        lambda row: extract_fiscal_year(
            row["fiscal_period"],
            row["announcement_date"],
        ),
        axis=1,
    )

    work["period_end"] = work.apply(
        lambda row: get_period_end(
            row["fiscal_year"],
            row["quarter"],
        ),
        axis=1,
    )

    work = work.dropna(subset=["quarter", "period_end"]).copy()

    work["lag_days"] = (
        work["announcement_date"] - work["period_end"]
    ).dt.days

    return work


def get_actual_history(df_base, company, quarter, before_fiscal_year):
    hist = df_base[
        (df_base["company"] == company)
        & (df_base["quarter"] == quarter)
        & (df_base["fiscal_year"] < before_fiscal_year)
        & (df_base["status_norm"] == "actual")
    ].copy()

    return hist.sort_values("fiscal_year")


def get_peer_history(df_base, company, quarter, before_fiscal_year):
    peer = df_base[
        (df_base["company"] != company)
        & (df_base["quarter"] == quarter)
        & (df_base["fiscal_year"] < before_fiscal_year)
        & (df_base["status_norm"] == "actual")
    ].copy()

    return peer.sort_values("fiscal_year")


def default_lag_by_quarter(quarter):
    default_map = {
        "Q1": 40,
        "Q2": 40,
        "Q3": 40,
        "Q4": 52,
    }

    return default_map.get(quarter, 40)


def predict_lag(df_base, company, quarter, target_fiscal_year):
    actual_hist = get_actual_history(
        df_base,
        company,
        quarter,
        target_fiscal_year,
    )

    peer_hist = get_peer_history(
        df_base,
        company,
        quarter,
        target_fiscal_year,
    )

    actual_lags = actual_hist["lag_days"].dropna().tolist()
    peer_lags = peer_hist["lag_days"].dropna().tolist()

    if len(actual_lags) >= 3:
        recent = actual_lags[-3:]
        return (
            round(weighted_average(recent, [0.2, 0.3, 0.5])),
            "High",
            "last_3_actuals_weighted",
        )

    if len(actual_lags) == 2:
        recent = actual_lags[-2:]
        return (
            round(weighted_average(recent, [0.4, 0.6])),
            "Mid",
            "last_2_actuals_weighted",
        )

    if len(actual_lags) == 1:
        company_lag = actual_lags[-1]

        if len(peer_lags) > 0:
            peer_avg = round(sum(peer_lags) / len(peer_lags))
            return (
                round(company_lag * 0.7 + peer_avg * 0.3),
                "Low",
                "1_actual_plus_peer_avg",
            )

        return company_lag, "Low", "1_actual_only"

    if len(peer_lags) > 0:
        return (
            round(sum(peer_lags) / len(peer_lags)),
            "Low",
            "peer_avg_only",
        )

    return (
        default_lag_by_quarter(quarter),
        "Low",
        "default_quarter_lag",
    )


def format_predicted_fiscal_period(fiscal_year, quarter):
    return f"{fiscal_year} {quarter}"


def generate_predictions(
    df_raw,
    start_year=PREDICT_START_YEAR,
    end_year=PREDICT_END_YEAR,
):
    columns = [
        "company",
        "fiscal_period",
        "announcement_date",
        "status",
        "source",
        "prediction_confidence",
        "prediction_basis",
    ]

    if df_raw.empty:
        return pd.DataFrame(columns=columns)

    today_ts = pd.Timestamp(date.today())

    df_base = prepare_base_dataframe(df_raw)

    if df_base.empty:
        return pd.DataFrame(columns=columns)

    companies = sorted(df_base["company"].dropna().unique().tolist())
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    existing_keys = set(
        zip(
            df_base["company"],
            df_base["quarter"],
            df_base["fiscal_year"],
        )
    )

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
                    working_base,
                    company,
                    quarter,
                    fiscal_year,
                )

                period_end = get_period_end(fiscal_year, quarter)

                if period_end is None:
                    continue

                predicted_date = adjust_to_business_day(
                    period_end + pd.Timedelta(days=int(predicted_lag))
                )

                # 오늘 이전 예측값은 생성하지 않음
                if predicted_date < today_ts:
                    existing_keys.add(key)
                    continue

                new_rows_for_year.append(
                    {
                        "company": company,
                        "fiscal_period": format_predicted_fiscal_period(
                            fiscal_year,
                            quarter,
                        ),
                        "announcement_date": predicted_date,
                        "status": "predicted",
                        "source": f"model:{basis}",
                        "prediction_confidence": confidence,
                        "prediction_basis": basis,
                    }
                )

                existing_keys.add(key)

        if new_rows_for_year:
            year_df = pd.DataFrame(new_rows_for_year)
            prediction_rows.append(year_df)

            base_like_year_df = year_df.copy()
            base_like_year_df["status_norm"] = "predicted"
            base_like_year_df["quarter"] = base_like_year_df["fiscal_period"].apply(
                extract_quarter
            )

            base_like_year_df["fiscal_year"] = base_like_year_df.apply(
                lambda row: extract_fiscal_year(
                    row["fiscal_period"],
                    row["announcement_date"],
                ),
                axis=1,
            )

            base_like_year_df["period_end"] = base_like_year_df.apply(
                lambda row: get_period_end(
                    row["fiscal_year"],
                    row["quarter"],
                ),
                axis=1,
            )

            base_like_year_df["lag_days"] = (
                base_like_year_df["announcement_date"]
                - base_like_year_df["period_end"]
            ).dt.days

            working_base = pd.concat(
                [working_base, base_like_year_df],
                ignore_index=True,
            )

    if prediction_rows:
        return pd.concat(prediction_rows, ignore_index=True)

    return pd.DataFrame(columns=columns)


# =============================
# 3) LOAD DATA
# =============================
earnings_df = load_earnings_data(
    str(EARNINGS_CSV_PATH),
    get_file_mtime(EARNINGS_CSV_PATH),
)

predicted_df = generate_predictions(
    earnings_df,
    PREDICT_START_YEAR,
    PREDICT_END_YEAR,
)


# =============================
# 4) SIDEBAR
# =============================
st.sidebar.header("설정")

if st.sidebar.button("캐시 전체 삭제 후 새로고침"):
    st.cache_data.clear()
    st.rerun()

today = date.today()

year_options = list(range(2018, PREDICT_END_YEAR + 1))
default_year_index = (
    year_options.index(today.year)
    if today.year in year_options
    else len(year_options) - 1
)

selected_year = st.sidebar.selectbox(
    "Calendar Year",
    year_options,
    index=default_year_index,
)

selected_month = st.sidebar.selectbox(
    "Calendar Month",
    list(range(1, 13)),
    index=today.month - 1,
)

show_predicted = st.sidebar.checkbox(
    "예측 일정 표시",
    value=True,
)

selected_statuses = st.sidebar.multiselect(
    "표시할 상태",
    ["actual", "past", "confirmed", "planned", "predicted"],
    default=["actual", "past", "confirmed", "planned", "predicted"],
)

company_options = (
    ["All"] + sorted(earnings_df["company"].dropna().unique().tolist())
    if not earnings_df.empty
    else ["All"]
)

selected_companies = st.sidebar.multiselect(
    "회사",
    company_options,
    default=["All"],
)


# =============================
# 5) SUMMARY
# =============================
actual_count = len(earnings_df)
predicted_count = len(predicted_df)

m1, m2, m3 = st.columns(3)

m1.metric("입력 일정 수", f"{actual_count:,}")
m2.metric("예측 일정 수", f"{predicted_count:,}")
m3.metric("CSV 파일", "있음" if EARNINGS_CSV_PATH.exists() else "없음")


# =============================
# 6) CALENDAR
# =============================
st.subheader("실적 캘린더")

frames = []

if not earnings_df.empty:
    frames.append(earnings_df.copy())

if show_predicted and not predicted_df.empty:
    frames.append(predicted_df.copy())

display_df = (
    pd.concat(frames, ignore_index=True)
    if frames
    else pd.DataFrame(
        columns=[
            "company",
            "fiscal_period",
            "announcement_date",
            "status",
            "source",
        ]
    )
)

if display_df.empty:
    st.info("earnings 발표일.CSV가 없거나 비어 있습니다.")
else:
    display_df["announcement_date"] = pd.to_datetime(
        display_df["announcement_date"],
        errors="coerce",
    )

    display_df = display_df.dropna(subset=["announcement_date"]).copy()

    today_ts = pd.Timestamp(date.today())

    display_df["status_lower"] = (
        display_df["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    display_df["status_lower"] = display_df["status_lower"].replace(
        {
            "prediction": "predicted",
            "actual": "actual",
            "confirmed": "confirmed",
            "past": "past",
            "planned": "planned",
        }
    )

    # 오늘 이전 predicted 일정은 제거
    display_df = display_df[
        ~(
            (display_df["status_lower"] == "predicted")
            & (display_df["announcement_date"] < today_ts)
        )
    ].copy()

    # 상태 필터
    if selected_statuses:
        selected_statuses_lower = [s.lower() for s in selected_statuses]
        display_df = display_df[
            display_df["status_lower"].isin(selected_statuses_lower)
        ].copy()

    # 회사 필터
    if selected_companies and "All" not in selected_companies:
        display_df = display_df[
            display_df["company"].isin(selected_companies)
        ].copy()

    filtered = display_df[
        (display_df["announcement_date"].dt.year == selected_year)
        & (display_df["announcement_date"].dt.month == selected_month)
    ].copy()

    event_map = {}

    for _, row in filtered.iterrows():
        d = row["announcement_date"].date()
        event_map.setdefault(d, []).append(row.to_dict())

    st.markdown(
        """
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

        .actual {
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
        """,
        unsafe_allow_html=True,
    )

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

            for event in day_events[:4]:
                status = str(event.get("status_lower", "unknown")).strip().lower()

                if status == "prediction":
                    status = "predicted"

                if status not in [
                    "actual",
                    "past",
                    "confirmed",
                    "predicted",
                    "planned",
                ]:
                    status = "unknown"

                company = html_lib.escape(str(event.get("company", "")))
                source = html_lib.escape(str(event.get("source", "")))
                confidence = html_lib.escape(
                    str(event.get("prediction_confidence", ""))
                )
                fiscal_period = html_lib.escape(str(event.get("fiscal_period", "")))

                meta_text = f"{source} ({confidence})" if confidence else source

                calendar_html += (
                    f'<div class="event-box {status}">'
                    f'<div><strong>{company}</strong></div>'
                    f'<div>{fiscal_period}</div>'
                    f'<div class="event-meta">{meta_text}</div>'
                    f'</div>'
                )

            if len(day_events) > 4:
                calendar_html += (
                    f'<div style="font-size:12px;color:#4b5563;">'
                    f'+{len(day_events) - 4} more'
                    f'</div>'
                )

            calendar_html += "</div>"

    calendar_html += "</div></div>"

    st.markdown(calendar_html, unsafe_allow_html=True)

    st.markdown("월간 이벤트 목록")

    show = filtered.sort_values("announcement_date").copy()
    show["announcement_date"] = show["announcement_date"].dt.strftime("%Y-%m-%d")

    cols = [
        "announcement_date",
        "company",
        "fiscal_period",
        "status",
        "source",
    ]

    if "prediction_confidence" in show.columns:
        cols.append("prediction_confidence")

    if "prediction_basis" in show.columns:
        cols.append("prediction_basis")

    st.dataframe(
        show[cols],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = show[cols].to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")

    st.download_button(
        label="월간 이벤트 CSV 다운로드",
        data=csv_bytes,
        file_name=f"earnings_calendar_{selected_year}_{selected_month:02d}.csv",
        mime="text/csv",
    )
