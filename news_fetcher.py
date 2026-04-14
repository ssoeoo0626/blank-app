import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import pandas as pd


DEFAULT_NEWS_ALLOWED_POOL_KEYS = {
    "trade",
    "data",
    "korea",
    "exhibition",
    "association",
    "competitor",
    "event",
}


def safe_lower(x):
    return str(x or "").strip().lower()


def infer_pool_keys_for_row(category, company, keyword, keyword_type):
    category = str(category or "").strip()
    keyword = str(keyword or "").strip()
    keyword_type = str(keyword_type or "").strip()
    keyword_l = safe_lower(keyword)

    pool_keys = set()
    pool_keys.add("trade")

    if category in ["경쟁사", "글로벌 스튜디오", "글로벌 배급사", "K배급사", "엔터"]:
        pool_keys.add("official")

    if keyword_type in ["재무", "재무/투자"] or any(
        t in keyword_l for t in [
            "earnings", "guidance", "outlook", "debt", "refinancing",
            "liquidity", "investor presentation", "shareholder letter"
        ]
    ):
        pool_keys.add("ir")
        pool_keys.add("data")

    if category in ["North America", "Europe", "Latin America", "APAC", "Middle East & Africa", "극장산업"]:
        pool_keys.add("exhibition")
        pool_keys.add("association")

    if category in ["콘텐츠", "글로벌 스튜디오", "글로벌 배급사", "K배급사", "K-콘텐츠/K무비", "엔터"]:
        pool_keys.add("trade")
        pool_keys.add("data")

    if category in ["K배급사", "K-콘텐츠/K무비", "엔터"]:
        pool_keys.add("korea")

    if category in ["경쟁사", "프리미엄포맷", "특별관/프리미엄"]:
        pool_keys.add("competitor")
        pool_keys.add("event")

    if category in ["라이브/이벤트 콘텐츠"]:
        pool_keys.add("event")
        pool_keys.add("trade")

    return sorted(list(pool_keys))


def classify_pool_key(site_row):
    name = safe_lower(site_row.get("사이트명", ""))
    domain = safe_lower(site_row.get("도메인", ""))
    pool = safe_lower(site_row.get("풀구분", ""))
    source_type = safe_lower(site_row.get("소스유형", ""))

    if "official" in source_type or "공식" in source_type:
        if "investor" in name or "ir" in name or "ir" in domain:
            return "ir"
        return "official"

    if "boxofficepro" in domain or "cinemaunited" in domain or "unic" in domain:
        if "association" in source_type or "협회" in source_type:
            return "association"
        return "exhibition"

    if "variety.com" in domain or "deadline.com" in domain:
        return "trade"

    if "the-numbers.com" in domain or "comscore" in domain:
        return "data"

    if "kofic" in domain or "kobis" in domain or "koreanfilm" in domain:
        return "korea"

    if (
        "imax" in domain or "dolby" in domain or "cj4dplex" in domain
        or "d-box" in domain or "mediamation" in domain or "lumma" in domain
    ):
        return "competitor"

    if "cinemacon" in domain:
        return "event"

    if pool in ["한국", "korea"]:
        return "korea"
    if pool in ["포맷/경쟁사"]:
        return "competitor"
    if pool in ["행사"]:
        return "event"

    return "trade"


def build_search_query(company, keyword, domain, use_site_filter=True):
    company = str(company or "").strip()
    keyword = str(keyword or "").strip()
    domain = str(domain or "").strip()

    if use_site_filter and domain:
        if company and company != "ALL":
            return 'site:{0} "{1}" "{2}"'.format(domain, company, keyword)
        return 'site:{0} "{1}"'.format(domain, keyword)

    if company and company != "ALL":
        return '"{0}" "{1}"'.format(company, keyword)
    return '"{0}"'.format(keyword)


def _is_news_searchable_pool(pool_key):
    return pool_key in DEFAULT_NEWS_ALLOWED_POOL_KEYS


def build_query_table(
    keyword_df,
    site_pool_df,
    company_domains_df=None,
    selected_pool_keys=None,
    selected_priorities=None,
    max_sites_per_keyword=4,
):
    if keyword_df is None or keyword_df.empty or site_pool_df is None or site_pool_df.empty:
        return pd.DataFrame(columns=[
            "분류", "회사", "키워드", "키워드유형", "사이트명", "도메인",
            "우선순위", "pool_key", "뉴스검색가능", "검색쿼리"
        ])

    selected_pool_keys = set(selected_pool_keys or [])
    selected_priorities = set(selected_priorities or [])

    pool_df = site_pool_df.copy()
    pool_df["pool_key"] = pool_df.apply(classify_pool_key, axis=1)
    pool_df["뉴스검색가능"] = pool_df["pool_key"].apply(_is_news_searchable_pool)

    if selected_pool_keys:
        pool_df = pool_df[pool_df["pool_key"].isin(list(selected_pool_keys))].copy()
    if selected_priorities:
        pool_df = pool_df[pool_df["우선순위"].isin(list(selected_priorities))].copy()

    company_domain_map = {}
    company_ir_domain_map = {}

    if company_domains_df is not None and not company_domains_df.empty:
        for _, r in company_domains_df.iterrows():
            company = str(r.get("회사", "")).strip()
            dom = str(r.get("도메인", "")).strip()
            ir_dom = str(r.get("IR도메인", "")).strip()
            if company:
                if dom:
                    company_domain_map[company] = dom
                if ir_dom:
                    company_ir_domain_map[company] = ir_dom

    rows = []

    for _, r in keyword_df.iterrows():
        category = str(r.get("분류", "")).strip()
        company = str(r.get("회사", "")).strip()
        keyword = str(r.get("키워드", "")).strip()
        keyword_type = str(r.get("키워드유형", "")).strip()

        if not keyword:
            continue

        allowed_keys = set(infer_pool_keys_for_row(category, company, keyword, keyword_type))
        if selected_pool_keys:
            allowed_keys = allowed_keys & set(selected_pool_keys)

        candidate_df = pool_df.copy()
        if allowed_keys:
            candidate_df = candidate_df[candidate_df["pool_key"].isin(list(allowed_keys))].copy()

        if not candidate_df.empty:
            candidate_df = candidate_df.sort_values(["우선순위", "사이트명"]).head(max_sites_per_keyword).copy()

            for _, s in candidate_df.iterrows():
                domain = str(s.get("도메인", "")).strip()
                site_name = str(s.get("사이트명", "")).strip()
                priority = str(s.get("우선순위", "")).strip()
                pool_key = str(s.get("pool_key", "")).strip()
                news_ok = bool(s.get("뉴스검색가능", False))

                query = build_search_query(company, keyword, domain, use_site_filter=False)

                rows.append({
                    "분류": category,
                    "회사": company,
                    "키워드": keyword,
                    "키워드유형": keyword_type,
                    "사이트명": site_name,
                    "도메인": domain,
                    "우선순위": priority,
                    "pool_key": pool_key,
                    "뉴스검색가능": 1 if news_ok else 0,
                    "검색쿼리": query,
                })

        if company in company_domain_map:
            rows.append({
                "분류": category,
                "회사": company,
                "키워드": keyword,
                "키워드유형": keyword_type,
                "사이트명": "{} Official".format(company),
                "도메인": company_domain_map[company],
                "우선순위": "P1",
                "pool_key": "official",
                "뉴스검색가능": 0,
                "검색쿼리": build_search_query(company, keyword, company_domain_map[company], use_site_filter=False),
            })

        if company in company_ir_domain_map:
            rows.append({
                "분류": category,
                "회사": company,
                "키워드": keyword,
                "키워드유형": keyword_type,
                "사이트명": "{} IR".format(company),
                "도메인": company_ir_domain_map[company],
                "우선순위": "P1",
                "pool_key": "ir",
                "뉴스검색가능": 0,
                "검색쿼리": build_search_query(company, keyword, company_ir_domain_map[company], use_site_filter=False),
            })

    query_df = pd.DataFrame(rows)
    if query_df.empty:
        return query_df

    query_df = query_df.drop_duplicates(subset=["회사", "키워드", "도메인", "검색쿼리"]).copy()
    query_df = query_df.sort_values(
        ["분류", "회사", "키워드", "우선순위", "사이트명"]
    ).reset_index(drop=True)
    return query_df


def fetch_url_text(url, timeout=20):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def google_news_rss_url(query, days_back=30, hl="en-US", gl="US", ceid="US:en"):
    date_filter = " when:{0}d".format(days_back)
    q = urllib.parse.quote_plus(str(query) + date_filter)
    return "https://news.google.com/rss/search?q={0}&hl={1}&gl={2}&ceid={3}".format(q, hl, gl, ceid)


def parse_pubdate(pub_text):
    try:
        return parsedate_to_datetime(pub_text)
    except Exception:
        return None


def fetch_google_news_rss(query, timeout=20, days_back=30):
    url = google_news_rss_url(query=query, days_back=days_back)
    try:
        raw = fetch_url_text(url, timeout=timeout)
        root = ET.fromstring(raw)
    except Exception as e:
        return [], str(e), url

    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        pub_date = item.findtext("pubDate", default="").strip()
        source_el = item.find("source")
        source_name = source_el.text.strip() if source_el is not None and source_el.text else ""
        published_at = parse_pubdate(pub_date)

        items.append({
            "title": title,
            "url": link,
            "published_at": published_at,
            "source": source_name,
        })

    return items, "", url


def extract_domain(url):
    try:
        netloc = urlparse(str(url)).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def domain_matches(target_domain, article_url, source_name=""):
    target = str(target_domain or "").strip().lower()
    article_domain = extract_domain(article_url)

    if not target:
        return True

    if article_domain == target:
        return True

    if article_domain.endswith("." + target):
        return True

    source_name_l = str(source_name or "").strip().lower()
    if target.replace(".com", "") in source_name_l:
        return True

    return False


def build_query_candidates(company, keyword):
    company = str(company or "").strip()
    keyword = str(keyword or "").strip()

    candidates = []

    if company and keyword:
        candidates.append(("company_keyword", '"{0}" "{1}"'.format(company, keyword)))

    if company:
        candidates.append(("company_only", '"{0}"'.format(company)))

    if keyword:
        candidates.append(("keyword_only", '"{0}"'.format(keyword)))

    seen = set()
    out = []
    for stage, query in candidates:
        if query not in seen:
            seen.add(query)
            out.append((stage, query))
    return out


def fetch_news_from_query_table(query_df, days_back=30, max_errors_to_keep=50):
    if query_df is None or query_df.empty:
        return pd.DataFrame(columns=[
            "published_at", "분류", "회사", "키워드", "사이트명", "도메인",
            "title", "url", "source", "검색쿼리", "수집상태", "오류메시지",
            "debug_query", "retry_stage", "raw_result_count"
        ])

    rows = []
    seen = set()

    use_df = query_df.copy()
    if "뉴스검색가능" in use_df.columns:
        use_df = use_df[use_df["뉴스검색가능"] == 1].copy()

    if use_df.empty:
        return pd.DataFrame(columns=[
            "published_at", "분류", "회사", "키워드", "사이트명", "도메인",
            "title", "url", "source", "검색쿼리", "수집상태", "오류메시지",
            "debug_query", "retry_stage", "raw_result_count"
        ])

    error_count = 0

    for _, r in use_df.iterrows():
        category = str(r.get("분류", "")).strip()
        company = str(r.get("회사", "")).strip()
        keyword = str(r.get("키워드", "")).strip()
        site_name = str(r.get("사이트명", "")).strip()
        domain = str(r.get("도메인", "")).strip()
        original_query = str(r.get("검색쿼리", "")).strip()

        if not company and not keyword:
            continue

        query_candidates = build_query_candidates(company, keyword)

        matched_items = []
        last_err = ""
        last_debug_query = ""
        last_retry_stage = ""
        last_raw_count = 0

        for retry_stage, debug_query in query_candidates:
            news_items, err, rss_url = fetch_google_news_rss(query=debug_query, days_back=days_back)
            time.sleep(0.2)

            last_err = err
            last_debug_query = debug_query
            last_retry_stage = retry_stage
            last_raw_count = len(news_items)

            if err:
                continue

            filtered_items = news_items[:]

            if filtered_items:
                matched_items = filtered_items
                break

        if last_err and not matched_items:
            if error_count < max_errors_to_keep:
                rows.append({
                    "published_at": pd.NaT,
                    "분류": category,
                    "회사": company,
                    "키워드": keyword,
                    "사이트명": site_name,
                    "도메인": domain,
                    "title": "",
                    "url": "",
                    "source": "",
                    "검색쿼리": original_query,
                    "수집상태": "error",
                    "오류메시지": last_err,
                    "debug_query": last_debug_query,
                    "retry_stage": last_retry_stage,
                    "raw_result_count": last_raw_count,
                })
                error_count += 1
            continue

        if not matched_items:
            rows.append({
                "published_at": pd.NaT,
                "분류": category,
                "회사": company,
                "키워드": keyword,
                "사이트명": site_name,
                "도메인": domain,
                "title": "",
                "url": "",
                "source": "",
                "검색쿼리": original_query,
                "수집상태": "empty",
                "오류메시지": "",
                "debug_query": last_debug_query,
                "retry_stage": last_retry_stage,
                "raw_result_count": last_raw_count,
            })
            continue

        for item in matched_items:
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            source = str(item.get("source", "")).strip()
            published_at = item.get("published_at")

            dedup_key = (company, keyword, domain, title, url)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            rows.append({
                "published_at": published_at,
                "분류": category,
                "회사": company,
                "키워드": keyword,
                "사이트명": site_name,
                "도메인": domain,
                "title": title,
                "url": url,
                "source": source,
                "검색쿼리": original_query,
                "수집상태": "ok",
                "오류메시지": "",
                "debug_query": last_debug_query,
                "retry_stage": last_retry_stage,
                "raw_result_count": last_raw_count,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    try:
        df["published_at"] = df["published_at"].dt.tz_localize(None)
    except Exception:
        try:
            df["published_at"] = df["published_at"].dt.tz_convert(None)
        except Exception:
            pass

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

    ok_df = df[df["수집상태"] == "ok"].copy()
    if not ok_df.empty:
        ok_df = ok_df[ok_df["published_at"].fillna(pd.Timestamp("1900-01-01")) >= cutoff].copy()

    other_df = df[df["수집상태"] != "ok"].copy()
    out = pd.concat([ok_df, other_df], ignore_index=True)

    if not out.empty:
        out = out.sort_values(["수집상태", "published_at"], ascending=[True, False]).reset_index(drop=True)

    return out
