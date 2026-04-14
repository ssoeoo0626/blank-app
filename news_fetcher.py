
from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, List

import pandas as pd


def safe_lower(x: str) -> str:
    return str(x or "").strip().lower()


def infer_pool_keys_for_row(category: str, company: str, keyword: str, keyword_type: str) -> list[str]:
    category_l = safe_lower(category)
    company_l = safe_lower(company)
    keyword_l = safe_lower(keyword)
    keyword_type_l = safe_lower(keyword_type)

    pool_keys = set()

    # defaults
    pool_keys.update(["trade"])

    # official / ir
    if category in ["경쟁사", "글로벌 스튜디오", "글로벌 배급사", "K배급사", "엔터"]:
        pool_keys.update(["official"])
    if keyword_type in ["재무", "재무/투자"] or any(
        t in keyword_l for t in ["earnings", "guidance", "outlook", "debt", "refinancing", "liquidity", "investor presentation"]
    ):
        pool_keys.update(["ir", "data"])

    # exhibition categories
    if category in ["North America", "Europe", "Latin America", "APAC", "Middle East & Africa", "극장산업"]:
        pool_keys.update(["exhibition", "association"])

    # content / studios / distributors
    if category in ["콘텐츠", "글로벌 스튜디오", "글로벌 배급사", "K배급사", "K-콘텐츠/K무비", "엔터"]:
        pool_keys.update(["trade", "data"])

    # Korea
    if category in ["K배급사", "K-콘텐츠/K무비", "엔터"]:
        pool_keys.update(["korea"])

    # competitor / premium formats
    if category in ["경쟁사", "프리미엄포맷", "특별관/프리미엄"]:
        pool_keys.update(["competitor", "event"])

    # event / live
    if category in ["라이브/이벤트 콘텐츠"]:
        pool_keys.update(["event", "trade"])

    return sorted(pool_keys)


def classify_pool_key(site_row: pd.Series) -> str:
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

    if "imax" in domain or "dolby" in domain or "cj4dplex" in domain or "d-box" in domain or "mediamation" in domain or "lumma" in domain:
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


def build_query_table(
    keyword_df: pd.DataFrame,
    site_pool_df: pd.DataFrame,
    company_domains_df: pd.DataFrame | None = None,
    selected_pool_keys: list[str] | None = None,
    selected_priorities: list[str] | None = None,
    max_sites_per_keyword: int = 4,
) -> pd.DataFrame:
    if keyword_df is None or keyword_df.empty or site_pool_df is None or site_pool_df.empty:
        return pd.DataFrame(columns=[
            "분류", "회사", "키워드", "키워드유형", "사이트명", "도메인", "우선순위", "검색쿼리"
        ])

    selected_pool_keys = set(selected_pool_keys or [])
    selected_priorities = set(selected_priorities or [])

    pool_df = site_pool_df.copy()
    pool_df["pool_key"] = pool_df.apply(classify_pool_key, axis=1)

    if selected_pool_keys:
        pool_df = pool_df[pool_df["pool_key"].isin(selected_pool_keys)].copy()
    if selected_priorities:
        pool_df = pool_df[pool_df["우선순위"].isin(selected_priorities)].copy()

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
        candidate_df = pool_df.copy()
        if selected_pool_keys:
            allowed_keys = allowed_keys & set(selected_pool_keys)

        if allowed_keys:
            candidate_df = candidate_df[candidate_df["pool_key"].isin(list(allowed_keys))].copy()

        # optional: prepend company official domains
        extra_sites = []
        if company in company_domain_map and ("official" in allowed_keys or not selected_pool_keys):
            extra_sites.append({
                "사이트명": f"{company} Official",
                "도메인": company_domain_map[company],
                "우선순위": "P1",
            })
        if company in company_ir_domain_map and ("ir" in allowed_keys or not selected_pool_keys):
            extra_sites.append({
                "사이트명": f"{company} IR",
                "도메인": company_ir_domain_map[company],
                "우선순위": "P1",
            })

        if not candidate_df.empty:
            candidate_df = candidate_df.sort_values(["우선순위", "사이트명"]).head(max_sites_per_keyword).copy()
            for _, s in candidate_df.iterrows():
                domain = str(s.get("도메인", "")).strip()
                site_name = str(s.get("사이트명", "")).strip()
                priority = str(s.get("우선순위", "")).strip()

                query = build_search_query(company, keyword, domain)
                rows.append({
                    "분류": category,
                    "회사": company,
                    "키워드": keyword,
                    "키워드유형": keyword_type,
                    "사이트명": site_name,
                    "도메인": domain,
                    "우선순위": priority,
                    "검색쿼리": query,
                })

        for s in extra_sites[:max_sites_per_keyword]:
            domain = s["도메인"]
            query = build_search_query(company, keyword, domain)
            rows.append({
                "분류": category,
                "회사": company,
                "키워드": keyword,
                "키워드유형": keyword_type,
                "사이트명": s["사이트명"],
                "도메인": domain,
                "우선순위": s["우선순위"],
                "검색쿼리": query,
            })

    query_df = pd.DataFrame(rows)
    if query_df.empty:
        return query_df

    query_df = query_df.drop_duplicates(subset=["회사", "키워드", "도메인", "검색쿼리"]).copy()
    query_df = query_df.sort_values(["분류", "회사", "키워드", "우선순위", "사이트명"]).reset_index(drop=True)
    return query_df


def build_search_query(company: str, keyword: str, domain: str) -> str:
    company = str(company or "").strip()
    keyword = str(keyword or "").strip()
    domain = str(domain or "").strip()

    if company and company != "ALL":
        return f'site:{domain} "{company}" "{keyword}"'
    return f'site:{domain} "{keyword}"'


def fetch_url_text(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def google_news_rss_url(query: str, days_back: int = 30, hl: str = "en-US", gl: str = "US", ceid: str = "US:en") -> str:
    date_filter = f" when:{days_back}d"
    q = urllib.parse.quote_plus(query + date_filter)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def parse_pubdate(pub_text: str):
    try:
        return parsedate_to_datetime(pub_text)
    except Exception:
        return None


def fetch_google_news_rss(query: str, timeout: int = 20) -> list[dict]:
    url = google_news_rss_url(query=query)
    try:
        raw = fetch_url_text(url, timeout=timeout)
        root = ET.fromstring(raw)
    except Exception:
        return []

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
    return items


def fetch_news_from_query_table(query_df: pd.DataFrame, days_back: int = 30) -> pd.DataFrame:
    if query_df is None or query_df.empty:
        return pd.DataFrame(columns=[
            "published_at", "분류", "회사", "키워드", "사이트명", "도메인", "title", "url", "source"
        ])

    rows = []
    seen = set()

    for _, r in query_df.iterrows():
        category = str(r.get("분류", "")).strip()
        company = str(r.get("회사", "")).strip()
        keyword = str(r.get("키워드", "")).strip()
        site_name = str(r.get("사이트명", "")).strip()
        domain = str(r.get("도메인", "")).strip()
        query = str(r.get("검색쿼리", "")).strip()

        if not query:
            continue

        news_items = fetch_google_news_rss(query=query)
        time.sleep(0.25)

        for item in news_items:
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            source = str(item.get("source", "")).strip()
            published_at = item.get("published_at")

            # de-dup key
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
                "검색쿼리": query,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_back)
    df["published_at"] = df["published_at"].dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")
    df = df[df["published_at"].fillna(pd.Timestamp("1900-01-01")) >= cutoff].copy()

    if not df.empty:
        df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    return df
