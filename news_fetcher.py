import re
import time
import requests
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timezone

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}

KEYWORDS = [
    "imax",
    "4d e-motion",
    "4d e motion",
    "lumma",
    "harkins",
    "cinemacon",
    "d-box",
    "d-box technologies",
    "mx4d",
    "mediamation",
    "screenx",
    "4dx",
]

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def contains_keyword(text: str, keywords=None) -> bool:
    keywords = keywords or KEYWORDS
    t = normalize_text(text)
    return any(k in t for k in keywords)

def parse_date_safe(date_str: str):
    if not date_str:
        return None
    try:
        dt = pd.to_datetime(date_str, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.tz_convert("Asia/Seoul").date()
    except Exception:
        return None

def fetch_rss_feed(feed_url: str, source_name: str, company_hint: str = ""):
    rows = []
    feed = feedparser.parse(feed_url)

    for entry in feed.entries:
        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        summary = getattr(entry, "summary", "") or ""

        published = None
        if hasattr(entry, "published"):
            published = parse_date_safe(entry.published)
        elif hasattr(entry, "updated"):
            published = parse_date_safe(entry.updated)

        text_blob = f"{title} {summary} {company_hint}"
        if contains_keyword(text_blob):
            rows.append({
                "date": published,
                "company": company_hint if company_hint else infer_company(title, summary),
                "category": "News",
                "title": clean_html(title),
                "url": link,
                "source": source_name,
            })

    return rows

def clean_html(text: str) -> str:
    return BeautifulSoup(text or "", "html.parser").get_text(" ", strip=True)

def infer_company(title: str, body: str = "") -> str:
    blob = normalize_text(f"{title} {body}")
    if "imax" in blob:
        return "IMAX"
    if "lumma" in blob or "4d e-motion" in blob or "4d e motion" in blob:
        return "Lumma"
    if "d-box" in blob:
        return "D-BOX"
    if "mx4d" in blob or "mediamation" in blob:
        return "MediaMation"
    if "4dx" in blob or "screenx" in blob:
        return "4DPLEX"
    return "Industry"

def scrape_imax_ir():
    url = "https://investors.imax.com/news-releases"
    rows = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # gcs-web 계열 일반 구조 대응
        links = soup.select("a[href*='/news-releases/news-release-details/']")
        seen = set()

        for a in links:
            href = a.get("href", "").strip()
            title = a.get_text(" ", strip=True)

            if not href or not title:
                continue

            if href.startswith("/"):
                href = "https://investors.imax.com" + href

            if href in seen:
                continue
            seen.add(href)

            parent_text = a.parent.get_text(" ", strip=True)
            m = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", parent_text)
            pub_date = parse_date_safe(m.group(1)) if m else None

            if contains_keyword(title):
                rows.append({
                    "date": pub_date,
                    "company": "IMAX",
                    "category": "IR",
                    "title": title,
                    "url": href,
                    "source": "IMAX IR",
                })
    except Exception:
        pass

    return rows

def scrape_4demotion_news():
    url = "https://4demotion.com/news/"
    rows = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.select("a"):
            title = a.get_text(" ", strip=True)
            href = a.get("href", "").strip()

            if not title or not href:
                continue

            if title.lower() == "read more":
                continue

            if not href.startswith("http"):
                href = requests.compat.urljoin(url, href)

            if contains_keyword(title):
                rows.append({
                    "date": None,
                    "company": "Lumma",
                    "category": "News",
                    "title": title,
                    "url": href,
                    "source": "4D E-Motion",
                })
    except Exception:
        pass

    return dedupe_rows(rows)

def scrape_celluloidjunkie_cinemacon():
    url = "https://celluloidjunkie.com/"
    rows = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.select("a"):
            title = a.get_text(" ", strip=True)
            href = a.get("href", "").strip()

            if not title or not href:
                continue

            if not href.startswith("http"):
                href = requests.compat.urljoin(url, href)

            text_blob = title
            if "cinemacon" in normalize_text(text_blob) or contains_keyword(text_blob):
                rows.append({
                    "date": None,
                    "company": infer_company(title),
                    "category": "Article",
                    "title": title,
                    "url": href,
                    "source": "Celluloid Junkie",
                })
    except Exception:
        pass

    return dedupe_rows(rows)

def dedupe_rows(rows):
    seen = set()
    out = []

    for row in rows:
        key = (normalize_text(row.get("title", "")), row.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out

def enrich_dates_from_article(df: pd.DataFrame) -> pd.DataFrame:
    # 날짜 없는 기사에 대해 URL별 기사 본문에서 time 태그를 읽어보는 보조 로직
    enriched = []

    for _, row in df.iterrows():
        row = row.copy()

        if pd.notna(row.get("date")) and row.get("date") is not None:
            enriched.append(row)
            continue

        try:
            r = requests.get(row["url"], headers=HEADERS, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")

            candidates = []

            for t in soup.select("time"):
                dt = t.get("datetime") or t.get_text(" ", strip=True)
                parsed = parse_date_safe(dt)
                if parsed:
                    candidates.append(parsed)

            meta_props = [
                ("meta", {"property": "article:published_time"}),
                ("meta", {"name": "pubdate"}),
                ("meta", {"name": "date"}),
                ("meta", {"property": "og:updated_time"}),
            ]

            for tag_name, attrs in meta_props:
                for tag in soup.find_all(tag_name, attrs=attrs):
                    content = tag.get("content", "")
                    parsed = parse_date_safe(content)
                    if parsed:
                        candidates.append(parsed)

            if candidates:
                row["date"] = sorted(candidates)[0]
        except Exception:
            pass

        enriched.append(row)

        # 너무 빠르게 때리지 않도록
        time.sleep(0.3)

    return pd.DataFrame(enriched)

def fetch_all_news(days_back: int = 30) -> pd.DataFrame:
    rows = []

    # RSS
    rows += fetch_rss_feed(
        "https://www.boxofficepro.com/feed/",
        "Boxoffice Pro",
        "Industry"
    )
    rows += fetch_rss_feed(
        "https://celluloidjunkie.com/feed/",
        "Celluloid Junkie",
        "Industry"
    )

    # Scraping
    rows += scrape_imax_ir()
    rows += scrape_4demotion_news()
    rows += scrape_celluloidjunkie_cinemacon()

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "company", "category", "title", "url", "source"])

    df = dedupe_dataframe(df)
    df = enrich_dates_from_article(df)

    today = pd.Timestamp.now(tz="Asia/Seoul").date()
    min_date = today - pd.Timedelta(days=days_back)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["title"].notna() & df["url"].notna()]
    df = df[(df["date"].isna()) | (df["date"] >= min_date)]
    df = df.sort_values(["date", "company", "title"], ascending=[False, True, True])

    return df.reset_index(drop=True)

def dedupe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["title_norm"] = df["title"].astype(str).map(normalize_text)
    df = df.drop_duplicates(subset=["title_norm", "url"])
    return df.drop(columns=["title_norm"], errors="ignore")
