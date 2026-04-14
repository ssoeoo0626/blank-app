
사용 방법

1) 아래 파일들을 같은 폴더에 둡니다.
- app.py
- news_fetcher.py
- earnings 발표일.CSV
- theater_keywords_expanded_v2.csv
- site_pool_master.csv

2) 선택 사항
- company_domains.csv
  형식:
  회사,도메인,IR도메인,비고
  AMC Theatres,amctheatres.com,investor.amctheatres.com,

3) 실행
streamlit run app.py

핵심 기능
- 기존 실적 캘린더 유지
- theater_keywords_expanded_v2.csv + site_pool_master.csv 기반 검색 쿼리 자동 생성
- Google News RSS로 최근 기사 수집
- 검색 쿼리 CSV / 뉴스 결과 CSV 다운로드

주의
- Google News RSS는 site: 검색을 완벽히 보장하지 않을 수 있음
- 정확도를 높이려면 company_domains.csv를 채워서 공식 사이트 도메인을 매핑하는 것이 좋음
