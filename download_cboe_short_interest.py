import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.cboe.com/us/equities/market_statistics/short_interest/{year}/Bats_Listed_Short_Interest-finra-{yyyymmdd}.csv-dl"

US_HOLIDAYS_2023_2026 = {
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}

HOLIDAYS = {pd.Timestamp(d).date() for d in US_HOLIDAYS_2023_2026}

def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS

def prev_business_day(d: date) -> date:
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d

def add_business_days(d: date, n: int) -> date:
    cur = d
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if is_business_day(cur):
            added += 1
    return cur

def settlement_dates_between(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        mid = date(cur.year, cur.month, 15)
        mid = prev_business_day(mid)
        if start <= mid <= end:
            yield mid

        if cur.month == 12:
            next_month = date(cur.year + 1, 1, 1)
        else:
            next_month = date(cur.year, cur.month + 1, 1)

        eom = prev_business_day(next_month - timedelta(days=1))
        if start <= eom <= end:
            yield eom

        cur = next_month

def candidate_publication_dates(settlement: date):
    for n in (7, 8, 6, 9):
        yield add_business_days(settlement, n)

def download_file(settlement: date, out_dir: Path, session: requests.Session, overwrite: bool = False) -> dict:
    result = {"settlement": settlement.isoformat(), "status": "pending", "url": None, "path": None}

    out_path = out_dir / f"Bats_Short_Interest_settle_{settlement.strftime('%Y%m%d')}.csv"
    if out_path.exists() and not overwrite:
        result.update(status="cached", path=str(out_path))
        return result

    for pub_date in candidate_publication_dates(settlement):
        url = BASE_URL.format(year=pub_date.year, yyyymmdd=pub_date.strftime("%Y%m%d"))
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException as e:
            result["status"] = f"error:{e.__class__.__name__}"
            continue

        if r.status_code == 200 and len(r.content) > 500:
            head = r.content[:500].decode("utf-8", errors="ignore")
            if "Cycle Settlement Date" in head or "BATS-Symbol" in head:
                out_path.write_bytes(r.content)
                result.update(status="downloaded", url=url, path=str(out_path))
                return result

    result["status"] = "not_found"
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01", help="Earliest settlement date (YYYY-MM-DD)")
    ap.add_argument("--end", default=date.today().isoformat(), help="Latest settlement date")
    ap.add_argument("--out", default="./cboe_history", help="Output directory")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between requests")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (research script; contact: your-email@example.com)"
    })

    settlements = list(settlement_dates_between(start, end))
    print(f"Planning to download {len(settlements)} cycles ({start} -> {end})")

    results = []
    for i, s in enumerate(settlements, 1):
        r = download_file(s, out_dir, session, overwrite=args.overwrite)
        results.append(r)

        tag = {"downloaded": "✓", "cached": "-", "not_found": "x"}.get(r["status"], "?")
        print(f"[{i}/{len(settlements)}] {tag} settle={s} status={r['status']}")

        if r["status"] == "downloaded":
            time.sleep(args.sleep)

    summary = pd.DataFrame(results)

    print("\nSummary:")
    print(summary["status"].value_counts().to_string())

    summary.to_csv(out_dir / "_download_log.csv", index=False)
    print(f"\nLog written to: {out_dir / '_download_log.csv'}")
    print(f"Files saved in: {out_dir}")

    not_found = summary[summary["status"] == "not_found"]
    if not not_found.empty:
        print(f"\n{len(not_found)} cycles not found. Run with an earlier --start year if older files are missing.")

if __name__ == "__main__":
    main()
