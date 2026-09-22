"""Download the HMDA Illinois 2024 home-purchase records from the CFPB Data Browser.

The CFPB endpoint rejects the default requests user-agent with HTTP 403, so a
browser user-agent header is required.

Usage:
    python src/download_hmda.py [--out data/hmda_illinois_2024.csv]
"""

import argparse
import os
import sys

import requests

URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
PARAMS = {"years": "2024", "states": "IL", "loan_purposes": "1"}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hmda_illinois_2024.csv")
    args = ap.parse_args()

    if os.path.exists(args.out) and os.path.getsize(args.out) > 1_000_000:
        print(f"Already present: {args.out} "
              f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Proxies are picked up from http_proxy/https_proxy if set, which is needed
    # on HPC compute nodes without direct outbound access.
    proxies = {k: os.environ.get(f"{k}_proxy") for k in ("http", "https")}
    proxies = {k: v for k, v in proxies.items() if v}

    print(f"Downloading {URL} with {PARAMS} ...")
    resp = requests.get(URL, params=PARAMS, headers=HEADERS,
                        proxies=proxies or None, stream=True, timeout=300)

    if resp.status_code != 200:
        print(f"Request failed with status {resp.status_code}", file=sys.stderr)
        print(resp.text[:500], file=sys.stderr)
        sys.exit(1)

    with open(args.out, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out} ({size:.1f} MB)")
    if size < 1:
        print("Warning: file is unexpectedly small; it may be an error page.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
