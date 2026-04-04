"""Build annual AI sentiment + spillover dataset into SQLite."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from finhack.annual_intelligence import AnnualIntelligenceBuilder

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build annual AI intelligence dataset")
    parser.add_argument("--years-back", type=int, default=5)
    parser.add_argument("--pre-days", type=int, default=30)
    parser.add_argument("--post-days", type=int, default=20)
    parser.add_argument("--max-news-per-event", type=int, default=25)
    args = parser.parse_args()

    builder = AnnualIntelligenceBuilder()
    summary = builder.build(
        years_back=max(2, min(args.years_back, 12)),
        pre_days=max(7, min(args.pre_days, 120)),
        post_days=max(5, min(args.post_days, 120)),
        max_news_per_event=max(5, min(args.max_news_per_event, 100)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
