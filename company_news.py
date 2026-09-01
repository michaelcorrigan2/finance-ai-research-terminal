from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yfinance as yf


def _get_value(item: dict[str, Any], key: str, default: Any = None) -> Any:
    """Support both older and newer yfinance news response formats."""
    content = item.get("content", {})
    return content.get(key, item.get(key, default))


def _format_date(timestamp: Any) -> str:
    """Convert Yahoo's timestamp into a readable date/time."""
    if not timestamp:
        return "Date unavailable"

    try:
        if isinstance(timestamp, str):
            return timestamp.replace("T", " ").replace("Z", "")[:16]

        return datetime.fromtimestamp(
            int(timestamp),
            tz=timezone.utc,
        ).strftime("%b %d, %Y %I:%M %p UTC")
    except (TypeError, ValueError, OSError):
        return "Date unavailable"


def get_company_news(
    ticker: str,
    company_name: str = "",
    max_articles: int = 8,
) -> list[dict[str, Any]]:
    """
    Return recent company news in a consistent format for app.py and ai_news.py.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        return []

    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    articles = []
    seen_urls = set()

    for item in raw_news:
        title = _get_value(item, "title", "")
        link = _get_value(item, "clickThroughUrl", {})

        if isinstance(link, dict):
            link = link.get("url", "")

        if not link:
            canonical_url = _get_value(item, "canonicalUrl", {})
            if isinstance(canonical_url, dict):
                link = canonical_url.get("url", "")

        publisher = _get_value(item, "provider", "")
        if isinstance(publisher, dict):
            publisher = publisher.get("displayName", "")

        summary = _get_value(item, "summary", "")
        published_at = _get_value(item, "pubDate", None)

        if not published_at:
            published_at = _get_value(item, "providerPublishTime", None)

        thumbnail = _get_value(item, "thumbnail", {})
        image_url = ""

        if isinstance(thumbnail, dict):
            resolutions = thumbnail.get("resolutions", [])
            if resolutions:
                image_url = resolutions[0].get("url", "")

        if not title or not link or link in seen_urls:
            continue

        seen_urls.add(link)

        articles.append(
            {
                "ticker": ticker,
                "company_name": company_name or ticker,
                "title": title,
                "publisher": publisher or "Yahoo Finance",
                "summary": summary or "No summary available.",
                "link": link,
                "published_at": _format_date(published_at),
                "image_url": image_url,
            }
        )

        if len(articles) >= max_articles:
            break

    return articles