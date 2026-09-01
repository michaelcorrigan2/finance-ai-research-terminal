from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

import pandas as pd


# The definitions below keep the calculations and display rules in one place.
# Ratios and percentages are stored as decimals (0.25 means 25%).
METRIC_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "category": "Valuation",
        "label": "Trailing P/E",
        "key": "trailing_pe",
        "format": "multiple",
        "direction": "lower",
        "positive_only": True,
    },
    {
        "category": "Valuation",
        "label": "Forward P/E",
        "key": "forward_pe",
        "format": "multiple",
        "direction": "lower",
        "positive_only": True,
    },
    {
        "category": "Valuation",
        "label": "PEG Ratio",
        "key": "peg_ratio",
        "format": "multiple",
        "direction": "lower",
        "positive_only": True,
    },
    {
        "category": "Valuation",
        "label": "Price / Sales",
        "key": "price_to_sales",
        "format": "multiple",
        "direction": "lower",
        "positive_only": True,
    },
    {
        "category": "Valuation",
        "label": "EV / EBITDA",
        "key": "enterprise_to_ebitda",
        "format": "multiple",
        "direction": "lower",
        "positive_only": True,
    },
    {
        "category": "Growth",
        "label": "Revenue Growth",
        "key": "revenue_growth",
        "format": "percent",
        "direction": "higher",
    },
    {
        "category": "Growth",
        "label": "Earnings Growth",
        "key": "earnings_growth",
        "format": "percent",
        "direction": "higher",
    },
    {
        "category": "Profitability",
        "label": "Gross Margin",
        "key": "gross_margin",
        "format": "percent",
        "direction": "higher",
    },
    {
        "category": "Profitability",
        "label": "Operating Margin",
        "key": "operating_margin",
        "format": "percent",
        "direction": "higher",
    },
    {
        "category": "Profitability",
        "label": "Net Profit Margin",
        "key": "profit_margin",
        "format": "percent",
        "direction": "higher",
    },
    {
        "category": "Profitability",
        "label": "Return on Equity",
        "key": "return_on_equity",
        "format": "percent",
        "direction": "higher",
    },
    {
        "category": "Profitability",
        "label": "Return on Assets",
        "key": "return_on_assets",
        "format": "percent",
        "direction": "higher",
    },
    {
        "category": "Financial Strength",
        "label": "Debt / Equity",
        "key": "debt_to_equity",
        "format": "multiple",
        "direction": "lower",
        "nonnegative_only": True,
    },
    {
        "category": "Financial Strength",
        "label": "Current Ratio",
        "key": "current_ratio",
        "format": "multiple",
        "direction": "higher",
    },
    {
        "category": "Financial Strength",
        "label": "Quick Ratio",
        "key": "quick_ratio",
        "format": "multiple",
        "direction": "higher",
    },
    {
        "category": "Financial Strength",
        "label": "Free Cash Flow Margin",
        "key": "free_cash_flow_margin",
        "format": "percent",
        "direction": "higher",
    },
    {
        "category": "Financial Strength",
        "label": "Net Debt / EBITDA",
        "key": "net_debt_to_ebitda",
        "format": "multiple",
        "direction": "lower",
    },
    {
        "category": "Market Performance",
        "label": "1-Month Return",
        "key": "one_month_return",
        "format": "percent",
        "direction": "higher",
        "source": "returns",
    },
    {
        "category": "Market Performance",
        "label": "YTD Return",
        "key": "ytd_return",
        "format": "percent",
        "direction": "higher",
        "source": "returns",
    },
    {
        "category": "Market Performance",
        "label": "1-Year Return",
        "key": "one_year_return",
        "format": "percent",
        "direction": "higher",
        "source": "returns",
    },
    {
        "category": "Market Performance",
        "label": "3-Year Annualized Return",
        "key": "three_year_annualized_return",
        "format": "percent",
        "direction": "higher",
        "source": "returns",
    },
)

CATEGORY_ORDER = (
    "Valuation",
    "Growth",
    "Profitability",
    "Financial Strength",
    "Market Performance",
)

CATEGORY_WEIGHTS = {
    "Valuation": 0.25,
    "Growth": 0.25,
    "Profitability": 0.25,
    "Financial Strength": 0.15,
    "Market Performance": 0.10,
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _ticker(company: Mapping[str, Any]) -> str:
    return str(company.get("ticker") or company.get("symbol") or "N/A").upper()


def _derived_company_data(company: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(company)
    revenue = _number(data.get("revenue"))
    free_cash_flow = _number(data.get("free_cash_flow"))
    total_debt = _number(data.get("total_debt"))
    total_cash = _number(data.get("total_cash"))
    ebitda = _number(data.get("ebitda"))

    data["free_cash_flow_margin"] = (
        free_cash_flow / revenue
        if free_cash_flow is not None and revenue not in (None, 0)
        else None
    )
    data["net_debt_to_ebitda"] = (
        ((total_debt or 0.0) - (total_cash or 0.0)) / ebitda
        if ebitda not in (None, 0) and (total_debt is not None or total_cash is not None)
        else None
    )
    return data


def _normalize_companies(
    companies: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    scores: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], Mapping[str, Any] | None]:
    """Accept both the new list input and the legacy two-dictionary call."""
    if isinstance(companies, Mapping):
        if isinstance(scores, Mapping) and ("ticker" in scores or "symbol" in scores):
            return [dict(companies), dict(scores)], None
        return [dict(companies)], scores

    normalized = [dict(company) for company in companies if isinstance(company, Mapping)]
    return normalized, scores


def _metric_value(
    definition: Mapping[str, Any],
    ticker: str,
    company: Mapping[str, Any],
    returns: Mapping[str, Mapping[str, Any]],
) -> float | None:
    source = definition.get("source", "company")
    if source == "returns":
        value = returns.get(ticker, {}).get(str(definition["key"]))
    else:
        value = company.get(str(definition["key"]))

    converted = _number(value)
    if converted is not None and definition.get("positive_only") and converted <= 0:
        return None
    if converted is not None and definition.get("nonnegative_only") and converted < 0:
        return None
    return converted


def _rank_scores(
    values: Mapping[str, float | None],
    direction: str,
) -> tuple[dict[str, float | None], str | None]:
    available = [(ticker, value) for ticker, value in values.items() if value is not None]
    if not available:
        return {ticker: None for ticker in values}, None

    reverse = direction == "higher"
    ordered = sorted(available, key=lambda item: item[1], reverse=reverse)
    best_ticker = ordered[0][0]

    ordered_unique_values = sorted(
        {value for _, value in available},
        reverse=reverse,
    )
    if len(ordered_unique_values) == 1:
        return {ticker: (50.0 if value is not None else None) for ticker, value in values.items()}, best_ticker

    scores: dict[str, float | None] = {ticker: None for ticker in values}
    denominator = len(ordered_unique_values) - 1
    value_scores = {
        value: 100.0 * (denominator - position) / denominator
        for position, value in enumerate(ordered_unique_values)
    }
    for ticker, value in available:
        scores[ticker] = value_scores[value]
    return scores, best_ticker


def compare_companies(
    companies: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    scores: Mapping[str, Any] | None = None,
    returns: Mapping[str, Mapping[str, Any]] | None = None,
    benchmark_ticker: str = "SPY",
    benchmark_returns: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a peer-relative comparison for two to five public companies."""
    company_list, scores = _normalize_companies(companies, scores)
    if len(company_list) < 2:
        raise ValueError("At least two companies are required for comparison.")
    if len(company_list) > 5:
        raise ValueError("A maximum of five companies can be compared at once.")

    tickers = [_ticker(company) for company in company_list]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Each company ticker must be unique.")

    company_map = {
        ticker: _derived_company_data(company)
        for ticker, company in zip(tickers, company_list)
    }
    return_map = dict(returns or {})
    benchmark_return_map = dict(benchmark_returns or {})

    metric_rows: list[dict[str, Any]] = []
    category_metric_scores: dict[str, dict[str, list[float]]] = {
        category: {ticker: [] for ticker in tickers}
        for category in CATEGORY_ORDER
    }

    for definition in METRIC_DEFINITIONS:
        values = {
            ticker: _metric_value(definition, ticker, company_map[ticker], return_map)
            for ticker in tickers
        }
        valid_values = [value for value in values.values() if value is not None]
        peer_median = median(valid_values) if valid_values else None
        rank_scores, best_ticker = _rank_scores(values, str(definition["direction"]))

        for ticker, rank_score in rank_scores.items():
            if rank_score is not None:
                category_metric_scores[str(definition["category"])][ticker].append(rank_score)

        metric_rows.append(
            {
                **definition,
                "values": values,
                "peer_median": peer_median,
                "benchmark_value": (
                    _number(benchmark_return_map.get(str(definition["key"])))
                    if definition.get("source") == "returns"
                    else None
                ),
                "rank_scores": rank_scores,
                "best_ticker": best_ticker,
            }
        )

    category_scores: dict[str, dict[str, float | None]] = {}
    for category in CATEGORY_ORDER:
        category_scores[category] = {}
        for ticker in tickers:
            values = category_metric_scores[category][ticker]
            category_scores[category][ticker] = (
                round(sum(values) / len(values), 1) if values else None
            )

    overall_scores: dict[str, float | None] = {}
    for ticker in tickers:
        weighted_total = 0.0
        available_weight = 0.0
        for category, weight in CATEGORY_WEIGHTS.items():
            value = category_scores[category].get(ticker)
            if value is not None:
                weighted_total += value * weight
                available_weight += weight
        overall_scores[ticker] = (
            round(weighted_total / available_weight, 1)
            if available_weight
            else None
        )

    category_scores["Overall"] = overall_scores

    def leader_for(score_map: Mapping[str, float | None]) -> str | None:
        available = [(ticker, value) for ticker, value in score_map.items() if value is not None]
        return max(available, key=lambda item: (item[1], item[0]))[0] if available else None

    quality_scores: dict[str, float | None] = {}
    for ticker in tickers:
        quality_values = [
            category_scores[category][ticker]
            for category in ("Profitability", "Financial Strength")
            if category_scores[category][ticker] is not None
        ]
        quality_scores[ticker] = (
            sum(quality_values) / len(quality_values) if quality_values else None
        )

    leaders = {
        "Best Overall": leader_for(overall_scores),
        "Best Value": leader_for(category_scores["Valuation"]),
        "Best Growth": leader_for(category_scores["Growth"]),
        "Best Quality": leader_for(quality_scores),
        "Best Momentum": leader_for(category_scores["Market Performance"]),
    }

    return {
        "tickers": tickers,
        "companies": company_map,
        "financial_scores": dict(scores or {}),
        "returns": return_map,
        "benchmark": {
            "ticker": benchmark_ticker.upper(),
            "returns": benchmark_return_map,
        },
        "metrics": metric_rows,
        "category_scores": category_scores,
        "leaders": leaders,
    }


def _format_value(value: Any, value_format: str) -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    if value_format == "percent":
        return f"{number * 100:.1f}%"
    if value_format == "multiple":
        return f"{number:.2f}x"
    if value_format == "large_currency":
        absolute = abs(number)
        if absolute >= 1_000_000_000_000:
            return f"${number / 1_000_000_000_000:.2f}T"
        if absolute >= 1_000_000_000:
            return f"${number / 1_000_000_000:.2f}B"
        if absolute >= 1_000_000:
            return f"${number / 1_000_000:.2f}M"
        return f"${number:,.0f}"
    return f"{number:,.2f}"


def comparison_to_dataframe(
    comparison: Mapping[str, Any],
    category: str | None = None,
    formatted: bool = False,
) -> pd.DataFrame:
    """Convert all or one category of comparison metrics to a DataFrame."""
    if not comparison:
        return pd.DataFrame()

    tickers = list(comparison.get("tickers", []))
    benchmark = comparison.get("benchmark", {})
    benchmark_ticker = str(benchmark.get("ticker") or "Benchmark")
    rows: list[dict[str, Any]] = []

    for metric in comparison.get("metrics", []):
        if category and metric.get("category") != category:
            continue

        value_format = str(metric.get("format", "number"))
        row: dict[str, Any] = {
            "Category": metric.get("category"),
            "Metric": metric.get("label"),
        }
        for ticker in tickers:
            value = metric.get("values", {}).get(ticker)
            row[ticker] = _format_value(value, value_format) if formatted else value

        peer_median = metric.get("peer_median")
        row["Peer Median"] = (
            _format_value(peer_median, value_format) if formatted else peer_median
        )

        if metric.get("category") == "Market Performance":
            benchmark_value = metric.get("benchmark_value")
            row[benchmark_ticker] = (
                _format_value(benchmark_value, value_format)
                if formatted
                else benchmark_value
            )
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    if category and "Category" in dataframe.columns:
        dataframe = dataframe.drop(columns="Category")
    return dataframe


def category_scores_to_dataframe(comparison: Mapping[str, Any]) -> pd.DataFrame:
    if not comparison:
        return pd.DataFrame()
    tickers = list(comparison.get("tickers", []))
    categories = ("Overall",) + CATEGORY_ORDER
    return pd.DataFrame(
        [
            {
                "Category": category,
                **{
                    ticker: comparison.get("category_scores", {}).get(category, {}).get(ticker)
                    for ticker in tickers
                },
            }
            for category in categories
        ]
    )


def valuation_premium_to_median_dataframe(
    comparison: Mapping[str, Any],
) -> pd.DataFrame:
    """Show each valuation multiple's premium or discount to the peer median."""
    rows: list[dict[str, Any]] = []
    tickers = list(comparison.get("tickers", []))
    for metric in comparison.get("metrics", []):
        if metric.get("category") != "Valuation":
            continue
        peer_median = _number(metric.get("peer_median"))
        if peer_median in (None, 0):
            continue
        valid_values = {
            ticker: _number(metric.get("values", {}).get(ticker))
            for ticker in tickers
        }
        ordered = sorted(
            ((ticker, value) for ticker, value in valid_values.items() if value is not None),
            key=lambda item: item[1],
        )
        ranks = {ticker: rank for rank, (ticker, _) in enumerate(ordered, start=1)}
        for ticker in tickers:
            value = valid_values[ticker]
            if value is None:
                continue
            rows.append(
                {
                    "Metric": metric.get("label"),
                    "Ticker": ticker,
                    "Multiple": f"{value:.2f}x",
                    "Peer Median": f"{peer_median:.2f}x",
                    "Premium / (Discount)": f"{(value / peer_median - 1) * 100:+.1f}%",
                    "Value Rank": ranks.get(ticker),
                }
            )
    return pd.DataFrame(rows)


def legacy_pair_comparison(
    first_company: Mapping[str, Any],
    second_company: Mapping[str, Any],
) -> dict[str, Any]:
    """Explicit compatibility helper for older code."""
    return compare_companies([first_company, second_company])
