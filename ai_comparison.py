from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found. Add it to your .env file or Streamlit Secrets.")
    return api_key


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format(value: Any, kind: str = "number") -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    if kind == "percent":
        return f"{number * 100:.1f}%"
    if kind == "multiple":
        return f"{number:.2f}x"
    if kind == "currency":
        absolute = abs(number)
        if absolute >= 1_000_000_000_000:
            return f"${number / 1_000_000_000_000:.2f} trillion"
        if absolute >= 1_000_000_000:
            return f"${number / 1_000_000_000:.2f} billion"
        if absolute >= 1_000_000:
            return f"${number / 1_000_000:.2f} million"
        return f"${number:,.0f}"
    return f"{number:,.2f}"


def _normalize_companies(
    companies: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    comparison_result: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], Mapping[str, Any] | None]:
    if isinstance(companies, Mapping):
        if isinstance(comparison_result, Mapping) and (
            "ticker" in comparison_result or "symbol" in comparison_result
        ):
            return [dict(companies), dict(comparison_result)], None
        return [dict(companies)], comparison_result
    return [dict(company) for company in companies if isinstance(company, Mapping)], comparison_result


def _company_overview(company: Mapping[str, Any]) -> str:
    ticker = str(company.get("ticker") or company.get("symbol") or "N/A").upper()
    name = str(company.get("company_name") or company.get("name") or ticker)
    summary = str(company.get("business_summary") or "Not available").strip()
    if len(summary) > 500:
        summary = summary[:497].rstrip() + "..."
    return "\n".join((
        f"- {ticker}: {name}",
        f"  Sector / industry: {company.get('sector') or 'N/A'} / {company.get('industry') or 'N/A'}",
        f"  Market cap: {_format(company.get('market_cap'), 'currency')}",
        f"  Revenue: {_format(company.get('revenue'), 'currency')}",
        f"  Business: {summary}",
    ))


def _comparison_summary(comparison_result: Mapping[str, Any]) -> str:
    tickers = list(comparison_result.get("tickers", []))
    lines: list[str] = []
    leaders = comparison_result.get("leaders", {})
    if leaders:
        lines.append("APPROVED PEER-RELATIVE LEADERS:")
        lines.extend(f"- {label}: {ticker or 'N/A'}" for label, ticker in leaders.items())

    lines.append("\nAPPROVED METRIC FACTS:")
    for metric in comparison_result.get("metrics", []):
        values = metric.get("values", {})
        kind = str(metric.get("format", "number"))
        values_text = ", ".join(f"{ticker}={_format(values.get(ticker), kind)}" for ticker in tickers)
        benchmark = metric.get("benchmark_value")
        benchmark_text = ""
        if benchmark is not None:
            benchmark_ticker = comparison_result.get("benchmark", {}).get("ticker", "Benchmark")
            benchmark_text = f", {benchmark_ticker}={_format(benchmark, kind)}"
        lines.append(
            f"- {metric.get('category')} | {metric.get('label')}: "
            f"{values_text}; peer median={_format(metric.get('peer_median'), kind)}{benchmark_text}"
        )
    return "\n".join(lines)


def _write_prompt(company_context: str, fact_sheet: str) -> str:
    return f"""
You are an equity research analyst. Write a concise comparative-company report.

COMPANIES:
{company_context}

CALCULATED FACT SHEET (the only source for rankings and numerical claims):
{fact_sheet}

Rules:
- Use only the fact sheet for all numeric and relative claims.
- Never write a self-correction, such as "actually," "correction," or "despite the prior statement."
- A multiple below the peer median is a discount; above is a premium. State it correctly.
- "Recent momentum" means only the 1-month return.
- Mention YTD, 1-year, and 3-year annualized performance separately.
- Do not make company-specific claims about geography, pricing power, product demand, competitors, events, or risks unless directly stated in the company description above.
- If evidence is missing, say "data not available." 
- Do not call a stock cheap, expensive, best, or weakest unless the fact sheet supports it.

Use these headings exactly:
## Executive Ranking
## Business and Peer Context
## Valuation
## Growth and Profitability
## Financial Strength and Cash Flow
## Market Performance vs Benchmark
## Key Risks and Data Gaps
## Investment Conclusion

Use bullets where useful. Keep the report under 900 words.
"""


def _review_prompt(draft: str, fact_sheet: str) -> str:
    return f"""
You are the fact-checking editor for an equity-research report.

APPROVED FACT SHEET:
{fact_sheet}

DRAFT REPORT:
{draft}

Return a corrected, publication-ready version of the full report only.

Mandatory checks:
1. Delete every contradiction and every self-correction.
2. Verify every ranking, premium/discount, leader, percentage, and multiple against the fact sheet.
3. A value below its peer median must be called a discount, never a premium.
4. Recent momentum must agree with the highest 1-month return.
5. Do not add unsupported company-specific risks or claims.
6. Retain the exact eight headings used in the draft.
"""


def compare_companies_with_ai(
    companies: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    comparison_result: Mapping[str, Any] | None = None,
    *_: Any,
    **__: Any,
) -> str:
    company_list, comparison_result = _normalize_companies(companies, comparison_result)
    if len(company_list) < 2:
        raise ValueError("At least two companies are required for AI comparison.")
    if not comparison_result:
        raise ValueError("Calculated comparison data is required for the AI comparison.")

    company_context = "\n".join(_company_overview(company) for company in company_list)
    fact_sheet = _comparison_summary(comparison_result)
    client = OpenAI(api_key=_get_api_key())

    draft_response = client.responses.create(
        model="gpt-4.1-mini",
        input=_write_prompt(company_context, fact_sheet),
        temperature=0.1,
    )
    draft = getattr(draft_response, "output_text", "")
    if not draft:
        raise ValueError("The AI comparison returned no text.")

    reviewed_response = client.responses.create(
        model="gpt-4.1-mini",
        input=_review_prompt(draft, fact_sheet),
        temperature=0,
    )
    report = getattr(reviewed_response, "output_text", "")
    if not report:
        raise ValueError("The AI fact-checking pass returned no text.")
    return report.strip()


def generate_ai_comparison(companies: Sequence[Mapping[str, Any]] | Mapping[str, Any], comparison_result: Mapping[str, Any] | None = None, *_: Any, **__: Any) -> str:
    return compare_companies_with_ai(companies, comparison_result)


def compare_companies_ai(companies: Sequence[Mapping[str, Any]] | Mapping[str, Any], comparison_result: Mapping[str, Any] | None = None, *_: Any, **__: Any) -> str:
    return compare_companies_with_ai(companies, comparison_result)


def ai_compare_companies(companies: Sequence[Mapping[str, Any]] | Mapping[str, Any], comparison_result: Mapping[str, Any] | None = None, *_: Any, **__: Any) -> str:
    return compare_companies_with_ai(companies, comparison_result)
