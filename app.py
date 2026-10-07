from __future__ import annotations
import re

import inspect
import importlib
import math
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

from pdf_report import generate_research_report_pdf
st.set_page_config(
    page_title="Finance AI Research Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# MODULE LOADING
# =============================================================================

MODULE_NAMES = (
    "ai_analysis",
    "ai_news",
    "investment_recommendation",
    "comparison",
    "ai_comparison",
    "dcf_model",
    "dcf_scenarios",
    "dcf_sensitivity",
    "financial_score",
    "market_data",
    "company_news",
)


@st.cache_resource(show_spinner=False)
def load_project_modules() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for module_name in MODULE_NAMES:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except Exception as exc:
            modules[module_name] = None
            errors[module_name] = f"{type(exc).__name__}: {exc}"

    modules["_errors"] = errors
    return modules


MODULES = load_project_modules()


# =============================================================================
# GENERIC HELPERS
# =============================================================================

Number = Optional[Union[int, float]]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_ticker(value: str) -> str:
    return "".join(value.strip().upper().split())


def first_present(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def format_currency(value: Number, decimals: int = 2) -> str:
    if not is_number(value):
        return "N/A"
    return f"${float(value):,.{decimals}f}"


def format_large_number(value: Number) -> str:
    if not is_number(value):
        return "N/A"

    number = float(value)
    absolute = abs(number)

    if absolute >= 1_000_000_000_000:
        return f"${number / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"${number / 1_000:.2f}K"
    return f"${number:,.2f}"


def format_number(value: Number, decimals: int = 2) -> str:
    if not is_number(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def format_multiple(value: Number) -> str:
    if not is_number(value):
        return "N/A"
    return f"{float(value):.2f}x"


def normalize_percent(value: Number) -> Optional[float]:
    """Convert decimal ratios such as 0.25 into 25.00 for display."""
    if not is_number(value):
        return None

    number = float(value)
    if abs(number) <= 1.5:
        return number * 100
    return number


def format_percent(value: Number) -> str:
    normalized = normalize_percent(value)
    if normalized is None:
        return "N/A"
    return f"{normalized:.2f}%"


def format_dividend_yield(value: Number) -> str:
    """Handle Yahoo values returned as either 0.0048 or 0.48 for 0.48%."""
    if not is_number(value):
        return "N/A"

    number = float(value)
    if abs(number) > 0.20:
        number /= 100

    return f"{number * 100:.2f}%"


def format_debt_to_equity(value: Number) -> str:
    """Handle Yahoo debtToEquity values such as 6.55, meaning 0.0655x."""
    if not is_number(value):
        return "N/A"

    number = float(value)
    if abs(number) > 1:
        number /= 100

    return f"{number:.2f}x"


def calculate_upside(fair_value: Number, current_price: Number) -> Optional[float]:
    if not is_number(fair_value) or not is_number(current_price):
        return None
    if float(current_price) <= 0:
        return None
    return (float(fair_value) / float(current_price) - 1) * 100


def get_callable(module_name: str, candidates: Sequence[str]) -> Optional[Callable[..., Any]]:
    module = MODULES.get(module_name)
    if module is None:
        return None

    for name in candidates:
        function = getattr(module, name, None)
        if callable(function):
            return function

    public_functions = [
        member
        for name, member in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and member.__module__ == module.__name__
    ]

    if len(public_functions) == 1:
        return public_functions[0]

    return None


def call_compatible(
    function: Callable[..., Any],
    context: Mapping[str, Any],
    positional_attempts: Sequence[Sequence[Any]] = (),
) -> Any:
    """
    Call a project function while accommodating common parameter-name variations.
    The function's real signature is inspected first. If that cannot fully resolve
    the call, several explicit positional forms are attempted.
    """
    signature = inspect.signature(function)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    unresolved_required: list[str] = []

    aliases: dict[str, tuple[str, ...]] = {
        "ticker": ("ticker", "ticker_symbol", "symbol"),
        "ticker_symbol": ("ticker_symbol", "ticker", "symbol"),
        "symbol": ("symbol", "ticker", "ticker_symbol"),
        "data": ("data", "stock_data", "company_data", "financial_data"),
        "stock_data": ("stock_data", "data", "company_data", "financial_data"),
        "company_data": ("company_data", "stock_data", "data", "financial_data"),
        "financial_data": ("financial_data", "stock_data", "company_data", "data"),
        "scores": ("scores", "financial_scores", "score"),
        "financial_scores": ("financial_scores", "scores", "score"),
        "news": ("news", "news_items", "articles", "company_news"),
        "news_items": ("news_items", "news", "articles", "company_news"),
        "articles": ("articles", "news", "news_items", "company_news"),
        "company_news": ("company_news", "news", "news_items", "articles"),
        "company_1": ("company_1", "first_company", "company1", "data_1", "first_data"),
        "company_2": ("company_2", "second_company", "company2", "data_2", "second_data"),
        "first_company": ("first_company", "company_1", "company1", "data_1", "first_data"),
        "second_company": ("second_company", "company_2", "company2", "data_2", "second_data"),
        "first_scores": ("first_scores", "scores_1", "company_1_scores"),
        "second_scores": ("second_scores", "scores_2", "company_2_scores"),
        "growth_rate": ("growth_rate", "fcf_growth_rate", "annual_growth_rate"),
        "fcf_growth_rate": ("fcf_growth_rate", "growth_rate", "annual_growth_rate"),
        "discount_rate": ("discount_rate", "wacc", "required_return"),
        "terminal_growth_rate": ("terminal_growth_rate", "terminal_rate", "perpetual_growth_rate"),
        "years": ("years", "projection_years", "forecast_years"),
        "projection_years": ("projection_years", "years", "forecast_years"),
        "base_growth_rate": ("base_growth_rate", "growth_rate", "fcf_growth_rate"),
        "base_discount_rate": ("base_discount_rate", "discount_rate", "wacc"),
    }

    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        lookup_names = aliases.get(parameter.name, (parameter.name,))
        matched = False

        for lookup_name in lookup_names:
            if lookup_name in context:
                value = context[lookup_name]
                if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                    args.append(value)
                else:
                    kwargs[parameter.name] = value
                matched = True
                break

        if not matched and parameter.default is inspect.Parameter.empty:
            unresolved_required.append(parameter.name)

    if not unresolved_required:
        return function(*args, **kwargs)

    last_error: Optional[Exception] = None
    for attempt in positional_attempts:
        try:
            return function(*attempt)
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise TypeError(
        f"Could not resolve required parameters for {function.__module__}."
        f"{function.__name__}: {', '.join(unresolved_required)}"
    )


def module_status_table() -> pd.DataFrame:
    errors = MODULES.get("_errors", {})
    rows = []

    for name in MODULE_NAMES:
        available = MODULES.get(name) is not None
        rows.append(
            {
                "Module": name,
                "Status": "Loaded" if available else "Unavailable",
                "Details": "" if available else errors.get(name, "Import failed"),
            }
        )

    return pd.DataFrame(rows)


def render_structured_markdown(report: str) -> None:
    """
    Render an AI-generated Markdown report as professional research cards.
    """
    import re

    cleaned_report = report.strip()

    if not cleaned_report:
        st.info("The module completed but returned no written analysis.")
        return

    icon_map = {
        "executive news summary": "📰",
        "executive summary": "📋",
        "investment summary": "📋",
        "investment thesis": "🎯",
        "business overview": "🏢",
        "company overview": "🏢",
        "financial strengths": "💪",
        "financial strength": "💪",
        "financial quality": "💪",
        "financial relevance": "💵",
        "profitability": "📊",
        "balance sheet": "🏦",
        "cash flow": "💵",
        "valuation assessment": "💰",
        "valuation": "💰",
        "bull case": "📈",
        "bear case": "📉",
        "negative catalysts and risks": "⚠️",
        "negative catalysts": "⚠️",
        "positive catalysts": "🚀",
        "growth catalysts": "🚀",
        "growth drivers": "🚀",
        "key risks": "⚠️",
        "risks": "⚠️",
        "expected market impact": "📉",
        "market impact": "📉",
        "what investors should watch": "👀",
        "investor takeaway": "🎯",
        "investment outlook": "🔭",
        "overall assessment": "🎯",
        "bottom line": "✅",
        "key takeaways": "🔑",
        "recommendation": "🧭",
        "overall news sentiment": "📊",
        "news sentiment": "📊",
        "sentiment": "📊",
        "comparison summary": "⚖️",
        "catalysts": "🚀",
    }

    heading_pattern = re.compile(
        r"(?m)^##\s+(.+?)\s*$"
    )
    matches = list(heading_pattern.finditer(cleaned_report))

    if not matches:
        with st.container(border=True):
            st.markdown(cleaned_report)
        return

    preamble = cleaned_report[:matches[0].start()].strip()

    if preamble:
        with st.container(border=True):
            st.markdown(preamble)

    for index, match in enumerate(matches):
        heading = match.group(1).strip().replace("**", "")
        content_start = match.end()
        content_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(cleaned_report)
        )
        section_content = cleaned_report[
            content_start:content_end
        ].strip()

        normalized_heading = re.sub(
            r"[^a-z0-9 ]",
            "",
            heading.lower(),
        ).strip()

        icon = icon_map.get(normalized_heading, "▪️")

        if icon == "▪️":
            ordered_icon_matches = sorted(
                icon_map.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )

            for keyword, mapped_icon in ordered_icon_matches:
                if keyword in normalized_heading:
                    icon = mapped_icon
                    break

        with st.container(border=True):
            st.markdown(f"### {icon} {heading}")

            if section_content:
                st.markdown(section_content)
            else:
                st.caption(
                    "No additional detail was returned for this section."
                )


def render_text_result(result: Any) -> None:
    if result is None:
        st.info(
            "The module completed but returned no displayable result."
        )

    elif isinstance(result, str):
        cleaned_result = result.strip()

        if not cleaned_result:
            st.info(
                "The module completed but returned no written analysis."
            )
            return

        render_structured_markdown(cleaned_result)

        st.download_button(
            label="Download Report",
            data=cleaned_result.encode("utf-8"),
            file_name="finance_ai_research_report.md",
            mime="text/markdown",
            key=f"download_text_report_{abs(hash(cleaned_result))}",
        )

    elif isinstance(result, pd.DataFrame):
        st.dataframe(
            result,
            width="stretch",
        )

    elif isinstance(result, Mapping):
        st.json(dict(result))

    elif isinstance(result, (list, tuple)):
        if all(isinstance(item, Mapping) for item in result):
            st.dataframe(
                pd.DataFrame(result),
                width="stretch",
            )
        else:
            for item in result:
                st.write(item)

    else:
        st.write(result)


def render_news_cards(news_items: Any) -> None:
    """
    Render recent company news as clean, investor-focused article cards.
    """
    if not isinstance(news_items, (list, tuple)) or not news_items:
        st.info("No recent company news was available.")
        return

    valid_articles = [
        article
        for article in news_items
        if isinstance(article, Mapping)
        and str(article.get("title", "")).strip()
    ]

    if not valid_articles:
        st.info("No displayable company news articles were returned.")
        return

    st.caption(
        f"Showing {len(valid_articles)} recent article"
        f"{'' if len(valid_articles) == 1 else 's'}."
    )

    for index, article in enumerate(valid_articles, start=1):
        title = str(
            article.get("title")
            or "Untitled Article"
        ).strip()

        publisher = str(
            article.get("publisher")
            or "Unknown Publisher"
        ).strip()

        published_at = str(
            article.get("published_at")
            or "Date unavailable"
        ).strip()

        summary = str(
            article.get("summary")
            or "No summary available."
        ).strip()

        link = article.get("link")

        with st.container(border=True):
            st.markdown(f"### {title}")

            metadata_columns = st.columns(
                [1.2, 1.5, 0.7]
            )

            with metadata_columns[0]:
                st.caption(f"**Source:** {publisher}")

            with metadata_columns[1]:
                st.caption(f"**Published:** {published_at}")

            with metadata_columns[2]:
                st.caption(f"**Article {index}**")

            st.markdown(summary)

            if isinstance(link, str) and link.strip():
                st.link_button(
                    "Read Full Article",
                    link.strip(),
                    width="content",
                )


# =============================================================================
# DATA ACCESS LAYER
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_company_data(ticker: str) -> dict[str, Any]:
    function = get_callable(
        "market_data",
        (
            "get_stock_data",
            "get_market_data",
            "fetch_stock_data",
            "fetch_market_data",
            "get_company_data",
        ),
    )
    if function is None:
        raise RuntimeError(
            "market_data.py loaded, but no supported market-data function was found."
        )

    result = call_compatible(
        function,
        {"ticker": ticker, "ticker_symbol": ticker, "symbol": ticker},
        positional_attempts=((ticker,),),
    )
    data = as_mapping(result)

    if not data:
        raise ValueError(f"No market data was returned for {ticker}.")

    data.setdefault("ticker", ticker)
    data.setdefault(
        "company_name",
        first_present(data, "long_name", "short_name", "name", default=ticker),
    )
    return data


@st.cache_data(ttl=900, show_spinner=False)
def get_price_history(ticker: str, period: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, auto_adjust=False)

    if history is None or history.empty:
        return pd.DataFrame()

    history = history.copy()
    history.index = pd.to_datetime(history.index)
    return history


@st.cache_data(ttl=900, show_spinner=False)
def get_multi_period_returns(ticker: str) -> dict[str, Optional[float]]:
    """Calculate comparable market returns from a single five-year history."""
    history = get_price_history(ticker, "5y")
    if history.empty or "Close" not in history.columns:
        return {
            "one_month_return": None,
            "ytd_return": None,
            "one_year_return": None,
            "three_year_annualized_return": None,
        }

    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(close) < 2:
        return {
            "one_month_return": None,
            "ytd_return": None,
            "one_year_return": None,
            "three_year_annualized_return": None,
        }

    # yfinance may return a timezone-aware index. Removing the timezone makes
    # date comparisons consistent across tickers and deployed environments.
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    latest_date = close.index[-1]
    latest_price = float(close.iloc[-1])

    def return_since(cutoff: pd.Timestamp) -> Optional[float]:
        available = close.loc[close.index >= cutoff]
        if available.empty:
            return None
        starting_price = float(available.iloc[0])
        if starting_price == 0:
            return None
        return latest_price / starting_price - 1

    one_month_return = return_since(latest_date - pd.DateOffset(months=1))
    ytd_return = return_since(pd.Timestamp(year=latest_date.year, month=1, day=1))
    one_year_return = return_since(latest_date - pd.DateOffset(years=1))

    three_year_prices = close.loc[
        close.index >= latest_date - pd.DateOffset(years=3)
    ]
    three_year_annualized_return = None
    if len(three_year_prices) >= 2:
        starting_price = float(three_year_prices.iloc[0])
        elapsed_years = (
            three_year_prices.index[-1] - three_year_prices.index[0]
        ).days / 365.25
        if starting_price > 0 and latest_price > 0 and elapsed_years > 0:
            three_year_annualized_return = (
                latest_price / starting_price
            ) ** (1 / elapsed_years) - 1

    return {
        "one_month_return": one_month_return,
        "ytd_return": ytd_return,
        "one_year_return": one_year_return,
        "three_year_annualized_return": three_year_annualized_return,
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_financial_statements(
    ticker: str,
    frequency: str,
) -> dict[str, pd.DataFrame]:
    """
    Fetch annual or quarterly company statements with multiple yfinance
    fallbacks so the feature continues working across yfinance versions.
    """
    stock = yf.Ticker(ticker)
    freq = "quarterly" if frequency == "quarterly" else "yearly"

    def fetch_statement(
        method_name: str,
        annual_attribute: str,
        quarterly_attribute: str,
    ) -> pd.DataFrame:
        statement = pd.DataFrame()

        method = getattr(stock, method_name, None)
        if callable(method):
            try:
                statement = method(freq=freq)
            except TypeError:
                try:
                    statement = method()
                except Exception:
                    statement = pd.DataFrame()
            except Exception:
                statement = pd.DataFrame()

        if not isinstance(statement, pd.DataFrame) or statement.empty:
            attribute_name = (
                quarterly_attribute
                if frequency == "quarterly"
                else annual_attribute
            )
            try:
                statement = getattr(stock, attribute_name)
            except Exception:
                statement = pd.DataFrame()

        if not isinstance(statement, pd.DataFrame) or statement.empty:
            return pd.DataFrame()

        prepared = statement.copy()

        prepared.columns = [
            pd.to_datetime(column).strftime("%Y-%m-%d")
            if not isinstance(column, str)
            else column
            for column in prepared.columns
        ]

        prepared.index = [
            str(index)
            .replace("_", " ")
            .replace("  ", " ")
            .strip()
            for index in prepared.index
        ]

        prepared = prepared.dropna(axis=0, how="all")
        prepared = prepared.dropna(axis=1, how="all")
        return prepared

    return {
        "Income Statement": fetch_statement(
            "get_income_stmt",
            "financials",
            "quarterly_financials",
        ),
        "Balance Sheet": fetch_statement(
            "get_balance_sheet",
            "balance_sheet",
            "quarterly_balance_sheet",
        ),
        "Cash Flow Statement": fetch_statement(
            "get_cash_flow",
            "cashflow",
            "quarterly_cashflow",
        ),
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_financial_trends(ticker: str) -> pd.DataFrame:
    """Build annual financial trend data from yfinance statements."""
    stock = yf.Ticker(ticker)

    try:
        income = stock.get_income_stmt(freq="yearly")
    except Exception:
        income = getattr(stock, "financials", pd.DataFrame())

    try:
        cash_flow = stock.get_cash_flow(freq="yearly")
    except Exception:
        cash_flow = getattr(stock, "cashflow", pd.DataFrame())

    if not isinstance(income, pd.DataFrame):
        income = pd.DataFrame()

    if not isinstance(cash_flow, pd.DataFrame):
        cash_flow = pd.DataFrame()

    def row_values(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype="float64")

        for candidate in candidates:
            if candidate in frame.index:
                values = pd.to_numeric(frame.loc[candidate], errors="coerce")
                values.index = pd.to_datetime(values.index)
                return values

        normalized_index = {
            str(index).replace(" ", "").replace("_", "").lower(): index
            for index in frame.index
        }

        for candidate in candidates:
            normalized_candidate = (
                candidate.replace(" ", "").replace("_", "").lower()
            )
            matched = normalized_index.get(normalized_candidate)
            if matched is not None:
                values = pd.to_numeric(frame.loc[matched], errors="coerce")
                values.index = pd.to_datetime(values.index)
                return values

        return pd.Series(dtype="float64")

    revenue = row_values(
        income,
        ("TotalRevenue", "OperatingRevenue"),
    )
    net_income = row_values(
        income,
        ("NetIncome", "NetIncomeCommonStockholders"),
    )
    ebitda = row_values(
        income,
        ("EBITDA", "NormalizedEBITDA"),
    )
    diluted_eps = row_values(
        income,
        ("DilutedEPS", "BasicEPS"),
    )
    gross_profit = row_values(
        income,
        ("GrossProfit",),
    )
    operating_income = row_values(
        income,
        ("OperatingIncome", "TotalOperatingIncomeAsReported"),
    )
    free_cash_flow = row_values(
        cash_flow,
        ("FreeCashFlow",),
    )

    all_dates = sorted(
        set(revenue.index)
        | set(net_income.index)
        | set(ebitda.index)
        | set(diluted_eps.index)
        | set(gross_profit.index)
        | set(operating_income.index)
        | set(free_cash_flow.index)
    )

    if not all_dates:
        return pd.DataFrame()

    trends = pd.DataFrame(index=all_dates)
    trends.index.name = "Date"

    trends["Revenue"] = revenue.reindex(all_dates)
    trends["Net Income"] = net_income.reindex(all_dates)
    trends["EBITDA"] = ebitda.reindex(all_dates)
    trends["Diluted EPS"] = diluted_eps.reindex(all_dates)
    trends["Free Cash Flow"] = free_cash_flow.reindex(all_dates)

    revenue_safe = trends["Revenue"].replace(0, pd.NA)
    trends["Gross Margin"] = (
        gross_profit.reindex(all_dates) / revenue_safe * 100
    )
    trends["Operating Margin"] = (
        operating_income.reindex(all_dates) / revenue_safe * 100
    )

    trends = trends.sort_index()

    # Remove years where all displayed trend metrics are missing
    trends = trends.dropna(
        subset=[
            "Revenue",
            "Net Income",
            "EBITDA",
            "Diluted EPS",
            "Free Cash Flow",
        ],
        how="all",
    )

    return trends


@st.cache_data(ttl=900, show_spinner=False)
def get_analyst_estimates(ticker: str) -> dict[str, Any]:
    """
    Fetch analyst recommendations, price targets, and forward estimates
    using yfinance with version-compatible fallbacks.
    """
    stock = yf.Ticker(ticker)
    info = {}

    try:
        info = stock.get_info()
    except Exception:
        try:
            info = stock.info
        except Exception:
            info = {}

    if not isinstance(info, Mapping):
        info = {}

    current_price = first_present(
        info,
        "currentPrice",
        "regularMarketPrice",
        "previousClose",
    )
    target_mean = first_present(info, "targetMeanPrice")
    target_high = first_present(info, "targetHighPrice")
    target_low = first_present(info, "targetLowPrice")
    target_median = first_present(info, "targetMedianPrice")
    recommendation_key = first_present(info, "recommendationKey")
    analyst_count = first_present(info, "numberOfAnalystOpinions")

    recommendation_summary = pd.DataFrame()
    recommendations = pd.DataFrame()
    revenue_estimates = pd.DataFrame()
    earnings_estimates = pd.DataFrame()

    try:
        recommendation_summary = stock.recommendations_summary
    except Exception:
        recommendation_summary = pd.DataFrame()

    try:
        recommendations = stock.recommendations
    except Exception:
        recommendations = pd.DataFrame()

    try:
        revenue_estimates = stock.revenue_estimate
    except Exception:
        try:
            revenue_estimates = stock.get_revenue_estimate()
        except Exception:
            revenue_estimates = pd.DataFrame()

    try:
        earnings_estimates = stock.earnings_estimate
    except Exception:
        try:
            earnings_estimates = stock.get_earnings_estimate()
        except Exception:
            earnings_estimates = pd.DataFrame()

    def safe_dataframe(value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            return pd.DataFrame()
        return value.copy()

    return {
        "current_price": current_price,
        "target_mean": target_mean,
        "target_high": target_high,
        "target_low": target_low,
        "target_median": target_median,
        "recommendation_key": recommendation_key,
        "analyst_count": analyst_count,
        "recommendation_summary": safe_dataframe(recommendation_summary),
        "recommendations": safe_dataframe(recommendations),
        "revenue_estimates": safe_dataframe(revenue_estimates),
        "earnings_estimates": safe_dataframe(earnings_estimates),
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_news_items(ticker: str, company_data: Mapping[str, Any]) -> Any:
    function = get_callable(
        "company_news",
        (
            "get_company_news",
            "fetch_company_news",
            "get_news",
            "fetch_news",
            "company_news",
        ),
    )
    if function is None:
        raise RuntimeError(
            "company_news.py loaded, but no supported news-fetching function was found."
        )

    company_name = str(
        first_present(
            company_data,
            "company_name",
            "longName",
            "shortName",
            "name",
            default=ticker,
        )
    ).strip()

    context = {
        "ticker": ticker,
        "ticker_symbol": ticker,
        "symbol": ticker,
        "data": company_data,
        "company_data": company_data,
        "stock_data": company_data,
        "company_name": company_name,
        "limit": 8,
    }

    return call_compatible(
        function,
        context,
        positional_attempts=(
            (ticker,),
            (ticker, 8),
            (ticker, 8, company_name),
        ),
    )


def calculate_scores(company_data: Mapping[str, Any]) -> dict[str, Any]:
    function = get_callable(
        "financial_score",
        (
            "calculate_financial_scores",
            "calculate_financial_score",
            "get_financial_scores",
            "score_company",
        ),
    )
    if function is None:
        raise RuntimeError(
            "financial_score.py loaded, but no supported scoring function was found."
        )

    result = call_compatible(
        function,
        {
            "data": company_data,
            "stock_data": company_data,
            "company_data": company_data,
            "financial_data": company_data,
        },
        positional_attempts=((company_data,),),
    )
    return as_mapping(result)


# =============================================================================
# MODULE-SPECIFIC CALLS
# =============================================================================

def generate_analysis(company_data: Mapping[str, Any], scores: Mapping[str, Any]) -> Any:
    function = get_callable(
        "ai_analysis",
        (
            "generate_ai_analysis",
            "generate_analysis",
            "analyze_company",
            "create_ai_analysis",
        ),
    )
    if function is None:
        raise RuntimeError(
            "ai_analysis.py loaded, but no supported analysis function was found."
        )

    return call_compatible(
        function,
        {
            "data": company_data,
            "stock_data": company_data,
            "company_data": company_data,
            "financial_data": company_data,
            "scores": scores,
            "financial_scores": scores,
        },
        positional_attempts=((company_data,), (company_data, scores)),
    )


def generate_news_analysis(
    company_data: Mapping[str, Any],
    news_items: Any,
) -> Any:
    function = get_callable(
        "ai_news",
        (
            "generate_ai_news_analysis",
            "analyze_news_with_ai",
            "generate_news_analysis",
            "analyze_news",
            "generate_ai_news",
        ),
    )
    if function is None:
        raise RuntimeError(
            "ai_news.py loaded, but no supported AI-news function was found."
        )

    return call_compatible(
        function,
        {
            "data": company_data,
            "stock_data": company_data,
            "company_data": company_data,
            "news": news_items,
            "news_items": news_items,
            "articles": news_items,
            "company_news": news_items,
            "news_articles": news_items,
            "ticker": first_present(company_data, "ticker"),
            "ticker_symbol": first_present(company_data, "ticker"),
            "company_name": first_present(company_data, "company_name", "name"),
        },
        positional_attempts=(
            (news_items,),
            (company_data, news_items),
            (first_present(company_data, "ticker"), news_items),
        ),
    )


def generate_recommendation(
    company_data: Mapping[str, Any],
    scores: Mapping[str, Any],
    analysis: Any = None,
    news_analysis: Any = None,
) -> Any:
    function = get_callable(
        "investment_recommendation",
        (
            "generate_investment_recommendation",
            "get_investment_recommendation",
            "create_investment_recommendation",
            "generate_recommendation",
            "recommend_investment",
        ),
    )
    if function is None:
        raise RuntimeError(
            "investment_recommendation.py loaded, but no supported recommendation function was found."
        )

    context = {
        "data": company_data,
        "stock_data": company_data,
        "company_data": company_data,
        "financial_data": company_data,
        "scores": scores,
        "financial_scores": scores,
        "analysis": analysis,
        "ai_analysis": analysis,
        "news_analysis": news_analysis,
        "ai_news_analysis": news_analysis,
    }

    return call_compatible(
        function,
        context,
        positional_attempts=(
            (company_data,),
            (company_data, scores),
            (company_data, scores, analysis),
            (company_data, scores, analysis, news_analysis),
        ),
    )


def run_comparison(
    companies: Sequence[Mapping[str, Any]],
    score_map: Mapping[str, Mapping[str, Any]],
    returns: Mapping[str, Mapping[str, Any]],
    benchmark_ticker: str,
    benchmark_returns: Mapping[str, Any],
) -> Any:
    function = get_callable(
        "comparison",
        (
            "compare_companies",
            "compare_stocks",
            "build_comparison",
            "company_comparison",
        ),
    )
    if function is None:
        raise RuntimeError(
            "comparison.py loaded, but no supported comparison function was found."
        )

    return call_compatible(
        function,
        {
            "companies": list(companies),
            "company_data": list(companies),
            "scores": score_map,
            "score_map": score_map,
            "returns": returns,
            "benchmark_ticker": benchmark_ticker,
            "benchmark_returns": benchmark_returns,
        },
        positional_attempts=(
            (list(companies),),
            (
                list(companies),
                score_map,
                returns,
                benchmark_ticker,
                benchmark_returns,
            ),
        ),
    )


def run_ai_comparison(
    companies: Sequence[Mapping[str, Any]],
    comparison_result: Mapping[str, Any],
) -> Any:
    function = get_callable(
        "ai_comparison",
        (
            "generate_ai_comparison",
            "compare_companies_with_ai",
            "generate_comparison_analysis",
            "analyze_comparison",
        ),
    )
    if function is None:
        raise RuntimeError(
            "ai_comparison.py loaded, but no supported AI-comparison function was found."
        )

    context = {
        "companies": list(companies),
        "company_data": list(companies),
        "comparison_result": comparison_result,
        "comparison": comparison_result,
    }

    return call_compatible(
        function,
        context,
        positional_attempts=(
            (list(companies),),
            (list(companies), comparison_result),
        ),
    )



def render_dcf_result(
    result: Mapping[str, Any],
    current_price: Any = None,
    title: str = "DCF Valuation",
) -> None:
    """Render a professional base-case DCF valuation."""

    if not isinstance(result, Mapping) or not result:
        st.info("No DCF valuation result is available.")
        return

    def safe_float(value: Any) -> Optional[float]:
        try:
            number = float(value)

            if number == number:
                return number
        except (TypeError, ValueError):
            pass

        return None

    def format_currency(value: Any) -> str:
        number = safe_float(value)

        if number is None:
            return "N/A"

        absolute_value = abs(number)

        if absolute_value >= 1_000_000_000_000:
            return f"${number / 1_000_000_000_000:,.2f}T"

        if absolute_value >= 1_000_000_000:
            return f"${number / 1_000_000_000:,.2f}B"

        if absolute_value >= 1_000_000:
            return f"${number / 1_000_000:,.2f}M"

        return f"${number:,.2f}"

    fair_value = safe_float(result.get("fair_value_per_share"))
    market_price = safe_float(current_price)

    upside_downside = None

    if (
        fair_value is not None
        and market_price is not None
        and market_price > 0
    ):
        upside_downside = (
            fair_value / market_price - 1
        ) * 100

    if upside_downside is None:
        valuation_status = "Price Unavailable"
    elif upside_downside >= 20:
        valuation_status = "Potentially Undervalued"
    elif upside_downside >= 5:
        valuation_status = "Moderately Undervalued"
    elif upside_downside > -5:
        valuation_status = "Approximately Fair Value"
    elif upside_downside > -20:
        valuation_status = "Moderately Overvalued"
    else:
        valuation_status = "Potentially Overvalued"

    st.subheader(title)

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Intrinsic Value",
        format_currency(fair_value),
    )

    metric_columns[1].metric(
        "Current Price",
        format_currency(market_price),
    )

    metric_columns[2].metric(
        "Upside / Downside",
        (
            f"{upside_downside:+.1f}%"
            if upside_downside is not None
            else "N/A"
        ),
    )

    metric_columns[3].metric(
        "Valuation View",
        valuation_status,
    )

    if upside_downside is None:
        st.info(
            "Valuation comparison unavailable because a current market "
            "price could not be retrieved."
        )
    elif upside_downside >= 5:
        st.success(
            f"DCF Valuation Signal: {valuation_status} — "
            f"estimated intrinsic value implies {upside_downside:+.1f}% "
            f"upside versus the current market price."
        )
    elif upside_downside > -5:
        st.warning(
            f"DCF Valuation Signal: {valuation_status} — "
            f"estimated intrinsic value is within 5% of the current "
            f"market price."
        )
    else:
        st.error(
            f"DCF Valuation Signal: {valuation_status} — "
            f"estimated intrinsic value implies {upside_downside:+.1f}% "
            f"downside versus the current market price."
        )

    if market_price is None:
        st.caption(
            "Current market price was unavailable, so implied upside or "
            "downside could not be calculated."
        )

    st.markdown("#### Core Assumptions")

    assumption_columns = st.columns(4)

    growth_rate = safe_float(result.get("growth_rate"))
    discount_rate = safe_float(result.get("discount_rate"))
    terminal_growth_rate = safe_float(
        result.get("terminal_growth_rate")
    )
    forecast_years = result.get("forecast_years")

    assumption_columns[0].metric(
        "FCF Growth",
        (
            f"{growth_rate * 100:.1f}%"
            if growth_rate is not None
            else "N/A"
        ),
    )

    assumption_columns[1].metric(
        "Discount Rate / WACC",
        (
            f"{discount_rate * 100:.1f}%"
            if discount_rate is not None
            else "N/A"
        ),
    )

    assumption_columns[2].metric(
        "Terminal Growth",
        (
            f"{terminal_growth_rate * 100:.1f}%"
            if terminal_growth_rate is not None
            else "N/A"
        ),
    )

    assumption_columns[3].metric(
        "Projection Period",
        (
            f"{int(forecast_years)} Years"
            if safe_float(forecast_years) is not None
            else "N/A"
        ),
    )

    projected_cash_flows = result.get(
        "projected_cash_flows",
        [],
    )
    present_values = result.get(
        "present_values",
        [],
    )

    if isinstance(projected_cash_flows, (list, tuple)):
        forecast_rows = []

        for index, projected_fcf in enumerate(
            projected_cash_flows,
            start=1,
        ):
            present_value = (
                present_values[index - 1]
                if isinstance(present_values, (list, tuple))
                and index - 1 < len(present_values)
                else None
            )

            discount_factor = None
            projected_number = safe_float(projected_fcf)
            present_number = safe_float(present_value)

            if (
                projected_number is not None
                and present_number is not None
                and projected_number != 0
            ):
                discount_factor = (
                    present_number / projected_number
                )

            forecast_rows.append(
                {
                    "Forecast Year": f"Year {index}",
                    "Projected Free Cash Flow": projected_number,
                    "Present Value": present_number,
                    "PV as % of FCF": (
                        discount_factor * 100
                        if discount_factor is not None
                        else None
                    ),
                }
            )

        if forecast_rows:
            st.markdown("#### Free Cash Flow Forecast")

            forecast_df = pd.DataFrame(forecast_rows)

            st.dataframe(
                forecast_df.style.format(
                    {
                        "Projected Free Cash Flow": format_currency,
                        "Present Value": format_currency,
                        "PV as % of FCF": lambda value: (
                            f"{value:.1f}%"
                            if pd.notna(value)
                            else "N/A"
                        ),
                    }
                ),
                hide_index=True,
                width="stretch",
            )

            st.download_button(
                "Download DCF Forecast",
                data=forecast_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="dcf_forecast.csv",
                mime="text/csv",
                key="download_dcf_forecast",
            )

    st.markdown("#### Valuation Bridge")

    valuation_rows = [
        {
            "Valuation Component": "PV of Forecast Cash Flows",
            "Value": safe_float(
                result.get(
                    "present_value_of_forecast_cash_flows"
                )
            ),
        },
        {
            "Valuation Component": "PV of Terminal Value",
            "Value": safe_float(
                result.get("discounted_terminal_value")
            ),
        },
        {
            "Valuation Component": "Enterprise Value",
            "Value": safe_float(
                result.get("enterprise_value")
            ),
        },
        {
            "Valuation Component": "Cash",
            "Value": safe_float(result.get("total_cash")),
        },
        {
            "Valuation Component": "Debt",
            "Value": safe_float(result.get("total_debt")),
        },
        {
            "Valuation Component": "Net Cash / (Debt)",
            "Value": safe_float(result.get("net_cash")),
        },
        {
            "Valuation Component": "Equity Value",
            "Value": safe_float(result.get("equity_value")),
        },
        {
            "Valuation Component": "Fair Value per Share",
            "Value": fair_value,
        },
    ]

    valuation_df = pd.DataFrame(valuation_rows)

    st.dataframe(
        valuation_df.style.format(
            {"Value": format_currency}
        ),
        hide_index=True,
        width="stretch",
    )

    enterprise_value = safe_float(
        result.get("enterprise_value")
    )
    discounted_terminal_value = safe_float(
        result.get("discounted_terminal_value")
    )

    terminal_value_share = None

    if (
        enterprise_value is not None
        and enterprise_value > 0
        and discounted_terminal_value is not None
    ):
        terminal_value_share = (
            discounted_terminal_value
            / enterprise_value
            * 100
        )

    if terminal_value_share is not None:
        st.caption(
            f"Terminal value represents "
            f"{terminal_value_share:.1f}% of estimated enterprise value."
        )

        if terminal_value_share >= 75:
            st.warning(
                "A large portion of this valuation comes from terminal "
                "value. The result is therefore highly sensitive to the "
                "discount-rate and terminal-growth assumptions."
            )

    st.caption(
        "DCF estimates are assumption-sensitive and should be evaluated "
        "alongside comparable-company valuation, financial quality, "
        "industry conditions, and company-specific risks."
    )


def render_scenario_result(result: Any) -> None:
    """Render bull, base, and bear DCF scenarios."""

    if not isinstance(result, Mapping) or not result:
        st.info("No DCF scenario results are available.")
        return

    def safe_float(value: Any) -> Optional[float]:
        try:
            number = float(value)

            if number == number:
                return number
        except (TypeError, ValueError):
            pass

        return None

    scenario_rows = []

    preferred_order = ("Bear", "Base", "Bull")

    ordered_names = [
        name for name in preferred_order
        if name in result
    ]

    ordered_names.extend(
        name for name in result
        if name not in ordered_names
    )

    for scenario_name in ordered_names:
        scenario = result.get(scenario_name)

        if not isinstance(scenario, Mapping):
            continue

        assumptions = scenario.get("assumptions", {})

        if not isinstance(assumptions, Mapping):
            assumptions = {}

        growth_rate = safe_float(
            assumptions.get(
                "growth_rate",
                scenario.get("growth_rate"),
            )
        )
        discount_rate = safe_float(
            assumptions.get(
                "discount_rate",
                scenario.get("discount_rate"),
            )
        )
        terminal_growth_rate = safe_float(
            assumptions.get(
                "terminal_growth_rate",
                scenario.get("terminal_growth_rate"),
            )
        )

        scenario_rows.append(
            {
                "Scenario": scenario_name,
                "FCF Growth": (
                    growth_rate * 100
                    if growth_rate is not None
                    else None
                ),
                "WACC": (
                    discount_rate * 100
                    if discount_rate is not None
                    else None
                ),
                "Terminal Growth": (
                    terminal_growth_rate * 100
                    if terminal_growth_rate is not None
                    else None
                ),
                "Enterprise Value": safe_float(
                    scenario.get("enterprise_value")
                ),
                "Equity Value": safe_float(
                    scenario.get("equity_value")
                ),
                "Fair Value per Share": safe_float(
                    scenario.get("fair_value_per_share")
                ),
            }
        )

    if not scenario_rows:
        st.info("No displayable DCF scenario results were returned.")
        return

    scenario_df = pd.DataFrame(scenario_rows)

    metric_columns = st.columns(len(scenario_rows))

    for column, row in zip(
        metric_columns,
        scenario_rows,
    ):
        fair_value = row.get("Fair Value per Share")

        column.metric(
            f"{row['Scenario']} Case",
            (
                f"${fair_value:,.2f}"
                if fair_value is not None
                else "N/A"
            ),
        )

    st.dataframe(
        scenario_df.style.format(
            {
                "FCF Growth": lambda value: (
                    f"{value:.1f}%"
                    if pd.notna(value)
                    else "N/A"
                ),
                "WACC": lambda value: (
                    f"{value:.1f}%"
                    if pd.notna(value)
                    else "N/A"
                ),
                "Terminal Growth": lambda value: (
                    f"{value:.1f}%"
                    if pd.notna(value)
                    else "N/A"
                ),
                "Enterprise Value": lambda value: (
                    f"${value / 1_000_000_000:,.2f}B"
                    if pd.notna(value)
                    else "N/A"
                ),
                "Equity Value": lambda value: (
                    f"${value / 1_000_000_000:,.2f}B"
                    if pd.notna(value)
                    else "N/A"
                ),
                "Fair Value per Share": lambda value: (
                    f"${value:,.2f}"
                    if pd.notna(value)
                    else "N/A"
                ),
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download Scenario Analysis",
        data=scenario_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="dcf_scenarios.csv",
        mime="text/csv",
        key="download_dcf_scenarios",
    )

    st.caption(
        "Scenario values are not price targets. They illustrate how changes "
        "in growth, WACC, and terminal-growth assumptions affect estimated "
        "intrinsic value."
    )


def render_sensitivity_result(result: Any) -> None:
    """Render and export a DCF sensitivity matrix."""

    if result is None:
        st.info("No DCF sensitivity result is available.")
        return

    if isinstance(result, pd.DataFrame):
        sensitivity_df = result.copy()
    elif isinstance(result, Mapping):
        try:
            sensitivity_df = pd.DataFrame(result)
        except Exception:
            st.write(result)
            return
    else:
        st.write(result)
        return

    if sensitivity_df.empty:
        st.info("The DCF sensitivity table is empty.")
        return

    display_df = sensitivity_df.copy()

    value_columns = [
        column
        for column in display_df.columns
        if column != "FCF Growth"
    ]

    def format_sensitivity_value(value: Any) -> str:
        try:
            number = float(value)

            if number == number:
                return f"${number:,.2f}"
        except (TypeError, ValueError):
            pass

        return "N/A"

    formatters = {
        column: format_sensitivity_value
        for column in value_columns
    }

    st.dataframe(
        display_df.style.format(formatters),
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Rows vary projected free-cash-flow growth. Columns vary WACC. "
        "Higher growth and lower WACC generally produce higher estimated "
        "intrinsic value."
    )

    st.download_button(
        "Download Sensitivity Table",
        data=sensitivity_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="dcf_sensitivity.csv",
        mime="text/csv",
        key="download_dcf_sensitivity",
    )

def run_dcf(
    company_data: Mapping[str, Any],
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    years: int,
) -> Any:
    function = get_callable(
        "dcf_model",
        (
            "calculate_dcf",
            "run_dcf",
            "dcf_valuation",
            "calculate_dcf_valuation",
        ),
    )

    if function is None:
        raise RuntimeError(
            "dcf_model.py loaded, but no supported DCF function was found."
        )

    def get_first_number(*keys: str) -> Optional[float]:
        for key in keys:
            value = company_data.get(key)

            if value is None:
                continue

            try:
                number = float(value)

                if number == number:
                    return number
            except (TypeError, ValueError):
                continue

        return None

    free_cash_flow = get_first_number(
        "free_cash_flow",
        "freeCashflow",
        "freeCashFlow",
        "fcf",
    )

    shares_outstanding = get_first_number(
        "shares_outstanding",
        "sharesOutstanding",
        "shares",
        "impliedSharesOutstanding",
    )

    total_cash = get_first_number(
        "total_cash",
        "totalCash",
        "cash",
        "cashAndCashEquivalents",
    )

    total_debt = get_first_number(
        "total_debt",
        "totalDebt",
        "debt",
    )

    if free_cash_flow is None:
        raise ValueError(
            "Free cash flow could not be found in the company data."
        )

    if shares_outstanding is None:
        raise ValueError(
            "Shares outstanding could not be found in the company data."
        )

    return function(
        free_cash_flow=free_cash_flow,
        shares_outstanding=shares_outstanding,
        total_cash=total_cash,
        total_debt=total_debt,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        forecast_years=years,
    )

def run_dcf_scenarios(
    company_data: Mapping[str, Any],
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    years: int,
) -> Any:
    function = get_callable(
        "dcf_scenarios",
        (
            "calculate_dcf_scenarios",
            "run_dcf_scenarios",
            "generate_dcf_scenarios",
            "scenario_analysis",
        ),
    )
    if function is None:
        raise RuntimeError(
            "dcf_scenarios.py loaded, but no supported scenario function was found."
        )

    context = {
        "data": company_data,
        "stock_data": company_data,
        "company_data": company_data,
        "growth_rate": growth_rate,
        "base_growth_rate": growth_rate,
        "discount_rate": discount_rate,
        "base_discount_rate": discount_rate,
        "terminal_growth_rate": terminal_growth_rate,
        "years": years,
        "projection_years": years,
    }

    return call_compatible(
        function,
        context,
        positional_attempts=(
            (company_data,),
            (company_data, growth_rate, discount_rate, terminal_growth_rate),
            (company_data, growth_rate, discount_rate, terminal_growth_rate, years),
        ),
    )


def run_dcf_sensitivity(
    company_data: Mapping[str, Any],
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    years: int,
) -> Any:
    function = get_callable(
        "dcf_sensitivity",
        (
            "calculate_sensitivity_table",
            "calculate_dcf_sensitivity",
            "run_sensitivity_analysis",
            "generate_sensitivity_table",
        ),
    )
    if function is None:
        raise RuntimeError(
            "dcf_sensitivity.py loaded, but no supported sensitivity function was found."
        )

    context = {
        "data": company_data,
        "stock_data": company_data,
        "company_data": company_data,
        "growth_rate": growth_rate,
        "base_growth_rate": growth_rate,
        "discount_rate": discount_rate,
        "base_discount_rate": discount_rate,
        "terminal_growth_rate": terminal_growth_rate,
        "years": years,
        "projection_years": years,
    }

    return call_compatible(
        function,
        context,
        positional_attempts=(
            (company_data,),
            (company_data, growth_rate, discount_rate, terminal_growth_rate),
            (company_data, growth_rate, discount_rate, terminal_growth_rate, years),
        ),
    )


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def render_scorecard(scores: Mapping[str, Any]) -> None:
    score_keys = [
        ("overall", "Overall"),
        ("profitability", "Profitability"),
        ("balance_sheet", "Balance Sheet"),
        ("valuation", "Valuation"),
        ("market_performance", "Market Performance"),
    ]

    columns = st.columns(len(score_keys))

    for column, (key, label) in zip(columns, score_keys):
        value = first_present(scores, key, key.replace("_", " "), default=None)
        if is_number(value):
            column.metric(label, f"{float(value):.0f} / 100")
        else:
            column.metric(label, "N/A")


def render_price_chart(history: pd.DataFrame, ticker: str) -> None:
    if history.empty or "Close" not in history.columns:
        st.warning("Price history was not available.")
        return

    chart_data = history.copy()
    chart_data = chart_data.dropna(subset=["Close"])

    if chart_data.empty:
        st.warning("Price history was not available.")
        return

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Close"],
            mode="lines",
            name=ticker,
            line={"width": 2.5},
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Close: $%{y:,.2f}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=f"{ticker} Price History",
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        height=430,
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title_text="Ticker",
    )

    figure.update_xaxes(
        rangeslider_visible=False,
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
    )

    st.plotly_chart(figure, use_container_width=True)


def render_financial_overview(data: Mapping[str, Any]) -> None:
    """Render all financial metrics without allowing one missing value to stop the section."""
    valuation, profitability, health = st.columns(3)

    with valuation:
        st.markdown("#### Valuation")
        st.write(
            f"**Enterprise Value:** "
            f"{format_large_number(first_present(data, 'enterprise_value'))}"
        )
        st.write(
            f"**Trailing P/E:** "
            f"{format_number(first_present(data, 'trailing_pe'))}"
        )
        st.write(
            f"**Forward P/E:** "
            f"{format_number(first_present(data, 'forward_pe'))}"
        )
        st.write(
            f"**PEG Ratio:** "
            f"{format_number(first_present(data, 'peg_ratio'))}"
        )
        st.write(
            f"**Price / Book:** "
            f"{format_number(first_present(data, 'price_to_book'))}"
        )
        st.write(
            f"**Beta:** "
            f"{format_number(first_present(data, 'beta'))}"
        )
        st.write(
            f"**Dividend Yield:** "
            f"{format_dividend_yield(first_present(data, 'dividend_yield'))}"
        )

    with profitability:
        st.markdown("#### Profitability")
        st.write(
            f"**Revenue:** "
            f"{format_large_number(first_present(data, 'revenue', 'total_revenue'))}"
        )
        st.write(
            f"**Net Income:** "
            f"{format_large_number(first_present(data, 'net_income'))}"
        )
        st.write(
            f"**EBITDA:** "
            f"{format_large_number(first_present(data, 'ebitda'))}"
        )
        st.write(
            f"**Profit Margin:** "
            f"{format_percent(first_present(data, 'profit_margin'))}"
        )
        st.write(
            f"**Operating Margin:** "
            f"{format_percent(first_present(data, 'operating_margin'))}"
        )
        st.write(
            f"**Return on Equity:** "
            f"{format_percent(first_present(data, 'return_on_equity'))}"
        )
        st.write(
            f"**Return on Assets:** "
            f"{format_percent(first_present(data, 'return_on_assets'))}"
        )

    with health:
        st.markdown("#### Financial Health")
        st.write(
            f"**Total Cash:** "
            f"{format_large_number(first_present(data, 'total_cash'))}"
        )
        st.write(
            f"**Total Debt:** "
            f"{format_large_number(first_present(data, 'total_debt'))}"
        )
        st.write(
            f"**Debt / Equity:** "
            f"{format_debt_to_equity(first_present(data, 'debt_to_equity'))}"
        )
        st.write(
            f"**Current Ratio:** "
            f"{format_number(first_present(data, 'current_ratio'))}"
        )
        st.write(
            f"**Quick Ratio:** "
            f"{format_number(first_present(data, 'quick_ratio'))}"
        )
        st.write(
            f"**Operating Cash Flow:** "
            f"{format_large_number(first_present(data, 'operating_cash_flow'))}"
        )
        st.write(
            f"**Free Cash Flow:** "
            f"{format_large_number(first_present(data, 'free_cash_flow'))}"
        )



STATEMENT_LABELS = {
    "TotalRevenue": "Revenue",
    "OperatingRevenue": "Operating Revenue",
    "CostOfRevenue": "Cost of Revenue",
    "ReconciledCostOfRevenue": "Cost of Revenue",
    "GrossProfit": "Gross Profit",
    "OperatingExpense": "Operating Expenses",
    "SellingGeneralAndAdministration": "SG&A Expense",
    "ResearchAndDevelopment": "Research & Development",
    "OperatingIncome": "Operating Income",
    "TotalOperatingIncomeAsReported": "Reported Operating Income",
    "EBIT": "EBIT",
    "EBITDA": "EBITDA",
    "NormalizedEBITDA": "Normalized EBITDA",
    "InterestIncome": "Interest Income",
    "InterestExpense": "Interest Expense",
    "NetInterestIncome": "Net Interest Income",
    "PretaxIncome": "Pre-Tax Income",
    "TaxProvision": "Income Tax Expense",
    "NetIncome": "Net Income",
    "NetIncomeCommonStockholders": "Net Income to Common Shareholders",
    "NetIncomeContinuousOperations": "Net Income from Continuing Operations",
    "BasicEPS": "Basic EPS",
    "DilutedEPS": "Diluted EPS",
    "BasicAverageShares": "Basic Weighted-Average Shares",
    "DilutedAverageShares": "Diluted Weighted-Average Shares",

    "TotalAssets": "Total Assets",
    "CurrentAssets": "Current Assets",
    "CashAndCashEquivalents": "Cash & Cash Equivalents",
    "CashCashEquivalentsAndShortTermInvestments":
        "Cash & Short-Term Investments",
    "AccountsReceivable": "Accounts Receivable",
    "Inventory": "Inventory",
    "NetPPE": "Property, Plant & Equipment",
    "GoodwillAndOtherIntangibleAssets": "Goodwill & Intangible Assets",
    "TotalLiabilitiesNetMinorityInterest": "Total Liabilities",
    "CurrentLiabilities": "Current Liabilities",
    "AccountsPayable": "Accounts Payable",
    "CurrentDebt": "Current Debt",
    "LongTermDebt": "Long-Term Debt",
    "TotalDebt": "Total Debt",
    "StockholdersEquity": "Shareholders' Equity",
    "TotalEquityGrossMinorityInterest": "Total Equity",
    "RetainedEarnings": "Retained Earnings",
    "WorkingCapital": "Working Capital",
    "OrdinarySharesNumber": "Shares Outstanding",

    "OperatingCashFlow": "Operating Cash Flow",
    "CashFlowFromContinuingOperatingActivities":
        "Cash Flow from Operating Activities",
    "InvestingCashFlow": "Investing Cash Flow",
    "CashFlowFromContinuingInvestingActivities":
        "Cash Flow from Investing Activities",
    "FinancingCashFlow": "Financing Cash Flow",
    "CashFlowFromContinuingFinancingActivities":
        "Cash Flow from Financing Activities",
    "CapitalExpenditure": "Capital Expenditures",
    "FreeCashFlow": "Free Cash Flow",
    "RepurchaseOfCapitalStock": "Share Repurchases",
    "CashDividendsPaid": "Dividends Paid",
    "CommonStockDividendPaid": "Common Dividends Paid",
    "IssuanceOfDebt": "Debt Issued",
    "RepaymentOfDebt": "Debt Repaid",
    "BeginningCashPosition": "Beginning Cash Balance",
    "EndCashPosition": "Ending Cash Balance",
    "ChangesInCash": "Net Change in Cash",
}


PRIMARY_STATEMENT_ROWS = {
    "Income Statement": [
        "TotalRevenue",
        "CostOfRevenue",
        "GrossProfit",
        "ResearchAndDevelopment",
        "SellingGeneralAndAdministration",
        "OperatingExpense",
        "OperatingIncome",
        "EBIT",
        "EBITDA",
        "InterestIncome",
        "InterestExpense",
        "PretaxIncome",
        "TaxProvision",
        "NetIncome",
        "NetIncomeCommonStockholders",
        "BasicEPS",
        "DilutedEPS",
        "BasicAverageShares",
        "DilutedAverageShares",
    ],
    "Balance Sheet": [
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalents",
        "AccountsReceivable",
        "Inventory",
        "CurrentAssets",
        "NetPPE",
        "GoodwillAndOtherIntangibleAssets",
        "TotalAssets",
        "AccountsPayable",
        "CurrentDebt",
        "CurrentLiabilities",
        "LongTermDebt",
        "TotalDebt",
        "TotalLiabilitiesNetMinorityInterest",
        "RetainedEarnings",
        "StockholdersEquity",
        "TotalEquityGrossMinorityInterest",
        "WorkingCapital",
        "OrdinarySharesNumber",
    ],
    "Cash Flow Statement": [
        "OperatingCashFlow",
        "CashFlowFromContinuingOperatingActivities",
        "CapitalExpenditure",
        "FreeCashFlow",
        "InvestingCashFlow",
        "CashFlowFromContinuingInvestingActivities",
        "FinancingCashFlow",
        "CashFlowFromContinuingFinancingActivities",
        "RepurchaseOfCapitalStock",
        "CashDividendsPaid",
        "CommonStockDividendPaid",
        "IssuanceOfDebt",
        "RepaymentOfDebt",
        "BeginningCashPosition",
        "ChangesInCash",
        "EndCashPosition",
    ],
}


def humanize_statement_label(value: Any) -> str:
    raw_label = str(value)

    if raw_label in STATEMENT_LABELS:
        return STATEMENT_LABELS[raw_label]

    readable = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw_label)
    readable = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", readable)
    readable = readable.replace("And", "and")
    return readable.strip()


def format_compact_statement_number(
    value: float,
    currency: bool = True,
) -> str:
    absolute_value = abs(value)

    if absolute_value >= 1_000_000_000_000:
        formatted = f"{absolute_value / 1_000_000_000_000:.2f}T"
    elif absolute_value >= 1_000_000_000:
        formatted = f"{absolute_value / 1_000_000_000:.2f}B"
    elif absolute_value >= 1_000_000:
        formatted = f"{absolute_value / 1_000_000:.2f}M"
    elif absolute_value >= 1_000:
        formatted = f"{absolute_value / 1_000:.2f}K"
    else:
        formatted = f"{absolute_value:,.0f}"

    prefix = "$" if currency else ""
    result = f"{prefix}{formatted}"

    if value < 0:
        return f"({result})"

    return result


def format_statement_value(line_item: str, value: Any) -> Any:
    if pd.isna(value):
        return ""

    if not is_number(value):
        return str(value)

    numeric_value = float(value)
    normalized_item = line_item.lower()

    if "eps" in normalized_item:
        return f"{numeric_value:.2f}"

    if "share" in normalized_item:
        return format_compact_statement_number(
            numeric_value,
            currency=False,
        )

    if "tax rate" in normalized_item:
        percent_value = (
            numeric_value * 100
            if abs(numeric_value) <= 1
            else numeric_value
        )
        return f"{percent_value:.1f}%"

    return format_compact_statement_number(numeric_value)


def statement_download_bytes(statement: pd.DataFrame) -> bytes:
    return statement.to_csv(index=True).encode("utf-8")


def prepare_statement_for_display(
    statement_name: str,
    statement: pd.DataFrame,
    show_detailed: bool,
) -> pd.DataFrame:
    source = statement.copy()

    if not show_detailed:
        preferred_rows = PRIMARY_STATEMENT_ROWS.get(statement_name, [])
        available_rows = [
            row for row in preferred_rows if row in source.index
        ]

        if available_rows:
            source = source.loc[available_rows]

    # Remove reporting periods where every displayed line item is missing
    source = source.dropna(axis=1, how="all")

    source.index.name = "Line Item"
    display_table = source.reset_index()

    raw_line_items = display_table["Line Item"].astype(str).tolist()
    display_table["Line Item"] = [
        humanize_statement_label(item)
        for item in raw_line_items
    ]

    for column in display_table.columns:
        if column == "Line Item":
            continue

        display_table[column] = [
            format_statement_value(raw_item, value)
            for raw_item, value in zip(
                raw_line_items,
                display_table[column],
            )
        ]

    return display_table


def render_financial_statement(
    statement_name: str,
    statement: pd.DataFrame,
    ticker: str,
    frequency: str,
) -> None:
    st.subheader(statement_name)

    if not isinstance(statement, pd.DataFrame) or statement.empty:
        st.warning(
            f"No {statement_name.lower()} data was returned for {ticker}. "
            "Try the other frequency or another ticker."
        )
        return

    show_detailed = st.checkbox(
        "Show detailed statement",
        value=False,
        key=(
            f"show_detailed_{ticker}_{frequency}_"
            f"{statement_name}"
        ),
    )

    display_table = prepare_statement_for_display(
        statement_name,
        statement,
        show_detailed,
    )

    st.dataframe(
        display_table,
        hide_index=True,
        height=650,
        column_config={
            "Line Item": st.column_config.TextColumn(
                "Line Item",
                width="large",
            )
        },
    )

    visible_description = (
        "all available line items"
        if show_detailed
        else "key financial line items"
    )

    st.caption(
        f"Showing {visible_description} · "
        f"{len(statement.columns):,} reporting periods"
    )

    filename = (
        f"{ticker}_{frequency}_"
        f"{statement_name.lower().replace(' ', '_')}.csv"
    )

    st.download_button(
        label=f"Download Full {statement_name} CSV",
        data=statement_download_bytes(statement),
        file_name=filename,
        mime="text/csv",
        key=f"download_{ticker}_{frequency}_{statement_name}",
    )


def render_financial_trends_chart(
    trends: pd.DataFrame,
    metrics: list[str],
    title: str,
    percent_metrics: Optional[set[str]] = None,
) -> None:
    if trends.empty:
        st.info("No financial trend data is available.")
        return

    available_metrics = [
        metric for metric in metrics if metric in trends.columns
    ]

    if not available_metrics:
        st.info(f"No data is available for {title.lower()}.")
        return

    chart_data = trends[available_metrics].copy()
    chart_data.index = chart_data.index.strftime("%Y")

    is_percent_chart = bool(percent_metrics)
    is_eps_chart = all(
        "EPS" in metric.upper()
        for metric in available_metrics
    )
    is_currency_chart = not is_percent_chart and not is_eps_chart

    if is_currency_chart:
        plot_data = chart_data / 1_000_000_000
        y_axis_title = "USD Billions"
        hover_template = (
            "<b>%{fullData.name}</b><br>"
            "Fiscal Year: %{x}<br>"
            "Value: $%{y:,.2f}B"
            "<extra></extra>"
        )
    elif is_percent_chart:
        plot_data = chart_data
        y_axis_title = "Percent"
        hover_template = (
            "<b>%{fullData.name}</b><br>"
            "Fiscal Year: %{x}<br>"
            "Value: %{y:,.1f}%"
            "<extra></extra>"
        )
    else:
        plot_data = chart_data
        y_axis_title = "EPS"
        hover_template = (
            "<b>%{fullData.name}</b><br>"
            "Fiscal Year: %{x}<br>"
            "Value: $%{y:,.2f}"
            "<extra></extra>"
        )

    fig = go.Figure()

    for metric in available_metrics:
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data[metric],
                mode="lines+markers",
                name=metric,
                line={"width": 4},
                marker={"size": 10},
                hovertemplate=hover_template,
            )
        )

    fig.update_layout(
        title={"text": ""},
        xaxis_title="Fiscal Year",
        yaxis_title=y_axis_title,
        hovermode="x unified",
        height=560,
        margin={"l": 20, "r": 20, "t": 25, "b": 25},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 13},
        },
        font={"size": 14},
    )

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=list(plot_data.index),
        tickmode="array",
        tickvals=list(plot_data.index),
        ticktext=list(plot_data.index),
        showgrid=False,
        tickfont={"size": 13},
        title_font={"size": 14},
    )

    fig.update_yaxes(
        separatethousands=True,
        gridcolor="rgba(128, 128, 128, 0.18)",
        zerolinecolor="rgba(128, 128, 128, 0.35)",
        tickfont={"size": 13},
        title_font={"size": 14},
    )

    if is_percent_chart:
        fig.update_yaxes(ticksuffix="%")
    elif is_currency_chart:
        fig.update_yaxes(tickprefix="$", ticksuffix="B")
    else:
        fig.update_yaxes(tickprefix="$")

    st.plotly_chart(fig, use_container_width=True)


def render_growth_metrics(trends: pd.DataFrame) -> None:
    if trends.empty or len(trends.index) < 2:
        return

    latest = trends.iloc[-1]
    previous = trends.iloc[-2]

    metric_columns = st.columns(4)

    metric_map = [
        ("Revenue", "Revenue Growth"),
        ("Net Income", "Net Income Growth"),
        ("EBITDA", "EBITDA Growth"),
        ("Free Cash Flow", "FCF Growth"),
    ]

    for column, (source_metric, label) in zip(metric_columns, metric_map):
        current_value = latest.get(source_metric)
        prior_value = previous.get(source_metric)

        growth = None
        if (
            is_number(current_value)
            and is_number(prior_value)
            and float(prior_value) != 0
        ):
            growth = (
                (float(current_value) - float(prior_value))
                / abs(float(prior_value))
                * 100
            )

        with column:
            st.metric(
                label,
                "N/A" if growth is None else f"{growth:.1f}%",
            )



def format_estimate_table(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    display = frame.copy()
    display.index = [str(index) for index in display.index]
    display.index.name = "Period"
    display = display.reset_index()

    for column in display.columns:
        if column == "Period":
            display[column] = display[column].astype(str)
            continue

        display[column] = display[column].apply(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{float(value):,.2f}"
                if is_number(value)
                else str(value)
            )
        )

    return display


def render_analyst_rating_summary(data: Mapping[str, Any]) -> None:
    recommendation_summary = data.get("recommendation_summary")

    if (
        not isinstance(recommendation_summary, pd.DataFrame)
        or recommendation_summary.empty
    ):
        recommendation_key = data.get("recommendation_key")
        if recommendation_key:
            st.info(
                f"Current consensus recommendation: "
                f"**{str(recommendation_key).replace('_', ' ').title()}**"
            )
        else:
            st.info("No analyst recommendation summary was available.")
        return

    summary = recommendation_summary.copy()

    rename_map = {
        "strongBuy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strongSell": "Strong Sell",
    }
    summary = summary.rename(columns=rename_map)

    preferred_columns = [
        column
        for column in (
            "period",
            "Strong Buy",
            "Buy",
            "Hold",
            "Sell",
            "Strong Sell",
        )
        if column in summary.columns
    ]

    if preferred_columns:
        summary = summary[preferred_columns]

    st.dataframe(
        summary,
        hide_index=True,
    )

    numeric_columns = [
        column
        for column in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")
        if column in summary.columns
    ]

    if numeric_columns:
        latest_row = summary.iloc[0]
        labels = []
        values = []

        for column in numeric_columns:
            value = latest_row.get(column)
            if is_number(value):
                labels.append(column)
                values.append(float(value))

        if values:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=labels,
                        y=values,
                    )
                ]
            )
            fig.update_layout(
                title="Latest Analyst Recommendation Breakdown",
                xaxis_title="Recommendation",
                yaxis_title="Number of Analysts",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_estimate_dataframe(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    """
    Display analyst estimate tables with finance-friendly formatting.
    """

    st.subheader(title)

    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        st.info(f"No {title.lower()} were available.")
        return

    df = dataframe.copy()

    if "avg" in df.columns:
        df = df.rename(columns={"avg": "Average"})

    if "low" in df.columns:
        df = df.rename(columns={"low": "Low"})

    if "high" in df.columns:
        df = df.rename(columns={"high": "High"})

    if "growth" in df.columns:
        df = df.rename(columns={"growth": "Growth"})

    money_keywords = (
        "Average",
        "Low",
        "High",
        "Revenue",
        "EPS",
        "yearAgoRevenue",
        "yearAgoEps",
    )

    for column in df.columns:

        if any(keyword.lower() in column.lower() for keyword in money_keywords):

            def format_billions(value):
                try:
                    value = float(value)

                    if abs(value) >= 1_000_000_000:
                        return f"${value/1_000_000_000:.1f}B"

                    if abs(value) >= 1_000_000:
                        return f"${value/1_000_000:.1f}M"

                    return f"${value:,.2f}"

                except Exception:
                    return value

            df[column] = df[column].apply(format_billions)

        elif column.lower() == "growth":

            def format_growth(value):
                try:
                    return f"{float(value)*100:.1f}%"
                except Exception:
                    return value

            df[column] = df[column].apply(format_growth)

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        height=min(320, 45 * (len(df) + 1)),
    )


# =============================================================================
# SESSION STATE
# =============================================================================

STATE_DEFAULTS = {
    "analysis_data": None,
    "analysis_scores": None,
    "analysis_history": None,
    "ai_report": None,
    "news_items": None,
    "ai_news_report": None,
    "investment_recommendation": None,
    "comparison_result": None,
    "ai_comparison_result": None,
    "comparison_companies": None,
    "dcf_result": None,
    "dcf_scenarios_result": None,
    "dcf_sensitivity_result": None,
    "dcf_company_data": None,
    "financial_statements": None,
    "financial_statements_ticker": None,
    "financial_statements_frequency": None,
}

for state_key, default_value in STATE_DEFAULTS.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📈 Finance AI Research Terminal")
    st.caption("AI-powered public-equity research, analytics, and valuation.")

    main_ticker = clean_ticker(
        st.text_input(
            "Primary ticker",
            value=st.session_state.get("primary_ticker", "NVDA"),
            key="primary_ticker_input",
        )
    )
    st.session_state["primary_ticker"] = main_ticker

    history_period = st.selectbox(
        "Price-history period",
        options=("1mo", "3mo", "6mo", "1y", "2y", "5y", "max"),
        index=3,
    )

# =============================================================================
# HEADER
# =============================================================================

st.title("Finance AI Research Terminal")
st.caption(
    "End-to-end public-company research platform combining market data, "
    "financial statements, scoring, AI research, news intelligence, "
    "peer comparison, investment recommendations, and DCF valuation."
)

if not main_ticker:
    st.warning("Enter a ticker in the sidebar to begin.")
    st.stop()


# =============================================================================
# MAIN NAVIGATION
# =============================================================================

(
    overview_tab,
    statements_tab,
    trends_tab,
    analyst_tab,
    research_tab,
    news_tab,
    recommendation_tab,
    comparison_tab,
    dcf_tab,
    about_tab,
) = st.tabs(
    (
        "Company Overview",
        "Financial Statements",
        "Financial Trends",
        "Analyst Estimates",
        "AI Research",
        "News Intelligence",
        "Investment Recommendation",
        "Company Comparison",
        "DCF Valuation",
        "About Project",
    )
)


# =============================================================================
# COMPANY OVERVIEW TAB
# =============================================================================

with overview_tab:
    st.header("Company Overview")

    if st.button("Analyze Company", type="primary", key="overview_analyze_button"):
        try:
            with st.spinner(f"Loading market and financial data for {main_ticker}..."):
                company_data = get_company_data(main_ticker)
                scores = calculate_scores(company_data)
                history = get_price_history(main_ticker, history_period)

            st.session_state.analysis_data = company_data
            st.session_state.analysis_scores = scores
            st.session_state.analysis_history = history

            # Clear ticker-dependent generated outputs when a new company is analyzed.
            st.session_state.ai_report = None
            st.session_state.news_items = None
            st.session_state.ai_news_report = None
            st.session_state.investment_recommendation = None

        except Exception as exc:
            st.error(f"Could not analyze {main_ticker}: {exc}")

    company_data = st.session_state.analysis_data
    scores = st.session_state.analysis_scores
    history = st.session_state.analysis_history

    if company_data and clean_ticker(str(first_present(company_data, "ticker", default=""))) == main_ticker:
        company_name = first_present(company_data, "company_name", "name", default=main_ticker)
        sector = first_present(company_data, "sector", default="N/A")
        industry = first_present(company_data, "industry", default="N/A")

        st.subheader(company_name)
        st.write(f"**Ticker:** {main_ticker} · **Sector:** {sector} · **Industry:** {industry}")

        current_price = first_present(company_data, "current_price", "regular_market_price")
        market_cap = first_present(company_data, "market_cap")
        one_year_return = first_present(company_data, "one_year_return", "one_year_change")
        trailing_pe = first_present(company_data, "trailing_pe")

        headline_metrics = st.columns(4)
        headline_metrics[0].metric("Current Price", format_currency(current_price))
        headline_metrics[1].metric("Market Cap", format_large_number(market_cap))
        headline_metrics[2].metric("1-Year Return", format_percent(one_year_return))
        headline_metrics[3].metric("Trailing P/E", format_number(trailing_pe))

        st.subheader("Financial Scorecard")
        render_scorecard(scores or {})

        st.subheader(f"{history_period.upper()} Price History")
        render_price_chart(history if isinstance(history, pd.DataFrame) else pd.DataFrame(), main_ticker)

        st.subheader("Financial Overview")
        render_financial_overview(company_data)

        st.divider()
        st.subheader("Export Research Report")
        st.caption(
            "Generate a professional PDF containing company data, "
            "financial metrics, analyst estimates, AI research, "
            "recommendations, financial trends, and available DCF results."
        )

        if st.button(
            "Prepare Research Report PDF",
            key=f"prepare_pdf_report_{main_ticker}",
        ):
            try:
                with st.spinner(
                    f"Preparing the {main_ticker} research report..."
                ):
                    report_analyst_data = get_analyst_estimates(
                        main_ticker
                    )

                    report_financial_trends = get_financial_trends(
                        main_ticker
                    )

                    raw_recommendation = (
                        st.session_state.investment_recommendation
                    )
                    report_recommendation = as_mapping(
                        raw_recommendation
                    )

                    if (
                        not report_recommendation
                        and raw_recommendation is not None
                    ):
                        report_recommendation = {
                            "summary": str(raw_recommendation)
                        }

                    raw_dcf_result = st.session_state.get(
                        "dcf_result"
                    )
                    raw_dcf_company = st.session_state.get(
                        "dcf_company_data"
                    )

                    report_dcf_data = {}

                    if isinstance(raw_dcf_company, Mapping):
                        report_dcf_ticker = clean_ticker(
                            str(
                                first_present(
                                    raw_dcf_company,
                                    "ticker",
                                    "symbol",
                                    default="",
                                )
                            )
                        )

                        if report_dcf_ticker == main_ticker:
                            report_dcf_data = as_mapping(
                                raw_dcf_result
                            )

                    raw_ai_report = st.session_state.get(
                        "ai_report"
                    )
                    raw_news_report = st.session_state.get(
                        "ai_news_report"
                    )

                    report_ai_text = (
                        None
                        if raw_ai_report is None
                        else str(raw_ai_report)
                    )

                    report_news_text = (
                        None
                        if raw_news_report is None
                        else str(raw_news_report)
                    )

                    report_pdf = generate_research_report_pdf(
                        ticker=main_ticker,
                        company_data=company_data,
                        scores=scores or {},
                        analyst_data=report_analyst_data,
                        recommendation=report_recommendation,
                        dcf_data=report_dcf_data,
                        financial_trends=report_financial_trends,
                        ai_research=report_ai_text,
                        news_summary=report_news_text,
                    )

                    st.session_state[
                        "research_report_pdf"
                    ] = report_pdf

                    st.session_state[
                        "research_report_ticker"
                    ] = main_ticker

                st.success(
                    f"{main_ticker} research report is ready."
                )

            except Exception as exc:
                st.session_state[
                    "research_report_pdf"
                ] = None

                st.error(
                    f"Could not prepare the PDF report: {exc}"
                )

        saved_report = st.session_state.get(
            "research_report_pdf"
        )
        saved_report_ticker = st.session_state.get(
            "research_report_ticker"
        )

        if (
            isinstance(saved_report, bytes)
            and saved_report
            and saved_report_ticker == main_ticker
        ):
            st.download_button(
                label="Download Full Research Report PDF",
                data=saved_report,
                file_name=(
                    f"{main_ticker}_finance_ai_research_report.pdf"
                ),
                mime="application/pdf",
                key=f"download_pdf_report_{main_ticker}",
                type="primary",
            )

    else:
        st.info("Click **Analyze Company** to load the full overview.")


# =============================================================================
# FINANCIAL STATEMENTS TAB
# =============================================================================

with statements_tab:
    st.header("Financial Statements")
    st.caption(
        "Review annual or quarterly income statements, balance sheets, "
        "and cash flow statements. Download any statement as a CSV file."
    )

    statement_frequency = st.radio(
        "Statement frequency",
        options=("Annual", "Quarterly"),
        horizontal=True,
        key="statement_frequency",
    )

    if st.button(
        f"Load {statement_frequency} Financial Statements",
        type="primary",
        key="load_financial_statements",
    ):
        try:
            with st.spinner(
                f"Loading {statement_frequency.lower()} financial statements "
                f"for {main_ticker}..."
            ):
                st.session_state["financial_statements"] = get_financial_statements(
                    main_ticker,
                    statement_frequency.lower(),
                )
                st.session_state["financial_statements_ticker"] = main_ticker
                st.session_state["financial_statements_frequency"] = (
                    statement_frequency.lower()
                )
        except Exception as exc:
            st.error(f"Could not load financial statements: {exc}")

    statements = st.session_state.get("financial_statements")
    statements_ticker = st.session_state.get("financial_statements_ticker")
    statements_frequency = st.session_state.get(
        "financial_statements_frequency"
    )

    if statements and statements_ticker == main_ticker:
        income_tab, balance_tab, cash_flow_tab = st.tabs(
            ("Income Statement", "Balance Sheet", "Cash Flow Statement")
        )

        with income_tab:
            render_financial_statement(
                "Income Statement",
                statements.get("Income Statement", pd.DataFrame()),
                main_ticker,
                statements_frequency or "annual",
            )

        with balance_tab:
            render_financial_statement(
                "Balance Sheet",
                statements.get("Balance Sheet", pd.DataFrame()),
                main_ticker,
                statements_frequency or "annual",
            )

        with cash_flow_tab:
            render_financial_statement(
                "Cash Flow Statement",
                statements.get("Cash Flow Statement", pd.DataFrame()),
                main_ticker,
                statements_frequency or "annual",
            )
    else:
        st.info(
            "Select annual or quarterly data and click "
            "**Load Financial Statements**."
        )


# =============================================================================
# FINANCIAL TRENDS TAB
# =============================================================================

with trends_tab:
    st.header("Financial Trends")
    st.caption(
        "Visualize multi-year revenue, profitability, cash flow, EPS, "
        "and margin trends."
    )

    try:
        with st.spinner(f"Loading financial trends for {main_ticker}..."):
            financial_trends = get_financial_trends(main_ticker)
    except Exception as exc:
        financial_trends = pd.DataFrame()
        st.error(f"Could not load financial trends: {exc}")

    if financial_trends.empty:
        st.info("No financial trend data was returned for this ticker.")
    else:
        render_growth_metrics(financial_trends)

        st.markdown("### Revenue and Profitability")
        render_financial_trends_chart(
            financial_trends,
            ["Revenue", "Net Income", "EBITDA"],
            "Revenue, Net Income, and EBITDA",
        )

        st.markdown("### Cash Flow")
        render_financial_trends_chart(
            financial_trends,
            ["Free Cash Flow"],
            "Free Cash Flow Trend",
        )

        st.markdown("### Earnings per Share")
        render_financial_trends_chart(
            financial_trends,
            ["Diluted EPS"],
            "Diluted EPS Trend",
        )

        st.markdown("### Margin Analysis")
        render_financial_trends_chart(
            financial_trends,
            ["Gross Margin", "Operating Margin"],
            "Gross and Operating Margins",
            percent_metrics={"Gross Margin", "Operating Margin"},
        )

        st.download_button(
            label="Download Financial Trends CSV",
            data=financial_trends.to_csv(index=True).encode("utf-8"),
            file_name=f"{main_ticker}_financial_trends.csv",
            mime="text/csv",
            key="download_financial_trends",
        )


# =============================================================================
# ANALYST ESTIMATES TAB
# =============================================================================

with analyst_tab:
    st.header("Analyst Estimates and Price Targets")
    st.caption(
        "Review Wall Street consensus ratings, target-price expectations, "
        "and forward revenue and earnings estimates."
    )

    try:
        with st.spinner(f"Loading analyst estimates for {main_ticker}..."):
            analyst_data = get_analyst_estimates(main_ticker)
    except Exception as exc:
        analyst_data = {}
        st.error(f"Could not load analyst estimates: {exc}")

    current_price = analyst_data.get("current_price")
    target_mean = analyst_data.get("target_mean")
    target_median = analyst_data.get("target_median")
    target_high = analyst_data.get("target_high")
    target_low = analyst_data.get("target_low")
    analyst_count = analyst_data.get("analyst_count")
    recommendation_key = analyst_data.get("recommendation_key")

    implied_upside = None
    if (
        is_number(current_price)
        and is_number(target_mean)
        and float(current_price) != 0
    ):
        implied_upside = (
            (float(target_mean) - float(current_price))
            / float(current_price)
            * 100
        )

    normalized_recommendation = (
        str(recommendation_key)
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
        if recommendation_key
        else "N/A"
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Current Price",
        format_currency(current_price),
    )

    metric_columns[1].metric(
        "Mean Target",
        format_currency(target_mean),
        (
            None
            if implied_upside is None
            else f"{implied_upside:+.1f}% implied"
        ),
    )

    metric_columns[2].metric(
        "Median Target",
        format_currency(target_median),
    )

    metric_columns[3].metric(
        "Consensus",
        normalized_recommendation,
    )

    metric_columns[4].metric(
        "Analyst Coverage",
        (
            "N/A"
            if not is_number(analyst_count)
            else f"{int(float(analyst_count))} analysts"
        ),
    )

    if implied_upside is not None:
        direction = "upside" if implied_upside >= 0 else "downside"
        st.markdown(
            f"**Wall Street takeaway:** The mean analyst target implies "
            f"**{abs(implied_upside):.1f}% {direction}** from the current "
            f"share price."
        )

    valid_target_values = all(
        is_number(value)
        for value in (
            current_price,
            target_low,
            target_mean,
            target_high,
        )
    )

    if valid_target_values:
        current_value = float(current_price)
        low_value = float(target_low)
        mean_value = float(target_mean)
        high_value = float(target_high)

        median_value = (
            float(target_median)
            if is_number(target_median)
            else mean_value
        )

        chart_min = min(low_value, current_value) * 0.94
        chart_max = max(high_value, current_value) * 1.06

        target_chart = go.Figure()

        target_chart.add_trace(
            go.Scatter(
                x=[low_value, high_value],
                y=["Analyst Range", "Analyst Range"],
                mode="lines",
                name="Target Range",
                line={"width": 12},
                hovertemplate=(
                    "Analyst target range<br>"
                    f"Low: {format_currency(low_value)}<br>"
                    f"High: {format_currency(high_value)}"
                    "<extra></extra>"
                ),
            )
        )

        target_chart.add_trace(
            go.Scatter(
                x=[low_value, high_value],
                y=["Analyst Range", "Analyst Range"],
                mode="markers+text",
                name="Low / High",
                marker={"size": 16},
                text=[
                    f"Low {format_currency(low_value)}",
                    f"High {format_currency(high_value)}",
                ],
                textposition=["bottom center", "bottom center"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

        target_chart.add_trace(
            go.Scatter(
                x=[current_value],
                y=["Analyst Range"],
                mode="markers+text",
                name="Current Price",
                marker={
                    "size": 20,
                    "symbol": "diamond",
                },
                text=[f"Current {format_currency(current_value)}"],
                textposition="top center",
                hovertemplate="%{text}<extra></extra>",
            )
        )

        target_chart.add_trace(
            go.Scatter(
                x=[mean_value],
                y=["Analyst Range"],
                mode="markers+text",
                name="Mean Target",
                marker={
                    "size": 20,
                    "symbol": "star",
                },
                text=[f"Mean {format_currency(mean_value)}"],
                textposition="top center",
                hovertemplate="%{text}<extra></extra>",
            )
        )

        if abs(median_value - mean_value) > 0.01:
            target_chart.add_trace(
                go.Scatter(
                    x=[median_value],
                    y=["Analyst Range"],
                    mode="markers+text",
                    name="Median Target",
                    marker={
                        "size": 16,
                        "symbol": "circle",
                    },
                    text=[f"Median {format_currency(median_value)}"],
                    textposition="bottom center",
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        target_chart.update_layout(
            title={"text": ""},
            xaxis_title="Share Price",
            yaxis_title="",
            height=350,
            hovermode="closest",
            margin={"l": 20, "r": 20, "t": 35, "b": 45},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
            font={"size": 13},
        )

        target_chart.update_xaxes(
            range=[chart_min, chart_max],
            tickprefix="$",
            tickformat=",.0f",
            gridcolor="rgba(128, 128, 128, 0.18)",
            zeroline=False,
        )

        target_chart.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        )

        st.markdown("### Price Target Range")
        st.plotly_chart(
            target_chart,
            key=f"analyst_target_range_{main_ticker}",
        )
    else:
        st.info(
            "Complete analyst price-target data was not available "
            "for this ticker."
        )

    st.markdown("### Recommendation Consensus")
    render_analyst_rating_summary(analyst_data)

    estimate_tab_1, estimate_tab_2, estimate_tab_3 = st.tabs(
        (
            "Revenue Estimates",
            "Earnings Estimates",
            "Recommendation History",
        )
    )

    with estimate_tab_1:
        render_estimate_dataframe(
            "Revenue Estimates",
            analyst_data.get(
                "revenue_estimates",
                pd.DataFrame(),
            ),
        )

    with estimate_tab_2:
        render_estimate_dataframe(
            "Earnings Estimates",
            analyst_data.get(
                "earnings_estimates",
                pd.DataFrame(),
            ),
        )

    with estimate_tab_3:
        recommendation_history = analyst_data.get(
            "recommendations",
            pd.DataFrame(),
        )

        if (
            not isinstance(
                recommendation_history,
                pd.DataFrame,
            )
            or recommendation_history.empty
        ):
            st.info(
                "No recommendation history was available."
            )
        else:
            history = recommendation_history.copy()

            if len(history) > 100:
                history = history.tail(100)

            history = history.reset_index()

            st.dataframe(
                history,
                width="stretch",
                hide_index=True,
                height=450,
            )


# =============================================================================
# AI RESEARCH TAB
# =============================================================================

with research_tab:
    st.header("AI Equity Research")

    if st.session_state.analysis_data is None or clean_ticker(
        str(first_present(st.session_state.analysis_data or {}, "ticker", default=""))
    ) != main_ticker:
        st.info("Analyze the company in the Company Overview tab first.")
    else:
        if st.button("Generate AI Research Report", type="primary", key="generate_ai_report"):
            try:
                with st.spinner("Generating the AI equity-research report..."):
                    st.session_state.ai_report = generate_analysis(
                        st.session_state.analysis_data,
                        st.session_state.analysis_scores or {},
                    )
            except Exception as exc:
                st.error(f"Could not generate the AI research report: {exc}")

        if st.session_state.ai_report is not None:
            render_text_result(st.session_state.ai_report)
        else:
            st.info("Generate a report to view the AI investment thesis, risks, and catalysts.")


# =============================================================================
# NEWS TAB
# =============================================================================

with news_tab:
    st.header("Company News Intelligence")

    if st.session_state.analysis_data is None or clean_ticker(
        str(first_present(st.session_state.analysis_data or {}, "ticker", default=""))
    ) != main_ticker:
        st.info("Analyze the company in the Company Overview tab first.")
    else:
        news_button, ai_news_button = st.columns(2)

        with news_button:
            fetch_news = st.button(
                "Fetch Company News",
                type="primary",
                key="fetch_company_news",
                width="stretch",
            )

        with ai_news_button:
            analyze_news_button = st.button(
                "Run AI News Analysis",
                key="run_ai_news_analysis",
                width="stretch",
            )

        if fetch_news:
            try:
                with st.spinner(f"Fetching company news for {main_ticker}..."):
                    st.session_state.news_items = get_news_items(
                        main_ticker,
                        st.session_state.analysis_data,
                    )
                    st.session_state.ai_news_report = None
            except Exception as exc:
                st.error(f"Could not fetch company news: {exc}")

        if analyze_news_button:
            try:
                if st.session_state.news_items is None:
                    with st.spinner(f"Fetching company news for {main_ticker}..."):
                        st.session_state.news_items = get_news_items(
                            main_ticker,
                            st.session_state.analysis_data,
                        )

                with st.spinner("Analyzing news sentiment, catalysts, and risks..."):
                    st.session_state.ai_news_report = generate_news_analysis(
                        st.session_state.analysis_data,
                        st.session_state.news_items,
                    )
            except Exception as exc:
                st.error(f"Could not run the AI news analysis: {exc}")

        news_list_tab, news_analysis_tab = st.tabs(("Latest Articles", "AI News Analysis"))

        with news_list_tab:
            if st.session_state.news_items is not None:
                render_news_cards(st.session_state.news_items)
            else:
                st.info("Fetch company news to display the latest articles.")

        with news_analysis_tab:
            if st.session_state.ai_news_report is not None:
                render_text_result(st.session_state.ai_news_report)
            else:
                st.info("Run AI News Analysis to summarize sentiment, catalysts, and risks.")


# =============================================================================
# INVESTMENT RECOMMENDATION TAB
# =============================================================================

with recommendation_tab:
    st.header("Investment Recommendation")

    if st.session_state.analysis_data is None or clean_ticker(
        str(first_present(st.session_state.analysis_data or {}, "ticker", default=""))
    ) != main_ticker:
        st.info("Analyze the company in the Company Overview tab first.")
    else:
        st.warning(
            "This output is a research aid, not personalized financial advice. "
            "Verify all market data and assumptions before making an investment decision."
        )

        if st.button(
            "Generate Investment Recommendation",
            type="primary",
            key="generate_investment_recommendation",
        ):
            try:
                with st.spinner("Building the investment recommendation..."):
                    if st.session_state.ai_report is None:
                        st.session_state.ai_report = generate_analysis(
                            st.session_state.analysis_data,
                            st.session_state.analysis_scores or {},
                        )

                    if st.session_state.news_items is None:
                        try:
                            st.session_state.news_items = get_news_items(
                                main_ticker,
                                st.session_state.analysis_data,
                            )
                        except Exception:
                            st.session_state.news_items = []

                    if st.session_state.ai_news_report is None and st.session_state.news_items:
                        try:
                            st.session_state.ai_news_report = generate_news_analysis(
                                st.session_state.analysis_data,
                                st.session_state.news_items,
                            )
                        except Exception:
                            st.session_state.ai_news_report = None

                    st.session_state.investment_recommendation = generate_recommendation(
                        st.session_state.analysis_data,
                        st.session_state.analysis_scores or {},
                        st.session_state.ai_report,
                        st.session_state.ai_news_report,
                    )
            except Exception as exc:
                st.error(f"Could not generate the investment recommendation: {exc}")

        if st.session_state.investment_recommendation is not None:
            render_text_result(st.session_state.investment_recommendation)
        else:
            st.info(
                "Generate the recommendation to combine market data, financial scoring, "
                "AI equity research, and available news intelligence."
            )


# =============================================================================
# COMPANY COMPARISON TAB
# =============================================================================

with comparison_tab:
    st.header("Comparable Company Analysis")
    st.caption(
        "Compare two to five peers across valuation, growth, profitability, "
        "financial strength, and market performance versus a benchmark."
    )

    peer_input_column, benchmark_input_column = st.columns([3, 1])
    with peer_input_column:
        default_peer_text = (
            "NVDA, AMD, AVGO"
            if main_ticker in {"NVDA", "AMD", "AVGO"}
            else f"{main_ticker}, AMD, AVGO"
        )
        peer_ticker_text = st.text_input(
            "Peer tickers",
            value=default_peer_text,
            help="Enter 2-5 ticker symbols separated by commas.",
            key="comparison_peer_tickers",
        )

    with benchmark_input_column:
        benchmark_ticker = clean_ticker(
            st.text_input(
                "Market benchmark",
                value="SPY",
                help="Examples: SPY, QQQ, or ^GSPC.",
                key="comparison_benchmark_ticker",
            )
        )

    peer_tickers: list[str] = []
    for ticker_value in re.split(r"[\s,;]+", peer_ticker_text):
        ticker = clean_ticker(ticker_value)
        if ticker and ticker not in peer_tickers:
            peer_tickers.append(ticker)

    if st.button(
        "Run Comparable Analysis",
        type="primary",
        key="compare_companies_button",
    ):
        if len(peer_tickers) < 2:
            st.error("Enter at least two different company tickers.")
        elif len(peer_tickers) > 5:
            st.error("Enter no more than five company tickers.")
        elif not benchmark_ticker:
            st.error("Enter a benchmark ticker.")
        else:
            try:
                with st.spinner(
                    "Loading peer fundamentals, financial scores, and market returns..."
                ):
                    companies = [get_company_data(ticker) for ticker in peer_tickers]
                    score_map = {
                        ticker: calculate_scores(company)
                        for ticker, company in zip(peer_tickers, companies)
                    }
                    return_map = {
                        ticker: get_multi_period_returns(ticker)
                        for ticker in peer_tickers
                    }
                    benchmark_returns = get_multi_period_returns(benchmark_ticker)

                    comparison_result = run_comparison(
                        companies,
                        score_map,
                        return_map,
                        benchmark_ticker,
                        benchmark_returns,
                    )

                st.session_state.comparison_result = comparison_result
                st.session_state.comparison_companies = {
                    "companies": companies,
                    "score_map": score_map,
                    "tickers": peer_tickers,
                    "benchmark_ticker": benchmark_ticker,
                }
                st.session_state.ai_comparison_result = None

            except Exception as exc:
                st.error(f"Could not complete the comparable analysis: {exc}")

    comparison_state = st.session_state.comparison_companies
    comparison_result = st.session_state.comparison_result

    current_comparison_ready = (
        isinstance(comparison_state, Mapping)
        and isinstance(comparison_result, Mapping)
        and "companies" in comparison_state
        and "metrics" in comparison_result
    )

    if current_comparison_ready:
        companies = comparison_state["companies"]
        score_map = comparison_state["score_map"]
        active_tickers = comparison_state["tickers"]
        active_benchmark = comparison_state["benchmark_ticker"]

        st.subheader("Peer Group Snapshot")
        company_columns = st.columns(len(active_tickers))
        for column, ticker, company in zip(
            company_columns,
            active_tickers,
            companies,
        ):
            with column:
                st.markdown(f"### {ticker}")
                st.caption(
                    str(first_present(company, "company_name", "name", default=ticker))
                )
                st.metric(
                    "Financial Score",
                    f"{score_map.get(ticker, {}).get('overall', 'N/A')} / 100",
                )
                st.metric(
                    "Current Price",
                    format_currency(first_present(company, "current_price")),
                )
                st.metric(
                    "Market Cap",
                    format_large_number(first_present(company, "market_cap")),
                )

        (
            comp_summary_tab,
            comp_metrics_tab,
            comp_valuation_tab,
            comp_performance_tab,
            ai_comparison_tab,
        ) = st.tabs(
            (
                "Ranking Summary",
                "Fundamental Metrics",
                "Valuation vs Peers",
                "Market Performance",
                "AI Analyst Comparison",
            )
        )

        with comp_summary_tab:
            leader_columns = st.columns(5)
            for column, (label, ticker) in zip(
                leader_columns,
                comparison_result.get("leaders", {}).items(),
            ):
                with column:
                    st.metric(label, ticker or "N/A")

            st.markdown("#### Peer-Relative Category Scores")
            st.caption(
                "Scores rank each company against this selected peer group. "
                "They are not absolute investment ratings."
            )
            category_score_df = MODULES[
                "comparison"
            ].category_scores_to_dataframe(comparison_result)
            score_columns = [
                ticker for ticker in active_tickers if ticker in category_score_df.columns
            ]

            def color_peer_score(value: Any) -> str:
                if not is_number(value):
                    return ""
                if float(value) >= 66:
                    return "background-color: rgba(34, 197, 94, 0.22)"
                if float(value) >= 33:
                    return "background-color: rgba(234, 179, 8, 0.20)"
                return "background-color: rgba(239, 68, 68, 0.18)"

            category_score_style = category_score_df.style.format(
                {ticker: "{:.1f}" for ticker in score_columns},
                na_rep="N/A",
            ).map(color_peer_score, subset=score_columns)
            st.dataframe(
                category_score_style,
                width="stretch",
                hide_index=True,
            )

            radar_categories = [
                "Valuation",
                "Growth",
                "Profitability",
                "Financial Strength",
                "Market Performance",
            ]
            radar_figure = go.Figure()
            for ticker in active_tickers:
                radar_values = [
                    comparison_result.get("category_scores", {})
                    .get(category, {})
                    .get(ticker)
                    or 0
                    for category in radar_categories
                ]
                radar_figure.add_trace(
                    go.Scatterpolar(
                        r=radar_values + [radar_values[0]],
                        theta=radar_categories + [radar_categories[0]],
                        fill="toself",
                        name=ticker,
                        opacity=0.68,
                    )
                )
            radar_figure.update_layout(
                polar={"radialaxis": {"visible": True, "range": [0, 100]}},
                margin={"l": 40, "r": 40, "t": 30, "b": 30},
                height=500,
            )
            st.plotly_chart(radar_figure, width="stretch")

        with comp_metrics_tab:
            metric_categories = (
                "Valuation",
                "Growth",
                "Profitability",
                "Financial Strength",
            )
            metric_tabs = st.tabs(metric_categories)
            for metric_tab, category in zip(metric_tabs, metric_categories):
                with metric_tab:
                    metric_df = MODULES["comparison"].comparison_to_dataframe(
                        comparison_result,
                        category=category,
                        formatted=True,
                    )
                    metric_lookup = {
                        metric.get("label"): metric.get("best_ticker")
                        for metric in comparison_result.get("metrics", [])
                        if metric.get("category") == category
                    }

                    def highlight_best_metric(row: pd.Series) -> list[str]:
                        styles = [""] * len(row)
                        best_ticker = metric_lookup.get(row.get("Metric"))
                        if best_ticker in row.index:
                            styles[row.index.get_loc(best_ticker)] = (
                                "background-color: rgba(34, 197, 94, 0.22); "
                                "font-weight: 600"
                            )
                        return styles

                    st.dataframe(
                        metric_df.style.apply(highlight_best_metric, axis=1),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Green identifies the strongest reported value for each "
                        "metric within the selected peer group."
                    )

            full_export_df = MODULES["comparison"].comparison_to_dataframe(
                comparison_result,
                formatted=False,
            )
            st.download_button(
                "Download Comparison CSV",
                data=full_export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{'_'.join(active_tickers)}_comparable_analysis.csv",
                mime="text/csv",
                key="download_comparable_analysis",
            )

        with comp_valuation_tab:
            st.markdown("#### Valuation Premium / (Discount) to Peer Median")
            st.caption(
                "A negative percentage indicates a discount to the peer median. "
                "Value Rank 1 is the lowest positive multiple."
            )
            valuation_df = MODULES[
                "comparison"
            ].valuation_premium_to_median_dataframe(comparison_result)
            if valuation_df.empty:
                st.info("No comparable valuation multiples were available.")
            else:
                st.dataframe(
                    valuation_df,
                    width="stretch",
                    hide_index=True,
                )

        with comp_performance_tab:
            performance_df = MODULES["comparison"].comparison_to_dataframe(
                comparison_result,
                category="Market Performance",
                formatted=True,
            )
            st.dataframe(
                performance_df,
                width="stretch",
                hide_index=True,
            )

            performance_period = st.selectbox(
                "Normalized price chart period",
                options=("1y", "3y", "5y"),
                index=1,
                key="comparison_performance_period",
            )
            performance_figure = go.Figure()
            for ticker in active_tickers + [active_benchmark]:
                ticker_history = get_price_history(ticker, performance_period)
                if ticker_history.empty or "Close" not in ticker_history.columns:
                    continue
                close = pd.to_numeric(
                    ticker_history["Close"],
                    errors="coerce",
                ).dropna()
                if close.empty or float(close.iloc[0]) == 0:
                    continue
                normalized = close / float(close.iloc[0]) * 100
                performance_figure.add_trace(
                    go.Scatter(
                        x=normalized.index,
                        y=normalized.values,
                        mode="lines",
                        name=ticker,
                        line={
                            "width": 3 if ticker == active_benchmark else 2,
                            "dash": "dash" if ticker == active_benchmark else "solid",
                        },
                    )
                )
            performance_figure.update_layout(
                title=f"Normalized Price Performance (Start = 100) vs {active_benchmark}",
                xaxis_title="Date",
                yaxis_title="Indexed Value",
                hovermode="x unified",
                height=500,
                margin={"l": 40, "r": 30, "t": 60, "b": 40},
            )
            st.plotly_chart(performance_figure, width="stretch")

        with ai_comparison_tab:
            st.caption(
                "The AI report uses the calculated peer medians, relative rankings, "
                "and benchmark returns shown in the other tabs."
            )
            if st.button(
                "Generate AI Comparable Analysis",
                type="primary",
                key="generate_ai_comparison_button",
            ):
                try:
                    with st.spinner("Writing the equity-research comparison..."):
                        ai_result = run_ai_comparison(
                            companies,
                            comparison_result,
                        )
                    if isinstance(ai_result, str):
                        ai_result = ai_result.strip()
                    st.session_state.ai_comparison_result = (
                        ai_result
                        or "The AI comparison completed but returned no written analysis."
                    )
                except Exception as exc:
                    st.error(f"Could not generate the AI comparison: {exc}")

            if st.session_state.ai_comparison_result:
                render_text_result(st.session_state.ai_comparison_result)
            else:
                st.info(
                    "Generate the report after reviewing the calculated peer comparison."
                )
    else:
        st.info(
            "Enter two to five company tickers and a market benchmark, then run "
            "the comparable analysis."
        )


# =============================================================================
# DCF TAB
# =============================================================================

with dcf_tab:
    st.header("Discounted Cash Flow Valuation")

    dcf_ticker = clean_ticker(
        st.text_input(
            "DCF ticker",
            value=main_ticker,
            key="dcf_ticker",
        )
    )

    assumption_columns = st.columns(4)

    with assumption_columns[0]:
        growth_rate_percent = st.number_input(
            "Annual FCF Growth (%)",
            min_value=-50.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            key="dcf_growth_rate",
        )

    with assumption_columns[1]:
        discount_rate_percent = st.number_input(
            "Discount Rate / WACC (%)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=0.25,
            key="dcf_discount_rate",
        )

    with assumption_columns[2]:
        terminal_growth_percent = st.number_input(
            "Terminal Growth (%)",
            min_value=-5.0,
            max_value=10.0,
            value=3.0,
            step=0.25,
            key="dcf_terminal_growth_rate",
        )

    with assumption_columns[3]:
        projection_years = st.number_input(
            "Projection Years",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="dcf_projection_years",
        )

    growth_rate = growth_rate_percent / 100
    discount_rate = discount_rate_percent / 100
    terminal_growth_rate = terminal_growth_percent / 100

    if terminal_growth_rate >= discount_rate:
        st.error("Terminal growth must be lower than the discount rate.")

    run_dcf_button = st.button(
        "Run Complete DCF Analysis",
        type="primary",
        key="run_complete_dcf",
        disabled=terminal_growth_rate >= discount_rate or not dcf_ticker,
    )

    if run_dcf_button:
        try:
            with st.spinner(f"Running DCF valuation for {dcf_ticker}..."):
                dcf_company_data = get_company_data(dcf_ticker)

                base_result = run_dcf(
                    dcf_company_data,
                    growth_rate,
                    discount_rate,
                    terminal_growth_rate,
                    int(projection_years),
                )

                scenarios_result = run_dcf_scenarios(
                    dcf_company_data,
                    growth_rate,
                    discount_rate,
                    terminal_growth_rate,
                    int(projection_years),
                )

                sensitivity_result = run_dcf_sensitivity(
                    dcf_company_data,
                    growth_rate,
                    discount_rate,
                    terminal_growth_rate,
                    int(projection_years),
                )

                st.session_state.dcf_company_data = dcf_company_data
                st.session_state.dcf_result = base_result
                st.session_state.dcf_scenarios_result = scenarios_result
                st.session_state.dcf_sensitivity_result = sensitivity_result

        except Exception as exc:
            st.error(f"Could not complete the DCF analysis: {exc}")

    if st.session_state.dcf_result is not None and st.session_state.dcf_company_data:
        dcf_company_data = st.session_state.dcf_company_data
        current_price = first_present(
            dcf_company_data,
            "current_price",
            "regular_market_price",
        )

        base_tab, scenarios_tab, sensitivity_tab = st.tabs(
            ("Base Case", "Scenarios", "Sensitivity")
        )

        with base_tab:
            render_dcf_result(
                st.session_state.dcf_result,
                current_price,
                title=f"{first_present(dcf_company_data, 'company_name', default=dcf_ticker)} DCF",
            )

        with scenarios_tab:
            st.subheader("Bull, Base, and Bear Scenarios")
            render_scenario_result(st.session_state.dcf_scenarios_result)

        with sensitivity_tab:
            st.subheader("DCF Sensitivity Table")
            render_sensitivity_result(st.session_state.dcf_sensitivity_result)
    else:
        st.info(
            "Set the assumptions and click **Run Complete DCF Analysis** to calculate "
            "the base valuation, scenarios, and sensitivity table."
        )


# =============================================================================
# ABOUT PROJECT TAB
# =============================================================================

with about_tab:
    st.header("About This Project")
    st.caption(
        "A portfolio-ready equity research platform combining financial "
        "analysis, valuation, market intelligence, and AI-assisted research."
    )

    overview_column, stack_column = st.columns(2)

    with overview_column:
        st.subheader("Project Overview")
        st.write(
            "The Finance AI Research Terminal is an end-to-end public-equity "
            "analysis application. Users can enter a ticker and review company "
            "fundamentals, price performance, financial statements, analyst "
            "expectations, AI research, news, peer comparisons, investment "
            "recommendations, and DCF valuation outputs."
        )

        st.subheader("Core Capabilities")
        st.markdown(
            """
- Public-company market and fundamental data
- Financial scoring across profitability, valuation, balance sheet, and performance
- Annual and quarterly financial statements
- Multi-year revenue, earnings, margin, EPS, and cash-flow trends
- Analyst ratings, price targets, and forward estimates
- AI-generated company research and investment recommendations
- News intelligence and sentiment analysis
- Peer-company comparison
- DCF valuation, scenarios, and sensitivity analysis
- CSV exports for financial statements and trend data
"""
        )

        st.subheader("Platform Highlights")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Modules", "9")
        col2.metric("AI Reports", "3")
        col3.metric("DCF Models", "3")
        col4.metric("Exports", "CSV")

    with stack_column:
        st.subheader("Technology Stack")
        technology_stack = pd.DataFrame(
            {
                "Technology": [
                    "Python",
                    "Streamlit",
                    "yfinance",
                    "Pandas",
                    "Plotly",
                    "OpenAI API",
                ],
                "Purpose": [
                    "Application logic and financial calculations",
                    "Interactive web interface",
                    "Market and company data",
                    "Data transformation and analysis",
                    "Interactive financial visualizations",
                    "AI research and recommendation generation",
                ],
            }
        )
        st.dataframe(
            technology_stack,
            width="stretch",
            hide_index=True,
        )

        st.subheader("Data Sources")

        st.markdown(
            """
- **Yahoo Finance (yfinance)** — Market prices, company fundamentals, financial statements, analyst estimates, and historical pricing
- **OpenAI API** — AI-generated equity research, investment recommendations, and news intelligence
- **Plotly** — Interactive financial charts and visualizations
"""
        )

        st.subheader("Analytical Methods")
        st.markdown(
            """
- Fundamental ratio analysis
- Profitability and balance-sheet scoring
- Relative valuation
- Historical trend analysis
- Analyst-consensus analysis
- Discounted cash flow valuation
- Scenario and sensitivity analysis
- AI-assisted qualitative research
"""
        )

    st.divider()
    st.subheader("Methodology")

    methodology_one, methodology_two, methodology_three = st.columns(3)

    with methodology_one:
        st.markdown("#### 1. Data Collection")
        st.write(
            "Collects market prices, company fundamentals, financial "
            "statements, analyst estimates, and news from external providers."
        )

    with methodology_two:
        st.markdown("#### 2. Financial Analysis")
        st.write(
            "Transforms raw data into standardized metrics, growth rates, "
            "financial scores, trend charts, comparisons, and valuations."
        )

    st.divider()

    st.subheader("Important Notes")

    st.info(
        """
Finance AI Research Terminal is designed as an educational and equity research platform.

Market data is provided by external sources and may occasionally be delayed or incomplete.

AI-generated research should complement—not replace—independent financial analysis and investment due diligence.
"""
    )

    with methodology_three:
        st.markdown("#### 3. AI Interpretation")
        st.write(
            "Summarizes performance, identifies risks and catalysts, evaluates "
            "news, and produces structured investment commentary."
        )

    st.divider()
    st.subheader("Project Metrics")

    project_metric_one, project_metric_two, project_metric_three, project_metric_four = st.columns(4)

    with project_metric_one:
        integrated_module_count = sum(
            MODULES.get(name) is not None
            for name in MODULE_NAMES
        )
        st.metric(
            "Integrated Modules",
            f"{integrated_module_count}/{len(MODULE_NAMES)}",
        )

    with project_metric_two:
        st.metric("Research Sections", "10")

    with project_metric_three:
        st.metric("Valuation Tools", "3")

    with project_metric_four:
        st.metric("Export Format", "CSV")

    st.divider()
    st.subheader("Data and Model Disclosures")
    st.info(
        "Market and company data may be delayed, incomplete, or differently "
        "defined across providers. Analyst estimates change frequently. "
        "AI-generated research may contain errors and is not personalized "
        "investment advice."
    )



# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(
    "Finance AI Research Terminal · Data may be delayed or incomplete. "
    "AI-generated analysis can be wrong and should be independently verified."
)
