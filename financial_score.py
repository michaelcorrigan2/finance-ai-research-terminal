from typing import Any, Dict


def clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def score_profitability(data: Dict[str, Any]) -> int:
    score = 0.0

    profit_margin = data.get("profit_margin")
    operating_margin = data.get("operating_margin")
    return_on_equity = data.get("return_on_equity")
    return_on_assets = data.get("return_on_assets")

    if profit_margin is not None:
        score += max(0, min(profit_margin / 0.30, 1)) * 30
    if operating_margin is not None:
        score += max(0, min(operating_margin / 0.30, 1)) * 30
    if return_on_equity is not None:
        score += max(0, min(return_on_equity / 0.25, 1)) * 25
    if return_on_assets is not None:
        score += max(0, min(return_on_assets / 0.15, 1)) * 15

    return clamp_score(score)


def score_balance_sheet(data: Dict[str, Any]) -> int:
    score = 0.0

    total_cash = data.get("total_cash")
    total_debt = data.get("total_debt")
    debt_to_equity = data.get("debt_to_equity")
    free_cash_flow = data.get("free_cash_flow")

    if total_cash is not None and total_debt is not None:
        if total_cash >= total_debt:
            score += 40
        elif total_debt > 0:
            score += max(0, 40 * (total_cash / total_debt))

    if debt_to_equity is not None and debt_to_equity >= 0:
        # market_data.py already normalizes Yahoo's percentage into a ratio.
        # For example, Yahoo's 65 becomes 0.65x before reaching this function.
        debt_to_equity_ratio = debt_to_equity

        if debt_to_equity_ratio <= 0.50:
            score += 30
        elif debt_to_equity_ratio <= 1.00:
            score += 20
        elif debt_to_equity_ratio <= 2.00:
            score += 10

    if free_cash_flow is not None and free_cash_flow > 0:
        score += 30

    return clamp_score(score)


def score_valuation(data: Dict[str, Any]) -> int:
    score = 0.0

    trailing_pe = data.get("trailing_pe")
    forward_pe = data.get("forward_pe")
    peg_ratio = data.get("peg_ratio")

    # Negative multiples generally indicate negative earnings, not cheapness.
    if trailing_pe is not None and trailing_pe > 0:
        if trailing_pe <= 15:
            score += 35
        elif trailing_pe <= 25:
            score += 28
        elif trailing_pe <= 35:
            score += 20
        elif trailing_pe <= 50:
            score += 10

    if forward_pe is not None and forward_pe > 0:
        if forward_pe <= 15:
            score += 35
        elif forward_pe <= 25:
            score += 28
        elif forward_pe <= 35:
            score += 20
        elif forward_pe <= 50:
            score += 10

    if peg_ratio is not None and peg_ratio > 0:
        if peg_ratio <= 1:
            score += 30
        elif peg_ratio <= 1.5:
            score += 20
        elif peg_ratio <= 2:
            score += 10

    return clamp_score(score)


def score_market_performance(data: Dict[str, Any]) -> int:
    one_year_return = data.get("one_year_return")
    beta = data.get("beta")

    score = 50.0

    if one_year_return is not None:
        if one_year_return >= 0.30:
            score += 35
        elif one_year_return >= 0.15:
            score += 25
        elif one_year_return >= 0:
            score += 10
        elif one_year_return <= -0.20:
            score -= 30
        else:
            score -= 15

    if beta is not None:
        if beta <= 1:
            score += 15
        elif beta <= 1.5:
            score += 8
        elif beta >= 2:
            score -= 10

    return clamp_score(score)


def calculate_financial_scores(data: Dict[str, Any]) -> Dict[str, int]:
    profitability = score_profitability(data)
    balance_sheet = score_balance_sheet(data)
    valuation = score_valuation(data)
    market_performance = score_market_performance(data)

    overall = round(
        profitability * 0.35
        + balance_sheet * 0.30
        + valuation * 0.20
        + market_performance * 0.15
    )

    return {
        "profitability": profitability,
        "balance_sheet": balance_sheet,
        "valuation": valuation,
        "market_performance": market_performance,
        "overall": overall,
    }
