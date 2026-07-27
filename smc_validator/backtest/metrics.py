import pandas as pd


def compute_metrics(trades: pd.DataFrame) -> dict:
    """Win rate alone is insufficient with staged partial exits (per Phase 1
    plan) — expectancy, the full R-distribution, profit factor, max
    drawdown (in R), and trades/month (a tradeability check) all matter.
    """
    if len(trades) == 0:
        return {"n_trades": 0}

    wins = trades["total_r"] > 0
    gross_win = trades.loc[wins, "total_r"].sum()
    gross_loss = -trades.loc[~wins, "total_r"].sum()

    equity = trades["total_r"].cumsum()
    drawdown = equity - equity.cummax()

    span = trades["signal_time"].max() - trades["signal_time"].min()
    span_days = span.days if hasattr(span, "days") else float(span)
    trades_per_month = len(trades) / (span_days / 30.0) if span_days > 0 else float("nan")

    return {
        "n_trades": len(trades),
        "win_rate": float(wins.mean()),
        "expectancy_r": float(trades["total_r"].mean()),
        "avg_win_r": float(trades.loc[wins, "total_r"].mean()) if wins.any() else 0.0,
        "avg_loss_r": float(trades.loc[~wins, "total_r"].mean()) if (~wins).any() else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_r": float(drawdown.min()),
        "trades_per_month": trades_per_month,
        "r_distribution": trades["total_r"].describe().to_dict(),
    }
