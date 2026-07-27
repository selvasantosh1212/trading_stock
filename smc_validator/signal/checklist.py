import pandas as pd


def assemble_signals(
    htf_bias: pd.Series,
    ltf_confirmation: pd.Series,
    zone_direction: pd.Series,
    entry_price: pd.Series,
    stop_price: pd.Series,
    target1_price: pd.Series,
    target2_price: pd.Series,
) -> pd.DataFrame:
    """Final entry checklist assembly (Direction -> Liquidity targets ->
    Location -> Confirmation -> Entry, per the source's full checklist).
    Every upstream piece (bias, zone, confirmation, targets) is computed by
    its own dedicated, independently-tested module — this only combines
    them into the final tradeable signal. A signal fires only when LTF
    confirmation has occurred AND the zone's own direction still agrees with
    the current HTF bias (guards against acting on a zone left over from a
    bias that has since flipped).
    """
    zone_matches_bias = (zone_direction == htf_bias) & htf_bias.notna()
    fires = ltf_confirmation & zone_matches_bias

    signals = pd.DataFrame(index=htf_bias.index)
    signals["direction"] = htf_bias.where(fires)
    signals["entry"] = entry_price.where(fires)
    signals["stop"] = stop_price.where(fires)
    signals["target1"] = target1_price.where(fires)
    signals["target2"] = target2_price.where(fires)
    return signals.dropna(subset=["direction"])
