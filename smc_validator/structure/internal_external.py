import pandas as pd

from smc_validator.structure.swings import detect_fractal_pivots


def label_internal_external(
    df: pd.DataFrame,
    internal_left: int,
    internal_right: int,
    external_left: int,
    external_right: int,
) -> pd.DataFrame:
    """Two-tier pivot classification: a small-lookback (internal) pivot that
    doesn't also register under the larger (external) lookback is noise
    inside the current leg. Per the source rule, only external/swing
    structure should be used to frame trend bias — internal pivots are
    identified here so callers can exclude them, not to compute their own
    trend state.
    """
    internal = detect_fractal_pivots(df, internal_left, internal_right)
    external = detect_fractal_pivots(df, external_left, external_right)

    is_internal_pivot = internal["pivot_high"] | internal["pivot_low"]
    is_external_pivot = external["pivot_high"] | external["pivot_low"]

    label = pd.Series("none", index=df.index, dtype="object")
    label[is_internal_pivot & ~is_external_pivot] = "internal"
    label[is_external_pivot] = "external"

    return pd.DataFrame(
        {
            "internal_pivot_high": internal["pivot_high"],
            "internal_pivot_low": internal["pivot_low"],
            "external_pivot_high": external["pivot_high"],
            "external_pivot_low": external["pivot_low"],
            "structure_label": label,
        },
        index=df.index,
    )
