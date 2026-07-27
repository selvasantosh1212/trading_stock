from smc_validator.structure.internal_external import label_internal_external


def test_choppy_dip_labeled_internal_not_external(load_fixture):
    df = load_fixture("choppy_internal_leg.csv")
    labels = label_internal_external(
        df, internal_left=1, internal_right=1, external_left=3, external_right=3
    )

    assert labels["structure_label"].iloc[4] == "internal"
    assert not labels["external_pivot_low"].iloc[4]


def test_choppy_wiggle_labeled_internal_not_external(load_fixture):
    df = load_fixture("choppy_internal_leg.csv")
    labels = label_internal_external(
        df, internal_left=1, internal_right=1, external_left=3, external_right=3
    )

    assert labels["structure_label"].iloc[7] == "internal"
    assert not labels["external_pivot_high"].iloc[7]


def test_a_pivot_confirmed_at_both_scales_is_labeled_external(load_fixture):
    df = load_fixture("choppy_internal_leg.csv")
    labels = label_internal_external(
        df, internal_left=1, internal_right=1, external_left=1, external_right=1
    )

    # with identical left/right params, everything internal also qualifies as
    # external, so nothing should be classified as internal-only noise
    assert (labels["structure_label"] == "internal").sum() == 0
