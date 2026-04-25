"""
logic.py — FairScan core audit & mitigation functions.

run_audit        → always returns EXACTLY 4 values:
                   (protected_rate: float, reference_rate: float,
                    ratio: float, is_weighted: bool)

apply_reweighing → returns EXACTLY 4 values (one weight per group×outcome cell):
                   (w_prot_pos: float, w_prot_neg: float,
                    w_ref_pos:  float, w_ref_neg:  float)

                   Accepts an optional target_ratio (default 1.0).  Weights are
                   computed so that Σ(Y×w)/Σ(w) for the protected group equals
                   target_ratio × p_ref, making the re-audited DI ratio = target_ratio.

run_audit_intersectional        → same 4-tuple contract, but accepts a
                                   pre-built 'combined_protected' column.

apply_reweighing_intersectional → same 4-tuple contract + target_ratio param,
                                   operates on the synthetic combined_protected column.

compute_sensitivity_curve       → DataFrame of weights across a ratio range.
compute_sensitivity_curve_intersectional → same, for intersectional combined col.
"""
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# WEIGHTED AUDIT  (standard, single protected attribute)
# ──────────────────────────────────────────────────────────────────────────────

def run_audit(
    df: pd.DataFrame,
    target_col: str = "target",
    protected_col: str = "is_young",
) -> tuple[float, float, float, bool]:
    """
    Compute disparate-impact metrics between two groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset. Must contain ``target_col`` and ``protected_col``
        as binary (0/1) columns.  If a ``sample_weight`` column is present
        it is used in the calculation.
    target_col : str
        Name of the outcome column (binary 0/1).
    protected_col : str
        Name of the protected-attribute column (binary 0/1, 1 = protected).

    Returns   ← ALWAYS exactly 4 values
    -------
    protected_rate : float
        Success rate for the protected group (protected_col == 1).
    reference_rate : float
        Success rate for the reference group (protected_col == 0).
    ratio : float
        Disparate impact ratio = protected_rate / reference_rate, rounded to 2 dp.
    is_weighted : bool
        True  → ``sample_weight`` column was found and used.
        False → simple unweighted mean was used.

    Weighted formula (applied per group):
        rate = Σ(target * sample_weight) / Σ(sample_weight)
    """
    # --- masks -----------------------------------------------------------------
    protected_mask = df[protected_col] == 1
    reference_mask = ~protected_mask

    is_weighted: bool = "sample_weight" in df.columns

    if is_weighted:
        # ── Weighted success rate ─────────────────────────────────────────────
        # Formula: Σ(outcome × weight) / Σ(weight)  — computed per group
        prot_weights  = df.loc[protected_mask, "sample_weight"]
        ref_weights   = df.loc[reference_mask, "sample_weight"]

        prot_outcomes = df.loc[protected_mask, target_col]
        ref_outcomes  = df.loc[reference_mask, target_col]

        protected_rate: float = (
            (prot_outcomes * prot_weights).sum() / prot_weights.sum()
            if prot_weights.sum() > 0 else 0.0
        )
        reference_rate: float = (
            (ref_outcomes * ref_weights).sum() / ref_weights.sum()
            if ref_weights.sum() > 0 else 0.0
        )
    else:
        # ── Unweighted (simple mean) ──────────────────────────────────────────
        protected_rate = float(df.loc[protected_mask, target_col].mean())
        reference_rate = float(df.loc[reference_mask, target_col].mean())

    ratio: float = round(protected_rate / reference_rate, 2) if reference_rate > 0 else 0.0

    # Explicit 4-tuple — no ambiguity
    return protected_rate, reference_rate, ratio, is_weighted


# ──────────────────────────────────────────────────────────────────────────────
# REWEIGHING  (4-weight, group × outcome)  — standard single-attribute
# ──────────────────────────────────────────────────────────────────────────────

def apply_reweighing(
    df: pd.DataFrame,
    target_col: str = "target",
    protected_col: str = "is_young",
    target_ratio: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Compute 4 sample weights to achieve a specific Disparate Impact Ratio.

    The reference group keeps weights of 1.0 (natural rate preserved).
    The protected group receives weights that shift its weighted success rate
    to  ``target_ratio × p_ref``, so that the re-audited DI ratio equals
    ``target_ratio`` exactly.

    Formula
    -------
    target_prot_rate = clamp(target_ratio × p_ref, ε, 1−ε)
    w_prot_pos = target_prot_rate / p_prot
    w_prot_neg = (1 − target_prot_rate) / (1 − p_prot)
    w_ref_pos  = 1.0   (reference group unchanged)
    w_ref_neg  = 1.0

    Parameters
    ----------
    target_ratio : float
        Desired DI Ratio after reweighing.  Default = 1.0 (perfect parity).
        Slider range 0.80 – 1.20 is typical; values are clamped so that the
        resulting target_prot_rate stays in (ε, 1−ε).

    Returns  ← ALWAYS exactly 4 values
    -------
    w_prot_pos, w_prot_neg, w_ref_pos, w_ref_neg
    """
    _EPS = 1e-9
    protected_mask = df[protected_col] == 1

    p_prot: float = float(df.loc[protected_mask,  target_col].mean())
    p_ref:  float = float(df.loc[~protected_mask, target_col].mean())

    # Desired protected-group weighted success rate
    target_prot_rate: float = float(np.clip(target_ratio * p_ref, _EPS, 1.0 - _EPS))

    # Weights for protected group
    w_prot_pos = target_prot_rate / p_prot             if p_prot        > _EPS else 1.0
    w_prot_neg = (1.0 - target_prot_rate) / (1.0 - p_prot) if (1.0 - p_prot) > _EPS else 1.0

    # Reference group — keep at natural rate
    w_ref_pos = 1.0
    w_ref_neg = 1.0

    return round(w_prot_pos, 4), round(w_prot_neg, 4), round(w_ref_pos, 4), round(w_ref_neg, 4)


# ──────────────────────────────────────────────────────────────────────────────
# SENSITIVITY CURVE  (standard single-attribute)
# ──────────────────────────────────────────────────────────────────────────────

def compute_sensitivity_curve(
    df: pd.DataFrame,
    target_col: str,
    protected_col: str,
    ratio_range: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """
    Return a DataFrame showing how weights change across a range of target ratios.

    Columns: target_ratio, w_prot_pos, w_prot_neg, w_ref_pos, w_ref_neg
    """
    if ratio_range is None:
        ratio_range = np.round(np.arange(0.50, 1.55, 0.05), 2)

    rows = []
    for t in ratio_range:
        wp, wn, rp, rn = apply_reweighing(
            df, target_col, protected_col, target_ratio=float(t)
        )
        rows.append(
            {"target_ratio": round(float(t), 2),
             "w_prot_pos": wp, "w_prot_neg": wn,
             "w_ref_pos": rp, "w_ref_neg": rn}
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# INTERSECTIONAL AUDIT  (two protected attributes → combined binary group)
# ──────────────────────────────────────────────────────────────────────────────

def run_audit_intersectional(
    df: pd.DataFrame,
    target_col: str,
    combined_col: str = "_combined_protected",
) -> tuple[float, float, float, bool]:
    """
    Disparate-impact audit for a pre-built intersectional group column.

    The caller must create ``combined_col`` as a binary column where
    1 = member of *both* protected groups (e.g. Female AND Young), 0 = otherwise.

    The function delegates to ``run_audit`` using that synthetic column —
    so all weighted-math guarantees are preserved.

    Returns   ← ALWAYS exactly 4 values  (same contract as run_audit)
    -------
    protected_rate, reference_rate, ratio, is_weighted
    """
    return run_audit(df, target_col=target_col, protected_col=combined_col)


# ──────────────────────────────────────────────────────────────────────────────
# INTERSECTIONAL REWEIGHING  (same 4-weight approach on combined group)
# ──────────────────────────────────────────────────────────────────────────────

def apply_reweighing_intersectional(
    df: pd.DataFrame,
    target_col: str,
    combined_col: str = "_combined_protected",
    target_ratio: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Compute 4 reweighing weights for the intersectional (combined) group.

    Delegates to ``apply_reweighing`` with the synthetic combined column.
    Accepts the same ``target_ratio`` parameter — pass the slider value.

    Returns  ← ALWAYS exactly 4 values  (same contract as apply_reweighing)
    -------
    w_prot_pos, w_prot_neg, w_ref_pos, w_ref_neg
    """
    return apply_reweighing(
        df, target_col=target_col, protected_col=combined_col,
        target_ratio=target_ratio,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SENSITIVITY CURVE  (intersectional combined group)
# ──────────────────────────────────────────────────────────────────────────────

def compute_sensitivity_curve_intersectional(
    df: pd.DataFrame,
    target_col: str,
    combined_col: str = "_combined_protected",
    ratio_range: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """
    Sensitivity curve for the intersectional combined group.
    Delegates to ``compute_sensitivity_curve``.
    """
    return compute_sensitivity_curve(
        df, target_col=target_col, protected_col=combined_col,
        ratio_range=ratio_range,
    )