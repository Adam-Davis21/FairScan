"""
app.py — FairScan Smart Bias Scanner (Streamlit UI)

Requires Python 3.13 + the packages in requirements.txt.
Run with:  streamlit run app.py

New in this version
-------------------
• Smart Encoding   : any column with exactly 2 unique values (strings or ints)
                     is accepted; non-numeric values are encoded 1/0 on-the-fly.
• Column Cleaning  : column names are stripped of leading/trailing whitespace on load.
• Headerless .data : files that aren't the 21-column German Credit get generic
                     column names (C1, C2, …) instead of crashing.
• All existing German Credit logic and 4-weight parity math are unchanged.
"""
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from logic import (
    apply_reweighing,
    apply_reweighing_intersectional,
    compute_sensitivity_curve,
    compute_sensitivity_curve_intersectional,
    run_audit,
    run_audit_intersectional,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FairScan – Smart Bias Scanner",
    page_icon="⚖️",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Responsible AI Glossary & Explainability
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center;margin-bottom:4px;'>📚 Understanding the Audit</h2>"
        "<p style='text-align:center;color:#888;font-size:0.85rem;margin-top:0;'>"
        "Responsible AI · Glossary"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    with st.expander("⚖️ Disparate Impact Ratio", expanded=False):
        st.markdown(
            "The **Disparate Impact Ratio** measures how fairly a positive outcome "
            "is distributed between a protected group and a reference group — calculated "
            "as *protected success rate ÷ reference success rate*.\n\n"
            "The **80% Rule** (Four-Fifths Rule), introduced by the U.S. Equal Employment "
            "Opportunity Commission, flags a potential legal violation when this ratio "
            "falls below **0.80**, meaning the protected group succeeds at less than "
            "80 % of the rate of the reference group."
        )

    with st.expander("🔄 Reweighing Mitigation", expanded=False):
        st.markdown(
            "**Reweighing** does *not* delete or alter any rows — every record "
            "in your dataset is preserved exactly as-is.\n\n"
            "Instead, it assigns a **sample weight** to each row so that "
            "underrepresented success stories (e.g., a qualified applicant from a "
            "marginalised group who was approved) carry *more mathematical voice* "
            "in downstream model training, balancing the numbers without "
            "discarding real data."
        )

    with st.expander("🔀 Intersectional Bias", expanded=False):
        st.markdown(
            "**Intersectional bias** arises when discrimination hides at the "
            "*overlap* of multiple identities — for example, a model may treat "
            "women and young people fairly when each group is audited separately, "
            "yet still disadvantage individuals who are *both* young *and* female.\n\n"
            "FairScan's intersectional mode creates a combined protected-group column "
            "(identity A **AND** identity B = 1) so this hidden bias becomes measurable "
            "with the same Disparate Impact Ratio framework."
        )

    st.markdown("---")
    st.caption(
        "📖 **References** · EEOC Four-Fifths Rule · "
        "AIF360 Reweighing Algorithm · Crenshaw (1989) Intersectionality"
    )

st.title("⚖️ FairScan: Smart Bias Scanner")
st.markdown(
    "<p style='color:#888;margin-top:-12px;'>"
    "Upload a dataset and let FairScan automatically detect bias targets and protected groups."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_binary_columns(df: pd.DataFrame) -> list[str]:
    """
    Return every column that has exactly 2 unique non-null values.

    Accepts both numeric (0/1) and string columns (e.g. 'Male'/'Female',
    '>50K'/'<=50K').  This is the 'Smart Encoding' gate.
    """
    binary_cols: list[str] = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            binary_cols.append(col)
    return binary_cols


def encode_column(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Return a 0/1-encoded Series for *col*.

    • If already numeric with values in {0, 1} → return as-is (int).
    • Otherwise → map the two unique values: the one that sorts/compares
      'higher' (e.g. '>50K', 'Male', 1) becomes 1; the other becomes 0.
      The mapping is shown in the UI so the user can verify it.

    Raises ValueError if the column doesn't have exactly 2 unique values.
    """
    unique_vals = sorted(df[col].dropna().unique(), key=str)
    if len(unique_vals) != 2:
        raise ValueError(f"Column '{col}' has {len(unique_vals)} unique values (expected 2).")

    # Already 0/1 integers — nothing to do
    if set(unique_vals) <= {0, 1}:
        return df[col].astype(int)

    # Map: second value (higher sort) → 1, first → 0
    val0, val1 = unique_vals[0], unique_vals[1]
    mapping = {val0: 0, val1: 1}
    return df[col].map(mapping).astype(int)


def show_encoding_info(col: str, original: pd.Series, encoded: pd.Series) -> None:
    """Display a compact badge explaining what the encoding did."""
    unique_orig = sorted(original.dropna().unique(), key=str)
    if set(unique_orig) <= {0, 1}:
        return  # already binary — no message needed
    val0, val1 = unique_orig[0], unique_orig[1]
    st.caption(
        f"🔢 **Auto-encoded `{col}`** → `{val0}` = 0 · `{val1}` = 1  "
        f"*(protected / positive class = 1)*"
    )


def auto_detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Heuristically suggest outcome and protected-attribute columns.

    Priority rules
    --------------
    Outcome   : column named 'target' or 'income'  →  first binary column found
    Protected : first column starting with 'is_' or named 'sex'/'gender'/'race'
                →  second binary column found
    """
    binary_cols: list[str] = detect_binary_columns(df)
    if not binary_cols:
        return None, None

    # ── Outcome ───────────────────────────────────────────────────────────────
    preferred_outcomes = {"target", "income", "class", "label", "y"}
    outcome: str | None = next(
        (c for c in binary_cols if c.lower() in preferred_outcomes), None
    )
    if outcome is None:
        outcome = binary_cols[0]

    # ── Protected attribute ───────────────────────────────────────────────────
    preferred_protected = {"sex", "gender", "race", "age", "is_young"}
    is_cols = [c for c in binary_cols if c.startswith("is_")]
    pref_cols = [c for c in binary_cols if c.lower() in preferred_protected and c != outcome]

    if is_cols:
        protected: str | None = is_cols[0]
    elif pref_cols:
        protected = pref_cols[0]
    else:
        candidates = [c for c in binary_cols if c != outcome]
        protected = candidates[0] if candidates else None

    return outcome, protected


def load_dataframe(uploaded_file) -> pd.DataFrame | None:
    """
    Parse an uploaded file into a DataFrame.

    Supports
    --------
    • .data / .txt  — space-separated, with or without header
    • .csv          — any comma-separated file (including Adult dataset)

    New behaviours
    ---------------
    • Column names are stripped of leading/trailing whitespace.
    • Headerless .data files get generic names C1, C2, … instead of crashing.
    • The Adult CSV works automatically via the CSV branch.

    Returns None (and shows an error) if loading fails.
    """
    try:
        name: str = uploaded_file.name

        if name.endswith(".data") or name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
            if df.shape[1] == 21:          # German Credit: 20 features + target
                df.columns = [f"A{i}" for i in range(1, 20)] + ["A20", "target"]
                df["target"]   = df["target"].map({1: 1, 2: 0})
                df["is_young"] = (df["A13"] < 25).astype(int)
                st.info("📂 Raw `.data` file — auto-mapped as German Credit dataset.")
            else:
                # Generic headerless .data — use C1, C2, …
                df.columns = [f"C{i+1}" for i in range(df.shape[1])]
                st.warning(
                    f"⚠️ Headerless `.data` file ({df.shape[1]} columns) — "
                    "loaded with generic column names (C1, C2, …).  "
                    "Select 'Target' and 'Protected' columns manually below."
                )
        else:
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV loaded successfully.")

        if df.empty:
            st.error("❌ The uploaded file is empty. Please try a different file.")
            return None

        # ── Column-name cleaning (fixes Adult dataset's leading spaces) ────────
        df.columns = df.columns.str.strip()

        # ── String value cleaning (strip leading/trailing spaces in cells) ─────
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].str.strip()

        return df

    except pd.errors.EmptyDataError:
        st.error("❌ The file appears to be empty or has no parsable content.")
        return None
    except pd.errors.ParserError as exc:
        st.error(f"❌ Could not parse the file: {exc}")
        return None
    except Exception as exc:
        st.error(f"❌ Unexpected error while loading file: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 1: Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, .data, or .txt)",
    type=["csv", "data", "txt"],
)

if uploaded_file is None:
    st.info("👆 Please upload a file to begin.  Supported formats: CSV · .data · .txt")
    st.stop()

df: pd.DataFrame | None = load_dataframe(uploaded_file)
if df is None:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREVIEW — always shown above selectors so users can map generic names
# (e.g. C10, C15) to real fields before choosing Target / Protected columns.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
st.dataframe(df.head(10), width="stretch")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — COLUMN SELECTION  (auto-detect + manual override, ALL columns shown)
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 2: Configure Audit Columns")

binary_cols: list[str]  = detect_binary_columns(df)
auto_outcome, auto_prot = auto_detect_columns(df)

# ── Informational banners (no hard stop — user may pick any column) ───────────
if binary_cols:
    if auto_outcome and auto_prot:
        st.success(
            f"🤖 **Auto-detected** → Outcome: `{auto_outcome}` | "
            f"Protected Attribute: `{auto_prot}`  *(override below if needed)*"
        )
    else:
        st.info(
            f"🔎 Found {len(binary_cols)} two-valued column(s): "
            + ", ".join(f"`{c}`" for c in binary_cols[:6])
            + ("  …" if len(binary_cols) > 6 else "")
            + "  — select your Target and Protected columns below."
        )
else:
    st.warning(
        "⚠️ No column with exactly 2 unique values was auto-detected.  "
        "You can still select any column manually — FairScan will attempt "
        "to encode it for the audit."
    )

# ── All columns available in dropdowns ───────────────────────────────────────
all_cols: list[str] = df.columns.tolist()

col_a, col_b = st.columns(2)

with col_a:
    # Default to auto-detected outcome; fall back to first column
    outcome_default_idx: int = (
        all_cols.index(auto_outcome) if auto_outcome in all_cols else 0
    )
    target_col: str = st.selectbox(
        "🎯 Select Outcome (Target) Column",
        options=all_cols,
        index=outcome_default_idx,
        help="Any column whose values represent the model's outcome.  "
             "Works best with exactly 2 unique values.",
    )

with col_b:
    # Default to auto-detected protected attribute; fall back to second column
    prot_default_idx: int = (
        all_cols.index(auto_prot)
        if auto_prot and auto_prot in all_cols
        else min(1, len(all_cols) - 1)
    )
    protected_col: str = st.selectbox(
        "🛡️ Select Protected Group Column",
        options=all_cols,
        index=prot_default_idx,
        help="Any column identifying a demographic group.  "
             "Works best with exactly 2 unique values (0/1 or strings like 'Male').",
    )

if target_col == protected_col:
    st.error("❌ Target and protected-group columns cannot be the same. Please choose different columns.")
    st.stop()

# ── Intersectional Audit (optional second attribute) ──────────────────────────
use_intersectional: bool = st.checkbox(
    "🔀 Enable Intersectional Audit  *(optional — select a second protected attribute)*",
    value=False,
    help="When enabled, the 'Protected Group' becomes individuals who belong to "
         "the protected class of BOTH attributes simultaneously (e.g. Female AND Young).",
)

secondary_col: str | None = None
if use_intersectional:
    remaining_cols = [c for c in all_cols if c != target_col and c != protected_col]
    if not remaining_cols:
        st.warning("⚠️ No additional columns available for a second protected attribute.")
        use_intersectional = False
    else:
        secondary_col = st.selectbox(
            "🛡️🛡️ Select Secondary Protected Attribute",
            options=remaining_cols,
            index=0,
            help="The second demographic column.  Combined with the primary attribute "
                 "above to form the intersectional protected group.",
        )
        st.info(
            f"🔀 **Intersectional mode active** — Protected group = "
            f"`{protected_col}` = 1  **AND**  `{secondary_col}` = 1"
        )

# ── Adaptive on-the-fly encoding ─────────────────────────────────────────────
# Build an encoded copy of df for audit/mitigation; keep original df intact.
# If a column has >2 unique values we warn the user but still encode the
# top-2 most-frequent values (→1 and →0) so the audit can proceed.

def _safe_encode(col: str) -> pd.Series:
    """Encode *col* adaptively; warns on >2 unique values."""
    unique_vals = df[col].dropna().unique()
    n_unique = len(unique_vals)

    if n_unique == 2:
        return encode_column(df, col)

    if n_unique > 2:
        # Keep only the two most-frequent values; map them 1 / 0
        top2 = df[col].value_counts().index[:2].tolist()
        val1, val0 = top2[0], top2[1]   # most frequent → 1, second → 0
        st.warning(
            f"⚠️ Column **`{col}`** has {n_unique} unique values — "
            f"FairScan works best on binary groups.  "
            f"Auto-encoding the two most frequent: "
            f"`{val1}` → 1 (protected/positive),  `{val0}` → 0 (reference/negative).  "
            "Select a different column above if this is incorrect."
        )
        mapping = {val1: 1, val0: 0}
        return df[col].map(mapping).fillna(0).astype(int)

    # 0 or 1 unique values — cannot audit
    st.error(f"❌ Column `{col}` has {n_unique} non-null unique value(s) — cannot audit.")
    st.stop()

try:
    df_encoded = df.copy()
    df_encoded[target_col]    = _safe_encode(target_col)
    df_encoded[protected_col] = _safe_encode(protected_col)

    show_encoding_info(target_col,    df[target_col],    df_encoded[target_col])
    show_encoding_info(protected_col, df[protected_col], df_encoded[protected_col])

    if use_intersectional and secondary_col:
        df_encoded[secondary_col] = _safe_encode(secondary_col)
        show_encoding_info(secondary_col, df[secondary_col], df_encoded[secondary_col])
        # Combined group: 1 only if member of BOTH protected groups
        df_encoded["_combined_protected"] = (
            (df_encoded[protected_col] == 1) & (df_encoded[secondary_col] == 1)
        ).astype(int)

except Exception as enc_err:
    st.error(f"❌ Encoding error: {enc_err}")
    st.stop()

# ── Column Alias (display rename) ─────────────────────────────────────────────
with st.expander("✏️ Rename columns for chart labels  *(optional)*", expanded=False):
    st.caption(
        "These labels are display-only — they do not modify your data. "
        "Leave blank to use the original column name."
    )
    _alias_cols = 3 if (use_intersectional and secondary_col) else 2
    _alias_widgets = st.columns(_alias_cols)
    with _alias_widgets[0]:
        target_alias: str = st.text_input(
            f"Label for Outcome  `{target_col}`",
            value=st.session_state.get("alias_target", ""),
            placeholder=target_col,
            key="alias_target_input",
        )
    with _alias_widgets[1]:
        protected_alias: str = st.text_input(
            f"Label for Primary Protected  `{protected_col}`",
            value=st.session_state.get("alias_protected", ""),
            placeholder=protected_col,
            key="alias_protected_input",
        )
    secondary_alias: str = ""
    if use_intersectional and secondary_col and _alias_cols == 3:
        with _alias_widgets[2]:
            secondary_alias = st.text_input(
                f"Label for Secondary Protected  `{secondary_col}`",
                value=st.session_state.get("alias_secondary", ""),
                placeholder=secondary_col,
                key="alias_secondary_input",
            )

# Resolve display names
target_label:    str = target_alias.strip()    or target_col
prot_label_a:    str = protected_alias.strip() or protected_col
prot_label_b:    str = secondary_alias.strip() or (secondary_col or "")

# Combined intersectional label used in charts
if use_intersectional and secondary_col:
    protected_label: str = f"{prot_label_a} & {prot_label_b}"
else:
    protected_label = prot_label_a

# Persist aliases
st.session_state["alias_target"]    = target_alias
st.session_state["alias_protected"] = protected_alias
st.session_state["alias_secondary"] = secondary_alias

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FAIRNESS AUDIT
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 3: Fairness Audit")

st.info(
    "**📖 What is Disparate Impact?**\n\n"
    "Disparate Impact measures whether a protected group (e.g. women, a racial minority) "
    "receives a favourable outcome at a rate that is substantially lower than the reference group.  "
    "The widely-used **80% rule** (also called the Four-Fifths Rule) flags a potential violation "
    "when the ratio falls below **0.80** — meaning the protected group's success rate is less than "
    "80 % of the reference group's.  "
    "A ratio of **1.0** means both groups receive the positive outcome at exactly the same rate "
    "(perfect parity)."
)

if st.button("🚀 Run FairScan Audit", use_container_width=True):
    try:
        # ── CALL: always unpacks exactly 4 values ─────────────────────────────
        if use_intersectional and secondary_col and "_combined_protected" in df_encoded.columns:
            prot_rate, ref_rate, ratio, is_weighted = run_audit_intersectional(
                df_encoded, target_col, combined_col="_combined_protected"
            )
        else:
            prot_rate, ref_rate, ratio, is_weighted = run_audit(
                df_encoded, target_col, protected_col
            )

        # ── Audit-mode badge ──────────────────────────────────────────────────
        if is_weighted:
            st.success(
                "⚖️ **Mode: Weighted Parity Audit** — `sample_weight` column detected.  "
                "Rates = Σ(outcome × weight) / Σ(weight) per group."
            )
        else:
            st.info(
                "📊 **Mode: Standard Audit** — No sample weights found; "
                "using simple mean success rates."
            )

        # ── Three-column metric cards ─────────────────────────────────────────
        m1, m2, m3 = st.columns(3)

        m1.metric(
            label=f"Protected Group Rate  ({protected_label} = 1)",
            value=f"{prot_rate * 100:.1f}%",
        )
        m2.metric(
            label=f"Reference Group Rate  ({protected_label} = 0)",
            value=f"{ref_rate * 100:.1f}%",
        )

        delta_from_threshold: float = round(ratio - 0.8, 3)
        parity_label: str = (
            "✅ Meets 80% rule"
            if ratio >= 0.8
            else f"⚠️ {abs(delta_from_threshold):.3f} below threshold"
        )
        m3.metric(
            label="Disparate Impact Ratio",
            value=ratio,
            delta=parity_label,
            delta_color="normal" if ratio >= 0.8 else "inverse",
        )

        # ── Bar chart  (uses human-readable alias labels) ─────────────────────
        mode_label: str = "Weighted" if is_weighted else "Standard"
        fig, ax = plt.subplots(figsize=(8, 4))

        bars = ax.bar(
            [f"{protected_label}\n(protected group)", f"{protected_label}\n(reference group)"],
            [prot_rate, ref_rate],
            color=["#FF4B4B", "#1C83E1"],
            width=0.5,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.axhline(
            0.8 * ref_rate,
            color="#FFA500",
            linestyle="--",
            linewidth=1.5,
            label=f"80% Rule Threshold ({0.8 * ref_rate * 100:.1f}%)",
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height() * 100:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{target_label} Success Rate")
        ax.set_title(
            f"'{target_label}' Outcome Rate by '{protected_label}' Group  [{mode_label} Audit]",
            fontweight="bold",
        )
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

        # ── Dynamic Contextual Hint based on ratio ────────────────────────────
        if ratio < 0.8:
            st.error(
                "🚨 **Action Required:** Significant bias detected against the protected group. "
                f"The Disparate Impact Ratio is **{ratio}** — well below the 0.80 fairness threshold. "
                "Consider applying **Reweighing Mitigation** (Step 4 below) to correct this imbalance."
            )
        elif ratio > 1.25:
            st.warning(
                "⚠️ **High Variance:** The model significantly *favours* the protected group. "
                f"A ratio of **{ratio}** (above 1.25) can indicate reverse bias or data imbalance. "
                "Investigate whether the protected group is overrepresented in positive outcomes."
            )
        else:
            st.success(
                "✅ **Fair Representation:** The dataset meets standard fairness thresholds. "
                f"A Disparate Impact Ratio of **{ratio}** falls within the accepted 0.80 – 1.25 range. "
                "Continue monitoring as new data is collected."
            )

        # ── Technical Card: Formula ───────────────────────────────────────────
        with st.expander("🧮 Technical Card — Audit Formula", expanded=False):
            st.markdown("**Weighted Audit Rate (per group)**")
            st.latex(
                r"\text{Success Rate} = "
                r"\frac{\sum (\text{Outcome} \times \text{Weight})}{\sum \text{Weight}}"
            )
            st.markdown(
                "where **Outcome** ∈ {0, 1} is the target column value and "
                "**Weight** is the `sample_weight` per row (defaults to 1.0 for "
                "unweighted audits).\n\n"
                "The **Disparate Impact Ratio** is then:"
            )
            st.latex(
                r"\text{DI Ratio} = "
                r"\frac{\text{Success Rate}_{\,\text{protected}}}"
                r"{\text{Success Rate}_{\,\text{reference}}}"
            )
            st.caption(
                "⚖️ A ratio < 0.80 indicates potential adverse impact under the "
                "EEOC Four-Fifths Rule.  A ratio > 1.25 may indicate reverse bias."
            )

        # ── Persist audit state across reruns ────────────────────────────────
        st.session_state["audit_done"]              = True
        st.session_state["audit_target"]            = target_col
        st.session_state["audit_protected"]         = protected_col
        st.session_state["audit_secondary"]         = secondary_col
        st.session_state["audit_intersectional"]    = use_intersectional
        st.session_state["audit_target_label"]      = target_label
        st.session_state["audit_protected_label"]   = protected_label

    except ValueError as exc:
        st.error(
            f"❌ **ValueError during audit** — most likely a return-value mismatch.\n\n"
            f"`{exc}`\n\n"
            "```\n" + traceback.format_exc() + "\n```"
        )
    except KeyError as exc:
        st.error(
            f"❌ **Column not found**: {exc}  "
            "Please verify your column selections above."
        )
    except ZeroDivisionError:
        st.error(
            "❌ A group has zero members or zero successes — "
            "cannot compute a valid ratio."
        )
    except Exception as exc:
        st.error(
            f"❌ **Unexpected audit failure**:\n\n"
            "```\n" + traceback.format_exc() + "\n```"
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MITIGATION & DOWNLOAD
# Shown only after a successful audit AND when no weights exist yet
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("audit_done") and "sample_weight" not in df_encoded.columns:
    st.markdown("---")
    st.header("Step 4: Apply Bias Mitigation & Export")

    st.markdown(
        "The **reweighing** technique assigns compensatory sample weights so that a downstream "
        "model trained on this data will see a fairer representation of each group."
    )

    _target:         str       = st.session_state["audit_target"]
    _protected:      str       = st.session_state["audit_protected"]
    _secondary:      str | None= st.session_state.get("audit_secondary")
    _intersectional: bool      = st.session_state.get("audit_intersectional", False)
    _prot_label:     str       = st.session_state.get("audit_protected_label", _protected)

    # Determine which column drives the reweighing
    _reweigh_col: str = (
        "_combined_protected"
        if _intersectional and _secondary and "_combined_protected" in df_encoded.columns
        else _protected
    )

    # ── Fairness Target Slider ───────────────────────────────────────────────
    target_ratio: float = st.slider(
        "🎯 Set Fairness Target (Disparate Impact Ratio)",
        min_value=0.80,
        max_value=1.20,
        value=1.00,
        step=0.05,
        help=(
            "Drag to choose the DI Ratio you want the mitigated dataset to achieve. "
            "1.0 = perfect parity.  0.8 = minimum legal threshold (EEOC Four-Fifths Rule). "
            "Weights update instantly as you move the slider."
        ),
    )

    try:
        # Unpack all 4 (group × outcome) weights
        if _reweigh_col == "_combined_protected":
            w_prot_pos, w_prot_neg, w_ref_pos, w_ref_neg = apply_reweighing_intersectional(
                df_encoded, _target, combined_col="_combined_protected",
                target_ratio=target_ratio,
            )
            st.caption(
                f"🔀 Intersectional reweighing: protected group = "
                f"`{_protected}` = 1  **AND**  `{_secondary}` = 1"
            )
        else:
            w_prot_pos, w_prot_neg, w_ref_pos, w_ref_neg = apply_reweighing(
                df_encoded, _target, _protected,
                target_ratio=target_ratio,
            )

        # Display weights in a 2×2 grid
        st.markdown(f"**Computed sample weights per cell** *(targeting DI Ratio = {target_ratio})*:")
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric(f"`{_prot_label}`=1, `{_target}`=1", w_prot_pos,
                   help="Protected group, positive outcome")
        wc2.metric(f"`{_prot_label}`=1, `{_target}`=0", w_prot_neg,
                   help="Protected group, negative outcome")
        wc3.metric(f"`{_prot_label}`=0, `{_target}`=1", w_ref_pos,
                   help="Reference group, positive outcome (always 1.0)")
        wc4.metric(f"`{_prot_label}`=0, `{_target}`=0", w_ref_neg,
                   help="Reference group, negative outcome (always 1.0)")

        # ── Sensitivity Graph ───────────────────────────────────────────────
        with st.expander("📈 Sensitivity Graph — Weights vs. Fairness Target", expanded=True):
            ratio_range = np.round(np.arange(0.80, 1.25, 0.05), 2)

            if _reweigh_col == "_combined_protected":
                sens_df = compute_sensitivity_curve_intersectional(
                    df_encoded, _target,
                    combined_col="_combined_protected",
                    ratio_range=ratio_range,
                )
            else:
                sens_df = compute_sensitivity_curve(
                    df_encoded, _target, _reweigh_col,
                    ratio_range=ratio_range,
                )

            fig_s, ax_s = plt.subplots(figsize=(8, 3))
            ax_s.plot(
                sens_df["target_ratio"], sens_df["w_prot_pos"],
                label="w⁺ (protected, positive outcome)",
                color="#FF4B4B", linewidth=2, marker="o", markersize=5,
            )
            ax_s.plot(
                sens_df["target_ratio"], sens_df["w_prot_neg"],
                label="w⁻ (protected, negative outcome)",
                color="#FFA500", linewidth=2, marker="s", markersize=5,
            )
            ax_s.axhline(1.0, color="#888", linestyle="--", linewidth=1,
                         label="Reference weight baseline (1.0)")
            ax_s.axvline(
                target_ratio, color="#1C83E1", linestyle=":", linewidth=2,
                label=f"Current target ({target_ratio})",
            )
            ax_s.set_xlabel("Fairness Target (DI Ratio)")
            ax_s.set_ylabel("Sample Weight")
            ax_s.set_title(
                "How Reweighing Weights Change with the Fairness Target",
                fontweight="bold",
            )
            ax_s.legend(fontsize=8)
            fig_s.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)
            st.caption(
                "💡 **w⁺ rises** as the target increases (protected group success "
                "is amplified). **w⁻ falls** correspondingly.  "
                "Reference weights remain fixed at 1.0 (grey dashed line)."
            )

        # Assign weight per row based on group membership AND outcome
        def _assign_weight(row) -> float:
            if row[_reweigh_col] == 1:
                return w_prot_pos if row[_target] == 1 else w_prot_neg
            else:
                return w_ref_pos  if row[_target] == 1 else w_ref_neg

        df_export = df_encoded.copy()
        df_export["sample_weight"] = df_export.apply(_assign_weight, axis=1)
        st.success(
            f"✅ 4-cell reweighing complete — targeting **DI Ratio = {target_ratio}**. "
            "Re-audit the downloaded CSV to confirm the ratio matches your target."
        )

        csv_bytes: bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Fair-Ready CSV",
            data=csv_bytes,
            file_name="fairscan_mitigated.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            f"❌ Mitigation failed:\n\n"
            "```\n" + traceback.format_exc() + "\n```"
        )


elif "sample_weight" in df_encoded.columns:
    st.markdown("---")
    st.info(
        "ℹ️ This dataset already contains a `sample_weight` column — "
        "mitigation step skipped.  Re-upload the original file to generate new weights."
    )