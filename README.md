# ⚖️ FairScan — Algorithmic Fairness Auditor

> **Detect, quantify, and mitigate bias in any tabular dataset — in minutes, not months.**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![UN SDG 10](https://img.shields.io/badge/UN%20SDG-10%20Reduced%20Inequalities-DD1367)](https://sdgs.un.org/goals/goal10)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌍 SDG Alignment

FairScan directly addresses **UN Sustainable Development Goal 10: Reduced Inequalities**.  
Algorithmic bias in hiring, lending, healthcare, and education silently reinforces systemic
disadvantage. FairScan gives any practitioner — regardless of ML expertise — a transparent,
evidence-based tool to detect and correct those inequalities before a model goes into production.

---

## 🎯 What Is FairScan?

FairScan is an interactive, no-code bias auditing tool built with Python 3.13 and Streamlit.
Upload any CSV dataset, point it at an outcome column and a demographic attribute, and FairScan
instantly computes the **Disparate Impact Ratio** — the gold-standard legal metric for algorithmic
fairness — and generates mathematically rigorous sample weights to mitigate bias without discarding
a single row of data.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **🔁 Universal Data Loading** | Accepts CSV, `.data`, and `.txt` files. Auto-detects binary columns, smart-encodes string values (`Male`/`Female` → 1/0), and strips whitespace — the Adult Income, German Credit, and any custom dataset work out of the box. |
| **🤖 Auto-Detection** | Heuristically suggests the most likely Outcome and Protected Attribute columns the moment a file is loaded, with manual override. |
| **🔀 Intersectional Auditing** | Goes beyond single-attribute analysis. Enable a second protected attribute (e.g. `Gender` **AND** `Age`) to surface bias that hides at the intersection of multiple identities — a capability absent from most commercial fairness tools. |
| **⚖️ Weighted Parity Audit** | Supports datasets with pre-existing `sample_weight` columns, computing fairness metrics via the weighted-mean formula rather than a simple average. |
| **🎯 What-If Sensitivity Slider** | Drag a slider from **0.80 → 1.20** to choose any target Disparate Impact Ratio. Reweighing weights update instantly, and a live **Sensitivity Graph** shows exactly how each weight responds to every possible fairness target. |
| **📥 Fair-Ready Export** | Download the mitigated dataset as a CSV with `sample_weight` column attached — ready to feed directly into scikit-learn, XGBoost, or any ML pipeline. |
| **📚 Responsible AI Glossary** | A persistent sidebar explains Disparate Impact, Reweighing, and Intersectionality in plain language, with academic citations. |

---

## 🔬 Methodology

### Disparate Impact Ratio

The primary fairness metric, codified by the U.S. Equal Employment Opportunity Commission
as the **Four-Fifths (80%) Rule**:

$$\text{DI Ratio} = \frac{\text{Success Rate}_{\,\text{protected}}}{\text{Success Rate}_{\,\text{reference}}}$$

A ratio below **0.80** indicates potential adverse impact against the protected group.
A ratio above **1.25** indicates potential reverse bias.

### Weighted Audit (when `sample_weight` is present)

$$\text{Success Rate} = \frac{\displaystyle\sum_{i} \left( y_i \times w_i \right)}{\displaystyle\sum_{i} w_i}$$

where $y_i \in \{0, 1\}$ is the binary outcome and $w_i$ is the sample weight for row $i$.

### Target-Aware Reweighing

For a user-specified target ratio $T$, FairScan computes:

$$w^{+}_{\text{prot}} = \frac{T \cdot p_{\text{ref}}}{p_{\text{prot}}}, \qquad w^{-}_{\text{prot}} = \frac{1 - T \cdot p_{\text{ref}}}{1 - p_{\text{prot}}}$$

where $p_{\text{prot}}$ and $p_{\text{ref}}$ are the unweighted success rates of the protected
and reference groups respectively. This guarantees that re-auditing the exported CSV
returns a DI Ratio of exactly $T$.

---

## 🗂️ Project Structure

```
FairScan_Project/
├── app.py            # Streamlit UI — all pages, layout, and interactive widgets
├── logic.py          # Core audit & mitigation functions (pure Python, no UI deps)
├── requirements.txt  # Pinned dependencies
├── data/             # Sample datasets for local testing
└── README.md
```

**Architecture principle:** `logic.py` contains zero Streamlit calls. Every public function
returns plain Python primitives or DataFrames, making the core logic independently testable
and reusable in non-Streamlit contexts (e.g., CI fairness gates, Jupyter notebooks).

---

## 🚀 Getting Started

### Prerequisites

- Python **3.13** or later
- `pip` (comes with Python)

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/FairScan.git
cd FairScan
```

### 2 — Create and activate a virtual environment *(recommended)*

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run FairScan

```bash
streamlit run app.py
```

Streamlit will open **http://localhost:8501** in your default browser automatically.

---

## 🧪 Quick Demo

1. Upload any CSV with a binary outcome column (e.g., `income`, `target`, `approved`).
2. FairScan auto-selects the most likely **Outcome** and **Protected Attribute** columns.
3. Click **🚀 Run FairScan Audit** — the Disparate Impact Ratio and bar chart appear instantly.
4. Scroll to **Step 4**, drag the **🎯 Fairness Target slider** to your desired ratio, and download the fair-ready CSV.

---

## 📦 Dependencies

| Package | Role |
|---|---|
| `streamlit` | Interactive web UI |
| `pandas` | Data loading, encoding, and manipulation |
| `matplotlib` | Audit bar chart and sensitivity graph |
| `scikit-learn` | (Available for downstream model integration) |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.  
Please open an issue to discuss significant changes before submitting a pull request.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 📖 References

- EEOC (1978). *Uniform Guidelines on Employee Selection Procedures* — Four-Fifths Rule.  
- Feldman, M. et al. (2015). *Certifying and Removing Disparate Impact*. KDD 2015.  
- Kamiran, F. & Calders, T. (2012). *Data preprocessing techniques for classification without discrimination*. KLDS.  
- Crenshaw, K. (1989). *Demarginalizing the Intersection of Race and Sex*. University of Chicago Legal Forum.  
- Agarwal, A. et al. (2018). *A Reductions Approach to Fairness*. ICML 2018.

---

<p align="center">
  Built with ❤️ for the <strong>Google Solution Challenge</strong> · Addressing <strong>UN SDG 10: Reduced Inequalities</strong>
</p>
