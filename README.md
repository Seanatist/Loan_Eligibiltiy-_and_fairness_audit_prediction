# Loan_Eligibiltiy-_and_fairness_audit_prediction
# 🏦 Loan Approval Fairness Audit Dashboard

> An end-to-end ML pipeline that trains a loan approval prediction model and audits it for bias across sensitive demographic attributes — built with Streamlit, Scikit-learn, and Fairlearn.

---

## 📌 Overview

AI models used in financial decision-making can encode and amplify societal biases — sometimes without anyone noticing. This project addresses that risk head-on by building a **transparent, interactive fairness auditing tool** for loan approval models.

The dashboard walks through three stages:
1. **Data Cleaning & Analysis** — understand what's in the data before modeling
2. **Model Training & Evaluation** — train a classifier and measure overall performance
3. **Fairness Audit** — audit the model across sensitive attributes like gender and race using industry-standard fairness metrics

---

## 🖥️ Demo

| Stage | What You See |
|---|---|
| Data Cleaning | Raw vs Cleaned data, Summary Stats, Group Distributions |
| Model Evaluation | Accuracy, ROC-AUC score |
| Fairness Audit | Per-group Precision & Recall, Demographic Parity, Equalized Odds |

---

## 🚀 Features

- 🔄 **Synthetic Data Generation** — Generates a realistic 2,000-record loan dataset with features: `income`, `age`, `loan_amount`, `credit_score`, `gender`, `race`, `approved`
- "Development was conducted on a real-world loan dataset (614 rows, 12 features). The deployed demo uses a synthetic dataset to protect privacy while deliberately demonstrating fairness disparities."
- 🧹 **Data Cleaning Pipeline** — Inspect raw vs cleaned data side by side
- 📊 **Exploratory Analysis** — Summary statistics, class balance, and group distribution tabs
- 🤖 **Model Training** — Trains a classification model to predict loan approval outcomes
- ⚖️ **Fairness Audit** — Select any sensitive attribute and instantly see:
  - Group-wise Precision & Recall
  - Demographic Parity Difference
  - Equalized Odds Difference
- 🔀 **Switchable Sensitive Attributes** — Audit by `gender` or `race` with a single dropdown

---

## 📂 Project Structure

```
loan-fairness-audit/
│
├── app.py                  # Main Streamlit application
├── data_generator.py       # Synthetic loan data generation
├── model.py                # Model training & evaluation logic
├── fairness.py             # Fairness metrics using Fairlearn
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/loan-fairness-audit.git
cd loan-fairness-audit

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `scikit-learn` | Model training, evaluation metrics |
| `fairlearn` | Fairness metrics (core dependency) |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |

### Why Fairlearn?

Fairlearn is the **critical dependency** for this project's core value proposition. It provides:

- `demographic_parity_difference()` — measures whether approval rates are equal across groups
- `equalized_odds_difference()` — measures whether error rates are equal across groups
- `MetricFrame` — automatically slices any sklearn metric by sensitive attribute

Without Fairlearn, the entire Fairness Audit section (Section 3) would not exist. Make sure the version is pinned in `requirements.txt` to avoid silent compatibility issues with scikit-learn.

---

## 📈 Fairness Metrics Explained

### Demographic Parity Difference
Measures the gap in approval rates between demographic groups.
- **Ideal value:** 0.0
- **Acceptable threshold:** < 0.10
- **This project (race):** 0.189 🔴 — significant disparity detected

### Equalized Odds Difference
Measures whether false positive and false negative rates are equal across groups.
- **Ideal value:** 0.0
- **Acceptable threshold:** < 0.10
- **This project (race):** 0.198 🔴 — model errors are unevenly distributed

### Key Finding
Auditing by **race** revealed nearly **3× more bias** than auditing by gender (19–20% vs 6%), highlighting race as the dominant source of unfairness in the baseline model.

---

## ⚠️ Constraints & Limitations

- **Synthetic data only** — the dataset is procedurally generated, not sourced from real loan records. Results are illustrative, not production-ready.
- **Baseline model performance** — the current model achieves ~55.5% accuracy and ROC-AUC ~0.497, intentionally kept simple to highlight the auditing pipeline rather than model optimization.
- **Binary sensitive attributes** — the current implementation supports binary group splits (Male/Female, GroupA/GroupB). Multi-group fairness is not yet implemented.
- **Fairlearn maturity** — Fairlearn is still evolving; regression fairness and automated debiasing pipelines have limited support.
- **No debiasing yet** — the tool audits and surfaces bias but does not yet apply mitigation strategies (e.g. reweighing, threshold calibration, adversarial debiasing).

---

## 🔮 Future Improvements

- [ ] Upload your own real-world dataset instead of synthetic data
- [ ] Add debiasing techniques (reweighing, Fairlearn's `ExponentiatedGradient`)
- [ ] Support multi-class sensitive attributes (e.g. multiple racial groups)
- [ ] Add visual charts for group distributions and fairness metrics
- [ ] Export fairness audit report as PDF
- [ ] Add model comparison — compare a biased vs. debiased model side by side

---

## 🧠 Why This Matters

Loan approval AI is used by real banks to make decisions that affect people's financial futures. Regulatory frameworks like the **EU AI Act** and the **US Equal Credit Opportunity Act (ECOA)** increasingly require that automated decision systems be auditable for fairness. This project demonstrates what that auditing pipeline looks like in practice — making fairness a first-class concern, not an afterthought.

## 🙋 Author

Built by **[Egbayelo Olusegun]**
- GitHub: (https://github.com/Seanatist)
- LinkedIn: (www.linkedin.com/in/olusegun-egbayelo-74386b2a6)

---

*If this project was helpful, please consider giving it a ⭐ on GitHub!*
