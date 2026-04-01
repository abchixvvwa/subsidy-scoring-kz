# Subsidy Scoring System

> **Hackathon:** Decentrathon 5.0 · **Case:** #2 — Agricultural subsidy applicant scoring

---

## Problem

The current subsidy distribution operates on a first-come, first-served basis. This results in:

- Subsidies allocated to the fastest applicants rather than the most effective producers
- Absence of historical performance analysis across applicants
- No mechanism to detect systematic abuse or anomalous behavior
- Lack of transparent, auditable selection criteria

---

## Solution

An automated ML scoring system that analyzes **36,651 applications** from **457 unique applicants** and produces an objective, merit-based ranking using four complementary models.

### Design principles

- **Merit-based** — ranking derived from real performance indicators, not submission timing
- **Transparent** — every score is fully decomposable into factor contributions
- **Auditable** — anomaly detection flags suspicious applicants with human-readable explanations
- **Robust** — multiple independent models cross-validate each other

---

## ML Architecture

### Models

| Model | Type | Purpose | Metric |
|-------|------|---------|--------|
| LightGBM | Supervised classification | Predict applicant reliability | ROC-AUC |
| K-Means | Unsupervised clustering | Market segmentation | Silhouette score |
| Isolation Forest | Anomaly detection | Flag suspicious applicants | Contamination 5% |
| Composite Scoring | Rule-based | Interpretable baseline | Weighted sum |

### Why four models

A single model cannot satisfy all requirements simultaneously:

| Requirement | LightGBM | Composite | K-Means | Isolation Forest |
|-------------|----------|-----------|---------|-----------------|
| Predictive accuracy | + | — | — | — |
| Full explainability | — | + | + | + |
| No labeled data required | — | + | + | + |
| Strategic segmentation | — | — | + | — |
| Corruption / fraud detection | — | — | — | + |

### LightGBM — supervised model

**Target:** `y = 1` если выполнены все три условия:
`approval_rate >= 0.80`, `total_applications >= 10`,
`total_amount_received >= 40-й перцентиль по выборке`.
Это отсеивает заявителей с единичными заявками и фокусирует
модель на устойчивых производителях.  
**Features (7):** engineered financial and behavioral signals — no direct leakage from the target

| Feature | Description |
|---------|-------------|
| `total_applications` | Experience with the subsidy system |
| `total_amount_received` | Scale of the enterprise |
| `avg_amount` | Average subsidy per application |
| `max_amount` | Peak subsidy received |
| `unique_directions` | Diversification across directions |
| `unique_subsidy_types` | Diversification across programs |
| `last_activity_days` | Recency of activity |

**Note:** `approval_rate` and `rejection_rate` are explicitly excluded from X to prevent data leakage, since the target is derived from `approval_rate`.

**Cross-validation:** 5-fold, ROC-AUC 0.76 ± 0.07 — a realistic score for predicting reliability from behavioral signals only.

### K-Means — segmentation

Clusters applicants by behavioral patterns using five features: `approval_rate`, `total_amount_received`, `total_applications`, `unique_directions`, `last_activity_days`. Optimal K selected via elbow method. Labels assigned automatically by ranking cluster means.

### Isolation Forest — anomaly detection

Uses all 13 numeric features. Contamination fixed at 5%. Anomaly reasons expressed in business language (multiples of the population mean), not technical Z-scores.

### Composite Scoring — interpretable baseline

Weighted sum of five normalized factors:

| Factor | Weight |
|--------|--------|
| Approval rate | 0.30 |
| Total amount received | 0.25 |
| Total applications | 0.20 |
| Unique directions | 0.15 |
| Recency score | 0.10 |

---

## System Architecture

```
data/raw/                  Raw Excel data (36,651 rows)
    |
    v
src/preprocessing.py       Cleaning, aggregation, feature engineering
    |
    v
data/processed/
  features.csv             457 applicants × 16 features
    |
    +----> src/scoring.py         Composite score + explain_score()
    |           |
    |           v
    |      scored output
    |
    +----> src/ml_model.py        KMeans + IsolationForest + LightGBM
                |
                v
          data/processed/
            ml_results.csv        457 rows × 23 columns
            lgbm_model.pkl        Serialized LightGBM model
                |
                +----> src/api.py            FastAPI REST endpoints
                +----> app/dashboard.py      Streamlit dashboard
```

---

## API Endpoints

| Endpoint | Description |
|---------|-------------|
| `GET /health` | Service status and data load info |
| `GET /shortlist?top_n=20` | Top applicants by composite score |
| `GET /ml-shortlist?top_n=20` | Top applicants by LightGBM probability |
| `GET /applicant/{id}` | Full applicant profile with ML fields |
| `GET /explain/{id}` | Composite score factor decomposition |
| `GET /compare/{id}` | Both scores + discrepancy flag if divergence > 0.3 |
| `GET /clusters` | K-Means cluster statistics |
| `GET /anomalies` | List of Isolation Forest flagged applicants |

---

## Output Schema

`data/processed/ml_results.csv` — 23 columns:

```
applicant_id, total_applications, approved_count, rejected_count,
withdrawn_count, approval_rate, rejection_rate, total_amount_received,
avg_amount, max_amount, unique_directions, unique_subsidy_types,
last_activity_days, first_application_year, oblast, primary_direction,
cluster, cluster_label, is_anomaly, anomaly_score, anomaly_reason,
ml_probability, ml_rank
```

---

## Setup

```bash
git clone https://github.com/abchixvvwa/subsidy-scoring-kz.git
cd subsidy-scoring-kz
pip install -r requirements.txt
```

Place the source Excel file at `data/raw/subsidies_2025.xlsx`, then:

```bash
# 1. Feature engineering
python src/preprocessing.py

# 2. ML pipeline (KMeans + IsolationForest + LightGBM)
python src/ml_model.py

# 3. Dashboard
streamlit run app/dashboard.py

# 4. API (optional)
uvicorn src.api:app --reload
# Docs: http://localhost:8000/docs
```

---

## Data

- **Applications:** 36,651
- **Unique applicants:** 457
- **Format:** Microsoft Excel (.xlsx)
- **Regions (oblasts):** all 17 regions of Kazakhstan

---

## Limitations

1. **Data quality** — the model is sensitive to missing values and errors in source data
2. **Temporal drift** — features relevant today may lose predictive power within 1–2 years without retraining
3. **Geographic specificity** — trained on Kazakhstani data; not intended for direct transfer to other jurisdictions
4. **Class distribution** — reduced accuracy possible for regions with very few approved applications
5. **Decision support only** — the system is a tool for human decision-makers; final decisions remain with the authorized body

---

## Project

**Hackathon:** [Decentrathon 5.0](https://decentrathon.kz)  
**Case:** #2 — Automated scoring for agricultural subsidy applications  
**Organizer:** inDrive

---

## License

MIT — see [LICENSE](LICENSE)
