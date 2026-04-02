# КазнаЛинза — GovTech-аналитика субсидий

> **Hackathon:** Decentrathon 5.0 · **Case:** #2 — Agricultural subsidy applicant scoring (Kazakhstan)

Система анализирует историю заявок на субсидии, строит **merit-based** рейтинг заявителей, **сегментирует** их (K-Means), **детектирует аномалии** (Isolation Forest) и даёт **интерпретируемый composite-score** (взвешенная формула) + **ML-скор** (LightGBM).

---

## Быстрый старт (после клонирования)

### 1. Окружение

```bash
cd subsidy-scoring-kz   # или ваша папка с репозиторием
python -m pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
cd C:\path\to\hack
python -m pip install -r requirements.txt
```

### 2. Данные

Положите исходный Excel в `data/raw/subsidies_2025.xlsx` (или путь из `src/preprocessing.py`).

### 3. Пайплайн

```bash
python src/preprocessing.py    # features.csv
python src/ml_model.py         # ml_results.csv + lgbm_model.pkl
```

### 4. Интерфейс (выберите один)

| Вариант | Команда | URL |
|--------|---------|-----|
| **Веб-дашборд (рекомендуется для демо)** | `python app/server.py` | http://localhost:8502 |
| Streamlit | `streamlit run app/dashboard.py` | http://localhost:8501 (порт по умолчанию) |
| FastAPI | `uvicorn src.api:app --reload` | http://localhost:8000/docs |

Для веб-UI сервер отдаёт `app/index.html` и JSON API (`/api/all`, `/api/explain/{id}`, …). Запускайте **`python app/server.py` из корня репозитория**, чтобы пути к `data/` были корректны.

---

## Архитектура данных

```
data/raw/*.xlsx
    → src/preprocessing.py
    → data/processed/features.csv          (457 заявителей × признаки)

features.csv
    → src/scoring.py           → composite: final_score, rank
    → src/ml_model.py          → KMeans, IsolationForest, LightGBM

    → data/processed/ml_results.csv
    → data/processed/lgbm_model.pkl
```

---

## Модели (как в коде)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| **Composite score** | `src/scoring.py` | Интерпретируемый скор из 5 факторов (approval, суммы, заявки, направления, актуальность) |
| **LightGBM** | `src/ml_model.py` | Бинарная классификация: **target = 1** для заявителей с `approval_rate >= median(approval_rate)` (топ-50% по одобрению). Признаки **без** `approval_rate` / `rejection_rate` (анти-утечка). Вероятность → `ml_probability`, ранг → `ml_rank`. Калибровка: `CalibratedClassifierCV` |
| **K-Means** | `src/ml_model.py` | Кластеризация, читаемые `cluster_label` |
| **Isolation Forest** | `src/ml_model.py` | Аномалии, `contamination=0.05`, `is_anomaly`, `anomaly_reason` |

> В более ранних описаниях встречался «сложный» таргет LightGBM (медианы по сумме и т.д.). **Фактическая реализация в `train_lightgbm()` — таргет top-50% по `approval_rate`.** Для жюри лучше опираться на этот абзац и на код.

---

## REST API (FastAPI)

Полноценное API в `src/api.py` (при запуске `uvicorn`):

| Метод | Описание |
|-------|----------|
| `GET /health` | Статус и загрузка данных |
| `GET /shortlist` | Топ по composite |
| `GET /ml-shortlist` | Топ по `ml_probability` |
| `GET /applicant/{id}` | Профиль заявителя |
| `GET /explain/{id}` | Разложение composite по факторам |
| `GET /compare/{id}` | Сравнение ML vs composite |
| `GET /clusters` | Статистика кластеров |
| `GET /anomalies` | Список аномалий |

Лёгкий сервер для фронта: `app/server.py` — `/api/all`, `/api/explain/{id}`, `/api/debug`, отдача `index.html`.

---

## Схема `ml_results.csv`

Базовые колонки из пайплайна (плюс при необходимости):

`applicant_id`, поведенческие и суммовые признаки, `cluster`, `cluster_label`, `is_anomaly`, `anomaly_score`, `anomaly_reason`, `ml_probability`, `ml_rank`.

При объединении с composite в веб-сервере в ответ добавляются **`final_score`** и **`rank`** (если есть `features.csv`).

---

## Известные ограничения и честный чеклист для хакатона

1. **Качество данных** — пропуски и ошибки в Excel сильно влияют на скоринг.  
2. **ML probability** — не «вероятность одобрения следующей заявки», а **калиброванная оценка модели** по выбранному таргету; интерпретируйте как вспомогательный сигнал.  
3. **Экстремальные `ml_probability`** — после калибровки мало кто может быть ровно 0.0/1.0; на UI не округляйте до «0% / 100%» без подписи сырого значения.  
4. **Решение поддержки** — система для аналитики и приоритизации, не замена юридическому решению.

**Что хорошо для демо:**  
есть данные, воспроизводимый пайплайн, несколько моделей, explainability (composite + факторы), аномалии, UI.

---

## Данные (типовой датасет)

- **Заявок:** 36 651  
- **Уникальных заявителей:** 457  
- **Регионы:** области Казахстана  

---

## Кейс и лицензия

- **Hackathon:** [Decentrathon 5.0](https://decentrathon.kz)  
- **Case:** #2 — Automated scoring for agricultural subsidy applications  
- **License:** MIT — см. [LICENSE](LICENSE)
