"""
Subsidy Scoring System — ML Pipeline
=====================================
Два ML-модуля для анализа заявителей на сельскохозяйственные субсидии:

  1. Кластеризация K-Means
     Группирует заявителей по поведенческим паттернам, автоматически
     присваивая каждому кластеру читаемое русскоязычное название.

  2. Обнаружение аномалий — Isolation Forest
     Выявляет подозрительных заявителей и объясняет причину аномалии
     путём сравнения их признаков со средними по выборке.

Запуск:
    python src/ml_model.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    print("[WARN] lightgbm не установлен. train_lightgbm() будет пропущена.")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

FEATURES_PATH   = "data/processed/features.csv"
ML_RESULTS_PATH = "data/processed/ml_results.csv"
MODEL_PATH      = "data/processed/lgbm_model.pkl"

# Признаки для LightGBM (approval_rate и rejection_rate исключены — утечка данных:
# таргет y строится из approval_rate, поэтому оставлять её в X нельзя)
LGBM_FEATURES = [
    "total_applications",
    "total_amount_received",
    "avg_amount",
    "max_amount",
    "unique_directions",
    "unique_subsidy_types",
    "last_activity_days",
]

# Признаки для кластеризации
CLUSTER_FEATURES = [
    "approval_rate",
    "total_amount_received",
    "total_applications",
    "unique_directions",
    "last_activity_days",
]

# Признаки, которые НЕ используются при обнаружении аномалий
# (нечисловые или ненужные идентификаторы)
EXCLUDE_FROM_ANOMALY = [
    "applicant_id",
    "oblast",
    "primary_direction",
]

# Порог для флага аномалии: используем contamination IsolationForest
CONTAMINATION = 0.05
RANDOM_STATE  = 42


# ---------------------------------------------------------------------------
# МОДЕЛЬ 1: Кластеризация K-Means
# ---------------------------------------------------------------------------

def _calc_inertia(X_scaled: np.ndarray, k_range: range) -> list[float]:
    """Вычисляет инерцию KMeans для каждого K в диапазоне."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias


def _find_elbow_k(inertias: list[float], k_range: range) -> int:
    """
    Метод локтя: находит K, в котором происходит наибольшее
    относительное снижение прироста инерции.
    """
    k_list = list(k_range)
    if len(k_list) <= 2:
        return k_list[0]

    # Вычисляем «вторую производную» кривой инерции
    diffs1 = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    diffs2 = [diffs1[i] - diffs1[i + 1] for i in range(len(diffs1) - 1)]

    # Локоть — там, где вторая производная максимальна (наибольший перегиб)
    elbow_idx = int(np.argmax(diffs2)) + 1   # +1 потому что diffs2 сдвинут на 2
    optimal_k = k_list[elbow_idx]
    return int(optimal_k)


def _assign_cluster_labels(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.Series:
    """
    Автоматически определяет читаемое название для каждого кластера
    на основе средних значений ключевых признаков внутри кластера.

    Логика:
    - «Крупные эффективные»  — высокий total_amount_received + высокий approval_rate
    - «Средние стабильные»   — средние значения всех показателей
    - «Мелкие рискованные»   — низкий approval_rate / мало заявок
    - «Неактивные»           — большое last_activity_days (давно не подавали)
    - «Диверсифицированные»  — много уникальных направлений
    """
    cluster_means = df.groupby(cluster_col)[CLUSTER_FEATURES].mean()
    n_clusters    = len(cluster_means)

    # Нормализуем средние по кластерам в [0,1] для сравнения
    normed = (cluster_means - cluster_means.min()) / (
        cluster_means.max() - cluster_means.min() + 1e-9
    )

    # Вычисляем композитный «рейтинг» для каждого кластера
    # (approval_rate и total_amount_received — позитивные; last_activity_days — негативный)
    cluster_score = (
        normed["approval_rate"]         * 0.35 +
        normed["total_amount_received"] * 0.30 +
        normed["total_applications"]    * 0.15 +
        normed["unique_directions"]     * 0.10 +
        (1 - normed["last_activity_days"]) * 0.10  # инверсия: меньше дней = лучше
    )

    # Ранжируем кластеры: 0 = лучший, n-1 = худший
    ranked = cluster_score.rank(ascending=False).astype(int)

    # Стандартный набор меток (от лучшего к худшему)
    label_templates = [
        "Крупные эффективные",
        "Средние стабильные",
        "Малые активные",
        "Малые рискованные",
        "Неактивные низкоэффективные",
    ]
    # Если кластеров меньше, обрезаем
    label_templates = label_templates[:n_clusters]

    # Дополняем, если кластеров больше заготовленных меток
    while len(label_templates) < n_clusters:
        label_templates.append(f"Кластер {len(label_templates) + 1}")

    # Собираем маппинг: cluster_id → label
    mapping = {}
    for cluster_id, rank_pos in ranked.items():
        mapping[cluster_id] = label_templates[int(rank_pos) - 1]

    return df[cluster_col].map(mapping)


def cluster_applicants(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Кластеризует заявителей методом K-Means.

    Шаги:
    1. Выбирает признаки из CLUSTER_FEATURES, нормализует StandardScaler.
    2. Подбирает оптимальное K методом локтя (K ∈ [2..8]).
    3. Обучает KMeans с оптимальным K.
    4. Добавляет колонки ``cluster`` и ``cluster_label``.

    Parameters
    ----------
    features_df : pd.DataFrame
        Датафрейм признаков заявителей.

    Returns
    -------
    pd.DataFrame
        Исходный датафрейм с добавленными колонками ``cluster`` и ``cluster_label``.
    """
    df = features_df.copy()

    # ── 1. Подготовка данных ─────────────────────────────────────────────────
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют признаки для кластеризации: {missing}")

    X = df[CLUSTER_FEATURES].fillna(df[CLUSTER_FEATURES].median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 2. Метод локтя ───────────────────────────────────────────────────────
    k_range = range(2, 9)   # K от 2 до 8
    print("  [K-Means] Подборка оптимального K методом локтя...")
    inertias = _calc_inertia(X_scaled, k_range)

    optimal_k = _find_elbow_k(inertias, k_range)
    print(f"  [K-Means] Инерции по K: { {k: round(v, 1) for k, v in zip(k_range, inertias)} }")
    print(f"  [K-Means] Оптимальное K = {optimal_k}")

    # ── 3. Обучение KMeans ───────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # ── 4. Читаемые названия кластеров ───────────────────────────────────────
    df["cluster_label"] = _assign_cluster_labels(df, cluster_col="cluster")

    print(f"  [K-Means] Распределение по кластерам:\n"
          + df.groupby(["cluster", "cluster_label"])
               .size()
               .reset_index(name="count")
               .to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# МОДЕЛЬ 2: Обнаружение аномалий — Isolation Forest
# ---------------------------------------------------------------------------

def _build_anomaly_reason(row: pd.Series, means: pd.Series, stds: pd.Series) -> str:
    """
    Сравнивает значения заявителя со средними по всей выборке,
    выбирает топ-3 наиболее отклоняющихся признака.
    """
    # Метки для признаков (числовые)
    feature_labels = {
        "total_applications":    "кол-во заявок",
        "approved_count":        "кол-во одобренных",
        "rejected_count":        "кол-во отклонённых",
        "withdrawn_count":       "кол-во отозванных",
        "approval_rate":         "доля одобрения",
        "rejection_rate":        "доля отклонения",
        "total_amount_received": "сумма субсидий",
        "avg_amount":            "средняя сумма",
        "max_amount":            "максимальная сумма",
        "unique_directions":     "уникальных направлений",
        "unique_subsidy_types":  "типов субсидий",
        "last_activity_days":    "дней с последней заявки",
        "first_application_year":"год первой заявки",
    }

    deviations = {}
    for feat, label in feature_labels.items():
        if feat not in row.index or feat not in means.index:
            continue
        std = stds.get(feat, 0)
        if std < 1e-9:
            continue
        z = (row[feat] - means[feat]) / std
        deviations[feat] = (z, label)

    # Сортируем по величине отклонения (убывание), z используем только для ранга
    sorted_devs = sorted(deviations.items(), key=lambda x: abs(x[1][0]), reverse=True)
    top_devs = sorted_devs[:3]

    if not top_devs:
        return "Необычное сочетание признаков без явного доминирующего отклонения."

    def _fmt_tenge(v: float) -> str:
        """Умное форматирование суммы: тг / млн тг / млрд тг."""
        if abs(v) >= 1_000_000_000:
            return f"{v / 1_000_000_000:.2f} млрд тг"
        elif abs(v) >= 1_000_000:
            return f"{v / 1_000_000:.1f} млн тг"
        else:
            return f"{v:,.0f} тг"

    def _ratio_phrase(val: float, mean_val: float) -> str:
        """Кратность превышения / доля от среднего — без упоминания z-score."""
        if mean_val < 1e-9:
            return "значительно отличается от среднего значения"
        ratio = val / mean_val
        if ratio >= 1.05:
            return f"в {ratio:.1f} раз выше среднего значения"
        elif ratio <= 0.95:
            return f"составляет лишь {ratio * 100:.0f}% от среднего значения"
        else:
            return "близко к среднему значению"

    parts = []
    for feat, (_z, label) in top_devs:
        val      = row[feat]
        mean_val = means[feat]
        rphrase  = _ratio_phrase(val, mean_val)

        if feat in ("approval_rate", "rejection_rate"):
            parts.append(
                f"{label}: {val:.1%} ({rphrase} {mean_val:.1%})"
            )
        elif feat in ("total_amount_received", "avg_amount", "max_amount"):
            parts.append(
                f"{label}: {_fmt_tenge(val)} ({rphrase} {_fmt_tenge(mean_val)})"
            )
        elif feat == "last_activity_days":
            parts.append(
                f"{label}: {int(val)} дн. ({rphrase} {int(mean_val)} дн.)"
            )
        elif feat == "first_application_year":
            diff     = int(val - mean_val)
            diff_str = f"на {abs(diff)} г. {'раньше' if diff < 0 else 'позже'} среднего ({int(mean_val)} г.)"
            parts.append(f"{label}: {int(val)} ({diff_str})")
        else:
            parts.append(
                f"{label}: {val:.1f} ({rphrase} {mean_val:.1f})"
            )

    return "Аномалия: " + "; ".join(parts) + "."


def detect_anomalies(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Обнаруживает аномальных заявителей с помощью Isolation Forest.

    Шаги:
    1. Выбирает все числовые признаки (кроме EXCLUDE_FROM_ANOMALY).
    2. Обучает IsolationForest(contamination=0.05).
    3. Добавляет колонки ``is_anomaly``, ``anomaly_score``, ``anomaly_reason``.

    Parameters
    ----------
    features_df : pd.DataFrame
        Датафрейм признаков заявителей (уже прошедший кластеризацию или нет).

    Returns
    -------
    pd.DataFrame
        Исходный датафрейм с добавленными колонками.
    """
    df = features_df.copy()

    # ── 1. Числовые признаки ─────────────────────────────────────────────────
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_FROM_ANOMALY
           and c not in ("cluster",)          # не берём синтетические колонки
    ]

    X = df[num_cols].fillna(df[num_cols].median())

    # ── 2. Обучение IsolationForest ──────────────────────────────────────────
    print(f"\n  [IsolationForest] Признаки ({len(num_cols)}): {num_cols}")
    iso = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200,
    )
    preds  = iso.fit_predict(X)       # -1 = аномалия, 1 = норма
    scores = iso.score_samples(X)     # чем ниже — тем подозрительнее

    df["is_anomaly"]    = preds == -1
    df["anomaly_score"] = scores.round(6)

    # ── 3. Объяснения аномалий ───────────────────────────────────────────────
    means = X.mean()
    stds  = X.std()

    def _reason(row: pd.Series) -> str:
        if not row["is_anomaly"]:
            return ""
        return _build_anomaly_reason(row[num_cols], means, stds)

    df["anomaly_reason"] = df.apply(_reason, axis=1)

    n_anomalies = df["is_anomaly"].sum()
    print(f"  [IsolationForest] Обнаружено аномалий: {n_anomalies} "
          f"из {len(df)} ({n_anomalies / len(df):.1%})")

    return df


# ---------------------------------------------------------------------------
# МОДЕЛЬ 3: LightGBM — Supervised Classification
# ---------------------------------------------------------------------------

def train_lightgbm(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает LightGBM classifier для предсказания надёжности заявителя.

    Таргет (y): 1 если заявитель входит в топ-33% по approval_rate
                И total_amount_received > медианы — иначе 0.
    Признаки:   LGBM_FEATURES (без approval_rate и rejection_rate — нет leakage).
    """
    if not _LGBM_AVAILABLE:
        print("  [LightGBM] Пропускаем — lightgbm не установлен.")
        features_df["ml_probability"] = np.nan
        features_df["ml_rank"]        = np.nan
        return features_df

    df = features_df.copy()

    # ── 1. Подготовка данных ─────────────────────────────────────────────────
    missing = [c for c in LGBM_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют признаки для LightGBM: {missing}")

    X = df[LGBM_FEATURES].fillna(df[LGBM_FEATURES].median())

    # Таргет: надёжный заявитель с реальным опытом
    # approval_rate >= 0.80  — высокая доля одобрений
    # total_applications >= 10  — исключаем единичные заявки
    # total_amount_received >= quantile(0.40)  — значимый объём субсидий
    amount_q40 = features_df["total_amount_received"].quantile(0.40)
    y = (
        (features_df["approval_rate"] >= 0.80) &
        (features_df["total_applications"] >= 10) &
        (features_df["total_amount_received"] >= amount_q40)
    ).astype(int)

    pos_rate = y.mean()
    print(f"\n  [LightGBM] Target: approval_rate >= 0.80 "
          f"AND applications >= 10 AND amount >= {amount_q40/1e6:.1f}M")
    print(f"  Позитивных примеров: {int(y.sum())} из {len(y)}")
    print(f"  [LightGBM] Подготовка: {len(X)} записей, "
          f"target=1: {int(y.sum())} ({pos_rate:.1%}), "
          f"target=0: {int((~y.astype(bool)).sum())}")

    # ── 2. Кросс-валидация ───────────────────────────────────────────────────
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    cv_scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    mean_auc = cv_scores.mean()
    std_auc  = cv_scores.std()
    print(f"  [LightGBM] 5-Fold CV ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  [LightGBM] По фолдам: {[round(s, 4) for s in cv_scores]}")

    # ── 3. Обучение на всех данных ───────────────────────────────────────────
    model.fit(X, y)
    print("  [LightGBM] Модель обучена на полных данных.")

    # ── 4. Предсказания ──────────────────────────────────────────────────────
    df["ml_probability"] = model.predict_proba(X)[:, 1].round(6)
    df["ml_rank"]        = df["ml_probability"].rank(
        ascending=False, method="first"
    ).astype(int)

    # ── 5. Feature importance ────────────────────────────────────────────────
    importance_df = pd.DataFrame({
        "Признак":              LGBM_FEATURES,
        "Важность (split)":    model.feature_importances_,
    }).sort_values("Важность (split)", ascending=False).reset_index(drop=True)
    importance_df.index += 1
    print("\n  [LightGBM] Feature Importances:")
    print(importance_df.to_string())

    # ── 6. Сохранение модели ─────────────────────────────────────────────────
    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"  [LightGBM] Модель сохранена → {model_path.resolve()}")

    return df


# ---------------------------------------------------------------------------
# Полный ML-пайплайн
# ---------------------------------------------------------------------------

def run_full_ml_pipeline(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Запускает все три модели последовательно и сохраняет результат.

    1. cluster_applicants  → добавляет ``cluster``, ``cluster_label``
    2. detect_anomalies    → добавляет ``is_anomaly``, ``anomaly_score``, ``anomaly_reason``
    3. train_lightgbm      → добавляет ``ml_probability``, ``ml_rank``
    4. Сохраняет итоговый датафрейм в ML_RESULTS_PATH

    Parameters
    ----------
    features_df : pd.DataFrame
        Исходный датафрейм признаков.

    Returns
    -------
    pd.DataFrame
        Датафрейм со всеми новыми колонками от всех трёх моделей.
    """
    print("\n" + "=" * 60)
    print("  SUBSIDY ML PIPELINE — запуск")
    print("=" * 60)

    # Шаг 1: Кластеризация
    print("\n[1/3] Кластеризация K-Means...")
    df = cluster_applicants(features_df)

    # Шаг 2: Обнаружение аномалий
    print("\n[2/3] Обнаружение аномалий (Isolation Forest)...")
    df = detect_anomalies(df)

    # Шаг 3: LightGBM supervised classification
    print("\n[3/3] Supervised ML — LightGBM...")
    df = train_lightgbm(df)

    # Сохранение
    out_path = Path(ML_RESULTS_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Результаты сохранены → {out_path.resolve()}")

    return df


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    # Корень проекта — на уровень выше src/
    _root = Path(__file__).resolve().parent.parent
    os.chdir(_root)

    # ── Загрузка данных ──────────────────────────────────────────────────────
    features = pd.read_csv(FEATURES_PATH)
    print(f"Загружено заявителей: {len(features)} | Признаков: {len(features.columns)}")

    # ── Запуск пайплайна ─────────────────────────────────────────────────────
    result = run_full_ml_pipeline(features)

    # ── Статистика по кластерам ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  СТАТИСТИКА ПО КЛАСТЕРАМ")
    print("=" * 60)

    cluster_stats = (
        result.groupby(["cluster", "cluster_label"])
        .agg(
            count               = ("applicant_id",       "count"),
            avg_approval_rate   = ("approval_rate",       "mean"),
            avg_amount_received = ("total_amount_received","mean"),
            avg_applications    = ("total_applications",  "mean"),
            avg_directions      = ("unique_directions",   "mean"),
            avg_activity_days   = ("last_activity_days",  "mean"),
        )
        .reset_index()
    )

    cluster_stats["avg_approval_rate"]   = cluster_stats["avg_approval_rate"].map("{:.1%}".format)
    cluster_stats["avg_amount_received"] = cluster_stats["avg_amount_received"].map("{:,.0f} тг".format)
    cluster_stats["avg_applications"]    = cluster_stats["avg_applications"].map("{:.1f}".format)
    cluster_stats["avg_directions"]      = cluster_stats["avg_directions"].map("{:.1f}".format)
    cluster_stats["avg_activity_days"]   = cluster_stats["avg_activity_days"].map("{:.0f} дн.".format)

    cluster_stats.columns = [
        "Кластер", "Название", "Кол-во", "Доля одобр.",
        "Ср. сумма субсидий", "Ср. заявок", "Ср. направлений", "Ср. активность",
    ]
    print(cluster_stats.to_string(index=False))

    # ── Список аномалий ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ОБНАРУЖЕННЫЕ АНОМАЛИИ")
    print("=" * 60)

    anomalies = (
        result[result["is_anomaly"]]
        .sort_values("anomaly_score")   # самые подозрительные — первые
        [["applicant_id", "anomaly_score", "cluster_label", "anomaly_reason"]]
        .reset_index(drop=True)
    )

    if anomalies.empty:
        print("  Аномалий не обнаружено.")
    else:
        print(f"  Всего аномалий: {len(anomalies)}\n")
        for _, row in anomalies.iterrows():
            print(f"  ID: {row['applicant_id']}")
            print(f"  Кластер: {row['cluster_label']}")
            print(f"  Anomaly score: {row['anomaly_score']:.4f}")
            print(f"  {row['anomaly_reason']}")
            print()

    print(f"Результаты сохранены в: {ML_RESULTS_PATH}")

    # ── Проверка распределения ml_probability ────────────────────────────────
    print("\n" + "=" * 60)
    print("  ml_probability DISTRIBUTION CHECK")
    print("=" * 60)
    print(result["ml_probability"].describe().round(4).to_string())
    print(f"\n  Unique values (top 10): {sorted(result['ml_probability'].unique())[:10]}")
    print(f"  Values == 1.0: {(result['ml_probability'] == 1.0).sum()}")
    print(f"  Values == 0.0: {(result['ml_probability'] == 0.0).sum()}")
    print(f"  Values in [0.2, 0.8]: {((result['ml_probability'] > 0.2) & (result['ml_probability'] < 0.8)).sum()}")

