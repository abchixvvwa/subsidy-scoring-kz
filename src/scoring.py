"""
Subsidy Scoring System — Composite Scoring Model
=================================================
Взвешенная система оценки заявителей на сельскохозяйственные субсидии
Казахстана на основе 5 факторов, обоснованных нормативами субсидирования.

ВЕСА И ОБОСНОВАНИЕ:
  approval_rate          × 0.30  Надёжность заявителя. Субсидии выдаются только
                                  при полном пакете документов (племенные свидетель-
                                  ства, ветсправки и пр.). Высокий approval_rate
                                  означает стабильное соответствие требованиям.

  total_amount_received  × 0.25  Масштаб производства. Нормативы (150 000 тг/гол.
                                  за племенной КРС, 25 194 тг/гол. за овец и т.д.)
                                  пропорциональны поголовью/объёму — суммарные
                                  субсидии являются прокси реального размера хозяйства.

  total_applications     × 0.20  Опыт и активность. Заявитель с большим числом
                                  заявок хорошо знает систему, регулярно взаимо-
                                  действует с акиматом и своевременно обновляет
                                  документацию.

  unique_directions      × 0.15  Диверсификация производства. Хозяйство, рабо-
                                  тающее сразу в скотоводстве + овцеводстве +
                                  коневодстве, устойчивее к отраслевым рискам и
                                  использует весь спектр программ субсидирования.

  recency_score          × 0.10  Актуальность хозяйства. Рассчитывается как
                                  1 − normalize(last_activity_days): чем меньше
                                  дней с последней заявки, тем выше балл.

Запуск:
    python src/scoring.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Веса composite score (сумма == 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "approval_rate":         0.30,
    "total_amount_received": 0.25,
    "total_applications":    0.20,
    "unique_directions":     0.15,
    "recency_score":         0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Сумма весов должна равняться 1.0"

# Признаки, по которым делаем прямую Min-Max нормализацию
_DIRECT_NORM_COLS = [
    "approval_rate",
    "total_amount_received",
    "total_applications",
    "unique_directions",
]

# Читаемые названия для объяснений
_FACTOR_LABELS: dict[str, str] = {
    "approval_rate":         "Доля одобренных заявок",
    "total_amount_received": "Суммарно получено субсидий (тг)",
    "total_applications":    "Всего подано заявок",
    "unique_directions":     "Направлений субсидирования",
    "recency_score":         "Актуальность (дней с последней заявки)",
}


# ---------------------------------------------------------------------------
# normalize_features
# ---------------------------------------------------------------------------

def normalize_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-Max нормализация числовых признаков в диапазон [0, 1].

    Для каждой колонки из _DIRECT_NORM_COLS добавляет ``<col>_norm``.
    ``recency_score = 1 − normalize(last_activity_days)``:
    чем меньше дней с последней активности → тем выше балл актуальности.

    Parameters
    ----------
    features_df : pd.DataFrame
        Датафрейм профилей заявителей (выход build_applicant_features).

    Returns
    -------
    pd.DataFrame
        Копия датафрейма с дополнительными колонками ``*_norm``
        и ``recency_score``.
    """
    df = features_df.copy()

    # Прямая нормализация
    for col in _DIRECT_NORM_COLS:
        if col not in df.columns:
            raise KeyError(
                f"Ожидалась колонка '{col}', но она отсутствует в датафрейме."
            )
        col_min = df[col].min()
        col_max = df[col].max()
        denom = col_max - col_min
        df[f"{col}_norm"] = (
            (df[col] - col_min) / denom if denom != 0 else 0.0
        )

    # Обратная нормализация для last_activity_days → recency_score
    if "last_activity_days" not in df.columns:
        raise KeyError("Ожидалась колонка 'last_activity_days'.")

    col_min = df["last_activity_days"].min()
    col_max = df["last_activity_days"].max()
    denom = col_max - col_min
    last_norm = (
        (df["last_activity_days"] - col_min) / denom if denom != 0 else 0.0
    )
    df["recency_score"] = 1.0 - last_norm

    return df


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------

def compute_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет взвешенный composite score для каждого заявителя.

    Parameters
    ----------
    features_df : pd.DataFrame
        Датафрейм профилей заявителей.

    Returns
    -------
    pd.DataFrame
        Исходный датафрейм + колонки ``final_score`` и ``rank``
        (rank=1 — лучший заявитель), отсортированный по rank.
    """
    df = normalize_features(features_df)

    # Маппинг: логическое имя фактора → нормализованная колонка
    norm_col_map: dict[str, str] = {
        "approval_rate":         "approval_rate_norm",
        "total_amount_received": "total_amount_received_norm",
        "total_applications":    "total_applications_norm",
        "unique_directions":     "unique_directions_norm",
        "recency_score":         "recency_score",  # уже в [0,1]
    }

    score = pd.Series(0.0, index=df.index)
    for factor, weight in WEIGHTS.items():
        norm_col = norm_col_map[factor]
        score += df[norm_col] * weight

    df["final_score"] = score.round(6)
    df["rank"] = (
        df["final_score"].rank(ascending=False, method="min").astype(int)
    )

    return df.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# get_shortlist
# ---------------------------------------------------------------------------

def get_shortlist(
    features_df: pd.DataFrame,
    top_n: int = 20,
    scored_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Возвращает топ-N заявителей по composite score.

    Parameters
    ----------
    features_df : pd.DataFrame
        Датафрейм профилей заявителей.
    top_n : int
        Количество позиций в шортлисте.
    scored_df : pd.DataFrame | None
        Уже посчитанный датафрейм (передайте, чтобы избежать
        повторного вычисления при нескольких вызовах).

    Returns
    -------
    pd.DataFrame
        Топ-N строк с колонками:
        rank, applicant_id, final_score, approval_rate,
        total_amount_received, primary_direction, oblast, total_applications.
    """
    if scored_df is None:
        scored_df = compute_scores(features_df)

    desired_cols = [
        "rank",
        "applicant_id",
        "final_score",
        "approval_rate",
        "total_amount_received",
        "primary_direction",
        "oblast",
        "total_applications",
    ]
    available = [c for c in desired_cols if c in scored_df.columns]

    return (
        scored_df.nsmallest(top_n, "rank")[available]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# explain_score
# ---------------------------------------------------------------------------

def explain_score(applicant_id: str, features_df: pd.DataFrame) -> dict:
    """
    Детальное объяснение скора конкретного заявителя.

    Parameters
    ----------
    applicant_id : str
        ID заявителя (первые 10 символов app_num).
    features_df : pd.DataFrame
        Датафрейм профилей заявителей.

    Returns
    -------
    dict
        applicant_id   — str
        final_score    — float
        rank           — int
        factors        — dict[str, dict] (по одному на каждый из 5 факторов):
                           label, raw_value, normalized, weight, contribution
        top_factor     — str  (фактор с наибольшим вкладом)
        explanation_ru — str  (читаемое объяснение на русском)
    """
    scored = compute_scores(features_df)

    rows = scored[scored["applicant_id"].astype(str) == str(applicant_id)]
    if rows.empty:
        raise ValueError(
            f"Заявитель '{applicant_id}' не найден в датафрейме."
        )
    row = rows.iloc[0]

    # (нормализованная колонка, исходная колонка для raw_value)
    factor_cols: dict[str, tuple[str, str]] = {
        "approval_rate":         ("approval_rate_norm",        "approval_rate"),
        "total_amount_received": ("total_amount_received_norm", "total_amount_received"),
        "total_applications":    ("total_applications_norm",    "total_applications"),
        "unique_directions":     ("unique_directions_norm",     "unique_directions"),
        "recency_score":         ("recency_score",              "last_activity_days"),
    }

    factors: dict[str, dict] = {}
    for factor, (norm_col, raw_col) in factor_cols.items():
        weight       = WEIGHTS[factor]
        normalized   = float(row[norm_col])
        raw_val      = float(row[raw_col]) if raw_col in row.index else np.nan
        contribution = round(weight * normalized, 6)

        factors[factor] = {
            "label":        _FACTOR_LABELS[factor],
            "raw_value":    round(raw_val, 4),
            "normalized":   round(normalized, 4),
            "weight":       weight,
            "contribution": contribution,
        }

    top_factor  = max(factors, key=lambda f: factors[f]["contribution"])
    final_score = float(row["final_score"])
    rank        = int(row["rank"])

    # ── Русскоязычное объяснение ─────────────────────────────────────────
    sorted_factors = sorted(
        factors.items(), key=lambda x: x[1]["contribution"], reverse=True
    )
    parts = []
    for factor, info in sorted_factors:
        raw  = info["raw_value"]
        cont = info["contribution"]
        if factor == "approval_rate":
            parts.append(f"доля одобренных заявок {raw * 100:.1f}% (вклад {cont:.3f})")
        elif factor == "total_amount_received":
            parts.append(f"получено субсидий {raw:,.0f} тг (вклад {cont:.3f})")
        elif factor == "total_applications":
            parts.append(f"подано заявок {int(raw)} шт. (вклад {cont:.3f})")
        elif factor == "unique_directions":
            parts.append(f"направлений субсидирования {int(raw)} (вклад {cont:.3f})")
        elif factor == "recency_score":
            parts.append(f"последняя активность {int(raw)} дн. назад (вклад {cont:.3f})")

    explanation_ru = (
        f"Заявитель {applicant_id} | скор: {final_score:.4f} | место: #{rank}\n"
        f"Ключевой фактор: «{factors[top_factor]['label']}»\n"
        "Обусловлен (от большего вклада к меньшему):\n"
        + "\n".join(f"  • {p}" for p in parts)
    )

    return {
        "applicant_id":   applicant_id,
        "final_score":    round(final_score, 6),
        "rank":           rank,
        "factors":        factors,
        "top_factor":     top_factor,
        "explanation_ru": explanation_ru,
    }


# ---------------------------------------------------------------------------
# save_scores
# ---------------------------------------------------------------------------

def save_scores(scored_df: pd.DataFrame, output_path: str) -> None:
    """Сохраняет полный ranked датафрейм в CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[save_scores] Сохранено → {out.resolve()}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FEATURES_PATH = "data/processed/features.csv"
    SCORES_PATH   = "data/processed/scores.csv"

    print("=" * 60)
    print("  SUBSIDY SCORING SYSTEM — Composite Score")
    print("=" * 60)

    features = pd.read_csv(FEATURES_PATH)
    print(f"\nЗагружено заявителей: {len(features)}")

    # Расчёт скоров и сохранение
    scored = compute_scores(features)
    save_scores(scored, SCORES_PATH)

    # Топ-5
    shortlist = get_shortlist(features, top_n=5, scored_df=scored)
    print("\n=== ТОП-5 ЗАЯВИТЕЛЕЙ ===")
    print(shortlist.to_string(index=False))

    # Объяснение для лучшего
    top_id = str(shortlist.iloc[0]["applicant_id"])
    exp = explain_score(top_id, features)

    print(f"\n=== ОБЪЯСНЕНИЕ ДЛЯ {top_id} ===")
    print(exp["explanation_ru"])

    print("\n--- Вклад каждого фактора ---")
    for factor, info in sorted(
        exp["factors"].items(), key=lambda x: x[1]["contribution"], reverse=True
    ):
        bar = chr(9608) * int(info["contribution"] * 50)
        print(f"  {info['label']:<44} {bar}  {info['contribution']:.4f}")
