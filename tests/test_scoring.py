"""
Subsidy Scoring System — Pytest Test Suite
==========================================
Запуск всех тестов:
    pytest tests/ -v

Запуск конкретного теста:
    pytest tests/test_scoring.py::test_scores_range -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Добавляем корень репозитория в sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.scoring import compute_scores, get_shortlist, explain_score  # noqa: E402

# ---------------------------------------------------------------------------
# Ожидаемые колонки в features.csv
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "applicant_id",
    "total_applications",
    "approved_count",
    "rejected_count",
    "withdrawn_count",
    "approval_rate",
    "rejection_rate",
    "total_amount_received",
    "avg_amount",
    "max_amount",
    "unique_directions",
    "unique_subsidy_types",
    "last_activity_days",
    "first_application_year",
    "oblast",
    "primary_direction",
]

EXPECTED_FACTOR_KEYS = {
    "approval_rate",
    "total_amount_received",
    "total_applications",
    "unique_directions",
    "recency_score",
}


# ---------------------------------------------------------------------------
# Тесты датасета
# ---------------------------------------------------------------------------

def test_features_shape(features_df: pd.DataFrame) -> None:
    """features.csv содержит > 100 строк и все обязательные колонки."""
    assert len(features_df) > 100, (
        f"Ожидалось > 100 строк, получено {len(features_df)}"
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in features_df.columns]
    assert not missing, f"Отсутствуют обязательные колонки: {missing}"


def test_approval_rate_valid(features_df: pd.DataFrame) -> None:
    """Все значения approval_rate строго в диапазоне [0.0, 1.0]."""
    col = features_df["approval_rate"]
    assert col.between(0.0, 1.0).all(), (
        f"approval_rate выходит за [0, 1]: "
        f"min={col.min():.4f}, max={col.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Тесты скоринга
# ---------------------------------------------------------------------------

def test_scores_range(scored_df: pd.DataFrame) -> None:
    """Все final_score строго в диапазоне [0.0, 1.0]."""
    col = scored_df["final_score"]
    assert col.between(0.0, 1.0).all(), (
        f"final_score выходит за [0, 1]: "
        f"min={col.min():.6f}, max={col.max():.6f}"
    )


def test_no_nulls_in_score(scored_df: pd.DataFrame) -> None:
    """В колонке final_score нет пропущенных значений (NaN)."""
    null_count = scored_df["final_score"].isna().sum()
    assert null_count == 0, f"Найдено {null_count} NaN в final_score"


def test_rank_starts_at_one(scored_df: pd.DataFrame) -> None:
    """Минимальный ранг равен 1 (лучший заявитель)."""
    assert scored_df["rank"].min() == 1, (
        f"Ожидался минимальный ранг 1, получено {scored_df['rank'].min()}"
    )


# ---------------------------------------------------------------------------
# Тесты шортлиста
# ---------------------------------------------------------------------------

def test_shortlist_length(features_df: pd.DataFrame) -> None:
    """get_shortlist(df, top_n=10) возвращает ровно 10 строк."""
    sl = get_shortlist(features_df, top_n=10)
    assert len(sl) == 10, f"Ожидалось 10 строк, получено {len(sl)}"


def test_shortlist_sorted(features_df: pd.DataFrame) -> None:
    """Ранги в шортлисте идут строго по возрастанию: 1, 2, 3 ... 10."""
    sl = get_shortlist(features_df, top_n=10)
    ranks = sl["rank"].tolist()
    assert ranks == sorted(ranks), f"Ранги не отсортированы: {ranks}"
    assert ranks[0] == 1, f"Первый ранг должен быть 1, получено {ranks[0]}"


def test_shortlist_columns(features_df: pd.DataFrame) -> None:
    """Шортлист содержит обязательные колонки."""
    expected = {
        "rank", "applicant_id", "final_score", "approval_rate",
        "total_amount_received", "primary_direction", "oblast", "total_applications",
    }
    sl = get_shortlist(features_df, top_n=5)
    missing = expected - set(sl.columns)
    assert not missing, f"Шортлисту не хватает колонок: {missing}"


# ---------------------------------------------------------------------------
# Тесты explain_score
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def top_applicant_id(features_df: pd.DataFrame) -> str:
    """Возвращает applicant_id заявителя с рангом #1."""
    sl = get_shortlist(features_df, top_n=1)
    return str(sl.iloc[0]["applicant_id"])


def test_explain_keys(features_df: pd.DataFrame, top_applicant_id: str) -> None:
    """explain_score() возвращает dict со всеми обязательными ключами верхнего уровня."""
    result = explain_score(top_applicant_id, features_df)
    expected_keys = {"applicant_id", "final_score", "factors", "top_factor", "explanation_ru"}
    missing = expected_keys - set(result.keys())
    assert not missing, f"explain_score не вернул ключи: {missing}"


def test_explain_factors(features_df: pd.DataFrame, top_applicant_id: str) -> None:
    """В словаре factors ровно 5 ключей (по числу факторов скоринга)."""
    result = explain_score(top_applicant_id, features_df)
    n = len(result["factors"])
    assert n == 5, f"Ожидалось 5 факторов, получено {n}: {list(result['factors'].keys())}"


def test_explain_factor_keys(features_df: pd.DataFrame, top_applicant_id: str) -> None:
    """Каждый фактор содержит обязательные поля: raw_value, normalized, weight, contribution."""
    result = explain_score(top_applicant_id, features_df)
    required_fields = {"raw_value", "normalized", "weight", "contribution"}
    for factor_name, factor_data in result["factors"].items():
        missing = required_fields - set(factor_data.keys())
        assert not missing, (
            f"Фактор '{factor_name}' не содержит поля: {missing}"
        )


def test_explain_contributions_sum(features_df: pd.DataFrame, top_applicant_id: str) -> None:
    """Сумма contribution по всем факторам ≈ final_score (погрешность < 1e-4)."""
    result = explain_score(top_applicant_id, features_df)
    total = sum(v["contribution"] for v in result["factors"].values())
    diff  = abs(total - result["final_score"])
    assert diff < 1e-4, (
        f"Сумма вкладов ({total:.6f}) ≠ final_score ({result['final_score']:.6f}), "
        f"разница: {diff:.6f}"
    )


def test_explain_not_found(features_df: pd.DataFrame) -> None:
    """explain_score() бросает ValueError для несуществующего applicant_id."""
    with pytest.raises(ValueError, match="не найден"):
        explain_score("NONEXISTENT_ID_XYZ", features_df)


def test_explanation_ru_nonempty(features_df: pd.DataFrame, top_applicant_id: str) -> None:
    """explanation_ru — непустая строка на русском языке."""
    result = explain_score(top_applicant_id, features_df)
    text = result["explanation_ru"]
    assert isinstance(text, str) and len(text) > 10, (
        f"explanation_ru слишком короткий: '{text}'"
    )
