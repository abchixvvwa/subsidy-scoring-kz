"""
Subsidy Scoring System — Conftest
==================================
Общие фикстуры pytest для всех тестов.

Фикстуры:
    features_df — загружает data/processed/features.csv один раз на сессию.
    scored_df   — рассчитанный датафрейм со скорами (один раз на сессию).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Добавляем корень репозитория в sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FEATURES_PATH = _ROOT / "data" / "processed" / "features.csv"


@pytest.fixture(scope="session")
def features_df() -> pd.DataFrame:
    """
    Загружает data/processed/features.csv один раз на всю тестовую сессию.
    Пропускает всю сессию если файл не найден.
    """
    if not FEATURES_PATH.exists():
        pytest.skip(
            f"features.csv не найден: {FEATURES_PATH}. "
            "Запустите: python src/preprocessing.py"
        )
    df = pd.read_csv(FEATURES_PATH)
    return df


@pytest.fixture(scope="session")
def scored_df(features_df: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает scored датафрейм один раз на всю тестовую сессию."""
    from src.scoring import compute_scores
    return compute_scores(features_df)
