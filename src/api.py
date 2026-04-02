"""
Subsidy Scoring System — FastAPI Application
============================================
REST API для системы скоринга сельхозпроизводителей Казахстана.

Запуск:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Документация (Swagger):
    http://localhost:8000/docs

Документация (ReDoc):
    http://localhost:8000/redoc
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Путь к данным
# ---------------------------------------------------------------------------
# Поддерживаем запуск как из корня репозитория, так и из папки src/
_BASE_DIR = Path(__file__).resolve().parent.parent  # всегда корень проекта
FEATURES_PATH   = _BASE_DIR / "data" / "processed" / "features.csv"
ML_RESULTS_PATH = _BASE_DIR / "data" / "processed" / "ml_results.csv"

# ---------------------------------------------------------------------------
# Импорт модулей проекта
# ---------------------------------------------------------------------------
# Добавляем корень в sys.path, чтобы импорт работал при любом CWD
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.scoring import compute_scores, get_shortlist, explain_score  # noqa: E402
from src.ml_model import run_full_ml_pipeline                         # noqa: E402

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="КазнаЛинза API",
    description=(
        "API системы merit-based скоринга сельхозпроизводителей Казахстана "
        "для получения субсидий. Хакатон Decentrathon 5.0, Кейс №2."
    ),
    version="1.0.0",
    contact={
        "name": "Decentrathon 5.0 Team",
        "url": "https://decentrathon.kz",
    },
    license_info={"name": "MIT"},
)

# CORS — разрешаем все origins (для хакатон-демо)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Глобальное состояние — данные загружаются один раз при старте
# ---------------------------------------------------------------------------

class _AppState:
    features_df: Optional[pd.DataFrame] = None
    scored_df:   Optional[pd.DataFrame] = None
    ml_df:       Optional[pd.DataFrame] = None   # ml_results.csv (includes cluster/anomaly cols)
    loaded_at:   Optional[datetime]     = None
    load_error:  Optional[str]          = None


_state = _AppState()


def _load_data() -> None:
    """
    Загружает данные при старте приложения.

    Логика:
    1. Если ml_results.csv существует — загружает его (уже содержит ML-колонки).
    2. Иначе — загружает features.csv и запускает run_full_ml_pipeline().
    3. Дополнительно рассчитывает scored_df (composite score) по features_df.
    """
    if not FEATURES_PATH.exists():
        _state.load_error = (
            f"Файл признаков не найден: {FEATURES_PATH}. "
            "Сначала запустите: python src/preprocessing.py"
        )
        print(f"[WARN] {_state.load_error}")
        return

    try:
        _state.features_df = pd.read_csv(FEATURES_PATH)
        _state.scored_df   = compute_scores(_state.features_df)

        if ML_RESULTS_PATH.exists():
            _state.ml_df = pd.read_csv(ML_RESULTS_PATH)
            print(
                f"[INFO] ML-результаты загружены: {len(_state.ml_df)} заявителей "
                f"из {ML_RESULTS_PATH}"
            )
        else:
            print("[INFO] ml_results.csv не найден — запускаю ML-пайплайн...")
            _state.ml_df = run_full_ml_pipeline(_state.features_df)
            print("[INFO] ML-пайплайн завершён.")

        _state.loaded_at = datetime.now(timezone.utc)
        print(
            f"[INFO] Загружено заявителей: {len(_state.features_df)} "
            f"из {FEATURES_PATH}"
        )
    except Exception as exc:
        _state.load_error = str(exc)
        print(f"[ERROR] Ошибка загрузки данных: {exc}")


@app.on_event("startup")
async def startup_event() -> None:
    _load_data()


def _require_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (features_df, scored_df); бросает 503 если данные не загружены."""
    if _state.features_df is None or _state.scored_df is None:
        raise HTTPException(
            status_code=503,
            detail=(
                _state.load_error
                or "Данные ещё не загружены. Повторите запрос через несколько секунд."
            ),
        )
    return _state.features_df, _state.scored_df


def _require_ml() -> pd.DataFrame:
    """Возвращает ml_df; бросает 503 если ML-данные не загружены."""
    if _state.ml_df is None:
        raise HTTPException(
            status_code=503,
            detail=(
                _state.load_error
                or "ML-результаты ещё не готовы. Повторите запрос через несколько секунд."
            ),
        )
    return _state.ml_df


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    applicants_loaded: int = Field(..., example=457)
    data_file: str = Field(..., example="data/processed/features.csv")
    ml_results_file: str = Field(..., example="data/processed/ml_results.csv")
    loaded_at: Optional[str] = Field(None, example="2025-03-19T10:00:00Z")
    error: Optional[str] = Field(None, example=None)


class ShortlistApplicant(BaseModel):
    rank: int = Field(..., example=1)
    applicant_id: str = Field(..., example="0130010025")
    final_score: float = Field(..., example=0.8412)
    approval_rate: float = Field(..., example=0.905)
    total_amount_received: float = Field(..., example=156750000.0)
    primary_direction: str = Field(..., example="Субсидирование в скотоводстве")
    oblast: str = Field(..., example="область Абай")
    total_applications: int = Field(..., example=42)


class ShortlistResponse(BaseModel):
    total_applicants: int = Field(..., example=457)
    shortlist_size: int = Field(..., example=20)
    generated_at: str = Field(..., example="2025-03-19T10:00:00Z")
    applicants: list[ShortlistApplicant]


class ApplicantProfile(BaseModel):
    applicant_id: str
    total_applications: int
    approved_count: int
    rejected_count: int
    withdrawn_count: int
    approval_rate: float
    rejection_rate: float
    total_amount_received: float
    avg_amount: float
    max_amount: float
    unique_directions: int
    unique_subsidy_types: int
    last_activity_days: Optional[int]
    first_application_year: Optional[int]
    oblast: str
    primary_direction: str
    # ML-поля (None если ml_results.csv не содержит этого заявителя)
    cluster_label:   Optional[str]   = Field(None, example="Крупные эффективные")
    is_anomaly:      Optional[bool]  = Field(None, example=False)
    anomaly_reason:  Optional[str]   = Field(None, example="")


class FactorDetail(BaseModel):
    label: str = Field(..., example="Доля одобренных заявок")
    raw_value: float = Field(..., example=0.905)
    normalized: float = Field(..., example=0.87)
    weight: float = Field(..., example=0.30)
    contribution: float = Field(..., example=0.261)


class ScoreExplanation(BaseModel):
    applicant_id: str
    final_score: float
    rank: int
    factors: dict[str, FactorDetail]
    top_factor: str
    explanation_ru: str


class ApplicantDetailResponse(BaseModel):
    profile: ApplicantProfile
    score: ScoreExplanation


# ── ML response models ────────────────────────────────────────────────────────

class ClusterInfo(BaseModel):
    cluster_id:        int   = Field(..., example=0)
    label:             str   = Field(..., example="Крупные эффективные")
    count:             int   = Field(..., example=45)
    avg_score:         float = Field(..., example=0.72)
    avg_approval_rate: float = Field(..., example=0.91)
    avg_amount:        float = Field(..., example=500_000_000.0)


class ClustersResponse(BaseModel):
    total_clusters: int
    clusters: list[ClusterInfo]


class AnomalyItem(BaseModel):
    applicant_id:      str   = Field(..., example="0130010025")
    anomaly_score:     float = Field(..., example=-0.15)
    anomaly_reason:    str   = Field(..., example="Аномалия: ...")
    oblast:            str   = Field(..., example="область Абай")
    primary_direction: str   = Field(..., example="Субсидирование в скотоводстве")


class AnomaliesResponse(BaseModel):
    total_anomalies: int
    anomalies: list[AnomalyItem]


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка работоспособности сервиса",
    tags=["System"],
)
async def health() -> HealthResponse:
    """
    Возвращает статус сервиса и количество загруженных заявителей.
    """
    n = len(_state.features_df) if _state.features_df is not None else 0
    loaded_at_str = (
        _state.loaded_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        if _state.loaded_at else None
    )
    return HealthResponse(
        status="ok" if _state.features_df is not None else "degraded",
        applicants_loaded=n,
        data_file=str(FEATURES_PATH.relative_to(_BASE_DIR)),
        ml_results_file=str(ML_RESULTS_PATH.relative_to(_BASE_DIR)),
        loaded_at=loaded_at_str,
        error=_state.load_error,
    )


@app.get(
    "/shortlist",
    response_model=ShortlistResponse,
    summary="Топ-N заявителей по composite score",
    tags=["Scoring"],
)
async def shortlist(
    top_n: int = Query(
        default=20,
        ge=1,
        le=457,
        description="Количество лучших заявителей в шортлисте (1–457)",
    )
) -> ShortlistResponse:
    """
    Возвращает топ-N заявителей, ранжированных по взвешенному composite score.

    Факторы и веса:
    - **approval_rate** × 0.30 — надёжность (доля одобренных заявок)
    - **total_amount_received** × 0.25 — масштаб хозяйства
    - **total_applications** × 0.20 — опыт работы с системой субсидий
    - **unique_directions** × 0.15 — диверсификация производства
    - **recency_score** × 0.10 — актуальность (свежесть последней заявки)
    """
    features_df, scored_df = _require_data()

    sl = get_shortlist(features_df, top_n=top_n, scored_df=scored_df)

    applicants = []
    for _, row in sl.iterrows():
        applicants.append(
            ShortlistApplicant(
                rank=int(row["rank"]),
                applicant_id=str(row["applicant_id"]),
                final_score=round(float(row["final_score"]), 6),
                approval_rate=round(float(row.get("approval_rate", 0)), 4),
                total_amount_received=round(float(row.get("total_amount_received", 0)), 2),
                primary_direction=str(row.get("primary_direction", "")),
                oblast=str(row.get("oblast", "")),
                total_applications=int(row.get("total_applications", 0)),
            )
        )

    return ShortlistResponse(
        total_applicants=len(features_df),
        shortlist_size=len(applicants),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        applicants=applicants,
    )


@app.get(
    "/applicant/{applicant_id}",
    response_model=ApplicantDetailResponse,
    summary="Полный профиль заявителя + скор",
    tags=["Applicants"],
)
async def get_applicant(applicant_id: str) -> ApplicantDetailResponse:
    """
    Возвращает профиль заявителя и детальное объяснение его скора.
    Дополнительно включает ML-поля: cluster_label, is_anomaly, anomaly_reason.

    - **applicant_id** — первые 10 символов номера заявки (app_num)
    """
    features_df, _ = _require_data()

    mask = features_df["applicant_id"].astype(str) == applicant_id
    if not mask.any():
        raise HTTPException(
            status_code=404,
            detail=f"Заявитель не найден: '{applicant_id}'",
        )

    row = features_df[mask].iloc[0]

    # ML-поля из ml_df (если доступны)
    cluster_label  = None
    is_anomaly     = None
    anomaly_reason = None
    if _state.ml_df is not None:
        ml_mask = _state.ml_df["applicant_id"].astype(str) == applicant_id
        if ml_mask.any():
            ml_row         = _state.ml_df[ml_mask].iloc[0]
            cluster_label  = str(ml_row.get("cluster_label", "")) or None
            is_anomaly     = bool(ml_row.get("is_anomaly", False))
            anomaly_reason = str(ml_row.get("anomaly_reason", "")) or None

    # Профиль
    profile = ApplicantProfile(
        applicant_id=str(row["applicant_id"]),
        total_applications=int(row.get("total_applications", 0)),
        approved_count=int(row.get("approved_count", 0)),
        rejected_count=int(row.get("rejected_count", 0)),
        withdrawn_count=int(row.get("withdrawn_count", 0)),
        approval_rate=round(float(row.get("approval_rate", 0)), 4),
        rejection_rate=round(float(row.get("rejection_rate", 0)), 4),
        total_amount_received=round(float(row.get("total_amount_received", 0)), 2),
        avg_amount=round(float(row.get("avg_amount", 0)), 2),
        max_amount=round(float(row.get("max_amount", 0)), 2),
        unique_directions=int(row.get("unique_directions", 0)),
        unique_subsidy_types=int(row.get("unique_subsidy_types", 0)),
        last_activity_days=_safe_int(row.get("last_activity_days")),
        first_application_year=_safe_int(row.get("first_application_year")),
        oblast=str(row.get("oblast", "")),
        primary_direction=str(row.get("primary_direction", "")),
        cluster_label=cluster_label,
        is_anomaly=is_anomaly,
        anomaly_reason=anomaly_reason,
    )

    # Скор + объяснение
    score = _build_score_explanation(applicant_id, features_df)

    return ApplicantDetailResponse(profile=profile, score=score)


@app.get(
    "/explain/{applicant_id}",
    response_model=ScoreExplanation,
    summary="Объяснение скора заявителя",
    tags=["Scoring"],
)
async def explain(applicant_id: str) -> ScoreExplanation:
    """
    Возвращает детальное объяснение composite score для конкретного заявителя:
    вклад каждого из 5 факторов, ключевой фактор и читаемое объяснение на русском.
    """
    features_df, _ = _require_data()

    mask = features_df["applicant_id"].astype(str) == applicant_id
    if not mask.any():
        raise HTTPException(
            status_code=404,
            detail=f"Заявитель не найден: '{applicant_id}'",
        )

    return _build_score_explanation(applicant_id, features_df)


@app.get(
    "/clusters",
    response_model=ClustersResponse,
    summary="Статистика по кластерам заявителей",
    tags=["ML"],
)
async def get_clusters() -> ClustersResponse:
    """
    Возвращает агрегированную статистику по каждому кластеру K-Means:
    количество заявителей, средние значения approval_rate, суммы субсидий
    и composite score.
    """
    ml_df      = _require_ml()
    _, scored  = _require_data()

    # Присоединяем final_score из scored_df для avg_score
    score_col = scored[["applicant_id", "final_score"]].copy()
    merged    = ml_df.merge(
        score_col, on="applicant_id", how="left", suffixes=("", "_scored")
    )

    clusters: list[ClusterInfo] = []
    for (cid, label), grp in merged.groupby(["cluster", "cluster_label"]):
        clusters.append(
            ClusterInfo(
                cluster_id=int(cid),
                label=str(label),
                count=len(grp),
                avg_score=round(float(grp["final_score"].mean(skipna=True)), 4)
                          if "final_score" in grp.columns else 0.0,
                avg_approval_rate=round(float(grp["approval_rate"].mean()), 4),
                avg_amount=round(float(grp["total_amount_received"].mean()), 2),
            )
        )

    # Сортируем по avg_score убывающим (лучший кластер первым)
    clusters.sort(key=lambda c: c.avg_score, reverse=True)

    return ClustersResponse(total_clusters=len(clusters), clusters=clusters)


@app.get(
    "/anomalies",
    response_model=AnomaliesResponse,
    summary="Список подозрительных заявителей (аномалии)",
    tags=["ML"],
)
async def get_anomalies() -> AnomaliesResponse:
    """
    Возвращает заявителей, помеченных Isolation Forest как аномальные.
    Отсортированы по anomaly_score по возрастанию (наиболее подозрительные — первые).
    """
    ml_df = _require_ml()

    anom_df = (
        ml_df[ml_df["is_anomaly"]]
        .sort_values("anomaly_score")
        .reset_index(drop=True)
    )

    anomalies: list[AnomalyItem] = []
    for _, row in anom_df.iterrows():
        anomalies.append(
            AnomalyItem(
                applicant_id=str(row["applicant_id"]),
                anomaly_score=round(float(row["anomaly_score"]), 6),
                anomaly_reason=str(row.get("anomaly_reason", "")),
                oblast=str(row.get("oblast", "")),
                primary_direction=str(row.get("primary_direction", "")),
            )
        )

    return AnomaliesResponse(total_anomalies=len(anomalies), anomalies=anomalies)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _build_score_explanation(
    applicant_id: str, features_df: pd.DataFrame
) -> ScoreExplanation:
    """Вызывает explain_score и оборачивает результат в Pydantic-модель."""
    raw: dict[str, Any] = explain_score(applicant_id, features_df)

    factors_model = {
        key: FactorDetail(
            label=v["label"],
            raw_value=v["raw_value"],
            normalized=v["normalized"],
            weight=v["weight"],
            contribution=v["contribution"],
        )
        for key, v in raw["factors"].items()
    }

    return ScoreExplanation(
        applicant_id=raw["applicant_id"],
        final_score=raw["final_score"],
        rank=raw["rank"],
        factors=factors_model,
        top_factor=raw["top_factor"],
        explanation_ru=raw["explanation_ru"],
    )


def _safe_int(val: Any) -> Optional[int]:
    """Конвертирует значение в int, возвращает None при ошибке (NaN и пр.)."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
