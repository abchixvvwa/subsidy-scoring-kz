"""
Subsidy Scoring System — Data Preprocessing
============================================
Загрузка и очистка датасета выданных субсидий Казахстана 2025 г.,
построение признаков (features) на уровне заявителя.

Запуск:
    python src/preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Дата "сегодня" для расчёта last_activity_days (фиксируем по данным)
REFERENCE_DATE = pd.Timestamp("2026-03-19")

# Колонки в исходном Excel после header=3 → наши имена
COLUMN_MAP = {
    "Unnamed: 0":  "num",
    "Unnamed: 1":  "date",
    "Unnamed: 4":  "oblast",
    "Unnamed: 5":  "akimat",
    "Unnamed: 6":  "app_num",
    "Unnamed: 7":  "direction",
    "Unnamed: 8":  "subsidy_name",
    "Unnamed: 9":  "status",
    "Unnamed: 10": "normativ",
    "Unnamed: 11": "amount",
    "Unnamed: 12": "district",
}

# Статусы, считающиеся одобрением
APPROVED_STATUSES = {"Исполнена", "Одобрена"}

# Статусы отклонения / отзыва
REJECTED_STATUSES  = {"Отклонена"}
WITHDRAWN_STATUSES = {"Отозвано", "Отозвана"}


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Читает Excel-файл с выданными субсидиями.

    Parameters
    ----------
    filepath : str
        Путь к .xlsx файлу (данные начинаются с 4-й строки, header=3).

    Returns
    -------
    pd.DataFrame
        Очищенный датафрейм с типизированными колонками.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {filepath}\n"
            "Положите Excel-файл в data/raw/ и укажите правильный путь."
        )

    print(f"[load_raw_data] Читаю файл: {filepath}")
    raw = pd.read_excel(filepath, header=3)

    # --- Переименование колонок -------------------------------------------
    # Берём только те, которые есть в файле (защита от версий Excel)
    present_map = {k: v for k, v in COLUMN_MAP.items() if k in raw.columns}
    missing     = set(COLUMN_MAP.keys()) - set(present_map.keys())
    if missing:
        print(f"  [WARN] Колонки не найдены и пропущены: {missing}")

    df = raw.rename(columns=present_map)

    # Оставляем только нужные колонки (те, которые удалось замапить)
    keep = [v for v in COLUMN_MAP.values() if v in df.columns]
    df = df[keep].copy()

    # --- Фильтрация служебной строки "№ п/п" --------------------------------
    # Первая строка данных содержит текстовые заголовки
    before = len(df)
    df = df[df["num"] != "№ п/п"].dropna(subset=["num"])
    print(f"  Отфильтровано служебных строк: {before - len(df)}")

    # --- Типизация ----------------------------------------------------------

    # num → int (может быть строкой)
    df["num"] = pd.to_numeric(df["num"], errors="coerce")

    # date → datetime
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # amount и normativ → float
    df["amount"]  = pd.to_numeric(df["amount"].astype(str).str.replace(" ", "").str.replace(",", "."),
                                  errors="coerce")
    df["normativ"] = pd.to_numeric(df["normativ"].astype(str).str.replace(" ", "").str.replace(",", "."),
                                   errors="coerce")

    # Строковые поля — убираем пробелы по краям
    for col in ["oblast", "akimat", "app_num", "direction", "subsidy_name", "status", "district"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # --- applicant_id = первые 10 символов app_num -------------------------
    df["applicant_id"] = df["app_num"].str[:10]

    print(f"  Строк после очистки : {len(df):,}")
    print(f"  Уникальных заявителей: {df['applicant_id'].nunique():,}")
    print(f"  Период: {df['date'].min().date()} — {df['date'].max().date()}")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# build_applicant_features
# ---------------------------------------------------------------------------

def build_applicant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует строки заявок до уровня заявителя (1 строка = 1 заявитель).

    Parameters
    ----------
    df : pd.DataFrame
        Чистый датафрейм, полученный из load_raw_data().

    Returns
    -------
    pd.DataFrame
        Датафрейм признаков; индекс = applicant_id.
    """
    print("[build_applicant_features] Строю признаки...")

    ref = REFERENCE_DATE

    # Вспомогательные булевы маски
    df = df.copy()
    df["is_approved"]  = df["status"].isin(APPROVED_STATUSES)
    df["is_rejected"]  = df["status"].isin(REJECTED_STATUSES)
    df["is_withdrawn"] = df["status"].isin(WITHDRAWN_STATUSES)

    # Сумма только по одобренным заявкам
    df["approved_amount"] = df["amount"].where(df["is_approved"], other=0.0)

    # -----------------------------------------------------------------------
    grp = df.groupby("applicant_id")

    features = pd.DataFrame()

    # Базовые счётчики
    features["total_applications"]    = grp["app_num"].count()
    features["approved_count"]        = grp["is_approved"].sum().astype(int)
    features["rejected_count"]        = grp["is_rejected"].sum().astype(int)
    features["withdrawn_count"]       = grp["is_withdrawn"].sum().astype(int)

    # Доли
    features["approval_rate"]         = features["approved_count"] / features["total_applications"]
    features["rejection_rate"]        = features["rejected_count"] / features["total_applications"]

    # Финансовые агрегаты
    features["total_amount_received"] = grp["approved_amount"].sum()
    features["avg_amount"]            = grp["amount"].mean()
    features["max_amount"]            = grp["amount"].max()

    # Разнообразие
    features["unique_directions"]     = grp["direction"].nunique()
    features["unique_subsidy_types"]  = grp["subsidy_name"].nunique()

    # Временны́е признаки
    features["last_activity_days"]    = grp["date"].max().apply(
        lambda d: (ref - d).days if pd.notnull(d) else np.nan
    ).astype("Int64")

    features["first_application_year"] = grp["date"].min().dt.year.astype("Int64")

    # Категориальные (режим — наиболее частое значение)
    features["oblast"]             = grp["oblast"].agg(lambda s: _mode(s))
    features["primary_direction"]  = grp["direction"].agg(lambda s: _mode(s))

    features = features.reset_index()  # applicant_id становится обычной колонкой

    # Типизация
    features["applicant_id"]           = features["applicant_id"].astype(str)
    features["total_applications"]     = features["total_applications"].astype(int)
    features["approved_count"]         = features["approved_count"].astype(int)
    features["rejected_count"]         = features["rejected_count"].astype(int)
    features["withdrawn_count"]        = features["withdrawn_count"].astype(int)
    features["approval_rate"]          = features["approval_rate"].round(4)
    features["rejection_rate"]         = features["rejection_rate"].round(4)
    features["total_amount_received"]  = features["total_amount_received"].round(2)
    features["avg_amount"]             = features["avg_amount"].round(2)
    features["max_amount"]             = features["max_amount"].round(2)

    print(f"  Признаков построено    : {len(features.columns)}")
    print(f"  Заявителей в датафрейме: {len(features):,}")

    return features


# ---------------------------------------------------------------------------
# save_features
# ---------------------------------------------------------------------------

def save_features(features_df: pd.DataFrame, output_path: str) -> None:
    """
    Сохраняет датафрейм признаков в CSV.

    Parameters
    ----------
    features_df : pd.DataFrame
        Датафрейм, полученный из build_applicant_features().
    output_path : str
        Путь к выходному CSV-файлу.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[save_features] Сохранено → {out.resolve()}")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _mode(series: pd.Series) -> str:
    """Возвращает наиболее частое значение в серии; при пустой — пустую строку."""
    vals = series.dropna()
    if vals.empty:
        return ""
    return vals.mode().iloc[0]


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RAW_PATH = "data/raw/Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx"
    OUT_PATH = "data/processed/features.csv"

    # 1. Загрузка и очистка
    df = load_raw_data(RAW_PATH)

    # Быстрый overview
    print("\n--- Распределение по статусам ---")
    print(df["status"].value_counts().to_string())

    print("\n--- Топ-5 направлений ---")
    print(df["direction"].value_counts().head().to_string())

    # 2. Построение признаков
    features = build_applicant_features(df)

    # 3. Сохранение
    save_features(features, OUT_PATH)

    # 4. Отчёт
    print(f"\n✅ Обработано заявителей: {len(features)}")
    print("\n--- Первые 3 строки признаков ---")
    print(features.head(3).to_string())

    print("\n--- Статистика числовых признаков ---")
    print(features.describe().round(2).to_string())
