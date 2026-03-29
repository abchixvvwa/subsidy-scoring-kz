"""
Subsidy Scoring System — Demo Script
=====================================
Хакатон Decentrathon 5.0 | Кейс 2: Скоринг сельхозпроизводителей Казахстана

Запуск:
    python demo.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Проверка зависимостей
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    print("❌ Пакет pandas не установлен. Выполните: pip install -r requirements.txt")
    sys.exit(1)

try:
    from tabulate import tabulate
except ImportError:
    print("❌ Пакет tabulate не установлен. Выполните: pip install tabulate")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"

# Добавляем корень в sys.path для импорта src.*
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.scoring import compute_scores, get_shortlist, explain_score  # noqa: E402


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 65) -> str:
    return char * width


def _header(title: str) -> None:
    print()
    print(_sep("═"))
    print(f"  {title}")
    print(_sep("═"))


def _section(title: str) -> None:
    print()
    print(_sep())
    print(f"  {title}")
    print(_sep())


def _fmt_millions(val: float) -> str:
    """Форматирует сумму в миллионах тенге с 1 знаком после запятой."""
    return f"{val / 1_000_000:.1f}"


def _factor_bar(contribution: float, width: int = 30) -> str:
    """ASCII-прогресс бар для вклада фактора."""
    filled = int(contribution * width / 0.30)  # нормируем к макс весу 0.30
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:

    # ── Заголовок ───────────────────────────────────────────────────────────
    _header("SUBSIDY SCORING SYSTEM — ДЕМО")
    print("  Хакатон Decentrathon 5.0 | Кейс 2: Скоринг сельхозпроизводителей")

    # ── Загрузка данных ─────────────────────────────────────────────────────
    if not FEATURES_PATH.exists():
        print()
        print("❌ Файл признаков не найден:")
        print(f"   {FEATURES_PATH}")
        print()
        print("   Сначала запустите:")
        print("   python src/preprocessing.py")
        sys.exit(1)

    features = pd.read_csv(FEATURES_PATH)
    print(f"\n  ✓ Загружено заявителей: {len(features)}")

    # ── Статистика датасета ─────────────────────────────────────────────────
    _section("СТАТИСТИКА ДАТАСЕТА")

    print(f"  Всего заявителей:          {len(features)}")

    avg_rate = features["approval_rate"].mean()
    print(f"  Средний approval_rate:     {avg_rate * 100:.1f}%")

    min_sum = features["total_amount_received"].min() / 1_000_000
    max_sum = features["total_amount_received"].max() / 1_000_000
    print(f"  Диапазон субсидий:         {min_sum:.1f} млн — {max_sum:.0f} млн тг")

    print()
    print("  Топ-3 направления по числу заявителей:")
    top_dirs = (
        features["primary_direction"]
        .value_counts()
        .head(3)
        .reset_index()
    )
    top_dirs.columns = ["Направление", "Заявителей"]
    for _, r in top_dirs.iterrows():
        print(f"    • {r['Направление']}: {r['Заявителей']}")

    # ── Расчёт скоров ───────────────────────────────────────────────────────
    print()
    print("  Рассчитываю composite scores...", end=" ", flush=True)
    scored = compute_scores(features)
    print("✓")

    # ── Топ-10 шортлист ─────────────────────────────────────────────────────
    _section("ТОП-10 ЗАЯВИТЕЛЕЙ")

    sl = get_shortlist(features, top_n=10, scored_df=scored)

    table_rows = []
    for _, row in sl.iterrows():
        table_rows.append([
            int(row["rank"]),
            str(row["applicant_id"]),
            f"{row['final_score']:.4f}",
            f"{row['approval_rate'] * 100:.1f}%",
            f"{_fmt_millions(row['total_amount_received'])} млн",
            str(row["primary_direction"])[:35],
            str(row["oblast"])[:20],
        ])

    headers = ["Ранг", "ID заявителя", "Скор", "Одобрений%",
               "Сумма (млн тг)", "Направление", "Область"]

    print()
    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline",
                   colalign=("right", "left", "right", "right", "right", "left", "left")))

    # ── Объяснение #1 ───────────────────────────────────────────────────────
    top1_id = str(sl.iloc[0]["applicant_id"])
    _explain_block(top1_id, features, rank_label=1)

    # ── Объяснение #10 ──────────────────────────────────────────────────────
    top10_id = str(sl.iloc[9]["applicant_id"])
    _explain_block(top10_id, features, rank_label=10)

    # ── Финал ───────────────────────────────────────────────────────────────
    print()
    print(_sep("═"))
    print("  ✓ СИСТЕМА ГОТОВА.")
    print("  ✓ Запуск API:   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload")
    print("  ✓ Swagger UI:   http://localhost:8000/docs")
    print("  ✓ ReDoc:        http://localhost:8000/redoc")
    print(_sep("═"))
    print()


def _explain_block(applicant_id: str, features: pd.DataFrame, rank_label: int) -> None:
    """Выводит детальное объяснение скора для одного заявителя."""
    _section(f"ОБЪЯСНЕНИЕ СКОРА ДЛЯ ЗАЯВИТЕЛЯ #{rank_label}  (ID: {applicant_id})")

    exp = explain_score(applicant_id, features)

    print(f"  Итоговый скор : {exp['final_score']:.4f}")
    print(f"  Место в рейтинге: #{exp['rank']}")
    print(f"  Ключевой фактор : {exp['factors'][exp['top_factor']]['label']}")
    print()
    print("  Разбивка по факторам:")
    print()

    factor_rows = []
    for factor, info in sorted(
        exp["factors"].items(),
        key=lambda x: x[1]["contribution"],
        reverse=True,
    ):
        bar = _factor_bar(info["contribution"])
        factor_rows.append([
            info["label"],
            f"{info['raw_value']:.3g}",
            f"{info['normalized']:.3f}",
            f"×{info['weight']:.2f}",
            f"{info['contribution']:.4f}",
            bar,
        ])

    factor_headers = ["Фактор", "Исх. знач.", "Норм.", "Вес", "Вклад", ""]
    print(tabulate(factor_rows, headers=factor_headers,
                   tablefmt="simple", colalign=("left","right","right","right","right","left")))

    print()
    print("  Пояснение:")
    for line in exp["explanation_ru"].split("\n"):
        print(f"  {line}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
