import sys
import traceback
sys.path.insert(0, '.')

def section(title):
    print(f"\n{'='*50}")
    print(f"=== {title} ===")
    print('='*50)

# ─── PREPROCESSING ───────────────────────────────────
section("PREPROCESSING")
try:
    import pandas as pd
    from src.preprocessing import load_raw_data, build_applicant_features

    df = load_raw_data('data/raw/Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx')
    features = build_applicant_features(df)
    print(f'Заявителей: {len(features)}')
    print(f'Колонки: {list(features.columns)}')
    print(f'NaN: {features.isnull().sum().sum()}')
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()

# ─── SCORING ─────────────────────────────────────────
section("SCORING")
try:
    import pandas as pd
    from src.scoring import compute_scores, get_shortlist, explain_score

    df = pd.read_csv('data/processed/features.csv')
    scored = compute_scores(df)
    print(f'Score range: {scored.final_score.min():.3f} - {scored.final_score.max():.3f}')
    print(f'NaN в score: {scored.final_score.isnull().sum()}')
    sl = get_shortlist(df, 5)
    print(sl[['applicant_id', 'final_score', 'approval_rate']].to_string())
    top_id = sl.iloc[0]['applicant_id']
    exp = explain_score(top_id, df)
    print(f'Факторов в explain: {len(exp["factors"])}')
    print(f'explanation_ru: {exp["explanation_ru"][:80]}')
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()

# ─── ML MODEL ────────────────────────────────────────
section("ML MODEL")
try:
    import pandas as pd

    df = pd.read_csv('data/processed/ml_results.csv')
    print(f'Колонок: {len(df.columns)}')
    print(f'Колонки: {list(df.columns)}')
    print(f'ml_probability:\n{df.ml_probability.describe()}')
    print(f'Аномалий: {df.is_anomaly.sum()}')
    print(f'Кластеров: {df.cluster_label.nunique()} - {df.cluster_label.unique()}')
    print(f'NaN в ml_probability: {df.ml_probability.isnull().sum()}')
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()

# ─── API ─────────────────────────────────────────────
section("API")
try:
    from src.api import app
    routes = [r.path for r in app.routes if hasattr(r, 'path')]
    print(f'Маршруты: {sorted(routes)}')
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()

# ─── TESTS ───────────────────────────────────────────
section("TESTS")
import subprocess
result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
    capture_output=True, text=True, cwd='.'
)
output = (result.stdout + result.stderr).strip().split('\n')
# последние 40 строк
print('\n'.join(output[-40:]))
