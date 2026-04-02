import http.server
import json
import urllib.parse
import sys
import math
from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.scoring import compute_scores, explain_score

ML_RESULTS_PATH = _BASE_DIR / "data" / "processed" / "ml_results.csv"
FEATURES_PATH = _BASE_DIR / "data" / "processed" / "features.csv"


def _normalize_one_applicant_id(aid: str) -> str:
    """Совпадение с `_normalize_applicant_id_series` для запросов /api/explain и /api/applicant."""
    n = pd.to_numeric(str(aid).strip(), errors="coerce")
    if pd.isna(n):
        return str(aid).strip()
    return str(int(n))


def _normalize_applicant_id_series(s: pd.Series) -> pd.Series:
    """
    Единый ключ заявителя: в features.csv часто строка с ведущими нулями (0010010025),
    в ml_results — число (10010025). Приводим к одной строке цифр без ведущих нулей.
    """
    num = pd.to_numeric(s, errors="coerce")
    return num.astype("Int64").astype(str)


def _load_merged_rows() -> tuple[list[dict], pd.DataFrame | None]:
    """
    Возвращает список dict-строк для фронта и (опционально) features_df для /api/explain.

    Мы объединяем:
      - ml_results.csv (ml_probability, anomaly_reason, etc)
      - composite final_score/rank из compute_scores(features.csv)
    """
    if not ML_RESULTS_PATH.exists():
        return [], None

    ml_df = pd.read_csv(ML_RESULTS_PATH)
    features_df = None

    # composite (final_score) добавляем, если есть features.csv
    if FEATURES_PATH.exists():
        features_df = pd.read_csv(FEATURES_PATH)
        features_df = features_df.copy()
        features_df["applicant_id"] = _normalize_applicant_id_series(features_df["applicant_id"])
        scored_df = compute_scores(features_df)
        ml_df = ml_df.copy()
        ml_df["applicant_id"] = _normalize_applicant_id_series(ml_df["applicant_id"])
        scored_df["applicant_id"] = _normalize_applicant_id_series(scored_df["applicant_id"])
        ml_df = ml_df.merge(
            scored_df[["applicant_id", "final_score", "rank"]],
            on="applicant_id",
            how="left",
        )
    else:
        # фронт сможет показать ML-score, но composite будет N/A
        ml_df["final_score"] = None
        ml_df["rank"] = None

    rows = ml_df.to_dict(orient="records")

    # В данных могут встречаться NaN (float) — браузерный JSON.parse
    # часто не принимает литерал NaN как валидный JSON.
    def _sanitize_value(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    rows = [{k: _sanitize_value(v) for k, v in row.items()} for row in rows]
    return rows, features_df


ALL_ROWS, _FEATURES_DF = _load_merged_rows()

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send_json(self, data, status=200):
        # numpy / pandas скаляры и прочие типы — в валидный JSON
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/debug":
            rows = ALL_ROWS
            if rows:
                self.send_json({
                    "total": len(rows),
                    "columns": list(rows[0].keys()),
                    "row0": rows[0],
                    "row1": rows[1] if len(rows)>1 else {}
                })
            else:
                self.send_json({"error": "empty"})
        elif path == "/api/all":
            self.send_json(ALL_ROWS)
        elif path == "/api/columns":
            if ALL_ROWS:
                self.send_json({"columns": list(ALL_ROWS[0].keys()), "sample": ALL_ROWS[0]})
            else:
                self.send_json({"error": "no data"})
        elif path.startswith("/api/applicant/"):
            aid = _normalize_one_applicant_id(urllib.parse.unquote(path.split("/")[-1]))
            found = [r for r in ALL_ROWS if str(r.get("applicant_id")) == aid]
            self.send_json(found[0] if found else {}, 404 if not found else 200)
        elif path.startswith("/api/explain/"):
            aid = _normalize_one_applicant_id(urllib.parse.unquote(path.split("/")[-1]))
            if _FEATURES_DF is None:
                self.send_json({"error": "features.csv not loaded"}, status=503)
                return
            try:
                exp = explain_score(aid, _FEATURES_DF)
                # JSON: round-trip friendly (plain dict)
                self.send_json(exp)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=404)
        elif path == "/":
            html = (_BASE_DIR / "app" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_json({"error":"not found"}, 404)

if __name__ == "__main__":
    import webbrowser, threading
    server = http.server.HTTPServer(("localhost", 8502), Handler)
    threading.Timer(0.5, lambda: webbrowser.open("http://localhost:8502")).start()
    print("Открываю http://localhost:8502")
    server.serve_forever()
