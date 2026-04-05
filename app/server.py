import http.server
import json
import os
import re
import urllib.parse
import sys
import math
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.scoring import compute_scores, explain_score

ML_RESULTS_PATH = _BASE_DIR / "data" / "processed" / "ml_results.csv"
FEATURES_PATH = _BASE_DIR / "data" / "processed" / "features.csv"


def _normalize_one_applicant_id(aid: str) -> str:
    n = pd.to_numeric(str(aid).strip(), errors="coerce")
    if pd.isna(n):
        return str(aid).strip()
    return str(int(n))


def _normalize_applicant_id_series(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    return num.astype("Int64").astype(str)


def _load_merged_rows() -> tuple[list[dict], pd.DataFrame | None]:
    if not ML_RESULTS_PATH.exists():
        return [], None

    ml_df = pd.read_csv(ML_RESULTS_PATH)
    features_df = None

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
        ml_df["final_score"] = None
        ml_df["rank"] = None

    rows = ml_df.to_dict(orient="records")

    def _sanitize_value(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    rows = [{k: _sanitize_value(v) for k, v in row.items()} for row in rows]
    return rows, features_df


ALL_ROWS, _FEATURES_DF = _load_merged_rows()


def _load_merged_dataframe() -> pd.DataFrame:
    if not ML_RESULTS_PATH.exists():
        return pd.DataFrame()
    ml_df = pd.read_csv(ML_RESULTS_PATH)
    ml_df = ml_df.copy()
    ml_df["applicant_id"] = _normalize_applicant_id_series(ml_df["applicant_id"])
    if FEATURES_PATH.exists():
        features_df = pd.read_csv(FEATURES_PATH)
        features_df = features_df.copy()
        features_df["applicant_id"] = _normalize_applicant_id_series(features_df["applicant_id"])
        scored_df = compute_scores(features_df)
        scored_df["applicant_id"] = _normalize_applicant_id_series(scored_df["applicant_id"])
        ml_df = ml_df.merge(
            scored_df[["applicant_id", "final_score", "rank"]],
            on="applicant_id",
            how="left",
        )
    else:
        ml_df["final_score"] = None
        ml_df["rank"] = None
    return ml_df


def _anomaly_mask(s: pd.Series) -> pd.Series:
    def one(x):
        if x is True or x is False:
            return bool(x)
        t = str(x).strip().lower()
        return t in ("true", "1", "yes")

    return s.map(one)


def _sanitize_row_for_prompt(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def _extract_applicant_id_from_message(message: str, df: pd.DataFrame) -> str | None:
    if df.empty or "applicant_id" not in df.columns:
        return None
    valid = set(df["applicant_id"].astype(str))
    for m in re.finditer(r"\d{5,}", message):
        raw = m.group(0)
        cand = _normalize_one_applicant_id(raw)
        if cand in valid:
            return cand
    return None


def _compute_cluster_stats_table(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty or "cluster_label" not in df.columns:
        return None

    agg: dict = {}
    if "approval_rate" in df.columns:
        agg["approval_rate"] = "mean"
    if "total_amount_received" in df.columns:
        agg["total_amount_received"] = "mean"
    elif "total_amount" in df.columns:
        agg["total_amount"] = "mean"
    if "total_applications" in df.columns:
        agg["total_applications"] = "mean"
    if "ml_probability" in df.columns:
        agg["ml_probability"] = "mean"
    if not agg:
        return None

    g = df.groupby("cluster_label", dropna=False).agg(agg).round(3)
    if "total_amount_received" in g.columns:
        g = g.rename(columns={"total_amount_received": "total_amount"})
    return g


def _build_chat_system_prompt(df: pd.DataFrame, user_message: str = "") -> str:
    if df.empty:
        return (
            "Ты AI-аналитик системы КазнаЛинза — инструмента для государственных служащих "
            "Министерства сельского хозяйства Казахстана.\n"
            "Сейчас нет загруженных данных (файлы data/processed/ml_results.csv отсутствуют или пусты). "
            "Сообщи пользователю, что нужно выполнить пайплайн preprocessing + ml_model. "
            "Отвечай кратко на русском языке."
        )

    n = len(df)
    ml_col = "ml_probability" if "ml_probability" in df.columns else None

    # Топ-5 надёжных — читаемый текст
    top5 = "—"
    if ml_col and df[ml_col].notna().any():
        t5 = df.nlargest(5, ml_col)
        lines = []
        for _, row in t5.iterrows():
            idx = round(float(row.get(ml_col, 0)) * 100)
            cl = row.get("cluster_label", "—")
            ob = row.get("oblast", "—")
            lines.append(f"  - Заявитель {row['applicant_id']}: индекс {idx}/100, группа «{cl}», регион {ob}")
        top5 = "\n".join(lines)

    am = _anomaly_mask(df["is_anomaly"]) if "is_anomaly" in df.columns else pd.Series([False] * len(df))
    anomaly_count = int(am.sum())

    # Топ аномалии — читаемый текст
    top_anomalies = "нет"
    if anomaly_count and "anomaly_score" in df.columns:
        sub = df.loc[am].nsmallest(min(5, anomaly_count), "anomaly_score")
        lines = []
        for _, row in sub.iterrows():
            idx = round(float(row.get(ml_col, 0)) * 100) if ml_col else "—"
            reason = str(row.get("anomaly_reason", "") or "").strip() or "не указана"
            lines.append(f"  - Заявитель {row['applicant_id']}: индекс {idx}/100, причина: {reason}")
        top_anomalies = "\n".join(lines)

    g_cluster = _compute_cluster_stats_table(df)
    cluster_stats_text = "—"
    cluster_stats_json = "{}"
    if g_cluster is not None and not g_cluster.empty:
        cluster_stats_text = g_cluster.to_string()
        cluster_stats_json = json.dumps(
            g_cluster.to_dict("index"),
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    applicant_extra = ""
    found_id = _extract_applicant_id_from_message(user_message, df)
    if found_id is not None and "applicant_id" in df.columns:
        sub = df[df["applicant_id"].astype(str) == str(found_id)]
        if len(sub) == 0:
            sub = df[df["applicant_id"] == found_id]
        if len(sub) > 0:
            r = sub.iloc[0]

            def _pct(v):
                try: return f"{round(float(v) * 100)}%"
                except: return "—"

            def _mln(v):
                try: return f"{round(float(v) / 1_000_000)} млн тг"
                except: return "—"

            def _idx(v):
                try: return str(round(float(v) * 100))
                except: return "—"

            def _int(v):
                try: return str(int(float(v)))
                except: return "—"

            is_anom = str(r.get("is_anomaly", "")).strip().lower() in ("true", "1", "yes")
            anom_text = "отмечен системой как требующий проверки" if is_anom else "не вызывает подозрений"

            reason = str(r.get("anomaly_reason", "") or "").strip()
            reason_line = f"\n- Причина внимания: {reason}" if is_anom and reason else ""

            applicant_extra = (
                f"\n\nДАННЫЕ ЗАЯВИТЕЛЯ (ID: {found_id}):\n"
                f"- Индекс надёжности: {_idx(r.get('ml_probability'))} из 100\n"
                f"- Общий индекс: {_idx(r.get('final_score'))} из 100\n"
                f"- Место в рейтинге: #{r.get('rank', '—')}\n"
                f"- Группа заявителей: {r.get('cluster_label', '—')}\n"
                f"- Регион: {r.get('oblast', '—')}\n"
                f"- Направление субсидий: {str(r.get('primary_direction', '—')).replace('Субсидирование в ', '')}\n"
                f"- Доля одобренных заявок: {_pct(r.get('approval_rate'))}\n"
                f"- Количество поданных заявок: {_int(r.get('total_applications'))}\n"
                f"- Сумма полученных субсидий: {_mln(r.get('total_amount_received'))}\n"
                f"- Статус: {anom_text}"
                f"{reason_line}"
            )

    top_region = "—"
    if "oblast" in df.columns and "total_amount_received" in df.columns:
        reg_sum = df.groupby("oblast", dropna=False)["total_amount_received"].sum().sort_values(ascending=False)
        if len(reg_sum):
            top_region = f"{reg_sum.index[0]} — сумма субсидий {reg_sum.iloc[0]:,.0f} тг"

    cluster_block = (
        "ГРУППЫ ЗАЯВИТЕЛЕЙ (средние показатели по каждой группе):\n"
        f"{cluster_stats_text}\n\n"
        "Когда объясняешь группу заявителя — сравни его реальные показатели "
        "со средними по группе и скажи почему он туда попал, используя конкретные цифры. "
        "Не упоминай названия колонок — переводи их в понятные слова."
    )

    # ── ФИНАЛЬНЫЙ СИСТЕМНЫЙ ПРОМПТ ────────────────────────────────────────────
    return (
        "Ты AI-аналитик системы КазнаЛинза — инструмента для государственных служащих "
        "Министерства сельского хозяйства Казахстана.\n"
        "Отвечай ТОЛЬКО на основе данных ниже. Пиши на русском языке, кратко и понятно.\n\n"

        "ПРАВИЛА ЯЗЫКА (строго обязательно):\n"
        "- НИКОГДА не используй технические термины в ответе: ml_probability, is_anomaly, "
        "anomaly_score, final_score, cluster, contamination, isolation, composite, ml_rank.\n"
        "- ml_probability → называй 'индекс надёжности', показывай как целое число 0–100 "
        "(например: 0.87 → '87 из 100')\n"
        "- final_score → называй 'общий индекс', тоже умножай на 100\n"
        "- is_anomaly=True → пиши 'отмечен системой как требующий проверки'\n"
        "- is_anomaly=False → пиши 'не вызывает подозрений' или 'без особых отметок'\n"
        "- anomaly_score → не упоминай число вообще, только вывод: высокий / средний риск\n"
        "- approval_rate → называй 'доля одобренных заявок', показывай в процентах (0.82 → 82%)\n"
        "- total_amount_received → называй 'сумма полученных субсидий', показывай в млн тг\n"
        "- total_applications → называй 'количество поданных заявок'\n"
        "- cluster_label → называй 'группа заявителей' или просто называй название группы\n"
        "- Никогда не пиши числа с 4+ знаками после запятой — только округлённые значения\n\n"

        "ДАННЫЕ СИСТЕМЫ:\n"
        f"- Всего заявителей в базе: {n}\n"
        f"- Топ-5 по надёжности:\n{top5}\n"
        f"- Заявителей на проверке: {anomaly_count} из {n}\n"
        f"- Топ заявители на проверке:\n{top_anomalies}\n"
        f"- Лидирующий регион по субсидиям: {top_region}\n\n"
        f"{cluster_block}\n"
        f"{applicant_extra}\n\n"

        "СТИЛЬ ОТВЕТА:\n"
        "- Пиши как опытный аналитик-консультант, не как программист\n"
        "- Про конкретного заявителя: дай чёткий вывод — надёжный / требует внимания / "
        "входит в группу X, и объясни почему конкретными цифрами на понятном языке\n"
        "- Сравнивай показатели заявителя со средними по его группе\n"
        "- Не придумывай данные которых нет выше\n"
        "- Максимум 5–6 предложений если не просят подробнее\n"
    )


def _groq_chat(system_prompt: str, user_message: str, history: list) -> tuple[str, int]:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key or key == "your_key_here":
        return "Задайте переменную GROQ_API_KEY в файле .env в корне проекта (ключ Groq API).", 503

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history[-20:]:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_message})
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": msgs,
        "max_tokens": 600,
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        if r.status_code != 200:
            return f"Ошибка Groq API ({r.status_code}): {r.text[:500]}", 502
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        text = text.strip() or "(пустой ответ модели)"
        return text, 200
    except requests.RequestException as e:
        return f"Сеть или Groq недоступны: {e}", 502


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path != "/api/chat":
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self.send_json({"error": "empty body"}, 400)
            return
        try:
            raw = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json({"error": "invalid JSON"}, 400)
            return
        message = (raw.get("message") or "").strip()
        history = raw.get("history") or []
        if not isinstance(history, list):
            history = []
        if not message:
            self.send_json({"error": "empty message"}, 400)
            return

        df = _load_merged_dataframe()
        system_prompt = _build_chat_system_prompt(df, message)
        reply, status = _groq_chat(system_prompt, message, history)
        self.send_json({"reply": reply}, status)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/debug":
            rows = ALL_ROWS
            if rows:
                self.send_json({
                    "total": len(rows),
                    "columns": list(rows[0].keys()),
                    "row0": rows[0],
                    "row1": rows[1] if len(rows) > 1 else {}
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
                self.send_json(exp)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=404)
        elif path == "/":
            html = (_BASE_DIR / "app" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    import webbrowser, threading
    server = http.server.HTTPServer(("localhost", 8502), Handler)
    threading.Timer(0.5, lambda: webbrowser.open("http://localhost:8502")).start()
    print("Открываю http://localhost:8502")
    server.serve_forever()