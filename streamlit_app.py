from __future__ import annotations

import os
import re
import time
import io
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
SCORE_URL = f"{API_URL.rstrip('/')}/score"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "model_rf.joblib"),
)

FEATURES_CSV = os.getenv(
    "FEATURES_CSV",
    str(Path(__file__).resolve().parent / "data" / "Nova_pay_features.csv"),
)

DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

FEATURE_ORDER: List[str] = [
    "home_country",
    "source_currency",
    "dest_currency",
    "channel",
    "kyc_tier",
    "ip_country",
    "new_device",
    "location_mismatch",
    "ip_country_missing",
    "amount_src",
    "amount_usd",
    "fee",
    "ip_risk_score",
    "device_trust_score",
    "account_age_days",
    "txn_velocity_1h",
    "txn_velocity_24h",
    "corridor_risk",
    "risk_score_internal",
    "hour",
    "dayofweek",
]

CATEGORICAL_COLS = [
    "home_country",
    "source_currency",
    "dest_currency",
    "channel",
    "kyc_tier",
    "ip_country",
]


# ----------------------------
# HELPERS
# ----------------------------
@st.cache_resource
def load_pipeline_from_path(model_path: str, mtime: float):
    # mtime is only used to bust the cache when the file changes.
    return joblib.load(model_path)


@st.cache_resource
def get_explainer_from_path(model_path: str, mtime: float):
    pipe = load_pipeline_from_path(model_path, mtime)
    model = pipe.named_steps["model"]
    return shap.TreeExplainer(model)


def get_pipe_and_explainer(model_path: str):
    """
    Centralized loader to avoid forgetting to pass mtime.
    If model_path is overwritten on disk, mtime changes => cache invalidates.
    """
    mtime = os.path.getmtime(model_path)
    pipe = load_pipeline_from_path(model_path, mtime)
    explainer = get_explainer_from_path(model_path, mtime)
    return pipe, explainer


def ensure_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def load_batch_csv(file) -> pd.DataFrame:
    """
    Ultra-robust loader for Excel / 'fake CSV' exports.

    Fixes the specific failure mode you're seeing:
      - Pandas reads the entire header row as ONE column name:
        "a,b,c,d,..."
    Strategy:
      1) Read raw bytes
      2) Decode as utf-8-sig (removes BOM)
      3) If first line contains commas, split header/rows ourselves
      4) Otherwise fall back to pandas auto-detect
    """
    raw = file.getvalue()
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")

    text = raw.decode("utf-8-sig", errors="replace")

    # Normalize newlines and drop empty lines
    lines = [
        ln.strip()
        for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if ln.strip()
    ]
    if not lines:
        raise RuntimeError("Uploaded file appears empty.")

    header_line = lines[0]

    # If the header line has commas, we force comma parsing ourselves
    if "," in header_line:
        headers = [h.strip().strip('"') for h in header_line.split(",")]

        rows = []
        for ln in lines[1:]:
            vals = [v.strip().strip('"') for v in ln.split(",")]

            # pad/truncate to header length
            if len(vals) < len(headers):
                vals = vals + [""] * (len(headers) - len(vals))
            elif len(vals) > len(headers):
                vals = vals[: len(headers)]

            rows.append(vals)

        df = pd.DataFrame(rows, columns=headers)
    else:
        # fallback: try pandas
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def coerce_batch_types(Xb: pd.DataFrame) -> pd.DataFrame:
    """
    Convert types safely for batch scoring.
    Manual parsing can leave everything as string.
    """
    Xb = Xb.copy()

    # booleans (accept 0/1, true/false, yes/no)
    bool_map = {"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False}
    for b in ["new_device", "location_mismatch", "ip_country_missing"]:
        Xb[b] = (
            Xb[b].astype(str).str.strip().str.lower().map(bool_map)
        )

    # numeric columns
    numeric_cols = [
        c for c in FEATURE_ORDER
        if c not in CATEGORICAL_COLS and c not in ["new_device", "location_mismatch", "ip_country_missing"]
    ]
    for c in numeric_cols:
        Xb[c] = pd.to_numeric(Xb[c], errors="coerce")

    return Xb


def api_score(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Send many rows to FastAPI /score and return scores + decisions."""
    items = df[FEATURE_ORDER].to_dict(orient="records")
    payload = {"items": items}
    resp = requests.post(SCORE_URL, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    out = pd.DataFrame(resp.json().get("results", []))
    if out.empty:
        raise RuntimeError("API returned empty results.")
    if "score" not in out.columns:
        raise RuntimeError(f"API response missing 'score'. Got: {out.columns.tolist()}")

    out["score"] = out["score"].astype(float)
    out["decision"] = np.where(out["score"] >= threshold, "flag", "allow")
    return out


def make_group_map(feature_names: np.ndarray, base_cols: List[str]) -> Dict[str, List[int]]:
    """
    Group encoded feature names back to original logical columns.
    Works across ColumnTransformer naming patterns:
      cat__/num__/remainder__/passthrough__
    and one-hot expansions like:
      channel_mobile, home_country_US, etc.
    """
    groups: Dict[str, List[int]] = {}
    base_set = set(base_cols)

    for i, fn in enumerate(feature_names):
        s = str(fn)
        s2 = re.sub(r"^(cat__|num__|remainder__|passthrough__)", "", s)

        matched = False
        for base in base_set:
            if s2.startswith(base + "_"):
                groups.setdefault(base, []).append(i)
                matched = True
                break

        if not matched and s2 in base_set:
            groups.setdefault(s2, []).append(i)
            matched = True

        if not matched:
            groups.setdefault(s2, []).append(i)

    return groups


def group_shap_values(sv_row: np.ndarray, groups: Dict[str, List[int]]) -> pd.Series:
    grouped = {g: float(np.sum(sv_row[idxs])) for g, idxs in groups.items()}
    return pd.Series(grouped).sort_values(key=lambda s: np.abs(s), ascending=False)


def normalize_binary_shap_output(explainer, shap_values):
    """
    Normalize SHAP output into (sv, base_value) for class 1.
    Supports:
      - list of arrays [class0, class1]
      - 2D array (n, p)
      - 3D array (n, p, 2)
    """
    if isinstance(shap_values, list):
        sv = shap_values[1]
        base_value = explainer.expected_value[1]
        return sv, float(np.array(base_value).reshape(-1)[0])

    sv = np.array(shap_values)
    base_value = explainer.expected_value

    if sv.ndim == 3 and sv.shape[-1] == 2:
        sv = sv[:, :, 1]
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

    return sv, float(np.array(base_value).reshape(-1)[0])


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="FraudLens", layout="wide")
st.title("FraudLens")
st.caption("Streamlit dashboard for scoring + SHAP explainability (global & local).")

with st.sidebar:
    st.subheader("Settings")
    st.write("FastAPI:", API_URL)
    st.write("Model:", MODEL_PATH)

    threshold = st.slider("Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    use_api = st.toggle("Use FastAPI for scoring", value=True)

    if st.button("Reload model / clear cache"):
        st.cache_resource.clear()
        st.rerun()

tabs = st.tabs(["Single Transaction", "Batch CSV Scoring", "Global Explainability"])


# ----------------------------
# TAB 1: SINGLE TRANSACTION
# ----------------------------
with tabs[0]:
    st.subheader("1) Score a single transaction")

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            transaction_id = st.text_input("transaction_id", value="demo-ui-1")
            home_country = st.text_input("home_country", value="US")
            ip_country = st.text_input("ip_country", value="US")
            channel = st.selectbox("channel", ["mobile", "web", "agent", "api"], index=0)
            kyc_tier = st.selectbox("kyc_tier", ["standard", "basic", "enhanced"], index=0)

        with c2:
            source_currency = st.text_input("source_currency", value="USD")
            dest_currency = st.text_input("dest_currency", value="MXN")
            new_device = st.checkbox("new_device", value=False)
            location_mismatch = st.checkbox("location_mismatch", value=False)
            ip_country_missing = st.checkbox("ip_country_missing", value=False)

        with c3:
            amount_src = st.number_input("amount_src", value=120.0, step=10.0, min_value=0.0)
            amount_usd = st.number_input("amount_usd", value=120.0, step=10.0, min_value=0.0)
            fee = st.number_input("fee", value=2.1, step=0.1, min_value=0.0)
            ip_risk_score = st.number_input("ip_risk_score", value=0.2, step=0.05, min_value=0.0, max_value=1.0)
            device_trust_score = st.number_input("device_trust_score", value=0.7, step=0.05, min_value=0.0, max_value=1.0)

        c4, c5, c6 = st.columns(3)
        with c4:
            account_age_days = st.number_input("account_age_days", value=240, step=10, min_value=0)
            corridor_risk = st.number_input("corridor_risk", value=0.05, step=0.01, min_value=0.0)
        with c5:
            txn_velocity_1h = st.number_input("txn_velocity_1h", value=0, step=1, min_value=0)
            txn_velocity_24h = st.number_input("txn_velocity_24h", value=1, step=1, min_value=0)
        with c6:
            risk_score_internal = st.number_input("risk_score_internal", value=0.2, step=0.05, min_value=0.0, max_value=1.0)
            hour = st.number_input("hour", value=14, min_value=0, max_value=23, step=1)
            dayofweek = st.number_input("dayofweek", value=2, min_value=0, max_value=6, step=1)

        run_single = st.form_submit_button("Score + Explain")

    if run_single:
        st.caption(f"transaction_id: {transaction_id}")

        warnings = []
        if ip_country_missing and ip_country:
            warnings.append("ip_country_missing is True but ip_country is filled.")
        if (not ip_country_missing) and (not ip_country):
            warnings.append("ip_country is empty but ip_country_missing is False.")
        if amount_usd <= 0:
            warnings.append("amount_usd should be > 0.")
        if warnings:
            st.warning(" • " + "\n • ".join(warnings))

        row = {
            "home_country": home_country,
            "source_currency": source_currency,
            "dest_currency": dest_currency,
            "channel": channel,
            "kyc_tier": kyc_tier,
            "ip_country": ip_country,
            "new_device": bool(new_device),
            "location_mismatch": bool(location_mismatch),
            "ip_country_missing": bool(ip_country_missing),
            "amount_src": float(amount_src),
            "amount_usd": float(amount_usd),
            "fee": float(fee),
            "ip_risk_score": float(ip_risk_score),
            "device_trust_score": float(device_trust_score),
            "account_age_days": int(account_age_days),
            "txn_velocity_1h": int(txn_velocity_1h),
            "txn_velocity_24h": int(txn_velocity_24h),
            "corridor_risk": float(corridor_risk),
            "risk_score_internal": float(risk_score_internal),
            "hour": int(hour),
            "dayofweek": int(dayofweek),
        }

        X = pd.DataFrame([row], columns=FEATURE_ORDER)

        latency_ms = None
        if use_api:
            t0 = time.perf_counter()
            scored = api_score(X, threshold)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            score = float(scored.loc[0, "score"])
            decision = str(scored.loc[0, "decision"])
        else:
            pipe, _ = get_pipe_and_explainer(MODEL_PATH)
            score = float(pipe.predict_proba(X)[:, 1][0])
            decision = "flag" if score >= threshold else "allow"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fraud Score", f"{score:.3f}")
        m2.metric("Decision", decision.upper())
        m3.metric("Threshold", f"{threshold:.2f}")
        if latency_ms is not None:
            m4.metric("API latency", f"{latency_ms:.0f} ms")
        else:
            m4.metric("Scoring", "Local")

        st.divider()
        st.subheader("Local SHAP explanation (why this score?)")

        pipe, explainer = get_pipe_and_explainer(MODEL_PATH)
        prep = pipe.named_steps["prep"]

        X_enc = ensure_dense(prep.transform(X))
        feature_names = np.array(prep.get_feature_names_out())
        X_enc_df = pd.DataFrame(X_enc, columns=feature_names)

        shap_values = explainer.shap_values(X_enc)
        sv_all, base_value = normalize_binary_shap_output(explainer, shap_values)
        sv_row = np.array(sv_all[0]).reshape(-1)

        groups = make_group_map(
            feature_names,
            CATEGORICAL_COLS + [c for c in FEATURE_ORDER if c not in CATEGORICAL_COLS],
        )
        grouped = group_shap_values(sv_row, groups)

        topk = grouped.head(3)
        reasons = ", ".join([f"{idx} ({val:+.3f})" for idx, val in topk.items()])
        st.info(f"Top drivers: {reasons}")

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Grouped feature contributions (clean)")
            st.dataframe(grouped.head(15).to_frame("Grouped SHAP"), use_container_width=True)

        with right:
            st.markdown("### Raw force plot (encoded columns)")
            try:
                fig = shap.plots.force(
                    base_value,
                    sv_row,
                    X_enc_df.iloc[0, :],
                    matplotlib=True,
                    show=False,
                )
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.warning(f"Force plot failed: {e}")


# ----------------------------
# TAB 2: BATCH CSV SCORING
# ----------------------------
with tabs[1]:
    st.subheader("2) Batch scoring (upload CSV)")
    st.write("Upload a CSV containing the model input columns. You’ll get scores + decisions and can download results.")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = load_batch_csv(file)
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.write("Detected columns:", df.columns.tolist())

        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            st.error("❌ Uploaded CSV is missing required columns.")
            st.write("Required columns:", FEATURE_ORDER)
            st.write("Missing columns:", missing)
            st.stop()

        Xb = df[FEATURE_ORDER].copy()
        Xb = coerce_batch_types(Xb)

        # Validate parsing
        bad_bool = [c for c in ["new_device", "location_mismatch", "ip_country_missing"] if Xb[c].isna().any()]
        if bad_bool:
            st.error(f"Invalid boolean values in columns: {bad_bool}. Use 0/1 or TRUE/FALSE.")
            st.stop()

        bad_num = Xb.columns[Xb.isna().any()].tolist()
        if bad_num:
            st.error(f"Invalid or missing numeric values in columns: {bad_num}")
            st.stop()

        st.write("Preview (model inputs):")
        st.dataframe(Xb.head(10), use_container_width=True)

        if st.button("Score uploaded file"):
            if use_api:
                t0 = time.perf_counter()
                scored = api_score(Xb, threshold)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                base = df.reset_index(drop=True)
                sc = scored.reset_index(drop=True)

                # If API returns fields already in base (e.g., transaction_id), drop them
                overlap = [c for c in sc.columns if c in base.columns]
                sc = sc.drop(columns=overlap)
                out = pd.concat([base, sc], axis=1)

                st.caption(f"Scored via API in {latency_ms:.0f} ms")
            else:
                pipe, _ = get_pipe_and_explainer(MODEL_PATH)
                proba = pipe.predict_proba(Xb)[:, 1]
                out = df.copy()
                out["score"] = proba.astype(float)
                out["decision"] = np.where(out["score"] >= threshold, "flag", "allow")

            st.success("Scoring complete!")
            st.dataframe(out.head(20), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="scored_transactions.csv",
                mime="text/csv",
            )


# ----------------------------
# TAB 3: GLOBAL EXPLAINABILITY
# ----------------------------
with tabs[2]:
    st.subheader("3) Global SHAP explainability (model behaviour)")
    st.write(
        "This shows which features drive predictions overall. "
        "For best results, point FEATURES_CSV to a feature dataset."
    )

    pipe, explainer = get_pipe_and_explainer(MODEL_PATH)
    prep = pipe.named_steps["prep"]

    if Path(FEATURES_CSV).exists():
        df_g = pd.read_csv(FEATURES_CSV)
        usable_cols = [c for c in FEATURE_ORDER if c in df_g.columns]
        Xg = df_g[usable_cols].dropna().head(500)
        st.caption(f"Using {len(Xg)} rows from: {FEATURES_CSV}")
    else:
        st.warning("No FEATURES_CSV found. Set env var FEATURES_CSV to your dataset path.")
        st.stop()

    Xg_enc = ensure_dense(prep.transform(Xg))
    feature_names = np.array(prep.get_feature_names_out())
    Xg_enc_df = pd.DataFrame(Xg_enc, columns=feature_names)

    shap_values = explainer.shap_values(Xg_enc)
    sv_global, _ = normalize_binary_shap_output(explainer, shap_values)

    st.markdown("### SHAP summary plot (global)")
    fig, ax = plt.subplots()
    shap.summary_plot(sv_global, Xg_enc_df, show=False)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Grouped global importance (clean)")
    groups = make_group_map(
        feature_names,
        CATEGORICAL_COLS + [c for c in FEATURE_ORDER if c not in CATEGORICAL_COLS],
    )

    mean_abs = np.mean(np.abs(sv_global), axis=0)
    grouped_importance = {g: float(np.sum(mean_abs[idxs])) for g, idxs in groups.items()}
    grouped_imp = pd.Series(grouped_importance).sort_values(ascending=False)

    st.dataframe(grouped_imp.to_frame("Mean |SHAP| (grouped)").head(25), use_container_width=True)
