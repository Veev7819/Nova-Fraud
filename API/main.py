from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

app = FastAPI(title="NovaPay Transaction Risk Scoring API", version="0.1.0")


class Transaction(BaseModel):
    # minimal subset for demo; extend as needed
    transaction_id: Optional[str] = None
    timestamp: Optional[str] = None
    home_country: str
    source_currency: str
    dest_currency: str
    channel: str
    kyc_tier: str
    ip_country: str
    new_device: bool
    location_mismatch: bool
    ip_country_missing: bool
    amount_src: float
    amount_usd: float
    fee: float
    ip_risk_score: float
    device_trust_score: float
    account_age_days: int
    txn_velocity_1h: int
    txn_velocity_24h: int
    corridor_risk: float
    risk_score_internal: float
    hour: int
    dayofweek: int


class ScoreRequest(BaseModel):
    items: List[Transaction]


class ScoreResponseItem(BaseModel):
    transaction_id: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0)
    decision: str


class ScoreResponse(BaseModel):
    results: List[ScoreResponseItem]


MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models"/"model_rf.joblib")
THRESHOLD: float = 0.5
PIPELINE = None


def load_pipeline() -> Any:
    global PIPELINE
    if PIPELINE is None:
        try:
            PIPELINE = joblib.load(MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model pipeline from {MODEL_PATH}: {exc}")
    return PIPELINE


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
    "dayofweek"
]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided")
    pipe = load_pipeline()

    records: List[Dict[str, Any]] = []
    for t in req.items:
        rec = t.model_dump()
        # Ensure all model features exist; cast types
        formatted: Dict[str, Any] = {}
        for col in FEATURE_ORDER:
            val = rec.get(col)
            formatted[col] = val
        records.append(formatted)

    X = pd.DataFrame.from_records(records, columns=FEATURE_ORDER)
    try:
        proba = pipe.predict_proba(X)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    results: List[ScoreResponseItem] = []
    for i, t in enumerate(req.items):
        s = float(np.clip(proba[i], 0.0, 1.0))
        decision = "flag" if s >= THRESHOLD else "allow"
        results.append(ScoreResponseItem(transaction_id=t.transaction_id, score=s, decision=decision))

    return ScoreResponse(results=results)

