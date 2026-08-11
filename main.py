
# Solar Power Production Prediction API


import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field




ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

# Features the API can build from a single request
AVAILABLE_FEATURES = {
    "WindSpeed", "Sunshine", "AirPressure", "Radiation",
    "AirTemperature", "RelativeAirHumidity", "Hour", "Month",
}




registry = json.loads((MODELS_DIR / "registry.json").read_text())

models = {}
metadata = {}

for version in registry["versions"]:
    models[version] = joblib.load(MODELS_DIR / version / "model.pkl")
    metadata[version] = json.loads((MODELS_DIR / version / "metadata.json").read_text())

    # Fail at startup, not on the first user request
    missing = set(metadata[version]["features"]) - AVAILABLE_FEATURES
    if missing:
        raise RuntimeError(f"Version {version} needs features the API cannot build: {missing}")



app = FastAPI(
    title="Solar Power Production Prediction API",
    description="Predicts hourly solar power production using versioned ML models.",
    version="1.0.0",
)



class SolarInput(BaseModel):
    timestamp: datetime = Field(..., examples=["2017-07-15T12:00:00"])
    WindSpeed: float = Field(..., ge=0, le=60, description="m/s")
    Sunshine: float = Field(..., ge=0, le=60, description="minutes per hour")
    AirPressure: float = Field(..., ge=850, le=1100, description="hPa")
    Radiation: float = Field(..., ge=-50, le=1500, description="W/m2")
    AirTemperature: float = Field(..., ge=-50, le=60, description="degrees C")
    RelativeAirHumidity: float = Field(..., ge=0, le=100, description="%")




def resolve_model_version(version: str | None) -> str:
    """Return the requested version, or the promoted one if none given."""
    selected = version or registry["promoted"]

    if selected not in models:
        raise HTTPException(404, f"Unknown model version: {selected}")

    return selected


def prepare_input(data: SolarInput, version: str) -> pd.DataFrame:
    """Build the feature row in the exact order used during training."""
    row = data.model_dump()
    timestamp = row.pop("timestamp")

    row["Hour"] = timestamp.hour
    row["Month"] = timestamp.month
    row["Radiation"] = max(row["Radiation"], 0)   # same fix applied in training

    return pd.DataFrame([row])[metadata[version]["features"]]




@app.get("/health")
def health():
    """Liveness and model status."""
    return {
        "status": "ok",
        "promoted_model": registry["promoted"],
        "models_loaded": len(models),
    }


@app.get("/versions")
def get_versions():
    """All trained versions and which one is live."""
    return registry


@app.get("/metadata")
def get_metadata(model_version: str | None = None):
    """Feature contract, CV score and training environment."""
    return metadata[resolve_model_version(model_version)]


@app.post("/predict")
def predict(data: SolarInput, model_version: str | None = None):
    """Predict hourly solar power production."""
    version = resolve_model_version(model_version)

    prediction = models[version].predict(prepare_input(data, version))[0]
    prediction = max(float(prediction), 0)   # production cannot be negative

    return {
        "prediction": round(prediction, 2),
        "unit": "Wh",
        "model_version": version,
    }