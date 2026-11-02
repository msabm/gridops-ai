from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from src.api.schemas import ForecastResponse
from src.api.services import (
    load_model_artifacts,
    generate_forecast,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # loading the model and preprocessing artifacts at startup
    model, preprocessing = load_model_artifacts()

    app.state.model = model
    app.state.preprocessing = preprocessing

    yield

app = FastAPI(
    title="GridOps AI API",
    description="API for forecasting total power consumption using LSTM model", 
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root ():
    return {
        "message": "Welcome to the GridOps AI API.",
        "status": "online",
    }

@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "model": "LSTM",
        "features": len(request.app.state.preprocessing["feature_columns"]),
    }

@app.get("/forecast", response_model=ForecastResponse)
def forecast(request: Request):
    return generate_forecast(
        request.app.state.model,
        request.app.state.preprocessing,
    )