# setting the structure of the output files

from datetime import datetime
from pydantic import BaseModel, Field

class ForecastResponse(BaseModel):
    forecast_origin: datetime
    target_timestamp: datetime
    predicted_total_power_consumption: float = Field(
        description="Predicted total demand"
        )
    model: str = "LSTM"
    features: int
    lookback_steps: int

# this will produce consistend JSON output for the API endpoint
