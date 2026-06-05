from pydantic import BaseModel, Field, ConfigDict

class CustomerPredictionInput(BaseModel):
    # 1. We remove 'example' from the Field definitions entirely
    Amount: float = Field(..., description="Transaction raw amount")
    Value: float = Field(..., description="Transaction value metric")
    TransactionHour: int = Field(..., ge=0, le=23, description="Hour of transaction")
    ProductCategory_airtime: int = Field(..., ge=0, le=1, description="One-hot feature flag")

    # 2. We use the modern ConfigDict to define the global JSON Schema examples
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Amount": 2500.0,
                "Value": 2500.0,
                "TransactionHour": 18,
                "ProductCategory_airtime": 0
            }
        }
    )

class RiskPredictionResponse(BaseModel):
    is_high_risk: int = Field(..., description="Binary risk assignment (0=Low, 1=High)")
    probability: float = Field(..., description="Calculated model confidence score")
    
    # Optional example for your response schema as well
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_high_risk": 1,
                "probability": 0.8742
            }
        }
    )