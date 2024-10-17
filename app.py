from fastapi import FastAPI
import pandas as pd
import model_app
from pydantic import BaseModel, validator

class HouseData(BaseModel):
    size: float
    nb_rooms: int
    garden: int  # Garden can only be 0 or 1

    # Validator to enforce that garden is either 0 or 1
    @validator('garden')
    def check_garden_value(cls, v):
        if v not in [0, 1]:
            raise ValueError('garden must be either 0 or 1')
        return v
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(data: HouseData):
    input_data = pd.DataFrame({
        'size': [data.size],
        'nb_rooms': [data.nb_rooms],
        'garden': [data.garden],
    })
    
    predictions = model_app.predict_model(input_data)
    return {"prediction_price": predictions[0]}