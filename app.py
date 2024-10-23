import os
import pandas as pd
import model_app
import train_model

from fastapi import FastAPI
from pydantic import BaseModel, validator

if not os.path.exists('./bert_regression_model') :
    train_model.build_bert_model()
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

@app.post("/predict_with_bert")
async def predict_bert(data: HouseData):
    predictions = model_app.predict_bert_model(data.size, data.nb_rooms, data.garden)
    return {"prediction_price": predictions}