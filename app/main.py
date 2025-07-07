from fastapi import FastAPI
from app.schemas import PredictRequest
from app.model import predict

app = FastAPI()

@app.post("/predict")
def make_prediction(request: PredictRequest):
    prediction = predict(request.features)
    return {"predicted_addiction_level": prediction}