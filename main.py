from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predictor import train_model, predict_next_hours

app = FastAPI()

# Allow React Native frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = train_model()

@app.get("/predict/{hours}")
def get_prediction(hours: int):
    return predict_next_hours(model, hours)
