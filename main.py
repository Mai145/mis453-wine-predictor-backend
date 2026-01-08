import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the App
app = FastAPI(
    title="Wine Variety Predictor",
    description="A Machine Learning API that predicts wine cultivator based on chemical ingredients.",
    version="1.0"
)

# 2. Define the Input Data Format (13 features)
class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

# 3. Load the Model (Global Variable)
# We load it once when the script starts
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ["Class 0", "Class 1", "Class 2"]

@app.get("/")
def home():
    """Root endpoint to check if API is running."""
    return {"message": "Wine Predictor API is Live"}

@app.post("/predict")
def predict_wine(features: WineInput):
    """
    Predicts the wine variety based on 13 chemical features.
    """
    # Convert input object to a list of values
    input_data = [[
        features.alcohol, features.malic_acid, features.ash, 
        features.alcalinity_of_ash, features.magnesium, features.total_phenols,
        features.flavanoids, features.nonflavanoid_phenols, features.proanthocyanins,
        features.color_intensity, features.hue, features.od280_od315_of_diluted_wines,
        features.proline
    ]]
    
    # Make prediction
    prediction_idx = model.predict(input_data)[0]
    prediction_name = class_names[prediction_idx]
    
    return {
        "prediction_index": int(prediction_idx),
        "prediction_name": prediction_name
    }