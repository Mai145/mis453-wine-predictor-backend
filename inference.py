import pickle
import numpy as np
from sklearn.datasets import load_wine

# 1. Load the saved model
with open('wine_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load class names for better output (e.g., 'class_0', 'class_1')
class_names = load_wine().target_names

# 2. Define a sample input (13 features of a specific wine)
# This is a sample row from the dataset for testing
sample_input = np.array([[13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050.0]])

# 3. Make Prediction
prediction_index = loaded_model.predict(sample_input)[0]
prediction_name = class_names[prediction_index]

print(f"Predicted Wine Cultivator: {prediction_name} (Class {prediction_index})")