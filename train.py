import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Dataset
print("Loading Wine dataset...")
data = load_wine()
X = data.data
y = data.target

# 2. Split Data (80% Train, 20% Test)
# I the set random_state=42 for reproducibility as mentioned in the report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Model
# I set the set max_depth=3 to prevent overfitting (Hyper-parameter decision)
print("Training Decision Tree Classifier...")
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Complete. Accuracy: {accuracy*100:.2f}%")

# 5. Save the Model (Serialization)
with open('wine_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to 'wine_model.pkl'")