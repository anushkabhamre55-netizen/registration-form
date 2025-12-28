import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the saved model
model = joblib.load("models/job_pipeline.joblib")

# Load your dataset
data = pd.read_csv("data/dataset.csv")  # update path if your file is elsewhere

# Split into features (X) and target (y)
X = data.drop("role", axis=1)
y = data["role"]

# Make predictions
y_pred = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
