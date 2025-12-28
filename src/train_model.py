# src/train_model.py (patched)
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import random

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "job_pipeline.joblib")
RANDOM_STATE = 42

def generate_dataset(n=2000, seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    ages = np.random.randint(18, 60, size=n)
    educations = np.random.choice(["High School", "Diploma", "Bachelors", "Masters", "PhD"], size=n, p=[0.2,0.15,0.35,0.25,0.05])
    years_exp = np.maximum(0, (ages - np.random.randint(18,25,size=n)) )
    skills = np.random.randint(0, 10, size=n)
    communication = np.random.randint(0, 11, size=n)
    leadership = np.random.randint(0, 11, size=n)
    technical = np.random.randint(0, 11, size=n)
    certifications = np.random.randint(0, 5, size=n)
    willing_to_travel = np.random.choice([0,1], size=n, p=[0.7,0.3])
    domains = np.random.choice(["IT", "Finance", "Operations", "Marketing", "HR", "Manufacturing"], size=n)

    roles = []
    for i in range(n):
        # simple heuristic for synthetic label
        if technical[i] >= 7 and educations[i] in ["Bachelors","Masters","PhD"] and domains[i] in ["IT","Finance"]:
            roles.append("DATA ANALYST" if skills[i] >= 2 else "PROJECT MANAGER")
        elif leadership[i] >= 7 and years_exp[i] >= 8:
            roles.append("PROJECT MANAGER")
        elif communication[i] >= 7 and skills[i] <= 3:
            roles.append("HR")
        elif technical[i] >= 5 and skills[i] >= 3:
            roles.append("BUSINESS ANALYST")
        elif skills[i] <= 1 and years_exp[i] < 3:
            roles.append("CLERK")
        else:
            roles.append(random.choice(["MANAGER","BUSINESS ANALYST","DATA ANALYST","HR"]))

    df = pd.DataFrame({
        "age": ages,
        "education": educations,
        "years_experience": years_exp,
        "skills_count": skills,
        "communication": communication,
        "leadership": leadership,
        "technical": technical,
        "certifications": certifications,
        "willing_to_travel": willing_to_travel,
        "preferred_domain": domains,
        "role": roles
    })
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print("Generated dataset at", DATA_PATH, "shape:", df.shape)
    return df

def load_or_generate_dataset():
    if os.path.exists(DATA_PATH):
        print("Loading existing dataset:", DATA_PATH)
        return pd.read_csv(DATA_PATH)
    else:
        print("Dataset not found — generating synthetic dataset.")
        return generate_dataset(2000)

def train_and_save(df=None, model_path=MODEL_PATH):
    if df is None:
        df = load_or_generate_dataset()
    # features and target
    X = df.drop(columns=["role"])
    y = df["role"]
    numeric_feats = ["age","years_experience","skills_count","communication","leadership","technical","certifications","willing_to_travel"]
    cat_feats = ["education","preferred_domain"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to {model_path}")
    return pipeline

if __name__ == "__main__":
    df = load_or_generate_dataset()
    train_and_save(df)
