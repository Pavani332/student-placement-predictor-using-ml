"""
Student Placement Prediction - Model Training

Run from the project folder (the one that contains app.py):
    python src/train.py

What it does:
    1. Loads data/placement_data.csv if you have one with a "Placement"
       column ("Placed" / "Not Placed"). Otherwise it creates synthetic data.
    2. Compares two models using cross-validation and keeps the better one.
    3. Saves the trained model to models/placement_model.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

# This file lives in src/, so the project folder is one level up.
# If you put train.py next to app.py instead, this still works.
HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent if HERE.name == "src" else HERE

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "placement_data.csv"
MODEL_FILE = MODEL_DIR / "placement_model.pkl"

TARGET = "Placement"
POSITIVE = "Placed"
NEGATIVE = "Not Placed"

# Column names must match the dictionary built in app.py
NUMERIC_COLS = [
    "Age", "CGPA", "Tenth_Percentage", "Twelfth_Percentage",
    "Attendance_Percentage", "Communication_Skills", "Technical_Skills",
    "Aptitude_Score", "Internships", "Projects", "Certifications",
    "Backlogs", "Coding_Skills", "Work_Experience_Months",
]
CATEGORICAL_COLS = ["Gender", "Extracurricular_Activities"]


# ============================================================
# SYNTHETIC DATA
# ============================================================

def make_synthetic_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Create realistic-looking student data. Strong students are more
    likely to be placed, with some randomness so it is not a perfect rule."""
    rng = np.random.default_rng(seed)

    def half_steps(x):
        return np.round(np.clip(x, 0, 10) * 2) / 2

    cgpa = np.clip(rng.normal(7.2, 1.0, n), 4.0, 10.0).round(2)
    tenth = np.clip(rng.normal(78, 10, n), 45, 100).round(1)
    twelfth = np.clip(rng.normal(74, 10, n), 45, 100).round(1)
    attendance = np.clip(rng.normal(82, 10, n), 40, 100).round(1)
    aptitude = np.clip(rng.normal(62, 15, n), 0, 100).round(0)

    # Skills are loosely linked to CGPA
    communication = half_steps(rng.normal(6.0, 1.8, n))
    technical = half_steps(rng.normal(5.5 + (cgpa - 7) * 0.5, 1.8, n))
    coding = half_steps(rng.normal(5.5 + (cgpa - 7) * 0.5, 2.0, n))

    internships = np.clip(rng.poisson(1.0, n), 0, 20)
    projects = np.clip(rng.poisson(2.5, n), 0, 50)
    certifications = np.clip(rng.poisson(2.0, n), 0, 50)
    backlogs = np.clip(rng.poisson(np.where(cgpa < 6.5, 1.8, 0.4)), 0, 30)
    work_exp = np.where(rng.random(n) < 0.7, 0, rng.poisson(8, n))
    work_exp = np.clip(work_exp, 0, 120)

    gender = rng.choice(["Male", "Female"], n)
    extra = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])
    age = np.clip(rng.normal(22, 1.2, n).round(), 18, 30).astype(int)

    logit = (
        -16.2
        + 0.90 * cgpa
        + 0.02 * tenth
        + 0.02 * twelfth
        + 0.02 * attendance
        + 0.15 * communication
        + 0.20 * technical
        + 0.20 * coding
        + 0.02 * aptitude
        + 0.35 * internships
        + 0.15 * projects
        + 0.10 * certifications
        - 0.60 * backlogs
        + 0.03 * np.minimum(work_exp, 12)
        + 0.20 * (extra == "Yes")
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-1.6 * logit))
    placed = rng.random(n) < prob

    return pd.DataFrame({
        "Age": age,
        "CGPA": cgpa,
        "Tenth_Percentage": tenth,
        "Twelfth_Percentage": twelfth,
        "Attendance_Percentage": attendance,
        "Communication_Skills": communication,
        "Technical_Skills": technical,
        "Aptitude_Score": aptitude,
        "Internships": internships,
        "Projects": projects,
        "Certifications": certifications,
        "Backlogs": backlogs,
        "Coding_Skills": coding,
        "Work_Experience_Months": work_exp,
        "Gender": gender,
        "Extracurricular_Activities": extra,
        TARGET: np.where(placed, POSITIVE, NEGATIVE),
    })


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        print(f"Loading dataset: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        missing = [c for c in NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Your CSV is missing these columns: {missing}")
        return df

    print("No dataset found. Creating synthetic data...")
    df = make_synthetic_data()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved synthetic dataset to: {DATA_FILE}")
    return df


# ============================================================
# MODEL
# ============================================================

def build_pipeline(classifier) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def main() -> None:
    df = load_data()
    df = df.dropna(subset=NUMERIC_COLS + CATEGORICAL_COLS + [TARGET])

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]

    print(f"\nRows: {len(df)}")
    print(y.value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    best_name, best_pipe, best_score = None, None, -1.0
    print("\nCross-validation (ROC AUC):")
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        score = cross_val_score(
            pipe, X_train, y_train, cv=5,
            scoring="roc_auc", n_jobs=-1
        ).mean()
        print(f"  {name}: {score:.3f}")
        if score > best_score:
            best_name, best_pipe, best_score = name, pipe, score

    print(f"\nBest model: {best_name}")
    best_pipe.fit(X_train, y_train)

    # Test-set evaluation
    pred = best_pipe.predict(X_test)
    placed_idx = list(best_pipe.classes_).index(POSITIVE)
    proba = best_pipe.predict_proba(X_test)[:, placed_idx]

    print(f"\nTest accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"Test ROC AUC:  {roc_auc_score(y_test == POSITIVE, proba):.3f}\n")
    print(classification_report(y_test, pred))

    # Retrain on all data, then save
    best_pipe.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()