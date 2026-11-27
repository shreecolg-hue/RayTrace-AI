import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

CLASS_NAME_TO_ID = {
    "good condition": 0,
    "electric faulty": 1,
    "physical damage": 2,
}

FEATURE_COLUMNS = [
    "panel_class_id",
    "class_confidence",
    "defect_area_ratio",
    "num_cut_segments",
    "total_cut_length_mm",
    "laser_power_w",
]


def generate_recovery_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create a structured dataset linking panel triage, defect coverage, and laser cut plan to yield (%)."""
    rng = np.random.default_rng(seed)

    panel_class_id = rng.integers(0, 3, size=n_samples)
    class_confidence = rng.uniform(0.55, 0.995, size=n_samples)

    base_defect_ratio = np.where(
        panel_class_id == 0,
        rng.uniform(0.0, 0.12, size=n_samples),
        np.where(
            panel_class_id == 1,
            rng.uniform(0.05, 0.35, size=n_samples),
            rng.uniform(0.15, 0.70, size=n_samples),
        ),
    )
    defect_area_ratio = np.clip(
        base_defect_ratio + rng.normal(0.0, 0.04, size=n_samples),
        0.0,
        0.85,
    )

    num_cut_segments = rng.integers(1, 16, size=n_samples)
    total_cut_length_mm = rng.uniform(60.0, 1400.0, size=n_samples)
    laser_power_w = rng.uniform(25.0, 85.0, size=n_samples)

    raw_yield_pct = (
        96.5
        - 78.0 * defect_area_ratio
        - 10.0 * (panel_class_id == 2).astype(float)
        - 3.5 * (panel_class_id == 1).astype(float)
        - 0.007 * total_cut_length_mm
        - 0.45 * np.maximum(0.0, num_cut_segments - 4)
        - 0.03 * np.maximum(0.0, laser_power_w - 60.0)
        + 2.0 * (class_confidence - 0.8)
        + rng.normal(0.0, 2.2, size=n_samples)
    )
    recoverable_yield_pct = np.clip(raw_yield_pct, 5.0, 98.5)

    return pd.DataFrame({
        "panel_class_id": panel_class_id,
        "class_confidence": class_confidence,
        "defect_area_ratio": defect_area_ratio,
        "num_cut_segments": num_cut_segments,
        "total_cut_length_mm": total_cut_length_mm,
        "laser_power_w": laser_power_w,
        "recoverable_yield_pct": recoverable_yield_pct,
    })


def train_random_forest_model(
    model_path: str = "recovery_yield_model.joblib",
    n_samples: int = 2500,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = generate_recovery_dataset(n_samples=n_samples, seed=random_state)

    X = df[FEATURE_COLUMNS]
    y = df["recoverable_yield_pct"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = root_mean_squared_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)

    print("=== RandomForestRegressor Validation Metrics ===")
    print(f"MAE:  {mae:.2f}% recoverable yield")
    print(f"RMSE: {rmse:.2f}% recoverable yield")
    print(f"R2:   {r2:.4f}")

    print("\n=== Feature Importances ===")
    for col, importance in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {col:<22}: {importance:.4f}")

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "class_name_to_id": CLASS_NAME_TO_ID,
    }
    joblib.dump(artifact, model_path, compress=3)
    print(f"\nSaved trained RandomForestRegressor to: {os.path.abspath(model_path)}")
    return model


def load_recovery_model(model_path: str = "recovery_yield_model.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Run train_random_forest_model() first."
        )
    return joblib.load(model_path)


def predict_recovery_yield(
    panel_class,
    class_confidence: float,
    defect_area_ratio: float,
    num_cut_segments: int,
    total_cut_length_mm: float,
    laser_power_w: float = 45.0,
    model_path: str = "recovery_yield_model.joblib",
) -> float:
    artifact = load_recovery_model(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    class_map = artifact.get("class_name_to_id", CLASS_NAME_TO_ID)

    if isinstance(panel_class, str):
        class_id = class_map.get(panel_class.strip().lower(), 1)
    else:
        class_id = int(panel_class)

    input_df = pd.DataFrame([{
        "panel_class_id": class_id,
        "class_confidence": float(np.clip(class_confidence, 0.0, 1.0)),
        "defect_area_ratio": float(np.clip(defect_area_ratio, 0.0, 1.0)),
        "num_cut_segments": max(0, int(num_cut_segments)),
        "total_cut_length_mm": max(0.0, float(total_cut_length_mm)),
        "laser_power_w": max(0.0, float(laser_power_w)),
    }])[feature_columns]

    prediction = float(model.predict(input_df)[0])
    return round(float(np.clip(prediction, 0.0, 100.0)), 2)


if __name__ == "__main__":
    train_random_forest_model()

    sample_prediction = predict_recovery_yield(
        panel_class="electric faulty",
        class_confidence=0.92,
        defect_area_ratio=0.18,
        num_cut_segments=5,
        total_cut_length_mm=360.0,
        laser_power_w=45.0,
    )
    print(f"\nSample Predicted Recoverable Yield: {sample_prediction}%")