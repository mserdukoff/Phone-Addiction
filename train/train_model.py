import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error,
    mean_absolute_percentage_error,
    median_absolute_error
)
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/teen_phone_addiction_dataset.csv")

# Drop unnecessary columns
df = df.drop(columns=["ID", "Name", "Location"])

# Handle missing values
df = df.dropna()

# Encode categorical features
le = LabelEncoder()
categorical_cols = ["Gender", "School_Grade", "Phone_Usage_Purpose"]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop("Addiction_Level", axis=1)
y = df["Addiction_Level"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("TeenPhoneAddiction")

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    max_err = max_error(y_test, preds)
    median_ae = median_absolute_error(y_test, preds)

    mlflow.log_param("model_type", "RandomForestRegressor")

    # Log all metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("max_error", max_err)
    mlflow.log_metric("median_absolute_error", median_ae)

    mlflow.sklearn.log_model(model, "model")

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Max Error: {max_err:.4f}")
    print(f"Median AE: {median_ae:.4f}")