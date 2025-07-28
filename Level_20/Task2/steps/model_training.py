import pandas as pd
from sklearn.linear_model import LinearRegression
from zenml import step
import mlflow


@step(experiment_tracker="mlflow_tracker")
def model_training(df: pd.DataFrame) -> LinearRegression:
    """Training the LinearRegression model"""
    if mlflow.active_run():
        mlflow.end_run()

    X = df.drop(columns=["target"])
    y = df["target"]

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X, y)
        return model
