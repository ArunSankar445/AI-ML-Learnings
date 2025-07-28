from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from zenml import step
from sklearn.linear_model import LinearRegression
import mlflow


@step(experiment_tracker="mlflow_tracker")
def evaluation_model(model: LinearRegression, df: pd.DataFrame):
    """Evaluating the model by MSE and R2 score"""
    X = df.drop(columns=["target"])
    y_true = df["target"]
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mlflow.set_experiment("my_experiment_name")

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
