from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from zenml import step
from sklearn.linear_model import LinearRegression


@step
def evaluation_model(model: LinearRegression, df: pd.DataFrame):
    """Evaluating the model by MSE and R2 score"""
    X = df.drop(columns=["target"])
    y_true = df["target"]
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Square Error: {mse}")
    print(f"R2 Score: {r2}")
