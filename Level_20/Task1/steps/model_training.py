import pandas as pd
from sklearn.linear_model import LinearRegression
from zenml import step


@step
def model_training(df: pd.DataFrame) -> LinearRegression:
    """Training the LinearRegression model"""

    X = df.drop(columns=["target"])
    y = df["target"]

    model = LinearRegression()
    model.fit(X, y)
    return model
