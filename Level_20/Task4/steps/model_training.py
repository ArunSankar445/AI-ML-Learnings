# steps/model_training.py
from zenml.steps import step
from sklearn.linear_model import LinearRegression


@step
def model_training(data: dict) -> object:
    X_train = data["X_train"]
    y_train = data["y_train"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model
