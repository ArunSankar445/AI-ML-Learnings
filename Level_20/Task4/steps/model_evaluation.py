# steps/model_evaluation.py
from zenml.steps import step
from sklearn.metrics import r2_score


@step
def model_evaluation(model: object, data: dict) -> float:
    X_test = data["X_test"]
    y_test = data["y_test"]

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    return r2
