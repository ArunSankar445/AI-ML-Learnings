from zenml import step, pipeline
from sklearn.datasets import load_diabetes
from model_dev import LinearRegressionModel, RandomForestModel
from sklearn.model_selection import train_test_split
from evaluation import Evaluator


@step
def load_data():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@step
def train_evaluate(data):
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    models = {
        "LinearRegression": LinearRegressionModel(),
        "RandomForest": RandomForestModel(),
    }

    evaluation = Evaluator()
    result = {}

    for name, model in models.items():
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluation.evaluate(y_test, y_pred)
        result[name] = metrics

    return result


@pipeline
def model_comparison_pipeline():
    data = load_data()
    train_evaluate(data)
