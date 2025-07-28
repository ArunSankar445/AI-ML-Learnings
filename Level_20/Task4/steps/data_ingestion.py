# steps/data_ingestion.py
from zenml.steps import step
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


@step
def data_ingestion():
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
