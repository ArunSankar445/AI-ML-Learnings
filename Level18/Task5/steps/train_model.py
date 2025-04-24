import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from zenml import step


@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    print("Starting model Training")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Model training complete.")
    print(f"R² on training data: {train_score:.4f}")
    print(f"R² on test data: {test_score:.4f}")

    return model
