from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class LinearRegressionModel(Model):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class RandomForestModel(Model):
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
