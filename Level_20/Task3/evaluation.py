from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


class Evaluator:
    def evaluate(self, y_true, y_pred):
        mse = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {"mse": mse, "rmse": rmse, "r2": r2}
