from zenml.steps import step
import mlflow
import mlflow.sklearn


@step
def deploy_model(model: object, deploy: bool):
    if deploy:
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model")
        print("Model deployed successfully!")
    else:
        print("Model deployment skipped due to low R2 score.")
