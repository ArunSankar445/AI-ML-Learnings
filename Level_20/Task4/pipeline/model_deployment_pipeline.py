# pipeline/model_deployment_pipeline.py
from zenml.pipelines import pipeline
from steps.data_ingestion import data_ingestion
from steps.model_training import model_training
from steps.model_evaluation import model_evaluation
from steps.deployment_trigger import deployment_trigger
from steps.deploy_model import deploy_model


@pipeline
def model_deployment_pipeline():
    data = data_ingestion()
    model = model_training(data)
    r2 = model_evaluation(model, data)
    deploy = deployment_trigger(r2)
    deploy_model(model, deploy)
