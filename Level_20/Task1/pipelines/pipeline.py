from steps.data_ingestion import data_ingestion
from steps.cleaning_data import cleaning_data
from steps.model_training import model_training
from steps.evaluation_model import evaluation_model
from zenml import pipeline


@pipeline
def regression_pipeline():
    data = data_ingestion()
    clean_data = cleaning_data(data)
    model = model_training(clean_data)
    evaluation_model(model, clean_data)
