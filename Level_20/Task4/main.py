from pipeline.model_deployment_pipeline import model_deployment_pipeline

pipeline_instance = model_deployment_pipeline()
print(f"Pipeline run ID: {pipeline_instance.id}")
