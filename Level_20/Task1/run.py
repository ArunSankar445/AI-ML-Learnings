from pipelines.pipeline import regression_pipeline

if __name__ == "__main__":
    p = regression_pipeline()
    print("Pipeline run ID: {p.id}")
