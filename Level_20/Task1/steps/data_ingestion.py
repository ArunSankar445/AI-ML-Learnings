import pandas as pd
from sklearn.datasets import load_diabetes
from zenml import step


@step
def data_ingestion() -> pd.DataFrame:
    """Getting the data from the sklearn.datasets.load_diabetes"""
    data = load_diabetes(as_frame=True)
    df = data.frame
    return df
