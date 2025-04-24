from zenml import step
import pandas as pd


@step
def ingest_data() -> pd.DataFrame:
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "feature3": [0.1, 0.2, 0.3, 0.4, 0.5],
        "target": [2, 4, 6, 8, 10],
    }
    df = pd.DataFrame(data)
    return df
