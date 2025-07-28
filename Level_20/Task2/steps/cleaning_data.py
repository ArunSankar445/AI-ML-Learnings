import pandas as pd
from zenml import step


@step(experiment_tracker="mlflow_tracker")
def cleaning_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handling missing values"""

    cleaned_df = df.dropna()
    return cleaned_df
