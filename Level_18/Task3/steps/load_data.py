from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.datasets import load_digits


@step
def load_digits_data() -> Tuple[np.ndarray, np.ndarray]:
    digits = load_digits()
    return digits.data, digits.target
