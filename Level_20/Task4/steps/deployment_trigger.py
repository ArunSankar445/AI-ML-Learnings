# steps/deployment_trigger.py
from zenml.steps import step


@step
def deployment_trigger(r2: float) -> bool:
    return r2 >= 0.7
