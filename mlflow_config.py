import dagshub
import mlflow


def setup_mlflow():
    """
    Connect MLflow to the DagsHub repository
    and configure the experiment.
    """

    dagshub.init(
        repo_owner="rajsengar1804",
        repo_name="emi-prediction",
        mlflow=True
    )

    mlflow.set_experiment("EMI Classification")