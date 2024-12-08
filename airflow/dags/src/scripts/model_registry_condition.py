from mlflow.tracking import MlflowClient
from src.utils.config import MlFlowConfig
import logging
import mlflow

LOGGER = logging.getLogger(__name__)


class ModelRegistryCondition:
    """
    Condition pour register le model.

    Args:
        criteria (float): Coefficient a appliqué au modèle pour qui'il soit registré.
        metric (str): métrique de réference ca peut etre  `["precision", "recall", "roc_auc"]`.
    """

    def __init__(self, criteria: float, metric: str) -> None:
        if metric not in ["roc_auc", "precision", "recall"]:
            raise ValueError("Metric must be one of ['roc_auc', 'precision', 'recall']")
        self.criteria = criteria
        self.metric = metric

    def __call__(self, mlflow_run_id: str) -> None:
        """
        Compare the metric from the last run to the model in the registry.
        If `metric_run > registered_metric * (1 + self.criteria)`, then the model is registered.
        """
        LOGGER.info(f"Run_id: {mlflow_run_id}")
        mlflow.set_tracking_uri(MlFlowConfig.uri)

        # Récupérer les métriques de la dernière exécution
        run = mlflow.get_run(run_id=mlflow_run_id)
        metric_value = run.data.metrics[self.metric]
        LOGGER.info(f"Metric value from the current run: {metric_value}")

        # Initialiser le client MLflow
        client = MlflowClient()

        # Vérifier les modèles déjà enregistrés
        registered_models = client.search_registered_models(
            filter_string=f"name='{MlFlowConfig.registered_model_name}'"
        )

        if not registered_models:
            client.create_registered_model(MlFlowConfig.registered_model_name)
            client.create_model_version(
                name=MlFlowConfig.registered_model_name,
                source=f"runs:/{mlflow_run_id}/{MlFlowConfig.artifact_path}",
                run_id=mlflow_run_id
            )
            LOGGER.info("No models found in the registry. New model registered.")
            return

        # Comparer avec la dernière version du modèle
        latest_version = registered_models[0].latest_versions[-1]
        registered_model_run = mlflow.get_run(latest_version.run_id)
        registered_metric_value = registered_model_run.data.metrics[self.metric]

        LOGGER.info(f"Registered metric value: {registered_metric_value}")

        if metric_value > registered_metric_value * (1 + self.criteria):
            client.create_model_version(
                name=MlFlowConfig.registered_model_name,
                source=f"runs:/{mlflow_run_id}/{MlFlowConfig.artifact_path}",
                run_id=mlflow_run_id
            )
            LOGGER.info("New model version registered due to improved metric.")
        else:
            LOGGER.info("Current model does not meet the criteria for registration.")
