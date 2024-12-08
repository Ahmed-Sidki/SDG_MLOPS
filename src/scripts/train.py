from src.utils.config import MlFlowConfig, TrainerConfig, S3Helper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os


class Train:
    """
    Classe pour l'entraînement d'un modèle Logistic Regression.
    """
    def __init__(self, params, model_name, paths):
        self.params = params
        self.model_name = model_name
        self.paths = paths
        self.s3_helper = S3Helper()  # Utilisation de la classe S3Helper

    def __call__(self, s3_bucket):
        """
        Exécute le processus d'entraînement et de validation du modèle.
        """
        # Configuration de MLflow
        mlflow.set_tracking_uri(MlFlowConfig.uri)
        mlflow.set_experiment(MlFlowConfig.experiment_name)

        with mlflow.start_run():
            # Lecture des données depuis S3
            train_features, target_train = self.read_features_from_s3(
                self.paths['train']['feature_path'], s3_bucket
            )
            test_features, target_test = self.read_features_from_s3(
                self.paths['test']['feature_path'], s3_bucket
            )

            # Entraînement du modèle
            model = LogisticRegression(
                random_state=TrainerConfig.random_state,
                verbose=True,
                **self.params
            ).fit(train_features, target_train)

            # Prédictions et évaluation
            y_pred = model.predict(test_features)
            metrics = self.evaluate_model(target_test, y_pred)

            # Enregistrement des métriques et du modèle avec MLflow
            self.log_mlflow(metrics, model)

            return {"mlflow_run_id": mlflow.active_run().info.run_id}

    def read_features_from_s3(self, file_path, s3_bucket):
        """
        Lit les caractéristiques et cibles depuis un fichier stocké sur S3.
        """
        local_path = f'/tmp/{os.path.basename(file_path)}'
        self.s3_helper.download_file(s3_bucket, file_path, local_path)
        features, target = joblib.load(local_path)
        return features, target

    def evaluate_model(self, y_true, y_pred):
        """
        Évalue le modèle et retourne les métriques.
        """
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, y_pred, average='macro')

        print("Classification Report:")
        print(classification_report(y_true, y_pred))

        return {
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }

    def log_mlflow(self, metrics, model):
        """
        Enregistre les paramètres, métriques et le modèle dans MLflow.
        """
        mlflow.log_params(self.params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model", self.model_name)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=MlFlowConfig.artifact_path
        )
