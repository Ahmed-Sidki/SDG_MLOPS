from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.scripts.data_ingestion import ScopusAPI
from src.scripts.preprocess import Preprocess
from src.scripts.feature_engineering import FeatureEngineering
from src.scripts.train import Train
from src.utils.config import FeaturePaths, TrainerConfig
from src.scripts.model_registry_condition import ModelRegistryCondition

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

paths = FeaturePaths.PATHS
model = TrainerConfig.model_name
params = TrainerConfig.params
s3_bucket = 'sdg'

with DAG(
    dag_id='article_processing_dag',
    default_args=default_args,
    description='DAG to fetch and preprocess articles',
    schedule_interval=timedelta(days=1),
    catchup=False
) as dag:

    def fetch_and_store_articles(**kwargs):
        """
        Initialise ScopusAPI et stocke des articles de test.
        """
        try:
            api_key = 'your_api_key_here'
            scopus_api = ScopusAPI(api_key)
            scopus_api.test_insertion()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch articles: {e}")

    def preprocess_and_upload(**kwargs):
        """
        Initialise Preprocess et exécute le prétraitement.
        """
        try:
            preprocessor = Preprocess(inference_mode=False)
            preprocessor(s3_bucket)
        except Exception as e:
            raise RuntimeError(f"Failed during preprocessing: {e}")

    def upload_features(**kwargs):
        """
        Initialise FeatureEngineering et génère les features.
        """
        try:
            feature_engineer = FeatureEngineering(inference_mode=False, paths=paths)
            feature_engineer(s3_bucket)
        except Exception as e:
            raise RuntimeError(f"Failed to upload features: {e}")

    def train_model(**kwargs):
        """
        Initialise Train et entraîne le modèle.
        """
        try:
            trainer = Train(params=params, model_name=model, paths=paths)
            #trainer(s3_bucket)
            result = trainer(s3_bucket)
            #Condition pour enregistrer le modèle
            condition_step = ModelRegistryCondition(criteria=0.05, metric="roc_auc") 
            condition_step(result["mlflow_run_id"])
        except Exception as e:
            raise RuntimeError(f"Failed to train the model: {e}")

    fetch_articles_task = PythonOperator(
        task_id='fetch_and_store_articles',
        python_callable=fetch_and_store_articles
    )

    preprocess_upload_task = PythonOperator(
        task_id='preprocess_and_upload',
        python_callable=preprocess_and_upload
    )

    upload_features_task = PythonOperator(
        task_id='upload_features',
        python_callable=upload_features
    )

    train_task = PythonOperator(
        task_id='train',
        python_callable=train_model
    )

    # Définition de la dépendance des tâches
    fetch_articles_task >> preprocess_upload_task >> upload_features_task >> train_task
