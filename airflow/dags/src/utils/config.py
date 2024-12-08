import psycopg2
import os
import logging
import boto3

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PostgresHelper:
    """
    Classe d'aide pour gérer les connexions PostgreSQL et la création de tables.
    """
    db_config = {
        'dbname': os.getenv('POSTGRES_DB_AIRFLOW', 'airflow'),
        'user': os.getenv('POSTGRES_USER_AIRFLOW', 'airflow'),
        'password': os.getenv('POSTGRES_PASSWORD_AIRFLOW', 'airflow'),
        'host': 'postgres-airflow',
        'port': '5432'
    }

    @classmethod
    def connect_to_db(cls):
        """
        Établit une connexion à la base de données et crée la table 'articles' si elle n'existe pas.
        """
        try:
            conn = psycopg2.connect(**cls.db_config)
            logger.info("Database connection successful")

            with conn.cursor() as cur:
                # Création de la table 
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS articles (
                        title TEXT,
                        author_keywords TEXT,
                        abstract TEXT,
                        odd INTEGER
                    );
                ''')
                conn.commit()
                logger.info("Table 'articles' vérifiée/créée avec succès.")
            return conn

        except Exception as e:
            logger.error(f"Erreur lors de la connexion ou de la création de la table : {e}")
            return None



class S3Helper:
    """
    Classe d'aide pour les interactions avec S3 (compatible MinIO).
    """
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT', 'http://minio:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            config=boto3.session.Config(signature_version='s3v4')
        )

    def upload_file(self, file_path, bucket, object_name):
        try:
            with open(file_path, 'rb') as data:
                self.client.upload_fileobj(data, bucket, object_name)
            logger.info(f"File {file_path} uploaded to {bucket}/{object_name}")
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")

    def download_file(self, bucket, object_name, local_path):
        try:
            self.client.download_file(bucket, object_name, local_path)
            logger.info(f"File {bucket}/{object_name} downloaded to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download file from S3: {e}")


class FeaturePaths:
    BASE_PATH = os.getenv('BASE_PATH', '/tmp')

    PATHS = {
        'train': {
            'data_path': 'data/train.parquet', 
            'feature_path': 'features/train_features.parquet'
        },
        'test': {
            'data_path': 'data/test.parquet',
            'feature_path': 'features/test_features.parquet'
        },
        'batch': {
            'data_path': 'data/batch.parquet',
            'feature_path': 'features/batch_features.parquet'
        },
        'vectorizer': {
            'path': 'features/vectorizer.joblib'
        }
    }


class TrainerConfig:
    """
    Configuration du modèle d'entraînement.
    """
    model_name = "Logistic Regression"
    random_state = 42
    train_size = 0.2
    shuffle = True
    params = {
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 100,
        "penalty": 'l2'
    }

class MlFlowConfig:
    """
    Configuration pour MLflow.
    """
    uri = "http://mlflow-webserver:5000"
    experiment_name = "sdg_recommandation"
    artifact_path = "model-artifact"
    registered_model_name = "sdg_model"
