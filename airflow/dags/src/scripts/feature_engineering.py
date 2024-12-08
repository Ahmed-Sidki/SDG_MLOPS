from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.config import S3Helper, FeaturePaths
import pandas as pd
import joblib
import os


class FeatureEngineering:
    """
    Classe pour l'ingénierie des caractéristiques textuelles utilisant TF-IDF.
    """
    def __init__(self, inference_mode, paths):
        self.inference_mode = inference_mode
        self.target_column = 'odd'
        self.s3_helper = S3Helper()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.paths = paths

    def __call__(self, s3_bucket):
        """
        Point d'entrée principal pour générer ou transformer les caractéristiques.
        """
        modes = ['train', 'test'] if not self.inference_mode else ['batch']

        for mode in modes:
            data_path = self.paths[mode]['data_path']
            feature_path = self.paths[mode]['feature_path']
            df = self.read_parquet_from_s3(data_path, s3_bucket)

            if mode == 'batch':
                X = df['text']
                features = self.transform(X, s3_bucket, self.paths['vectorizer']['path'])
                self.write_features_to_s3(features, None, feature_path, mode, s3_bucket)
            else:
                target = df[self.target_column]
                X = df['text']
                features = (
                    self.fit_transform(X, s3_bucket, self.paths['vectorizer']['path'])
                    if mode == 'train' else
                    self.transform(X, s3_bucket, self.paths['vectorizer']['path'])
                )
                self.write_features_to_s3(features, target, feature_path, mode, s3_bucket)

    def fit_transform(self, X, s3_bucket, vectorizer_path):
        """
        Ajuste et transforme les données d'entraînement, puis sauvegarde le vectorizer.
        """
        features = self.tfidf_vectorizer.fit_transform(X)
        local_path = '/tmp/tfidf_vectorizer.joblib'
        joblib.dump(self.tfidf_vectorizer, local_path)
        self.s3_helper.upload_file(local_path, s3_bucket, vectorizer_path)
        return features

    def transform(self, X, s3_bucket, vectorizer_path):
        """
        Transforme les données à l'aide d'un vectorizer existant.
        """
        local_path = '/tmp/tfidf_vectorizer.joblib'
        self.s3_helper.download_file(s3_bucket, vectorizer_path, local_path)
        self.tfidf_vectorizer = joblib.load(local_path)
        return self.tfidf_vectorizer.transform(X)

    def read_parquet_from_s3(self, file_path, s3_bucket):
        """
        Lit un fichier parquet depuis S3.
        """
        local_path = f'/tmp/{os.path.basename(file_path)}'
        self.s3_helper.download_file(s3_bucket, file_path, local_path)
        return pd.read_parquet(local_path)

    def write_features_to_s3(self, features, target, object_name, suffix, s3_bucket):
        """
        Sauvegarde les caractéristiques et cibles dans un fichier puis les télécharge vers S3.
        """
        local_path = f'/tmp/features_{suffix}.joblib'
        features_to_save = (features, target) if target is not None else features
        joblib.dump(features_to_save, local_path)
        self.s3_helper.upload_file(local_path, s3_bucket, object_name)
