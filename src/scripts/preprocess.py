from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.utils.config import PostgresHelper, TrainerConfig, S3Helper
from sklearn.model_selection import train_test_split

import pandas as pd
import re
import string
import logging

# Configuration du logger
logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self, inference_mode, use_lemmatization=False):
        """
        Classe de prétraitement des données textuelles.

        :param inference_mode: True pour inférence (batch), False pour entraînement.
        :param use_lemmatization: Active la lemmatisation au lieu du stemming.
        """
        self.inference_mode = inference_mode
        self.use_lemmatization = use_lemmatization
        self.conn = PostgresHelper.connect_to_db()
        self.s3_helper = S3Helper()
        self.stop_words = set(stopwords.words("english"))

        # Choix entre stemming ou lemmatisation
        if use_lemmatization:
            self.processor = WordNetLemmatizer()
        else:
            self.processor = PorterStemmer()

    def __call__(self, s3_bucket):
        """
        Point d'entrée principal pour le prétraitement.
        """
        try:
            logger.info("Fetching data from database...")
            df = self.fetch_data()

            logger.info("Starting preprocessing...")
            preprocessed_df = self.preprocess(df)

            if not self.inference_mode:
                logger.info("Splitting data into train and test sets...")
                train_df, test_df = train_test_split(
                    preprocessed_df, 
                    test_size=0.2, 
                    random_state=TrainerConfig.random_state,
                    stratify=preprocessed_df['odd']  
                )

                self.save_and_upload(train_df, "train", s3_bucket)
                self.save_and_upload(test_df, "test", s3_bucket)
            else:
                self.save_and_upload(preprocessed_df, "batch", s3_bucket)

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
        finally:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed.")

    def save_and_upload(self, df, suffix, s3_bucket):
        """
        Sauvegarde les données localement et les télécharge vers S3.
        """
        parquet_path = f'/tmp/preprocessed_data_{suffix}.parquet'
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved preprocessed data locally: {parquet_path}")

        self.s3_helper.upload_file(parquet_path, s3_bucket, f'data/{suffix}.parquet')

    def fetch_data(self):
        """
        Récupère les données de la table PostgreSQL.
        """
        query = "SELECT * FROM articles"
        return pd.read_sql(query, self.conn)

    def clean_text(self, text):
        """
        Nettoie le texte : URL, HTML, emojis, ponctuation, stopwords, stemming/lemmatisation.
        """
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove emojis
        text = re.sub(
            r"[" 
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", "", text, flags=re.UNICODE
        )

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove stopwords and apply stemming/lemmatisation
        words = [
            self.processor.lemmatize(word.lower()) if self.use_lemmatization else self.processor.stem(word.lower())
            for word in text.split() if word.lower() not in self.stop_words
        ]
        return " ".join(words)

    def preprocess(self, df: pd.DataFrame):
        """
        Applique les transformations de nettoyage de texte sur un DataFrame.
        """
        df.fillna(" ", inplace=True)
        df["text"] = df["title"] + " " + df["author_keywords"] + " " + df["abstract"]

        logger.info("Cleaning text data...")
        df["text"] = df["text"].apply(self.clean_text)

        logger.info("Preprocessing completed.")
        return df
