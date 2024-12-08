from src.utils.config import PostgresHelper
import requests
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ScopusAPI:
    """
    Classe pour interagir avec l'API Scopus afin de scraper des articles
    et les insérer dans une base de données PostgreSQL.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.url = 'https://api.elsevier.com/content/search/scopus'
        self.conn = PostgresHelper.connect_to_db()
        self.cur = self.conn.cursor() if self.conn else None

    def scrap_articles(self, query, odd):
        """
        Récupère des articles à partir de l'API Scopus et les stocke dans la base de données.

        Args:
            query (str): Requête à envoyer à l'API Scopus.
            odd (int): Identifiant ou tag pour les articles collectés.
        """
        if not self.cur:
            logger.error("Impossible de scraper des articles : la connexion à la base de données a échoué.")
            return

        params = {'query': query, 'start': 0, 'count': 25}
        total_results = None

        try:
            while total_results is None or params['start'] < total_results:
                headers = {
                    'X-ELS-APIKey': self.api_key,
                    'Accept': 'application/json'
                }
                response = requests.get(self.url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()
                    search_results = data.get('search-results', {})
                    total_results = int(search_results.get('opensearch:totalResults', 0))
                    entries = search_results.get('entry', [])

                    for article in entries:
                        title = article.get('dc:title')
                        author_keywords = article.get('authkeywords', '')
                        abstract = article.get('dc:description', '')
                        self.store_article(title, author_keywords, abstract, odd)

                    params['start'] += 25
                    logger.info(f"Progress: {params['start']} / {total_results}")
                else:
                    logger.error(f"Échec de récupération des données : HTTP {response.status_code}")
                    break

            logger.info("Scraping terminé avec succès.")

        except Exception as e:
            logger.error(f"Une erreur est survenue lors du scraping : {e}")

    def store_article(self, title, author_keywords, abstract, odd):
        """
        Insère un article dans la base de données.

        Args:
            title (str): Titre de l'article.
            author_keywords (str): Mots-clés de l'article.
            abstract (str): Résumé de l'article.
            odd (int): Tag/identifiant associé à l'article.
        """
        if not self.cur:
            logger.error("Impossible d'insérer l'article : la connexion est inexistante.")
            return

        try:
            self.cur.execute('''
                INSERT INTO articles (title, author_keywords, abstract, odd)
                VALUES (%s, %s, %s, %s);
            ''', (title, author_keywords, abstract, odd))
            self.conn.commit()
            logger.info(f"Article inséré : {title[:30]}...") 

        except Exception as e:
            logger.error(f"Échec de l'insertion de l'article '{title}': {e}")

    def test_insertion(self):
        """
        Méthode de test pour insérer des articles d'exemple.
        """
        test_articles = [
            ("Test Title 1", "Keyword1; Keyword2", "Abstract of the first test article.", 5),
            ("Test Title 2", "Keyword3; Keyword4", "Abstract of the second test article.", 7)
        ]

        for article in test_articles:
            self.store_article(*article)

    def close(self):
        """
        Ferme les connexions à la base de données.
        """
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            logger.info("Connexion à la base de données fermée.")


if __name__ == "__main__":
    # Exemple d'utilisation
    API_KEY = '3c86d8982063dee5ceed3bed53187a9f'

    # Exemple de requête
    QUERY = """
        TITLE-ABS-KEY ( ( {extreme poverty} OR {poverty alleviation} OR {poverty eradication} OR {poverty reduction}
        OR {international poverty line} OR ( {financial aid} AND {poverty} ) OR ( {financial aid} AND {poor} )
        OR ( {financial aid} AND {north-south divide} ) OR ( {financial development} AND {poverty} )
        OR {financial empowerment} OR {distributional effect} OR {distributional effects}
        OR {child labor} OR {child labour} OR {development aid} OR {social protection}
        OR {social protection system} OR ( {social protection} AND access ) OR microfinanc* OR micro-financ*
        OR {resilience of the poor} OR ( {safety net} AND {poor} OR {vulnerable} )
        OR ( {economic resource} AND access ) OR ( {economic resources} AND access )
        OR {food bank} OR {food banks} ) ) AND PUBYEAR < 2018 AND PUBYEAR > 2012
    """

    scopus_api = ScopusAPI(API_KEY)
    scopus_api.scrap_articles(QUERY, odd=2012)
    scopus_api.test_insertion()
    scopus_api.close()
