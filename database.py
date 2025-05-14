import sqlite3
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path='bitcoin_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise la base de données"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Table pour les données de marché
        c.execute('''CREATE TABLE IF NOT EXISTS market_data
                    (timestamp TEXT, open REAL, high REAL, low REAL, 
                     close REAL, volume REAL)''')
        
        # Table pour les sentiments
        c.execute('''CREATE TABLE IF NOT EXISTS sentiment_data
                    (timestamp TEXT, fear_greed_value REAL, 
                     fear_greed_classification TEXT, market_sentiment REAL,
                     google_trends_value REAL)''')
        
        # Table pour Google Trends
        c.execute('''CREATE TABLE IF NOT EXISTS google_trends
                    (timestamp TEXT, bitcoin REAL, crypto REAL, BTC REAL)''')
        
        conn.commit()
        conn.close()
    
    def get_latest_data(self):
        """Récupère les dernières données"""
        conn = sqlite3.connect(self.db_path)
        
        market_data = pd.read_sql('SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 1000', conn)
        sentiment_data = pd.read_sql('SELECT * FROM sentiment_data ORDER BY timestamp DESC LIMIT 1', conn)
        trends_data = pd.read_sql('SELECT * FROM google_trends ORDER BY timestamp DESC LIMIT 24', conn)
        
        conn.close()
        return market_data, sentiment_data, trends_data

    def save_bulk_data(self, df):
        """
        Stocke un DataFrame d'historique dans la base.
        """
        with self.conn:
            df.to_sql('bitcoin_data', self.conn, if_exists='append', index=False)