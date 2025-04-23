import pandas as pd
import ccxt
import pymongo
from datetime import datetime, timedelta
import time
from typing import Dict, Any
import logging

class BitcoinDataCollector:
    def __init__(self):
        # Initialisation de la connexion MongoDB
        self.mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.mongo_client["bitcoin_analysis"]
        self.historical_collection = self.db["historical_data"]
        self.realtime_collection = self.db["realtime_data"]
        
        # Initialisation de l'exchange (Binance par défaut)
        self.exchange = ccxt.binance()
        
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_historical_data(self, years: int = 5) -> pd.DataFrame:
        """Récupère les données historiques des 5 dernières années"""
        try:
            start_date = int((datetime.now() - timedelta(days=years*365)).timestamp() * 1000)
            all_data = []
            
            while start_date < datetime.now().timestamp() * 1000:
                self.logger.info(f"Fetching data from {datetime.fromtimestamp(start_date/1000)}")
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol='BTC/USDT',
                    timeframe='1h',
                    since=start_date,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                start_date = ohlcv[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)  # Respect rate limits
                
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise

    def store_historical_data(self, data: pd.DataFrame):
        """Stocke les données historiques dans MongoDB"""
        try:
            records = data.to_dict('records')
            self.historical_collection.insert_many(records)
            self.logger.info(f"Stored {len(records)} historical records")
        except Exception as e:
            self.logger.error(f"Error storing historical data: {e}")
            raise

    async def realtime_data_stream(self):
        """Génère un flux de données en temps réel"""
        while True:
            try:
                ticker = self.exchange.fetch_ticker('BTC/USDT')
                data = {
                    'timestamp': datetime.now(),
                    'price': ticker['last'],
                    'volume': ticker['quoteVolume'],
                    'high': ticker['high'],
                    'low': ticker['low']
                }
                
                await self.store_realtime_data(data)
                await asyncio.sleep(1)  # Mise à jour toutes les secondes
                
            except Exception as e:
                self.logger.error(f"Error in realtime stream: {e}")
                await asyncio.sleep(5)  # Attente avant nouvelle tentative

    async def store_realtime_data(self, data: Dict[str, Any]):
        """Stocke les données en temps réel"""
        try:
            self.realtime_collection.insert_one(data)
        except Exception as e:
            self.logger.error(f"Error storing realtime data: {e}") 