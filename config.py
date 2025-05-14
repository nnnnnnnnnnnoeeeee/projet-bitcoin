import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # API Keys (à mettre dans un .env en production)
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    
    # Paramètres de trading
    TRADING_PAIRS: List[str] = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    DEFAULT_TIMEFRAME: str = "1h"
    
    # Paramètres des indicateurs
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: int = 2
    
    # Paramètres de cache
    CACHE_EXPIRATION: int = 300  # 5 minutes
    
    # Paramètres ML
    PREDICTION_WINDOW: int = 24  # heures
    FEATURES: List[str] = [
        "rsi", "macd", "bb_upper", "bb_lower", 
        "volume_sma", "close_sma_20", "close_sma_50"
    ] 