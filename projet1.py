import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import time
import csv

# Part 1: Historical Data Download
def download_historical_data():
    start_date = (datetime.now() - timedelta(days=8*365)).strftime('%Y-%m-%d')
    btc_data = yf.download('BTC-USD', start=start_date)
    btc_data.to_csv('btc_historical_8years.csv')
    print(f"Downloaded BTC price history from {start_date} to {datetime.now().strftime('%Y-%m-%d')}")

# Part 2: Real-time Data Ingestion
def get_realtime_btc(max_retries=3):
    for attempt in range(max_retries):
        try:
            btc = yf.download('BTC-USD', period='1d', interval='1h')
            
            if btc.empty:
                print(f"Attempt {attempt + 1}: No data received, retrying...")
                time.sleep(5)
                continue
                
            current_price = float(btc['Close'].iloc[-1])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open('btc_realtime.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, current_price])
                
            print(f"BTC price recorded at {timestamp}: ${current_price:,.2f}")
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("All attempts to fetch BTC price failed")
                return False

# Part 3: Technical Indicators
class TechnicalIndicators:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df.set_index('Timestamp', inplace=True)
    
    def calculate_sma(self, period=20):
        """Calculate Simple Moving Average"""
        return self.df['Close'].rolling(window=period).mean()
    
    def calculate_ema(self, period=20):
        """Calculate Exponential Moving Average"""
        return self.df['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, short_period=12, long_period=26, signal_period=9):
        """Calculate MACD"""
        short_ema = self.df['Close'].ewm(span=short_period, adjust=False).mean()
        long_ema = self.df['Close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

# Initialize the system
if __name__ == "__main__":
    # Download historical data first
    download_historical_data()
    
    # Schedule hourly data collection
    schedule.every().hour.do(get_realtime_btc)
    
    # Run first collection immediately
    if get_realtime_btc():
        print("Initial BTC price collection successful")
    else:
        print("Initial BTC price collection failed")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)
