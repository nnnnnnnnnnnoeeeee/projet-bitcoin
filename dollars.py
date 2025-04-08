import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time
import csv

# Download historical volume data
def download_historical_volume():
    start_date = (datetime.now() - timedelta(days=8*365)).strftime('%Y-%m-%d')
    btc_data = yf.download('BTC-USD', start=start_date)
    btc_data.to_csv('btc_volume_historical.csv')
    print(f"Downloaded BTC volume history from {start_date} to {datetime.now().strftime('%Y-%m-%d')}")

# Get real-time volume data
def get_realtime_volume(max_retries=3):
    for attempt in range(max_retries):
        try:
            btc = yf.download('BTC-USD', period='1d', interval='1h')
            
            if btc.empty:
                print(f"Attempt {attempt + 1}: No volume data received, retrying...")
                time.sleep(5)
                continue
                
            current_volume = float(btc['Volume'].iloc[-1])
            current_price = float(btc['Close'].iloc[-1])
            volume_usd = current_volume * current_price
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open('btc_volume_realtime.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, current_volume, volume_usd])
                
            print(f"BTC volume recorded at {timestamp}: {current_volume:,.2f} BTC (${volume_usd:,.2f})")
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("All attempts to fetch BTC volume failed")
                return False

# Initialize volume tracking
if __name__ == "__main__":
    # Download historical volume first
    download_historical_volume()
    
    # Schedule hourly volume collection
    schedule.every().hour.do(get_realtime_volume)
    
    # Run first collection immediately
    if get_realtime_volume():
        print("Initial BTC volume collection successful")
    else:
        print("Initial BTC volume collection failed")
