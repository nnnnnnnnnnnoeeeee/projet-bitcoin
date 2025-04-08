import yfinance as yf
import csv
from datetime import datetime
import schedule
import time
import pandas as pd

def get_realtime_btc(max_retries=3):
    for attempt in range(max_retries):
        try:
            # Fetch latest BTC price with a slightly larger period to ensure we get data
            btc = yf.download('BTC-USD', period='1d', interval='1h')
            
            if btc.empty:
                print(f"Attempt {attempt + 1}: No data received, retrying...")
                time.sleep(5)  # Wait 5 seconds before retrying
                continue
                
            current_price = float(btc['Close'].iloc[-1])  # Convert to float explicitly
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Append to CSV file
            with open('btc_realtime.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, current_price])
                
            print(f"BTC price recorded at {timestamp}: ${current_price:,.2f}")
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5 seconds before retrying
            else:
                print("All attempts to fetch BTC price failed")
                return False

# Schedule data collection every hour
schedule.every().hour.do(get_realtime_btc)

# Run the first collection immediately
if get_realtime_btc():
    print("Initial BTC price collection successful")
else:
    print("Initial BTC price collection failed")

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)
