import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Calculate the start date (8 years ago from today)
start_date = (datetime.now() - timedelta(days=8*365)).strftime('%Y-%m-%d')

# Download BTC historical data
btc_data = yf.download('BTC-USD', start=start_date)

# Save to CSV file
btc_data.to_csv('btc_historical_8years.csv')

print(f"Downloaded BTC price history from {start_date} to {datetime.now().strftime('%Y-%m-%d')}")
# Import yfinance package
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance package...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf
    import schedule
    import time
    import csv
    from datetime import datetime

    def get_realtime_btc():
        # Fetch latest BTC price
        btc = yf.download('BTC-USD', period='1h', interval='1h')
        current_price = btc['Close'].iloc[-1]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Append to CSV file
        with open('btc_realtime.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, current_price])
            
        print(f"BTC price recorded at {timestamp}: ${current_price:.2f}")

    # Schedule data collection every hour
    schedule.every().hour.do(get_realtime_btc)

    # Run the first collection immediately
    get_realtime_btc()

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)
        # Function to calculate technical indicators
        def calculate_technical_indicators(price_data):
            # Calculate various technical indicators
            # Moving averages
            price_data['SMA_20'] = price_data['Close'].rolling(window=20).mean()
            price_data['SMA_50'] = price_data['Close'].rolling(window=50).mean()
            price_data['SMA_200'] = price_data['Close'].rolling(window=200).mean()
            
            # RSI (Relative Strength Index)
            delta = price_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            price_data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            price_data['BB_middle'] = price_data['Close'].rolling(window=20).mean()
            price_data['BB_upper'] = price_data['BB_middle'] + 2 * price_data['Close'].rolling(window=20).std()
            price_data['BB_lower'] = price_data['BB_middle'] - 2 * price_data['Close'].rolling(window=20).std()
            
            # Save indicators to CSV
            price_data.to_csv('btc_indicators.csv')
            
            return price_data

        # Read the real-time data and calculate indicators
        def update_indicators():
            try:
                # Read the real-time price data
                price_data = pd.read_csv('btc_realtime.csv', names=['Timestamp', 'Close'])
                price_data['Timestamp'] = pd.to_datetime(price_data['Timestamp'])
                price_data.set_index('Timestamp', inplace=True)
                
                # Calculate and save indicators
                indicators_data = calculate_technical_indicators(price_data)
                print(f"Technical indicators updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                print(f"Error updating indicators: {str(e)}")

        # Add indicator calculation to hourly schedule
        schedule.every().hour.do(update_indicators)
