import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime
import numpy as np

def calculate_rsi(data, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def get_realtime_btc_data():
    """Get real-time Bitcoin data"""
    btc_data = yf.download('BTC-USD', period='1d', interval='1m')
    return btc_data

# Create output directory if it doesn't exist
output_dir = 'indicateurs_techniques'
os.makedirs(output_dir, exist_ok=True)

while True:
    try:
        # Get real-time data
        df = get_realtime_btc_data()
        
        if len(df) >= 14:  # Need at least 14 periods for RSI calculation
            # Calculate indicators
            try:
                rsi = calculate_rsi(df)
                macd, signal = calculate_macd(df)
            except Exception as e:
                print(f"Error calculating indicators: {str(e)}")
                continue
            
            # Create DataFrame with results
            indicators_df = pd.DataFrame({
                'Timestamp': df.index,
                'Close': df['Close'],
                'RSI': rsi.values.flatten() if rsi is not None and rsi.size > 0 else np.nan,
                'MACD': macd.values.flatten() if macd is not None and macd.size > 0 else np.nan,
                'Signal': signal.values.flatten() if signal is not None and signal.size > 0 else np.nan
            })
            
            # Save to CSV
            output_file = os.path.join(output_dir, 'btc_indicateurs_techniques.csv')
            indicators_df.to_csv(output_file, index=False)
            
            print(f"\nLatest indicators saved to {output_file}")
            print(f"Timestamp: {datetime.now()}")
            print(f"RSI: {rsi.iloc[-1]:.2f}")
            print(f"MACD: {macd.iloc[-1]:.2f}")
            print(f"Signal: {signal.iloc[-1]:.2f}")
            
        else:
            print("Not enough data points to calculate indicators")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    time.sleep(60)  # Wait 60 seconds before next update
