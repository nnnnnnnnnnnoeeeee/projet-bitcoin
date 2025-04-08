import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import schedule
import time
import csv

# Part 1: Real-time data ingestion
def get_realtime_btc(max_retries=3):
    for attempt in range(max_retries):
        try:
            # Fetch latest BTC price
            btc = yf.download('BTC-USD', period='1d', interval='1h')
            
            if btc.empty:
                print(f"Attempt {attempt + 1}: No data received, retrying...")
                time.sleep(5)
                continue
                
            current_price = btc['Close'].iloc[-1].item()  # Using .item() to convert to Python scalar
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
                time.sleep(5)
            else:
                print("All attempts to fetch BTC price failed")
                return False

# Part 2: Technical Indicators Calculation
class TechnicalIndicators:
    def __init__(self, data):
        self.df = pd.read_csv(data)
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
        """Calculate MACD (Moving Average Convergence Divergence)"""
        short_ema = self.df['Close'].ewm(span=short_period, adjust=False).mean()
        long_ema = self.df['Close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(period)
        std = self.df['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

# Schedule data collection every hour
schedule.every().hour.do(get_realtime_btc)

if __name__ == "__main__":
    # Run initial data collection
    if get_realtime_btc():
        print("Initial BTC price collection successful")
    else:
        print("Initial BTC price collection failed")
        
    # Keep the script running for data collection
    while True:
        schedule.run_pending()
        time.sleep(60)
        
        # Example of calculating technical indicators (run periodically)
        if datetime.now().minute == 0:  # Run at the start of each hour
            try:
                indicators = TechnicalIndicators('btc_realtime.csv')
                
                # Calculate various indicators
                rsi = indicators.calculate_rsi()
                macd, signal = indicators.calculate_macd()
                upper, middle, lower = indicators.calculate_bollinger_bands()
                
                print("\nLatest Technical Indicators:")
                print(f"RSI: {rsi.iloc[-1]:.2f}")
                print(f"MACD: {macd.iloc[-1]:.2f}")
                print(f"Bollinger Bands: Upper={upper.iloc[-1]:.2f}, Middle={middle.iloc[-1]:.2f}, Lower={lower.iloc[-1]:.2f}")
            except Exception as e:
                print(f"Error calculating indicators: {str(e)}")


                # Create a simple web interface using Streamlit
                import streamlit as st
                
                def create_dashboard():
                    """Create a Streamlit dashboard to display technical indicators"""
                    st.title('Bitcoin Technical Analysis Dashboard')
                    
                    # Display current price
                    st.header('Current BTC Price')
                    current_price = indicators.df['Close'].iloc[-1]
                    st.metric("BTC/USD", f"${current_price:,.2f}")
                    
                    # Technical Indicators Section
                    st.header('Technical Indicators')
                    
                    # Create columns for indicators
                    col1, col2, col3 = st.columns(3)
                    
                    # RSI
                    with col1:
                        st.subheader('RSI')
                        rsi_value = rsi.iloc[-1]
                        st.metric("RSI Value", f"{rsi_value:.2f}")
                        if rsi_value > 70:
                            st.warning("Overbought")
                        elif rsi_value < 30:
                            st.warning("Oversold")
                        else:
                            st.success("Normal Range")
                    
                    # MACD
                    with col2:
                        st.subheader('MACD')
                        macd_value = macd.iloc[-1]
                        signal_value = signal.iloc[-1]
                        st.metric("MACD", f"{macd_value:.2f}")
                        st.metric("Signal", f"{signal_value:.2f}")
                        if macd_value > signal_value:
                            st.success("Bullish Signal")
                        else:
                            st.warning("Bearish Signal")
                    
                    # Bollinger Bands
                    with col3:
                        st.subheader('Bollinger Bands')
                        st.metric("Upper Band", f"${upper.iloc[-1]:,.2f}")
                        st.metric("Middle Band", f"${middle.iloc[-1]:,.2f}")
                        st.metric("Lower Band", f"${lower.iloc[-1]:,.2f}")
                        
                        if current_price > upper.iloc[-1]:
                            st.warning("Price above upper band - Potential overbought")
                        elif current_price < lower.iloc[-1]:
                            st.warning("Price below lower band - Potential oversold")
                        else:
                            st.success("Price within bands")
                    
                    # Chart Section
                    st.header('Price Chart')
                    chart_data = indicators.df['Close'].tail(100)
                    st.line_chart(chart_data)
                
                # Run the dashboard
                if datetime.now().minute == 0:  # Update dashboard hourly
                    create_dashboard()
