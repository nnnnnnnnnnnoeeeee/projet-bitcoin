def calculate_bollinger_bands(self, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = self.df['Close'].rolling(window=period).mean()
    std = self.df['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic_oscillator(self, period=14):
    """Calculate Stochastic Oscillator"""
    low_min = self.df['Low'].rolling(window=period).min()
    high_max = self.df['High'].rolling(window=period).max()
    k = 100 * ((self.df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=3).mean()
    return k, d

def calculate_atr(self, period=14):
    """Calculate Average True Range"""
    high_low = self.df['High'] - self.df['Low']
    high_close = abs(self.df['High'] - self.df['Close'].shift())
    low_close = abs(self.df['Low'] - self.df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_obv(self):
    """Calculate On Balance Volume"""
    obv = []
    obv.append(0)
    
    for i in range(1, len(self.df.index)):
        if self.df['Close'].iloc[i] > self.df['Close'].iloc[i-1]:
            obv.append(obv[-1] + self.df['Volume'].iloc[i])
        elif self.df['Close'].iloc[i] < self.df['Close'].iloc[i-1]:
            obv.append(obv[-1] - self.df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    return pd.Series(obv, index=self.df.index)

def calculate_moving_averages(self):
    """Calculate Multiple Moving Averages"""
    ma_periods = [5, 10, 20, 50, 200]  # Common MA periods
    mas = {}
    for period in ma_periods:
        mas[f'SMA_{period}'] = self.df['Close'].rolling(window=period).mean()
        mas[f'EMA_{period}'] = self.df['Close'].ewm(span=period, adjust=False).mean()
    return mas
    def create_visualization(self):
        """Create visualization dashboard for technical indicators"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Price and Moving Averages
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax1.plot(self.df.index, self.df['Close'], label='Price', alpha=0.8)
        mas = self.calculate_moving_averages()
        for ma_name, ma_values in mas.items():
            ax1.plot(self.df.index, ma_values, label=ma_name, alpha=0.6)
        ax1.set_title('Price and Moving Averages')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True)
        
        # RSI
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        rsi = self.calculate_rsi()
        ax2.plot(self.df.index, rsi)
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_title('RSI')
        ax2.grid(True)
        
        # MACD
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        macd, signal = self.calculate_macd()
        ax3.plot(self.df.index, macd, label='MACD')
        ax3.plot(self.df.index, signal, label='Signal')
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True)
        
        # Bollinger Bands
        ax4 = plt.subplot2grid((3, 2), (2, 0))
        upper, middle, lower = self.calculate_bollinger_bands()
        ax4.plot(self.df.index, self.df['Close'], label='Price')
        ax4.plot(self.df.index, upper, label='Upper Band')
        ax4.plot(self.df.index, middle, label='Middle Band')
        ax4.plot(self.df.index, lower, label='Lower Band')
        ax4.set_title('Bollinger Bands')
        ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax4.grid(True)
        
        # Volume
        ax5 = plt.subplot2grid((3, 2), (2, 1))
        ax5.bar(self.df.index, self.df['Volume'])
        ax5.set_title('Volume')
        ax5.grid(True)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    def display_current_signals(self):
        """Display current trading signals based on indicators"""
        # Get latest values
        current_price = self.df['Close'].iloc[-1]
        rsi = self.calculate_rsi().iloc[-1]
        macd, signal = self.calculate_macd()
        macd_latest = macd.iloc[-1]
        signal_latest = signal.iloc[-1]
        upper, middle, lower = self.calculate_bollinger_bands()
        
        # Print analysis
        print("\n=== Technical Analysis Summary ===")
        print(f"Current Price: ${current_price:.2f}")
        print(f"\nRSI ({rsi:.2f}):")
        if rsi > 70:
            print("⚠️ Overbought condition")
        elif rsi < 30:
            print("⚠️ Oversold condition")
        else:
            print("✓ Normal range")
            
        print(f"\nMACD:")
        if macd_latest > signal_latest:
            print("↗️ Bullish signal")
        else:
            print("↘️ Bearish signal")
            
        print(f"\nBollinger Bands:")
        if current_price > upper.iloc[-1]:
            print("⚠️ Price above upper band - potential overbought")
        elif current_price < lower.iloc[-1]:
            print("⚠️ Price below lower band - potential oversold")
        else:
            print("✓ Price within bands")
