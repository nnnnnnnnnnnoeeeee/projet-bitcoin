import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data_collector import BitcoinDataCollector
from sentiment_analyzer import EnhancedSentimentAnalyzer
import asyncio
from datetime import datetime, timedelta

class BitcoinDashboard:
    def __init__(self):
        self.data_collector = BitcoinDataCollector()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        
    def run(self):
        st.title("Bitcoin Analysis Dashboard")
        
        # Sidebar pour les contrôles
        st.sidebar.header("Paramètres")
        timeframe = st.sidebar.selectbox(
            "Période d'analyse",
            ["1j", "1s", "1m", "3m", "6m", "1a", "5a"]
        )
        
        # Onglets principaux
        tab1, tab2, tab3 = st.tabs(["Prix & Indicateurs", "Analyse des Sentiments", "Données Historiques"])
        
        with tab1:
            self.display_price_indicators()
            
        with tab2:
            self.display_sentiment_analysis()
            
        with tab3:
            self.display_historical_data()
    
    def display_price_indicators(self):
        # Récupération des données en temps réel
        current_data = self.data_collector.realtime_collection.find().sort([("timestamp", -1)]).limit(1000)
        df = pd.DataFrame(list(current_data))
        
        # Création du graphique
        fig = make_subplots(rows=2, cols=1, shared_xaxis=True)
        
        # Graphique des prix
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            ),
            row=1, col=1
        )
        
        # Graphique du volume
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume']),
            row=2, col=1
        )
        
        st.plotly_chart(fig)
        
    def display_sentiment_analysis(self):
        # Récupération des sentiments récents
        sentiments = self.sentiment_analyzer.sentiment_collection.find().sort([("timestamp", -1)]).limit(100)
        df_sentiment = pd.DataFrame(list(sentiments))
        
        # Création du graphique
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_sentiment['timestamp'],
            y=df_sentiment['composite_sentiment'],
            name="Sentiment Composite"
        ))
        
        st.plotly_chart(fig)
        
    def display_historical_data(self):
        # Affichage des données historiques
        historical_data = self.data_collector.historical_collection.find()
        df_historical = pd.DataFrame(list(historical_data))
        
        st.dataframe(df_historical)

if __name__ == "__main__":
    dashboard = BitcoinDashboard()
    dashboard.run() 