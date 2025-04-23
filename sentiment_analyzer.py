import praw
import pandas as pd
from textblob import TextBlob
import tweepy
from newsapi import NewsApi
from googleapiclient.discovery import build
import os
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any
import pymongo

class EnhancedSentimentAnalyzer:
    def __init__(self):
        # Twitter setup
        self.twitter_auth = tweepy.OAuthHandler(
            os.getenv('TWITTER_API_KEY'),
            os.getenv('TWITTER_API_SECRET')
        )
        self.twitter_auth.set_access_token(
            os.getenv('TWITTER_ACCESS_TOKEN'),
            os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )
        self.twitter_api = tweepy.API(self.twitter_auth)
        
        # Reddit setup
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent="bitcoin_sentiment_analyzer"
        )
        
        # Google News setup
        self.news_api = NewsApi(os.getenv('NEWS_API_KEY'))
        
        # MongoDB setup for sentiment storage
        self.mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.mongo_client["bitcoin_analysis"]
        self.sentiment_collection = self.db["sentiment_data"]

    async def analyze_twitter_sentiment(self) -> Dict[str, float]:
        """Analyse avancée du sentiment Twitter"""
        tweets = tweepy.Cursor(
            self.twitter_api.search_tweets,
            q="bitcoin OR btc -filter:retweets",
            lang="en",
            tweet_mode="extended"
        ).items(1000)
        
        sentiments = []
        for tweet in tweets:
            analysis = TextBlob(tweet.full_text)
            sentiments.append({
                'polarity': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity,
                'followers': tweet.user.followers_count
            })
        
        # Sentiment pondéré par le nombre de followers
        weighted_sentiment = sum(s['polarity'] * s['followers'] for s in sentiments) / \
                           sum(s['followers'] for s in sentiments)
        
        return {
            'weighted_sentiment': weighted_sentiment,
            'average_sentiment': sum(s['polarity'] for s in sentiments) / len(sentiments),
            'volume': len(sentiments)
        }

    async def analyze_reddit_sentiment(self) -> Dict[str, float]:
        """Analyse du sentiment Reddit"""
        subreddits = ['bitcoin', 'cryptocurrency', 'btc']
        all_sentiments = []
        
        for subreddit_name in subreddits:
            subreddit = self.reddit.subreddit(subreddit_name)
            for post in subreddit.hot(limit=100):
                analysis = TextBlob(post.title + " " + post.selftext)
                all_sentiments.append({
                    'polarity': analysis.sentiment.polarity,
                    'score': post.score
                })
        
        weighted_sentiment = sum(s['polarity'] * s['score'] for s in all_sentiments) / \
                           sum(s['score'] for s in all_sentiments)
        
        return {
            'weighted_sentiment': weighted_sentiment,
            'volume': len(all_sentiments)
        }

    async def analyze_google_news(self) -> Dict[str, float]:
        """Analyse du sentiment des actualités Google"""
        news_results = self.news_api.get_everything(
            q='bitcoin OR cryptocurrency',
            language='en',
            sort_by='relevancy',
            page_size=100
        )
        
        sentiments = []
        for article in news_results['articles']:
            if article['description']:
                analysis = TextBlob(article['title'] + " " + article['description'])
                sentiments.append(analysis.sentiment.polarity)
        
        return {
            'average_sentiment': sum(sentiments) / len(sentiments),
            'volume': len(sentiments)
        }

    async def aggregate_sentiments(self) -> Dict[str, Any]:
        """Agrège tous les sentiments"""
        twitter_sentiment = await self.analyze_twitter_sentiment()
        reddit_sentiment = await self.analyze_reddit_sentiment()
        news_sentiment = await self.analyze_google_news()
        
        timestamp = datetime.now()
        
        aggregate_data = {
            'timestamp': timestamp,
            'twitter': twitter_sentiment,
            'reddit': reddit_sentiment,
            'news': news_sentiment,
            'composite_sentiment': (
                twitter_sentiment['weighted_sentiment'] * 0.4 +
                reddit_sentiment['weighted_sentiment'] * 0.3 +
                news_sentiment['average_sentiment'] * 0.3
            )
        }
        
        # Store in MongoDB
        self.sentiment_collection.insert_one(aggregate_data)
        
        return aggregate_data 