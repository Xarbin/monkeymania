import pandas as pd
from textblob import TextBlob
import praw
from pytrends.request import TrendReq
import requests

# --- Reddit Credentials (REPLACE WITH YOURS) ---
REDDIT_CLIENT_ID = 'ytafakHMYee5sJFAo5VM-w'
REDDIT_CLIENT_SECRET = 'W8j1fWa7an8ERGUPUtXXIS-rl2ZVxA'
REDDIT_USER_AGENT = 'sentiment_scraper/0.1 by xarbin'
REDDIT_USERNAME = 'xarbin'
REDDIT_PASSWORD = 'GRIF9350#'

# --- Initialize Reddit ---
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

# --- Optional Crypto Mapping ---
crypto_map = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'DOGE': 'dogecoin',
    'ADA': 'cardano',
    'SOL': 'solana'
}

def reddit_sentiment(keyword, limit=30):
    try:
        subreddit = reddit.subreddit('all')
        posts = subreddit.search(keyword, limit=limit, sort='new', time_filter='week')
        sentiments = [TextBlob(post.title).sentiment.polarity for post in posts]
        return round(sum(sentiments) / len(sentiments), 3) if sentiments else 0
    except Exception as e:
        return 0

def google_trends_interest(keyword):
    try:
        pytrends = TrendReq()
        pytrends.build_payload([keyword], timeframe='now 7-d')
        data = pytrends.interest_over_time()
        return int(data[keyword].mean()) if not data.empty else 0
    except:
        return 0

def coingecko_social_and_price(crypto_id):
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}'
        response = requests.get(url)
        data = response.json()
        social = data.get('community_data', {}).get('twitter_followers', 0)
        price = data.get('market_data', {}).get('current_price', {}).get('usd', 0)
        return int(social), float(price)
    except:
        return 0, 0

def analyze_row(row):
    ticker = str(row['Ticker']).upper()

    reddit_score = reddit_sentiment(ticker)
    trend_score = google_trends_interest(ticker)

    # Optional: Try to use extra columns
    short_float = float(row.get('Short Float %', 0)) / 100
    rvol = float(row.get('Relative Volume', 1))
    news_sent = float(row.get('News Sentiment', 0))
    institutional_ownership = float(row.get('Institutional Ownership %', 0)) / 100
    atr = float(row.get('ATR', 1))

    # Optional: Crypto data
    crypto_id = crypto_map.get(ticker)
    if crypto_id:
        twitter_followers, crypto_price = coingecko_social_and_price(crypto_id)
    else:
        twitter_followers, crypto_price = 0, 0

    # --- Composite Score ---
    score = reddit_score * 5 + trend_score / 10 + news_sent
    score *= (1 + short_float)
    score *= (rvol if rvol > 0 else 1)
    score *= (1 - institutional_ownership) if institutional_ownership < 1 else 1

    return {
        'Ticker': ticker,
        'Reddit Sentiment': reddit_score,
        'Google Trends': trend_score,
        'Crypto Twitter Followers': twitter_followers,
        'Crypto Price USD': crypto_price,
        'News Sentiment': news_sent,
        'Short Float %': short_float * 100,
        'Relative Volume': rvol,
        'Institutional Ownership %': institutional_ownership * 100,
        'ATR': atr,
        'Score': round(score, 3)
    }

def analyze_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    results = [analyze_row(row) for _, row in df.iterrows()]
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ Saved results to {output_path}")

# Example usage:
# analyze_csv("movers_pre.csv", "sentiment_results.csv")