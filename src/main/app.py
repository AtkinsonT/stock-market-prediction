import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import pytz
import yfinance as yf
from flask import Flask, render_template, request, jsonify

from sentiment.finbert import FinbertSentimentAnalysis
from sentiment.vader import VaderSentimentAnalysis
from src.historical.historical_backtesting import HistoricalPrediction
from src.util.mboum_api import API

BST = pytz.timezone('Europe/London')
user_sentiment = FinbertSentimentAnalysis()

app = Flask(__name__, template_folder="../../data/templates")


def get_live_price_history(ticker: str, earliest_datetime: pd.Timestamp) -> pd.DataFrame:
    # Fetch live price history for a ticker up to a certain date
    return API.get_price_history(ticker, earliest_datetime)


def get_news_articles(ticker) -> pd.DataFrame:
    # Fetch and return news articles for a specific ticker
    user_sentiment.set_symbol(ticker)
    return API.get_live_news(ticker)


def score_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    # Score the sentiment of news articles using Finbert model
    user_sentiment.set_data(news_df)
    return user_sentiment.calculate_sentiment_scores(news_df)


def plot_sentiment_scores(ticker: str) -> go.Figure:
    # Plot sentiment scores using plotly
    return user_sentiment.plot_sentiment()


def get_earliest_date(df: pd.DataFrame) -> Any | None:
    # Extract the earliest date from a DataFrame
    # Check if the DataFrame is empty
    if df.empty:
        print("df is empty")
        return None
    # Ensure the 'Date Time' column exists
    if 'Date Time' not in df.columns:
        raise ValueError("DataFrame does not contain a 'Date Time' column")
    # Assuming 'Date Time' is already in a suitable format
    date = df['Date Time'].iloc[-1]
    py_date = date.to_pydatetime()
    return py_date.replace(tzinfo=BST)


def plot_price(df, ticker) -> go.Figure:
    # Plot prices for a ticker using plotly
    if not {'Date Time', 'Price'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'Date Time' and 'Price' columns.")

    fig = px.line(data_frame=df, x='Date Time', y='Price', title=f"{ticker} Price",
                  labels={'Date Time': 'Date and Time', 'Price': 'Price (USD)'})
    return fig


def get_business_name(ticker):
    # Fetch the long name of a business given a ticker using yfinance
    tickername = yf.Ticker(ticker)
    try:
        info = tickername.info
        return info['longName']
    except KeyError:
        return None


def evaluate_historical_accuracy(ticker):
    # Evaluate historical accuracy of predictions for a given ticker
    hist_stock_prediction = HistoricalPrediction(ticker)
    historical_scores = hist_stock_prediction.train_and_optimize()
    return historical_scores


def calculate_pearson_correlation(sentiment_df, prices_df):
    # Calculate Pearson correlation between sentiment scores and stock prices
    # Remove timezone info to ensure consistency
    sentiment_df['Date Time'] = sentiment_df['Date Time'].dt.tz_localize(None)
    sentiment_df['Date Time'] = sentiment_df['Date Time'].dt.round('1min')
    prices_df['Date Time'] = prices_df['Date Time'].dt.tz_localize(None)
    prices_df['Date Time'] = prices_df['Date Time'].dt.round('1min')
    # time range for merging
    sentiment_df['start_time'] = sentiment_df['Date Time'] - pd.Timedelta(minutes=5)
    sentiment_df['end_time'] = sentiment_df['Date Time'] + pd.Timedelta(minutes=5)
    # Merge using conditional that checks price timestamps fall within sentiment time range
    merged_df = pd.merge(sentiment_df.assign(key=1), prices_df.assign(key=1), on='key').drop('key', axis=1)
    merged_df = merged_df[
        (merged_df['Date Time_y'] >= merged_df['start_time']) & (merged_df['Date Time_y'] <= merged_df['end_time'])]
    # Drop extra columns used for merging
    merged_df.drop(['start_time', 'end_time'], axis=1, inplace=True)
    # cases with insufficient data
    if len(merged_df) < 2:
        print("Not enough data to calculate Pearson correlation.")
        return None
    # Ensure no NaN values which can affect correlation calculation
    merged_df.dropna(subset=['combined_sentiment_score', 'Price'], inplace=True)
    # Calculate correlation
    correlation = merged_df['combined_sentiment_score'].corr(merged_df['Price'])
    return correlation


def clean_interval_price(df):
    # Remove entries where the price is zero
    df = df[df['Price'] != 0.0]
    return df


def headline_to_link(df: pd.DataFrame) -> pd.DataFrame:
    # Transform headlines to clickable links
    df.insert(2, 'Headline', df['title + link'])
    df.drop(columns=['title + link', 'title'], inplace=True, axis=1)
    return df


@app.route('/', methods=['GET'])
def index():
    return render_template('menu.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    global user_sentiment
    # Main analysis endpoint to process requests
    try:
        ticker = request.form['ticker'].strip().upper()
        inputted_sentiment = request.form['analysis']
        historical_information = evaluate_historical_accuracy(ticker)
        business_name = get_business_name(ticker)
        news_df = get_news_articles(ticker)

        if inputted_sentiment == 'FinBERT':
            user_sentiment = FinbertSentimentAnalysis()
        elif inputted_sentiment == 'VADER':
            user_sentiment = VaderSentimentAnalysis()

        if news_df.empty:
            return jsonify({"error": "No news articles found for the ticker."}), 404

        scored_news_df = score_news_sentiment(news_df)
        first_row_sentiment = scored_news_df.iloc[0]
        fig_bar_sentiment = plot_sentiment_scores(ticker)
        graph_sentiment = json.dumps(fig_bar_sentiment, cls=PlotlyJSONEncoder)

        earliest_datetime = get_earliest_date(news_df)
        price_history_df = get_live_price_history(ticker, earliest_datetime)
        cleaned_price_history_df = clean_interval_price(price_history_df)

        correlation_coefficient = calculate_pearson_correlation(
            scored_news_df[['Date Time', 'combined_sentiment_score']],
            cleaned_price_history_df[['Date Time', 'Price']]
        )

        fig_line_price_history = plot_price(cleaned_price_history_df, ticker)
        graph_price = json.dumps(fig_line_price_history, cls=PlotlyJSONEncoder)

        scored_news_df = headline_to_link(scored_news_df)

        # generating results to template
        return render_template(
            'analysis.html',
            ticker=ticker,
            name=business_name,
            graph_price=graph_price,
            first_row_sentiment=first_row_sentiment,
            correlation_coefficient=correlation_coefficient,
            historical_information=historical_information,
            graph_sentiment=graph_sentiment,
            table=scored_news_df.to_html(classes='mystyle', render_links=True, escape=False)
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
