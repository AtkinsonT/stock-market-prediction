import json
from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import yfinance as yf
from flask import Flask, render_template, request
from plotly.utils import PlotlyJSONEncoder
from sentiment.finbert import FinbertSentiment
from src.util.yahoo_api import API
from src.historical.historical_backtesting import HistoricalPrediction

BST = pytz.timezone('Europe/London')
sentimentAlgorithm = FinbertSentiment()

app = Flask(__name__, template_folder="../../data/templates")


def get_price_history(ticker: str, earliest_datetime: pd.Timestamp) -> pd.DataFrame:
    return API.get_price_history(ticker, earliest_datetime)


def get_news(ticker) -> pd.DataFrame:
    sentimentAlgorithm.set_symbol(ticker)
    return API.get_news(ticker)


def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    sentimentAlgorithm.set_data(news_df)
    sentimentAlgorithm.calc_sentiment_score()
    return sentimentAlgorithm.df


def plot_sentiment(ticker: str) -> go.Figure:
    return sentimentAlgorithm.plot_sentiment()


def get_earliest_date(df: pd.DataFrame) -> Any | None:
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


def plot_hourly_price(df, ticker) -> go.Figure:
    fig = px.line(data_frame=df, x=df['Date Time'], y="Price", title=f"{ticker} Price")
    return fig


def get_business_name(ticker):
    tickername = yf.Ticker(ticker)
    try:
        info = tickername.info
        return info['longName']
    except KeyError:
        return None


def evaluate_historical_accuracy(ticker):
    hist_stock_prediction = HistoricalPrediction(ticker)
    historical_scores = hist_stock_prediction.train_and_optimize()
    return historical_scores


def calculate_pearson_correlation(sentiment_df, prices_df):
    # Remove timezone info to ensure consistency
    sentiment_df['Date Time'] = sentiment_df['Date Time'].dt.tz_localize(None)
    sentiment_df['Date Time'] = sentiment_df['Date Time'].dt.round('1min')
    prices_df['Date Time'] = prices_df['Date Time'].dt.tz_localize(None)
    prices_df['Date Time'] = prices_df['Date Time'].dt.round('1min')
    # Create a time range for merging
    sentiment_df['start_time'] = sentiment_df['Date Time'] - pd.Timedelta(minutes=5)
    sentiment_df['end_time'] = sentiment_df['Date Time'] + pd.Timedelta(minutes=5)
    # Merge using conditional that checks price timestamps fall within sentiment time range
    merged_df = pd.merge(sentiment_df.assign(key=1), prices_df.assign(key=1), on='key').drop('key', axis=1)
    merged_df = merged_df[
        (merged_df['Date Time_y'] >= merged_df['start_time']) & (merged_df['Date Time_y'] <= merged_df['end_time'])]
    # Drop extra columns used for merging
    merged_df.drop(['start_time', 'end_time'], axis=1, inplace=True)
    # Handle cases with insufficient data
    if len(merged_df) < 2:
        print("Not enough data to calculate Pearson correlation.")
        return None
    # Ensure no NaN values which can affect correlation calculation
    merged_df.dropna(subset=['sentiment_score', 'Price'], inplace=True)
    # Calculate correlation
    correlation = merged_df['sentiment_score'].corr(merged_df['Price'])
    return correlation


def clean_interval_price(df):
    df = df[df['Price'] != 0.0]
    # if any !=0, run 5m interval get price history
    return df


@app.route('/', methods=['GET'])
def index():
    return render_template('menu.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form['ticker'].strip().upper()
    historical_information = evaluate_historical_accuracy(ticker)
    business_name = get_business_name(ticker)
    news_df = get_news(ticker)
    scored_news_df = score_news(news_df)
    first_row_sentiment = scored_news_df.iloc[0]
    fig_bar_sentiment = plot_sentiment(ticker)
    graph_sentiment = json.dumps(fig_bar_sentiment, cls=PlotlyJSONEncoder)
    earliest_datetime = get_earliest_date(news_df)
    price_history_df = get_price_history(ticker, earliest_datetime)
    cleaned_price_history_df = clean_interval_price(price_history_df)
    correlation_coefficient = calculate_pearson_correlation(scored_news_df[['Date Time', 'sentiment_score']],
                                                            cleaned_price_history_df[['Date Time', 'Price']])
    fig_line_price_history = plot_hourly_price(cleaned_price_history_df, ticker)
    graph_price = json.dumps(fig_line_price_history, cls=PlotlyJSONEncoder)
    scored_news_df = convert_headline_to_link(scored_news_df)
    return render_template('analysis.html', ticker=ticker, name=business_name, graph_price=graph_price,
                           first_row_sentiment=first_row_sentiment, correlation_coefficient=correlation_coefficient,
                           historical_information=historical_information, graph_sentiment=graph_sentiment,
                           table=scored_news_df.to_html
                           (classes='mystyle', render_links=True, escape=False))


def convert_headline_to_link(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(2, 'Headline', df['title + link'])
    df.drop(columns=['sentiment', 'title + link', 'title'], inplace=True, axis=1)
    return df


if __name__ == '__main__':
    app.run(debug=True)
