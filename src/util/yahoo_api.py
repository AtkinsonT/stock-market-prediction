from datetime import datetime
import requests
from src.util.config import Config
import pandas as pd
import pytz

DATE_FORMAT = "%b-%d-%y %H:%M %S"
BST = pytz.timezone('Europe/London')


class API:
    def __init__(self) -> None:
        pass

    def fetch_stock_data(symbol, interval):
        querystring = {"symbol": symbol, "interval": interval, "diffandsplits": "false"}
        response = requests.get(Config.HISTORY_API_URL, headers=Config.headers, params=querystring)
        return response.json() if response.ok else None

    def get_news(ticker):
        querystring = {"symbol": f"{ticker}"}
        response = requests.get(url=Config.NEWS_API_URL, headers=Config.headers, params=querystring)
        respose_json = response.json()
        data_array = []
        if 'body' in respose_json:
            articles = respose_json['body']
            for article in articles:
                utc_datetime = datetime.strptime(
                    article['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
                title_i = article['title']
                description_i = article['description']
                link_i = article['link']
                data_array.append([utc_datetime, title_i, description_i, f'<a href="{link_i}">{title_i}</a>'])
            columns = ['Date Time', 'title', 'Description', 'title + link']
            df = pd.DataFrame(data_array, columns=columns)
            df['Date Time'] = pd.to_datetime(df['Date Time'], format=DATE_FORMAT, utc=True)
            df.set_index('Date Time', inplace=True)
            df.sort_values(by='Date Time', ascending=False)
            df.reset_index(inplace=True)
        else:
            print(f'No data returned for ticker: {ticker}, response: {respose_json}')
            df = pd.DataFrame()
        print(f"Response dataframe {df}")
        return df

    def get_price_history(ticker: str, earliest_datetime: pd.Timestamp) -> pd.DataFrame:
        querystring = {"symbol": {ticker}, "interval": "1m", "diffandsplits": "false"}
        response = requests.get(url=Config.HISTORY_API_URL, headers=Config.headers, params=querystring)
        respose_json = response.json()
        price_history = respose_json['body']
        data_dict = []
        for stock_price in price_history.values():
            date_time_num = stock_price["date_utc"]
            utc_datetime = datetime.fromtimestamp(date_time_num, tz=pytz.utc)
            est_datetime = utc_datetime.astimezone(tz=BST)
            if est_datetime < earliest_datetime:
                continue
            price = stock_price["open"]
            data_dict.append([est_datetime.strftime(DATE_FORMAT), price])
        # Set column names
        columns = ['Date Time', 'Price']
        df = pd.DataFrame(data_dict, columns=columns)
        df['Date Time'] = pd.to_datetime(df['Date Time'], format=DATE_FORMAT)
        df.sort_values(by='Date Time', ascending=True)
        df.reset_index(inplace=True)
        df.drop('index', axis=1, inplace=True)
        return df
