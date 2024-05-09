import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class HistoricalPrediction:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = self.download_data()

    def calculate_macd(self, df, short_window=12, long_window=26, signal=9):
        # Calculate MACD
        short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()  # short term EMA
        long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()  # long term EMA
        df['MACD'] = short_ema - long_ema
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    def calculate_rsi(self, df, periods=14):
        # Calculate Relative Strength Index (RSI)
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()  # avg gain
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()  # avg loss
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    def download_data(self):
        # Download historical data and calculate indicators
        nasdaq = yf.Ticker(self.symbol)
        data = nasdaq.history(period="10y")  # Use 10 years of data
        self.calculate_macd(data)
        self.calculate_rsi(data)
        data['Previous Close'] = data['Close'].shift(1)  # yesterday's close price
        data.dropna(inplace=True)
        return data

    def train_and_optimize(self):
        # Train the model and optimize parameters
        x = self.data[['Previous Close', 'MACD', 'Signal_Line', 'RSI']]
        y = (self.data['Close'] > self.data['Previous Close']).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # split data into sets

        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='precision', n_jobs=-1, verbose=1)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_  # best model found by GSCV
        predictions = best_model.predict(x_test)
        return {
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'accuracy': accuracy_score(y_test, predictions)
        }
