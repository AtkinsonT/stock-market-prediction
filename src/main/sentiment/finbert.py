from transformers import pipeline
from .sentiment_base import SentimentAnalysis


class FinbertSentimentAnalysis(SentimentAnalysis):
    def __init__(self):
        self._sentiment_analysis = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        super().__init__()

    def calculate_sentiment_scores(self, df):
        df['title_sentiment'] = df['title'].apply(lambda title: self.analyze_sentiment(title))
        df['description_sentiment'] = df['Description'].apply(lambda description: self.analyze_sentiment(description))
        # Calculate weighted merged score
        df['combined_sentiment_score'] = 0.2 * df['title_sentiment'] + 0.8 * df['description_sentiment']
        return df

    def analyze_sentiment(self, text):
        result = self._sentiment_analysis(text)[0]
        # map sentiment label to score
        scores = {'positive': 1, 'negative': -1, 'neutral': 0}
        return scores.get(result['label'], 0) * result['score']
