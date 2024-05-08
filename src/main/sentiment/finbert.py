from transformers import pipeline
from .sentiment_base import SentimentAnalysisBase


class FinbertSentiment(SentimentAnalysisBase):
    def __init__(self):
        # Initialize the sentiment analysis pipeline with the FinBERT model
        self._sentiment_analysis = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        super().__init__()

    def calc_sentiment_score(self):
        # sentiment analysis to each title and extract the sentiment and score
        self.df['sentiment'] = self.df['title'].apply(lambda title: self._sentiment_analysis(title)[0])
        # sentiment score based on output
        self.df['sentiment_score'] = self.df['sentiment'].apply(
            lambda sentiment: self._map_sentiment_to_score(sentiment))

    def _map_sentiment_to_score(self, sentiment):
        # Map sentiment label to a numerical score
        scores = {'positive': 1, 'negative': -1, 'neutral': 0}
        return scores.get(sentiment['label'], 0) * sentiment['score']
