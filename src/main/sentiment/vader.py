from nltk.sentiment import SentimentIntensityAnalyzer
from .sentiment_base import SentimentAnalysis


class VaderSentimentAnalysis(SentimentAnalysis):
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        super().__init__()

    def calculate_sentiment_scores(self, df):
        # Calculate sentiment for title and description
        df['title_sentiment'] = df['title'].apply(lambda title: self.vader.polarity_scores(title)['compound'])
        df['description_sentiment'] = df['Description'].apply(
            lambda description: self.vader.polarity_scores(description)['compound'])
        # Calculate weighted merged score
        df['combined_sentiment_score'] = 0.2 * df['title_sentiment'] + 0.8 * df['description_sentiment']
        return df

    def analyze_sentiment(self, text):
        result = self._sentiment_analysis(text)[0]
        # map sentiment label to score
        scores = {'positive': 1, 'negative': -1, 'neutral': 0}
        return scores.get(result['label'], 0) * result['score']

