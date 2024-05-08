import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from .sentiment_base import SentimentAnalysisBase

# Download VADER at module level
nltk.download('vader_lexicon')


class VaderSentiment(SentimentAnalysisBase):
    def __init__(self):
        # Initialize the sentiment intensity analyzer
        self.vader = SentimentIntensityAnalyzer()
        super().__init__()

    def calc_sentiment_score(self):
        # Calculate sentiment for each title and directly extract the compound score
        self.df['sentiment_score'] = self.df['title'].apply(
            lambda title: self.vader.polarity_scores(title)['compound']
        )
