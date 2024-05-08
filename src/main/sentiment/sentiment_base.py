import plotly.express as px
import plotly.graph_objects as go


class SentimentAnalysis:
    def __init__(self):
        self.symbol = None
        self.df = None

    def set_symbol(self, symbol: str):
        self.symbol = symbol

    def set_data(self, df):
        self.df = df

    def plot_sentiment(self) -> go.Figure:
        # filter out rows 'sentiment_score' is 0
        df_plot = self.df[self.df['combined_sentiment_score'] != 0]
        # colour for each bar based on sentiment
        colors = ['green' if x > 0 else 'red' for x in df_plot['combined_sentiment_score']]
        fig = go.Figure(data=[go.Bar(
            x=df_plot['Date Time'],
            y=df_plot['combined_sentiment_score'],
            marker={'color': colors}
        )])
        fig.update_layout(
            title=f"{self.symbol} News Sentiment Scores",
            xaxis_title='Date Time',
            yaxis_title='Sentiment Score',
            template='plotly_white'
        )
        return fig
