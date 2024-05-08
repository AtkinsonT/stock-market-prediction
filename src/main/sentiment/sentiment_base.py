import plotly.express as px
import plotly.graph_objects as go


class SentimentAnalysisBase:
    def __init__(self):
        self.symbol = None
        self.df = None

    def set_symbol(self, symbol: str):
        self.symbol = symbol

    def set_data(self, df):
        self.df = df

    def calc_sentiment_score(self):
        pass

    def get_sentiment_scores(self):
        return self.df

    def plot_sentiment(self) -> go.Figure:
        # Filter out rows where 'sentiment_score' is 0
        df_plot = self.df[self.df['sentiment_score'] != 0]
        # Define colors for each bar based on the sentiment score
        colors = ['green' if x > 0 else 'red' for x in df_plot['sentiment_score']]
        # Create a bar plot using Graph Objects for detailed customization
        fig = go.Figure(data=[go.Bar(
            x=df_plot['Date Time'],
            y=df_plot['sentiment_score'],
            marker={'color': colors}  # Correctly apply the colors using a dictionary
        )])
        # Update layout of the figure
        fig.update_layout(
            title=f"{self.symbol} Hourly Sentiment Scores",
            xaxis_title='Date Time',
            yaxis_title='Sentiment Score',
            template='plotly_white'
        )
        return fig
