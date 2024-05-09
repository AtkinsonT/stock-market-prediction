class APIConfiguration:
    # URL for accessing news from API
    NEWS_API_URL = "https://mboum-finance.p.rapidapi.com/v1/markets/news"

    # URL for accessing historical stock data from API
    HISTORY_API_URL = "https://mboum-finance.p.rapidapi.com/v1/markets/stock/history"

    # Headers used in HTTP requests to the API, authentication details included
    headers = {
        "X-RapidAPI-Key": "661503ddeemsh08dcbf64db98b48p15db98jsnb8437c93cd73",
        # API key for authenticating the requests
        "X-RapidAPI-Host": "mboum-finance.p.rapidapi.com"  # Host header required by the API for routing the request
    }
