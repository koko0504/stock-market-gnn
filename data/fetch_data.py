import yfinance as yf

tickers = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","ADBE","CRM","ORCL","CSCO",
    "JPM","BAC","WFC","GS","MS","C","AXP","BLK","SCHW","USB",
    "XOM","CVX","COP","SLB","EOG","PSX","VLO","MPC","OXY","WMB",
    "JNJ","PFE","UNH","ABBV","MRK","TMO","ABT","LLY","DHR","BMY",
    "PG","KO","PEP","WMT","COST","MCD","NKE","DIS","HD","SBUX"
]

def fetch_prices(start="2018-01-01", end="2024-01-01"):
    prices = yf.download(tickers, start=start, end=end)["Close"]
    return prices

def get_returns(prices):
    return prices.pct_change().dropna()
