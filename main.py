import argparse
from utils.stock_data_fetcher import fetch_stock_data_yfinance
from utils.news_scraper import scrape_news
from utils.sentiment_analyzer import analyze_news_sentiment
from models.lstm_price_predictor import predict_with_lstm
from models.ppo_trading_agent import PPOTradingAgent

def main(ticker, start, end):
    fetch_stock_data_yfinance(ticker=ticker, start_date=start, end_date=end)
    scrape_news(keyword=ticker[:4])
    analyze_news_sentiment("data/news_data.json", output_path="data/sentiment_result.json")
    price_prediction = predict_with_lstm("data/stock_data.csv")
    agent = PPOTradingAgent()
    agent.run(price_prediction, "data/sentiment_result.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 股票交易系統 CLI 工具")
    parser.add_argument("--ticker", type=str, default="2330.TW", help="股票代碼")
    parser.add_argument("--start", type=str, default="2023-01-01", help="開始日期")
    parser.add_argument("--end", type=str, default=None, help="結束日期")
    args = parser.parse_args()
    main(args.ticker, args.start, args.end)
