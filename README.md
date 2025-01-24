# Algorithmic Trading Bot

This repository contains an algorithmic trading bot designed for the Indian stock market using Zerodha's Kite Connect API.

## Features

- **Technical Indicators**: SMA, EMA, RSI.
- **ML-Based Signal Generation**: Placeholder for advanced machine learning models.
- **Order Book Analysis**: Analyzes bid/ask volumes to provide insights.
- **Live Monitoring and Trading**: Trades based on live data and market signals.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/algo_trading_bot.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file in the root directory with your Kite Connect credentials:
   ```
   KITE_API_KEY=your_api_key
   KITE_API_SECRET=your_api_secret
   KITE_ACCESS_TOKEN=your_access_token
   ```

4. Run the bot:
   ```bash
   python main.py
   ```

## Disclaimer

This bot is for educational purposes only. Use it at your own risk and ensure compliance with financial regulations.
