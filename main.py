
# Required libraries
import os
import logging as lg
from kiteconnect import KiteConnect
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Logger initialization
def initialize_logger():
    logs_path = './logs/'
    try:
        os.mkdir(logs_path)
    except OSError:
        pass

    log_name = datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
    currentLog_path = logs_path + log_name
    lg.basicConfig(filename=currentLog_path, 
                   format='%(asctime)s - %(levelname)s: %(message)s', 
                   level=lg.INFO)
    lg.getLogger().addHandler(lg.StreamHandler())
    lg.info('Logger initialized')

# Trading configuration
class Config:
    API_KEY = os.getenv("KITE_API_KEY")
    API_SECRET = os.getenv("KITE_API_SECRET")
    ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")  # Store after generating using KiteConnect
    STOP_LOSS_PERCENTAGE = 0.01
    TAKE_PROFIT_PERCENTAGE = 0.02
    MAX_SPENT_EQUITY = 10000  # Example value in INR
    ML_MODEL_PATH = './ml_model.pkl'  # Example path for ML model

# Initialize KiteConnect API
def initialize_kite():
    kite = KiteConnect(api_key=Config.API_KEY)
    kite.set_access_token(Config.ACCESS_TOKEN)
    return kite

# Function to check account validity
def check_account(kite):
    try:
        profile = kite.profile()
        lg.info(f"Account validated for user: {profile['user_name']}")
    except Exception as e:
        lg.error("Error validating account")
        lg.error(e)
        exit()

# Function to clean open orders
def clean_open_orders(kite):
    try:
        orders = kite.orders()
        for order in orders:
            if order['status'] in ['OPEN', 'TRIGGER PENDING']:
                kite.cancel_order(variety=order['variety'], order_id=order['order_id'])
        lg.info("All open orders cancelled")
    except Exception as e:
        lg.error("Failed to cancel open orders")
        lg.error(e)

# Function to check asset validity
def is_tradable(kite, instrument):
    try:
        instruments = kite.instruments(exchange="NSE")
        tradable = any(instr['tradingsymbol'] == instrument and instr['tradable'] for instr in instruments)
        if tradable:
            lg.info(f"Instrument {instrument} is tradable")
        else:
            lg.info(f"Instrument {instrument} is not tradable")
        return tradable
    except Exception as e:
        lg.error("Error checking tradability")
        lg.error(e)
        return False

# Function to calculate technical indicators
def calculate_indicators(data):
    try:
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        lg.info("Indicators calculated")
        return data
    except Exception as e:
        lg.error("Error calculating indicators")
        lg.error(e)
        return data

# Function for ML-based signal generation
def generate_ml_signals(data):
    try:
        lg.info("Generating ML-based signals...")
        model = RandomForestClassifier()
        features = ['SMA_20', 'EMA_20', 'RSI']
        scaler = StandardScaler()
        X = scaler.fit_transform(data[features].fillna(0))
        data['ML_Signal'] = model.predict(X)
        lg.info("ML signals added")
        return data
    except Exception as e:
        lg.error("Error in ML signal generation")
        lg.error(e)
        return data

# Order book analysis
def analyze_order_book(kite, instrument):
    try:
        ob = kite.quote(f"NSE:{instrument}")
        depth = ob['depth']
        bid_volume = sum([bid['quantity'] for bid in depth['buy']])
        ask_volume = sum([ask['quantity'] for ask in depth['sell']])
        lg.info(f"Order book: BID_VOL={bid_volume}, ASK_VOL={ask_volume}")
        return bid_volume, ask_volume
    except Exception as e:
        lg.error("Error analyzing order book")
        lg.error(e)
        return 0, 0

# Live data monitoring and decision making
def monitor_and_trade(kite, instrument):
    try:
        while True:
            ltp = kite.ltp(f"NSE:{instrument}")[f"NSE:{instrument}"]['last_price']
            lg.info(f"Live Price for {instrument}: {ltp}")
            bid_volume, ask_volume = analyze_order_book(kite, instrument)
            if bid_volume > ask_volume * 1.5:
                lg.info(f"Strong BUY signal for {instrument}")
            elif ask_volume > bid_volume * 1.5:
                lg.info(f"Strong SELL signal for {instrument}")
            time.sleep(2)
    except Exception as e:
        lg.error("Error in live trading")
        lg.error(e)

# Main trading bot logic
def main():
    # Initialize logger
    initialize_logger()

    # Initialize Kite API
    kite = initialize_kite()

    # Validate account
    check_account(kite)

    # Clean open orders
    clean_open_orders(kite)

    # Specify instrument (example: RELIANCE for NSE)
    instrument = "RELIANCE"

    if not is_tradable(kite, instrument):
        lg.info("Exiting as instrument is not tradable")
        return

    lg.info(f"Starting trading bot for {instrument}")

    # Fetch historical data for backtesting
    try:
        historical_data = kite.historical_data(instrument_token=738561, 
                                               from_date="2023-01-01", 
                                               to_date="2023-12-31", 
                                               interval="day")
        historical_df = pd.DataFrame(historical_data)
        historical_df = calculate_indicators(historical_df)
        historical_df = generate_ml_signals(historical_df)
    except Exception as e:
        lg.error("Error fetching historical data")
        lg.error(e)

    # Monitor and trade live
    monitor_and_trade(kite, instrument)

if __name__ == "__main__":
    main()
