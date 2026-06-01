#!/usr/bin/env python3
import os
import sys
import zipfile
import urllib.request
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.retrain_bookticker import BNNFeatureExtractor
import train_bnn_standalone as train_bnn
import numpy as np

BINANCE_URL = "https://data.binance.vision/data/spot/monthly/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-01.zip"
ZIP_PATH = ROOT / "BTCUSDT-bookTicker-2024-01.zip"
CSV_PATH = ROOT / "BTCUSDT-bookTicker-2024-01.csv"

# Transaction cost model parameters
TAKER_FEE = 0.0004       # 4 bps Binance VIP taker fee
MAKER_REBATE = 0.0000    # 0 bps maker
SLIPPAGE = 0.0001        # 1 bps slippage expectation

def download_data():
    if not CSV_PATH.exists():
        if not ZIP_PATH.exists():
            print(f"Downloading historical data from {BINANCE_URL}...")
            urllib.request.urlretrieve(BINANCE_URL, ZIP_PATH)
        print("Extracting CSV...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(ROOT)
    print("Historical data ready.")

def run_backtest():
    print("Loading compiled BNN model weights...")
    model = train_bnn.build_model()
    try:
        model.load_weights(str(ROOT / "bnn_weights.h5"))
    except Exception:
        print("Model weights not found. Please train the model first.")
        sys.exit(1)

    extractor = BNNFeatureExtractor()
    
    position = 0 # 1 for Long, -1 for Short, 0 for Flat
    entry_price = 0.0
    realized_pnl = 0.0
    trade_count = 0
    max_drawdown = 0.0
    peak_equity = 0.0
    
    print("Running historical backtest over 1 month of tick data...")
    print("Transaction Cost Model: Taker Fee 4 bps, Slippage 1 bps")
    
    tick_count = 0
    last_minute = None
    last_equity = 0.0
    minute_returns = []
    equity = 0.0
    
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Expected header: updateId, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time
        
        for row in reader:
            bid = float(row[1])
            bid_qty = float(row[2])
            ask = float(row[3])
            ask_qty = float(row[4])
            event_time = int(row[6]) # ms epoch
            price = 0.5 * (bid + ask)
            volume = bid_qty + ask_qty
            
            tick = {
                "price": price,
                "volume": volume,
                "bid": bid,
                "ask": ask
            }
            
            ready, ind, spike = extractor.update(tick)
            tick_count += 1
            
            if ready and tick_count % 10 == 0: # Downsample inference for speed in backtest
                # FIX: The BNN model expects a 16-dimensional bipolar {-1, +1} input
                # (the quantized spike vector), NOT 4 raw float indicators.
                # Convert the uint16 spike bitmask to a 16-element bipolar array.
                spike_bipolar = np.array([
                    1.0 if (spike >> b) & 1 else -1.0
                    for b in range(16)
                ], dtype=np.float32).reshape(1, 16)
                pred = model.predict(spike_bipolar, verbose=0)
                decision = np.argmax(pred[0]) # 0=BUY, 1=HOLD, 2=SELL
                
                # Trading Logic
                if decision == 0: # BUY signal
                    if position <= 0:
                        # Close short if any, open long
                        if position == -1:
                            # Buy to cover at ask price
                            pnl = (entry_price - ask) / entry_price - TAKER_FEE - SLIPPAGE
                            realized_pnl += pnl
                            trade_count += 1
                        
                        position = 1
                        entry_price = ask # Pay the spread
                        
                elif decision == 2: # SELL signal
                    if position >= 0:
                        # Close long if any, open short
                        if position == 1:
                            # Sell to close at bid price
                            pnl = (bid - entry_price) / entry_price - TAKER_FEE - SLIPPAGE
                            realized_pnl += pnl
                            trade_count += 1
                        
                        position = -1
                        entry_price = bid # Pay the spread
                        
                # Update peak equity / drawdown
                equity = realized_pnl
                if position == 1:
                    equity += (bid - entry_price) / entry_price - TAKER_FEE
                elif position == -1:
                    equity += (entry_price - ask) / entry_price - TAKER_FEE
                    
                peak_equity = max(peak_equity, equity)
                drawdown = peak_equity - equity
                max_drawdown = max(max_drawdown, drawdown)
                
            # Track minute-by-minute returns for Sharpe Ratio
            current_minute = event_time // 60000
            if last_minute is None:
                last_minute = current_minute
            elif current_minute > last_minute:
                minute_returns.append(equity - last_equity)
                last_equity = equity
                last_minute = current_minute
                
            if tick_count > 500000: # Stop early for the demo
                break

    print("\n--- Backtest Results (500,000 ticks) ---")
    print(f"Total Trades: {trade_count}")
    print(f"Cumulative PnL: {realized_pnl * 100:.2f}%")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    if trade_count > 0:
        print(f"Average Trade: {(realized_pnl / trade_count) * 10000:.1f} bps")
        
    ret_arr = np.array(minute_returns)
    if len(ret_arr) > 1 and np.std(ret_arr) > 0:
        # Crypto is 24/7/365 -> 525,600 minutes per year
        sharpe = (np.mean(ret_arr) / np.std(ret_arr)) * np.sqrt(365 * 24 * 60)
        print(f"Annualized Sharpe: {sharpe:.2f} (from 1-min intervals, sqrt(525600) annualized)")

if __name__ == "__main__":
    download_data()
    run_backtest()
