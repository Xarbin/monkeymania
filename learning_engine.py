# learning_engine.py

import pandas as pd

def evaluate_trade(ticker, buy_price, stop_price, take_profit_price, postmarket_path):
    try:
        df = pd.read_csv(postmarket_path)
        df.columns = [c.strip().lower() for c in df.columns]

        rename_map = {
            'symbol': 'ticker',
            'price': 'price',
            'post-market close': 'postmarket_close'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        row = df[df['ticker'].str.upper() == ticker.upper()]
        if row.empty:
            return False, f"🔍 {ticker} not found in postmarket file."

        post_close = float(row.iloc[0].get('price', 0))
        if post_close >= take_profit_price:
            return True, f"✅ Hit Take Profit (${post_close} ≥ ${take_profit_price})"
        elif post_close <= stop_price:
            return False, f"❌ Hit Stop Loss (${post_close} ≤ ${stop_price})"
        else:
            return False, f"⚠️ No Trigger (${post_close})"

    except Exception as e:
        return False, f"❌ Error reading postmarket data: {e}"