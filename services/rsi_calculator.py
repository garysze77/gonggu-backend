#!/usr/bin/env python3
"""
RSI Calculator for HK Stocks
用於計算RSI指標
"""

import json
import subprocess
from datetime import datetime

def calculate_rsi(prices, period=14):
    """
    Calculate RSI using Wilder's Smoothing Method
    
    Args:
        prices: list of closing prices
        period: RSI period (default 14)
    
    Returns:
        list of RSI values
    """
    if len(prices) < period + 1:
        return []
    
    # Calculate price changes
    changes = []
    for i in range(1, len(prices)):
        changes.append(prices[i] - prices[i-1])
    
    # Separate gains and losses
    gains = [c if c > 0 else 0 for c in changes]
    losses = [-c if c < 0 else 0 for c in changes]
    
    # Initial average gain and loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values = []
    
    # Calculate RSI for each subsequent period
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
    
    return rsi_values

def get_stock_data(symbol):
    """Get stock data from EJFQ"""
    cmd = [
        'curl', '-s', '--compressed', '-L',
        '-H', 'authority: www.ejfq.com',
        '-H', 'accept: */*',
        '-H', 'accept-language: zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        '-H', 'content-type: text/plain',
        '-H', 'referer: https://www.ejfq.com/home/tc/screener360.htm?method=techChart&type=inter&code=HSI&theme=360',
        '-H', 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
        '-H', 'x-requested-with: XMLHttpRequest',
        f'https://www.ejfq.com/home/tc/tradingview3_360/php/chartfeed.php?symbol={symbol}&resolution=D&method=history'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        data = json.loads(result.stdout)
        if data.get("s") == "ok":
            return {
                "symbol": symbol,
                "close": data.get("c", []),
                "timestamp": data.get("t", [])
            }
    except:
        pass
    return None

def generate_signal(rsi_14, rsi_30=None):
    """
    Generate trading signal based on RSI
    
    Args:
        rsi_14: 14-period RSI
        rsi_30: Optional 30-period RSI
    
    Returns:
        dict with signal and description
    """
    if rsi_14 is None:
        return {"signal": "HOLD", "reason": "No data"}
    
    # Overbought/Oversold strategy
    if rsi_14 <= 30:
        return {"signal": "BUY", "reason": f"RSI({rsi_14:.1f}) <= 30 - Oversold"}
    elif rsi_14 >= 70:
        return {"signal": "SELL", "reason": f"RSI({rsi_14:.1f}) >= 70 - Overbought"}
    elif rsi_14 >= 50:
        return {"signal": "HOLD", "reason": f"RSI({rsi_14:.1f}) - Neutral (above 50)"}
    else:
        return {"signal": "HOLD", "reason": f"RSI({rsi_14:.1f}) - Neutral (below 50)"}

def analyze_stock(symbol):
    """Complete stock analysis"""
    data = get_stock_data(symbol)
    
    if not data or not data.get("close"):
        return {"error": f"Could not fetch data for {symbol}"}
    
    closes = data["close"]
    timestamps = data["timestamp"]
    
    # Calculate RSI
    rsi_values = calculate_rsi(closes, 14)
    
    if not rsi_values:
        return {"error": "Not enough data"}
    
    # Latest RSI
    latest_rsi = rsi_values[-1]
    latest_close = closes[-1]
    latest_date = datetime.fromtimestamp(timestamps[-1]).strftime("%Y-%m-%d")
    
    # Generate signal
    signal = generate_signal(latest_rsi)
    
    return {
        "symbol": symbol,
        "date": latest_date,
        "close": latest_close,
        "rsi_14": round(latest_rsi, 2),
        "signal": signal["signal"],
        "reason": signal["reason"]
    }

# Example usage
if __name__ == "__main__":
    # Test with HSI
    print("=== Testing RSI Calculator ===")
    
    # Get HSIA data
    result = analyze_stock("HSI")
    print(json.dumps(result, indent=2, ensure_ascii=False))
