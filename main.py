#!/usr/bin/env python3
"""
HK Stock Prediction System
港股預測系統 - Sector Rotation + RSI Signals
"""

import subprocess
import json
from datetime import datetime

# 恆生指數主要藍籌股
STOCKS = {
    # Tech
    "700": "騰訊",
    "3690": "美團",
    "9988": "阿里巴巴",
    "9618": "京東",
    "1024": "快手",
    
    # Financials
    "939": "建設銀行",
    "3988": "中國銀行",
    "0005": "HSBC",
    "2388": "港交所",
    "1113": "長實",
    
    # Properties
    "0016": "恒大地產",
    "0012": "恒地",
    "0001": "長江",
    "1109": "置地",
    
    # Telecom
    "0762": "中國移動",
    "6822": "香港電訊",
    
    # Energy
    "0883": "中海油",
    "0857": "中石油",
    
    # Retail
    "291": "華潤啤酒",
    "2319": "蒙牛乳業",
    
    # Healthcare
    "1177": "中國生物製藥",
    "2269": "藥明生物",
}

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return None
    
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [c if c > 0 else 0 for c in changes]
    losses = [-c if c < 0 else 0 for c in changes]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values = []
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
    
    return rsi_values[-1] if rsi_values else None

def fetch_stock(symbol):
    """Fetch stock data from EJFQ"""
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
                "name": STOCKS.get(symbol, symbol),
                "close": data.get("c", []),
                "high": data.get("h", []),
                "low": data.get("l", []),
                "timestamp": data.get("t", [])
            }
    except:
        pass
    return None

def analyze_stock(symbol):
    """Complete stock analysis"""
    data = fetch_stock(symbol)
    
    if not data or not data.get("close"):
        return None
    
    closes = data["close"]
    timestamps = data["timestamp"]
    
    rsi_14 = calculate_rsi(closes, 14)
    
    if rsi_14 is None:
        return None
    
    # Generate signal
    if rsi_14 <= 30:
        signal = "BUY"
        reason = f"RSI({rsi_14:.1f}) 超賣"
    elif rsi_14 >= 70:
        signal = "SELL"
        reason = f"RSI({rsi_14:.1f}) 超買"
    else:
        signal = "HOLD"
        reason = f"RSI({rsi_14:.1f}) 中性"
    
    return {
        "symbol": data["symbol"],
        "name": data["name"],
        "close": closes[-1],
        "rsi_14": round(rsi_14, 2),
        "signal": signal,
        "reason": reason,
        "date": datetime.fromtimestamp(timestamps[-1]).strftime("%Y-%m-%d")
    }

def generate_daily_report():
    """Generate daily analysis report"""
    print("🇭🇰 HK Stock Daily Analysis")
    print("=" * 50)
    
    results = []
    buy_list = []
    sell_list = []
    
    for symbol in STOCKS.keys():
        analysis = analyze_stock(symbol)
        if analysis:
            results.append(analysis)
            
            if analysis["signal"] == "BUY":
                buy_list.append(analysis)
            elif analysis["signal"] == "SELL":
                sell_list.append(analysis)
            
            emoji = "🟢" if analysis["signal"] == "BUY" else ("🔴" if analysis["signal"] == "SELL" else "⚪")
            print(f"{emoji} {analysis['symbol']} {analysis['name']}: {analysis['close']} | RSI: {analysis['rsi_14']} | {analysis['signal']}")
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  🟢 BUY: {len(buy_list)} stocks")
    print(f"  🔴 SELL: {len(sell_list)} stocks")
    print(f"  ⚪ HOLD: {len(results) - len(buy_list) - len(sell_list)} stocks")
    
    if buy_list:
        print("\n🟢 Top BUY Recommendations:")
        for stock in sorted(buy_list, key=lambda x: x["rsi_14"]):
            print(f"   {stock['symbol']} {stock['name']} - RSI: {stock['rsi_14']}")
    
    # Save to JSON
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": len(results),
        "buy_count": len(buy_list),
        "sell_count": len(sell_list),
        "buy_signals": buy_list,
        "sell_signals": sell_list,
        "all_stocks": results
    }
    
    with open("/root/.openclaw/workspace/hk-stock-prediction/data/daily_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

if __name__ == "__main__":
    generate_daily_report()
