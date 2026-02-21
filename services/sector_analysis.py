#!/usr/bin/env python3
"""
HK Sector Analysis
分析每個Sector既表現同最佳股票
"""

from rsi_calculator import get_stock_data, calculate_rsi, generate_signal, analyze_stock
import json

# 恒生指數主要藍籌股 (Sample - 主要 sector 代表)
SECTOR_LEADERS = {
    "Financials": [
        "939",   # 建設銀行
        "3988",  # 中國銀行
        "0005",  # HSBC
        "2388",  # 港交所
        "1113",  # 長實
    ],
    "Tech": [
        "700",   # 騰訊
        "3690",  # 美團
        "9988",  # 阿里巴巴
        "9618",  # 京東
        "1024",  # 快手
    ],
    "Properties": [
        "00175", # 恒大
        "0016",  # 恒大地產
        "0012",  # 恒地
        "0011",  # 恆生銀行
        "0001",  # 長江
    ],
    "Telecom": [
        "0762",  # 中國移動
        "6822",  # 香港電訊
    ],
    "Energy": [
        "0883",  # 中海油
        "0857",  # 中國石油
    ],
    "Retail": [
        "291",   # 華潤啤酒
        "2319",  # 蒙牛乳業
    ],
    "Healthcare": [
        "1177",  # 中國生物製藥
        "0669",  # 創科實業
    ]
}

def analyze_sector(sector_name, symbols):
    """分析一個 sector"""
    results = []
    
    for symbol in symbols:
        try:
            analysis = analyze_stock(symbol)
            if "error" not in analysis:
                results.append(analysis)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # 根據 RSI 排序
    if results:
        results.sort(key=lambda x: x.get("rsi_14", 50))
    
    return {
        "sector": sector_name,
        "stocks": results,
        "count": len(results)
    }

def generate_sector_signals():
    """生成所有 sector 既信號"""
    all_sectors = []
    
    for sector_name, symbols in SECTOR_LEADERS.items():
        sector_analysis = analyze_sector(sector_name, symbols)
        
        # 搵最佳 buy signal
        buy_candidates = [s for s in sector_analysis["stocks"] if s["signal"] == "BUY"]
        sell_candidates = [s for s in sector_analysis["stocks"] if s["signal"] == "SELL"]
        
        sector_analysis["recommendations"] = {
            "buy": buy_candidates[:2] if buy_candidates else [],
            "sell": sell_candidates[:2] if sell_candidates else [],
            "neutral": len([s for s in sector_analysis["stocks"] if s["signal"] == "HOLD"])
        }
        
        all_sectors.append(sector_analysis)
    
    return all_sectors

def print_report():
    """打印 report"""
    print("=" * 60)
    print("🇭🇰 HK Stock Sector Analysis Report")
    print("=" * 60)
    
    sectors = generate_sector_signals()
    
    for sector in sectors:
        print(f"\n📊 {sector['sector']} ({sector['count']} stocks)")
        print("-" * 40)
        
        recs = sector["recommendations"]
        
        if recs["buy"]:
            print("  🟢 BUY Recommendations:")
            for stock in recs["buy"]:
                print(f"     {stock['symbol']}: RSI={stock['rsi_14']}, Close={stock['close']}")
        
        if recs["sell"]:
            print("  🔴 SELL Recommendations:")
            for stock in recs["sell"]:
                print(f"     {stock['symbol']}: RSI={stock['rsi_14']}, Close={stock['close']}")
        
        if recs["neutral"]:
            print(f"  ⚪ HOLD: {recs['neutral']} stocks")
    
    print("\n" + "=" * 60)
    
    return sectors

if __name__ == "__main__":
    sectors = print_report()
    
    # Save to JSON
    with open("/root/.openclaw/workspace/hk-stock-prediction/data/sector_analysis.json", "w", encoding="utf-8") as f:
        json.dump(sectors, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Report saved to data/sector_analysis.json")
