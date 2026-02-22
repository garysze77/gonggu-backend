#!/usr/bin/env python3
"""
GongGu ML Training Module
Uses all technical indicators as features for ML prediction
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Stock list - 51 stocks
STOCKS = [
    '0700.HK', '3690.HK', '9988.HK', '09988.HK', '9618.HK', '1024.HK', '9999.HK',
    '0939.HK', '3988.HK', '0005.HK', '0388.HK', '1113.HK', '0001.HK', '0012.HK',
    '0016.HK', '1109.HK', '0762.HK', '6822.HK', '0883.HK', '0857.HK',
    '291.HK', '2319.HK', '0019.HK', '1177.HK', '2269.HK', '0669.HK',
    '6690.HK', '6060.HK', '1858.HK', '2800.HK', '2828.HK', '3032.HK',
    'HSI.HK', 'HSCEI.HK'
]

STOCK_NAMES = {
    '0700': '騰訊', '3690': '美團', '9988': '阿里巴巴', '09988': '阿里-SW',
    '9618': '京東', '1024': '快手', '9999': '網易',
    '0939': '建設銀行', '3988': '中銀', '0005': 'HSBC', '0388': '港交所',
    '1113': '長實', '0001': '長江', '0012': '恒地', '0016': '恒大',
    '1109': '置地', '0762': '中移動', '6822': '香港電訊',
    '0883': '中海油', '0857': '中石油',
    '291': '華潤啤酒', '2319': '蒙牛', '0019': '太古',
    '1177': '中生製藥', '2269': '藥明生物', '0669': '創科',
    '6690': '海爾智家', '6060': '眾安', '1858': '青島啤酒',
    '2800': '盈富基金', '2828': '恒生ETF', '3032': '南方A50',
    'HSI': '恆生指數', 'HSCEI': '國企指數',
}

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return None
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None

def calculate_ema(prices, period):
    """Calculate EMA"""
    return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_ma(prices, period):
    """Calculate MA"""
    return pd.Series(prices).rolling(window=period).mean().iloc[-1]

def calculate_macd(prices):
    """Calculate MACD"""
    ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line.iloc[-1] - signal_line.iloc[-1]
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram

def calculate_bollinger(prices, period=20):
    """Calculate Bollinger Bands"""
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return upper.iloc[-1], ma.iloc[-1], lower.iloc[-1]

def fetch_and_features(symbol):
    """Fetch stock data and calculate all features"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='2y')
        
        if df.empty or len(df) < 60:
            return None
        
        closes = df['Close']
        highs = df['High']
        lows = df['Low']
        volumes = df['Volume']
        
        features = {}
        
        # RSI
        features['rsi14'] = calculate_rsi(closes, 14)
        
        # MACD
        macd, signal, hist = calculate_macd(closes)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = hist
        
        # Moving Averages
        features['ma5'] = calculate_ma(closes, 5)
        features['ma10'] = calculate_ma(closes, 10)
        features['ma20'] = calculate_ma(closes, 20)
        features['ma50'] = calculate_ma(closes, 50)
        
        # MA Crossovers
        features['ma5_above_ma20'] = 1 if features['ma5'] > features['ma20'] else 0
        features['ma5_above_ma50'] = 1 if features['ma5'] > features['ma50'] else 0
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger(closes)
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (closes.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # Volume
        vol_ma = volumes.rolling(window=20).mean().iloc[-1]
        features['volume_ratio'] = volumes.iloc[-1] / vol_ma if vol_ma > 0 else 1
        
        # Momentum
        features['momentum_5'] = (closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else 0
        features['momentum_10'] = (closes.iloc[-1] / closes.iloc[-11] - 1) if len(closes) >= 11 else 0
        
        # Volatility
        features['volatility'] = returns = closes.pct_change().rolling(window=20).std().iloc[-1]
        
        # Price position
        features['price_to_high'] = closes.iloc[-1] / highs.iloc[-20:].max() if len(highs) >= 20 else 1
        features['price_to_low'] = closes.iloc[-1] / lows.iloc[-20:].min() if len(lows) >= 20 else 1
        
        # Target: Next day direction (1=up, 0=down)
        features['target'] = 1 if closes.iloc[-1] > closes.iloc[-2] else 0
        
        features['symbol'] = symbol.replace('.HK', '')
        features['close'] = closes.iloc[-1]
        
        return features
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

def prepare_training_data():
    """Prepare training data from all stocks"""
    print("📊 Fetching data and calculating features...")
    
    all_features = []
    
    for i, symbol in enumerate(STOCKS):
        print(f"  [{i+1}/{len(STOCKS)}] Processing {symbol}...")
        features = fetch_and_features(symbol)
        if features:
            all_features.append(features)
    
    if not all_features:
        print("❌ No data collected")
        return None, None
    
    df = pd.DataFrame(all_features)
    
    # Feature columns (all indicators)
    feature_cols = [
        'rsi14', 'macd_histogram', 'ma5_above_ma20', 'ma5_above_ma50',
        'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10',
        'volatility', 'price_to_high', 'price_to_low'
    ]
    
    # Drop rows with NaN
    df = df.dropna(subset=feature_cols + ['target'])
    
    if len(df) < 30:
        print("❌ Not enough data for training")
        return None, None
    
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"✅ Total samples: {len(X)}")
    print(f"   Up days: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   Down days: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    return X, y

def train_model():
    """Train the ML model"""
    print("\n" + "="*50)
    print("🎯 Training ML Model")
    print("="*50)
    
    X, y = prepare_training_data()
    
    if X is None:
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n🔧 Training Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\n📈 Model Performance:")
    print(f"   Training accuracy: {train_score*100:.1f}%")
    print(f"   Test accuracy: {test_score*100:.1f}%")
    
    # Feature importance
    feature_names = [
        'RSI', 'MACD Hist', 'MA5>MA20', 'MA5>MA50',
        'BB Position', 'Volume Ratio', 'Momentum 5d', 'Momentum 10d',
        'Volatility', 'Price to High', 'Price to Low'
    ]
    
    print(f"\n🔍 Feature Importance:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"   {name}: {imp*100:.1f}%")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'trained_at': datetime.now().isoformat()
    }
    
    model_path = '/tmp/gonggu_ml_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to {model_path}")
    
    return model_data

def generate_predictions(model_data=None):
    """Generate predictions for all stocks"""
    print("\n" + "="*50)
    print("🔮 Generating Predictions")
    print("="*50)
    
    if model_data is None:
        model_path = '/tmp/gonggu_ml_model.pkl'
        if not os.path.exists(model_path):
            print("❌ Model not found! Train first.")
            return []
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    predictions = []
    
    for symbol in STOCKS:
        features = fetch_and_features(symbol)
        if not features:
            continue
        
        # Prepare features
        feature_cols = [
            'rsi14', 'macd_histogram', 'ma5_above_ma20', 'ma5_above_ma50',
            'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10',
            'volatility', 'price_to_high', 'price_to_low'
        ]
        
        X = np.array([[features[f] for f in feature_cols]])
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        predictions.append({
            'symbol': features['symbol'],
            'name': STOCK_NAMES.get(features['symbol'], features['symbol']),
            'close': round(features['close'], 2),
            'ml_prediction': 'BUY' if pred == 1 else 'SELL',
            'ml_confidence': round(max(proba) * 100, 1),
            'ml_buy_prob': round(proba[1] * 100, 1),
            'ml_sell_prob': round(proba[0] * 100, 1)
        })
        
        emoji = "🟢" if pred == 1 else "🔴"
        print(f"{emoji} {features['symbol']}: {'BUY' if pred == 1 else 'SELL'} ({max(proba)*100:.1f}%)")
    
    # Save predictions
    output_path = '/tmp/gonggu_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Predictions saved to {output_path}")
    return predictions

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        generate_predictions()
    else:
        model_data = train_model()
        if model_data:
            generate_predictions(model_data)
