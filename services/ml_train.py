#!/usr/bin/env python3
"""
GongGu ML Training Module - Version 2
Uses ALL historical daily data as training samples
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

def calculate_features_for_date(closes, highs, lows, volumes, idx):
    """Calculate all features for a specific date index"""
    if idx < 60:  # Need at least 60 days of history
        return None
    
    prices = closes[:idx+1]
    
    features = {}
    
    # RSI (last 14 days before current)
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    features['rsi14'] = rsi.iloc[-1] if not rsi.empty else 50
    
    # MACD
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features['macd_histogram'] = (macd_line.iloc[-1] - signal_line.iloc[-1]) if not macd_line.empty else 0
    
    # Moving Averages
    features['ma5'] = prices.rolling(window=5).mean().iloc[-1]
    features['ma10'] = prices.rolling(window=10).mean().iloc[-1]
    features['ma20'] = prices.rolling(window=20).mean().iloc[-1]
    features['ma50'] = prices.rolling(window=50).mean().iloc[-1]
    
    # MA Crossovers
    features['ma5_above_ma20'] = 1 if features['ma5'] > features['ma20'] else 0
    features['ma5_above_ma50'] = 1 if features['ma5'] > features['ma50'] else 0
    
    # Bollinger Bands
    ma20 = prices.rolling(window=20).mean()
    std20 = prices.rolling(window=20).std()
    bb_upper = ma20 + (std20 * 2)
    bb_lower = ma20 - (std20 * 2)
    features['bb_position'] = (prices.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5
    
    # Volume
    vol_ma = volumes.rolling(window=20).mean()
    features['volume_ratio'] = volumes.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1
    
    # Momentum
    features['momentum_5'] = prices.iloc[-1] / prices.iloc[-6] - 1 if len(prices) >= 6 else 0
    features['momentum_10'] = prices.iloc[-1] / prices.iloc[-11] - 1 if len(prices) >= 11 else 0
    
    # Volatility
    returns = prices.pct_change()
atility'] = features['vol.rolling(window=20).std().iloc[-1] if not returns.empty else 0
    
    # Price position
    features['price_to_high'] = prices.iloc[-1] / highs.iloc[-20:].max() if len(highs) >= 20 else 1
    features['price_to_low'] = prices.iloc[-1] / lows.iloc[-20:].min() if len(lows) >= 20 else 1
    
    # Target: Next day direction (1=up, 0=down)
    # Need to look at the NEXT day
    if idx + 1 < len(closes):
        features['target'] = 1 if closes.iloc[idx+1] > closes.iloc[idx] else 0
    else:
        return None  # Can't determine target for the last day
    
    return features

def fetch_and_prepare_all_data(symbol):
    """Fetch ALL historical data and create training samples for each day"""
    try:
        print(f"  Processing {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='2y')
        
        if df.empty or len(df) < 120:  # Need at least 120 days
            return []
        
        closes = df['Close']
        highs = df['High']
        lows = df['Low']
        volumes = df['Volume']
        
        all_samples = []
        
        # Create a training sample for each day (starting from day 60)
        for idx in range(60, len(closes) - 1):  # -1 because we need next day for target
            features = calculate_features_for_date(closes, highs, lows, volumes, idx)
            if features:
                features['symbol'] = symbol.replace('.HK', '')
                features['date'] = closes.index[idx].strftime('%Y-%m-%d')
                features['close'] = closes.iloc[idx]
                all_samples.append(features)
        
        return all_samples
        
    except Exception as e:
        print(f"    Error: {e}")
        return []

def prepare_training_data():
    """Prepare training data from ALL historical data"""
    print("\n📊 Fetching ALL historical data and calculating features...")
    
    all_samples = []
    
    for i, symbol in enumerate(STOCKS):
        print(f"  [{i+1}/{len(STOCKS)}]", end=" ")
        samples = fetch_and_prepare_all_data(symbol)
        all_samples.extend(samples)
        if samples:
            print(f"→ {len(samples)} samples")
    
    if not all_samples:
        print("❌ No data collected")
        return None, None
    
    df = pd.DataFrame(all_samples)
    
    # Feature columns
    feature_cols = [
        'rsi14', 'macd_histogram', 'ma5_above_ma20', 'ma5_above_ma50',
        'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10',
        'volatility', 'price_to_high', 'price_to_low'
    ]
    
    # Drop rows with NaN
    df = df.dropna(subset=feature_cols + ['target'])
    
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"\n✅ Total training samples: {len(X)}")
    print(f"   Up days: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   Down days: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    return X, y

def train_model():
    """Train the ML model"""
    print("\n" + "="*60)
    print("🎯 Training ML Model with ALL Historical Data")
    print("="*60)
    
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
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
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
    """Generate predictions for current day"""
    print("\n" + "="*60)
    print("🔮 Generating Predictions")
    print("="*60)
    
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
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='2y')
            
            if df.empty or len(df) < 60:
                continue
            
            closes = df['Close']
            highs = df['High']
            lows = df['Low']
            volumes = df['Volume']
            
            # Get features for the last day
            features = calculate_features_for_date(closes, highs, lows, volumes, len(closes) - 1)
            
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
            
        except Exception as e:
            print(f"  Error with {symbol}: {e}")
    
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
