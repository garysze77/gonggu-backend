#!/usr/bin/env python3
"""
GongGu ML Training Module
Stock Price Prediction using Machine Learning
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import pickle
import os
from datetime import datetime, timedelta

# HK Stocks to train
STOCKS = [
    '0700.HK', '3690.HK', '9988.HK', '9618.HK', '1024.HK',
    '0939.HK', '3988.HK', '0005.HK', '0388.HK', '1113.HK',
    '0001.HK', '0012.HK', '1109.HK', '0762.HK', '6822.HK',
    '0883.HK', '0857.HK', '0291.HK', '2319.HK', '1177.HK',
]

def fetch_stock_data(symbol, period='2y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def calculate_features(df):
    """Calculate technical indicators as features"""
    if df is None or len(df) < 60:
        return None
    
    df = df.copy()
    
    # Price changes
    df['returns'] = df['Close'].pct_change()
    df['price_change'] = df['Close'] - df['Close'].shift(1)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma10'] = df['Close'].rolling(window=10).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['ma50'] = df['Close'].rolling(window=50).mean()
    
    # MA signals
    df['ma5_above_ma20'] = (df['ma5'] > df['ma20']).astype(int)
    df['ma10_above_ma20'] = (df['ma10'] > df['ma20']).astype(int)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Volume features
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Momentum
    df['momentum'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Target: Next day direction (1=up, 0=down)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df

def prepare_training_data(stock_symbols):
    """Prepare training data from multiple stocks"""
    all_features = []
    all_targets = []
    
    for symbol in stock_symbols:
        print(f"Processing {symbol}...")
        df = fetch_stock_data(symbol)
        
        if df is None or len(df) < 60:
            continue
        
        df = calculate_features(df)
        
        if df is None:
            continue
        
        # Feature columns
        feature_cols = [
            'rsi', 'ma5_above_ma20', 'ma10_above_ma20',
            'volatility', 'volume_ratio', 'bb_position',
            'macd_histogram', 'momentum'
        ]
        
        # Drop NaN rows
        df_clean = df.dropna(subset=feature_cols + ['target'])
        
        if len(df_clean) < 30:
            continue
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        all_features.append(X)
        all_targets.append(y)
    
    if not all_features:
        return None, None
    
    # Combine all stocks
    X = np.vstack(all_features)
    y = np.concatenate(all_targets)
    
    return X, y

def train_model():
    """Train the ML model"""
    print("=" * 50)
    print("GongGu ML Training")
    print("=" * 50)
    
    # Prepare data
    X, y = prepare_training_data(STOCKS)
    
    if X is None:
        print("❌ Not enough data to train")
        return None
    
    print(f"\n📊 Total samples: {len(X)}")
    print(f"   Buy signals: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   Sell signals: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model (Gradient Boosting)
    print("\n🎯 Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
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
    feature_names = ['RSI', 'MA5>MA20', 'MA10>MA20', 'Volatility', 
                     'Volume Ratio', 'BB Position', 'MACD Hist', 'Momentum']
    importances = model.feature_importances_
    
    print(f"\n🔍 Feature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"   {name}: {imp*100:.1f}%")
    
    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'stocks': STOCKS,
        'trained_at': datetime.now().isoformat()
    }
    
    model_path = '/root/.openclaw/workspace/hk-stock-prediction/data/ml_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to {model_path}")
    
    return model_data

def predict_stock(symbol):
    """Predict next day movement for a stock"""
    model_path = '/root/.openclaw/workspace/hk-stock-prediction/data/ml_model.pkl'
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Run training first.")
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Fetch latest data
    df = fetch_stock_data(symbol)
    df = calculate_features(df)
    
    if df is None:
        return None
    
    feature_cols = [
        'rsi', 'ma5_above_ma20', 'ma10_above_ma20',
        'volatility', 'volume_ratio', 'bb_position',
        'macd_histogram', 'momentum'
    ]
    
    # Get latest features
    X = df[feature_cols].dropna().iloc[-1:].values
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return {
        'symbol': symbol,
        'prediction': 'BUY' if prediction == 1 else 'SELL',
        'confidence': max(probability) * 100,
        'buy_probability': probability[1] * 100,
        'sell_probability': probability[0] * 100
    }

def generate_predictions():
    """Generate predictions for all stocks"""
    print("\n" + "=" * 50)
    print("Generating Predictions")
    print("=" * 50)
    
    predictions = []
    
    for symbol in STOCKS:
        result = predict_stock(symbol)
        if result:
            predictions.append(result)
            emoji = "🟢" if result['prediction'] == 'BUY' else "🔴"
            print(f"{emoji} {symbol}: {result['prediction']} (confidence: {result['confidence']:.1f}%)")
    
    # Save predictions
    output_path = '/root/.openclaw/workspace/hk-stock-prediction/data/predictions.json'
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n✅ Predictions saved to {output_path}")
    
    return predictions

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'predict':
            generate_predictions()
        else:
            print("Usage: python3 ml_train.py [predict]")
    else:
        train_model()
        generate_predictions()
