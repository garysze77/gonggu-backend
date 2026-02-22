const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Serve static files from frontend
app.use(express.static(path.join(__dirname, '../frontend')));

// API: Get all stocks with RSI signals + ML predictions
app.get('/api/stocks', (req, res) => {
  try {
    const reportPath = path.join(__dirname, '../data/daily_report.json');
    const predictionsPath = path.join(__dirname, '../data/predictions.json');
    
    let report = { all_stocks: [], total_stocks: 0, buy_count: 0, sell_count: 0 };
    let predictions = [];
    
    if (fs.existsSync(reportPath)) {
      report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
    }
    
    if (fs.existsSync(predictionsPath)) {
      predictions = JSON.parse(fs.readFileSync(predictionsPath, 'utf-8'));
    }
    
    // Create prediction lookup
    const predMap = {};
    predictions.forEach(p => {
      predMap[p.symbol.replace('.HK', '')] = p;
    });
    
    // Transform to frontend format with ML predictions
    const stocks = report.all_stocks.map(s => {
      const pred = predMap[s.symbol] || {};
      return {
        symbol: s.symbol,
        name: s.name,
        close: s.close,
        change: '0',
        signal: s.signal,
        rsi14: s.rsi_14,
        // ML prediction fields
        ml_prediction: pred.prediction || null,
        ml_confidence: pred.confidence || null,
      };
    });
    
    // Count signals
    const buyStocks = stocks.filter(s => s.signal === 'BUY');
    const sellStocks = stocks.filter(s => s.signal === 'SELL');
    const mlBuy = stocks.filter(s => s.ml_prediction === 'BUY');
    const mlSell = stocks.filter(s => s.ml_prediction === 'SELL');
    
    res.json({
      total_stocks: stocks.length,
      buy_count: buyStocks.length,
      sell_count: sellStocks.length,
      ml_buy_count: mlBuy.length,
      ml_sell_count: mlSell.length,
      all_stocks: stocks
    });
    
  } catch (error) {
    console.error('Error:', error);
    res.json({ error: error.message, all_stocks: [] });
  }
});

// API: Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date() });
});

// API: Trigger data fetch + ML training
app.post('/api/fetch', (req, res) => {
  try {
    console.log('Fetching data and training ML...');
    
    // Run data fetch
    execSync('python3 /root/.openclaw/workspace/hk-stock-prediction/main.py', {
      cwd: '/root/.openclaw/workspace/hk-stock-prediction',
      timeout: 120000
    });
    
    // Run ML training and predictions
    execSync('python3 /root/.openclaw/workspace/hk-stock-prediction/ml_train.py', {
      cwd: '/root/.openclaw/workspace/hk-stock-prediction',
      timeout: 300000
    });
    
    res.json({ success: true, message: 'Data fetched and ML trained successfully' });
  } catch (error) {
    console.error('Error:', error);
    res.json({ success: false, error: error.message });
  }
});

// API: Get ML predictions only
app.get('/api/predictions', (req, res) => {
  try {
    const predictionsPath = path.join(__dirname, '../data/predictions.json');
    
    if (!fs.existsSync(predictionsPath)) {
      return res.json({ error: 'No predictions available', predictions: [] });
    }
    
    const predictions = JSON.parse(fs.readFileSync(predictionsPath, 'utf-8'));
    res.json({ predictions });
    
  } catch (error) {
    console.error('Error:', error);
    res.json({ error: error.message, predictions: [] });
  }
});

// Serve frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 GongGu server running on http://localhost:${PORT}`);
});
