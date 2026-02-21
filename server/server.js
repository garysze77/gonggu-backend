const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const cron = require('node-cron');
const axios = require('axios');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Database connection
const pool = new Pool({
  user: 'gonggu_user',
  password: 'gonggu123',
  host: 'localhost',
  database: 'gonggu',
  port: 5432,
});

// EJFQ Headers
const EJFQ_HEADERS = {
  'authority': 'www.ejfq.com',
  'accept': '*/*',
  'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
  'content-type': 'text/plain',
  'referer': 'https://www.ejfq.com/home/tc/screener360.htm',
  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
  'x-requested-with': 'XMLHttpRequest'
};

// Stock list
const STOCKS = {
  "700": "騰訊",
  "3690": "美團",
  "9988": "阿里巴巴",
  "939": "建設銀行",
  "3988": "中國銀行",
  "0005": "HSBC",
  "2388": "港交所",
  "0016": "恒大地產",
  "0001": "長江",
  "0762": "中國移動",
  "0883": "中海油"
};

// Calculate RSI
function calculateRSI(prices, period = 14) {
  if (!prices || prices.length < period + 1) return null;
  
  let gains = [], losses = [];
  for (let i = 1; i < prices.length; i++) {
    const change = prices[i] - prices[i-1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }
  
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
  }
  
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

// Fetch stock data
async function fetchStockData(symbol) {
  try {
    const url = `https://www.ejfq.com/home/tc/tradingview3_360/php/chartfeed.php?symbol=${symbol}&resolution=D&method=history`;
    const response = await axios.get(url, { headers: EJFQ_HEADERS });
    return response.data;
  } catch (error) {
    console.error(`Error fetching ${symbol}:`, error.message);
    return null;
  }
}

// Analyze stock
async function analyzeStock(symbol) {
  const data = await fetchStockData(symbol);
  
  if (!data || data.s !== 'ok') {
    return null;
  }
  
  const closes = data.c;
  const rsi14 = calculateRSI(closes, 14);
  
  let signal = 'HOLD';
  let reason = '';
  
  if (rsi14 !== null) {
    if (rsi14 <= 30) {
      signal = 'BUY';
      reason = `RSI(${rsi14.toFixed(1)}) 超賣`;
    } else if (rsi14 >= 70) {
      signal = 'SELL';
      reason = `RSI(${rsi14.toFixed(1)}) 超買`;
    } else {
      reason = `RSI(${rsi14.toFixed(1)}) 中性`;
    }
  }
  
  return {
    symbol,
    name: STOCKS[symbol] || symbol,
    close: closes[closes.length - 1],
    rsi14: rsi14 ? rsi14.toFixed(2) : null,
    signal,
    reason,
    date: new Date().toISOString().split('T')[0]
  };
}

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date() });
});

app.get('/api/stocks', async (req, res) => {
  try {
    const results = [];
    for (const symbol of Object.keys(STOCKS)) {
      const analysis = await analyzeStock(symbol);
      if (analysis) {
        results.push(analysis);
      }
    }
    
    // Save to database
    const buyStocks = results.filter(s => s.signal === 'BUY');
    const sellStocks = results.filter(s => s.signal === 'SELL');
    
    // Store in DB (simple version)
    await pool.query(`
      INSERT INTO daily_signals (date, buy_count, sell_count, signals)
      VALUES ($1, $2, $3, $4)
      ON CONFLICT (date) DO UPDATE SET
        buy_count = $2, sell_count = $3, signals = $4
    `, [
      new Date().toISOString().split('T')[0],
      buyStocks.length,
      sellStocks.length,
      JSON.stringify(results)
    ]);
    
    res.json({
      generated_at: new Date().toISOString(),
      total_stocks: results.length,
      buy_signals: buyStocks,
      sell_signals: sellStocks,
      all_stocks: results
    });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/signals', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM daily_signals ORDER BY date DESC LIMIT 1');
    if (result.rows.length > 0) {
      res.json(result.rows[0]);
    } else {
      res.json({ message: 'No data yet' });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Daily scan at 9:30 AM
cron.schedule('30 1 * * *', async () => {
  console.log('Running daily scan...');
  const results = [];
  for (const symbol of Object.keys(STOCKS)) {
    const analysis = await analyzeStock(symbol);
    if (analysis) {
      results.push(analysis);
    }
  }
  console.log('Daily scan complete:', results.filter(r => r.signal === 'BUY').length, 'BUY signals');
});

// Create table on startup
async function initDB() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS daily_signals (
      date DATE PRIMARY KEY,
      buy_count INTEGER,
      sell_count INTEGER,
      signals JSONB,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log('Database initialized');
}

// Start server
initDB().then(() => {
  app.listen(PORT, () => {
    console.log(`GongGu API running on port ${PORT}`);
  });
});
