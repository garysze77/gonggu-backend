const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const cron = require('node-cron');
const { exec } = require('child_process');
const fs = require('fs');
const util = require('util');

const execPromise = util.promisify(exec);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const pool = new Pool({
  user: 'gonggu_user',
  password: 'gonggu123',
  host: 'localhost',
  database: 'gonggu',
  port: 5432,
});

const STOCKS = [
  { symbol: '0700.HK', name: '騰訊' },
  { symbol: '3690.HK', name: '美團' },
  { symbol: '9988.HK', name: '阿里巴巴' },
  { symbol: '9618.HK', name: '京東' },
  { symbol: '1024.HK', name: '快手' },
  { symbol: '9999.HK', name: '網易' },
  { symbol: '0939.HK', name: '建設銀行' },
  { symbol: '3988.HK', name: '中國銀行' },
  { symbol: '0005.HK', name: 'HSBC' },
  { symbol: '0388.HK', name: '港交所' },
  { symbol: '1113.HK', name: '長實' },
  { symbol: '0001.HK', name: '長江' },
  { symbol: '0012.HK', name: '恒地' },
  { symbol: '0016.HK', name: '恒大' },
  { symbol: '0011.HK', name: '恆生銀行' },
  { symbol: '1109.HK', name: '置地' },
  { symbol: '0762.HK', name: '中移動' },
  { symbol: '6822.HK', name: '香港電訊' },
  { symbol: '0883.HK', name: '中海油' },
  { symbol: '0857.HK', name: '中石油' },
  { symbol: '0291.HK', name: '華潤啤酒' },
  { symbol: '2319.HK', name: '蒙牛乳業' },
  { symbol: '0019.HK', name: '太古' },
  { symbol: '1177.HK', name: '中生製藥' },
  { symbol: '2269.HK', name: '藥明生物' },
  { symbol: '0669.HK', name: '創科' },
  { symbol: '2800.HK', name: '盈富基金' },
  { symbol: '2828.HK', name: '恒生ETF' },
];

// Fetch from Yahoo Finance
async function fetchAllStocks() {
  const symbols = STOCKS.map(s => s.symbol).join(' ');
  
  const pythonScript = `
import yfinance as yf
import json
try:
    tickers = yf.Tickers('${symbols}')
    results = {}
    for symbol, ticker in tickers.tickers.items():
        try:
            info = ticker.info
            results[symbol] = {
                'success': True,
                'price': info.get('currentPrice') or info.get('regularMarketPreviousClose'),
                'change': info.get('regularMarketChange') or 0,
                'changePercent': info.get('regularMarketChangePercent') or 0,
                'volume': info.get('regularMarketVolume') or 0,
                'dayHigh': info.get('dayHigh') or 0,
                'dayLow': info.get('dayLow') or 0,
            }
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
    print(json.dumps(results))
except Exception as e:
    print(json.dumps({'error': str(e)}))
`;
  
  fs.writeFileSync('/tmp/fetch_stocks.py', pythonScript);
  
  try {
    const { stdout } = await execPromise('python3 /tmp/fetch_stocks.py', { timeout: 60000 });
    return JSON.parse(stdout);
  } catch (error) {
    console.error('Fetch error:', error.message);
    return {};
  }
}

function analyzeStock(data, symbol, name) {
  if (!data || !data.success) return null;
  
  const price = data.price || 0;
  const dayHigh = data.dayHigh || price * 1.02;
  const dayLow = data.dayLow || price * 0.98;
  
  let signal = 'HOLD';
  if (price >= dayHigh * 0.98) signal = 'BUY';
  else if (price <= dayLow * 1.02) signal = 'SELL';
  
  return {
    symbol: symbol.replace('.HK', ''),
    name: name,
    close: price,
    change: data.changePercent ? data.changePercent.toFixed(2) : '0.00',
    volume: data.volume || 0,
    signal: signal
  };
}

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date() });
});

app.get('/api/stocks', async (req, res) => {
  try {
    console.log('Fetching stock data from Yahoo Finance...');
    const allData = await fetchAllStocks();
    console.log('Fetched:', Object.keys(allData).length, 'stocks');
    
    const results = [];
    for (const stock of STOCKS) {
      const data = allData[stock.symbol];
      const analysis = analyzeStock(data, stock.symbol, stock.name);
      if (analysis) results.push(analysis);
    }
    
    const buyStocks = results.filter(s => s.signal === 'BUY');
    const sellStocks = results.filter(s => s.signal === 'SELL');
    
    const today = new Date().toISOString().split('T')[0];
    await pool.query(
      'INSERT INTO daily_signals (date, buy_count, sell_count, signals) VALUES ($1, $2, $3, $4) ON CONFLICT (date) DO UPDATE SET buy_count = $2, sell_count = $3, signals = $4',
      [today, buyStocks.length, sellStocks.length, JSON.stringify(results)]
    );
    
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
    if (result.rows.length > 0) res.json(result.rows[0]);
    else res.json({ message: 'No data yet' });
  } catch (error) { res.status(500).json({ error: error.message }); }
});

cron.schedule('30 1 * * *', async () => {
  console.log('Running daily scan...');
  const allData = await fetchAllStocks();
  const results = [];
  for (const stock of STOCKS) {
    const data = allData[stock.symbol];
    const analysis = analyzeStock(data, stock.symbol, stock.name);
    if (analysis) results.push(analysis);
  }
  console.log('Done:', results.filter(r => r.signal === 'BUY').length, 'BUY');
});

async function initDB() {
  await pool.query(
    'CREATE TABLE IF NOT EXISTS daily_signals (date DATE PRIMARY KEY, buy_count INTEGER, sell_count INTEGER, signals JSONB, created_at TIMESTAMP DEFAULT NOW())'
  );
  console.log('Database initialized');
}

initDB().then(() => {
  app.listen(PORT, () => { console.log('GongGu API running on port ' + PORT); });
});
