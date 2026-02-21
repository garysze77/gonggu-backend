const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const cron = require('node-cron');
const axios = require('axios');

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

const WORKER_URL = 'https://hkstockdata.garysze77.workers.dev/?url=';

const EJFQ_HEADERS = {
  'authority': 'www.ejfq.com',
  'accept': '*/*',
  'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
  'content-type': 'text/plain',
  'referer': 'https://www.ejfq.com/home/tc/screener360.htm',
  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
  'x-requested-with': 'XMLHttpRequest'
};

const STOCKS = {
  '700': '騰訊', '3690': '美團', '9988': '阿里巴巴', '09988': '阿里-SW',
  '9618': '京東', '1024': '快手', '9999': '網易',
  '939': '建設銀行', '3988': '中國銀行', '0005': 'HSBC', '2388': '港交所',
  '1113': '長實', '0001': '長江', '0012': '恒地', '0016': '恒大',
  '0011': '恆生銀行', '1109': '置地', '6830': '華潤置地',
  '6690': '海爾智家', '6060': '眾安', '1858': '青島啤酒',
  '0762': '中移動', '6822': '香港電訊',
  '0883': '中海油', '0857': '中石油',
  '291': '華潤啤酒', '2319': '蒙牛乳業', '0019': '太古',
  '1177': '中生製藥', '2269': '藥明生物', '0669': '創科',
  '0836': '華潤置地', '1108': '熊貓',
  '2800': '盈富基金', '2828': '恒生ETF', '3032': '南方A50',
  'HSI': '恆生指數', 'HSCEI': '國企指數', '2618': 'TCL電子', '0688': '中國海外', '0728': '中國電信', '0175': '吉利汽車', '1211': '比亞迪股份', '0027': '銀河娛樂', '0188': '中外運', '0269': '中遠海運', '0195': '新奧能源', '1171': '兗州煤業', '0116': 'Volvo', '0233': '創科實業', '0696': '首程控股', '0386': '中石化', '0857': '中石油', '0883': '中海油', '1109': '華潤置地', '0330': '思愛普', '0019': '太古股份' ,
};

function calculateRSI(prices, period) {
  period = period || 14;
  if (!prices || prices.length < period + 1) return null;
  let gains = [], losses = [];
  for (let i = 1; i < prices.length; i++) {
    const change = prices[i] - prices[i-1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const rsiValues = [];
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    if (avgLoss === 0) { rsiValues.push(100); }
    else { const rs = avgGain / avgLoss; rsiValues.push(100 - (100 / (1 + rs))); }
  }
  return rsiValues;
}

function calculateEMA(prices, period) {
  if (!prices || prices.length < period) return null;
  const multiplier = 2 / (period + 1);
  let ema = prices.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const emaValues = [];
  for (let i = period; i < prices.length; i++) { 
    ema = (prices[i] - ema) * multiplier + ema; 
    if (!isNaN(ema) && isFinite(ema)) emaValues.push(ema); 
  }
  return emaValues.length > 0 ? emaValues : null;
}

function calculateMACD(prices) {
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);
  if (!ema12 || !ema26 || ema12.length < 9) return null;
  const macdLine = ema12.map((v, i) => v - ema26[ema26.length - ema12.length + i]);
  const signalLine = calculateEMA(macdLine, 9);
  if (!signalLine || signalLine.length === 0) return null;
  return { macd: macdLine[macdLine.length - 1], signal: signalLine[signalLine.length - 1], histogram: macdLine[macdLine.length - 1] - signalLine[signalLine.length - 1] };
}

function calculateMA(prices, period) {
  if (!prices || prices.length < period) return null;
  const maValues = [];
  for (let i = period - 1; i < prices.length; i++) { maValues.push(prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period); }
  return maValues;
}

function calculateBollingerBands(prices, period) {
  period = period || 20;
  const ma = calculateMA(prices, period);
  if (!ma) return null;
  const upper = [], lower = [];
  for (let i = 0; i < ma.length; i++) {
    const slice = prices.slice(i, i + period);
    const mean = ma[i];
    const variance = slice.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / period;
    const std = Math.sqrt(variance);
    upper.push(mean + 2 * std);
    lower.push(mean - 2 * std);
  }
  return { upper: upper[upper.length - 1], middle: ma[ma.length - 1], lower: lower[lower.length - 1] };
}

function safeNum(val) {
  if (val === null || val === undefined || isNaN(val) || !isFinite(val)) return null;
  return parseFloat(val);
}

function safeFixed(val, decimals) {
  const n = safeNum(val);
  return n !== null ? n.toFixed(decimals) : null;
}

async function fetchStockData(symbol) {
  try {
    const targetUrl = encodeURIComponent('https://www.ejfq.com/home/tc/tradingview3_360/php/chartfeed.php?symbol=' + symbol + '&resolution=D&method=history');
    const url = WORKER_URL + targetUrl;
    const response = await axios.get(url, { headers: EJFQ_HEADERS });
    return response.data;
  } catch (error) {
    console.error('Error fetching ' + symbol + ':', error.message);
    return null;
  }
}

async function analyzeStock(symbol) {
  const data = await fetchStockData(symbol);
  if (!data || data.s !== 'ok') return null;
  
  const closes = data.c;
  const timestamps = data.t;
  const volumes = data.v;
  
  const result = {
    symbol: symbol,
    name: STOCKS[symbol] || symbol,
    close: closes[closes.length - 1],
    date: new Date(timestamps[timestamps.length - 1] * 1000).toISOString().split('T')[0]
  };
  
  const rsi14 = calculateRSI(closes, 14);
  result.rsi14 = safeFixed(rsi14 ? rsi14[rsi14.length - 1] : null, 2);
  
  const macd = calculateMACD(closes);
  if (macd) {
    result.macd = safeFixed(macd.macd, 4);
    result.macdSignal = safeFixed(macd.signal, 4);
    result.macdHistogram = safeFixed(macd.histogram, 4);
  }
  
  const ma5 = calculateMA(closes, 5), ma10 = calculateMA(closes, 10), ma20 = calculateMA(closes, 20), ma50 = calculateMA(closes, 50);
  result.ma5 = safeFixed(ma5 ? ma5[ma5.length - 1] : null, 2);
  result.ma10 = safeFixed(ma10 ? ma10[ma10.length - 1] : null, 2);
  result.ma20 = safeFixed(ma20 ? ma20[ma20.length - 1] : null, 2);
  result.ma50 = safeFixed(ma50 ? ma50[ma50.length - 1] : null, 2);
  
  const bb = calculateBollingerBands(closes);
  if (bb) { result.bbUpper = safeFixed(bb.upper, 2); result.bbMiddle = safeFixed(bb.middle, 2); result.bbLower = safeFixed(bb.lower, 2); }
  
  const volMA = volumes ? calculateMA(volumes, 20) : null;
  const currentVol = volumes ? volumes[volumes.length - 1] : 0;
  result.volume = currentVol;
  result.volumeSMA = safeFixed(volMA ? volMA[volMA.length - 1] : null, 0);
  result.volumeRatio = safeFixed((volMA && currentVol) ? currentVol / volMA[volMA.length - 1] : null, 2);
  
  if (closes.length >= 2) result.change = safeFixed(((closes[closes.length - 1] - closes[closes.length - 2]) / closes[closes.length - 2] * 100), 2);
  
  let buyScore = 0, sellScore = 0;
  if (result.rsi14 && result.rsi14 < 30) buyScore += 3;
  if (result.rsi14 && result.rsi14 > 70) sellScore += 3;
  if (result.rsi14 && result.rsi14 < 40) buyScore += 1;
  if (result.rsi14 && result.rsi14 > 60) sellScore += 1;
  if (result.macdHistogram && result.macdHistogram > 0) buyScore += 2;
  if (result.macdHistogram && result.macdHistogram < 0) sellScore += 2;
  if (result.ma5 && result.ma20 && result.ma5 > result.ma20) buyScore += 2;
  if (result.ma5 && result.ma20 && result.ma5 < result.ma20) sellScore += 2;
  if (result.volumeRatio && result.volumeRatio > 1.5) buyScore += 1;
  
  if (buyScore > sellScore + 2) result.signal = 'BUY';
  else if (sellScore > buyScore + 2) result.signal = 'SELL';
  else if (buyScore > sellScore) result.signal = 'BUY';
  else if (sellScore > buyScore) result.signal = 'SELL';
  else result.signal = 'HOLD';
  
  result.buyScore = buyScore;
  result.sellScore = sellScore;
  
  return result;
}

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date() });
});

app.get('/api/stocks', async (req, res) => {
  try {
    const results = [];
    for (const symbol of Object.keys(STOCKS)) {
      const analysis = await analyzeStock(symbol);
      if (analysis) results.push(analysis);
    }
    
    const buyStocks = results.filter(s => s.signal === 'BUY');
    const sellStocks = results.filter(s => s.signal === 'SELL');
    
    const today = new Date().toISOString().split('T')[0];
    const checkResult = await pool.query('SELECT date FROM daily_signals WHERE date = $1', [today]);
    
    if (checkResult.rows.length === 0) {
      await pool.query('INSERT INTO daily_signals (date, buy_count, sell_count, signals) VALUES ($1, $2, $3, $4)',
        [today, buyStocks.length, sellStocks.length, JSON.stringify(results)]);
    } else {
      await pool.query('UPDATE daily_signals SET buy_count = $1, sell_count = $2, signals = $3 WHERE date = $4',
        [buyStocks.length, sellStocks.length, JSON.stringify(results), today]);
    }
    
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

app.get('/api/stock/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const analysis = await analyzeStock(symbol);
    if (analysis) res.json(analysis);
    else res.status(404).json({ error: 'Stock not found' });
  } catch (error) { res.status(500).json({ error: error.message }); }
});

cron.schedule('30 1 * * *', async () => {
  console.log('Running daily scan...');
  const results = [];
  for (const symbol of Object.keys(STOCKS)) {
    const analysis = await analyzeStock(symbol);
    if (analysis) results.push(analysis);
  }
  console.log('Daily scan complete:', results.filter(r => r.signal === 'BUY').length, 'BUY signals');
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

// K-line data endpoint
app.get('/api/kline/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const data = await fetchStockData(symbol);
    if (!data || data.s !== 'ok') {
      return res.status(404).json({ error: 'Stock not found' });
    }
    
    const kline = [];
    for (let i = 0; i < data.t.length; i++) {
      kline.push({
        time: data.t[i],
        open: data.o[i],
        high: data.h[i],
        low: data.l[i],
        close: data.c[i],
        volume: data.v[i]
      });
    }
    
    res.json({ symbol: symbol, name: STOCKS[symbol] || symbol, kline: kline });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
