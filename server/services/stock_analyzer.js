// Stock Analyzer - Multiple Indicators + ML Features
const STOCKS = {
  // Blue Chips
  "700": "騰訊", "3690": "美團", "9988": "阿里巴巴", "09988": "阿里-SW",
  "9618": "京東", "1024": "快手", "9999": "網易",
  // Financials
  "939": "建設銀行", "3988": "中銀", "0005": "HSBC", "2388": "港交所",
  "1113": "長實", "0001": "長江", "0012": "恒地", "0016": "恒大",
  // Tech
  "6690": "海爾智家", "6060": "眾安", "1858": "青島啤酒",
  // Telecom
  "0762": "中移動", "6822": "香港電訊",
  // Energy
  "0883": "中海油", "0857": "中石油",
  // Food/Retail
  "291": "華潤啤酒", "2319": "蒙牛", "0019": "太古",
  // Healthcare
  "1177": "中生製藥", "2269": "藥明生物", "0669": "創科",
  // ETFs
  "2800": "盈富基金", "2828": "恒生ETF", "3032": "南方A50",
  // Index
  "HSI": "恆生指數", "HSCEI": "國企指數",
};

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
  for (let i = period; i < prices.length; i++) { ema = (prices[i] - ema) * multiplier + ema; emaValues.push(ema); }
  return emaValues;
}

function calculateMACD(prices) {
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);
  if (!ema12 || !ema26) return null;
  const macdLine = ema12.map((v, i) => v - ema26[ema26.length - ema12.length + i]);
  const signalLine = calculateEMA(macdLine, 9);
  if (!signalLine) return null;
  const hist = macdLine[macdLine.length - 1] - signalLine[signalLine.length - 1];
  return { macd: macdLine[macdLine.length - 1], signal: signalLine[signalLine.length - 1], histogram: hist };
}

function calculateMA(prices, period) {
  const maValues = [];
  for (let i = period - 1; i < prices.length; i++) {
    const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
    maValues.push(sum / period);
  }
  return maValues;
}

function calculateBollingerBands(prices, period = 20, stdDev = 2) {
  const ma = calculateMA(prices, period);
  if (!ma) return null;
  const upper = [], lower = [];
  for (let i = 0; i < ma.length; i++) {
    const slice = prices.slice(i, i + period);
    const mean = ma[i];
    const variance = slice.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / period;
    const std = Math.sqrt(variance);
    upper.push(mean + stdDev * std);
    lower.push(mean - stdDev * std);
  }
  return { upper: upper[upper.length - 1], middle: ma[ma.length - 1], lower: lower[lower.length - 1] };
}

function analyzeStock(data) {
  const { symbol, closes, timestamps, highs, lows, volumes } = data;
  const result = { symbol, name: STOCKS[symbol] || symbol, close: closes[closes.length - 1], date: new Date(timestamps[timestamps.length - 1] * 1000).toISOString().split('T')[0] };
  
  // RSI
  const rsi14 = calculateRSI(closes, 14);
  result.rsi14 = rsi14 ? rsi14[rsi14.length - 1].toFixed(2) : null;
  
  // MACD
  const macd = calculateMACD(closes);
  if (macd) {
    result.macd = macd.macd.toFixed(4);
    result.macdSignal = macd.signal.toFixed(4);
    result.macdHistogram = macd.histogram.toFixed(4);
  }
  
  // Moving Averages
  const ma5 = calculateMA(closes, 5), ma10 = calculateMA(closes, 10), ma20 = calculateMA(closes, 20), ma50 = calculateMA(closes, 50);
  result.ma5 = ma5 ? ma5[ma5.length - 1].toFixed(2) : null;
  result.ma10 = ma10 ? ma10[ma10.length - 1].toFixed(2) : null;
  result.ma20 = ma20 ? ma20[ma20.length - 1].toFixed(2) : null;
  result.ma50 = ma50 ? ma50[ma50.length - 1].toFixed(2) : null;
  
  // Bollinger Bands
  const bb = calculateBollingerBands(closes);
  if (bb) { result.bbUpper = bb.upper.toFixed(2); result.bbMiddle = bb.middle.toFixed(2); result.bbLower = bb.lower.toFixed(2); }
  
  // Volume
  const volSMA = volumes ? calculateMA(volumes, 20) : null;
  const currentVol = volumes ? volumes[volumes.length - 1] : 0;
  result.volume = currentVol;
  result.volumeSMA = volSMA ? volSMA[volSMA.length - 1].toFixed(0) : null;
  result.volumeRatio = (volSMA && currentVol) ? (currentVol / volSMA[volSMA.length - 1]).toFixed(2) : null;
  
  // Price change
  if (closes.length >= 2) result.change = ((closes[closes.length - 1] - closes[closes.length - 2]) / closes[closes.length - 2] * 100).toFixed(2);
  
  // Generate Signal
  let buyScore = 0, sellScore = 0;
  if (result.rsi14 < 30) buyScore += 3;
  if (result.rsi14 > 70) sellScore += 3;
  if (result.rsi14 < 40) buyScore += 1;
  if (result.rsi14 > 60) sellScore += 1;
  if (result.macdHistogram > 0) buyScore += 2;
  if (result.macdHistogram < 0) sellScore += 2;
  if (result.ma5 > result.ma20) buyScore += 2;
  if (result.ma5 < result.ma20) sellScore += 2;
  if (result.volumeRatio > 1.5) buyScore += 1;
  
  if (buyScore > sellScore + 2) result.signal = 'BUY';
  else if (sellScore > buyScore + 2) result.signal = 'SELL';
  else if (buyScore > sellScore) result.signal = 'BUY';
  else if (sellScore > buyScore) result.signal = 'SELL';
  else result.signal = 'HOLD';
  
  result.buyScore = buyScore;
  result.sellScore = sellScore;
  
  return result;
}

module.exports = { STOCKS, analyzeStock, calculateRSI, calculateMACD, calculateMA, calculateBollingerBands };
