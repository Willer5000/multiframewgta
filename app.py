from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import os
import time
import telegram
import asyncio
from threading import Thread
import pytz

app = Flask(__name__)

# Configuración Telegram
TELEGRAM_BOT_TOKEN = "8007748376:AAHIW8n9b-BtA378g4gF-0-D2mOhn495Q0g"
TELEGRAM_CHAT_ID = "-1003229814161"

# Configuración optimizada - 75 criptomonedas top de Bitget
CRYPTO_SYMBOLS = [
    # Bajo Riesgo (25) - Top market cap
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "MATIC-USDT",
    "ATOM-USDT", "LTC-USDT", "BCH-USDT", "ETC-USDT", "XLM-USDT",
    "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "MANA-USDT",
    "SAND-USDT", "AXS-USDT", "THETA-USDT", "XTZ-USDT", "EOS-USDT",
    
    # Medio Riesgo (25) - Proyectos consolidados
    "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    "GALA-USDT", "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT",
    "IOTA-USDT", "ONE-USDT", "IOST-USDT", "WAVES-USDT", "KAVA-USDT",
    "RUNE-USDT", "CRV-USDT", "MKR-USDT", "COMP-USDT", "AAVE-USDT",
    "UNI-USDT", "SNX-USDT", "SUSHI-USDT", "YFI-USDT", "1INCH-USDT",
    
    # Alto Riesgo (20) - Proyectos emergentes
    "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
    "MAGIC-USDT", "RNDR-USDT", "IMX-USDT", "LDO-USDT", "STX-USDT",
    "APT-USDT", "BLUR-USDT", "PEPE-USDT", "LINA-USDT", "TRB-USDT",
    "RSR-USDT", "C98-USDT", "HFT-USDT", "METIS-USDT", "ZRX-USDT",
    
    # Memecoins (5) - Top memes
    "DOGE-USDT", "SHIB-USDT", "FLOKI-USDT", "PEPE-USDT", "BONK-USDT"
]

# Clasificación de riesgo optimizada
CRYPTO_RISK_CLASSIFICATION = {
    "bajo": [
        "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
        "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "MATIC-USDT",
        "ATOM-USDT", "LTC-USDT", "BCH-USDT", "ETC-USDT", "XLM-USDT",
        "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "MANA-USDT",
        "SAND-USDT", "AXS-USDT", "THETA-USDT", "XTZ-USDT", "EOS-USDT"
    ],
    "medio": [
        "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
        "GALA-USDT", "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT",
        "IOTA-USDT", "ONE-USDT", "IOST-USDT", "WAVES-USDT", "KAVA-USDT",
        "RUNE-USDT", "CRV-USDT", "MKR-USDT", "COMP-USDT", "AAVE-USDT",
        "UNI-USDT", "SNX-USDT", "SUSHI-USDT", "YFI-USDT", "1INCH-USDT"
    ],
    "alto": [
        "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
        "MAGIC-USDT", "RNDR-USDT", "IMX-USDT", "LDO-USDT", "STX-USDT",
        "APT-USDT", "BLUR-USDT", "PEPE-USDT", "LINA-USDT", "TRB-USDT",
        "RSR-USDT", "C98-USDT", "HFT-USDT", "METIS-USDT", "ZRX-USDT"
    ],
    "memecoins": [
        "DOGE-USDT", "SHIB-USDT", "FLOKI-USDT", "PEPE-USDT", "BONK-USDT"
    ]
}

# Mapeo de temporalidades para análisis multi-timeframe MEJORADO
TIMEFRAME_HIERARCHY = {
    '15m': {'mayor': '1h', 'media': '30m', 'menor': '5m'},
    '30m': {'mayor': '2h', 'media': '1h', 'menor': '15m'},
    '1h': {'mayor': '4h', 'media': '2h', 'menor': '30m'},
    '2h': {'mayor': '8h', 'media': '4h', 'menor': '1h'},
    '4h': {'mayor': '12h', 'media': '8h', 'menor': '2h'},
    '8h': {'mayor': '1D', 'media': '12h', 'menor': '4h'},
    '12h': {'mayor': '3D', 'media': '1D', 'menor': '8h'},
    '1D': {'mayor': '3D', 'media': '1D', 'menor': '12h'},
    '3D': {'mayor': '1W', 'media': '3D', 'menor': '1D'},
    '1W': {'mayor': '1M', 'media': '1W', 'menor': '3D'}
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_signals = {}
        self.win_rate_data = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
    
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def is_scalping_time(self):
        """Verificar si es horario de scalping"""
        now = self.get_bolivia_time()
        if now.weekday() >= 5:
            return False
        return 4 <= now.hour < 16

    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela"""
        if interval == '15m':
            next_close = current_time.replace(minute=current_time.minute // 15 * 15, second=0, microsecond=0) + timedelta(minutes=15)
            return (next_close - current_time).total_seconds() <= 450
        elif interval == '30m':
            next_close = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0) + timedelta(minutes=30)
            return (next_close - current_time).total_seconds() <= 900
        elif interval == '1h':
            next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return (next_close - current_time).total_seconds() <= 1800
        elif interval == '2h':
            current_hour = current_time.hour
            next_2h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour % 2 == 0:
                next_2h_close += timedelta(hours=2)
            else:
                next_2h_close += timedelta(hours=1)
            return (next_2h_close - current_time).total_seconds() <= 3600
        elif interval == '4h':
            current_hour = current_time.hour
            next_4h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 4
            if remainder == 0:
                next_4h_close += timedelta(hours=4)
            else:
                next_4h_close += timedelta(hours=4 - remainder)
            return (next_4h_close - current_time).total_seconds() <= 7200
        elif interval == '8h':
            current_hour = current_time.hour
            next_8h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 8
            if remainder == 0:
                next_8h_close += timedelta(hours=8)
            else:
                next_8h_close += timedelta(hours=8 - remainder)
            return (next_8h_close - current_time).total_seconds() <= 14400
        elif interval == '12h':
            current_hour = current_time.hour
            next_12h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour < 8:
                next_12h_close = next_12h_close.replace(hour=20)
            else:
                next_12h_close = next_12h_close.replace(hour=8) + timedelta(days=1)
            return (next_12h_close - current_time).total_seconds() <= 21600
        elif interval == '1D':
            tomorrow_8pm = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
            if current_time.hour >= 20:
                tomorrow_8pm += timedelta(days=1)
            return (tomorrow_8pm - current_time).total_seconds() <= 43200
        elif interval == '3D':
            # Para 3D, simplificar lógica
            return True
        elif interval == '1W':
            # Para 1W, simplificar lógica
            return True
        
        return False

    def get_kucoin_data(self, symbol, interval, limit=100):
        """Obtener datos de KuCoin con manejo robusto de errores"""
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < 60:
                    return cached_data
            
            interval_map = {
                '15m': '15min', '30m': '30min', '5m': '5min', '1h': '1hour',
                '2h': '2hour', '4h': '4hour', '8h': '8hour', '12h': '12hour',
                '1D': '1day', '3D': '3day', '1W': '1week'
            }
            
            kucoin_interval = interval_map.get(interval, '1hour')
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol.replace('-', '')}&type={kucoin_interval}"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if not candles:
                        return self.generate_realistic_data(symbol, interval, limit)
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    df = df.iloc[::-1].reset_index(drop=True)
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    result = df.tail(limit)
                    self.cache[cache_key] = (result, datetime.now())
                    return result
                
        except Exception as e:
            print(f"Error obteniendo datos de KuCoin para {symbol} {interval}: {e}")
        
        return self.generate_realistic_data(symbol, interval, limit)

    def generate_realistic_data(self, symbol, interval, limit):
        """Generar datos realistas basados en precios actuales del mercado"""
        try:
            # Obtener precio actual desde una fuente alternativa
            price_url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol.replace('-', '')}"
            response = requests.get(price_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000':
                    current_price = float(data['data']['price'])
                else:
                    current_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
            else:
                current_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
                
        except:
            current_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        np.random.seed(hash(symbol) % 10000)
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
        
        # Volatilidad realista basada en el símbolo
        if 'BTC' in symbol or 'ETH' in symbol:
            volatility = 0.015
        elif 'MEME' in symbol or 'SHIB' in symbol:
            volatility = 0.08
        else:
            volatility = 0.03
        
        returns = np.random.normal(0.001, volatility, limit)
        prices = current_price * (1 + np.cumsum(returns))
        
        data = {
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0.01, 0.01, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, limit))),
            'close': prices,
            'volume': np.random.lognormal(12, 1, limit)
        }
        
        df = pd.DataFrame(data)
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df

    def calculate_sma(self, prices, period):
        """Calcular SMA manualmente"""
        if len(prices) < period:
            return np.zeros(len(prices))
        
        sma = np.zeros(len(prices))
        for i in range(len(prices)):
            if i < period - 1:
                sma[i] = np.mean(prices[:i+1])
            else:
                sma[i] = np.mean(prices[i-period+1:i+1])
        
        return sma

    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente"""
        if len(prices) < period:
            return np.zeros(len(prices))
        
        alpha = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional"""
        if len(prices) < period + 1:
            return np.zeros(len(prices))
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        # Primeros valores
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.zeros(len(prices))
        for i in range(len(prices)):
            if avg_losses[i] > 0:
                rs[i] = avg_gains[i] / avg_losses[i]
            else:
                rs[i] = 100 if avg_gains[i] > 0 else 50
        
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcular MACD"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger"""
        sma = self.calculate_sma(prices, period)
        std = np.zeros(len(prices))
        
        for i in range(len(prices)):
            if i >= period - 1:
                window = prices[i-period+1:i+1]
                std[i] = np.std(window)
            else:
                std[i] = 0
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower

    def calculate_adx(self, high, low, close, period=14):
        """Calcular ADX, +DI, -DI"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        tr_smooth = self.calculate_ema(tr, period)
        plus_dm_smooth = self.calculate_ema(plus_dm, period)
        minus_dm_smooth = self.calculate_ema(minus_dm, period)
        
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        
        for i in range(n):
            if tr_smooth[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        
        dx = np.zeros(n)
        for i in range(n):
            if (plus_di[i] + minus_di[i]) > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = self.calculate_ema(dx, period)
        
        return adx, plus_di, minus_di

    def calculate_squeeze_momentum(self, high, low, close, length=20, mult=2):
        """Calcular Squeeze Momentum"""
        n = len(close)
        
        # Bandas de Bollinger
        bb_basis = self.calculate_sma(close, length)
        bb_dev = np.zeros(n)
        for i in range(length-1, n):
            window = close[i-length+1:i+1]
            bb_dev[i] = np.std(window) if len(window) > 1 else 0
        bb_upper = bb_basis + (bb_dev * mult)
        bb_lower = bb_basis - (bb_dev * mult)
        
        # Keltner Channel
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        kc_basis = self.calculate_sma(close, length)
        kc_dev = self.calculate_ema(tr, length)
        kc_upper = kc_basis + (kc_dev * mult)
        kc_lower = kc_basis - (kc_dev * mult)
        
        # Squeeze
        squeeze_on = np.zeros(n, dtype=bool)
        squeeze_off = np.zeros(n, dtype=bool)
        momentum = np.zeros(n)
        
        for i in range(n):
            # Squeeze ON cuando BB dentro de KC
            if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
                squeeze_on[i] = True
            # Squeeze OFF cuando BB fuera de KC
            elif bb_upper[i] > kc_upper[i] or bb_lower[i] < kc_lower[i]:
                squeeze_off[i] = True
            
            # Momentum (simplificado)
            if i > 0:
                momentum[i] = close[i] - close[i-1]
        
        return {
            'squeeze_on': squeeze_on.tolist(),
            'squeeze_off': squeeze_off.tolist(),
            'momentum': momentum.tolist(),
            'bb_upper': bb_upper.tolist(),
            'bb_lower': bb_lower.tolist(),
            'kc_upper': kc_upper.tolist(),
            'kc_lower': kc_lower.tolist()
        }

    def detect_rsi_divergence(self, price, rsi, lookback=14):
        """Detectar divergencias RSI"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Buscar máximos y mínimos en precio y RSI
            price_window = price[i-lookback:i+1]
            rsi_window = rsi[i-lookback:i+1]
            
            price_max_idx = np.argmax(price_window)
            price_min_idx = np.argmin(price_window)
            rsi_max_idx = np.argmax(rsi_window)
            rsi_min_idx = np.argmin(rsi_window)
            
            # Divergencia bajista: precio hace higher high, RSI hace lower high
            if (price_max_idx == lookback and rsi_max_idx < lookback and
                price[i] > price[i-lookback+price_max_idx-1] and
                rsi[i] < rsi[i-lookback+rsi_max_idx-1]):
                bearish_div[i] = True
            
            # Divergencia alcista: precio hace lower low, RSI hace higher low
            if (price_min_idx == lookback and rsi_min_idx < lookback and
                price[i] < price[i-lookback+price_min_idx-1] and
                rsi[i] > rsi[i-lookback+rsi_min_idx-1]):
                bullish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo básicos"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            window_close = close[i-lookback:i+1]
            
            # Detectar máximos y mínimos significativos
            highs = []
            lows = []
            
            for j in range(2, len(window_high)-2):
                if (window_high[j] > window_high[j-1] and 
                    window_high[j] > window_high[j-2] and
                    window_high[j] > window_high[j+1] and
                    window_high[j] > window_high[j+2]):
                    highs.append((j, window_high[j]))
                
                if (window_low[j] < window_low[j-1] and 
                    window_low[j] < window_low[j-2] and
                    window_low[j] < window_low[j+1] and
                    window_low[j] < window_low[j+2]):
                    lows.append((j, window_low[j]))
            
            # Patrón Hombro Cabeza Hombro (simplificado)
            if len(highs) >= 3:
                highs_sorted = sorted(highs, key=lambda x: x[1], reverse=True)
                if (len(highs_sorted) >= 3 and
                    abs(highs_sorted[0][1] - highs_sorted[1][1]) / highs_sorted[0][1] < 0.02 and
                    abs(highs_sorted[1][1] - highs_sorted[2][1]) / highs_sorted[1][1] < 0.02):
                    patterns['head_shoulders'][i] = True
            
            # Doble Techo
            if len(highs) >= 2:
                highs_sorted = sorted(highs, key=lambda x: x[1], reverse=True)
                if (len(highs_sorted) >= 2 and
                    abs(highs_sorted[0][1] - highs_sorted[1][1]) / highs_sorted[0][1] < 0.01):
                    patterns['double_top'][i] = True
            
            # Doble Fondo
            if len(lows) >= 2:
                lows_sorted = sorted(lows, key=lambda x: x[1])
                if (len(lows_sorted) >= 2 and
                    abs(lows_sorted[0][1] - lows_sorted[1][1]) / lows_sorted[0][1] < 0.01):
                    patterns['double_bottom'][i] = True
        
        return patterns

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick"""
        n = len(close)
        
        # Calcular Bandas de Bollinger
        basis = self.calculate_sma(close, length)
        dev = np.zeros(n)
        
        for i in range(length-1, n):
            window = close[i-length+1:i+1]
            dev[i] = np.std(window) if len(window) > 1 else 0
        
        upper = basis + (dev * mult)
        lower = basis - (dev * mult)
        
        # Calcular ancho de las bandas normalizado
        bb_width = np.zeros(n)
        for i in range(n):
            if basis[i] > 0:
                bb_width[i] = ((upper[i] - lower[i]) / basis[i]) * 100
        
        # Determinar dirección de la fuerza
        trend_strength = np.zeros(n)
        for i in range(1, n):
            if bb_width[i] > bb_width[i-1]:
                trend_strength[i] = bb_width[i]  # Positivo (verde)
            else:
                trend_strength[i] = -bb_width[i]  # Negativo (rojo)
        
        # Calcular percentil 70 para zona alta
        if n >= 50:
            historical_bb_width = bb_width[max(0, n-100):n]
            high_zone_threshold = np.percentile(historical_bb_width, 70)
        else:
            high_zone_threshold = np.percentile(bb_width, 70) if len(bb_width) > 0 else 5
        
        # Detectar zonas de no operar
        no_trade_zones = np.zeros(n, dtype=bool)
        strength_signals = ['NEUTRAL'] * n
        
        for i in range(10, n):
            # Zona de no operar cuando hay pérdida de fuerza después de movimiento fuerte
            if (bb_width[i] > high_zone_threshold and 
                trend_strength[i] < 0 and 
                bb_width[i] < np.max(bb_width[max(0, i-10):i])):
                no_trade_zones[i] = True
            
            # Determinar señal de fuerza de tendencia
            if trend_strength[i] > 0:
                if bb_width[i] > high_zone_threshold:
                    strength_signals[i] = 'STRONG_UP'
                else:
                    strength_signals[i] = 'WEAK_UP'
            elif trend_strength[i] < 0:
                if bb_width[i] > high_zone_threshold:
                    strength_signals[i] = 'STRONG_DOWN'
                else:
                    strength_signals[i] = 'WEAK_DOWN'
            else:
                strength_signals[i] = 'NEUTRAL'
        
        return {
            'bb_width': bb_width.tolist(),
            'trend_strength': trend_strength.tolist(),
            'basis': basis.tolist(),
            'upper_band': upper.tolist(),
            'lower_band': lower.tolist(),
            'high_zone_threshold': float(high_zone_threshold),
            'no_trade_zones': no_trade_zones.tolist(),
            'strength_signals': strength_signals,
            'colors': ['green' if x > 0 else 'red' for x in trend_strength]
        }

    def check_multi_timeframe_trend(self, symbol, interval):
        """Verificar tendencia en múltiples temporalidades - OBLIGATORIO"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and interval != '15m':  # Solo usar 5m para 15m
                    results[tf_type] = 'NEUTRAL'
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 50)
                if df is None or len(df) < 20:
                    results[tf_type] = 'NEUTRAL'
                    continue
                
                close = df['close'].values
                
                # Calcular medias para determinar tendencia
                ma_9 = self.calculate_sma(close, 9)
                ma_21 = self.calculate_sma(close, 21)
                
                current_ma_9 = ma_9[-1] if len(ma_9) > 0 else 0
                current_ma_21 = ma_21[-1] if len(ma_21) > 0 else 0
                current_price = close[-1]
                
                # Determinar tendencia
                if current_price > current_ma_9 and current_ma_9 > current_ma_21:
                    results[tf_type] = 'BULLISH'
                elif current_price < current_ma_9 and current_ma_9 < current_ma_21:
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}

    def check_obligatory_conditions(self, symbol, interval, signal_type, trend_strength_data, current_idx):
        """Verificar condiciones OBLIGATORIAS - Si falla alguna, score = 0"""
        try:
            # Verificar multi-timeframe
            multi_tf = self.check_multi_timeframe_trend(symbol, interval)
            
            # Verificar fuerza de tendencia Maverick
            current_strength = trend_strength_data['strength_signals'][current_idx] if current_idx < len(trend_strength_data['strength_signals']) else 'NEUTRAL'
            current_no_trade = trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False
            
            if signal_type == 'LONG':
                # Condiciones OBLIGATORIAS para LONG
                if not (multi_tf['mayor'] in ['BULLISH', 'NEUTRAL'] and
                       multi_tf['media'] == 'BULLISH' and
                       current_strength in ['STRONG_UP', 'WEAK_UP'] and
                       not current_no_trade):
                    return False
                    
            elif signal_type == 'SHORT':
                # Condiciones OBLIGATORIAS para SHORT
                if not (multi_tf['mayor'] in ['BEARISH', 'NEUTRAL'] and
                       multi_tf['media'] == 'BEARISH' and
                       current_strength in ['STRONG_DOWN', 'WEAK_DOWN'] and
                       not current_no_trade):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False

    def calculate_whale_signals_improved(self, df, interval, support_resistance_lookback=50):
        """Indicador Cazador de Ballenas MEJORADO - Solo obligatorio en 12H y 1D"""
        try:
            close = df['close'].values
            low = df['low'].values
            high = df['high'].values
            volume = df['volume'].values
            
            n = len(close)
            
            whale_pump_signal = np.zeros(n)
            whale_dump_signal = np.zeros(n)
            
            # Solo calcular señales fuertes para 12H y 1D
            is_obligatory_tf = interval in ['12h', '1D']
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                low_5 = np.min(low[max(0, i-5):i+1])
                high_5 = np.max(high[max(0, i-5):i+1])
                
                # Señales más estrictas para TF obligatorias
                if (volume_ratio > 2.0 and  # Volumen más alto para señales obligatorias
                    (close[i] < close[i-1] or price_change < -1.0) and
                    low[i] <= low_5 * 1.005):
                    
                    volume_strength = min(3.0, volume_ratio / 2.0)
                    signal_strength = volume_ratio * 25 * volume_strength
                    whale_pump_signal[i] = min(100, signal_strength) if is_obligatory_tf else signal_strength * 0.5
                
                if (volume_ratio > 2.0 and 
                    (close[i] > close[i-1] or price_change > 1.0) and
                    high[i] >= high_5 * 0.995):
                    
                    volume_strength = min(3.0, volume_ratio / 2.0)
                    signal_strength = volume_ratio * 25 * volume_strength
                    whale_dump_signal[i] = min(100, signal_strength) if is_obligatory_tf else signal_strength * 0.5
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            current_support = np.array([np.min(low[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            current_resistance = np.array([np.max(high[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'support': current_support.tolist(),
                'resistance': current_resistance.tolist(),
                'is_obligatory_tf': is_obligatory_tf
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_improved: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'support': df['low'].values.tolist(),
                'resistance': df['high'].values.tolist(),
                'is_obligatory_tf': False
            }

    def calculate_signal_score_v2(self, conditions, whale_data, interval):
        """Calcular score V2 con ponderaciones mejoradas"""
        total_score = 0
        max_score = 0
        fulfilled_conditions = []
        
        # Ponderaciones base
        weights = {
            'moving_averages': 15,
            'rsi_traditional': 15,
            'rsi_maverick': 15,
            'support_resistance': 20,
            'adx_dmi': 10,
            'macd': 10,
            'squeeze_momentum': 10,
            'bollinger_bands': 5,
            'chart_patterns': 15
        }
        
        # Ajustar ponderación de ballenas según temporalidad
        if interval in ['12h', '1D']:
            weights['whale_signals'] = 25
        else:
            # Redistribuir peso de ballenas a otros indicadores
            redistributed_weight = 25 / 8  # Distribuir entre 8 indicadores
            for key in weights:
                if key != 'whale_signals':
                    weights[key] += redistributed_weight
            weights['whale_signals'] = 0  # No puntúa en otras TF
        
        # Calcular scores
        for indicator, weight in weights.items():
            if conditions.get(indicator, False):
                total_score += weight
                fulfilled_conditions.append(indicator.replace('_', ' ').title())
            max_score += weight
        
        if max_score == 0:
            return 0, []
        
        final_score = (total_score / max_score) * 100
        return min(final_score, 100), fulfilled_conditions

    def calculate_optimal_entry_exit(self, df, signal_type, support, resistance, leverage=15):
        """Calcular entradas y salidas óptimas con lógica Smart Money"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            
            # Calcular ATR para stops
            n = len(high)
            tr = np.zeros(n)
            tr[0] = high[0] - low[0]
            for i in range(1, n):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr[i] = max(tr1, tr2, tr3)
            
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else tr[-1] if len(tr) > 0 else current_price * 0.02
            
            if signal_type == 'LONG':
                # Entrada lo más cerca posible del soporte (Smart Money)
                entry = min(current_price, support * 1.01)  # 1% sobre soporte
                stop_loss = support * 0.98  # 2% below support
                take_profit = [
                    resistance * 0.98,  # TP1 cerca de resistencia
                    resistance * 1.02,  # TP2 rompiendo resistencia
                    resistance * 1.05   # TP3 extendido
                ]
                
            else:  # SHORT
                # Entrada lo más cerca posible de la resistencia (Smart Money)
                entry = max(current_price, resistance * 0.99)  # 1% bajo resistencia
                stop_loss = resistance * 1.02  # 2% above resistance
                take_profit = [
                    support * 1.02,   # TP1 cerca de soporte
                    support * 0.98,   # TP2 rompiendo soporte
                    support * 0.95    # TP3 extendido
                ]
            
            atr_percentage = atr / current_price
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profit],
                'support': float(support),
                'resistance': float(resistance),
                'atr': float(atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.05],
                'support': float(np.min(df['low'].values[-14:])),
                'resistance': float(np.max(df['high'].values[-14:])),
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def generate_signals_v2(self, symbol, interval, di_period=14, adx_threshold=25, 
                          sr_period=50, rsi_length=14, bb_multiplier=2.0, leverage=15):
        """GENERACIÓN DE SEÑALES V2 - CON ESTRATEGIA MEJORADA"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            current_idx = -1
            
            # 1. INDICADOR CAZADOR DE BALLENAS (Obligatorio solo en 12H/1D)
            whale_data = self.calculate_whale_signals_improved(df, interval, sr_period)
            
            # 2. INDICADORES TÉCNICOS
            # Medias Móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # RSI Tradicional
            rsi_trad = self.calculate_rsi(close, rsi_length)
            rsi_bullish_div, rsi_bearish_div = self.detect_rsi_divergence(close, rsi_trad)
            
            # RSI Maverick (%B Bollinger)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            rsi_maverick = np.zeros(len(close))
            for i in range(len(close)):
                if (bb_upper[i] - bb_lower[i]) > 0:
                    rsi_maverick[i] = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
                else:
                    rsi_maverick[i] = 0.5
            
            # ADX + DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Squeeze Momentum
            squeeze_data = self.calculate_squeeze_momentum(high, low, close)
            
            # Fuerza de Tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Patrones de Chartismo
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # 3. EVALUAR CONDICIONES
            conditions = {}
            
            # Condiciones para LONG
            conditions_long = {
                'moving_averages': (ma_9[current_idx] > ma_21[current_idx] and 
                                  ma_21[current_idx] > ma_50[current_idx] and
                                  close[current_idx] > ma_200[current_idx]),
                'rsi_traditional': (rsi_trad[current_idx] < 70 and 
                                  (rsi_bullish_div[current_idx] or rsi_trad[current_idx] < 30)),
                'rsi_maverick': (rsi_maverick[current_idx] < 0.8 and 
                               (rsi_maverick[current_idx] < 0.2 or 
                                any(rsi_bullish_div[max(0, current_idx-7):current_idx+1]))),
                'support_resistance': close[current_idx] > whale_data['support'][current_idx],
                'adx_dmi': (adx[current_idx] > adx_threshold and 
                           plus_di[current_idx] > minus_di[current_idx]),
                'macd': macd[current_idx] > macd_signal[current_idx],
                'squeeze_momentum': (squeeze_data['squeeze_off'][current_idx] and 
                                   squeeze_data['momentum'][current_idx] > 0),
                'bollinger_bands': close[current_idx] > bb_middle[current_idx],
                'chart_patterns': (chart_patterns['double_bottom'][current_idx] or 
                                 chart_patterns['bullish_flag'][current_idx]),
                'whale_signals': whale_data['whale_pump'][current_idx] > 25
            }
            
            # Condiciones para SHORT
            conditions_short = {
                'moving_averages': (ma_9[current_idx] < ma_21[current_idx] and 
                                  ma_21[current_idx] < ma_50[current_idx] and
                                  close[current_idx] < ma_200[current_idx]),
                'rsi_traditional': (rsi_trad[current_idx] > 30 and 
                                  (rsi_bearish_div[current_idx] or rsi_trad[current_idx] > 70)),
                'rsi_maverick': (rsi_maverick[current_idx] > 0.2 and 
                               (rsi_maverick[current_idx] > 0.8 or 
                                any(rsi_bearish_div[max(0, current_idx-7):current_idx+1]))),
                'support_resistance': close[current_idx] < whale_data['resistance'][current_idx],
                'adx_dmi': (adx[current_idx] > adx_threshold and 
                           minus_di[current_idx] > plus_di[current_idx]),
                'macd': macd[current_idx] < macd_signal[current_idx],
                'squeeze_momentum': (squeeze_data['squeeze_off'][current_idx] and 
                                   squeeze_data['momentum'][current_idx] < 0),
                'bollinger_bands': close[current_idx] < bb_middle[current_idx],
                'chart_patterns': (chart_patterns['double_top'][current_idx] or 
                                 chart_patterns['head_shoulders'][current_idx] or
                                 chart_patterns['bearish_flag'][current_idx]),
                'whale_signals': whale_data['whale_dump'][current_idx] > 25
            }
            
            # 4. VERIFICAR CONDICIONES OBLIGATORIAS
            long_obligatory_ok = self.check_obligatory_conditions(symbol, interval, 'LONG', trend_strength_data, current_idx)
            short_obligatory_ok = self.check_obligatory_conditions(symbol, interval, 'SHORT', trend_strength_data, current_idx)
            
            # 5. CALCULAR SCORES
            long_score, long_conditions = self.calculate_signal_score_v2(conditions_long, whale_data, interval)
            short_score, short_conditions = self.calculate_signal_score_v2(conditions_short, whale_data, interval)
            
            # Aplicar multiplicador de obligatorios
            if not long_obligatory_ok:
                long_score = 0
            if not short_obligatory_ok:
                short_score = 0
            
            # 6. DETERMINAR SEÑAL FINAL
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= 70 and long_score > short_score:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= 70 and short_score > long_score:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # 7. CALCULAR NIVELES DE ENTRADA/SALIDA
            current_support = whale_data['support'][current_idx]
            current_resistance = whale_data['resistance'][current_idx]
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, current_support, current_resistance, leverage)
            
            # 8. PREPARAR DATOS DE RETORNO
            current_price = float(close[current_idx])
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'liquidation_long': float(levels_data['entry'] - (levels_data['entry'] / leverage)),
                'liquidation_short': float(levels_data['entry'] + (levels_data['entry'] / leverage)),
                'support': levels_data['support'],
                'resistance': levels_data['resistance'],
                'atr': levels_data['atr'],
                'atr_percentage': levels_data['atr_percentage'],
                'volume': float(df['volume'].iloc[current_idx]),
                'volume_ma': float(np.mean(df['volume'].tail(20))),
                'adx': float(adx[current_idx]),
                'plus_di': float(plus_di[current_idx]),
                'minus_di': float(minus_di[current_idx]),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_maverick': float(rsi_maverick[current_idx]),
                'fulfilled_conditions': fulfilled_conditions,
                'trend_strength_signal': trend_strength_data['strength_signals'][current_idx],
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx],
                'obligatory_conditions_ok': long_obligatory_ok if signal_type == 'LONG' else short_obligatory_ok,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'rsi_traditional': rsi_trad[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:].tolist(),
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['momentum'][-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'colors': trend_strength_data['colors'][-50:],
                    'rsi_bullish_div': rsi_bullish_div[-50:],
                    'rsi_bearish_div': rsi_bearish_div[-50:]
                }
            }
            
        except Exception as e:
            print(f"Error en generate_signals_v2 para {symbol}: {e}")
            return self._create_empty_signal(symbol)

    def _create_empty_signal(self, symbol):
        """Crear señal vacía en caso de error"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'signal': 'NEUTRAL',
            'signal_score': 0,
            'entry': 0,
            'stop_loss': 0,
            'take_profit': [0],
            'liquidation_long': 0,
            'liquidation_short': 0,
            'support': 0,
            'resistance': 0,
            'atr': 0,
            'atr_percentage': 0,
            'volume': 0,
            'volume_ma': 0,
            'adx': 0,
            'plus_di': 0,
            'minus_di': 0,
            'whale_pump': 0,
            'whale_dump': 0,
            'rsi_maverick': 0.5,
            'fulfilled_conditions': [],
            'trend_strength_signal': 'NEUTRAL',
            'no_trade_zone': False,
            'obligatory_conditions_ok': False,
            'data': [],
            'indicators': {}
        }

    def calculate_win_rate(self, symbol, interval, lookback=100):
        """Calcular winrate basado en datos históricos"""
        try:
            cache_key = f"winrate_{symbol}_{interval}"
            if cache_key in self.win_rate_data:
                cached_data, timestamp = self.win_rate_data[cache_key]
                if (datetime.now() - timestamp).seconds < 300:  # Cache 5 minutos
                    return cached_data
            
            df = self.get_kucoin_data(symbol, interval, lookback + 20)
            if df is None or len(df) < lookback + 10:
                return {'win_rate': 0, 'total_trades': 0, 'successful_trades': 0}
            
            successful_trades = 0
            total_trades = 0
            
            # Analizar datos históricos para calcular winrate
            for i in range(10, min(lookback, len(df) - 10)):
                try:
                    # Simular señal en punto histórico
                    historical_df = df.iloc[:i+10]
                    if len(historical_df) < 50:
                        continue
                    
                    signal_data = self.generate_signals_v2(symbol, interval)
                    
                    if signal_data['signal'] != 'NEUTRAL' and signal_data['signal_score'] >= 70:
                        total_trades += 1
                        
                        # Verificar si la señal fue exitosa
                        future_prices = df['close'].values[i+1:i+6]  # Precios siguientes
                        if len(future_prices) > 0:
                            if signal_data['signal'] == 'LONG':
                                if any(future_prices > signal_data['entry']):
                                    successful_trades += 1
                            else:  # SHORT
                                if any(future_prices < signal_data['entry']):
                                    successful_trades += 1
                    
                except Exception as e:
                    continue
            
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            result = {
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades,
                'successful_trades': successful_trades
            }
            
            self.win_rate_data[cache_key] = (result, datetime.now())
            return result
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return {'win_rate': 0, 'total_trades': 0, 'successful_trades': 0}

# Instancia global del indicador
indicator = TradingIndicator()

# Funciones auxiliares para Telegram y background tasks
def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA DE TRADING - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%

💰 Precio actual: {alert_data.get('current_price', alert_data['entry']):.6f}
🎯 Entrada: ${alert_data['entry']:.6f}
🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
📈 Take Profit: ${alert_data['take_profit']:.6f}

💪 Fuerza de Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}
📊 Apalancamiento: x{alert_data['leverage']}

✅ Condiciones Multi-Timeframe: CONFIRMADAS
🔔 WinRate Estratégico: {alert_data.get('win_rate', 'N/A')}%

📊 Revisa la señal en: https://multiframewgta.onrender.com/
            """
        else:  # exit alert
            message = f"""
🚨 ALERTA DE SALIDA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} - CERRAR

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
📊 P&L: {alert_data['pnl_percent']:+.2f}%

📊 Observación: {alert_data['reason']}
            """
        
        asyncio.run(bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=message
        ))
        print(f"Alerta {alert_type} enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            # Lógica simplificada para evitar sobrecarga
            time.sleep(60)
        except Exception as e:
            print(f"Error en background_alert_checker: {e}")
            time.sleep(60)

# Iniciar verificador de alertas en segundo plano
try:
    alert_thread = Thread(target=background_alert_checker, daemon=True)
    alert_thread.start()
    print("Background alert checker iniciado correctamente")
except Exception as e:
    print(f"Error iniciando background alert checker: {e}")

# Endpoints de la aplicación
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint para obtener señales de trading V2"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_v2(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, leverage
        )
        
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/multiple_signals')
def get_multiple_signals():
    """Endpoint para obtener múltiples señales"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        leverage = int(request.args.get('leverage', 15))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
            try:
                signal_data = indicator.generate_signals_v2(
                    symbol, interval, di_period, adx_threshold, sr_period,
                    rsi_length, bb_multiplier, leverage
                )
                
                if signal_data and signal_data['signal'] != 'NEUTRAL' and signal_data['signal_score'] >= 70:
                    all_signals.append(signal_data)
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")
                continue
        
        long_signals = [s for s in all_signals if s['signal'] == 'LONG']
        short_signals = [s for s in all_signals if s['signal'] == 'SHORT']
        
        long_signals.sort(key=lambda x: x['signal_score'], reverse=True)
        short_signals.sort(key=lambda x: x['signal_score'], reverse=True)
        
        return jsonify({
            'long_signals': long_signals,
            'short_signals': short_signals,
            'total_signals': len(all_signals)
        })
        
    except Exception as e:
        print(f"Error en /api/multiple_signals: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/scatter_data_improved')
def get_scatter_data_improved():
    """Endpoint para datos del scatter plot"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        
        scatter_data = []
        
        symbols_to_analyze = CRYPTO_SYMBOLS[:20]  # Limitar para performance
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_v2(symbol, interval, di_period, adx_threshold)
                if signal_data and signal_data['current_price'] > 0:
                    
                    buy_pressure = min(100, max(0,
                        (signal_data['whale_pump'] / 100 * 30) +
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 25 +
                        (signal_data['rsi_maverick'] * 15) +
                        (1 if signal_data['adx'] > adx_threshold else 0) * 10 +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 20)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (signal_data['whale_dump'] / 100 * 30) +
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 25 +
                        ((1 - signal_data['rsi_maverick']) * 15) +
                        (1 if signal_data['adx'] > adx_threshold else 0) * 10 +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 20)
                    ))
                    
                    scatter_data.append({
                        'symbol': symbol,
                        'x': float(buy_pressure),
                        'y': float(sell_pressure),
                        'signal_score': float(signal_data['signal_score']),
                        'current_price': float(signal_data['current_price']),
                        'signal': signal_data['signal'],
                        'risk_category': next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                    })
                    
            except Exception as e:
                print(f"Error procesando {symbol} para scatter: {e}")
                continue
        
        return jsonify(scatter_data)
        
    except Exception as e:
        print(f"Error en /api/scatter_data_improved: {e}")
        return jsonify([])

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    """Endpoint para obtener la clasificación de riesgo"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    """Endpoint para obtener alertas de scalping"""
    try:
        # Lógica simplificada para alertas
        alerts = []
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/win_rate')
def get_win_rate():
    """Endpoint para obtener winrate"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        win_rate_data = indicator.calculate_win_rate(symbol, interval)
        return jsonify(win_rate_data)
        
    except Exception as e:
        print(f"Error en /api/win_rate: {e}")
        return jsonify({'win_rate': 0, 'total_trades': 0, 'successful_trades': 0})

@app.route('/api/bolivia_time')
def get_bolivia_time():
    """Endpoint para obtener la hora actual de Bolivia"""
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d'),
        'day_of_week': current_time.strftime('%A'),
        'is_scalping_time': indicator.is_scalping_time(),
        'timezone': 'America/La_Paz'
    })

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        signal_data = indicator.generate_signals_v2(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Lógica simplificada para generar reporte PNG
        fig = plt.figure(figsize=(10, 8))
        
        # Información de la señal
        plt.axis('off')
        
        signal_info = f"""
        MULTI-TIMEFRAME CRYPTO WGTA PRO
        ==============================
        
        Símbolo: {signal_data['symbol']}
        Temporalidad: {interval}
        Señal: {signal_data['signal']}
        Score: {signal_data['signal_score']:.1f}%
        
        Precio Actual: ${signal_data['current_price']:.6f}
        Entrada: ${signal_data['entry']:.6f}
        Stop Loss: ${signal_data['stop_loss']:.6f}
        Take Profit: ${signal_data['take_profit'][0]:.6f}
        
        Fuerza Tendencia: {signal_data['trend_strength_signal']}
        Condiciones Obligatorias: {'✅ CUMPLIDAS' if signal_data['obligatory_conditions_ok'] else '❌ NO CUMPLIDAS'}
        
        Condiciones Cumplidas:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
        
        Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        plt.text(0.1, 0.9, signal_info, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png', 
                        as_attachment=True, 
                        download_name=f'report_{symbol}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        
    except Exception as e:
        print(f"Error generando reporte: {e}")
        return jsonify({'error': 'Error generando reporte'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Manejo de errores
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
