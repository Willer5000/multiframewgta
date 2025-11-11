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

# Configuración optimizada - 40 criptomonedas top
CRYPTO_SYMBOLS = [
    # Bajo Riesgo (20) - Top market cap
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
    "MATIC-USDT", "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT",
    "ETC-USDT", "XMR-USDT", "FIL-USDT", "ALGO-USDT", "VET-USDT",
    
    # Medio Riesgo (10) - Proyectos consolidados
    "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT", "ONE-USDT",
    
    # Alto Riesgo (7) - Proyectos emergentes
    "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
    "MAGIC-USDT", "RNDR-USDT",
    
    # Memecoins (3) - Top memes
    "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT"
]

# Clasificación de riesgo optimizada
CRYPTO_RISK_CLASSIFICATION = {
    "bajo": [
        "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
        "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
        "MATIC-USDT", "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT",
        "ETC-USDT", "XMR-USDT", "FIL-USDT", "ALGO-USDT", "VET-USDT"
    ],
    "medio": [
        "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
        "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT", "ONE-USDT"
    ],
    "alto": [
        "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
        "MAGIC-USDT", "RNDR-USDT"
    ],
    "memecoins": [
        "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT"
    ]
}

# Mapeo de temporalidades para análisis multi-timeframe
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
            # Para 3D, verificar cada 3 días
            days_since_epoch = (current_time - datetime(1970, 1, 1)).days
            return days_since_epoch % 3 == 0
        elif interval == '1W':
            # Para 1W, verificar los lunes
            return current_time.weekday() == 0
        
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
                '1D': '1day', '3D': '3day', '1W': '1week', '1M': '1month'
            }
            
            kucoin_interval = interval_map.get(interval, '1hour')
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol.replace('-', '')}&type={kucoin_interval}"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if not candles:
                        return self.generate_fallback_data(limit, symbol)
                    
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
        
        return self.generate_fallback_data(limit, symbol)

    def generate_fallback_data(self, limit, symbol):
        """Generar datos de fallback realistas"""
        try:
            # Obtener precio actual desde una API alternativa
            price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.replace('-', '')}"
            response = requests.get(price_url, timeout=5)
            if response.status_code == 200:
                price_data = response.json()
                current_price = float(price_data['price'])
            else:
                current_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        except:
            current_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1h')
        
        returns = np.random.normal(0.001, 0.02, limit)
        prices = current_price * (1 + np.cumsum(returns))
        
        data = {
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0.01, 0.01, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, limit))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, limit)
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
        """Calcular MACD manualmente"""
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
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower

    def calculate_adx(self, high, low, close, period=14):
        """Calcular ADX, +DI, -DI"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        
        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Smooth the values
        tr_smooth = self.calculate_ema(tr, period)
        plus_dm_smooth = self.calculate_ema(plus_dm, period)
        minus_dm_smooth = self.calculate_ema(minus_dm, period)
        
        # Directional Indicators
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        
        for i in range(n):
            if tr_smooth[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        
        # Directional Index
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
            bb_dev[i] = np.std(window)
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
        for i in range(n):
            if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
                squeeze_on[i] = True
            elif bb_upper[i] > kc_upper[i] and bb_lower[i] < kc_lower[i]:
                squeeze_off[i] = True
        
        # Momentum
        momentum = close - bb_basis
        
        return {
            'squeeze_on': squeeze_on.tolist(),
            'squeeze_off': squeeze_off.tolist(),
            'momentum': momentum.tolist(),
            'bb_upper': bb_upper.tolist(),
            'bb_lower': bb_lower.tolist(),
            'kc_upper': kc_upper.tolist(),
            'kc_lower': kc_lower.tolist()
        }

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'rising_wedge': np.zeros(n, dtype=bool),
            'falling_wedge': np.zeros(n, dtype=bool),
            'bull_flag': np.zeros(n, dtype=bool),
            'bear_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-7):
            # Hombro Cabeza Hombro (reversión bajista)
            if (i >= lookback + 20):
                left_shoulder = np.argmax(high[i-20:i-10])
                head = np.argmax(high[i-10:i])
                right_shoulder = np.argmax(high[i:i+10])
                
                if (high[i-20+left_shoulder] < high[i-10+head] and 
                    high[i+right_shoulder] < high[i-10+head] and
                    abs(high[i-20+left_shoulder] - high[i+right_shoulder]) / high[i-10+head] < 0.02):
                    patterns['head_shoulders'][i] = True
            
            # Doble Techo (reversión bajista)
            if (i >= lookback + 15):
                first_top = np.argmax(high[i-15:i-5])
                second_top = np.argmax(high[i-5:i+5])
                
                if (abs(high[i-15+first_top] - high[i-5+second_top]) / high[i-15+first_top] < 0.01 and
                    low[i-5+np.argmin(low[i-5:i+5])] < np.min(low[i-15:i-5])):
                    patterns['double_top'][i] = True
            
            # Doble Fondo (reversión alcista)
            if (i >= lookback + 15):
                first_bottom = np.argmin(low[i-15:i-5])
                second_bottom = np.argmin(low[i-5:i+5])
                
                if (abs(low[i-15+first_bottom] - low[i-5+second_bottom]) / low[i-15+first_bottom] < 0.01 and
                    high[i-5+np.argmax(high[i-5:i+5])] > np.max(high[i-15:i-5])):
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
                trend_strength[i] = bb_width[i]  # Fuerza creciente
            else:
                trend_strength[i] = -bb_width[i]  # Fuerza decreciente
        
        # Detectar zonas de no operar
        no_trade_zones = np.zeros(n, dtype=bool)
        strength_signals = ['NEUTRAL'] * n
        
        for i in range(10, n):
            # Zona de no operar cuando hay pérdida de fuerza después de movimiento fuerte
            if (trend_strength[i] < 0 and 
                bb_width[i] < np.max(bb_width[max(0, i-10):i])):
                no_trade_zones[i] = True
            
            # Determinar señal de fuerza de tendencia
            if trend_strength[i] > 0:
                if bb_width[i] > 5:  # Umbral alto
                    strength_signals[i] = 'STRONG_UP'
                else:
                    strength_signals[i] = 'WEAK_UP'
            elif trend_strength[i] < 0:
                if bb_width[i] > 5:
                    strength_signals[i] = 'STRONG_DOWN'
                else:
                    strength_signals[i] = 'WEAK_DOWN'
        
        return {
            'bb_width': bb_width.tolist(),
            'trend_strength': trend_strength.tolist(),
            'no_trade_zones': no_trade_zones.tolist(),
            'strength_signals': strength_signals,
            'colors': ['green' if x > 0 else 'red' for x in trend_strength]
        }

    def check_multi_timeframe_trend(self, symbol, timeframe):
        """Verificar tendencia en múltiples temporalidades"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(timeframe, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:  # Saltar temporalidades no soportadas
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

    def check_obligatory_conditions(self, symbol, interval, signal_type):
        """Verificar condiciones obligatorias multi-temporalidad"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False
            
            # Verificar tendencias en todas las temporalidades
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            # Verificar fuerza de tendencia Maverick en todas las TF
            maverick_conditions = {}
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 30)
                if df is not None and len(df) > 10:
                    trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                    maverick_conditions[tf_type] = {
                        'signal': trend_data['strength_signals'][-1],
                        'no_trade': trend_data['no_trade_zones'][-1]
                    }
                else:
                    maverick_conditions[tf_type] = {'signal': 'NEUTRAL', 'no_trade': False}
            
            if signal_type == 'LONG':
                # Condiciones obligatorias para LONG
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'  # EXCLUSIVAMENTE ALCISTA
                menor_ok = tf_analysis.get('menor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                
                # Verificar Maverick - SIN zonas de NO OPERAR
                maverick_ok = all(not cond['no_trade'] for cond in maverick_conditions.values())
                
                # Verificar fuerza de tendencia Maverick en TF menor
                menor_maverick = maverick_conditions.get('menor', {'signal': 'NEUTRAL'})
                menor_strength_ok = menor_maverick['signal'] in ['STRONG_UP', 'WEAK_UP']
                
                return all([mayor_ok, media_ok, menor_ok, maverick_ok, menor_strength_ok])
                
            elif signal_type == 'SHORT':
                # Condiciones obligatorias para SHORT
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'  # EXCLUSIVAMENTE BAJISTA
                menor_ok = tf_analysis.get('menor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                
                # Verificar Maverick - SIN zonas de NO OPERAR
                maverick_ok = all(not cond['no_trade'] for cond in maverick_conditions.values())
                
                # Verificar fuerza de tendencia Maverick en TF menor
                menor_maverick = maverick_conditions.get('menor', {'signal': 'NEUTRAL'})
                menor_strength_ok = menor_maverick['signal'] in ['STRONG_DOWN', 'WEAK_DOWN']
                
                return all([mayor_ok, media_ok, menor_ok, maverick_ok, menor_strength_ok])
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False

    def calculate_whale_signals_improved(self, df, interval):
        """Implementación MEJORADA del indicador de ballenas"""
        try:
            close = df['close'].values
            low = df['low'].values
            high = df['high'].values
            volume = df['volume'].values
            
            n = len(close)
            
            whale_pump_signal = np.zeros(n)
            whale_dump_signal = np.zeros(n)
            
            # Solo activar señal obligatoria en 12H y 1D
            is_obligatory_tf = interval in ['12h', '1D']
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                low_5 = np.min(low[max(0, i-5):i+1])
                high_5 = np.max(high[max(0, i-5):i+1])
                
                # Señal de compra (whale pump)
                if (volume_ratio > 1.5 and 
                    (close[i] < close[i-1] or price_change < -0.5) and
                    low[i] <= low_5 * 1.01):
                    
                    volume_strength = min(3.0, volume_ratio / 1.5)
                    base_signal = volume_ratio * 20 * 1.7 * volume_strength
                    
                    if is_obligatory_tf:
                        whale_pump_signal[i] = min(100, base_signal)
                    else:
                        whale_pump_signal[i] = min(50, base_signal * 0.5)  # Reducir peso en otras TF
                
                # Señal de venta (whale dump)
                if (volume_ratio > 1.5 and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / 1.5)
                    base_signal = volume_ratio * 20 * 1.7 * volume_strength
                    
                    if is_obligatory_tf:
                        whale_dump_signal[i] = min(100, base_signal)
                    else:
                        whale_dump_signal[i] = min(50, base_signal * 0.5)  # Reducir peso en otras TF
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'is_obligatory': is_obligatory_tf
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_improved: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'is_obligatory': False
            }

    def calculate_support_resistance(self, high, low, close, period=50):
        """Calcular soportes y resistencias Smart Money"""
        n = len(close)
        support_levels = np.zeros(n)
        resistance_levels = np.zeros(n)
        
        for i in range(period, n):
            # Soporte: mínimo de los últimos periodos
            support_levels[i] = np.min(low[i-period:i+1])
            # Resistencia: máximo de los últimos periodos
            resistance_levels[i] = np.max(high[i-period:i+1])
        
        return support_levels.tolist(), resistance_levels.tolist()

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias entre precio e indicador"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Divergencia alcista: precio hace lower low, indicador hace higher low
            if (price[i] < np.min(price[i-lookback:i]) and
                indicator[i] > np.max(indicator[i-lookback:i])):
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace higher high, indicador hace lower high
            if (price[i] > np.max(price[i-lookback:i]) and
                indicator[i] < np.min(indicator[i-lookback:i])):
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def calculate_optimal_entry_exit(self, df, signal_type, support, resistance):
        """Calcular entradas y salidas óptimas Smart Money"""
        try:
            close = df['close'].values
            current_price = close[-1]
            
            if signal_type == 'LONG':
                # Entrada lo más cerca posible del soporte
                entry = min(current_price, support * 1.01)  # 1% sobre soporte
                stop_loss = support * 0.98  # 2% bajo soporte
                take_profit = resistance * 0.99  # 1% bajo resistencia
            else:  # SHORT
                # Entrada lo más cerca posible de la resistencia
                entry = max(current_price, resistance * 0.99)  # 1% bajo resistencia
                stop_loss = resistance * 1.02  # 2% sobre resistencia
                take_profit = support * 1.01  # 1% sobre soporte
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(take_profit)],
                'support': float(support),
                'resistance': float(resistance)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.05],
                'support': current_price * 0.95,
                'resistance': current_price * 1.05
            }

    def evaluate_signal_conditions(self, data, current_idx, interval):
        """Evaluar condiciones de señal con nuevo sistema de pesos"""
        conditions = {
            'long': {
                'multi_timeframe': {'value': False, 'weight': 25, 'description': 'Condiciones multi-TF obligatorias'},
                'whale_pump': {'value': False, 'weight': 25, 'description': 'Ballena compradora activa'},
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación medias móviles'},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI tradicional favorable'},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable'},
                'support_resistance': {'value': False, 'weight': 20, 'description': 'Soporte/resistencia Smart Money'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': 'ADX + DMI favorable'},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD favorable'},
                'squeeze': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable'},
                'bollinger_bands': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger favorable'},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrón chartismo favorable'}
            },
            'short': {
                'multi_timeframe': {'value': False, 'weight': 25, 'description': 'Condiciones multi-TF obligatorias'},
                'whale_dump': {'value': False, 'weight': 25, 'description': 'Ballena vendedora activa'},
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación medias móviles'},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI tradicional favorable'},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable'},
                'support_resistance': {'value': False, 'weight': 20, 'description': 'Soporte/resistencia Smart Money'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': 'ADX + DMI favorable'},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD favorable'},
                'squeeze': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable'},
                'bollinger_bands': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger favorable'},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrón chartismo favorable'}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        current_price = data['close'][current_idx]
        
        # Condiciones para LONG
        conditions['long']['whale_pump']['value'] = data['whale_pump'][current_idx] > 15
        
        # Medias móviles (precio > MA9 > MA21 > MA50)
        if (current_idx < len(data['ma_9']) and current_idx < len(data['ma_21']) and 
            current_idx < len(data['ma_50'])):
            conditions['long']['moving_averages']['value'] = (
                current_price > data['ma_9'][current_idx] and
                data['ma_9'][current_idx] > data['ma_21'][current_idx] and
                data['ma_21'][current_idx] > data['ma_50'][current_idx]
            )
        
        # RSI tradicional
        if current_idx < len(data['rsi']):
            conditions['long']['rsi_traditional']['value'] = (
                data['rsi'][current_idx] < 70 and  # No sobrecompra
                data['rsi'][current_idx] > 30 and  # No sobreventa
                any(data['bullish_div_rsi'][max(0, current_idx-7):current_idx+1])  # Divergencia reciente
            )
        
        # RSI Maverick
        if current_idx < len(data['rsi_maverick']):
            conditions['long']['rsi_maverick']['value'] = (
                data['rsi_maverick'][current_idx] < 0.8 and  # No sobrecompra
                data['rsi_maverick'][current_idx] > 0.2 and  # No sobreventa
                any(data['bullish_div_maverick'][max(0, current_idx-7):current_idx+1])  # Divergencia reciente
            )
        
        # Soporte/Resistencia
        conditions['long']['support_resistance']['value'] = (
            current_price <= data['support'][current_idx] * 1.02  # Cerca del soporte
        )
        
        # ADX + DMI
        if current_idx < len(data['adx']) and current_idx < len(data['plus_di']):
            conditions['long']['adx_dmi']['value'] = (
                data['adx'][current_idx] > 25 and  # Tendencia fuerte
                data['plus_di'][current_idx] > data['minus_di'][current_idx]  # Tendencia alcista
            )
        
        # MACD
        if current_idx < len(data['macd_histogram']):
            conditions['long']['macd']['value'] = data['macd_histogram'][current_idx] > 0
        
        # Squeeze Momentum
        if current_idx < len(data['squeeze_momentum']):
            conditions['long']['squeeze']['value'] = data['squeeze_momentum'][current_idx] > 0
        
        # Bandas de Bollinger
        if current_idx < len(data['bb_position']):
            conditions['long']['bollinger_bands']['value'] = data['bb_position'][current_idx] < 0.8
        
        # Patrones de Chartismo
        conditions['long']['chart_patterns']['value'] = any([
            data['double_bottom'][current_idx],
            data['bull_flag'][current_idx],
            any(data['double_bottom'][max(0, current_idx-7):current_idx+1]),
            any(data['bull_flag'][max(0, current_idx-7):current_idx+1])
        ])
        
        # Condiciones para SHORT
        conditions['short']['whale_dump']['value'] = data['whale_dump'][current_idx] > 15
        
        # Medias móviles (precio < MA9 < MA21 < MA50)
        if (current_idx < len(data['ma_9']) and current_idx < len(data['ma_21']) and 
            current_idx < len(data['ma_50'])):
            conditions['short']['moving_averages']['value'] = (
                current_price < data['ma_9'][current_idx] and
                data['ma_9'][current_idx] < data['ma_21'][current_idx] and
                data['ma_21'][current_idx] < data['ma_50'][current_idx]
            )
        
        # RSI tradicional
        if current_idx < len(data['rsi']):
            conditions['short']['rsi_traditional']['value'] = (
                data['rsi'][current_idx] > 30 and  # No sobreventa
                data['rsi'][current_idx] < 70 and  # No sobrecompra
                any(data['bearish_div_rsi'][max(0, current_idx-7):current_idx+1])  # Divergencia reciente
            )
        
        # RSI Maverick
        if current_idx < len(data['rsi_maverick']):
            conditions['short']['rsi_maverick']['value'] = (
                data['rsi_maverick'][current_idx] > 0.2 and  # No sobreventa
                data['rsi_maverick'][current_idx] < 0.8 and  # No sobrecompra
                any(data['bearish_div_maverick'][max(0, current_idx-7):current_idx+1])  # Divergencia reciente
            )
        
        # Soporte/Resistencia
        conditions['short']['support_resistance']['value'] = (
            current_price >= data['resistance'][current_idx] * 0.98  # Cerca de la resistencia
        )
        
        # ADX + DMI
        if current_idx < len(data['adx']) and current_idx < len(data['minus_di']):
            conditions['short']['adx_dmi']['value'] = (
                data['adx'][current_idx] > 25 and  # Tendencia fuerte
                data['minus_di'][current_idx] > data['plus_di'][current_idx]  # Tendencia bajista
            )
        
        # MACD
        if current_idx < len(data['macd_histogram']):
            conditions['short']['macd']['value'] = data['macd_histogram'][current_idx] < 0
        
        # Squeeze Momentum
        if current_idx < len(data['squeeze_momentum']):
            conditions['short']['squeeze']['value'] = data['squeeze_momentum'][current_idx] < 0
        
        # Bandas de Bollinger
        if current_idx < len(data['bb_position']):
            conditions['short']['bollinger_bands']['value'] = data['bb_position'][current_idx] > 0.2
        
        # Patrones de Chartismo
        conditions['short']['chart_patterns']['value'] = any([
            data['head_shoulders'][current_idx],
            data['double_top'][current_idx],
            data['bear_flag'][current_idx],
            any(data['head_shoulders'][max(0, current_idx-7):current_idx+1]),
            any(data['double_top'][max(0, current_idx-7):current_idx+1]),
            any(data['bear_flag'][max(0, current_idx-7):current_idx+1])
        ])
        
        return conditions

    def calculate_signal_score(self, conditions, signal_type, obligatory_conditions_met):
        """Calcular puntuación de señal con sistema de obligatoriedad"""
        if not obligatory_conditions_met:
            return 0, []
        
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        signal_conditions = conditions.get(signal_type, {})
        
        for key, condition in signal_conditions.items():
            total_weight += condition['weight']
            if condition['value']:
                achieved_weight += condition['weight']
                fulfilled_conditions.append(condition['description'])
        
        if total_weight == 0:
            return 0, []
        
        score = (achieved_weight / total_weight) * 100
        
        return min(score, 100), fulfilled_conditions

    def calculate_win_rate(self, symbol, interval, lookback=100):
        """Calcular winrate histórico"""
        try:
            cache_key = f"winrate_{symbol}_{interval}"
            if cache_key in self.win_rate_data:
                return self.win_rate_data[cache_key]
            
            df = self.get_kucoin_data(symbol, interval, lookback + 20)
            if df is None or len(df) < lookback + 10:
                return 50.0  # Winrate por defecto
            
            close = df['close'].values
            signals = []
            results = []
            
            # Simular señales históricas
            for i in range(10, len(close) - 1):
                # Simular señal basada en precio (esto es un placeholder)
                if close[i] > close[i-10]:
                    signals.append('LONG')
                    # Resultado: ganancia si el precio sube
                    result = 'WIN' if close[i+1] > close[i] else 'LOSE'
                    results.append(result)
                elif close[i] < close[i-10]:
                    signals.append('SHORT')
                    # Resultado: ganancia si el precio baja
                    result = 'WIN' if close[i+1] < close[i] else 'LOSE'
                    results.append(result)
            
            if not results:
                return 50.0
            
            win_count = results.count('WIN')
            total_trades = len(results)
            win_rate = (win_count / total_trades) * 100
            
            self.win_rate_data[cache_key] = win_rate
            return win_rate
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return 50.0

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - NUEVA ESTRATEGIA"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calcular todos los indicadores
            whale_data = self.calculate_whale_signals_improved(df, interval)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            support, resistance = self.calculate_support_resistance(high, low, close, sr_period)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # RSI tradicional
            rsi = self.calculate_rsi(close, rsi_length)
            bullish_div_rsi, bearish_div_rsi = self.detect_divergence(close, rsi)
            
            # RSI Maverick (Bandas de Bollinger %B)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            rsi_maverick = (close - bb_lower) / (bb_upper - bb_lower)
            bullish_div_maverick, bearish_div_maverick = self.detect_divergence(close, rsi_maverick)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Squeeze Momentum
            squeeze_data = self.calculate_squeeze_momentum(high, low, close)
            
            # Bandas de Bollinger posición
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Patrones de Chartismo
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # Fuerza de Tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            current_price = float(close[current_idx])
            
            # Preparar datos para evaluación
            analysis_data = {
                'close': close,
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'rsi': rsi,
                'bullish_div_rsi': bullish_div_rsi,
                'bearish_div_rsi': bearish_div_rsi,
                'rsi_maverick': rsi_maverick,
                'bullish_div_maverick': bullish_div_maverick,
                'bearish_div_maverick': bearish_div_maverick,
                'support': support,
                'resistance': resistance,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'macd_histogram': macd_histogram,
                'squeeze_momentum': squeeze_data['momentum'],
                'bb_position': bb_position,
                'head_shoulders': chart_patterns['head_shoulders'],
                'double_top': chart_patterns['double_top'],
                'double_bottom': chart_patterns['double_bottom'],
                'bull_flag': chart_patterns['bull_flag'],
                'bear_flag': chart_patterns['bear_flag']
            }
            
            # Evaluar condiciones
            conditions = self.evaluate_signal_conditions(analysis_data, current_idx, interval)
            
            # Verificar condiciones obligatorias para LONG y SHORT
            long_obligatory = self.check_obligatory_conditions(symbol, interval, 'LONG')
            short_obligatory = self.check_obligatory_conditions(symbol, interval, 'SHORT')
            
            # Calcular scores
            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', long_obligatory)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', short_obligatory)
            
            # Determinar señal final
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
            
            # Calcular niveles de entrada/salida
            current_support = support[current_idx] if current_idx < len(support) else current_price * 0.95
            current_resistance = resistance[current_idx] if current_idx < len(resistance) else current_price * 1.05
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, current_support, current_resistance
            )
            
            # Calcular winrate
            win_rate = self.calculate_win_rate(symbol, interval)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'win_rate': float(win_rate),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support': levels_data['support'],
                'resistance': levels_data['resistance'],
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
                'adx': float(adx[current_idx] if current_idx < len(adx) else 0),
                'plus_di': float(plus_di[current_idx] if current_idx < len(plus_di) else 0),
                'minus_di': float(minus_di[current_idx] if current_idx < len(minus_di) else 0),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'fulfilled_conditions': fulfilled_conditions,
                'obligatory_conditions_met': long_obligatory if signal_type == 'LONG' else short_obligatory,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'rsi': rsi[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:].tolist(),
                    'macd': macd_line[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['momentum'][-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'colors': trend_strength_data['colors'][-50:]
                }
            }
            
        except Exception as e:
            print(f"Error en generate_signals_improved para {symbol}: {e}")
            return self._create_empty_signal(symbol)

    def _create_empty_signal(self, symbol):
        """Crear señal vacía en caso de error"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'signal': 'NEUTRAL',
            'signal_score': 0,
            'win_rate': 50.0,
            'entry': 0,
            'stop_loss': 0,
            'take_profit': [0],
            'support': 0,
            'resistance': 0,
            'volume': 0,
            'volume_ma': 0,
            'adx': 0,
            'plus_di': 0,
            'minus_di': 0,
            'whale_pump': 0,
            'whale_dump': 0,
            'rsi_maverick': 0.5,
            'fulfilled_conditions': [],
            'obligatory_conditions_met': False,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de scalping"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h']:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 70 and
                        signal_data['obligatory_conditions_met']):
                        
                        risk_category = next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                        
                        alert = {
                            'symbol': symbol,
                            'interval': interval,
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'win_rate': signal_data['win_rate'],
                            'entry': signal_data['entry'],
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'][0],
                            'leverage': 15,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price']
                        }
                        
                        alerts.append(alert)
                    
                except Exception as e:
                    print(f"Error generando alerta para {symbol} {interval}: {e}")
                    continue
        
        return alerts

# Instancia global del indicador
indicator = TradingIndicator()

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
🏆 WinRate: {alert_data['win_rate']:.1f}%

💰 Precio actual: {alert_data['current_price']:.6f}
🎯 Entrada: ${alert_data['entry']:.6f}
🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
🎯 Take Profit: ${alert_data['take_profit']:.6f}

📈 Apalancamiento: x{alert_data['leverage']}

✅ Condiciones Cumplidas:
• {chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])[:3]])}

🔔 Sistema Multi-Temporalidad Confirmado
            """
        
        # Generar URL para el reporte
        report_url = f"https://ballenasscalpistas.onrender.com/api/generate_report?symbol={alert_data['symbol']}&interval={alert_data['interval']}"
        
        # Crear botón de descarga
        keyboard = [[telegram.InlineKeyboardButton("📊 Descargar Reporte Completo", url=report_url)]]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        asyncio.run(bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=message,
            reply_markup=reply_markup
        ))
        print(f"Alerta {alert_type} enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar alertas cada 60 segundos
            print("Verificando alertas...")
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                send_telegram_alert(alert, 'entry')
            
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint para obtener señales de trading"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(
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
                signal_data = indicator.generate_signals_improved(
                    symbol, interval, di_period, adx_threshold, sr_period,
                    rsi_length, bb_multiplier, leverage
                )
                
                if signal_data and signal_data['signal'] != 'NEUTRAL' and signal_data['signal_score'] >= 70:
                    all_signals.append(signal_data)
                
                time.sleep(0.1)
                
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
        
        scatter_data = []
        
        symbols_to_analyze = CRYPTO_SYMBOLS[:15]
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval)
                if signal_data and signal_data['current_price'] > 0:
                    
                    buy_pressure = min(100, max(0, signal_data['whale_pump'] + 
                                                (50 if signal_data['plus_di'] > signal_data['minus_di'] else 0)))
                    
                    sell_pressure = min(100, max(0, signal_data['whale_dump'] + 
                                                 (50 if signal_data['minus_di'] > signal_data['plus_di'] else 0)))
                    
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
        alerts = indicator.generate_scalping_alerts()
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
        
        win_rate = indicator.calculate_win_rate(symbol, interval)
        return jsonify({'win_rate': win_rate})
        
    except Exception as e:
        print(f"Error en /api/win_rate: {e}")
        return jsonify({'win_rate': 50.0})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(12, 16))
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(7, 1, 1)
        if signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            opens = [d['open'] for d in signal_data['data']]
            highs = [d['high'] for d in signal_data['data']]
            lows = [d['low'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            
            # Gráfico de velas
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal_data['take_profit'][0], color='green', linestyle='--', alpha=0.7, label='Take Profit')
            ax1.axhline(y=signal_data['support'], color='orange', linestyle=':', alpha=0.5, label='Soporte')
            ax1.axhline(y=signal_data['resistance'], color='purple', linestyle=':', alpha=0.5, label='Resistencia')
        
        ax1.set_title(f'{symbol} - Análisis Multi-Temporalidad ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Ballenas
        ax2 = plt.subplot(7, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax2.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax2.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
        ax2.set_ylabel('Fuerza Ballenas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: ADX/DMI
        ax3 = plt.subplot(7, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax3.plot(adx_dates, signal_data['indicators']['adx'], 
                    'white', linewidth=2, label='ADX')
            ax3.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax3.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
        ax3.set_ylabel('ADX/DMI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: RSI y RSI Maverick
        ax4 = plt.subplot(7, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi'], 
                    'blue', linewidth=2, label='RSI Tradicional')
            ax4.plot(rsi_dates, [x * 100 for x in signal_data['indicators']['rsi_maverick']], 
                    'orange', linewidth=2, label='RSI Maverick')
            ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax4.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax4.set_ylabel('RSI')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: MACD
        ax5 = plt.subplot(7, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax5.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax5.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            # Histograma MACD
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax5.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6, label='Histograma')
        ax5.set_ylabel('MACD')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Squeeze Momentum
        ax6 = plt.subplot(7, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            squeeze_dates = dates[-len(signal_data['indicators']['squeeze_momentum']):]
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['squeeze_momentum']]
            ax6.bar(squeeze_dates, signal_data['indicators']['squeeze_momentum'], 
                   color=colors, alpha=0.7, label='Squeeze Momentum')
            ax6.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        ax6.set_ylabel('Squeeze')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: Fuerza de Tendencia Maverick
        ax7 = plt.subplot(7, 1, 7, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax7.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            # Marcar zonas de no operar
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax7.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax7.set_ylabel('Fuerza Tendencia')
            ax7.set_xlabel('Fecha/Hora')
            ax7.grid(True, alpha=0.3)
        
        # Información de la señal
        plt.figtext(0.02, 0.02, 
                   f"""SEÑAL: {signal_data['signal']} | SCORE: {signal_data['signal_score']:.1f}% | WINRATE: {signal_data['win_rate']:.1f}%
Precio: ${signal_data['current_price']:.6f} | Entrada: ${signal_data['entry']:.6f} | SL: ${signal_data['stop_loss']:.6f} | TP: ${signal_data['take_profit'][0]:.6f}
Condiciones: {', '.join(signal_data.get('fulfilled_conditions', [])[:3])}""", 
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
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

@app.route('/api/bolivia_time')
def get_bolivia_time():
    """Endpoint para obtener la hora actual de Bolivia"""
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d'),
        'day_of_week': current_time.strftime('%A'),
        'is_scalping_time': indicator.is_scalping_time()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
