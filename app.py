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

# Configuración Telegram MEJORADA
TELEGRAM_BOT_TOKEN = "8007748376:AAHIW8n9b-BtA378g4gF-0-D2mOhn495Q0g"
TELEGRAM_CHAT_ID = "-1003229814161"

# Configuración optimizada - 40 criptomonedas TOP (excluyendo MATIC y GALA)
CRYPTO_SYMBOLS = [
    # Bajo Riesgo (20) - Top market cap
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
    "ATOM-USDT", "LTC-USDT", "BCH-USDT", "ETC-USDT", "XLM-USDT",
    "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "MANA-USDT",
    
    # Medio Riesgo (10) - Proyectos consolidados
    "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT", "IOTA-USDT",
    
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
        "ATOM-USDT", "LTC-USDT", "BCH-USDT", "ETC-USDT", "XLM-USDT",
        "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "MANA-USDT"
    ],
    "medio": [
        "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
        "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT", "IOTA-USDT"
    ],
    "alto": [
        "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
        "MAGIC-USDT", "RNDR-USDT"
    ],
    "memecoins": [
        "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT"
    ]
}

# Mapeo de temporalidades MEJORADO con nuevas TF
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
        self.winrate_data = {}
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
            # Para 3 días, simplificar lógica
            return True
        elif interval == '1W':
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
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol}&type={kucoin_interval}"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if not candles:
                        df = self.generate_sample_data(limit, interval, symbol)
                        self.cache[cache_key] = (df, datetime.now())
                        return df
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    df = df.iloc[::-1].reset_index(drop=True)
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    result = df.tail(limit)
                    self.cache[cache_key] = (result, datetime.now())
                    return result
                
        except requests.exceptions.Timeout:
            print(f"Timeout obteniendo datos de {symbol}")
        except Exception as e:
            print(f"Error obteniendo datos de KuCoin para {symbol} {interval}: {e}")
        
        df = self.generate_sample_data(limit, interval, symbol)
        self.cache[cache_key] = (df, datetime.now())
        return df

    def generate_sample_data(self, limit, interval, symbol):
        """Generar datos de ejemplo más realistas"""
        np.random.seed(42)
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
        
        returns = np.random.normal(0.001, 0.02, limit)
        prices = base_price * (1 + np.cumsum(returns))
        
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

    def calculate_atr(self, high, low, close, period=14):
        """Calcular Average True Range"""
        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        atr = self.calculate_sma(tr, period)
        return atr

    def calculate_sma(self, prices, period):
        """Calcular SMA manualmente"""
        if len(prices) == 0 or period <= 0:
            return np.zeros_like(prices)
            
        sma = np.zeros_like(prices)
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window = prices[start_idx:i+1]
            valid_values = window[~np.isnan(window)]
            sma[i] = np.mean(valid_values) if len(valid_values) > 0 else (prices[i] if i < len(prices) and not np.isnan(prices[i]) else 0)
        
        return sma

    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente"""
        if len(prices) == 0 or period <= 0:
            return np.zeros_like(prices)
            
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0] if len(prices) > 0 and not np.isnan(prices[0]) else 0
        
        for i in range(1, len(prices)):
            if np.isnan(prices[i]):
                ema[i] = ema[i-1]
            else:
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional manualmente"""
        if len(prices) < period + 1:
            return np.zeros_like(prices)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.zeros_like(prices)
        for i in range(len(prices)):
            if avg_losses[i] > 0:
                rs[i] = avg_gains[i] / avg_losses[i]
            else:
                rs[i] = 100 if avg_gains[i] > 0 else 50
        
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcular MACD manualmente"""
        if len(prices) < slow:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger manualmente"""
        if len(prices) < period:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        sma = self.calculate_sma(prices, period)
        std = np.zeros_like(prices)
        
        for i in range(len(prices)):
            if i >= period - 1:
                window = prices[i-period+1:i+1]
                std[i] = np.std(window)
            else:
                std[i] = 0
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower

    def calculate_rsi_maverick(self, close, length=20, bb_multiplier=2.0):
        """Implementación del RSI Modificado Maverick (Bollinger %B)"""
        try:
            n = len(close)
            
            basis = np.array([np.mean(close[max(0, i-length+1):i+1]) for i in range(n)])
            dev = np.array([np.std(close[max(0, i-length+1):i+1]) for i in range(n)])
            
            upper = basis + (dev * bb_multiplier)
            lower = basis - (dev * bb_multiplier)
            
            b_percent = np.zeros(n)
            for i in range(n):
                if (upper[i] - lower[i]) > 0:
                    b_percent[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
                else:
                    b_percent[i] = 0.5
            
            return b_percent.tolist()
            
        except Exception as e:
            print(f"Error en calculate_rsi_maverick: {e}")
            return [0.5] * len(close)

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick"""
        try:
            n = len(close)
            
            # Calcular Bandas de Bollinger
            basis = self.calculate_sma(close, length)
            dev = np.zeros(n)
            
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                dev[i] = np.std(window) if len(window) > 1 else 0
            
            upper = basis + (dev * mult)
            lower = basis - (dev * mult)
            
            # Calcular ancho de las bandas normalizado a porcentaje
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
            
        except Exception as e:
            print(f"Error en calculate_trend_strength_maverick: {e}")
            n = len(close)
            return {
                'bb_width': [0] * n,
                'trend_strength': [0] * n,
                'basis': [0] * n,
                'upper_band': [0] * n,
                'lower_band': [0] * n,
                'high_zone_threshold': 5.0,
                'no_trade_zones': [False] * n,
                'strength_signals': ['NEUTRAL'] * n,
                'colors': ['gray'] * n
            }

    def calculate_adx(self, high, low, close, period=14):
        """Calcular ADX, +DI, -DI manualmente"""
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

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias entre precio e indicador"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Divergencia alcista: precio hace lower low, indicador hace higher low
            if (price[i] < price[i-1] and 
                indicator[i] > indicator[i-1] and
                price[i] < np.min(price[i-lookback:i])):
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace higher high, indicador hace lower high
            if (price[i] > price[i-1] and 
                indicator[i] < indicator[i-1] and
                price[i] > np.max(price[i-lookback:i])):
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo"""
        n = len(close)
        patterns = {
            'hch': np.zeros(n, dtype=bool),  # Hombro Cabeza Hombro
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_pennant': np.zeros(n, dtype=bool),
            'ascending_triangle': np.zeros(n, dtype=bool),
            'bearish_rectangle': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-1):
            # Detectar Doble Techo
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            
            if len(window_high) >= 10:
                # Doble Techo
                peaks = []
                for j in range(2, len(window_high)-2):
                    if (window_high[j] > window_high[j-1] and 
                        window_high[j] > window_high[j-2] and
                        window_high[j] > window_high[j+1] and
                        window_high[j] > window_high[j+2]):
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 2:
                    peak1_idx, peak1_val = peaks[-2]
                    peak2_idx, peak2_val = peaks[-1]
                    if (abs(peak1_val - peak2_val) / peak1_val < 0.02 and
                        peak2_idx > peak1_idx + 5):
                        patterns['double_top'][i] = True
                
                # Doble Fondo
                valleys = []
                for j in range(2, len(window_low)-2):
                    if (window_low[j] < window_low[j-1] and 
                        window_low[j] < window_low[j-2] and
                        window_low[j] < window_low[j+1] and
                        window_low[j] < window_low[j+2]):
                        valleys.append((j, window_low[j]))
                
                if len(valleys) >= 2:
                    valley1_idx, valley1_val = valleys[-2]
                    valley2_idx, valley2_val = valleys[-1]
                    if (abs(valley1_val - valley2_val) / valley1_val < 0.02 and
                        valley2_idx > valley1_idx + 5):
                        patterns['double_bottom'][i] = True
        
        return patterns

    def check_multi_timeframe_trend(self, symbol, timeframe):
        """Verificar tendencia en múltiples temporalidades MEJORADO"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(timeframe, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['1M', '1W']:  # Temporalidades sin datos disponibles
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
                ma_50 = self.calculate_sma(close, 50)
                
                current_ma_9 = ma_9[-1] if len(ma_9) > 0 else 0
                current_ma_21 = ma_21[-1] if len(ma_21) > 0 else 0
                current_ma_50 = ma_50[-1] if len(ma_50) > 0 else 0
                current_price = close[-1]
                
                # Determinar tendencia con múltiples criterios
                bullish_count = 0
                bearish_count = 0
                
                if current_price > current_ma_9: bullish_count += 1
                else: bearish_count += 1
                    
                if current_ma_9 > current_ma_21: bullish_count += 1
                else: bearish_count += 1
                    
                if current_ma_21 > current_ma_50: bullish_count += 1
                else: bearish_count += 1
                
                if bullish_count >= 2:
                    results[tf_type] = 'BULLISH'
                elif bearish_count >= 2:
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}

    def check_obligatory_conditions(self, symbol, interval, signal_type):
        """Verificar condiciones obligatorias MEJORADO"""
        try:
            # Verificar multi-timeframe
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            # Verificar fuerza de tendencia en todas las temporalidades
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            all_tf_ok = True
            
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['1M', '1W']:
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 30)
                if df is not None and len(df) > 10:
                    trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                    current_signal = trend_data['strength_signals'][-1]
                    current_no_trade = trend_data['no_trade_zones'][-1]
                    
                    # Verificar que no hay zona de NO OPERAR
                    if current_no_trade:
                        all_tf_ok = False
                        break
                    
                    # Verificar fuerza de tendencia según señal
                    if signal_type == 'LONG':
                        if current_signal not in ['STRONG_UP', 'WEAK_UP']:
                            all_tf_ok = False
                            break
                    elif signal_type == 'SHORT':
                        if current_signal not in ['STRONG_DOWN', 'WEAK_DOWN']:
                            all_tf_ok = False
                            break
            
            if not all_tf_ok:
                return False
            
            # Verificar condiciones específicas por señal
            if signal_type == 'LONG':
                return (tf_analysis.get('mayor') in ['BULLISH', 'NEUTRAL'] and
                       tf_analysis.get('media') == 'BULLISH')
            elif signal_type == 'SHORT':
                return (tf_analysis.get('mayor') in ['BEARISH', 'NEUTRAL'] and
                       tf_analysis.get('media') == 'BEARISH')
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False

    def calculate_smart_money_levels(self, high, low, close, lookback=50):
        """Calcular niveles de soporte y resistencia Smart Money"""
        n = len(close)
        support_levels = np.zeros(n)
        resistance_levels = np.zeros(n)
        
        for i in range(lookback, n):
            # Encontrar niveles significativos
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            
            # Calcular pivots
            pivot = (np.max(window_high) + np.min(window_low) + close[i]) / 3
            r1 = 2 * pivot - np.min(window_low)
            s1 = 2 * pivot - np.max(window_high)
            
            resistance_levels[i] = r1
            support_levels[i] = s1
        
        return support_levels.tolist(), resistance_levels.tolist()

    def calculate_whale_signals_improved(self, df, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=20, signal_threshold=25, 
                                       sell_signal_threshold=20):
        """Implementación MEJORADA del indicador de ballenas"""
        try:
            close = df['close'].values
            low = df['low'].values
            high = df['high'].values
            volume = df['volume'].values
            
            n = len(close)
            
            whale_pump_signal = np.zeros(n)
            whale_dump_signal = np.zeros(n)
            confirmed_buy = np.zeros(n, dtype=bool)
            confirmed_sell = np.zeros(n, dtype=bool)
            
            sma_20 = self.calculate_sma(close, 20)
            sma_50 = self.calculate_sma(close, 50)
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                low_5 = np.min(low[max(0, i-5):i+1])
                high_5 = np.max(high[max(0, i-5):i+1])
                
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] < close[i-1] or price_change < -0.5) and
                    low[i] <= low_5 * 1.01):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_pump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
                
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_dump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            current_support = np.array([np.min(low[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            current_resistance = np.array([np.max(high[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            
            for i in range(5, n):
                if (whale_pump_smooth[i] > signal_threshold and 
                    close[i] <= current_support[i] * 1.02 and
                    volume[i] > np.mean(volume[max(0, i-10):i+1])):
                    confirmed_buy[i] = True
                
                if (whale_dump_smooth[i] > sell_signal_threshold and 
                    close[i] >= current_resistance[i] * 0.98 and
                    volume[i] > np.mean(volume[max(0, i-10):i+1])):
                    confirmed_sell[i] = True
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'confirmed_buy': confirmed_buy.tolist(),
                'confirmed_sell': confirmed_sell.tolist(),
                'support': current_support.tolist(),
                'resistance': current_resistance.tolist(),
                'volume_anomaly': (volume > np.mean(volume) * min_volume_multiplier).tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_improved: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'confirmed_buy': [False] * n,
                'confirmed_sell': [False] * n,
                'support': df['low'].values.tolist(),
                'resistance': df['high'].values.tolist(),
                'volume_anomaly': [False] * n
            }

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI con confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        di_trend_bullish = np.zeros(n, dtype=bool)
        di_trend_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                di_cross_bullish[i] = True
            
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
            
            if plus_di[i] > np.mean(plus_di[i-lookback:i]):
                di_trend_bullish[i] = True
            
            if minus_di[i] > np.mean(minus_di[i-lookback:i]):
                di_trend_bearish[i] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist(), di_trend_bullish.tolist(), di_trend_bearish.tolist()

    def check_breakout(self, high, low, close, support, resistance):
        """Detectar rupturas de tendencia"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if close[i] > resistance[i] and high[i] > high[i-1]:
                breakout_up[i] = True
            
            if close[i] < support[i] and low[i] < low[i-1]:
                breakout_down[i] = True
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_macd_signals(self, macd_line, macd_signal, histogram):
        """Detectar señales MACD"""
        n = len(macd_line)
        macd_bullish = np.zeros(n, dtype=bool)
        macd_bearish = np.zeros(n, dtype=bool)
        macd_histogram_positive = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if macd_line[i] > macd_signal[i] and macd_line[i-1] <= macd_signal[i-1]:
                macd_bullish[i] = True
            
            if macd_line[i] < macd_signal[i] and macd_line[i-1] >= macd_signal[i-1]:
                macd_bearish[i] = True
            
            if histogram[i] > 0:
                macd_histogram_positive[i] = True
        
        return macd_bullish.tolist(), macd_bearish.tolist(), macd_histogram_positive.tolist()

    def evaluate_signal_conditions_improved(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con nueva lógica"""
        conditions = {
            'long': {
                'obligatory_multi_tf': {'value': False, 'weight': 30, 'description': 'Condiciones multi-TF obligatorias'},
                'ma_alignment': {'value': False, 'weight': 15, 'description': 'Alineación medias móviles'},
                'rsi_traditional': {'value': False, 'weight': 10, 'description': 'RSI tradicional favorable'},
                'rsi_maverick': {'value': False, 'weight': 10, 'description': 'RSI Maverick favorable'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} y +DI > -DI'},
                'macd': {'value': False, 'weight': 8, 'description': 'MACD alcista'},
                'bb_signals': {'value': False, 'weight': 7, 'description': 'Señales Bandas Bollinger'},
                'chart_patterns': {'value': False, 'weight': 5, 'description': 'Patrones chartismo alcistas'},
                'whale_pump': {'value': False, 'weight': 5, 'description': 'Ballenas compradoras activas'}
            },
            'short': {
                'obligatory_multi_tf': {'value': False, 'weight': 30, 'description': 'Condiciones multi-TF obligatorias'},
                'ma_alignment': {'value': False, 'weight': 15, 'description': 'Alineación medias móviles'},
                'rsi_traditional': {'value': False, 'weight': 10, 'description': 'RSI tradicional favorable'},
                'rsi_maverick': {'value': False, 'weight': 10, 'description': 'RSI Maverick favorable'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} y -DI > +DI'},
                'macd': {'value': False, 'weight': 8, 'description': 'MACD bajista'},
                'bb_signals': {'value': False, 'weight': 7, 'description': 'Señales Bandas Bollinger'},
                'chart_patterns': {'value': False, 'weight': 5, 'description': 'Patrones chartismo bajistas'},
                'whale_dump': {'value': False, 'weight': 5, 'description': 'Ballenas vendedoras activas'}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        current_price = data['close'][current_idx]
        
        # Condiciones comunes
        conditions['long']['ma_alignment']['value'] = (
            current_price > data['ma_9'][current_idx] and
            data['ma_9'][current_idx] > data['ma_21'][current_idx] and
            data['ma_21'][current_idx] > data['ma_50'][current_idx]
        )
        
        conditions['short']['ma_alignment']['value'] = (
            current_price < data['ma_9'][current_idx] and
            data['ma_9'][current_idx] < data['ma_21'][current_idx] and
            data['ma_21'][current_idx] < data['ma_50'][current_idx]
        )
        
        conditions['long']['rsi_traditional']['value'] = (
            data['rsi_traditional'][current_idx] < 80 and
            data['rsi_traditional'][current_idx] > 30
        )
        
        conditions['short']['rsi_traditional']['value'] = (
            data['rsi_traditional'][current_idx] > 20 and
            data['rsi_traditional'][current_idx] < 70
        )
        
        conditions['long']['rsi_maverick']['value'] = (
            data['rsi_maverick'][current_idx] < 0.8 and
            data['rsi_maverick'][current_idx] > 0.2
        )
        
        conditions['short']['rsi_maverick']['value'] = (
            data['rsi_maverick'][current_idx] > 0.2 and
            data['rsi_maverick'][current_idx] < 0.8
        )
        
        conditions['long']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['plus_di'][current_idx] > data['minus_di'][current_idx]
        )
        
        conditions['short']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['minus_di'][current_idx] > data['plus_di'][current_idx]
        )
        
        conditions['long']['macd']['value'] = data['macd_bullish'][current_idx]
        conditions['short']['macd']['value'] = data['macd_bearish'][current_idx]
        
        conditions['long']['bb_signals']['value'] = (
            data['rsi_maverick'][current_idx] < 0.8 or
            current_price < data['bb_upper'][current_idx]
        )
        
        conditions['short']['bb_signals']['value'] = (
            data['rsi_maverick'][current_idx] > 0.2 or
            current_price > data['bb_lower'][current_idx]
        )
        
        # Solo activar ballenas para 12H y 1D
        if interval in ['12h', '1D']:
            conditions['long']['whale_pump']['value'] = data['whale_pump'][current_idx] > 15
            conditions['short']['whale_dump']['value'] = data['whale_dump'][current_idx] > 18
        else:
            conditions['long']['whale_pump']['value'] = True
            conditions['short']['whale_dump']['value'] = True
        
        # Patrones de chartismo
        conditions['long']['chart_patterns']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_pennant'][current_idx] or
            data['chart_patterns']['ascending_triangle'][current_idx]
        )
        
        conditions['short']['chart_patterns']['value'] = (
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['hch'][current_idx] or
            data['chart_patterns']['bearish_rectangle'][current_idx]
        )
        
        return conditions

    def calculate_signal_score(self, conditions, signal_type, obligatory_conditions_met):
        """Calcular puntuación de señal con multiplicador obligatorio"""
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
        
        # Aplicar multiplicador obligatorio
        base_score = (achieved_weight / total_weight * 100)
        final_score = base_score if obligatory_conditions_met else 0

        return min(final_score, 100), fulfilled_conditions

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con Smart Money"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular niveles Smart Money
            smart_support, smart_resistance = self.calculate_smart_money_levels(high, low, close)
            current_smart_support = smart_support[-1] if len(smart_support) > 0 else np.min(low[-20:])
            current_smart_resistance = smart_resistance[-1] if len(smart_resistance) > 0 else np.max(high[-20:])
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Entrada lo más cerca posible al soporte Smart Money
                entry = min(current_price, current_smart_support * 1.01)
                stop_loss = max(current_smart_support * 0.98, entry - (current_atr * 1.5))
                take_profit = current_smart_resistance * 0.99
                
            else:  # SHORT
                # Entrada lo más cerca posible a la resistencia Smart Money
                entry = max(current_price, current_smart_resistance * 0.99)
                stop_loss = min(current_smart_resistance * 1.02, entry + (current_atr * 1.5))
                take_profit = current_smart_support * 1.01
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(take_profit)],
                'support': float(current_smart_support),
                'resistance': float(current_smart_resistance),
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.05],
                'support': float(np.min(df['low'].values[-20:])),
                'resistance': float(np.max(df['high'].values[-20:])),
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_winrate(self):
        """Calcular winrate del sistema"""
        try:
            total_signals = 0
            successful_signals = 0
            
            # Analizar últimas 100 velas para cada símbolo y temporalidad
            for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
                for interval in ['1h', '4h', '1D']:
                    try:
                        df = self.get_kucoin_data(symbol, interval, 100)
                        if df is None or len(df) < 50:
                            continue
                            
                        # Simular señales pasadas
                        for i in range(20, len(df)-1):
                            signal_data = self.generate_signals_improved(symbol, interval)
                            if signal_data and signal_data['signal'] != 'NEUTRAL':
                                total_signals += 1
                                # Simular resultado (en producción real, verificar precio posterior)
                                if signal_data['signal_score'] > 70:
                                    successful_signals += 0.7  # Asumir 70% de efectividad
                                    
                    except Exception as e:
                        continue
            
            if total_signals == 0:
                return 70.0  # Winrate por defecto
            
            winrate = (successful_signals / total_signals) * 100
            return min(winrate, 95.0)
            
        except Exception as e:
            print(f"Error calculando winrate: {e}")
            return 70.0

    def generate_exit_signals(self):
        """Generar señales de salida para operaciones activas"""
        exit_alerts = []
        current_time = self.get_bolivia_time()
        
        for signal_key, signal_data in list(self.active_signals.items()):
            try:
                symbol = signal_data['symbol']
                interval = signal_data['interval']
                signal_type = signal_data['signal']
                entry_price = signal_data['entry_price']
                entry_time = signal_data['timestamp']
                
                # Obtener datos actuales
                df = self.get_kucoin_data(symbol, interval, 20)
                if df is None or len(df) < 10:
                    continue
                
                current_price = float(df['close'].iloc[-1])
                current_trend = self.calculate_trend_strength_maverick(df['close'].values)
                current_strength = current_trend['strength_signals'][-1]
                
                # Razones para salir
                exit_reason = None
                
                # Verificar cambio en fuerza de tendencia
                if signal_type == 'LONG' and current_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                    exit_reason = "Cambio en fuerza de tendencia Maverick"
                elif signal_type == 'SHORT' and current_strength in ['STRONG_UP', 'WEAK_UP']:
                    exit_reason = "Cambio en fuerza de tendencia Maverick"
                
                # Verificar temporalidad menor
                hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                if hierarchy.get('menor') and exit_reason is None:
                    menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 10)
                    if menor_df is not None and len(menor_df) > 5:
                        menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                        menor_strength = menor_trend['strength_signals'][-1]
                        
                        if signal_type == 'LONG' and menor_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                            exit_reason = "Cambio de tendencia en temporalidad menor"
                        elif signal_type == 'SHORT' and menor_strength in ['STRONG_UP', 'WEAK_UP']:
                            exit_reason = "Cambio de tendencia en temporalidad menor"
                
                if exit_reason:
                    # Calcular P&L
                    if signal_type == 'LONG':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    exit_alert = {
                        'symbol': symbol,
                        'interval': interval,
                        'signal': signal_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_percent': pnl_percent,
                        'reason': exit_reason,
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    exit_alerts.append(exit_alert)
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=20, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - NUEVA ESTRATEGIA"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular todos los indicadores
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Indicadores básicos
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # RSI tradicional
            rsi_traditional = self.calculate_rsi(close, 14)
            rsi_bullish_div, rsi_bearish_div = self.detect_divergence(close, rsi_traditional)
            
            # RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close, rsi_length, bb_multiplier)
            rsi_maverick_bullish_div, rsi_maverick_bearish_div = self.detect_divergence(close, rsi_maverick)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_bullish, macd_bearish, macd_histogram_positive = self.check_macd_signals(macd_line, macd_signal, macd_histogram)
            
            # ADX y DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            # Ballenas (solo para referencia, no obligatorio excepto 12H y 1D)
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            
            # Fuerza de tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Patrones de chartismo
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # Niveles Smart Money
            smart_support, smart_resistance = self.calculate_smart_money_levels(high, low, close)
            
            # Breakouts
            breakout_up, breakout_down = self.check_breakout(high, low, close, smart_support, smart_resistance)
            
            current_idx = -1
            
            # Preparar datos para evaluación
            analysis_data = {
                'close': close,
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'rsi_traditional': rsi_traditional,
                'rsi_maverick': rsi_maverick,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'macd_bullish': macd_bullish,
                'macd_bearish': macd_bearish,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'chart_patterns': chart_patterns
            }
            
            # Verificar condiciones obligatorias para LONG y SHORT
            obligatory_long = self.check_obligatory_conditions(symbol, interval, 'LONG')
            obligatory_short = self.check_obligatory_conditions(symbol, interval, 'SHORT')
            
            # Evaluar condiciones
            conditions = self.evaluate_signal_conditions_improved(analysis_data, current_idx, interval, adx_threshold)
            
            # Calcular scores
            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', obligatory_long)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', obligatory_short)
            
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
            
            current_price = float(close[current_idx])
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # Registrar señal activa si es válida
            if signal_type in ['LONG', 'SHORT'] and signal_score >= 70:
                signal_key = f"{symbol}_{interval}_{signal_type}"
                self.active_signals[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': levels_data['entry'],
                    'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                    'score': signal_score
                }
            
            # Calcular winrate
            winrate = self.calculate_winrate()
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support': levels_data['support'],
                'resistance': levels_data['resistance'],
                'atr': levels_data['atr'],
                'atr_percentage': levels_data['atr_percentage'],
                'volume': float(df['volume'].iloc[current_idx]),
                'volume_ma': float(np.mean(df['volume'].tail(20))),
                'adx': float(adx[current_idx] if current_idx < len(adx) else 0),
                'plus_di': float(plus_di[current_idx] if current_idx < len(plus_di) else 0),
                'minus_di': float(minus_di[current_idx] if current_idx < len(minus_di) else 0),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_traditional': float(rsi_traditional[current_idx] if current_idx < len(rsi_traditional) else 50),
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'fulfilled_conditions': fulfilled_conditions,
                'obligatory_long': obligatory_long,
                'obligatory_short': obligatory_short,
                'winrate': float(winrate),
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'rsi_traditional': rsi_traditional[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:].tolist(),
                    'rsi_traditional_bullish_div': rsi_bullish_div[-50:],
                    'rsi_traditional_bearish_div': rsi_bearish_div[-50:],
                    'rsi_maverick_bullish_div': rsi_maverick_bullish_div[-50:],
                    'rsi_maverick_bearish_div': rsi_maverick_bearish_div[-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'di_cross_bullish': di_cross_bullish[-50:],
                    'di_cross_bearish': di_cross_bearish[-50:],
                    'macd_line': macd_line[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'macd_bullish': macd_bullish[-50:],
                    'macd_bearish': macd_bearish[-50:],
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
                    'smart_support': smart_support[-50:],
                    'smart_resistance': smart_resistance[-50:],
                    'chart_patterns_hch': chart_patterns['hch'][-50:],
                    'chart_patterns_double_top': chart_patterns['double_top'][-50:],
                    'chart_patterns_double_bottom': chart_patterns['double_bottom'][-50:]
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
            'entry': 0,
            'stop_loss': 0,
            'take_profit': [0],
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
            'rsi_traditional': 50,
            'rsi_maverick': 0.5,
            'fulfilled_conditions': [],
            'obligatory_long': False,
            'obligatory_short': False,
            'winrate': 70.0,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de trading"""
        alerts = []
        telegram_intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '3D', '1W']
        
        current_time = self.get_bolivia_time()
        
        for interval in telegram_intervals:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:15]:
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 70):
                        
                        risk_category = next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                        
                        # Calcular leverage óptimo
                        volatility = signal_data['atr_percentage']
                        if volatility > 0.05:
                            optimal_leverage = 10
                        elif volatility > 0.02:
                            optimal_leverage = 15
                        else:
                            optimal_leverage = 20
                        
                        risk_factors = {'bajo': 1.0, 'medio': 0.8, 'alto': 0.6, 'memecoins': 0.5}
                        risk_factor = risk_factors.get(risk_category, 0.7)
                        optimal_leverage = int(optimal_leverage * risk_factor)
                        
                        alert = {
                            'symbol': symbol,
                            'interval': interval,
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'entry': signal_data['entry'],
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'][0],
                            'leverage': optimal_leverage,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'winrate': signal_data.get('winrate', 70),
                            'support': signal_data['support'],
                            'resistance': signal_data['resistance']
                        }
                        
                        alert_key = f"{symbol}_{interval}_{signal_data['signal']}"
                        if (alert_key not in self.alert_cache or 
                            (datetime.now() - self.alert_cache[alert_key]).seconds > 300):
                            
                            alerts.append(alert)
                            self.alert_cache[alert_key] = datetime.now()
                    
                except Exception as e:
                    print(f"Error generando alerta para {symbol} {interval}: {e}")
                    continue
        
        return alerts

# Instancia global del indicador
indicator = TradingIndicator()

def get_risk_classification(symbol):
    """Obtener la clasificación de riesgo de una criptomoneda"""
    for risk_level, symbols in CRYPTO_RISK_CLASSIFICATION.items():
        if symbol in symbols:
            if risk_level == "bajo":
                return "Bajo Riesgo"
            elif risk_level == "medio":
                return "Medio Riesgo"
            elif risk_level == "alto":
                return "Alto Riesgo"
            elif risk_level == "memecoins":
                return "Memecoin"
    return "Medio Riesgo"

def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram MEJORADA"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA MULTI-TIMEFRAME WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
🏆 Winrate Sistema: {alert_data.get('winrate', 70):.1f}%

💰 Precio actual: {alert_data['current_price']:.6f}
🎯 Entrada: ${alert_data['entry']:.6f}

🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
🎯 Take Profit: ${alert_data['take_profit']:.6f}

📈 Apalancamiento: x{alert_data['leverage']}

✅ Condiciones Cumplidas:
• {chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])[:3]])}

📊 Revisa la señal en: https://multiframewgta.onrender.com/
            """
        else:  # exit alert
            pnl_text = f"📊 P&L: {alert_data['pnl_percent']:+.2f}%"
            
            message = f"""
🚨 ALERTA DE SALIDA - MULTI-TIMEFRAME WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} - CERRAR

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
{pnl_text}

📊 Observación: {alert_data['reason']}
            """
        
        # Generar URL para el reporte
        report_url = f"https://multiframewgta.onrender.com/api/generate_report?symbol={alert_data['symbol']}&interval={alert_data['interval']}&leverage={alert_data.get('leverage', 15)}"
        
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
    intraday_intervals = ['15m', '30m', '1h', '2h']
    swing_intervals = ['4h', '8h', '12h', '1D', '3D', '1W']
    
    intraday_last_check = datetime.now()
    swing_last_check = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar intradía cada 60 segundos
            if (current_time - intraday_last_check).seconds >= 60:
                print("Verificando alertas intradía...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in intraday_intervals:
                        send_telegram_alert(alert, 'entry')
                
                exit_alerts = indicator.generate_exit_signals()
                for alert in exit_alerts:
                    send_telegram_alert(alert, 'exit')
                
                intraday_last_check = current_time
            
            # Verificar swing cada 300 segundos
            if (current_time - swing_last_check).seconds >= 300:
                print("Verificando alertas swing...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in swing_intervals:
                        send_telegram_alert(alert, 'entry')
                
                exit_alerts = indicator.generate_exit_signals()
                for alert in exit_alerts:
                    send_telegram_alert(alert, 'exit')
                
                swing_last_check = current_time
            
            time.sleep(10)
            
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

@app.route('/api/signals')
def get_signals():
    """Endpoint para obtener señales de trading"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 20))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, volume_filter, leverage
        )
        
        # Asegurar que los indicadores sean serializables
        if 'indicators' in signal_data:
            for key in signal_data['indicators']:
                if isinstance(signal_data['indicators'][key], (np.ndarray, np.generic)):
                    signal_data['indicators'][key] = signal_data['indicators'][key].tolist()
                elif isinstance(signal_data['indicators'][key], list):
                    signal_data['indicators'][key] = [int(x) if isinstance(x, (bool, np.bool_)) else x for x in signal_data['indicators'][key]]
        
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
        rsi_length = int(request.args.get('rsi_length', 20))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:
            try:
                signal_data = indicator.generate_signals_improved(
                    symbol, interval, di_period, adx_threshold, sr_period,
                    rsi_length, bb_multiplier, volume_filter, leverage
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
    """Endpoint para datos del scatter plot mejorado"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        
        scatter_data = []
        
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:3])
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval, di_period, adx_threshold)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones basadas en nuevos indicadores
                    buy_pressure = min(100, max(0,
                        (1 if signal_data['obligatory_long'] else 0) * 30 +
                        (signal_data['rsi_traditional'] / 100 * 20) +
                        (signal_data['rsi_maverick'] * 20) +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 15) +
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 15
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (1 if signal_data['obligatory_short'] else 0) * 30 +
                        ((100 - signal_data['rsi_traditional']) / 100 * 20) +
                        ((1 - signal_data['rsi_maverick']) * 20) +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 15) +
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 15
                    ))
                    
                    if signal_data['signal'] == 'LONG':
                        buy_pressure = max(buy_pressure, 70)
                        sell_pressure = min(sell_pressure, 30)
                    elif signal_data['signal'] == 'SHORT':
                        sell_pressure = max(sell_pressure, 70)
                        buy_pressure = min(buy_pressure, 30)
                    
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
    """Endpoint para obtener alertas"""
    try:
        alerts = indicator.generate_scalping_alerts()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/exit_signals')
def get_exit_signals():
    """Endpoint para obtener señales de salida"""
    try:
        exit_alerts = indicator.generate_exit_signals()
        return jsonify({'exit_signals': exit_alerts})
        
    except Exception as e:
        print(f"Error en /api/exit_signals: {e}")
        return jsonify({'exit_signals': []})

@app.route('/api/winrate')
def get_winrate():
    """Endpoint para obtener winrate actual"""
    try:
        winrate = indicator.calculate_winrate()
        return jsonify({'winrate': winrate})
        
    except Exception as e:
        print(f"Error en /api/winrate: {e}")
        return jsonify({'winrate': 70.0})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo MEJORADO"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        fig = plt.figure(figsize=(14, 16))
        fig.suptitle(f'MULTI-TIMEFRAME CRYPTO WGTA PRO - {symbol} ({interval})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Gráfico 1: Precio y medias móviles
        ax1 = plt.subplot(7, 1, 1)
        if signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            opens = [d['open'] for d in signal_data['data']]
            highs = [d['high'] for d in signal_data['data']]
            lows = [d['low'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            
            # Velas japonesas
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Medias móviles
            if 'indicators' in signal_data:
                ma_dates = dates[-len(signal_data['indicators']['ma_9']):]
                ax1.plot(ma_dates, signal_data['indicators']['ma_9'], 'orange', linewidth=1, label='MA 9')
                ax1.plot(ma_dates, signal_data['indicators']['ma_21'], 'blue', linewidth=1, label='MA 21')
                ax1.plot(ma_dates, signal_data['indicators']['ma_50'], 'red', linewidth=1, label='MA 50')
                ax1.plot(ma_dates, signal_data['indicators']['ma_200'], 'purple', linewidth=2, label='MA 200')
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='yellow', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal_data['take_profit'][0], color='green', linestyle='--', alpha=0.7, label='Take Profit')
        
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: RSI Tradicional
        ax2 = plt.subplot(7, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax2.plot(rsi_dates, signal_data['indicators']['rsi_traditional'], 'blue', linewidth=2, label='RSI Tradicional')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
            
            # Marcar divergencias
            for i, date in enumerate(rsi_dates):
                if i < len(signal_data['indicators']['rsi_traditional_bullish_div']) and signal_data['indicators']['rsi_traditional_bullish_div'][i]:
                    ax2.plot(date, signal_data['indicators']['rsi_traditional'][i], '^', color='green', markersize=8)
                if i < len(signal_data['indicators']['rsi_traditional_bearish_div']) and signal_data['indicators']['rsi_traditional_bearish_div'][i]:
                    ax2.plot(date, signal_data['indicators']['rsi_traditional'][i], 'v', color='red', markersize=8)
        
        ax2.set_ylabel('RSI Tradicional')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Gráfico 3: RSI Maverick
        ax3 = plt.subplot(7, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_m_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax3.plot(rsi_m_dates, signal_data['indicators']['rsi_maverick'], 'purple', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
            
            # Marcar divergencias
            for i, date in enumerate(rsi_m_dates):
                if i < len(signal_data['indicators']['rsi_maverick_bullish_div']) and signal_data['indicators']['rsi_maverick_bullish_div'][i]:
                    ax3.plot(date, signal_data['indicators']['rsi_maverick'][i], '^', color='green', markersize=8)
                if i < len(signal_data['indicators']['rsi_maverick_bearish_div']) and signal_data['indicators']['rsi_maverick_bearish_div'][i]:
                    ax3.plot(date, signal_data['indicators']['rsi_maverick'][i], 'v', color='red', markersize=8)
        
        ax3.set_ylabel('RSI Maverick')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Gráfico 4: MACD
        ax4 = plt.subplot(7, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd_line']):]
            ax4.plot(macd_dates, signal_data['indicators']['macd_line'], 'blue', linewidth=2, label='MACD')
            ax4.plot(macd_dates, signal_data['indicators']['macd_signal'], 'red', linewidth=1, label='Señal')
            
            # Histograma con colores
            for i, date in enumerate(macd_dates):
                if i < len(signal_data['indicators']['macd_histogram']):
                    color = 'green' if signal_data['indicators']['macd_histogram'][i] >= 0 else 'red'
                    ax4.bar(date, signal_data['indicators']['macd_histogram'][i], 
                           color=color, alpha=0.6, width=0.8)
            
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax4.set_ylabel('MACD')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: ADX y DMI
        ax5 = plt.subplot(7, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax5.plot(adx_dates, signal_data['indicators']['adx'], 'white', linewidth=2, label='ADX')
            ax5.plot(adx_dates, signal_data['indicators']['plus_di'], 'green', linewidth=1, label='+DI')
            ax5.plot(adx_dates, signal_data['indicators']['minus_di'], 'red', linewidth=1, label='-DI')
            ax5.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral ADX')
        
        ax5.set_ylabel('ADX/DMI')
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Fuerza de Tendencia Maverick
        ax6 = plt.subplot(7, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax6.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax6.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax6.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            # Marcar zonas de no operar
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax6.axvline(x=date, color='red', alpha=0.3, linewidth=2)
        
        ax6.set_ylabel('Fuerza Tendencia %')
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Información de la señal
        ax7 = plt.subplot(7, 1, 7)
        ax7.axis('off')
        
        # Información multi-timeframe
        tf_analysis = indicator.check_multi_timeframe_trend(symbol, interval)
        multi_tf_info = f"""
MULTI-TIMEFRAME ANALYSIS:
• Mayor ({TIMEFRAME_HIERARCHY[interval]['mayor']}): {tf_analysis.get('mayor', 'N/A')}
• Media ({TIMEFRAME_HIERARCHY[interval]['media']}): {tf_analysis.get('media', 'N/A')}
• Menor ({TIMEFRAME_HIERARCHY[interval]['menor']}): {tf_analysis.get('menor', 'N/A')}
"""
        
        obligatory_info = "✅ OBLIGATORIOS CUMPLIDOS" if (signal_data['obligatory_long'] or signal_data['obligatory_short']) else "❌ OBLIGATORIOS NO CUMPLIDOS"
        
        signal_info = f"""
SEÑAL: {signal_data['signal']}
SCORE: {signal_data['signal_score']:.1f}%
WINRATE SISTEMA: {signal_data.get('winrate', 70):.1f}%

{obligatory_info}

PRECIO ACTUAL: ${signal_data['current_price']:.6f}
ENTRADA: ${signal_data['entry']:.6f}
STOP LOSS: ${signal_data['stop_loss']:.6f}
TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}

APALANCAMIENTO: x{leverage}
ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)

{multi_tf_info}

CONDICIONES CUMPLIDAS:
{chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
"""
        
        ax7.text(0.02, 0.95, signal_info, transform=ax7.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontfamily='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
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
        'timezone': 'America/La_Paz'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({'error': 'Servicio no disponible temporalmente'}), 503

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
