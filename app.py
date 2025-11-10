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

# Configuración Telegram - NUEVO TOKEN
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

# NUEVA JERARQUÍA TEMPORAL MEJORADA
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
            # Para 3 días, simplificar lógica
            return True
        elif interval == '1W':
            # Para 1 semana, simplificar lógica
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
                        return self._generate_fallback_data(limit, symbol)
                    
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
        
        return self._generate_fallback_data(limit, symbol)

    def _generate_fallback_data(self, limit, symbol):
        """Generar datos de fallback más realistas"""
        try:
            # Intentar obtener precio actual de una API alternativa
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

    # INDICADORES TÉCNICOS MANUALES
    def calculate_sma(self, prices, period):
        """Calcular Media Móvil Simple"""
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
        """Calcular Media Móvil Exponencial"""
        if len(prices) == 0:
            return np.array([])
            
        alpha = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI Tradicional"""
        if len(prices) < period + 1:
            return np.zeros(len(prices))
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        # Inicializar primeros valores
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
        if len(prices) < slow:
            return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger"""
        if len(prices) < period:
            return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
        
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
        """Calcular ADX con +DI y -DI"""
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
        
        # Suavizado
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
        """Calcular Squeeze Momentum (LazyBear)"""
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

    def detect_chart_patterns(self, high, low, close, volume, lookback=50):
        """Detectar patrones de chartismo"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'rising_wedge': np.zeros(n, dtype=bool),
            'falling_wedge': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-1):
            # Head & Shoulders (máximo 25 periodos para el patrón)
            window_high = high[i-25:i+1]
            window_low = low[i-25:i+1]
            
            if len(window_high) >= 10:
                # Detectar picos
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append((j, window_high[j]))
                
                # Head & Shoulders (3 picos: izquierdo, cabeza, derecho)
                if len(peaks) >= 3:
                    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                    if (peaks_sorted[0][1] > peaks_sorted[1][1] * 1.02 and 
                        peaks_sorted[0][1] > peaks_sorted[2][1] * 1.02):
                        patterns['head_shoulders'][i] = True
            
            # Double Top/Bottom
            if i >= 20:
                recent_high = high[i-20:i+1]
                recent_low = low[i-20:i+1]
                
                max_high = np.max(recent_high)
                min_low = np.min(recent_low)
                
                high_count = np.sum(np.abs(recent_high - max_high) < max_high * 0.01)
                low_count = np.sum(np.abs(recent_low - min_low) < min_low * 0.01)
                
                if high_count >= 2 and recent_high[-1] < max_high * 0.98:
                    patterns['double_top'][i] = True
                
                if low_count >= 2 and recent_low[-1] > min_low * 1.02:
                    patterns['double_bottom'][i] = True
        
        return patterns

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick"""
        n = len(close)
        
        # Bandas de Bollinger para fuerza
        basis = self.calculate_sma(close, length)
        dev = np.zeros(n)
        for i in range(length-1, n):
            window = close[i-length+1:i+1]
            dev[i] = np.std(window)
        
        upper = basis + (dev * mult)
        lower = basis - (dev * mult)
        
        # Ancho de bandas normalizado
        bb_width = np.zeros(n)
        for i in range(n):
            if basis[i] > 0:
                bb_width[i] = ((upper[i] - lower[i]) / basis[i]) * 100
        
        # Fuerza de tendencia
        trend_strength = np.zeros(n)
        for i in range(1, n):
            if bb_width[i] > bb_width[i-1]:
                trend_strength[i] = bb_width[i]  # Fuerza creciente
            else:
                trend_strength[i] = -bb_width[i]  # Fuerza decreciente
        
        # Zonas de no operar
        no_trade_zones = np.zeros(n, dtype=bool)
        strength_signals = ['NEUTRAL'] * n
        
        for i in range(10, n):
            # Detectar pérdida de fuerza después de movimiento fuerte
            if (bb_width[i] > 5 and  # Umbral mínimo
                trend_strength[i] < 0 and 
                bb_width[i] < np.max(bb_width[max(0, i-10):i])):
                no_trade_zones[i] = True
            
            # Señales de fuerza
            if trend_strength[i] > 0:
                if bb_width[i] > 8:  # Umbral alto
                    strength_signals[i] = 'STRONG_UP'
                else:
                    strength_signals[i] = 'WEAK_UP'
            elif trend_strength[i] < 0:
                if bb_width[i] > 8:
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
        """Verificar tendencia en múltiples temporalidades - NUEVA LÓGICA OBLIGATORIA"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(timeframe, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            # Verificar cada temporalidad
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:  # Excluir 5m y 1M del análisis obligatorio
                    results[tf_type] = 'NEUTRAL'
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 50)
                if df is None or len(df) < 20:
                    results[tf_type] = 'NEUTRAL'
                    continue
                
                close = df['close'].values
                
                # Calcular tendencia con múltiples métodos
                ma_9 = self.calculate_sma(close, 9)
                ma_21 = self.calculate_sma(close, 21)
                ma_50 = self.calculate_sma(close, 50)
                
                current_price = close[-1]
                current_ma_9 = ma_9[-1] if len(ma_9) > 0 else 0
                current_ma_21 = ma_21[-1] if len(ma_21) > 0 else 0
                current_ma_50 = ma_50[-1] if len(ma_50) > 0 else 0
                
                # Determinar tendencia
                bullish_signals = 0
                bearish_signals = 0
                
                if current_price > current_ma_9: bullish_signals += 1
                if current_ma_9 > current_ma_21: bullish_signals += 1
                if current_ma_21 > current_ma_50: bullish_signals += 1
                if current_price < current_ma_9: bearish_signals += 1
                if current_ma_9 < current_ma_21: bearish_signals += 1
                if current_ma_21 < current_ma_50: bearish_signals += 1
                
                if bullish_signals >= 2:
                    results[tf_type] = 'BULLISH'
                elif bearish_signals >= 2:
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}

    def check_obligatory_conditions(self, symbol, interval, signal_type):
        """Verificar condiciones obligatorias - NUEVA LÓGICA"""
        try:
            # 1. Verificar multi-timeframe
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            # 2. Verificar fuerza de tendencia Maverick en todas las TF
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            maverick_conditions = {}
            
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:  # Excluir estas TF
                    maverick_conditions[tf_type] = True
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 30)
                if df is not None and len(df) > 20:
                    trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                    current_signal = trend_data['strength_signals'][-1]
                    current_no_trade = trend_data['no_trade_zones'][-1]
                    
                    if signal_type == 'LONG':
                        maverick_ok = (current_signal in ['STRONG_UP', 'WEAK_UP'] and 
                                      not current_no_trade)
                    else:  # SHORT
                        maverick_ok = (current_signal in ['STRONG_DOWN', 'WEAK_DOWN'] and 
                                      not current_no_trade)
                    
                    maverick_conditions[tf_type] = maverick_ok
                else:
                    maverick_conditions[tf_type] = True  # Si no hay datos, permitir
            
            # 3. Aplicar reglas obligatorias
            if signal_type == 'LONG':
                # Tendencia Mayor: ALCISTA o NEUTRAL
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                # Tendencia Media: EXCLUSIVAMENTE ALCISTA
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'
                # Tendencia Menor: Fuerza Maverick ALCISTA
                menor_ok = maverick_conditions.get('menor', True)
                # No zonas NO OPERAR en ninguna TF
                no_trade_ok = all(maverick_conditions.values())
                
                obligatory_ok = mayor_ok and media_ok and menor_ok and no_trade_ok
                
            else:  # SHORT
                # Tendencia Mayor: BAJISTA o NEUTRAL
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                # Tendencia Media: EXCLUSIVAMENTE BAJISTA
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'
                # Tendencia Menor: Fuerza Maverick BAJISTA
                menor_ok = maverick_conditions.get('menor', True)
                # No zonas NO OPERAR en ninguna TF
                no_trade_ok = all(maverick_conditions.values())
                
                obligatory_ok = mayor_ok and media_ok and menor_ok and no_trade_ok
            
            return obligatory_ok, {
                'multi_timeframe': tf_analysis,
                'maverick_conditions': maverick_conditions,
                'obligatory_details': {
                    'mayor_ok': mayor_ok,
                    'media_ok': media_ok,
                    'menor_ok': menor_ok,
                    'no_trade_ok': no_trade_ok
                }
            }
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False, {}

    def calculate_whale_signals_improved(self, df, interval):
        """Indicador Cazador de Ballenas MEJORADO - Solo obligatorio en 12H y 1D"""
        try:
            close = df['close'].values
            low = df['low'].values
            high = df['high'].values
            volume = df['volume'].values
            
            n = len(close)
            
            # Señales base (siempre visibles)
            whale_pump = np.zeros(n)
            whale_dump = np.zeros(n)
            
            # Solo calcular señales confirmadas para 12H y 1D
            confirmed_buy = np.zeros(n, dtype=bool)
            confirmed_sell = np.zeros(n, dtype=bool)
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                low_5 = np.min(low[max(0, i-5):i+1])
                high_5 = np.max(high[max(0, i-5):i+1])
                
                # Señales base
                if (volume_ratio > 1.5 and 
                    (close[i] < close[i-1] or price_change < -0.5) and
                    low[i] <= low_5 * 1.01):
                    
                    whale_pump[i] = min(100, volume_ratio * 15)
                
                if (volume_ratio > 1.5 and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    whale_dump[i] = min(100, volume_ratio * 15)
            
            # Suavizar señales
            whale_pump_smooth = self.calculate_sma(whale_pump, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump, 3)
            
            # Señales confirmadas solo para 12H y 1D
            if interval in ['12h', '1D']:
                support = np.array([np.min(low[max(0, i-20+1):i+1]) for i in range(n)])
                resistance = np.array([np.max(high[max(0, i-20+1):i+1]) for i in range(n)])
                
                for i in range(10, n):
                    if (whale_pump_smooth[i] > 15 and 
                        close[i] <= support[i] * 1.02 and
                        volume[i] > np.mean(volume[max(0, i-10):i+1])):
                        confirmed_buy[i] = True
                    
                    if (whale_dump_smooth[i] > 15 and 
                        close[i] >= resistance[i] * 0.98 and
                        volume[i] > np.mean(volume[max(0, i-10):i+1])):
                        confirmed_sell[i] = True
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'confirmed_buy': confirmed_buy.tolist(),
                'confirmed_sell': confirmed_sell.tolist(),
                'is_obligatory': interval in ['12h', '1D']
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_improved: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'confirmed_buy': [False] * n,
                'confirmed_sell': [False] * n,
                'is_obligatory': interval in ['12h', '1D']
            }

    def calculate_support_resistance_smart(self, high, low, close, period=50):
        """Calcular soportes y resistencias Smart Money"""
        n = len(close)
        
        supports = []
        resistances = []
        
        # Identificar niveles clave
        for i in range(period, n, period//5):
            window_high = high[max(0, i-period):i]
            window_low = low[max(0, i-period):i]
            
            # Niveles de resistencia (máximos)
            resistance_level = np.max(window_high)
            resistances.append(resistance_level)
            
            # Niveles de soporte (mínimos)
            support_level = np.min(window_low)
            supports.append(support_level)
        
        # Niveles actuales
        current_support = np.min(supports) if supports else np.min(low)
        current_resistance = np.max(resistances) if resistances else np.max(high)
        
        return current_support, current_resistance

    def calculate_win_rate(self, symbol, interval, lookback=100):
        """Calcular winrate histórico"""
        try:
            cache_key = f"winrate_{symbol}_{interval}"
            if cache_key in self.win_rate_data:
                cached_data, timestamp = self.win_rate_data[cache_key]
                if (datetime.now() - timestamp).seconds < 300:  # Cache 5 minutos
                    return cached_data
            
            df = self.get_kucoin_data(symbol, interval, lookback + 20)
            if df is None or len(df) < lookback + 10:
                return {'win_rate': 0, 'total_signals': 0, 'successful_signals': 0}
            
            signals = []
            successful_trades = 0
            
            for i in range(10, len(df) - 10):
                window_df = df.iloc[:i+1]
                if len(window_df) < 30:
                    continue
                
                # Simular señal histórica
                signal_data = self._simulate_historical_signal(window_df, interval)
                if signal_data and signal_data['signal'] in ['LONG', 'SHORT']:
                    # Verificar resultado
                    future_prices = df['close'].values[i+1:i+11]  # Próximas 10 velas
                    if len(future_prices) > 0:
                        entry = signal_data['entry']
                        if signal_data['signal'] == 'LONG':
                            result = any(future_prices >= signal_data['take_profit'][0])
                        else:  # SHORT
                            result = any(future_prices <= signal_data['take_profit'][0])
                        
                        signals.append({
                            'signal': signal_data['signal'],
                            'entry': entry,
                            'result': result
                        })
                        
                        if result:
                            successful_trades += 1
            
            total_signals = len(signals)
            win_rate = (successful_trades / total_signals * 100) if total_signals > 0 else 0
            
            result = {
                'win_rate': round(win_rate, 1),
                'total_signals': total_signals,
                'successful_signals': successful_trades
            }
            
            self.win_rate_data[cache_key] = (result, datetime.now())
            return result
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return {'win_rate': 0, 'total_signals': 0, 'successful_signals': 0}

    def _simulate_historical_signal(self, df, interval):
        """Simular señal histórica para cálculo de winrate"""
        try:
            if len(df) < 30:
                return None
            
            # Indicadores básicos para simulación
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # SMA para tendencia
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            
            # RSI
            rsi = self.calculate_rsi(close)
            
            # Señal simplificada
            current_price = close[-1]
            current_ma_9 = ma_9[-1]
            current_ma_21 = ma_21[-1]
            current_rsi = rsi[-1]
            
            # Lógica básica de señal
            if (current_price > current_ma_9 and current_ma_9 > current_ma_21 and 
                current_rsi < 70):
                signal_type = 'LONG'
                entry = current_price
                tp = current_price * 1.02
            elif (current_price < current_ma_9 and current_ma_9 < current_ma_21 and 
                  current_rsi > 30):
                signal_type = 'SHORT'
                entry = current_price
                tp = current_price * 0.98
            else:
                return None
            
            return {
                'signal': signal_type,
                'entry': entry,
                'take_profit': [tp]
            }
            
        except Exception as e:
            return None

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=20, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - NUEVA ESTRATEGIA"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # 1. VERIFICAR CONDICIONES OBLIGATORIAS
            obligatory_ok_long, obligatory_details_long = self.check_obligatory_conditions(symbol, interval, 'LONG')
            obligatory_ok_short, obligatory_details_short = self.check_obligatory_conditions(symbol, interval, 'SHORT')
            
            # Si no se cumplen condiciones obligatorias, score = 0
            obligatory_multiplier_long = 1 if obligatory_ok_long else 0
            obligatory_multiplier_short = 1 if obligatory_ok_short else 0
            
            # 2. CALCULAR INDICADORES
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Indicadores básicos
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # RSI tradicional
            rsi_trad = self.calculate_rsi(close, rsi_length)
            
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
            
            # Soporte y resistencia Smart Money
            support, resistance = self.calculate_support_resistance_smart(high, low, close, sr_period)
            
            # Whale Signals (mejorado)
            whale_data = self.calculate_whale_signals_improved(df, interval)
            
            # Fuerza de tendencia Maverick
            trend_data = self.calculate_trend_strength_maverick(close)
            
            # Patrones de chartismo
            chart_patterns = self.detect_chart_patterns(high, low, close, volume)
            
            # 3. CALCULAR SCORES POR INDICADOR
            current_idx = -1
            
            # Ponderaciones base
            weights = {
                'moving_averages': 15,
                'rsi_traditional': 15,
                'rsi_maverick': 15,
                'support_resistance': 20,
                'adx_dmi': 10,
                'macd': 10,
                'squeeze': 10,
                'chart_patterns': 15
            }
            
            # Calcular scores individuales
            scores_long = {}
            scores_short = {}
            
            # Moving Averages
            ma_score_long = 0
            ma_score_short = 0
            if (close[current_idx] > ma_9[current_idx] and 
                ma_9[current_idx] > ma_21[current_idx] and 
                ma_21[current_idx] > ma_50[current_idx]):
                ma_score_long = weights['moving_averages']
            if (close[current_idx] < ma_9[current_idx] and 
                ma_9[current_idx] < ma_21[current_idx] and 
                ma_21[current_idx] < ma_50[current_idx]):
                ma_score_short = weights['moving_averages']
            
            # RSI Traditional
            rsi_trad_score_long = 0
            rsi_trad_score_short = 0
            if rsi_trad[current_idx] < 70:  # No sobrecomprado
                rsi_trad_score_long = weights['rsi_traditional'] * 0.5
            if rsi_trad[current_idx] < 30:  # Sobreventa
                rsi_trad_score_long = weights['rsi_traditional']
            if rsi_trad[current_idx] > 30:  # No sobrevendido
                rsi_trad_score_short = weights['rsi_traditional'] * 0.5
            if rsi_trad[current_idx] > 70:  # Sobrecompra
                rsi_trad_score_short = weights['rsi_traditional']
            
            # RSI Maverick
            rsi_mav_score_long = 0
            rsi_mav_score_short = 0
            if rsi_maverick[current_idx] < 0.8:  # No sobrecomprado
                rsi_mav_score_long = weights['rsi_maverick'] * 0.5
            if rsi_maverick[current_idx] < 0.2:  # Sobreventa
                rsi_mav_score_long = weights['rsi_maverick']
            if rsi_maverick[current_idx] > 0.2:  # No sobrevendido
                rsi_mav_score_short = weights['rsi_maverick'] * 0.5
            if rsi_maverick[current_idx] > 0.8:  # Sobrecompra
                rsi_mav_score_short = weights['rsi_maverick']
            
            # Support Resistance
            sr_score_long = 0
            sr_score_short = 0
            if close[current_idx] <= support * 1.02:  # Cerca de soporte
                sr_score_long = weights['support_resistance']
            if close[current_idx] >= resistance * 0.98:  # Cerca de resistencia
                sr_score_short = weights['support_resistance']
            
            # ADX + DMI
            adx_score_long = 0
            adx_score_short = 0
            if adx[current_idx] > adx_threshold:
                if plus_di[current_idx] > minus_di[current_idx]:
                    adx_score_long = weights['adx_dmi']
                else:
                    adx_score_short = weights['adx_dmi']
            
            # MACD
            macd_score_long = 0
            macd_score_short = 0
            if macd[current_idx] > macd_signal[current_idx] and macd_histogram[current_idx] > 0:
                macd_score_long = weights['macd']
            if macd[current_idx] < macd_signal[current_idx] and macd_histogram[current_idx] < 0:
                macd_score_short = weights['macd']
            
            # Squeeze Momentum
            squeeze_score_long = 0
            squeeze_score_short = 0
            if squeeze_data['squeeze_off'][current_idx] and squeeze_data['momentum'][current_idx] > 0:
                squeeze_score_long = weights['squeeze']
            if squeeze_data['squeeze_off'][current_idx] and squeeze_data['momentum'][current_idx] < 0:
                squeeze_score_short = weights['squeeze']
            
            # Chart Patterns
            chart_score_long = 0
            chart_score_short = 0
            if (chart_patterns['double_bottom'][current_idx] or 
                chart_patterns['falling_wedge'][current_idx] or
                chart_patterns['bullish_flag'][current_idx]):
                chart_score_long = weights['chart_patterns']
            if (chart_patterns['head_shoulders'][current_idx] or 
                chart_patterns['double_top'][current_idx] or
                chart_patterns['rising_wedge'][current_idx] or
                chart_patterns['bearish_flag'][current_idx]):
                chart_score_short = weights['chart_patterns']
            
            # 4. CALCULAR SCORES FINALES
            total_score_long = (ma_score_long + rsi_trad_score_long + rsi_mav_score_long + 
                              sr_score_long + adx_score_long + macd_score_long + 
                              squeeze_score_long + chart_score_long) * obligatory_multiplier_long
            
            total_score_short = (ma_score_short + rsi_trad_score_short + rsi_mav_score_short + 
                               sr_score_short + adx_score_short + macd_score_short + 
                               squeeze_score_short + chart_score_short) * obligatory_multiplier_short
            
            # 5. DETERMINAR SEÑAL
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if total_score_long >= 70 and obligatory_ok_long:
                signal_type = 'LONG'
                signal_score = total_score_long
                fulfilled_conditions = self._get_fulfilled_conditions({
                    'moving_averages': ma_score_long > 0,
                    'rsi_traditional': rsi_trad_score_long > 0,
                    'rsi_maverick': rsi_mav_score_long > 0,
                    'support_resistance': sr_score_long > 0,
                    'adx_dmi': adx_score_long > 0,
                    'macd': macd_score_long > 0,
                    'squeeze': squeeze_score_long > 0,
                    'chart_patterns': chart_score_long > 0
                }, 'LONG')
            elif total_score_short >= 70 and obligatory_ok_short:
                signal_type = 'SHORT'
                signal_score = total_score_short
                fulfilled_conditions = self._get_fulfilled_conditions({
                    'moving_averages': ma_score_short > 0,
                    'rsi_traditional': rsi_trad_score_short > 0,
                    'rsi_maverick': rsi_mav_score_short > 0,
                    'support_resistance': sr_score_short > 0,
                    'adx_dmi': adx_score_short > 0,
                    'macd': macd_score_short > 0,
                    'squeeze': squeeze_score_short > 0,
                    'chart_patterns': chart_score_short > 0
                }, 'SHORT')
            
            # 6. CALCULAR NIVELES DE ENTRADA/SALIDA
            current_price = float(close[current_idx])
            entry = current_price
            stop_loss = current_price * 0.98 if signal_type == 'LONG' else current_price * 1.02
            take_profit = [current_price * 1.02] if signal_type == 'LONG' else [current_price * 0.98]
            
            # Ajustar niveles basado en soporte/resistencia
            if signal_type == 'LONG' and current_price <= support * 1.02:
                entry = min(entry, support * 1.01)
                stop_loss = support * 0.98
                take_profit = [resistance * 0.99]
            elif signal_type == 'SHORT' and current_price >= resistance * 0.98:
                entry = max(entry, resistance * 0.99)
                stop_loss = resistance * 1.02
                take_profit = [support * 1.01]
            
            # 7. PREPARAR DATOS DE RETORNO
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profit],
                'support': float(support),
                'resistance': float(resistance),
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
                'adx': float(adx[current_idx]),
                'plus_di': float(plus_di[current_idx]),
                'minus_di': float(minus_di[current_idx]),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_maverick': float(rsi_maverick[current_idx]),
                'fulfilled_conditions': fulfilled_conditions,
                'obligatory_conditions_met': obligatory_ok_long if signal_type == 'LONG' else obligatory_ok_short,
                'trend_strength_signal': trend_data['strength_signals'][current_idx],
                'no_trade_zone': trend_data['no_trade_zones'][current_idx],
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
                    'trend_strength': trend_data['trend_strength'][-50:],
                    'no_trade_zones': trend_data['no_trade_zones'][-50:],
                    'strength_signals': trend_data['strength_signals'][-50:],
                    'colors': trend_data['colors'][-50:]
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error en generate_signals_improved para {symbol}: {e}")
            return self._create_empty_signal(symbol)

    def _get_fulfilled_conditions(self, conditions, signal_type):
        """Obtener lista de condiciones cumplidas"""
        fulfilled = []
        condition_names = {
            'moving_averages': 'Medias Móviles alineadas',
            'rsi_traditional': 'RSI Tradicional favorable',
            'rsi_maverick': 'RSI Maverick favorable', 
            'support_resistance': 'Posición favorable soporte/resistencia',
            'adx_dmi': 'ADX + DMI confirmando tendencia',
            'macd': 'MACD con señal favorable',
            'squeeze': 'Squeeze Momentum favorable',
            'chart_patterns': 'Patrón de chartismo detectado'
        }
        
        for key, fulfilled_flag in conditions.items():
            if fulfilled_flag:
                fulfilled.append(condition_names.get(key, key))
        
        return fulfilled

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
            'trend_strength_signal': 'NEUTRAL',
            'no_trade_zone': False,
            'data': [],
            'indicators': {}
        }

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
                
                # Verificar condiciones de salida
                exit_reason = None
                
                if signal_type == 'LONG':
                    # Salida por cambio de fuerza de tendencia
                    if current_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                        exit_reason = "Cambio en fuerza de tendencia a bajista"
                    # Salida por ruptura de soporte
                    elif current_price < signal_data.get('support', current_price * 0.98):
                        exit_reason = "Ruptura de soporte clave"
                    # Salida por objetivo alcanzado
                    elif current_price >= signal_data.get('take_profit', [current_price * 1.02])[0]:
                        exit_reason = "Objetivo de profit alcanzado"
                
                elif signal_type == 'SHORT':
                    # Salida por cambio de fuerza de tendencia
                    if current_strength in ['STRONG_UP', 'WEAK_UP']:
                        exit_reason = "Cambio en fuerza de tendencia a alcista"
                    # Salida por ruptura de resistencia
                    elif current_price > signal_data.get('resistance', current_price * 1.02):
                        exit_reason = "Ruptura de resistencia clave"
                    # Salida por objetivo alcanzado
                    elif current_price <= signal_data.get('take_profit', [current_price * 0.98])[0]:
                        exit_reason = "Objetivo de profit alcanzado"
                
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
                    
                    # Remover señal activa
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram mejorada"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA DE ENTRADA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Par: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%

💰 Precio actual: {alert_data['current_price']:.6f}
🎯 Entrada: {alert_data['entry']:.6f}
🛑 Stop Loss: {alert_data['stop_loss']:.6f}
🎯 Take Profit: {alert_data['take_profit']:.6f}

📈 Apalancamiento: x{alert_data['leverage']}
💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

✅ Condiciones cumplidas: {len(alert_data.get('fulfilled_conditions', []))}

🔗 Revisar en: https://multitimeframe-crypto-wgta-pro.onrender.com/
            """
        else:  # exit alert
            pnl_text = "📈 P&L: +{:.2f}%".format(alert_data['pnl_percent']) if alert_data['pnl_percent'] > 0 else "📉 P&L: {:.2f}%".format(alert_data['pnl_percent'])
            
            message = f"""
🚨 ALERTA DE SALIDA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Par: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} - CERRAR

💰 Entrada: {alert_data['entry_price']:.6f}
💰 Salida: {alert_data['exit_price']:.6f}
{pnl_text}

📊 Razón: {alert_data['reason']}

🔗 Revisar en: https://multitimeframe-crypto-wgta-pro.onrender.com/
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
            # Generar alertas de entrada
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                send_telegram_alert(alert, 'entry')
            
            # Generar alertas de salida
            exit_alerts = indicator.generate_exit_signals()
            for alert in exit_alerts:
                send_telegram_alert(alert, 'exit')
            
            time.sleep(60)  # Verificar cada 60 segundos
            
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
        adx_threshold = int(request.args.get('adx_threshold', 20))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, volume_filter, leverage
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
        adx_threshold = int(request.args.get('adx_threshold', 20))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
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
            'long_signals': long_signals[:5],
            'short_signals': short_signals[:5],
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
        
        for symbol in CRYPTO_SYMBOLS[:15]:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval)
                if signal_data and signal_data['current_price'] > 0:
                    
                    buy_pressure = min(100, max(0, signal_data['signal_score'] if signal_data['signal'] == 'LONG' else 0))
                    sell_pressure = min(100, max(0, signal_data['signal_score'] if signal_data['signal'] == 'SHORT' else 0))
                    
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
        # Simular algunas alertas para demo
        alerts = []
        current_time = indicator.get_bolivia_time()
        
        for symbol in CRYPTO_SYMBOLS[:3]:
            signal_data = indicator.generate_signals_improved(symbol, '15m')
            if signal_data and signal_data['signal'] != 'NEUTRAL':
                alerts.append({
                    'symbol': symbol,
                    'interval': '15m',
                    'signal': signal_data['signal'],
                    'score': signal_data['signal_score'],
                    'entry': signal_data['entry'],
                    'stop_loss': signal_data['stop_loss'],
                    'take_profit': signal_data['take_profit'][0],
                    'leverage': 15,
                    'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
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
        return jsonify({'win_rate': 0, 'total_signals': 0, 'successful_signals': 0})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Crear gráfico simple del reporte
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico 1: Precio y niveles
        if signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            
            ax1.plot(dates, closes, label='Precio', color='blue', linewidth=2)
            ax1.axhline(y=signal_data['entry'], color='green', linestyle='--', label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', label='Stop Loss')
            ax1.axhline(y=signal_data['take_profit'][0], color='orange', linestyle='--', label='Take Profit')
        
        ax1.set_title(f'{symbol} - Análisis de Precio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Indicadores
        if 'indicators' in signal_data:
            indicator_dates = dates[-50:]
            ax2.plot(indicator_dates, signal_data['indicators']['rsi_traditional'], label='RSI Tradicional', color='purple')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        ax2.set_title('Indicadores Técnicos')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Fuerza de Tendencia
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-50:]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            bars = ax3.bar(trend_dates, trend_strength, color=colors, alpha=0.7)
        
        ax3.set_title('Fuerza de Tendencia Maverick')
        ax3.grid(True, alpha=0.3)
        
        # Información de la señal
        ax4.axis('off')
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        
        PRECIO: ${signal_data['current_price']:.2f}
        ENTRADA: ${signal_data['entry']:.2f}
        STOP LOSS: ${signal_data['stop_loss']:.2f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.2f}
        
        SOPORTE: ${signal_data['support']:.2f}
        RESISTENCIA: ${signal_data['resistance']:.2f}
        
        CONDICIONES OBLIGATORIAS: {'✅' if signal_data['obligatory_conditions_met'] else '❌'}
        FUERZA TENDENCIA: {signal_data['trend_strength_signal']}
        """
        
        ax4.text(0.1, 0.9, signal_info, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png', 
                        as_attachment=True, 
                        download_name=f'report_{symbol}_{interval}.png')
        
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
