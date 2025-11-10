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

# Configuración Telegram - ACTUALIZADA
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
            # Para 3 días, verificar cada 3 días
            days_since_epoch = (current_time - datetime(1970, 1, 1)).days
            return days_since_epoch % 3 == 0 and current_time.hour >= 18
        elif interval == '1W':
            # Para 1 semana, verificar los lunes
            return current_time.weekday() == 0 and current_time.hour >= 18
        
        return False

    def get_kucoin_data(self, symbol, interval, limit=100):
        """Obtener datos reales de KuCoin"""
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
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if not candles:
                        return self._generate_fallback_data(symbol, interval, limit)
                    
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
        
        return self._generate_fallback_data(symbol, interval, limit)

    def _generate_fallback_data(self, symbol, interval, limit):
        """Generar datos de fallback basados en precios reales aproximados"""
        try:
            # Precios reales aproximados de referencia
            price_reference = {
                'BTC-USDT': 45000,
                'ETH-USDT': 3000,
                'BNB-USDT': 600,
                'SOL-USDT': 100,
                'XRP-USDT': 0.6,
                'ADA-USDT': 0.5,
                'AVAX-USDT': 40,
                'DOT-USDT': 7,
                'LINK-USDT': 15,
                'MATIC-USDT': 0.8
            }
            
            base_price = price_reference.get(symbol, 100)
            
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
            np.random.seed(hash(symbol) % 10000)
            
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
            
        except Exception as e:
            print(f"Error generando datos de fallback: {e}")
            # Último recurso: datos básicos
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
            return pd.DataFrame({
                'timestamp': dates,
                'open': np.ones(limit) * 100,
                'high': np.ones(limit) * 101,
                'low': np.ones(limit) * 99,
                'close': np.ones(limit) * 100,
                'volume': np.ones(limit) * 1000
            })

    # INDICADORES TÉCNICOS (CÁLCULOS MANUALES)
    
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
        if len(prices) < period:
            return np.zeros(len(prices))
        
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
        """Calcular Squeeze Momentum (LazyBear)"""
        try:
            n = len(close)
            
            # Calcular Bandas de Bollinger
            bb_basis = self.calculate_sma(close, length)
            bb_dev = np.zeros(n)
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                bb_dev[i] = np.std(window) if len(window) > 1 else 0
            bb_upper = bb_basis + (bb_dev * mult)
            bb_lower = bb_basis - (bb_dev * mult)
            
            # Calcular Keltner Channel
            tr = self.calculate_atr(high, low, close, length)
            kc_basis = self.calculate_sma(close, length)
            kc_upper = kc_basis + (tr * mult)
            kc_lower = kc_basis - (tr * mult)
            
            # Detectar squeeze
            squeeze_on = np.zeros(n, dtype=bool)
            squeeze_off = np.zeros(n, dtype=bool)
            for i in range(n):
                if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
                    squeeze_on[i] = True
                elif bb_upper[i] > kc_upper[i] and bb_lower[i] < kc_lower[i]:
                    squeeze_off[i] = True
            
            # Calcular momentum
            momentum = np.zeros(n)
            for i in range(1, n):
                if close[i] > close[i-1]:
                    momentum[i] = 1
                elif close[i] < close[i-1]:
                    momentum[i] = -1
                else:
                    momentum[i] = momentum[i-1]
            
            return {
                'squeeze_on': squeeze_on.tolist(),
                'squeeze_off': squeeze_off.tolist(),
                'momentum': momentum.tolist(),
                'bb_upper': bb_upper.tolist(),
                'bb_lower': bb_lower.tolist(),
                'kc_upper': kc_upper.tolist(),
                'kc_lower': kc_lower.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_squeeze_momentum: {e}")
            n = len(close)
            return {
                'squeeze_on': [False] * n,
                'squeeze_off': [False] * n,
                'momentum': [0] * n,
                'bb_upper': [0] * n,
                'bb_lower': [0] * n,
                'kc_upper': [0] * n,
                'kc_lower': [0] * n
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
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-7):
            # Doble Techo (Double Top)
            if (high[i] > high[i-1] and high[i] > high[i-2] and 
                high[i] > high[i+1] and high[i] > high[i+2] and
                abs(high[i] - high[i-5]) < high[i] * 0.02):
                patterns['double_top'][i] = True
            
            # Doble Fondo (Double Bottom)
            if (low[i] < low[i-1] and low[i] < low[i-2] and 
                low[i] < low[i+1] and low[i] < low[i+2] and
                abs(low[i] - low[i-5]) < low[i] * 0.02):
                patterns['double_bottom'][i] = True
            
            # Cuña Ascendente (Rising Wedge)
            if (high[i] > high[i-1] and high[i-1] > high[i-2] and
                low[i] > low[i-1] and low[i-1] > low[i-2] and
                (high[i] - low[i]) < (high[i-5] - low[i-5]) * 0.8):
                patterns['rising_wedge'][i] = True
            
            # Cuña Descendente (Falling Wedge)
            if (high[i] < high[i-1] and high[i-1] < high[i-2] and
                low[i] < low[i-1] and low[i-1] < low[i-2] and
                (high[i] - low[i]) < (high[i-5] - low[i-5]) * 0.8):
                patterns['falling_wedge'][i] = True
        
        return patterns

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

    def check_multi_timeframe_trend(self, symbol, interval):
        """Verificar tendencia en múltiples temporalidades - MEJORADO"""
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
        """Verificar condiciones obligatorias - NUEVO SISTEMA"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False, "No hay jerarquía temporal definida"
            
            # Verificar todas las temporalidades
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            # Obtener datos de fuerza de tendencia para cada TF
            trend_strength_checks = {}
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and interval != '15m':
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 30)
                if df is not None and len(df) > 10:
                    trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                    trend_strength_checks[tf_type] = {
                        'signal': trend_data['strength_signals'][-1],
                        'no_trade': trend_data['no_trade_zones'][-1]
                    }
                else:
                    trend_strength_checks[tf_type] = {
                        'signal': 'NEUTRAL',
                        'no_trade': False
                    }
            
            # VERIFICACIONES OBLIGATORIAS PARA LONG
            if signal_type == 'LONG':
                # 1. Tendencia Mayor: ALCISTA o NEUTRAL
                if tf_analysis.get('mayor') not in ['BULLISH', 'NEUTRAL']:
                    return False, "Tendencia Mayor no es alcista/neutral"
                
                # 2. Tendencia Media: EXCLUSIVAMENTE ALCISTA
                if tf_analysis.get('media') != 'BULLISH':
                    return False, "Tendencia Media no es exclusivamente alcista"
                
                # 3. Tendencia Menor: Fuerza ALCISTA
                menor_strength = trend_strength_checks.get('menor', {}).get('signal', 'NEUTRAL')
                if menor_strength not in ['STRONG_UP', 'WEAK_UP']:
                    return False, "Fuerza de tendencia menor no es alcista"
                
                # 4. Sin zonas NO OPERAR en ninguna TF
                for tf_type, check in trend_strength_checks.items():
                    if check.get('no_trade', False):
                        return False, f"Zona NO OPERAR activa en {tf_type}"
            
            # VERIFICACIONES OBLIGATORIAS PARA SHORT
            elif signal_type == 'SHORT':
                # 1. Tendencia Mayor: BAJISTA o NEUTRAL
                if tf_analysis.get('mayor') not in ['BEARISH', 'NEUTRAL']:
                    return False, "Tendencia Mayor no es bajista/neutral"
                
                # 2. Tendencia Media: EXCLUSIVAMENTE BAJISTA
                if tf_analysis.get('media') != 'BEARISH':
                    return False, "Tendencia Media no es exclusivamente bajista"
                
                # 3. Tendencia Menor: Fuerza BAJISTA
                menor_strength = trend_strength_checks.get('menor', {}).get('signal', 'NEUTRAL')
                if menor_strength not in ['STRONG_DOWN', 'WEAK_DOWN']:
                    return False, "Fuerza de tendencia menor no es bajista"
                
                # 4. Sin zonas NO OPERAR en ninguna TF
                for tf_type, check in trend_strength_checks.items():
                    if check.get('no_trade', False):
                        return False, f"Zona NO OPERAR activa en {tf_type}"
            
            return True, "Todas las condiciones obligatorias cumplidas"
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False, f"Error en verificación: {str(e)}"

    def calculate_whale_signals_improved(self, df, interval, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=20, signal_threshold=25, 
                                       sell_signal_threshold=20):
        """Implementación MEJORADA del indicador de ballenas con temporalidad específica"""
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
            
            # AJUSTAR SENSIBILIDAD POR TEMPORALIDAD
            if interval in ['12h', '1D']:
                # Mayor sensibilidad para temporalidades largas
                sensitivity = 2.0
                signal_threshold = 20
                sell_signal_threshold = 18
            else:
                # Menor sensibilidad para temporalidades cortas
                sensitivity = 1.5
                signal_threshold = 25
                sell_signal_threshold = 22
            
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

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias entre precio e indicador"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Divergencia Alcista: Precio hace lower low, indicador hace higher low
            if (price[i] < np.min(price[i-lookback:i]) and
                indicator[i] > np.min(indicator[i-lookback:i]) and
                price[i] < price[i-1] and indicator[i] > indicator[i-1]):
                bullish_div[i] = True
            
            # Divergencia Bajista: Precio hace higher high, indicador hace lower high
            if (price[i] > np.max(price[i-lookback:i]) and
                indicator[i] < np.max(indicator[i-lookback:i]) and
                price[i] > price[i-1] and indicator[i] < indicator[i-1]):
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

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

    def calculate_smart_money_levels(self, high, low, close, lookback=50):
        """Calcular niveles de soporte y resistencia Smart Money"""
        n = len(close)
        support_levels = np.zeros(n)
        resistance_levels = np.zeros(n)
        
        for i in range(lookback, n):
            # Soporte: mínimos significativos
            support_candidates = []
            for j in range(i-lookback, i):
                if low[j] == np.min(low[max(i-lookback, 0):j+1]):
                    support_candidates.append(low[j])
            
            if support_candidates:
                support_levels[i] = np.mean(support_candidates[-3:]) if len(support_candidates) >= 3 else support_candidates[-1]
            
            # Resistencia: máximos significativos
            resistance_candidates = []
            for j in range(i-lookback, i):
                if high[j] == np.max(high[max(i-lookback, 0):j+1]):
                    resistance_candidates.append(high[j])
            
            if resistance_candidates:
                resistance_levels[i] = np.mean(resistance_candidates[-3:]) if len(resistance_candidates) >= 3 else resistance_candidates[-1]
        
        return support_levels.tolist(), resistance_levels.tolist()

    def evaluate_indicators(self, data, current_idx, interval):
        """Evaluar todos los indicadores con pesos específicos"""
        indicators = {
            'moving_averages': {'value': 0, 'weight': 15, 'description': 'Medias Móviles Alineadas'},
            'rsi_traditional': {'value': 0, 'weight': 15, 'description': 'RSI Tradicional Favorable'},
            'rsi_maverick': {'value': 0, 'weight': 15, 'description': 'RSI Maverick Favorable'},
            'smart_money': {'value': 0, 'weight': 20, 'description': 'Niveles Smart Money'},
            'adx_dmi': {'value': 0, 'weight': 10, 'description': 'ADX + DMI Favorable'},
            'macd': {'value': 0, 'weight': 10, 'description': 'MACD Favorable'},
            'squeeze': {'value': 0, 'weight': 10, 'description': 'Squeeze Momentum Favorable'},
            'bollinger': {'value': 0, 'weight': 5, 'description': 'Bandas Bollinger Favorable'},
            'chart_patterns': {'value': 0, 'weight': 15, 'description': 'Patrón Chartismo Detectado'}
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return indicators
        
        current_price = data['close'][current_idx]
        
        # 1. Medias Móviles (15%)
        ma_9 = self.calculate_sma(data['close'], 9)
        ma_21 = self.calculate_sma(data['close'], 21)
        ma_50 = self.calculate_sma(data['close'], 50)
        ma_200 = self.calculate_sma(data['close'], 200)
        
        if (current_price > ma_9[current_idx] and 
            ma_9[current_idx] > ma_21[current_idx] and 
            ma_21[current_idx] > ma_50[current_idx]):
            indicators['moving_averages']['value'] = 1
        
        # 2. RSI Tradicional (15%)
        rsi = self.calculate_rsi(data['close'])
        if current_idx < len(rsi):
            if (data.get('signal_type') == 'LONG' and rsi[current_idx] < 70 and rsi[current_idx] > 30) or \
               (data.get('signal_type') == 'SHORT' and rsi[current_idx] > 30 and rsi[current_idx] < 70):
                indicators['rsi_traditional']['value'] = 1
        
        # 3. RSI Maverick (15%)
        rsi_maverick = self.calculate_rsi_maverick(data['close'])
        if current_idx < len(rsi_maverick):
            if (data.get('signal_type') == 'LONG' and rsi_maverick[current_idx] < 0.8 and rsi_maverick[current_idx] > 0.2) or \
               (data.get('signal_type') == 'SHORT' and rsi_maverick[current_idx] > 0.2 and rsi_maverick[current_idx] < 0.8):
                indicators['rsi_maverick']['value'] = 1
        
        # 4. Smart Money Levels (20%)
        support, resistance = self.calculate_smart_money_levels(data['high'], data['low'], data['close'])
        if current_idx < len(support) and current_idx < len(resistance):
            if (data.get('signal_type') == 'LONG' and current_price <= support[current_idx] * 1.02) or \
               (data.get('signal_type') == 'SHORT' and current_price >= resistance[current_idx] * 0.98):
                indicators['smart_money']['value'] = 1
        
        # 5. ADX + DMI (10%)
        adx, plus_di, minus_di = self.calculate_adx(data['high'], data['low'], data['close'])
        if current_idx < len(adx) and current_idx < len(plus_di) and current_idx < len(minus_di):
            if adx[current_idx] > 25:
                if (data.get('signal_type') == 'LONG' and plus_di[current_idx] > minus_di[current_idx]) or \
                   (data.get('signal_type') == 'SHORT' and minus_di[current_idx] > plus_di[current_idx]):
                    indicators['adx_dmi']['value'] = 1
        
        # 6. MACD (10%)
        macd, signal, histogram = self.calculate_macd(data['close'])
        if current_idx < len(macd) and current_idx < len(signal):
            if (data.get('signal_type') == 'LONG' and macd[current_idx] > signal[current_idx] and histogram[current_idx] > 0) or \
               (data.get('signal_type') == 'SHORT' and macd[current_idx] < signal[current_idx] and histogram[current_idx] < 0):
                indicators['macd']['value'] = 1
        
        # 7. Squeeze Momentum (10%)
        squeeze_data = self.calculate_squeeze_momentum(data['high'], data['low'], data['close'])
        if current_idx < len(squeeze_data['momentum']):
            if (data.get('signal_type') == 'LONG' and squeeze_data['momentum'][current_idx] > 0 and squeeze_data['squeeze_off'][current_idx]) or \
               (data.get('signal_type') == 'SHORT' and squeeze_data['momentum'][current_idx] < 0 and squeeze_data['squeeze_off'][current_idx]):
                indicators['squeeze']['value'] = 1
        
        # 8. Bollinger Bands (5%)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['close'])
        if current_idx < len(bb_upper) and current_idx < len(bb_lower):
            bb_position = (current_price - bb_lower[current_idx]) / (bb_upper[current_idx] - bb_lower[current_idx])
            if (data.get('signal_type') == 'LONG' and bb_position < 0.8) or \
               (data.get('signal_type') == 'SHORT' and bb_position > 0.2):
                indicators['bollinger']['value'] = 1
        
        # 9. Chart Patterns (15%)
        patterns = self.detect_chart_patterns(data['high'], data['low'], data['close'])
        pattern_detected = False
        for pattern_name, pattern_data in patterns.items():
            if current_idx < len(pattern_data) and pattern_data[current_idx]:
                if (data.get('signal_type') == 'LONG' and pattern_name in ['double_bottom', 'falling_wedge', 'bullish_flag']) or \
                   (data.get('signal_type') == 'SHORT' and pattern_name in ['double_top', 'rising_wedge', 'bearish_flag']):
                    pattern_detected = True
                    break
        
        if pattern_detected:
            indicators['chart_patterns']['value'] = 1
        
        return indicators

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con gestión Smart Money"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular niveles Smart Money
            support_levels, resistance_levels = self.calculate_smart_money_levels(high, low, close)
            current_support = support_levels[-1] if support_levels else np.min(low[-20:])
            current_resistance = resistance_levels[-1] if resistance_levels else np.max(high[-20:])
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Entrada lo más cerca posible al soporte
                entry = min(current_price, current_support * 1.01)
                # Stop loss por debajo del soporte con margen ATR
                stop_loss = max(current_support * 0.98, entry - (current_atr * 1.5))
                # Take profit en resistencia
                take_profit = current_resistance * 0.99
                
            else:  # SHORT
                # Entrada lo más cerca posible a la resistencia
                entry = max(current_price, current_resistance * 0.99)
                # Stop loss por encima de la resistencia con margen ATR
                stop_loss = min(current_resistance * 1.02, entry + (current_atr * 1.5))
                # Take profit en soporte
                take_profit = current_support * 1.01
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(take_profit)],
                'support': float(current_support),
                'resistance': float(current_resistance),
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support': float(np.min(df['low'].values[-14:])),
                'resistance': float(np.max(df['high'].values[-14:])),
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_signal_score(self, indicators, obligatory_met):
        """Calcular puntuación de señal con sistema de obligatoriedad"""
        if not obligatory_met:
            return 0, []
        
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        for indicator_name, indicator_data in indicators.items():
            total_weight += indicator_data['weight']
            if indicator_data['value']:
                achieved_weight += indicator_data['weight']
                fulfilled_conditions.append(indicator_data['description'])
        
        if total_weight == 0:
            return 0, []
        
        score = (achieved_weight / total_weight) * 100
        return min(score, 100), fulfilled_conditions

    def calculate_win_rate(self, symbol, interval, lookback=200):
        """Calcular winrate histórico"""
        try:
            cache_key = f"winrate_{symbol}_{interval}"
            if cache_key in self.winrate_data:
                cached_data, timestamp = self.winrate_data[cache_key]
                if (datetime.now() - timestamp).seconds < 300:  # Cache 5 minutos
                    return cached_data
            
            df = self.get_kucoin_data(symbol, interval, lookback + 20)
            if df is None or len(df) < 50:
                return {'win_rate': 50, 'total_signals': 0, 'successful_signals': 0}
            
            signals = []
            successful_trades = 0
            total_trades = 0
            
            # Simular señales históricas
            for i in range(20, len(df) - 5):
                try:
                    # Datos para backtesting
                    test_data = df.iloc[:i+1]
                    if len(test_data) < 30:
                        continue
                    
                    # Generar señal simulada
                    signal_data = self.generate_signals_improved(symbol, interval, test_data=test_data)
                    
                    if signal_data['signal'] in ['LONG', 'SHORT'] and signal_data['signal_score'] >= 70:
                        entry_price = signal_data['entry']
                        exit_price = df['close'].iloc[i + 5]  # Salida después de 5 velas
                        
                        if signal_data['signal'] == 'LONG':
                            pnl = (exit_price - entry_price) / entry_price * 100
                            successful = pnl > 0
                        else:
                            pnl = (entry_price - exit_price) / entry_price * 100
                            successful = pnl > 0
                        
                        total_trades += 1
                        if successful:
                            successful_trades += 1
                            
                except Exception as e:
                    continue
            
            if total_trades > 0:
                win_rate = (successful_trades / total_trades) * 100
            else:
                win_rate = 50
            
            result = {
                'win_rate': round(win_rate, 1),
                'total_signals': total_trades,
                'successful_signals': successful_trades
            }
            
            self.winrate_data[cache_key] = (result, datetime.now())
            return result
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return {'win_rate': 50, 'total_signals': 0, 'successful_signals': 0}

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=20, 
                                sr_period=50, rsi_length=20, bb_multiplier=2.0, volume_filter='Todos', 
                                leverage=15, test_data=None):
        """GENERACIÓN DE SEÑALES MEJORADA - NUEVO SISTEMA ESTRATÉGICO"""
        try:
            if test_data is not None:
                df = test_data
            else:
                df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 30:
                return self._create_empty_signal(symbol)
            
            # Obtener datos actuales
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            current_idx = -1
            current_price = float(close[current_idx])
            
            # CALCULAR TODOS LOS INDICADORES
            whale_data = self.calculate_whale_signals_improved(df, interval, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            rsi_maverick = self.calculate_rsi_maverick(close, rsi_length, bb_multiplier)
            bullish_div, bearish_div = self.detect_divergence(close, rsi_maverick)
            squeeze_data = self.calculate_squeeze_momentum(high, low, close)
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            patterns = self.detect_chart_patterns(high, low, close)
            
            # EVALUAR SEÑALES POTENCIALES
            long_potential = False
            short_potential = False
            
            # Señal LONG potencial
            if (whale_data['whale_pump'][current_idx] > 15 or 
                di_cross_bullish[current_idx] or 
                bullish_div[current_idx] or
                squeeze_data['momentum'][current_idx] > 0):
                long_potential = True
            
            # Señal SHORT potencial  
            if (whale_data['whale_dump'][current_idx] > 18 or
                di_cross_bearish[current_idx] or
                bearish_div[current_idx] or
                squeeze_data['momentum'][current_idx] < 0):
                short_potential = True
            
            # EVALUAR CADA SEÑAL POTENCIAL CON CONDICIONES OBLIGATORIAS
            final_signal = 'NEUTRAL'
            final_score = 0
            fulfilled_conditions = []
            win_rate_data = {'win_rate': 50, 'total_signals': 0, 'successful_signals': 0}
            
            if long_potential:
                # Verificar condiciones obligatorias para LONG
                obligatory_met, obligatory_msg = self.check_obligatory_conditions(symbol, interval, 'LONG')
                
                if obligatory_met:
                    # Evaluar indicadores para LONG
                    indicator_data = {
                        'close': close, 'high': high, 'low': low, 'volume': volume,
                        'signal_type': 'LONG'
                    }
                    indicators = self.evaluate_indicators(indicator_data, current_idx, interval)
                    score, conditions = self.calculate_signal_score(indicators, True)
                    
                    if score >= 70:
                        final_signal = 'LONG'
                        final_score = score
                        fulfilled_conditions = conditions
                        fulfilled_conditions.append(obligatory_msg)
                        # Calcular winrate
                        win_rate_data = self.calculate_win_rate(symbol, interval)
            
            if short_potential and final_signal == 'NEUTRAL':
                # Verificar condiciones obligatorias para SHORT
                obligatory_met, obligatory_msg = self.check_obligatory_conditions(symbol, interval, 'SHORT')
                
                if obligatory_met:
                    # Evaluar indicadores para SHORT
                    indicator_data = {
                        'close': close, 'high': high, 'low': low, 'volume': volume,
                        'signal_type': 'SHORT'
                    }
                    indicators = self.evaluate_indicators(indicator_data, current_idx, interval)
                    score, conditions = self.calculate_signal_score(indicators, True)
                    
                    if score >= 70:
                        final_signal = 'SHORT'
                        final_score = score
                        fulfilled_conditions = conditions
                        fulfilled_conditions.append(obligatory_msg)
                        # Calcular winrate
                        win_rate_data = self.calculate_win_rate(symbol, interval)
            
            # Calcular niveles de entrada/salida
            levels_data = self.calculate_optimal_entry_exit(df, final_signal, leverage)
            
            # Calcular indicadores adicionales para el frontend
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            rsi = self.calculate_rsi(close)
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Preparar datos para el response
            signal_data = {
                'symbol': symbol,
                'current_price': current_price,
                'signal': final_signal,
                'signal_score': float(final_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'liquidation_long': float(levels_data['entry'] - (levels_data['entry'] / leverage)),
                'liquidation_short': float(levels_data['entry'] + (levels_data['entry'] / leverage)),
                'support': levels_data['support'],
                'resistance': levels_data['resistance'],
                'atr': levels_data['atr'],
                'atr_percentage': levels_data['atr_percentage'],
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
                'adx': float(adx[current_idx] if current_idx < len(adx) else 0),
                'plus_di': float(plus_di[current_idx] if current_idx < len(plus_di) else 0),
                'minus_di': float(minus_di[current_idx] if current_idx < len(minus_di) else 0),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'fulfilled_conditions': fulfilled_conditions,
                'trend_strength_signal': trend_strength_data['strength_signals'][current_idx] if current_idx < len(trend_strength_data['strength_signals']) else 'NEUTRAL',
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False,
                'win_rate': win_rate_data['win_rate'],
                'total_signals': win_rate_data['total_signals'],
                'successful_signals': win_rate_data['successful_signals'],
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'di_cross_bullish': di_cross_bullish[-50:],
                    'di_cross_bearish': di_cross_bearish[-50:],
                    'rsi_maverick': rsi_maverick[-50:],
                    'bullish_divergence': bullish_div[-50:],
                    'bearish_divergence': bearish_div[-50:],
                    'support': whale_data['support'][-50:],
                    'resistance': whale_data['resistance'][-50:],
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'rsi': rsi[-50:].tolist(),
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['momentum'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
                    'colors': trend_strength_data['colors'][-50:],
                    'chart_patterns': {
                        'head_shoulders': patterns['head_shoulders'][-50:],
                        'double_top': patterns['double_top'][-50:],
                        'double_bottom': patterns['double_bottom'][-50:],
                        'rising_wedge': patterns['rising_wedge'][-50:],
                        'falling_wedge': patterns['falling_wedge'][-50:],
                        'bullish_flag': patterns['bullish_flag'][-50:],
                        'bearish_flag': patterns['bearish_flag'][-50:]
                    }
                }
            }
            
            return signal_data
            
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
            'win_rate': 50,
            'total_signals': 0,
            'successful_signals': 0,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de trading mejoradas"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        # Temporalidades para monitoreo
        monitor_intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D']
        
        for interval in monitor_intervals:
            # Verificar si es momento de enviar alerta
            if not self.calculate_remaining_time(interval, current_time):
                continue
                
            for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 70):
                        
                        risk_category = next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                        
                        alert = {
                            'symbol': symbol,
                            'interval': interval,
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'entry': signal_data['entry'],
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'][0],
                            'leverage': 15,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'win_rate': signal_data.get('win_rate', 50),
                            'trend_strength': signal_data.get('trend_strength_signal', 'NEUTRAL')
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
                
                if signal_type == 'LONG':
                    # Salir si la fuerza de tendencia cambia a bajista
                    if current_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                        exit_reason = "Fuerza de tendencia cambió a bajista"
                    # Salir si se alcanza el objetivo de profit
                    elif current_price >= signal_data.get('take_profit', entry_price * 1.02):
                        exit_reason = "Objetivo de profit alcanzado"
                
                elif signal_type == 'SHORT':
                    # Salir si la fuerza de tendencia cambia a alcista
                    if current_strength in ['STRONG_UP', 'WEAK_UP']:
                        exit_reason = "Fuerza de tendencia cambió a alcista"
                    # Salir si se alcanza el objetivo de profit
                    elif current_price <= signal_data.get('take_profit', entry_price * 0.98):
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
                        'trend_strength': current_strength,
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

# Funciones auxiliares para Telegram y background tasks
def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram mejorada"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA DE TRADING - MULTI-TIMEFRAME CRYPTO PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
🎯 Winrate: {alert_data.get('win_rate', 50):.1f}%

💰 Precio actual: {alert_data['current_price']:.6f}
🎯 Entrada: ${alert_data['entry']:.6f}
🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
🎯 Take Profit: ${alert_data['take_profit']:.6f}

💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}
📊 Apalancamiento: x{alert_data['leverage']}

✅ Condiciones Cumplidas:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])][:3])}

🔔 Revisa la señal en el sistema completo.
            """
        else:  # exit alert
            message = f"""
🚨 ALERTA DE SALIDA - MULTI-TIMEFRAME CRYPTO PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} - CERRAR

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
📊 P&L: {alert_data['pnl_percent']:+.2f}%

💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}
📊 Razón: {alert_data['reason']}
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
            
            time.sleep(60)  # Verificar cada minuto
            
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

# ROUTES DEL SISTEMA
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint principal para obtener señales"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 20))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 20))
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

@app.route('/api/win_rate')
def get_win_rate():
    """Endpoint para obtener winrate específico"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        win_rate_data = indicator.calculate_win_rate(symbol, interval)
        return jsonify(win_rate_data)
        
    except Exception as e:
        print(f"Error en /api/win_rate: {e}")
        return jsonify({'win_rate': 50, 'total_signals': 0, 'successful_signals': 0})

@app.route('/api/multiple_signals')
def get_multiple_signals():
    """Endpoint para obtener múltiples señales"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 20))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
            try:
                signal_data = indicator.generate_signals_improved(
                    symbol, interval, di_period, adx_threshold
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
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 20))
        
        scatter_data = []
        
        symbols_to_analyze = CRYPTO_SYMBOLS[:15]  # Limitar para performance
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval, di_period, adx_threshold)
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
    """Endpoint para clasificación de riesgo"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    """Endpoint para alertas de trading"""
    try:
        alerts = indicator.generate_scalping_alerts()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/exit_signals')
def get_exit_signals():
    """Endpoint para señales de salida"""
    try:
        exit_alerts = indicator.generate_exit_signals()
        return jsonify({'exit_signals': exit_alerts})
        
    except Exception as e:
        print(f"Error en /api/exit_signals: {e}")
        return jsonify({'exit_signals': []})

@app.route('/api/bolivia_time')
def get_bolivia_time():
    """Endpoint para hora de Bolivia"""
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d'),
        'timezone': 'America/La_Paz',
        'is_scalping_time': indicator.is_scalping_time()
    })

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Crear gráfico del reporte (simplificado para este ejemplo)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Gráfico 1: Precio y niveles
        if signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            
            axes[0].plot(dates, closes, 'b-', linewidth=1, label='Precio')
            axes[0].axhline(y=signal_data['entry'], color='green', linestyle='--', alpha=0.7, label='Entrada')
            axes[0].axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            axes[0].axhline(y=signal_data['take_profit'][0], color='orange', linestyle='--', alpha=0.7, label='Take Profit')
        
        axes[0].set_title(f'Reporte {symbol} - {interval}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Indicadores
        if 'indicators' in signal_data:
            indicator_dates = dates[-len(signal_data['indicators']['rsi']):]
            axes[1].plot(indicator_dates, signal_data['indicators']['rsi'], 'purple', linewidth=1, label='RSI')
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        
        axes[1].set_ylabel('RSI')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Información de la señal
        axes[2].axis('off')
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        WINRATE: {signal_data.get('win_rate', 50):.1f}%
        
        PRECIO: ${signal_data['current_price']:.6f}
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
        CONDICIONES:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
        """
        
        axes[2].text(0.1, 0.9, signal_info, transform=axes[2].transAxes, fontsize=10,
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
