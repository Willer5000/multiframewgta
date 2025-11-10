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
        """Verificar si es horario de scalping (lunes a viernes de 4am a 4pm hora boliviana)"""
        now = self.get_bolivia_time()
        if now.weekday() >= 5:  # Sábado o Domingo
            return False
        return 4 <= now.hour < 16  # De 4am a 4pm

    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela"""
        if interval == '15m':
            next_close = current_time.replace(minute=current_time.minute // 15 * 15, second=0, microsecond=0) + timedelta(minutes=15)
            return (next_close - current_time).total_seconds() <= 450  # 7.5 minutos (50%)
        elif interval == '30m':
            next_close = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0) + timedelta(minutes=30)
            return (next_close - current_time).total_seconds() <= 900  # 15 minutos (50%)
        elif interval == '1h':
            next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return (next_close - current_time).total_seconds() <= 1800  # 30 minutos (50%)
        elif interval == '2h':
            current_hour = current_time.hour
            next_2h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour % 2 == 0:
                next_2h_close += timedelta(hours=2)
            else:
                next_2h_close += timedelta(hours=1)
            return (next_2h_close - current_time).total_seconds() <= 3600  # 1 hora (50%)
        elif interval == '4h':
            current_hour = current_time.hour
            next_4h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 4
            if remainder == 0:
                next_4h_close += timedelta(hours=4)
            else:
                next_4h_close += timedelta(hours=4 - remainder)
            return (next_4h_close - current_time).total_seconds() <= 7200  # 2 horas (50%)
        elif interval == '8h':
            current_hour = current_time.hour
            next_8h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 8
            if remainder == 0:
                next_8h_close += timedelta(hours=8)
            else:
                next_8h_close += timedelta(hours=8 - remainder)
            return (next_8h_close - current_time).total_seconds() <= 14400  # 4 horas (50%)
        elif interval == '12h':
            current_hour = current_time.hour
            next_12h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour < 8:
                next_12h_close = next_12h_close.replace(hour=20)
            else:
                next_12h_close = next_12h_close.replace(hour=8) + timedelta(days=1)
            return (next_12h_close - current_time).total_seconds() <= 21600  # 6 horas (50%)
        elif interval == '1D':
            tomorrow_8pm = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
            if current_time.hour >= 20:
                tomorrow_8pm += timedelta(days=1)
            return (tomorrow_8pm - current_time).total_seconds() <= 43200  # 12 horas (50%)
        
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
                        return self._generate_empty_data(limit)
                    
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
        
        return self._generate_empty_data(limit)

    def _generate_empty_data(self, limit):
        """Generar datos vacíos"""
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1h')
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.ones(limit),
            'high': np.ones(limit),
            'low': np.ones(limit),
            'close': np.ones(limit),
            'volume': np.zeros(limit)
        })

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

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con protección anti-mechazos"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Soporte y resistencia con lookback extendido
            support_1 = np.min(low[-20:])
            resistance_1 = np.max(high[-20:])
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                entry = min(current_price, support_1 * 1.02)
                stop_loss = max(support_1 * 0.97, entry - (current_atr * 1.8))
                tp1 = resistance_1 * 0.98
                
                min_tp = entry + (2 * (entry - stop_loss))
                tp1 = max(tp1, min_tp)
                
            else:  # SHORT
                entry = max(current_price, resistance_1 * 0.98)
                stop_loss = min(resistance_1 * 1.03, entry + (current_atr * 1.8))
                tp1 = support_1 * 1.02
                
                min_tp = entry - (2 * (stop_loss - entry))
                tp1 = min(tp1, min_tp)
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp1)],
                'support': float(support_1),
                'resistance': float(resistance_1),
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support': current_price * 0.98,
                'resistance': current_price * 1.02,
                'atr': 0.0,
                'atr_percentage': 0.0
            }

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

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger"""
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

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional"""
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
        """Calcular MACD"""
        if len(prices) < slow:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_squeeze_momentum(self, high, low, close, length=20, mult=2):
        """Calcular Squeeze Momentum"""
        try:
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
            tr = self.calculate_atr(high, low, close, length)
            kc_basis = self.calculate_sma(close, length)
            kc_upper = kc_basis + (tr * mult)
            kc_lower = kc_basis - (tr * mult)
            
            # Detectar squeeze
            squeeze_on = np.zeros(n, dtype=bool)
            squeeze_off = np.zeros(n, dtype=bool)
            momentum = np.zeros(n)
            
            for i in range(n):
                if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
                    squeeze_on[i] = True
                elif bb_upper[i] > kc_upper[i] and bb_lower[i] < kc_lower[i]:
                    squeeze_off[i] = True
                
                # Momentum simple
                if i > 0:
                    momentum[i] = close[i] - close[i-1]
            
            return {
                'squeeze_on': squeeze_on.tolist(),
                'squeeze_off': squeeze_off.tolist(),
                'squeeze_momentum': momentum.tolist(),
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
                'squeeze_momentum': [0] * n,
                'bb_upper': [0] * n,
                'bb_lower': [0] * n,
                'kc_upper': [0] * n,
                'kc_lower': [0] * n
            }

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo"""
        n = len(close)
        patterns = {
            'head_shoulders': [False] * n,
            'double_top': [False] * n,
            'double_bottom': [False] * n,
            'ascending_wedge': [False] * n,
            'descending_wedge': [False] * n,
            'bullish_flag': [False] * n,
            'bearish_flag': [False] * n
        }
        
        try:
            for i in range(lookback, n-1):
                # Doble Techo
                if (i >= lookback and 
                    high[i] >= np.max(high[i-lookback:i]) * 0.98 and
                    high[i-10] >= np.max(high[i-lookback:i-10]) * 0.98 and
                    abs(high[i] - high[i-10]) / high[i] < 0.02):
                    patterns['double_top'][i] = True
                
                # Doble Fondo
                if (i >= lookback and 
                    low[i] <= np.min(low[i-lookback:i]) * 1.02 and
                    low[i-10] <= np.min(low[i-lookback:i-10]) * 1.02 and
                    abs(low[i] - low[i-10]) / low[i] < 0.02):
                    patterns['double_bottom'][i] = True
                
                # Cuña Ascendente
                if i >= 20:
                    highs_20 = high[i-20:i+1]
                    lows_20 = low[i-20:i+1]
                    if (np.polyfit(range(21), highs_20, 1)[0] > 0 and
                        np.polyfit(range(21), lows_20, 1)[0] > 0 and
                        (highs_20[-1] - highs_20[0]) / highs_20[0] > 0.05):
                        patterns['ascending_wedge'][i] = True
                
                # Cuña Descendente
                if i >= 20:
                    highs_20 = high[i-20:i+1]
                    lows_20 = low[i-20:i+1]
                    if (np.polyfit(range(21), highs_20, 1)[0] < 0 and
                        np.polyfit(range(21), lows_20, 1)[0] < 0 and
                        (lows_20[0] - lows_20[-1]) / lows_20[0] > 0.05):
                        patterns['descending_wedge'][i] = True
            
        except Exception as e:
            print(f"Error detectando patrones: {e}")
        
        return patterns

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick"""
        try:
            n = len(close)
            
            basis = self.calculate_sma(close, length)
            dev = np.zeros(n)
            
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                dev[i] = np.std(window) if len(window) > 1 else 0
            
            upper = basis + (dev * mult)
            lower = basis - (dev * mult)
            
            bb_width = np.zeros(n)
            for i in range(n):
                if basis[i] > 0:
                    bb_width[i] = ((upper[i] - lower[i]) / basis[i]) * 100
            
            trend_strength = np.zeros(n)
            for i in range(1, n):
                if bb_width[i] > bb_width[i-1]:
                    trend_strength[i] = bb_width[i]
                else:
                    trend_strength[i] = -bb_width[i]
            
            if n >= 50:
                historical_bb_width = bb_width[max(0, n-100):n]
                high_zone_threshold = np.percentile(historical_bb_width, 70)
            else:
                high_zone_threshold = np.percentile(bb_width, 70) if len(bb_width) > 0 else 5
            
            no_trade_zones = np.zeros(n, dtype=bool)
            strength_signals = ['NEUTRAL'] * n
            
            for i in range(10, n):
                if (bb_width[i] > high_zone_threshold and 
                    trend_strength[i] < 0 and 
                    bb_width[i] < np.max(bb_width[max(0, i-10):i])):
                    no_trade_zones[i] = True
                
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

    def check_multi_timeframe_trend(self, symbol, timeframe):
        """Verificar tendencia en múltiples temporalidades con OBLIGATORIEDAD"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(timeframe, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and timeframe != '15m':
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

    def check_multi_timeframe_obligatory_conditions(self, symbol, interval, signal_type):
        """Verificar condiciones OBLIGATORIAS multi-temporalidad"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False
            
            # Obtener análisis de todas las temporalidades
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            # Verificar fuerza de tendencia Maverick en todas las TF
            maverick_conditions = True
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and interval != '15m':
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 30)
                if df is not None and len(df) > 10:
                    trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                    current_signal = trend_data['strength_signals'][-1]
                    current_no_trade = trend_data['no_trade_zones'][-1]
                    
                    # Verificar que NO hay zona de NO OPERAR
                    if current_no_trade:
                        maverick_conditions = False
                        break
                    
                    # Verificar fuerza de tendencia según señal
                    if signal_type == 'LONG':
                        if current_signal not in ['STRONG_UP', 'WEAK_UP']:
                            maverick_conditions = False
                            break
                    elif signal_type == 'SHORT':
                        if current_signal not in ['STRONG_DOWN', 'WEAK_DOWN']:
                            maverick_conditions = False
                            break
            
            if not maverick_conditions:
                return False
            
            # Verificar condiciones de tendencia según OBLIGATORIEDAD
            if signal_type == 'LONG':
                # Mayor: ALCISTA o NEUTRAL
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                # Media: EXCLUSIVAMENTE ALCISTA
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'
                # Menor: Fuerza ALCISTA (ya verificado arriba)
                menor_ok = True
                
                return mayor_ok and media_ok and menor_ok
                
            elif signal_type == 'SHORT':
                # Mayor: BAJISTA o NEUTRAL
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                # Media: EXCLUSIVAMENTE BAJISTA
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'
                # Menor: Fuerza BAJISTA (ya verificado arriba)
                menor_ok = True
                
                return mayor_ok and media_ok and menor_ok
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False

    def calculate_whale_signals_improved(self, df, interval, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=20, signal_threshold=25, 
                                       sell_signal_threshold=20):
        """Implementación MEJORADA del indicador de ballenas con CONDICIONAL TEMPORAL"""
        try:
            # VERIFICACIÓN TEMPORAL: Solo señal obligatoria en 12H y 1D
            is_obligatory_tf = interval in ['12h', '1D']
            
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
                'volume_anomaly': (volume > np.mean(volume) * min_volume_multiplier).tolist(),
                'is_obligatory_tf': is_obligatory_tf
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
                'volume_anomaly': [False] * n,
                'is_obligatory_tf': interval in ['12h', '1D']
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
            # Divergencia alcista: precio hace lower low, indicador hace higher low
            price_low = price[i] < np.min(price[i-lookback:i])
            indicator_high = indicator[i] > np.max(indicator[i-lookback:i])
            
            if price_low and indicator_high:
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace higher high, indicador hace lower high
            price_high = price[i] > np.max(price[i-lookback:i])
            indicator_low = indicator[i] < np.min(indicator[i-lookback:i])
            
            if price_high and indicator_low:
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

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

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI"""
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

    def evaluate_signal_conditions_improved(self, data, current_idx, interval, adx_threshold=20):
        """Evaluar condiciones de señal con NUEVA ESTRUCTURA y OBLIGATORIEDAD"""
        # PESOS DE INDICADORES
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
        
        conditions = {
            'long': {},
            'short': {}
        }
        
        if current_idx < 0:
            current_idx = len(data['whale_pump']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['whale_pump']):
            return conditions
        
        # 1. MEDIAS MÓVILES (15%)
        ma_conditions = {
            'ma9_above': data['ma_9'][current_idx] < data['close'][current_idx] if current_idx < len(data['ma_9']) else False,
            'ma21_above': data['ma_21'][current_idx] < data['close'][current_idx] if current_idx < len(data['ma_21']) else False,
            'ma50_above': data['ma_50'][current_idx] < data['close'][current_idx] if current_idx < len(data['ma_50']) else False,
            'ma200_above': data['ma_200'][current_idx] < data['close'][current_idx] if current_idx < len(data['ma_200']) else False
        }
        
        ma_long_score = sum([1 for condition in ma_conditions.values() if condition]) / len(ma_conditions) * weights['moving_averages']
        ma_short_score = sum([1 for condition in ma_conditions.values() if not condition]) / len(ma_conditions) * weights['moving_averages']
        
        # 2. RSI TRADICIONAL + DIVERGENCIAS (15%)
        rsi_trad = data['rsi_traditional'][current_idx] if current_idx < len(data['rsi_traditional']) else 50
        rsi_conditions = {
            'oversold': rsi_trad < 30,
            'overbought': rsi_trad > 70,
            'bullish_div': data['rsi_bullish_div'][current_idx] if current_idx < len(data['rsi_bullish_div']) else False,
            'bearish_div': data['rsi_bearish_div'][current_idx] if current_idx < len(data['rsi_bearish_div']) else False
        }
        
        rsi_long_score = (rsi_conditions['oversold'] or rsi_conditions['bullish_div']) * weights['rsi_traditional']
        rsi_short_score = (rsi_conditions['overbought'] or rsi_conditions['bearish_div']) * weights['rsi_traditional']
        
        # 3. RSI MAVERICK + DIVERGENCIAS (15%)
        rsi_mav = data['rsi_maverick'][current_idx] if current_idx < len(data['rsi_maverick']) else 0.5
        rsi_mav_conditions = {
            'oversold': rsi_mav < 0.2,
            'overbought': rsi_mav > 0.8,
            'bullish_div': data['rsi_maverick_bullish_div'][current_idx] if current_idx < len(data['rsi_maverick_bullish_div']) else False,
            'bearish_div': data['rsi_maverick_bearish_div'][current_idx] if current_idx < len(data['rsi_maverick_bearish_div']) else False
        }
        
        rsi_mav_long_score = (rsi_mav_conditions['oversold'] or rsi_mav_conditions['bullish_div']) * weights['rsi_maverick']
        rsi_mav_short_score = (rsi_mav_conditions['overbought'] or rsi_mav_conditions['bearish_div']) * weights['rsi_maverick']
        
        # 4. SOPORTES Y RESISTENCIAS SMART MONEY (20%)
        current_price = data['close'][current_idx]
        support = data['support'][current_idx] if current_idx < len(data['support']) else 0
        resistance = data['resistance'][current_idx] if current_idx < len(data['resistance']) else 0
        
        sr_conditions = {
            'near_support': current_price <= support * 1.02,
            'near_resistance': current_price >= resistance * 0.98,
            'above_support': current_price > support,
            'below_resistance': current_price < resistance
        }
        
        sr_long_score = (sr_conditions['near_support'] and sr_conditions['above_support']) * weights['support_resistance']
        sr_short_score = (sr_conditions['near_resistance'] and sr_conditions['below_resistance']) * weights['support_resistance']
        
        # 5. ADX + DMI (10%)
        adx_val = data['adx'][current_idx] if current_idx < len(data['adx']) else 0
        plus_di_val = data['plus_di'][current_idx] if current_idx < len(data['plus_di']) else 0
        minus_di_val = data['minus_di'][current_idx] if current_idx < len(data['minus_di']) else 0
        
        adx_conditions = {
            'strong_trend': adx_val > adx_threshold,
            'plus_di_above': plus_di_val > minus_di_val,
            'minus_di_above': minus_di_val > plus_di_val
        }
        
        adx_long_score = (adx_conditions['strong_trend'] and adx_conditions['plus_di_above']) * weights['adx_dmi']
        adx_short_score = (adx_conditions['strong_trend'] and adx_conditions['minus_di_above']) * weights['adx_dmi']
        
        # 6. MACD (10%)
        macd_hist = data['macd_histogram'][current_idx] if current_idx < len(data['macd_histogram']) else 0
        macd_conditions = {
            'histogram_positive': macd_hist > 0,
            'histogram_negative': macd_hist < 0,
            'macd_above_signal': data['macd'][current_idx] > data['macd_signal'][current_idx] if current_idx < len(data['macd']) else False
        }
        
        macd_long_score = (macd_conditions['histogram_positive'] or macd_conditions['macd_above_signal']) * weights['macd']
        macd_short_score = (macd_conditions['histogram_negative'] or not macd_conditions['macd_above_signal']) * weights['macd']
        
        # 7. SQUEEZE MOMENTUM (10%)
        squeeze_momentum = data['squeeze_momentum'][current_idx] if current_idx < len(data['squeeze_momentum']) else 0
        squeeze_conditions = {
            'momentum_positive': squeeze_momentum > 0,
            'momentum_negative': squeeze_momentum < 0,
            'squeeze_off': data['squeeze_off'][current_idx] if current_idx < len(data['squeeze_off']) else False
        }
        
        squeeze_long_score = (squeeze_conditions['momentum_positive'] and squeeze_conditions['squeeze_off']) * weights['squeeze_momentum']
        squeeze_short_score = (squeeze_conditions['momentum_negative'] and squeeze_conditions['squeeze_off']) * weights['squeeze_momentum']
        
        # 8. BANDAS DE BOLLINGER (5%)
        bb_conditions = {
            'near_upper': current_price >= data['bb_upper'][current_idx] * 0.98 if current_idx < len(data['bb_upper']) else False,
            'near_lower': current_price <= data['bb_lower'][current_idx] * 1.02 if current_idx < len(data['bb_lower']) else False
        }
        
        bb_long_score = bb_conditions['near_lower'] * weights['bollinger_bands']
        bb_short_score = bb_conditions['near_upper'] * weights['bollinger_bands']
        
        # 9. PATRONES DE CHARTISMO (15%)
        chart_conditions = {
            'bullish_patterns': any([
                data['double_bottom'][current_idx] if current_idx < len(data['double_bottom']) else False,
                data['ascending_wedge'][current_idx] if current_idx < len(data['ascending_wedge']) else False,
                data['bullish_flag'][current_idx] if current_idx < len(data['bullish_flag']) else False
            ]),
            'bearish_patterns': any([
                data['double_top'][current_idx] if current_idx < len(data['double_top']) else False,
                data['descending_wedge'][current_idx] if current_idx < len(data['descending_wedge']) else False,
                data['bearish_flag'][current_idx] if current_idx < len(data['bearish_flag']) else False
            ])
        }
        
        chart_long_score = chart_conditions['bullish_patterns'] * weights['chart_patterns']
        chart_short_score = chart_conditions['bearish_patterns'] * weights['chart_patterns']
        
        # CALCULAR SCORES FINALES
        total_long_score = (
            ma_long_score + rsi_long_score + rsi_mav_long_score + sr_long_score +
            adx_long_score + macd_long_score + squeeze_long_score + bb_long_score + chart_long_score
        )
        
        total_short_score = (
            ma_short_score + rsi_short_score + rsi_mav_short_score + sr_short_score +
            adx_short_score + macd_short_score + squeeze_short_score + bb_short_score + chart_short_score
        )
        
        # CONDICIONES OBLIGATORIAS
        obligatory_conditions_met = {
            'long': data.get('multi_tf_obligatory_long', False),
            'short': data.get('multi_tf_obligatory_short', False)
        }
        
        # APLICAR MULTIPLICADOR DE OBLIGATORIEDAD
        final_long_score = total_long_score if obligatory_conditions_met['long'] else 0
        final_short_score = total_short_score if obligatory_conditions_met['short'] else 0
        
        conditions['long'] = {
            'score': final_long_score,
            'obligatory_met': obligatory_conditions_met['long'],
            'components': {
                'moving_averages': ma_long_score,
                'rsi_traditional': rsi_long_score,
                'rsi_maverick': rsi_mav_long_score,
                'support_resistance': sr_long_score,
                'adx_dmi': adx_long_score,
                'macd': macd_long_score,
                'squeeze_momentum': squeeze_long_score,
                'bollinger_bands': bb_long_score,
                'chart_patterns': chart_long_score
            }
        }
        
        conditions['short'] = {
            'score': final_short_score,
            'obligatory_met': obligatory_conditions_met['short'],
            'components': {
                'moving_averages': ma_short_score,
                'rsi_traditional': rsi_short_score,
                'rsi_maverick': rsi_mav_short_score,
                'support_resistance': sr_short_score,
                'adx_dmi': adx_short_score,
                'macd': macd_short_score,
                'squeeze_momentum': squeeze_short_score,
                'bollinger_bands': bb_short_score,
                'chart_patterns': chart_short_score
            }
        }
        
        return conditions

    def calculate_win_rate(self, symbol, interval, lookback_periods=200):
        """Calcular winrate basado en datos históricos"""
        try:
            cache_key = f"winrate_{symbol}_{interval}"
            if cache_key in self.win_rate_data:
                cached_data, timestamp = self.win_rate_data[cache_key]
                if (datetime.now() - timestamp).seconds < 3600:  # Cache 1 hora
                    return cached_data
            
            df = self.get_kucoin_data(symbol, interval, lookback_periods + 50)
            if df is None or len(df) < lookback_periods:
                return {'win_rate': 0, 'total_signals': 0, 'successful_signals': 0}
            
            signals = []
            successful_trades = 0
            
            for i in range(20, len(df) - 5):  # Dejar espacio para verificar resultado
                # Simular señal en punto histórico
                current_data = df.iloc[:i+1]
                if len(current_data) < 30:
                    continue
                
                # Generar señal para punto histórico
                signal_data = self._generate_historical_signal(current_data, interval)
                
                if signal_data['signal'] in ['LONG', 'SHORT'] and signal_data['signal_score'] >= 65:
                    # Verificar resultado en las próximas 5 velas
                    future_data = df.iloc[i+1:i+6]
                    if len(future_data) > 0:
                        entry_price = signal_data['entry']
                        exit_price = future_data['close'].iloc[-1]
                        
                        if signal_data['signal'] == 'LONG':
                            profit = (exit_price - entry_price) / entry_price * 100
                            successful = profit > 1.0  # 1% de ganancia
                        else:
                            profit = (entry_price - exit_price) / entry_price * 100
                            successful = profit > 1.0
                        
                        signals.append({
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'profit': profit,
                            'successful': successful
                        })
                        
                        if successful:
                            successful_trades += 1
            
            total_signals = len(signals)
            win_rate = (successful_trades / total_signals * 100) if total_signals > 0 else 0
            
            result = {
                'win_rate': round(win_rate, 1),
                'total_signals': total_signals,
                'successful_signals': successful_trades,
                'average_profit': np.mean([s['profit'] for s in signals]) if signals else 0
            }
            
            self.win_rate_data[cache_key] = (result, datetime.now())
            return result
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol} {interval}: {e}")
            return {'win_rate': 0, 'total_signals': 0, 'successful_signals': 0}

    def _generate_historical_signal(self, df, interval):
        """Generar señal para datos históricos"""
        try:
            # Implementación simplificada para cálculo histórico
            close = df['close'].values
            current_price = close[-1]
            
            # Señal básica para cálculo de winrate
            ma_20 = self.calculate_sma(close, 20)
            ma_50 = self.calculate_sma(close, 50)
            
            if len(ma_20) > 0 and len(ma_50) > 0:
                if ma_20[-1] > ma_50[-1]:
                    signal = 'LONG'
                    score = 70
                else:
                    signal = 'SHORT'
                    score = 70
            else:
                signal = 'NEUTRAL'
                score = 0
            
            return {
                'signal': signal,
                'signal_score': score,
                'entry': current_price
            }
            
        except Exception as e:
            return {'signal': 'NEUTRAL', 'signal_score': 0, 'entry': 0}

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
                current_no_trade = current_trend['no_trade_zones'][-1]
                
                # Razones para salir
                exit_reason = None
                
                # 1. Salida por pérdida de fuerza
                if signal_type == 'LONG' and current_strength in ['WEAK_UP', 'STRONG_DOWN', 'WEAK_DOWN']:
                    exit_reason = "Fuerza de tendencia desfavorable"
                elif signal_type == 'SHORT' and current_strength in ['WEAK_DOWN', 'STRONG_UP', 'WEAK_UP']:
                    exit_reason = "Fuerza de tendencia desfavorable"
                
                # 2. Salida por zona no operar
                elif current_no_trade:
                    exit_reason = "Zona de NO OPERAR activa"
                
                # 3. Salida por cambio en temporalidad menor
                else:
                    hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                    if hierarchy.get('menor'):
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
                        'trend_strength': current_strength,
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    exit_alerts.append(exit_alert)
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=20, 
                                sr_period=50, rsi_length=20, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - CON ESTRATEGIA MULTI-TEMPORALIDAD"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # INDICADORES PRINCIPALES
            whale_data = self.calculate_whale_signals_improved(df, interval, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi_maverick = self.calculate_rsi_maverick(close, rsi_length, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, 14)
            
            # DETECCIÓN DE DIVERGENCIAS
            rsi_maverick_bullish_div, rsi_maverick_bearish_div = self.detect_divergence(close, rsi_maverick)
            rsi_traditional_bullish_div, rsi_traditional_bearish_div = self.detect_divergence(close, rsi_traditional)
            
            # MEDIAS MÓVILES
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # BANDAS DE BOLLINGER
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # SQUEEZE MOMENTUM
            squeeze_data = self.calculate_squeeze_momentum(high, low, close)
            
            # PATRONES DE CHARTISMO
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # FUERZA DE TENDENCIA MAVERICK
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            current_price = close[current_idx]
            
            # VERIFICAR CONDICIONES OBLIGATORIAS MULTI-TIMEFRAME
            obligatory_long = self.check_multi_timeframe_obligatory_conditions(symbol, interval, 'LONG')
            obligatory_short = self.check_multi_timeframe_obligatory_conditions(symbol, interval, 'SHORT')
            
            # PREPARAR DATOS PARA EVALUACIÓN
            analysis_data = {
                'close': close,
                'high': high,
                'low': low,
                'volume': volume,
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'di_cross_bullish': di_cross_bullish,
                'di_cross_bearish': di_cross_bearish,
                'di_trend_bullish': di_trend_bullish,
                'di_trend_bearish': di_trend_bearish,
                'rsi_maverick': rsi_maverick,
                'rsi_traditional': rsi_traditional,
                'rsi_maverick_bullish_div': rsi_maverick_bullish_div,
                'rsi_maverick_bearish_div': rsi_maverick_bearish_div,
                'rsi_traditional_bullish_div': rsi_traditional_bullish_div,
                'rsi_traditional_bearish_div': rsi_traditional_bearish_div,
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'squeeze_on': squeeze_data['squeeze_on'],
                'squeeze_off': squeeze_data['squeeze_off'],
                'squeeze_momentum': squeeze_data['squeeze_momentum'],
                'support': whale_data['support'],
                'resistance': whale_data['resistance'],
                'volume_anomaly': whale_data['volume_anomaly'],
                'double_top': chart_patterns['double_top'],
                'double_bottom': chart_patterns['double_bottom'],
                'ascending_wedge': chart_patterns['ascending_wedge'],
                'descending_wedge': chart_patterns['descending_wedge'],
                'bullish_flag': chart_patterns['bullish_flag'],
                'bearish_flag': chart_patterns['bearish_flag'],
                'multi_tf_obligatory_long': obligatory_long,
                'multi_tf_obligatory_short': obligatory_short
            }
            
            # EVALUAR CONDICIONES
            conditions = self.evaluate_signal_conditions_improved(analysis_data, current_idx, interval, adx_threshold)
            
            long_score = conditions['long']['score']
            short_score = conditions['short']['score']
            
            # DETERMINAR SEÑAL FINAL
            signal_type = 'NEUTRAL'
            signal_score = 0
            
            if long_score >= 65 and obligatory_long:
                signal_type = 'LONG'
                signal_score = long_score
            elif short_score >= 65 and obligatory_short:
                signal_type = 'SHORT'
                signal_score = short_score
            
            # CALCULAR NIVELES DE ENTRADA/SALIDA
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # REGISTRAR SEÑAL ACTIVA SI ES VÁLIDA
            if signal_type in ['LONG', 'SHORT']:
                signal_key = f"{symbol}_{interval}_{signal_type}"
                self.active_signals[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': levels_data['entry'],
                    'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                    'score': signal_score
                }
            
            # PREPARAR DATOS DE RETORNO
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
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
                'adx': float(adx[current_idx] if current_idx < len(adx) else 0),
                'plus_di': float(plus_di[current_idx] if current_idx < len(plus_di) else 0),
                'minus_di': float(minus_di[current_idx] if current_idx < len(minus_di) else 0),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'rsi_traditional': float(rsi_traditional[current_idx] if current_idx < len(rsi_traditional) else 50),
                'multi_tf_obligatory_long': obligatory_long,
                'multi_tf_obligatory_short': obligatory_short,
                'trend_strength_signal': trend_strength_data['strength_signals'][current_idx] if current_idx < len(trend_strength_data['strength_signals']) else 'NEUTRAL',
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:].tolist(),
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['squeeze_momentum'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
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
            'rsi_traditional': 50,
            'multi_tf_obligatory_long': False,
            'multi_tf_obligatory_short': False,
            'trend_strength_signal': 'NEUTRAL',
            'no_trade_zone': False,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de scalping"""
        alerts = []
        telegram_intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D']
        
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
                        signal_data['signal_score'] >= 65):
                        
                        risk_category = next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                        
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
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'trend_strength': signal_data.get('trend_strength_signal', 'NEUTRAL'),
                            'multi_tf_obligatory': signal_data.get('multi_tf_obligatory_long', False) or signal_data.get('multi_tf_obligatory_short', False)
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

💰 Precio actual: {alert_data['current_price']:.6f}
💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

🎯 ENTRADA: ${alert_data['entry']:.6f}
🛑 STOP LOSS: ${alert_data['stop_loss']:.6f}
🎯 TAKE PROFIT: ${alert_data['take_profit']:.6f}

📈 Apalancamiento: x{alert_data['leverage']}
✅ Multi-TF: {'CONFIRMADO' if alert_data.get('multi_tf_obligatory', False) else 'PENDIENTE'}

📊 Revisa la señal en: https://multiframewgta.onrender.com/
            """
        else:
            pnl_text = f"📊 P&L: {alert_data['pnl_percent']:+.2f}%"
            message = f"""
🚨 ALERTA DE SALIDA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} - CERRAR POSICIÓN

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
{pnl_text}

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
            # Generar alertas de entrada
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                send_telegram_alert(alert, 'entry')
            
            # Generar alertas de salida
            exit_alerts = indicator.generate_exit_signals()
            for alert in exit_alerts:
                send_telegram_alert(alert, 'exit')
            
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

@app.route('/api/multiple_signals')
def get_multiple_signals():
    """Endpoint para obtener múltiples señales"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 20))
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
                
                if signal_data and signal_data['signal'] != 'NEUTRAL' and signal_data['signal_score'] >= 65:
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
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 20))
        
        scatter_data = []
        
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:3])
        
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
    """Generar reporte técnico"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Implementación básica del reporte
        fig = plt.figure(figsize=(12, 10))
        
        # Gráfico simple de precio
        ax1 = plt.subplot(2, 1, 1)
        if signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            ax1.plot(dates, closes, 'b-', linewidth=2, label='Precio')
            
            ax1.axhline(y=signal_data['entry'], color='green', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal_data['take_profit'][0], color='orange', linestyle='--', alpha=0.7, label='Take Profit')
        
        ax1.set_title(f'Reporte {symbol} - {interval}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Información de la señal
        ax2 = plt.subplot(2, 1, 2)
        ax2.axis('off')
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        
        MULTI-TF OBLIGATORIO: {'✅' if signal_data.get('multi_tf_obligatory_long', False) or signal_data.get('multi_tf_obligatory_short', False) else '❌'}
        FUERZA TENDENCIA: {signal_data.get('trend_strength_signal', 'NEUTRAL')}
        
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
        APALANCAMIENTO: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        """
        
        ax2.text(0.1, 0.9, signal_info, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
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
        'timezone': 'America/La_Paz',
        'is_scalping_time': indicator.is_scalping_time()
    })

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
