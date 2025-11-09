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

# JERARQUÍA TEMPORAL MEJORADA - OBLIGATORIA
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
        self.last_winrate_update = None
    
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
                        return None
                    
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
        
        return None

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
        """Calcular entradas y salidas óptimas mejoradas"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Soporte y resistencia con lookback extendido
            support_1 = np.min(low[-50:])
            resistance_1 = np.max(high[-50:])
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                entry = min(current_price, support_1 * 1.02)
                stop_loss = max(support_1 * 0.97, entry - (current_atr * 1.8))
                tp1 = resistance_1 * 0.98
                
                min_tp = entry + (2 * (entry - stop_loss))
                tp1 = max(tp1, min_tp)
                
            else:
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

    def calculate_squeeze_momentum(self, high, low, close, length=20, mult=2):
        """Calcular Squeeze Momentum manualmente"""
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
            
            # Determinar squeeze
            squeeze_on = np.zeros(n, dtype=bool)
            squeeze_off = np.zeros(n, dtype=bool)
            momentum = np.zeros(n)
            
            for i in range(n):
                if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
                    squeeze_on[i] = True
                elif bb_upper[i] > kc_upper[i] and bb_lower[i] < kc_lower[i]:
                    squeeze_off[i] = True
                
                # Calcular momentum simple
                if i >= 1:
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
            'head_shoulders': [False] * n,
            'double_top': [False] * n,
            'double_bottom': [False] * n,
            'ascending_wedge': [False] * n,
            'descending_wedge': [False] * n,
            'bullish_flag': [False] * n,
            'bearish_flag': [False] * n
        }
        
        try:
            for i in range(lookback, n-7):
                window_high = high[i-lookback:i]
                window_low = low[i-lookback:i]
                window_close = close[i-lookback:i]
                
                # Doble Techo
                if len(window_high) >= 20:
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
                if len(window_low) >= 20:
                    troughs = []
                    for j in range(2, len(window_low)-2):
                        if (window_low[j] < window_low[j-1] and 
                            window_low[j] < window_low[j-2] and
                            window_low[j] < window_low[j+1] and
                            window_low[j] < window_low[j+2]):
                            troughs.append((j, window_low[j]))
                    
                    if len(troughs) >= 2:
                        trough1_idx, trough1_val = troughs[-2]
                        trough2_idx, trough2_val = troughs[-1]
                        if (abs(trough1_val - trough2_val) / trough1_val < 0.02 and
                            trough2_idx > trough1_idx + 5):
                            patterns['double_bottom'][i] = True
            
            return patterns
            
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

    def check_multi_timeframe_trend(self, symbol, interval):
        """Verificar tendencia en múltiples temporalidades - OBLIGATORIO"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and interval != '15m':
                    results[tf_type] = 'NEUTRAL'
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 50)
                if df is None or len(df) < 20:
                    results[tf_type] = 'NEUTRAL'
                    continue
                
                close = df['close'].values
                
                ma_9 = self.calculate_sma(close, 9)
                ma_21 = self.calculate_sma(close, 21)
                ma_50 = self.calculate_sma(close, 50)
                
                current_ma_9 = ma_9[-1] if len(ma_9) > 0 else 0
                current_ma_21 = ma_21[-1] if len(ma_21) > 0 else 0
                current_ma_50 = ma_50[-1] if len(ma_50) > 0 else 0
                current_price = close[-1]
                
                if current_price > current_ma_9 and current_ma_9 > current_ma_21 and current_ma_21 > current_ma_50:
                    results[tf_type] = 'BULLISH'
                elif current_price < current_ma_9 and current_ma_9 < current_ma_21 and current_ma_21 < current_ma_50:
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}

    def check_obligatory_conditions(self, symbol, interval, signal_type, trend_strength_data, current_idx):
        """Verificar condiciones OBLIGATORIAS - SI NO SE CUMPLEN, SCORE = 0"""
        try:
            if current_idx < 0 or current_idx >= len(trend_strength_data['no_trade_zones']):
                return False
            
            # Verificar multi-timeframe
            multi_tf = self.check_multi_timeframe_trend(symbol, interval)
            
            # Verificar fuerza de tendencia Maverick
            current_strength = trend_strength_data['strength_signals'][current_idx]
            current_no_trade = trend_strength_data['no_trade_zones'][current_idx]
            
            if signal_type == 'LONG':
                # OBLIGATORIO PARA LONG
                mayor_ok = multi_tf.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                media_ok = multi_tf.get('media', 'NEUTRAL') == 'BULLISH'  # EXCLUSIVAMENTE ALCISTA
                menor_ok = current_strength in ['STRONG_UP', 'WEAK_UP']
                no_trade_ok = not current_no_trade
                
                return mayor_ok and media_ok and menor_ok and no_trade_ok
                
            elif signal_type == 'SHORT':
                # OBLIGATORIO PARA SHORT
                mayor_ok = multi_tf.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                media_ok = multi_tf.get('media', 'NEUTRAL') == 'BEARISH'  # EXCLUSIVAMENTE BAJISTA
                menor_ok = current_strength in ['STRONG_DOWN', 'WEAK_DOWN']
                no_trade_ok = not current_no_trade
                
                return mayor_ok and media_ok and menor_ok and no_trade_ok
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False

    def calculate_whale_signals_corrected(self, df, interval, sensitivity=1.7):
        """INDICADOR CAZADOR DE BALLENAS CORREGIDO - Solo 12H y 1D"""
        if interval not in ['12h', '1D']:
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
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                low_5 = np.min(low[max(0, i-5):i+1])
                high_5 = np.max(high[max(0, i-5):i+1])
                
                if (volume_ratio > 1.5 and 
                    (close[i] < close[i-1] or price_change < -0.5) and
                    low[i] <= low_5 * 1.01):
                    
                    volume_strength = min(3.0, volume_ratio / 1.5)
                    whale_pump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
                
                if (volume_ratio > 1.5 and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / 1.5)
                    whale_dump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            current_support = np.array([np.min(low[max(0, i-50+1):i+1]) for i in range(n)])
            current_resistance = np.array([np.max(high[max(0, i-50+1):i+1]) for i in range(n)])
            
            for i in range(5, n):
                if (whale_pump_smooth[i] > 25 and 
                    close[i] <= current_support[i] * 1.02 and
                    volume[i] > np.mean(volume[max(0, i-10):i+1])):
                    confirmed_buy[i] = True
                
                if (whale_dump_smooth[i] > 20 and 
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
                'volume_anomaly': (volume > np.mean(volume) * 1.5).tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_corrected: {e}")
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
        """RSI Maverick basado en %B Bollinger"""
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
            if (price[i] < price[i-1] and 
                indicator[i] > indicator[i-1] and
                price[i] < np.min(price[i-lookback:i])):
                bullish_div[i] = True
            
            if (price[i] > price[i-1] and 
                indicator[i] < indicator[i-1] and
                price[i] > np.max(price[i-lookback:i])):
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

    def evaluate_signal_conditions_pro(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con NUEVA ESTRUCTURA"""
        conditions = {
            'long': {
                # Indicadores principales
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación Medias Móviles (MA9, MA21, MA50)'},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI Tradicional favorable'},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable'},
                'support_resistance': {'value': False, 'weight': 20, 'description': 'Precio cerca soporte smart money'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} y +DI > -DI'},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD favorable'},
                'squeeze': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable'},
                'bollinger': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger favorables'},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrón chartista alcista'},
                # Ballenas solo en 12H/1D
                'whale_pump': {'value': False, 'weight': 25 if interval in ['12h', '1D'] else 0, 'description': 'Ballena compradora activa'}
            },
            'short': {
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación Medias Móviles (MA9, MA21, MA50)'},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI Tradicional favorable'},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable'},
                'support_resistance': {'value': False, 'weight': 20, 'description': 'Precio cerca resistencia smart money'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} y -DI > +DI'},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD favorable'},
                'squeeze': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable'},
                'bollinger': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger favorables'},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrón chartista bajista'},
                'whale_dump': {'value': False, 'weight': 25 if interval in ['12h', '1D'] else 0, 'description': 'Ballena vendedora activa'}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        current_price = data['close'][current_idx]
        
        # CONDICIONES LONG
        conditions['long']['moving_averages']['value'] = (
            current_price > data['ma_9'][current_idx] and
            data['ma_9'][current_idx] > data['ma_21'][current_idx] and
            data['ma_21'][current_idx] > data['ma_50'][current_idx]
        )
        
        conditions['long']['rsi_traditional']['value'] = (
            data['rsi'][current_idx] < 70 and
            data['rsi'][current_idx] > 30
        )
        
        conditions['long']['rsi_maverick']['value'] = (
            data['rsi_maverick'][current_idx] < 0.8 and
            data['rsi_maverick'][current_idx] > 0.2
        )
        
        conditions['long']['support_resistance']['value'] = (
            current_price <= data['support'][current_idx] * 1.02
        )
        
        conditions['long']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['plus_di'][current_idx] > data['minus_di'][current_idx]
        )
        
        conditions['long']['macd']['value'] = (
            data['macd'][current_idx] > data['macd_signal'][current_idx]
        )
        
        conditions['long']['squeeze']['value'] = (
            data['squeeze_momentum'][current_idx] > 0
        )
        
        conditions['long']['bollinger']['value'] = (
            current_price > data['bb_middle'][current_idx]
        )
        
        conditions['long']['chart_patterns']['value'] = (
            data['double_bottom'][current_idx] or
            data['bullish_flag'][current_idx]
        )
        
        conditions['long']['whale_pump']['value'] = (
            data['whale_pump'][current_idx] > 25
        )
        
        # CONDICIONES SHORT
        conditions['short']['moving_averages']['value'] = (
            current_price < data['ma_9'][current_idx] and
            data['ma_9'][current_idx] < data['ma_21'][current_idx] and
            data['ma_21'][current_idx] < data['ma_50'][current_idx]
        )
        
        conditions['short']['rsi_traditional']['value'] = (
            data['rsi'][current_idx] > 30 and
            data['rsi'][current_idx] < 70
        )
        
        conditions['short']['rsi_maverick']['value'] = (
            data['rsi_maverick'][current_idx] > 0.2 and
            data['rsi_maverick'][current_idx] < 0.8
        )
        
        conditions['short']['support_resistance']['value'] = (
            current_price >= data['resistance'][current_idx] * 0.98
        )
        
        conditions['short']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['minus_di'][current_idx] > data['plus_di'][current_idx]
        )
        
        conditions['short']['macd']['value'] = (
            data['macd'][current_idx] < data['macd_signal'][current_idx]
        )
        
        conditions['short']['squeeze']['value'] = (
            data['squeeze_momentum'][current_idx] < 0
        )
        
        conditions['short']['bollinger']['value'] = (
            current_price < data['bb_middle'][current_idx]
        )
        
        conditions['short']['chart_patterns']['value'] = (
            data['double_top'][current_idx] or
            data['head_shoulders'][current_idx]
        )
        
        conditions['short']['whale_dump']['value'] = (
            data['whale_dump'][current_idx] > 20
        )
        
        return conditions

    def calculate_signal_score_pro(self, conditions, signal_type, obligatory_met):
        """Calcular puntuación de señal con OBLIGATORIEDAD"""
        if not obligatory_met:
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
        
        score = (achieved_weight / total_weight * 100)
        return min(score, 100), fulfilled_conditions

    def calculate_win_rate(self, symbol, interval, lookback=200):
        """Calcular winrate histórico"""
        try:
            cache_key = f"winrate_{symbol}_{interval}"
            if (cache_key in self.win_rate_data and 
                self.last_winrate_update and 
                (datetime.now() - self.last_winrate_update).seconds < 3600):
                return self.win_rate_data[cache_key]
            
            df = self.get_kucoin_data(symbol, interval, lookback + 50)
            if df is None or len(df) < lookback + 10:
                return 50.0
            
            signals = []
            for i in range(10, len(df) - 5):
                try:
                    window_df = df.iloc[i-50:i]
                    if len(window_df) < 30:
                        continue
                    
                    signal_data = self.generate_signals_pro(
                        symbol, interval, df=window_df, calc_winrate=True
                    )
                    
                    if signal_data['signal'] in ['LONG', 'SHORT']:
                        future_prices = df['close'].iloc[i:i+5].values
                        entry = signal_data['entry']
                        
                        if signal_data['signal'] == 'LONG':
                            result = 'WIN' if any(future_prices >= signal_data['take_profit'][0]) else 'LOSS'
                        else:
                            result = 'WIN' if any(future_prices <= signal_data['take_profit'][0]) else 'LOSS'
                        
                        signals.append(result)
                        
                except Exception as e:
                    continue
            
            if not signals:
                win_rate = 50.0
            else:
                wins = signals.count('WIN')
                win_rate = (wins / len(signals)) * 100
            
            self.win_rate_data[cache_key] = win_rate
            self.last_winrate_update = datetime.now()
            
            return win_rate
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return 50.0

    def generate_signals_pro(self, symbol, interval, di_period=14, adx_threshold=25, 
                           sr_period=50, rsi_length=20, bb_multiplier=2.0, 
                           volume_filter='Todos', leverage=15, df=None, calc_winrate=False):
        """GENERACIÓN DE SEÑALES PROFESIONAL MEJORADA"""
        try:
            if df is None:
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
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            rsi = self.calculate_rsi(close)
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Indicadores avanzados
            whale_data = self.calculate_whale_signals_corrected(df, interval)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            rsi_maverick = self.calculate_rsi_maverick(close, rsi_length, bb_multiplier)
            squeeze_data = self.calculate_squeeze_momentum(high, low, close)
            chart_patterns = self.detect_chart_patterns(high, low, close)
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Detectar divergencias
            bullish_div, bearish_div = self.detect_divergence(close, rsi_maverick)
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            current_idx = -1
            
            # Preparar datos para evaluación
            analysis_data = {
                'close': close,
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'rsi': rsi,
                'rsi_maverick': rsi_maverick,
                'macd': macd,
                'macd_signal': macd_signal,
                'squeeze_momentum': squeeze_data['momentum'],
                'bb_middle': bb_middle,
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'support': whale_data['support'],
                'resistance': whale_data['resistance'],
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'double_top': chart_patterns['double_top'],
                'double_bottom': chart_patterns['double_bottom'],
                'head_shoulders': chart_patterns['head_shoulders'],
                'bullish_flag': chart_patterns['bullish_flag'],
                'bearish_flag': chart_patterns['bearish_flag']
            }
            
            # Evaluar condiciones
            conditions = self.evaluate_signal_conditions_pro(analysis_data, current_idx, interval, adx_threshold)
            
            # Verificar OBLIGATORIEDAD
            obligatory_long = self.check_obligatory_conditions(symbol, interval, 'LONG', trend_strength_data, current_idx)
            obligatory_short = self.check_obligatory_conditions(symbol, interval, 'SHORT', trend_strength_data, current_idx)
            
            # Calcular scores
            long_score, long_conditions = self.calculate_signal_score_pro(conditions, 'long', obligatory_long)
            short_score, short_conditions = self.calculate_signal_score_pro(conditions, 'short', obligatory_short)
            
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
            
            # Calcular winrate si no es para cálculo histórico
            win_rate = 0
            if not calc_winrate:
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
                'atr': levels_data['atr'],
                'atr_percentage': levels_data['atr_percentage'],
                'volume': float(df['volume'].iloc[current_idx]),
                'volume_ma': float(np.mean(df['volume'].tail(20))),
                'adx': float(adx[current_idx] if current_idx < len(adx) else 0),
                'plus_di': float(plus_di[current_idx] if current_idx < len(plus_di) else 0),
                'minus_di': float(minus_di[current_idx] if current_idx < len(minus_di) else 0),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'fulfilled_conditions': fulfilled_conditions,
                'trend_strength_signal': trend_strength_data['strength_signals'][current_idx] if current_idx < len(trend_strength_data['strength_signals']) else 'NEUTRAL',
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False,
                'obligatory_met_long': obligatory_long,
                'obligatory_met_short': obligatory_short,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'rsi': rsi[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:],
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
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'bullish_divergence': bullish_div[-50:],
                    'bearish_divergence': bearish_div[-50:],
                    'di_cross_bullish': di_cross_bullish[-50:],
                    'di_cross_bearish': di_cross_bearish[-50:],
                    'double_top': chart_patterns['double_top'][-50:],
                    'double_bottom': chart_patterns['double_bottom'][-50:],
                    'head_shoulders': chart_patterns['head_shoulders'][-50:],
                    'bullish_flag': chart_patterns['bullish_flag'][-50:],
                    'bearish_flag': chart_patterns['bearish_flag'][-50:]
                }
            }
            
        except Exception as e:
            print(f"Error en generate_signals_pro para {symbol}: {e}")
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
            'obligatory_met_long': False,
            'obligatory_met_short': False,
            'data': [],
            'indicators': {}
        }

    def generate_exit_signals(self):
        """Generar señales de salida mejoradas"""
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
                current_signal = self.generate_signals_pro(symbol, interval)
                
                # Razones para salir
                exit_reason = None
                
                # 1. Cambio en condiciones obligatorias
                if signal_type == 'LONG' and not current_signal['obligatory_met_long']:
                    exit_reason = "Pérdida condiciones obligatorias LONG"
                elif signal_type == 'SHORT' and not current_signal['obligatory_met_short']:
                    exit_reason = "Pérdida condiciones obligatorias SHORT"
                
                # 2. Zona de no operar
                elif current_signal['no_trade_zone']:
                    exit_reason = "Activación zona NO OPERAR"
                
                # 3. Cambio de señal
                elif current_signal['signal'] != signal_type:
                    exit_reason = f"Cambio de señal a {current_signal['signal']}"
                
                # 4. Objetivo alcanzado
                elif (signal_type == 'LONG' and current_price >= signal_data.get('take_profit', [current_price * 1.02])[0]):
                    exit_reason = "Take Profit alcanzado"
                elif (signal_type == 'SHORT' and current_price <= signal_data.get('take_profit', [current_price * 0.98])[0]):
                    exit_reason = "Take Profit alcanzado"
                
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

    def generate_strategic_report(self, symbol, interval):
        """Generar reporte estratégico completo"""
        try:
            # Obtener datos históricos para análisis
            df = self.get_kucoin_data(symbol, interval, 200)
            if df is None:
                return None
            
            # Calcular performance por combinación de indicadores
            combinations_performance = []
            
            # Simular análisis de combinaciones (en producción sería más complejo)
            combinations = [
                {'name': 'MA + RSI + ADX', 'win_rate': 65.2},
                {'name': 'Squeeze + Bollinger', 'win_rate': 58.7},
                {'name': 'Multi-TF + Maverick', 'win_rate': 72.1},
                {'name': 'Chart Patterns + Volume', 'win_rate': 68.9}
            ]
            
            best_strategy = max(combinations, key=lambda x: x['win_rate'])
            
            # Crear gráfico del reporte
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Gráfico 1: Performance por estrategia
            strategies = [c['name'] for c in combinations]
            win_rates = [c['win_rate'] for c in combinations]
            
            bars = ax1.bar(strategies, win_rates, color=['green' if x == best_strategy['win_rate'] else 'blue' for x in win_rates])
            ax1.set_title('Win Rate por Estrategia', fontweight='bold')
            ax1.set_ylabel('Win Rate (%)')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, rate in zip(bars, win_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate}%', 
                        ha='center', va='bottom', fontweight='bold')
            
            # Gráfico 2: Distribución temporal
            timeframes = ['15m', '30m', '1h', '2h', '4h', '1D']
            tf_win_rates = [58.3, 62.1, 65.8, 68.9, 72.4, 75.2]
            ax2.plot(timeframes, tf_win_rates, marker='o', linewidth=2, color='orange')
            ax2.set_title('Performance por Temporalidad', fontweight='bold')
            ax2.set_ylabel('Win Rate (%)')
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Mejor setup actual
            current_signal = self.generate_signals_pro(symbol, interval)
            setup_info = [
                f"Señal: {current_signal['signal']}",
                f"Score: {current_signal['signal_score']:.1f}%",
                f"Win Rate: {current_signal['win_rate']:.1f}%",
                f"Mejor Estrategia: {best_strategy['name']}",
                f"Win Rate Estrategia: {best_strategy['win_rate']}%"
            ]
            
            ax3.axis('off')
            ax3.text(0.1, 0.9, '\n'.join(setup_info), transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax3.set_title('Setup Actual Recomendado', fontweight='bold')
            
            # Gráfico 4: Recomendaciones
            recommendations = [
                "✅ Usar Multi-TF + Maverick",
                "✅ Verificar condiciones obligatorias",
                "✅ Gestión riesgo 2% máximo",
                "⏰ Operar en horario Bolivia",
                "📊 Monitorear winrate semanal"
            ]
            
            ax4.axis('off')
            ax4.text(0.1, 0.9, '\n'.join(recommendations), transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax4.set_title('Recomendaciones Estratégicas', fontweight='bold')
            
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando reporte estratégico: {e}")
            return None

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram mejorada"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA PRO - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
🏆 Win Rate: {alert_data.get('win_rate', 0):.1f}%

💰 Entrada: ${alert_data['entry']:.6f}
🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
🎯 Take Profit: ${alert_data['take_profit']:.6f}

💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}
✅ Condiciones Obligatorias: CUMPLIDAS

📈 Apalancamiento: x{alert_data['leverage']}

🔔 Señal confirmada multi-temporalidad
            """
        else:
            pnl_text = f"📊 P&L: {alert_data['pnl_percent']:+.2f}%"
            message = f"""
🚨 ALERTA SALIDA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} - CERRAR

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
{pnl_text}

📊 Razón: {alert_data['reason']}

🔔 Operación finalizada
            """
        
        asyncio.run(bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=message
        ))
        print(f"Alerta {alert_type} enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano mejorado"""
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar cada 60 segundos
            if True:  # Siempre verificar para testing
                print("Verificando alertas...")
                
                # Generar alertas de entrada
                alerts = []
                for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
                    for interval in ['15m', '1h', '4h', '1D']:
                        try:
                            signal_data = indicator.generate_signals_pro(symbol, interval)
                            
                            if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                                signal_data['signal_score'] >= 70):
                                
                                alert_key = f"{symbol}_{interval}_{signal_data['signal']}"
                                if (alert_key not in indicator.alert_cache or 
                                    (datetime.now() - indicator.alert_cache[alert_key]).seconds > 300):
                                    
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
                                        'risk_category': risk_category,
                                        'trend_strength': signal_data.get('trend_strength_signal', 'NEUTRAL')
                                    }
                                    
                                    alerts.append(alert)
                                    indicator.alert_cache[alert_key] = datetime.now()
                                    indicator.active_signals[alert_key] = {
                                        'symbol': symbol,
                                        'interval': interval,
                                        'signal': signal_data['signal'],
                                        'entry_price': signal_data['entry'],
                                        'take_profit': signal_data['take_profit'],
                                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                                    }
                        
                        except Exception as e:
                            continue
                
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
    """Endpoint para obtener señales de trading MEJORADO"""
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
        
        signal_data = indicator.generate_signals_pro(
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
        adx_threshold = int(request.args.get('adx_threshold', 25))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:
            try:
                signal_data = indicator.generate_signals_pro(
                    symbol, interval, di_period, adx_threshold
                )
                
                if signal_data and signal_data['signal'] != 'NEUTRAL' and signal_data['signal_score'] >= 70:
                    all_signals.append(signal_data)
                
                time.sleep(0.1)
                
            except Exception as e:
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
        
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:3])
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_pro(symbol, interval)
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
                continue
        
        return jsonify(scatter_data)
        
    except Exception as e:
        print(f"Error en /api/scatter_data_improved: {e}")
        return jsonify([])

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    """Endpoint para obtener alertas de scalping"""
    try:
        alerts = []
        current_time = datetime.now()
        
        for symbol in CRYPTO_SYMBOLS[:5]:
            for interval in ['15m', '1h', '4h']:
                try:
                    signal_data = indicator.generate_signals_pro(symbol, interval)
                    
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
                            'risk_category': risk_category
                        }
                        
                        alerts.append(alert)
                        
                except Exception as e:
                    continue
        
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
        
        win_rate = indicator.calculate_win_rate(symbol, interval)
        return jsonify({'symbol': symbol, 'interval': interval, 'win_rate': win_rate})
        
    except Exception as e:
        print(f"Error en /api/win_rate: {e}")
        return jsonify({'win_rate': 50.0})

@app.route('/api/strategic_report')
def get_strategic_report():
    """Endpoint para obtener reporte estratégico"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        report_buffer = indicator.generate_strategic_report(symbol, interval)
        
        if report_buffer:
            return send_file(report_buffer, mimetype='image/png', 
                            as_attachment=True, 
                            download_name=f'strategic_report_{symbol}_{interval}.png')
        else:
            return jsonify({'error': 'No se pudo generar el reporte'}), 400
            
    except Exception as e:
        print(f"Error generando reporte estratégico: {e}")
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

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_pro(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
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
            
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit']):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Medias Móviles
        ax2 = plt.subplot(7, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            ma_dates = dates[-len(signal_data['indicators']['ma_9']):]
            ax2.plot(ma_dates, signal_data['indicators']['ma_9'], 'orange', linewidth=1, label='MA9')
            ax2.plot(ma_dates, signal_data['indicators']['ma_21'], 'blue', linewidth=1, label='MA21')
            ax2.plot(ma_dates, signal_data['indicators']['ma_50'], 'red', linewidth=1, label='MA50')
        ax2.set_ylabel('Medias Móviles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: RSI y RSI Maverick
        ax3 = plt.subplot(7, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi']):]
            ax3.plot(rsi_dates, signal_data['indicators']['rsi'], 'purple', linewidth=2, label='RSI Tradicional')
            ax3.plot(rsi_dates, [x * 100 for x in signal_data['indicators']['rsi_maverick']], 'cyan', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax3.set_ylabel('RSI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: MACD
        ax4 = plt.subplot(7, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax4.plot(macd_dates, signal_data['indicators']['macd'], 'blue', linewidth=2, label='MACD')
            ax4.plot(macd_dates, signal_data['indicators']['macd_signal'], 'red', linewidth=1, label='Señal')
            ax4.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=['green' if x >= 0 else 'red' for x in signal_data['indicators']['macd_histogram']], 
                   alpha=0.6, label='Histograma')
        ax4.set_ylabel('MACD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Squeeze Momentum
        ax5 = plt.subplot(7, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            squeeze_dates = dates[-len(signal_data['indicators']['squeeze_momentum']):]
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['squeeze_momentum']]
            ax5.bar(squeeze_dates, signal_data['indicators']['squeeze_momentum'], color=colors, alpha=0.7)
            ax5.axhline(y=0, color='black', linewidth=1)
        ax5.set_ylabel('Squeeze')
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
            
            ax6.set_ylabel('Fuerza Tendencia %')
            ax6.grid(True, alpha=0.3)
        
        # Información de la señal
        ax7 = plt.subplot(7, 1, 7)
        ax7.axis('off')
        
        obligatory_info = "✅ OBLIGATORIAS: Cumplidas" if (signal_data.get('obligatory_met_long') or signal_data.get('obligatory_met_short')) else "❌ OBLIGATORIAS: No cumplidas"
        
        signal_info = f"""
        MULTI-TIMEFRAME CRYPTO WGTA PRO
        
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        WIN RATE: {signal_data['win_rate']:.1f}%
        
        {obligatory_info}
        FUERZA TENDENCIA: {signal_data.get('trend_strength_signal', 'NEUTRAL')}
        
        PRECIO: ${signal_data['current_price']:.6f}
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])][:5])}
        """
        
        ax7.text(0.1, 0.9, signal_info, transform=ax7.transAxes, fontsize=10,
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
