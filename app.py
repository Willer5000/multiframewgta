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

# Configuración optimizada - 40 criptomonedas top (actualizadas)
CRYPTO_SYMBOLS = [
    # Bajo Riesgo (20) - Top market cap
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
    "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT", "ETC-USDT",
    "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "EOS-USDT",
    
    # Medio Riesgo (10) - Proyectos consolidados
    "NEAR-USDT", "AXS-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ONE-USDT", "WAVES-USDT",
    
    # Alto Riesgo (7) - Proyectos emergentes
    "APE-USDT", "GMT-USDT", "SAND-USDT", "OP-USDT", "ARB-USDT",
    "MAGIC-USDT", "RNDR-USDT",
    
    # Memecoins (3) - Top memes
    "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT"
]

# Clasificación de riesgo optimizada
CRYPTO_RISK_CLASSIFICATION = {
    "bajo": [
        "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
        "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
        "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT", "ETC-USDT",
        "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "EOS-USDT"
    ],
    "medio": [
        "NEAR-USDT", "AXS-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
        "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ONE-USDT", "WAVES-USDT"
    ],
    "alto": [
        "APE-USDT", "GMT-USDT", "SAND-USDT", "OP-USDT", "ARB-USDT",
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
    '12h': {'mayor': '1D', 'media': '8h', 'menor': '4h'},
    '1D': {'mayor': '1W', 'media': '12h', 'menor': '8h'},
    '1W': {'mayor': '1M', 'media': '1W', 'menor': '3D'}
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_operations = {}
        self.winrate_data = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.sent_exit_signals = set()
        self.sent_volume_ema_alerts = set()
        self.divergence_cache = {}  # Cache para divergencias (hasta 7 velas)
        self.chart_pattern_cache = {}  # Cache para patrones chartistas
        self.volume_cluster_cache = {}  # Cache para clusters de volumen
        self.ma_cross_cache = {}  # Cache para cruce de medias (1 vela)
        self.dmi_cross_cache = {}  # Cache para cruce DMI (1 vela)
        self.breakout_cache = {}  # Cache para rupturas (1 vela)
        self.macd_cross_cache = {}  # Cache para cruce MACD (1 vela)
    
    def get_bolivia_time(self):
        return datetime.now(self.bolivia_tz)
    
    def is_scalping_time(self):
        now = self.get_bolivia_time()
        if now.weekday() >= 5:
            return False
        return 4 <= now.hour < 16

    def calculate_remaining_time(self, interval, current_time, threshold_ratio=0.5):
        """Calcular tiempo restante para el cierre de la vela"""
        if interval == '15m':
            minutes = current_time.minute
            remainder = 15 - (minutes % 15)
            total_minutes = 15
        elif interval == '30m':
            minutes = current_time.minute
            remainder = 30 - (minutes % 30)
            total_minutes = 30
        elif interval == '1h':
            remainder = 60 - current_time.minute
            total_minutes = 60
        elif interval == '2h':
            hour = current_time.hour
            remainder_hours = 2 - (hour % 2)
            remainder = remainder_hours * 60 - current_time.minute
            total_minutes = 120
        elif interval == '4h':
            hour = current_time.hour
            remainder_hours = 4 - (hour % 4)
            remainder = remainder_hours * 60 - current_time.minute
            total_minutes = 240
        elif interval == '8h':
            hour = current_time.hour
            remainder_hours = 8 - (hour % 8)
            remainder = remainder_hours * 60 - current_time.minute
            total_minutes = 480
        elif interval == '12h':
            hour = current_time.hour
            if hour < 12:
                remainder = (12 - hour) * 60 - current_time.minute
            else:
                remainder = (24 - hour) * 60 - current_time.minute
            total_minutes = 720
        elif interval == '1D':
            hours_left = 24 - current_time.hour
            remainder = hours_left * 60 - current_time.minute
            total_minutes = 1440
        elif interval == '1W':
            days_left = 6 - current_time.weekday()  # Asumiendo semana de trading L-V
            remainder = days_left * 24 * 60
            total_minutes = 7 * 24 * 60
        else:
            return False
        
        elapsed_ratio = 1 - (remainder / total_minutes)
        return elapsed_ratio >= threshold_ratio

    def get_kucoin_data(self, symbol, interval, limit=100):
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < 60:
                    return cached_data
            
            interval_map = {
                '15m': '15min', '30m': '30min', '5m': '5min', '1h': '1hour',
                '2h': '2hour', '4h': '4hour', '8h': '8hour', '12h': '12hour',
                '1D': '1day', '1W': '1week', '1M': '1month'
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
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular múltiples soportes y resistencias
            supports, resistances = self.calculate_support_resistance_levels(high, low, close)
            
            if not supports:
                support_1 = np.min(low[-50:])
                supports = [support_1]
            
            if not resistances:
                resistance_1 = np.max(high[-50:])
                resistances = [resistance_1]
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Para LONG: entrada en el soporte más cercano por debajo del precio
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)
                else:
                    entry = current_price * 0.995
                
                stop_loss = entry - (current_atr * 1.5)
                tp1 = resistances[0] if resistances else entry * 1.02
                
            else:  # SHORT
                # Para SHORT: entrada en la resistencia más cercana por encima del precio
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)
                else:
                    entry = current_price * 1.005
                
                stop_loss = entry + (current_atr * 1.5)
                tp1 = supports[0] if supports else entry * 0.98
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp1)],
                'supports': [float(s) for s in supports[-4:]],  # Últimos 4 soportes
                'resistances': [float(r) for r in resistances[-4:]],  # Últimas 4 resistencias
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
                'supports': [current_price * 0.95],
                'resistances': [current_price * 1.05],
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_support_resistance_levels(self, high, low, close, window=50):
        """Calcular múltiples niveles de soporte y resistencia"""
        n = len(close)
        supports = []
        resistances = []
        
        # Usar ventanas deslizantes para encontrar máximos y mínimos locales
        for i in range(window, n, window//2):  # Superposición del 50%
            start = max(0, i - window)
            end = i
            
            window_high = high[start:end]
            window_low = low[start:end]
            
            # Encontrar máximos y mínimos locales
            for j in range(1, len(window_high)-1):
                # Pico de resistencia
                if (window_high[j] > window_high[j-1] and 
                    window_high[j] > window_high[j+1] and
                    window_high[j] > np.mean(window_high)):
                    resistances.append(window_high[j])
                
                # Valle de soporte
                if (window_low[j] < window_low[j-1] and 
                    window_low[j] < window_low[j+1] and
                    window_low[j] < np.mean(window_low)):
                    supports.append(window_low[j])
        
        # Eliminar duplicados cercanos y ordenar
        supports = self._clean_levels(supports)
        resistances = self._clean_levels(resistances)
        
        return supports[-6:], resistances[-6:]  # Máximo 6 niveles cada uno

    def _clean_levels(self, levels, threshold=0.005):
        """Limpiar niveles cercanos"""
        if not levels:
            return []
        
        levels.sort()
        cleaned = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - cleaned[-1]) / cleaned[-1] > threshold:
                cleaned.append(level)
        
        return cleaned

    def calculate_ema(self, prices, period):
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
        if len(prices) < slow:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
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

    def check_bollinger_conditions_corrected(self, df, interval, signal_type):
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            avg_volume = np.mean(volume[-20:])
            
            if signal_type == 'LONG':
                touch_lower = current_price <= bb_lower[current_idx] * 1.02
                break_middle = (current_price > bb_middle[current_idx] and 
                              current_volume > avg_volume * 1.1)
                bounce_lower = (current_price > bb_lower[current_idx] and 
                               close[current_idx-1] <= bb_lower[current_idx-1] * 1.01)
                
                return touch_lower or break_middle or bounce_lower
                
            else:
                touch_upper = current_price >= bb_upper[current_idx] * 0.98
                break_middle = (current_price < bb_middle[current_idx] and 
                              current_volume > avg_volume * 1.1)
                rejection_upper = (current_price < bb_upper[current_idx] and 
                                 close[current_idx-1] >= bb_upper[current_idx-1] * 0.99)
                
                return touch_upper or break_middle or rejection_upper
                
        except Exception as e:
            print(f"Error verificando condiciones Bollinger: {e}")
            return False

    def check_multi_timeframe_trend(self, symbol, timeframe):
        try:
            if timeframe in ['12h', '1D', '1W']:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
                
            hierarchy = TIMEFRAME_HIERARCHY.get(timeframe, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and timeframe != '15m':
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

    def check_multi_timeframe_obligatory(self, symbol, interval, signal_type):
        try:
            if interval in ['12h', '1D', '1W']:
                return True
                
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False
            
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            if signal_type == 'LONG':
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'
                
                menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
                if menor_df is not None and len(menor_df) > 10:
                    menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                    menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']
                    menor_no_trade = not menor_trend['no_trade_zones'][-1]
                else:
                    menor_ok = True
                    menor_no_trade = True
                
                return mayor_ok and media_ok and menor_ok and menor_no_trade
                
            elif signal_type == 'SHORT':
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'
                
                menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
                if menor_df is not None and len(menor_df) > 10:
                    menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                    menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
                    menor_no_trade = not menor_trend['no_trade_zones'][-1]
                else:
                    menor_ok = True
                    menor_no_trade = True
                
                return mayor_ok and media_ok and menor_ok and menor_no_trade
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones multi-timeframe obligatorias: {e}")
            return False

    def calculate_whale_signals_improved(self, df, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=50, signal_threshold=25):
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
                
                if (whale_dump_smooth[i] > signal_threshold and 
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
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            price_window = price[i-lookback:i+1]
            indicator_window = indicator[i-lookback:i+1]
            
            if (price[i] < np.min(price_window[:-1]) and 
                indicator[i] > np.min(indicator_window[:-1])):
                bullish_div[i] = True
            
            if (price[i] > np.max(price_window[:-1]) and 
                indicator[i] < np.max(indicator_window[:-1])):
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def check_divergence_with_cache(self, symbol, interval, indicator_type, current_idx):
        """Verificar divergencia con cache de 7 velas"""
        cache_key = f"{symbol}_{interval}_{indicator_type}"
        
        if cache_key in self.divergence_cache:
            cache_data, timestamp = self.divergence_cache[cache_key]
            # Verificar si la señal sigue vigente (7 velas)
            if current_idx <= cache_data['idx'] + 7:
                return cache_data['type']
        
        return None

    def check_breakout(self, high, low, close, supports, resistances):
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Ruptura de resistencia
            for resistance in resistances:
                if close[i] > resistance and high[i] > high[i-1]:
                    breakout_up[i] = True
                    break
            
            # Ruptura de soporte
            for support in supports:
                if close[i] < support and low[i] < low[i-1]:
                    breakout_down[i] = True
                    break
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
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

    def calculate_adx(self, high, low, close, period=14):
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

    def detect_chart_patterns(self, high, low, close, lookback=50):
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool),
            'rising_wedge': np.zeros(n, dtype=bool),
            'falling_wedge': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-7):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            window_close = close[i-lookback:i+1]
            
            if len(window_high) >= 20:
                max_idx = np.argmax(window_high)
                if (max_idx > 5 and max_idx < len(window_high)-5 and
                    window_high[max_idx-3] < window_high[max_idx] and
                    window_high[max_idx+3] < window_high[max_idx]):
                    patterns['head_shoulders'][i] = True
            
            if len(window_high) >= 15:
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 2:
                    last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                    if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                        patterns['double_top'][i] = True
            
            if len(window_low) >= 15:
                troughs = []
                for j in range(1, len(window_low)-1):
                    if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                        troughs.append((j, window_low[j]))
                
                if len(troughs) >= 2:
                    last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                    if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                        patterns['double_bottom'][i] = True
            
            if len(window_high) >= 10:
                highs_slope = np.polyfit(range(5), window_high[-5:], 1)[0]
                lows_slope = np.polyfit(range(5), window_low[-5:], 1)[0]
                
                if highs_slope > 0 and lows_slope > 0 and highs_slope > lows_slope:
                    patterns['rising_wedge'][i] = True
                elif highs_slope < 0 and lows_slope < 0 and highs_slope < lows_slope:
                    patterns['falling_wedge'][i] = True
        
        return patterns

    def calculate_volume_anomaly_improved(self, volume, close, period=21, std_multiplier=2.5):
        """Nueva estrategia de volumen: barras verdes para compra, rojas para venta"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_colors = ['gray'] * n
            volume_ma = np.zeros(n)
            
            for i in range(period, n):
                window_volume = volume[max(0, i-period+1):i+1]
                volume_ma[i] = np.mean(window_volume)
                
                if volume_ma[i] > 0:
                    volume_ratio[i] = volume[i] / volume_ma[i]
                else:
                    volume_ratio[i] = 1
                
                if i > 0:
                    if close[i] > close[i-1]:
                        volume_colors[i] = 'green'
                    else:
                        volume_colors[i] = 'red'
                
                if volume_ratio[i] > std_multiplier:
                    volume_anomaly[i] = True
                
                if i >= 10:
                    recent_anomalies = volume_anomaly[max(0, i-9):i+1]
                    if np.sum(recent_anomalies) >= 3:
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ma': volume_ma.tolist(),
                'volume_colors': volume_colors
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly_improved: {e}")
            n = len(volume)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ma': [0] * n,
                'volume_colors': ['gray'] * n
            }

    def check_volume_ema_ftm_signal(self, symbol, interval):
        """Nueva estrategia: Desplome por Volumen + EMA21 con Filtros FTMaverick/Multi-Timeframe"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular EMA21 de precio
            ema_21 = self.calculate_ema(close, 21)
            current_ema = ema_21[-1]
            current_close = close[-1]
            
            # Calcular Volume MA21
            volume_ma_21 = self.calculate_sma(volume, 21)
            current_volume_ma = volume_ma_21[-1]
            current_volume = volume[-1]
            
            # Condición 1: Volumen > 2.5x MA21
            volume_ratio = current_volume / current_volume_ma if current_volume_ma > 0 else 1
            if volume_ratio < 2.5:
                return None
            
            # Condición 2: Precio vs EMA21
            signal_type = None
            if current_close > current_ema:
                signal_type = 'LONG'
            elif current_close < current_ema:
                signal_type = 'SHORT'
            else:
                return None
            
            # Condición 3: FTMaverick - Fuera de zona NO OPERAR
            ftm_current = self.calculate_trend_strength_maverick(close)
            if ftm_current['no_trade_zones'][-1]:
                return None
            
            # Condición 4: Multi-Timeframe
            if interval not in ['12h', '1D', '1W']:
                hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                
                # Timeframe Mayor
                mayor_df = self.get_kucoin_data(symbol, hierarchy.get('mayor', '4h'), 50)
                if mayor_df is not None and len(mayor_df) > 20:
                    mayor_trend = self.check_multi_timeframe_trend(symbol, hierarchy['mayor'])
                    mayor_tendencia = mayor_trend.get('mayor', 'NEUTRAL')
                else:
                    mayor_tendencia = 'NEUTRAL'
                
                # Timeframe Menor
                menor_df = self.get_kucoin_data(symbol, hierarchy.get('menor', '30m'), 30)
                if menor_df is not None and len(menor_df) > 10:
                    menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                    menor_tendencia = menor_trend['strength_signals'][-1]
                else:
                    menor_tendencia = 'NEUTRAL'
                
                if signal_type == 'LONG':
                    if not (mayor_tendencia in ['BULLISH', 'NEUTRAL'] and 
                           menor_tendencia in ['STRONG_UP', 'WEAK_UP']):
                        return None
                else:  # SHORT
                    if not (mayor_tendencia in ['BEARISH', 'NEUTRAL'] and 
                           menor_tendencia in ['STRONG_DOWN', 'WEAK_DOWN']):
                        return None
            
            # Todas las condiciones cumplidas
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'close_price': float(current_close),
                'ema_21': float(current_ema),
                'volume_ratio': float(volume_ratio),
                'volume_ma_21': float(current_volume_ma),
                'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error en check_volume_ema_ftm_signal para {symbol} {interval}: {e}")
            return None

    def evaluate_signal_conditions_corrected(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con PESOS CORREGIDOS según temporalidad"""
        
        # Definir pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            weights = {
                'long': {
                    'multi_timeframe': 30,  # Obligatorio para estas TF
                    'trend_strength': 25,   # Obligatorio
                    'whale_signal': 0,      # Prohibido
                    'bollinger_bands': 8,
                    'adx_dmi': 5,           # ADX con pendiente positiva
                    'ma_cross': 10,         # Cruce MA9 y MA21 (solo el cruce)
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,             # Cruce MACD (solo el cruce)
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7     # Volumen anómalo y clúster >=1
                },
                'short': {
                    'multi_timeframe': 30,
                    'trend_strength': 25,
                    'whale_signal': 0,
                    'bollinger_bands': 8,
                    'adx_dmi': 5,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                }
            }
        elif interval in ['12h', '1D']:
            weights = {
                'long': {
                    'multi_timeframe': 0,   # Prohibido
                    'trend_strength': 25,   # Obligatorio
                    'whale_signal': 30,     # Obligatorio para 12h y 1D
                    'bollinger_bands': 8,
                    'adx_dmi': 5,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                },
                'short': {
                    'multi_timeframe': 0,
                    'trend_strength': 25,
                    'whale_signal': 30,
                    'bollinger_bands': 8,
                    'adx_dmi': 5,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                }
            }
        else:  # 1W
            weights = {
                'long': {
                    'multi_timeframe': 0,   # Prohibido
                    'trend_strength': 55,   # Obligatorio con 55%
                    'whale_signal': 0,      # Prohibido
                    'bollinger_bands': 8,
                    'adx_dmi': 5,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                },
                'short': {
                    'multi_timeframe': 0,
                    'trend_strength': 55,
                    'whale_signal': 0,
                    'bollinger_bands': 8,
                    'adx_dmi': 5,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                }
            }
        
        conditions = {
            'long': {},
            'short': {}
        }
        
        # Inicializar condiciones
        for signal_type in ['long', 'short']:
            for key, weight in weights[signal_type].items():
                conditions[signal_type][key] = {
                    'value': False, 
                    'weight': weight, 
                    'description': self.get_condition_description(key)
                }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        # Obtener valores actuales
        current_price = data['close'][current_idx]
        ma_9 = data['ma_9'][current_idx] if current_idx < len(data['ma_9']) else 0
        ma_21 = data['ma_21'][current_idx] if current_idx < len(data['ma_21']) else 0
        ma_50 = data['ma_50'][current_idx] if current_idx < len(data['ma_50']) else 0
        
        # Verificar condiciones con cache para indicadores de 1 vela
        symbol = data.get('symbol', '')
        interval = data.get('interval', '')
        
        # Condiciones LONG
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_long', False)
        elif interval in ['12h', '1D']:
            conditions['long']['whale_signal']['value'] = (
                current_idx < len(data['confirmed_buy']) and 
                data['confirmed_buy'][current_idx] and
                current_idx < len(data['whale_pump']) and
                data['whale_pump'][current_idx] > 20
            )
        
        conditions['long']['trend_strength']['value'] = (
            current_idx < len(data['trend_strength_signals']) and
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            current_idx < len(data['no_trade_zones']) and
            not data['no_trade_zones'][current_idx]
        )
        
        conditions['long']['bollinger_bands']['value'] = data.get('bollinger_conditions_long', False)
        
        # ADX con pendiente positiva (no solo > threshold)
        if current_idx > 0 and current_idx < len(data['adx']):
            adx_current = data['adx'][current_idx]
            adx_prev = data['adx'][current_idx-1]
            conditions['long']['adx_dmi']['value'] = (
                adx_current > adx_threshold and
                adx_current > adx_prev
            )
        
        # Cruce de medias (solo el cruce)
        if current_idx > 0:
            ma_9_prev = data['ma_9'][current_idx-1] if current_idx-1 < len(data['ma_9']) else 0
            ma_21_prev = data['ma_21'][current_idx-1] if current_idx-1 < len(data['ma_21']) else 0
            conditions['long']['ma_cross']['value'] = (
                ma_9 > ma_21 and
                ma_9_prev <= ma_21_prev
            )
        
        # Divergencia RSI Maverick (7 velas)
        if current_idx < len(data['rsi_maverick_bullish_divergence']):
            divergence_maverick = self.check_divergence_with_cache(
                symbol, interval, 'rsi_maverick', current_idx
            )
            if divergence_maverick == 'BULLISH' or data['rsi_maverick_bullish_divergence'][current_idx]:
                conditions['long']['rsi_maverick_divergence']['value'] = True
        
        # Divergencia RSI Tradicional (7 velas)
        if current_idx < len(data['rsi_bullish_divergence']):
            divergence_traditional = self.check_divergence_with_cache(
                symbol, interval, 'rsi_traditional', current_idx
            )
            if divergence_traditional == 'BULLISH' or data['rsi_bullish_divergence'][current_idx]:
                conditions['long']['rsi_traditional_divergence']['value'] = True
        
        # Cruce MACD (solo el cruce)
        if current_idx > 0 and current_idx < len(data['macd']) and current_idx < len(data['macd_signal']):
            macd_current = data['macd'][current_idx]
            macd_prev = data['macd'][current_idx-1]
            signal_current = data['macd_signal'][current_idx]
            signal_prev = data['macd_signal'][current_idx-1]
            conditions['long']['macd']['value'] = (
                macd_current > signal_current and
                macd_prev <= signal_prev
            )
        
        # Patrones chartistas (7 velas)
        if (current_idx < len(data['chart_patterns']['double_bottom']) and 
            data['chart_patterns']['double_bottom'][current_idx]):
            conditions['long']['chart_pattern']['value'] = True
            conditions['long']['chart_pattern']['description'] = 'Patrón Chartista: Doble fondo'
        elif (current_idx < len(data['chart_patterns']['bullish_flag']) and 
              data['chart_patterns']['bullish_flag'][current_idx]):
            conditions['long']['chart_pattern']['value'] = True
            conditions['long']['chart_pattern']['description'] = 'Patrón Chartista: Bandera alcista'
        elif (current_idx < len(data['chart_patterns']['falling_wedge']) and 
              data['chart_patterns']['falling_wedge'][current_idx]):
            conditions['long']['chart_pattern']['value'] = True
            conditions['long']['chart_pattern']['description'] = 'Patrón Chartista: Cuña descendente'
        
        # Rupturas (1 vela)
        if current_idx < len(data['breakout_up']) and data['breakout_up'][current_idx]:
            conditions['long']['breakout']['value'] = True
        
        # Volumen anómalo y clúster >=1
        if (current_idx < len(data['volume_anomaly']) and 
            current_idx < len(data['volume_clusters']) and
            data['volume_anomaly'][current_idx] and
            data['volume_clusters'][current_idx]):
            conditions['long']['volume_anomaly']['value'] = True
        
        # Condiciones SHORT (simétricas)
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        elif interval in ['12h', '1D']:
            conditions['short']['whale_signal']['value'] = (
                current_idx < len(data['confirmed_sell']) and 
                data['confirmed_sell'][current_idx] and
                current_idx < len(data['whale_dump']) and
                data['whale_dump'][current_idx] > 20
            )
        
        conditions['short']['trend_strength']['value'] = (
            current_idx < len(data['trend_strength_signals']) and
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            current_idx < len(data['no_trade_zones']) and
            not data['no_trade_zones'][current_idx]
        )
        
        conditions['short']['bollinger_bands']['value'] = data.get('bollinger_conditions_short', False)
        
        if current_idx > 0 and current_idx < len(data['adx']):
            adx_current = data['adx'][current_idx]
            adx_prev = data['adx'][current_idx-1]
            conditions['short']['adx_dmi']['value'] = (
                adx_current > adx_threshold and
                adx_current > adx_prev
            )
        
        if current_idx > 0:
            ma_9_prev = data['ma_9'][current_idx-1] if current_idx-1 < len(data['ma_9']) else 0
            ma_21_prev = data['ma_21'][current_idx-1] if current_idx-1 < len(data['ma_21']) else 0
            conditions['short']['ma_cross']['value'] = (
                ma_9 < ma_21 and
                ma_9_prev >= ma_21_prev
            )
        
        if current_idx < len(data['rsi_maverick_bearish_divergence']):
            divergence_maverick = self.check_divergence_with_cache(
                symbol, interval, 'rsi_maverick', current_idx
            )
            if divergence_maverick == 'BEARISH' or data['rsi_maverick_bearish_divergence'][current_idx]:
                conditions['short']['rsi_maverick_divergence']['value'] = True
        
        if current_idx < len(data['rsi_bearish_divergence']):
            divergence_traditional = self.check_divergence_with_cache(
                symbol, interval, 'rsi_traditional', current_idx
            )
            if divergence_traditional == 'BEARISH' or data['rsi_bearish_divergence'][current_idx]:
                conditions['short']['rsi_traditional_divergence']['value'] = True
        
        if current_idx > 0 and current_idx < len(data['macd']) and current_idx < len(data['macd_signal']):
            macd_current = data['macd'][current_idx]
            macd_prev = data['macd'][current_idx-1]
            signal_current = data['macd_signal'][current_idx]
            signal_prev = data['macd_signal'][current_idx-1]
            conditions['short']['macd']['value'] = (
                macd_current < signal_current and
                macd_prev >= signal_prev
            )
        
        if (current_idx < len(data['chart_patterns']['double_top']) and 
            data['chart_patterns']['double_top'][current_idx]):
            conditions['short']['chart_pattern']['value'] = True
            conditions['short']['chart_pattern']['description'] = 'Patrón Chartista: Doble techo'
        elif (current_idx < len(data['chart_patterns']['head_shoulders']) and 
              data['chart_patterns']['head_shoulders'][current_idx]):
            conditions['short']['chart_pattern']['value'] = True
            conditions['short']['chart_pattern']['description'] = 'Patrón Chartista: Hombro cabeza hombro'
        elif (current_idx < len(data['chart_patterns']['bearish_flag']) and 
              data['chart_patterns']['bearish_flag'][current_idx]):
            conditions['short']['chart_pattern']['value'] = True
            conditions['short']['chart_pattern']['description'] = 'Patrón Chartista: Bandera bajista'
        elif (current_idx < len(data['chart_patterns']['rising_wedge']) and 
              data['chart_patterns']['rising_wedge'][current_idx]):
            conditions['short']['chart_pattern']['value'] = True
            conditions['short']['chart_pattern']['description'] = 'Patrón Chartista: Cuña ascendente'
        
        if current_idx < len(data['breakout_down']) and data['breakout_down'][current_idx]:
            conditions['short']['breakout']['value'] = True
        
        if (current_idx < len(data['volume_anomaly']) and 
            current_idx < len(data['volume_clusters']) and
            data['volume_anomaly'][current_idx] and
            data['volume_clusters'][current_idx]):
            conditions['short']['volume_anomaly']['value'] = True
        
        return conditions

    def get_condition_description(self, condition_key):
        descriptions = {
            'multi_timeframe': 'Multi-TF obligatorio',
            'trend_strength': 'Fuerza tendencia Maverick',
            'whale_signal': 'Indicador Ballenas confirmado',
            'bollinger_bands': 'Bandas de Bollinger',
            'adx_dmi': 'ADX con pendiente positiva',
            'ma_cross': 'Cruce de Medias 9-21',
            'rsi_traditional_divergence': 'Divergencia RSI Tradicional',
            'rsi_maverick_divergence': 'Divergencia RSI Maverick',
            'macd': 'Cruce MACD',
            'chart_pattern': 'Patrón Chartista',
            'breakout': 'Ruptura',
            'volume_anomaly': 'Volumen anómalo con clúster'
        }
        return descriptions.get(condition_key, condition_key)

    def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal con umbrales por MA200"""
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        signal_conditions = conditions.get(signal_type, {})
        
        # Verificar condiciones obligatorias según temporalidad
        obligatory_conditions = []
        for key, condition in signal_conditions.items():
            if condition['weight'] >= 25:  # Condiciones con peso >= 25 son obligatorias
                obligatory_conditions.append(key)
        
        # Verificar que todas las condiciones obligatorias se cumplan
        all_obligatory_met = all(signal_conditions[cond]['value'] for cond in obligatory_conditions)
        
        if not all_obligatory_met:
            return 0, []
        
        for key, condition in signal_conditions.items():
            total_weight += condition['weight']
            if condition['value']:
                achieved_weight += condition['weight']
                fulfilled_conditions.append(condition['description'])
        
        if total_weight == 0:
            return 0, []
        
        base_score = (achieved_weight / total_weight * 100)
        
        # Aplicar umbrales según MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:  # short
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0

        return min(final_score, 100), fulfilled_conditions

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            supports, resistances = self.calculate_support_resistance_levels(high, low, close)
            breakout_up, breakout_down = self.check_breakout(high, low, close, supports, resistances)
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            bollinger_conditions_long = self.check_bollinger_conditions_corrected(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions_corrected(df, interval, 'SHORT')
            
            volume_anomaly_data = self.calculate_volume_anomaly_improved(volume, close)
            
            multi_timeframe_long = self.check_multi_timeframe_obligatory(symbol, interval, 'LONG')
            multi_timeframe_short = self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT')
            
            current_idx = -1
            
            analysis_data = {
                'symbol': symbol,
                'interval': interval,
                'close': close,
                'high': high,
                'low': low,
                'volume': volume,
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'confirmed_buy': whale_data['confirmed_buy'],
                'confirmed_sell': whale_data['confirmed_sell'],
                'plus_di': plus_di,
                'minus_di': minus_di,
                'adx': adx,
                'di_cross_bullish': di_cross_bullish,
                'di_cross_bearish': di_cross_bearish,
                'di_trend_bullish': di_trend_bullish,
                'di_trend_bearish': di_trend_bearish,
                'rsi_maverick': rsi_maverick,
                'rsi_traditional': rsi_traditional,
                'rsi_maverick_bullish_divergence': rsi_maverick_bullish,
                'rsi_maverick_bearish_divergence': rsi_maverick_bearish,
                'rsi_bullish_divergence': rsi_bullish,
                'rsi_bearish_divergence': rsi_bearish,
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'chart_patterns': chart_patterns,
                'trend_strength': trend_strength_data['trend_strength'],
                'no_trade_zones': trend_strength_data['no_trade_zones'],
                'trend_strength_signals': trend_strength_data['strength_signals'],
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'volume_anomaly': volume_anomaly_data['volume_anomaly'],
                'volume_clusters': volume_anomaly_data['volume_clusters'],
                'volume_ratio': volume_anomaly_data['volume_ratio'],
                'volume_ma': volume_anomaly_data['volume_ma'],
                'volume_colors': volume_anomaly_data['volume_colors'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short
            }
            
            conditions = self.evaluate_signal_conditions_corrected(analysis_data, current_idx, interval, adx_threshold)
            
            current_ma200 = ma_200[current_idx] if current_idx < len(ma_200) else 0
            current_price = close[current_idx]
            ma200_condition = 'above' if current_price > current_ma200 else 'below'

            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', ma200_condition)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', ma200_condition)
            
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= (65 if ma200_condition == 'above' else 70):
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= (65 if ma200_condition == 'below' else 70):
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            if signal_type in ['LONG', 'SHORT']:
                signal_key = f"{symbol}_{interval}_{signal_type}_{int(time.time())}"
                self.active_operations[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': levels_data['entry'],
                    'timestamp': self.get_bolivia_time(),
                    'score': signal_score
                }
                print(f"Señal generada: {symbol} {interval} {signal_type} Score: {signal_score}%")
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'supports': levels_data['supports'],
                'resistances': levels_data['resistances'],
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
                'rsi_traditional': float(rsi_traditional[current_idx] if current_idx < len(rsi_traditional) else 50),
                'fulfilled_conditions': fulfilled_conditions,
                'multi_timeframe_ok': multi_timeframe_long if signal_type == 'LONG' else multi_timeframe_short,
                'ma200_condition': ma200_condition,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'confirmed_buy': whale_data['confirmed_buy'][-50:],
                    'confirmed_sell': whale_data['confirmed_sell'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'di_cross_bullish': di_cross_bullish[-50:],
                    'di_cross_bearish': di_cross_bearish[-50:],
                    'di_trend_bullish': di_trend_bullish[-50:],
                    'di_trend_bearish': di_trend_bearish[-50:],
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:],
                    'rsi_maverick_bullish_divergence': rsi_maverick_bullish[-50:],
                    'rsi_maverick_bearish_divergence': rsi_maverick_bearish[-50:],
                    'rsi_bullish_divergence': rsi_bullish[-50:],
                    'rsi_bearish_divergence': rsi_bearish[-50:],
                    'breakout_up': breakout_up[-50:],
                    'breakout_down': breakout_down[-50:],
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'volume_anomaly': volume_anomaly_data['volume_anomaly'][-50:],
                    'volume_clusters': volume_anomaly_data['volume_clusters'][-50:],
                    'volume_ratio': volume_anomaly_data['volume_ratio'][-50:],
                    'volume_ma': volume_anomaly_data['volume_ma'][-50:],
                    'volume_colors': volume_anomaly_data['volume_colors'][-50:],
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
            import traceback
            traceback.print_exc()
            return self._create_empty_signal(symbol)

    def _create_empty_signal(self, symbol):
        return {
            'symbol': symbol,
            'current_price': 0,
            'signal': 'NEUTRAL',
            'signal_score': 0,
            'entry': 0,
            'stop_loss': 0,
            'take_profit': [0],
            'supports': [0],
            'resistances': [0],
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
            'fulfilled_conditions': [],
            'multi_timeframe_ok': False,
            'ma200_condition': 'below',
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de trading para la estrategia principal"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time, 0.5)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:15]:
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        alert_key = f"{symbol}_{interval}_{signal_data['signal']}"
                        if (alert_key not in self.alert_cache or 
                            (datetime.now() - self.alert_cache[alert_key]).seconds > 300):
                            
                            alert = {
                                'symbol': symbol,
                                'interval': interval,
                                'signal': signal_data['signal'],
                                'score': signal_data['signal_score'],
                                'entry': signal_data['entry'],
                                'current_price': signal_data['current_price'],
                                'ma200_condition': signal_data.get('ma200_condition', 'below'),
                                'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            alerts.append(alert)
                            self.alert_cache[alert_key] = datetime.now()
                    
                except Exception as e:
                    print(f"Error generando alerta para {symbol} {interval}: {e}")
                    continue
        
        return alerts

    def generate_volume_ema_alerts(self):
        """Generar alertas para la nueva estrategia de volumen"""
        alerts = []
        intervals = ['1h', '4h', '12h', '1D']
        current_time = self.get_bolivia_time()
        
        for interval in intervals:
            # Lógica de ejecución según el intervalo
            if interval == '1h':
                if not self.calculate_remaining_time(interval, current_time, 0.5):
                    continue
            elif interval in ['4h', '12h', '1D']:
                if not self.calculate_remaining_time(interval, current_time, 0.25):
                    continue
            
            for symbol in CRYPTO_SYMBOLS:
                try:
                    signal = self.check_volume_ema_ftm_signal(symbol, interval)
                    if signal:
                        alert_key = f"volume_ema_{symbol}_{interval}_{signal['signal']}"
                        if alert_key not in self.sent_volume_ema_alerts:
                            alerts.append(signal)
                            self.sent_volume_ema_alerts.add(alert_key)
                            print(f"Alerta Vol+EMA21 generada: {symbol} {interval} {signal['signal']}")
                except Exception as e:
                    print(f"Error en generate_volume_ema_alerts para {symbol} {interval}: {e}")
                    continue
        
        return alerts

# Instancia global del indicador
indicator = TradingIndicator()

def get_risk_classification(symbol):
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

async def send_telegram_alert_async(alert_data, alert_type='multiframe'):
    """Enviar alerta por Telegram con imagen"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'multiframe':
            # Mensaje para estrategia multiframe
            message = f"""
🚨 {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Score: {alert_data['score']:.1f}%

Precio: ${alert_data.get('current_price', alert_data['entry']):.6f}
Entrada: ${alert_data['entry']:.6f}
MA200: {'ABOVE' if alert_data.get('ma200_condition') == 'above' else 'BELOW'}

Condiciones cumplidas:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])])}
"""
            
            # Generar imagen para estrategia multiframe
            img_buffer = generate_telegram_image_multiframe(alert_data)
            
        else:  # volume_ema
            message = f"""
🚨 VOL+EMA21 | {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Entrada: ${alert_data['close_price']:.6f} | Vol: {alert_data['volume_ratio']:.1f}x
Filtros: FTMaverick OK | MF: OK
"""
            
            # Generar imagen para estrategia volumen
            img_buffer = generate_telegram_image_volume_ema(alert_data)
        
        # Enviar imagen primero
        await bot.send_photo(
            chat_id=TELEGRAM_CHAT_ID,
            photo=img_buffer,
            caption=message[:1024]  # Limitar caption a 1024 caracteres
        )
        
        print(f"Alerta {alert_type} enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def send_telegram_alert(alert_data, alert_type='multiframe'):
    """Wrapper síncrono para enviar alertas a Telegram"""
    asyncio.run(send_telegram_alert_async(alert_data, alert_type))

def generate_telegram_image_multiframe(alert_data):
    """Generar imagen para Telegram - Estrategia Multiframe"""
    try:
        symbol = alert_data['symbol']
        interval = alert_data['interval']
        
        # Obtener datos
        signal_data = indicator.generate_signals_improved(symbol, interval)
        if not signal_data or not signal_data['data']:
            return None
        
        # Crear figura con 8 subplots
        fig = plt.figure(figsize=(12, 20))
        
        # 1. Gráfico de Velas
        ax1 = plt.subplot(8, 1, 1)
        dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                else d['timestamp'] for d in signal_data['data'][-50:]]
        opens = [d['open'] for d in signal_data['data'][-50:]]
        highs = [d['high'] for d in signal_data['data'][-50:]]
        lows = [d['low'] for d in signal_data['data'][-50:]]
        closes = [d['close'] for d in signal_data['data'][-50:]]
        
        for i in range(len(dates)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
        
        # Añadir Bandas de Bollinger (transparentes)
        if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
            bb_upper = signal_data['indicators']['bb_upper'][-50:]
            bb_lower = signal_data['indicators']['bb_lower'][-50:]
            ax1.fill_between(dates, bb_lower, bb_upper, color='orange', alpha=0.1)
        
        # Añadir soportes y resistencias
        for support in signal_data.get('supports', [])[-4:]:
            ax1.axhline(y=support, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        
        for resistance in signal_data.get('resistances', [])[-4:]:
            ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_title(f'{symbol} - {interval} - Velas Japonesas', fontsize=10)
        ax1.set_ylabel('Precio')
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(8, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx'][-50:]):]
            ax2.plot(adx_dates, signal_data['indicators']['adx'][-50:], 'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, signal_data['indicators']['plus_di'][-50:], 'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, signal_data['indicators']['minus_di'][-50:], 'red', linewidth=1, label='-DI')
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Volumen con Anomalías
        ax3 = plt.subplot(8, 1, 3, sharex=ax1)
        volumes = [d['volume'] for d in signal_data['data'][-50:]]
        if 'indicators' in signal_data and 'volume_colors' in signal_data['indicators']:
            volume_colors = signal_data['indicators']['volume_colors'][-50:]
            for i, (date, vol, color) in enumerate(zip(dates, volumes, volume_colors)):
                ax3.bar(date, vol, color=color, alpha=0.7, width=0.8)
        
        if 'indicators' in signal_data and 'volume_ma' in signal_data['indicators']:
            ax3.plot(dates, signal_data['indicators']['volume_ma'][-50:], 'yellow', linewidth=1.5, label='MA Volumen')
        
        ax3.set_ylabel('Volumen')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick
        ax4 = plt.subplot(8, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_strength = signal_data['indicators']['trend_strength'][-50:]
            colors = signal_data['indicators']['colors'][-50:]
            
            for i, (date, strength, color) in enumerate(zip(dates, trend_strength, colors)):
                ax4.bar(date, strength, color=color, alpha=0.7, width=0.8)
        
        ax4.set_ylabel('FT Maverick')
        ax4.grid(True, alpha=0.3)
        
        # 5. Indicador Ballenas (solo para 12h y 1D)
        if interval in ['12h', '1D'] and 'indicators' in signal_data:
            ax5 = plt.subplot(8, 1, 5, sharex=ax1)
            if 'whale_pump' in signal_data['indicators']:
                whale_dates = dates[-len(signal_data['indicators']['whale_pump'][-50:]):]
                ax5.bar(whale_dates, signal_data['indicators']['whale_pump'][-50:], 
                       color='green', alpha=0.7, label='Compradoras')
                ax5.bar(whale_dates, signal_data['indicators']['whale_dump'][-50:], 
                       color='red', alpha=0.7, label='Vendedoras')
            ax5.set_ylabel('Ballenas')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
        
        # 6. RSI Maverick
        ax6 = plt.subplot(8, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_maverick' in signal_data['indicators']:
            rsi_maverick = signal_data['indicators']['rsi_maverick'][-50:]
            ax6.plot(dates, rsi_maverick, 'blue', linewidth=2)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
        ax6.set_ylabel('RSI Maverick')
        ax6.grid(True, alpha=0.3)
        
        # 7. RSI Tradicional
        ax7 = plt.subplot(8, 1, 7, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_traditional' in signal_data['indicators']:
            rsi_traditional = signal_data['indicators']['rsi_traditional'][-50:]
            ax7.plot(dates, rsi_traditional, 'purple', linewidth=2)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        ax7.set_ylabel('RSI Tradicional')
        ax7.grid(True, alpha=0.3)
        
        # 8. MACD
        ax8 = plt.subplot(8, 1, 8, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd'][-50:]):]
            ax8.plot(macd_dates, signal_data['indicators']['macd'][-50:], 'blue', linewidth=1, label='MACD')
            ax8.plot(macd_dates, signal_data['indicators']['macd_signal'][-50:], 'red', linewidth=1, label='Señal')
            
            histogram = signal_data['indicators']['macd_histogram'][-50:]
            colors = ['green' if x > 0 else 'red' for x in histogram]
            ax8.bar(macd_dates, histogram, color=colors, alpha=0.6, width=0.8)
        
        ax8.set_ylabel('MACD')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen para Telegram (multiframe): {e}")
        return None

def generate_telegram_image_volume_ema(alert_data):
    """Generar imagen para Telegram - Estrategia Volumen+EMA21"""
    try:
        symbol = alert_data['symbol']
        interval = alert_data['interval']
        
        df = indicator.get_kucoin_data(symbol, interval, 50)
        if df is None or len(df) < 30:
            return None
        
        close = df['close'].values
        volume = df['volume'].values
        
        # Calcular EMA21 y Volume MA21
        ema_21 = indicator.calculate_ema(close, 21)
        volume_ma_21 = indicator.calculate_sma(volume, 21)
        
        dates = [datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S') if isinstance(d, str) 
                else d for d in df['timestamp'].tail(30)]
        
        # Crear figura doble
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Gráfico superior: Velas + EMA21
        for i in range(len(dates)):
            color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
            ax1.plot([dates[i], dates[i]], 
                    [df['low'].iloc[i], df['high'].iloc[i]], 
                    color='black', linewidth=1)
            ax1.plot([dates[i], dates[i]], 
                    [df['open'].iloc[i], df['close'].iloc[i]], 
                    color=color, linewidth=3)
        
        # EMA21
        ax1.plot(dates, ema_21[-30:], 'blue', linewidth=2, label='EMA 21')
        
        # Destacar vela actual
        if alert_data['signal'] == 'LONG':
            marker_color = 'green'
        else:
            marker_color = 'red'
        
        ax1.scatter(dates[-1], close[-1], color=marker_color, s=100, 
                   edgecolor='black', zorder=5, label=f"Señal {alert_data['signal']}")
        
        ax1.set_title(f'{symbol} - {interval} - Señal {alert_data["signal"]} por Volumen+EMA21', fontsize=12)
        ax1.set_ylabel('Precio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico inferior: Volumen + Volume MA21
        volume_colors = []
        for i in range(len(dates)):
            if i > 0:
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    volume_colors.append('green')
                else:
                    volume_colors.append('red')
            else:
                volume_colors.append('gray')
        
        ax2.bar(dates, volume[-30:], color=volume_colors, alpha=0.7)
        ax2.plot(dates, volume_ma_21[-30:], 'orange', linewidth=2, label='Volume MA21')
        
        # Destacar volumen actual
        ax2.bar(dates[-1], volume[-1], color=marker_color, alpha=1.0, edgecolor='black')
        
        ax2.set_ylabel('Volumen')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen para Telegram (volume_ema): {e}")
        return None

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            current_time = datetime.now()
            
            # Estrategia principal (multiframe)
            print("Verificando alertas estrategia principal...")
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                send_telegram_alert(alert, 'multiframe')
            
            # Nueva estrategia volumen
            print("Verificando alertas estrategia volumen...")
            volume_alerts = indicator.generate_volume_ema_alerts()
            for alert in volume_alerts:
                send_telegram_alert(alert, 'volume_ema')
            
            time.sleep(60)  # Revisar cada 60 segundos
            
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
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, volume_filter, leverage
        )
        
        if 'indicators' in signal_data:
            for key in signal_data['indicators']:
                if isinstance(signal_data['indicators'][key], (np.ndarray, np.generic)):
                    signal_data['indicators'][key] = signal_data['indicators'][key].tolist()
        
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/multiple_signals')
def get_multiple_signals():
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
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
            'long_signals': long_signals,
            'short_signals': short_signals,
            'total_signals': len(all_signals)
        })
        
    except Exception as e:
        print(f"Error en /api/multiple_signals: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/scatter_data_improved')
def get_scatter_data_improved():
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        
        scatter_data = []
        
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:5])
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval, di_period, adx_threshold)
                if signal_data and signal_data['current_price'] > 0:
                    
                    buy_pressure = min(100, max(0,
                        (signal_data['whale_pump'] / 100 * 25) +
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 20 +
                        (signal_data['rsi_maverick'] * 20) +
                        (1 if signal_data['adx'] > adx_threshold else 0) * 15 +
                        (min(1, signal_data['volume'] / max(1, signal_data['volume_ma'])) * 20)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (signal_data['whale_dump'] / 100 * 25) +
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 20 +
                        ((1 - signal_data['rsi_maverick']) * 20) +
                        (1 if signal_data['adx'] > adx_threshold else 0) * 15 +
                        (min(1, signal_data['volume'] / max(1, signal_data['volume_ma'])) * 20)
                    ))
                    
                    if signal_data['signal'] == 'LONG':
                        buy_pressure = min(100, buy_pressure * 1.3)
                        sell_pressure = max(0, sell_pressure * 0.7)
                    elif signal_data['signal'] == 'SHORT':
                        sell_pressure = min(100, sell_pressure * 1.3)
                        buy_pressure = max(0, buy_pressure * 0.7)
                    
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
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    try:
        alerts = indicator.generate_scalping_alerts()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/volume_ema_signals')
def get_volume_ema_signals():
    try:
        alerts = indicator.generate_volume_ema_alerts()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/volume_ema_signals: {e}")
        return jsonify({'alerts': []})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo - CORREGIDO"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Verificar que tenemos datos
        if not signal_data.get('data') or len(signal_data['data']) == 0:
            return jsonify({'error': 'No hay datos de velas disponibles'}), 400
        
        fig = plt.figure(figsize=(14, 18))
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(9, 1, 1)
        if signal_data['data']:
            dates = []
            for d in signal_data['data'][-50:]:
                if isinstance(d['timestamp'], str):
                    try:
                        dates.append(datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S'))
                    except:
                        dates.append(datetime.now())
                else:
                    dates.append(d['timestamp'])
            
            opens = [d['open'] for d in signal_data['data'][-50:]]
            highs = [d['high'] for d in signal_data['data'][-50:]]
            lows = [d['low'] for d in signal_data['data'][-50:]]
            closes = [d['close'] for d in signal_data['data'][-50:]]
            
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Añadir niveles de trading
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit']):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
            
            # Añadir soportes y resistencias
            for support in signal_data.get('supports', [])[-4:]:
                ax1.axhline(y=support, color='orange', linestyle=':', alpha=0.5, label='Soporte')
            
            for resistance in signal_data.get('resistances', [])[-4:]:
                ax1.axhline(y=resistance, color='purple', linestyle=':', alpha=0.5, label='Resistencia')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: ADX/DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data and 'adx' in signal_data['indicators']:
            adx_data = signal_data['indicators']['adx'][-50:]
            plus_di = signal_data['indicators']['plus_di'][-50:]
            minus_di = signal_data['indicators']['minus_di'][-50:]
            
            if len(dates) == len(adx_data):
                ax2.plot(dates, adx_data, 'white', linewidth=2, label='ADX')
                ax2.plot(dates, plus_di, 'green', linewidth=1, label='+DI')
                ax2.plot(dates, minus_di, 'red', linewidth=1, label='-DI')
                ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Volumen
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        volumes = [d['volume'] for d in signal_data['data'][-50:]]
        if 'indicators' in signal_data and 'volume_colors' in signal_data['indicators']:
            volume_colors = signal_data['indicators']['volume_colors'][-50:]
            if len(dates) == len(volumes) == len(volume_colors):
                for i, (date, vol, color) in enumerate(zip(dates, volumes, volume_colors)):
                    ax3.bar(date, vol, color=color, alpha=0.7, width=0.8)
        
        ax3.set_ylabel('Volumen')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Fuerza de Tendencia Maverick
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_strength = signal_data['indicators']['trend_strength'][-50:]
            colors = signal_data['indicators']['colors'][-50:]
            
            if len(dates) == len(trend_strength) == len(colors):
                for i, (date, strength, color) in enumerate(zip(dates, trend_strength, colors)):
                    ax4.bar(date, strength, color=color, alpha=0.7, width=0.8)
        
        ax4.set_ylabel('FT Maverick')
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: RSI Tradicional
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_traditional' in signal_data['indicators']:
            rsi_traditional = signal_data['indicators']['rsi_traditional'][-50:]
            if len(dates) == len(rsi_traditional):
                ax5.plot(dates, rsi_traditional, 'cyan', linewidth=2, label='RSI Tradicional')
                ax5.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
                ax5.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
                ax5.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        ax5.set_ylabel('RSI Tradicional')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: RSI Maverick
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_maverick' in signal_data['indicators']:
            rsi_maverick = signal_data['indicators']['rsi_maverick'][-50:]
            if len(dates) == len(rsi_maverick):
                ax6.plot(dates, rsi_maverick, 'blue', linewidth=2, label='RSI Maverick')
                ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
                ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
                ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        
        ax6.set_ylabel('RSI Maverick')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: MACD
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data and 'macd' in signal_data['indicators']:
            macd_data = signal_data['indicators']['macd'][-50:]
            macd_signal = signal_data['indicators']['macd_signal'][-50:]
            macd_histogram = signal_data['indicators']['macd_histogram'][-50:]
            
            if len(dates) == len(macd_data):
                ax7.plot(dates, macd_data, 'blue', linewidth=1, label='MACD')
                ax7.plot(dates, macd_signal, 'red', linewidth=1, label='Señal')
                
                colors = ['green' if x > 0 else 'red' for x in macd_histogram]
                ax7.bar(dates, macd_histogram, color=colors, alpha=0.6, label='Histograma')
                
                ax7.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax7.set_ylabel('MACD')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: Ballenas (si aplica)
        if interval in ['12h', '1D'] and 'indicators' in signal_data:
            ax8 = plt.subplot(9, 1, 8, sharex=ax1)
            if 'whale_pump' in signal_data['indicators']:
                whale_pump = signal_data['indicators']['whale_pump'][-50:]
                whale_dump = signal_data['indicators']['whale_dump'][-50:]
                
                if len(dates) == len(whale_pump):
                    ax8.bar(dates, whale_pump, color='green', alpha=0.7, label='Compradoras')
                    ax8.bar(dates, whale_dump, color='red', alpha=0.7, label='Vendedoras')
            
            ax8.set_ylabel('Ballenas')
            ax8.legend(fontsize=8)
            ax8.grid(True, alpha=0.3)
        
        # Información de la señal
        ax9 = plt.subplot(9, 1, 9)
        ax9.axis('off')
        
        multi_tf_info = "✅ MULTI-TIMEFRAME: Confirmado" if signal_data.get('multi_timeframe_ok') else "❌ MULTI-TIMEFRAME: No confirmado"
        ma200_info = f"MA200: {'ENCIMA' if signal_data.get('ma200_condition') == 'above' else 'DEBAJO'}"
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        
        {multi_tf_info}
        {ma200_info}
        
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
        APALANCAMIENTO: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
        """
        
        ax9.text(0.1, 0.9, signal_info, transform=ax9.transAxes, fontsize=9,
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
        return jsonify({'error': 'Error generando reporte', 'details': str(e)}), 500

@app.route('/api/bolivia_time')
def get_bolivia_time():
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
