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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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

# Estrategias por temporalidad
VOLUME_EMA_STRATEGY_TFS = ['1h', '4h', '12h', '1D']
MULTI_TF_STRATEGY_TFS = ['15m', '30m', '1h', '2h', '4h', '8h']

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_operations = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.volume_strategy_cache = {}
        self.chart_pattern_cache = {}
        self.divergence_cache = {}
        
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
            remaining_seconds = (next_close - current_time).total_seconds()
            return remaining_seconds <= (15 * 60 * 0.75)
        elif interval == '30m':
            next_close = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0) + timedelta(minutes=30)
            remaining_seconds = (next_close - current_time).total_seconds()
            return remaining_seconds <= (30 * 60 * 0.75)
        elif interval == '1h':
            next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            remaining_seconds = (next_close - current_time).total_seconds()
            return remaining_seconds <= (60 * 60 * 0.5)
        elif interval == '2h':
            current_hour = current_time.hour
            next_2h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour % 2 == 0:
                next_2h_close += timedelta(hours=2)
            else:
                next_2h_close += timedelta(hours=1)
            remaining_seconds = (next_2h_close - current_time).total_seconds()
            return remaining_seconds <= (120 * 60 * 0.5)
        elif interval == '4h':
            current_hour = current_time.hour
            next_4h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 4
            if remainder == 0:
                next_4h_close += timedelta(hours=4)
            else:
                next_4h_close += timedelta(hours=4 - remainder)
            remaining_seconds = (next_4h_close - current_time).total_seconds()
            return remaining_seconds <= (240 * 60 * 0.25)
        elif interval == '8h':
            current_hour = current_time.hour
            next_8h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 8
            if remainder == 0:
                next_8h_close += timedelta(hours=8)
            else:
                next_8h_close += timedelta(hours=8 - remainder)
            remaining_seconds = (next_8h_close - current_time).total_seconds()
            return remaining_seconds <= (480 * 60 * 0.25)
        elif interval == '12h':
            current_hour = current_time.hour
            next_12h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour < 8:
                next_12h_close = next_12h_close.replace(hour=20)
            else:
                next_12h_close = next_12h_close.replace(hour=8) + timedelta(days=1)
            remaining_seconds = (next_12h_close - current_time).total_seconds()
            return remaining_seconds <= (720 * 60 * 0.25)
        elif interval == '1D':
            tomorrow_8pm = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
            if current_time.hour >= 20:
                tomorrow_8pm += timedelta(days=1)
            remaining_seconds = (tomorrow_8pm - current_time).total_seconds()
            return remaining_seconds <= (1440 * 60 * 0.25)
        elif interval == '1W':
            days_passed = current_time.weekday()
            next_monday = current_time + timedelta(days=(7 - days_passed))
            next_monday = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            remaining_seconds = (next_monday - current_time).total_seconds()
            return remaining_seconds <= (10080 * 60 * 0.1)
        
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

    def calculate_support_resistance_levels(self, high, low, close, num_levels=6, lookback=50):
        """Calcular soportes y resistencias dinámicas"""
        try:
            n = len(close)
            if n < lookback:
                lookback = n
            
            # Usar pivots de Fibonacci extendidos
            recent_high = np.max(high[-lookback:])
            recent_low = np.min(low[-lookback:])
            diff = recent_high - recent_low
            
            levels = []
            
            # Niveles de Fibonacci
            fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
            for fib in fib_levels:
                level = recent_low + (diff * fib)
                levels.append(level)
            
            # Agrupar niveles cercanos
            sorted_levels = sorted(levels)
            merged_levels = []
            current_group = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if level - current_group[-1] < diff * 0.05:
                    current_group.append(level)
                else:
                    merged_levels.append(np.mean(current_group))
                    current_group = [level]
            
            if current_group:
                merged_levels.append(np.mean(current_group))
            
            # Asegurar número de niveles
            if len(merged_levels) > num_levels:
                merged_levels = merged_levels[:num_levels]
            elif len(merged_levels) < num_levels:
                # Añadir niveles equidistantes
                while len(merged_levels) < num_levels:
                    extra_level = np.mean([merged_levels[-1], recent_high])
                    merged_levels.append(extra_level)
            
            # Separar soportes y resistencias
            current_price = close[-1]
            supports = [l for l in merged_levels if l < current_price]
            resistances = [l for l in merged_levels if l > current_price]
            
            # Asegurar al menos 2 de cada uno
            while len(supports) < 2 and len(merged_levels) > 0:
                min_level = min(merged_levels)
                if min_level < current_price:
                    supports.append(min_level)
                merged_levels.remove(min_level)
            
            while len(resistances) < 2 and len(merged_levels) > 0:
                max_level = max(merged_levels)
                if max_level > current_price:
                    resistances.append(max_level)
                merged_levels.remove(max_level)
            
            supports = sorted(supports)
            resistances = sorted(resistances)
            
            return supports, resistances
            
        except Exception as e:
            print(f"Error calculando soportes/resistencias: {e}")
            current_price = close[-1] if len(close) > 0 else 0
            supports = [current_price * 0.95, current_price * 0.90]
            resistances = [current_price * 1.05, current_price * 1.10]
            return supports, resistances

    def calculate_optimal_entry_exit_v2(self, df, signal_type, supports, resistances, leverage=15):
        """Calcular entradas y salidas óptimas V2 con soportes/resistencias"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Filtrar niveles relevantes
            relevant_supports = [s for s in supports if s < current_price]
            relevant_resistances = [r for r in resistances if r > current_price]
            
            if signal_type == 'LONG':
                # Entrada en el soporte más cercano o 1% por debajo del precio
                if relevant_supports:
                    entry = max(relevant_supports[-1], current_price * 0.99)
                else:
                    entry = current_price * 0.99
                
                # Stop loss debajo del siguiente soporte o -1.8 ATR
                if len(relevant_supports) > 1:
                    stop_loss = relevant_supports[-2] * 0.99
                else:
                    stop_loss = entry - (current_atr * 1.8)
                
                # Take profits en resistencias
                take_profits = []
                for i, resistance in enumerate(relevant_resistances[:3]):
                    tp = resistance * 0.99
                    if tp > entry:
                        take_profits.append(tp)
                
                # Si no hay resistencias, usar ATR
                if not take_profits:
                    take_profits = [entry + (2 * (entry - stop_loss))]
                
            else:  # SHORT
                # Entrada en la resistencia más cercana o 1% por encima del precio
                if relevant_resistances:
                    entry = min(relevant_resistances[0], current_price * 1.01)
                else:
                    entry = current_price * 1.01
                
                # Stop loss encima de la siguiente resistencia o +1.8 ATR
                if len(relevant_resistances) > 1:
                    stop_loss = relevant_resistances[1] * 1.01
                else:
                    stop_loss = entry + (current_atr * 1.8)
                
                # Take profits en soportes
                take_profits = []
                for i, support in enumerate(relevant_supports[:3]):
                    tp = support * 1.01
                    if tp < entry:
                        take_profits.append(tp)
                
                # Si no hay soportes, usar ATR
                if not take_profits:
                    take_profits = [entry - (2 * (stop_loss - entry))]
            
            # Soportes y resistencias para mostrar
            display_supports = relevant_supports[-2:] if len(relevant_supports) >= 2 else relevant_supports
            display_resistances = relevant_resistances[:2] if len(relevant_resistances) >= 2 else relevant_resistances
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits[:2]],
                'support_levels': [float(s) for s in display_supports],
                'resistance_levels': [float(r) for r in display_resistances],
                'atr': float(current_atr),
                'atr_percentage': float(current_atr / current_price) if current_price > 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas V2: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support_levels': [current_price * 0.95, current_price * 0.90],
                'resistance_levels': [current_price * 1.05, current_price * 1.10],
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente con validación"""
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
        """Calcular SMA manualmente con validación"""
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

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick"""
        try:
            n = len(close)
            
            basis = np.array([np.mean(close[max(0, i-length+1):i+1]) for i in range(n)])
            dev = np.array([np.std(close[max(0, i-length+1):i+1]) for i in range(n)])
            
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
        """Verificar condiciones de Bandas de Bollinger CORREGIDO"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            avg_volume = np.mean(volume[-20:])
            
            # Condiciones más flexibles para Bollinger
            if signal_type == 'LONG':
                # Precio toca o está cerca de la banda inferior + volumen
                touch_lower = current_price <= bb_lower[current_idx] * 1.02
                # Precio rompe la banda media hacia arriba con volumen
                break_middle = (current_price > bb_middle[current_idx] and 
                              current_volume > avg_volume * 1.1)
                # Precio rebota desde la banda inferior
                bounce_lower = (current_price > bb_lower[current_idx] and 
                               close[current_idx-1] <= bb_lower[current_idx-1] * 1.01)
                
                return touch_lower or break_middle or bounce_lower
                
            else:  # SHORT
                # Precio toca o está cerca de la banda superior + volumen
                touch_upper = current_price >= bb_upper[current_idx] * 0.98
                # Precio rompe la banda media hacia abajo con volumen
                break_middle = (current_price < bb_middle[current_idx] and 
                              current_volume > avg_volume * 1.1)
                # Precio rechaza la banda superior
                rejection_upper = (current_price < bb_upper[current_idx] and 
                                 close[current_idx-1] >= bb_upper[current_idx-1] * 0.99)
                
                return touch_upper or break_middle or rejection_upper
                
        except Exception as e:
            print(f"Error verificando condiciones Bollinger: {e}")
            return False

    def check_multi_timeframe_trend(self, symbol, timeframe):
        """Verificar tendencia en múltiples temporalidades"""
        try:
            # Para temporalidades 12h, 1D, 1W no aplica Multi-Timeframe
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
        """Verificar condiciones multi-timeframe obligatorias"""
        try:
            # Para temporalidades 12h, 1D, 1W no es obligatorio Multi-Timeframe
            if interval in ['12h', '1D', '1W']:
                return False
                
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False
            
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            if signal_type == 'LONG':
                # TF Mayor: Alcista o Neutral
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                # TF Medio: Alcista
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'
                # TF Menor: Fuerza Maverick Alcista sin zona no operar
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
                # TF Mayor: Bajista o Neutral
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                # TF Medio: Bajista
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'
                # TF Menor: Fuerza Maverick Bajista sin zona no operar
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
        """Implementación del RSI Modificado Maverick"""
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

    def detect_divergence_extended(self, price, indicator, lookback=20, confirmation_period=4):
        """Detectar divergencias extendidas con período de confirmación"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-confirmation_period):
            price_window = price[i-lookback:i+1]
            indicator_window = indicator[i-lookback:i+1]
            
            # Buscar mínimos en precio
            price_min_idx = np.argmin(price_window)
            price_min = price_window[price_min_idx]
            
            # Buscar mínimos en indicador
            indicator_min_idx = np.argmin(indicator_window)
            indicator_min = indicator_window[indicator_min_idx]
            
            # Divergencia alcista: precio hace menor mínimo pero indicador hace mayor mínimo
            if (price_min_idx > lookback//2 and 
                price_min < np.min(price_window[:price_min_idx]) * 0.995 and
                indicator_min > np.min(indicator_window[:indicator_min_idx]) * 1.005):
                
                # Marcar divergencia y las siguientes 4 velas como válidas
                for j in range(confirmation_period+1):
                    if i + j < n:
                        bullish_div[i + j] = True
            
            # Buscar máximos en precio
            price_max_idx = np.argmax(price_window)
            price_max = price_window[price_max_idx]
            
            # Buscar máximos en indicador
            indicator_max_idx = np.argmax(indicator_window)
            indicator_max = indicator_window[indicator_max_idx]
            
            # Divergencia bajista: precio hace mayor máximo pero indicador hace menor máximo
            if (price_max_idx > lookback//2 and 
                price_max > np.max(price_window[:price_max_idx]) * 1.005 and
                indicator_max < np.max(indicator_window[:indicator_max_idx]) * 0.995):
                
                # Marcar divergencia y las siguientes 4 velas como válidas
                for j in range(confirmation_period+1):
                    if i + j < n:
                        bearish_div[i + j] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def check_breakout_extended(self, high, low, close, support_levels, resistance_levels, confirmation_period=1):
        """Detectar rupturas extendidas con período de confirmación"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Verificar rupturas de resistencias
            for resistance in resistance_levels:
                if close[i] > resistance and high[i] > high[i-1]:
                    # Marcar ruptura y la siguiente vela como válida
                    for j in range(confirmation_period+1):
                        if i + j < n:
                            breakout_up[i + j] = True
            
            # Verificar rupturas de soportes
            for support in support_levels:
                if close[i] < support and low[i] < low[i-1]:
                    # Marcar ruptura y la siguiente vela como válida
                    for j in range(confirmation_period+1):
                        if i + j < n:
                            breakout_down[i + j] = True
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover_extended(self, plus_di, minus_di, lookback=3, confirmation_period=1):
        """Detectar cruces de +DI y -DI con período de confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                # Marcar cruce y la siguiente vela como válida
                for j in range(confirmation_period+1):
                    if i + j < n:
                        di_cross_bullish[i + j] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                # Marcar cruce y la siguiente vela como válida
                for j in range(confirmation_period+1):
                    if i + j < n:
                        di_cross_bearish[i + j] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

    def check_ma_crossover_extended(self, ma_fast, ma_slow, confirmation_period=1):
        """Detectar cruces de medias móviles con período de confirmación"""
        n = len(ma_fast)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MA rápida cruza por encima de MA lenta
            if (ma_fast[i] > ma_slow[i] and 
                ma_fast[i-1] <= ma_slow[i-1]):
                # Marcar cruce y la siguiente vela como válida
                for j in range(confirmation_period+1):
                    if i + j < n:
                        ma_cross_bullish[i + j] = True
            
            # Cruce bajista: MA rápida cruza por debajo de MA lenta
            if (ma_fast[i] < ma_slow[i] and 
                ma_fast[i-1] >= ma_slow[i-1]):
                # Marcar cruce y la siguiente vela como válida
                for j in range(confirmation_period+1):
                    if i + j < n:
                        ma_cross_bearish[i + j] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()

    def check_macd_crossover_extended(self, macd, macd_signal, confirmation_period=1):
        """Detectar cruces de MACD con período de confirmación"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MACD cruza por encima de la señal
            if (macd[i] > macd_signal[i] and 
                macd[i-1] <= macd_signal[i-1]):
                # Marcar cruce y la siguiente vela como válida
                for j in range(confirmation_period+1):
                    if i + j < n:
                        macd_cross_bullish[i + j] = True
            
            # Cruce bajista: MACD cruza por debajo de la señal
            if (macd[i] < macd_signal[i] and 
                macd[i-1] >= macd_signal[i-1]):
                # Marcar cruce y la siguiente vela como válida
                for j in range(confirmation_period+1):
                    if i + j < n:
                        macd_cross_bearish[i + j] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()

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

    def check_adx_slope(self, adx, lookback=5):
        """Verificar pendiente positiva del ADX"""
        n = len(adx)
        adx_slope_positive = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            if adx[i] > 25:  # ADX por encima del nivel
                # Calcular pendiente
                x = np.arange(lookback)
                y = adx[i-lookback+1:i+1]
                
                if len(y) == lookback:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    if slope > 0:
                        adx_slope_positive[i] = True
        
        return adx_slope_positive.tolist()

    def detect_chart_patterns_extended(self, high, low, close, lookback=50, confirmation_period=7):
        """Detectar patrones de chartismo extendidos"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-confirmation_period):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            window_close = close[i-lookback:i+1]
            
            # Hombro Cabeza Hombro (Head & Shoulders)
            if len(window_high) >= 20:
                # Buscar picos
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 3:
                    sorted_peaks = sorted(peaks, key=lambda x: x[0])
                    # Verificar patrón HCH: pico medio más alto, laterales similares
                    if (sorted_peaks[1][1] > sorted_peaks[0][1] * 1.02 and
                        sorted_peaks[1][1] > sorted_peaks[2][1] * 1.02 and
                        abs(sorted_peaks[0][1] - sorted_peaks[2][1]) / sorted_peaks[0][1] < 0.02):
                        
                        # Marcar patrón y las siguientes 7 velas como válidas
                        for j in range(confirmation_period+1):
                            if i + j < n:
                                patterns['head_shoulders'][i + j] = True
            
            # Doble Techo (Double Top)
            if len(window_high) >= 15:
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 2:
                    last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                    if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                        # Marcar patrón y las siguientes 7 velas como válidas
                        for j in range(confirmation_period+1):
                            if i + j < n:
                                patterns['double_top'][i + j] = True
            
            # Doble Fondo (Double Bottom)
            if len(window_low) >= 15:
                troughs = []
                for j in range(1, len(window_low)-1):
                    if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                        troughs.append((j, window_low[j]))
                
                if len(troughs) >= 2:
                    last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                    if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                        # Marcar patrón y las siguientes 7 velas como válidas
                        for j in range(confirmation_period+1):
                            if i + j < n:
                                patterns['double_bottom'][i + j] = True
        
        return patterns

    def calculate_volume_anomaly_improved(self, volume, close, period=21, std_multiplier=2.5):
        """Calcular anomalías de volumen mejorado"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_ma = np.zeros(n)
            
            # Calcular EMA de volumen (21 periodos)
            for i in range(period, n):
                volume_window = volume[max(0, i-period+1):i+1]
                volume_ma[i] = np.mean(volume_window)
                
                # Ratio volumen actual vs EMA
                if volume_ma[i] > 0:
                    volume_ratio[i] = volume[i] / volume_ma[i]
                else:
                    volume_ratio[i] = 1
                
                # Detectar anomalía (> 2.5x MA)
                if volume_ratio[i] > std_multiplier:
                    volume_anomaly[i] = True
                
                # Detectar clusters (múltiples anomalías en 5 periodos)
                if i >= 10:
                    recent_anomalies = volume_anomaly[max(0, i-4):i+1]
                    if np.sum(recent_anomalies) >= 2:
                        volume_clusters[i] = True
            
            # Color de barras basado en dirección del precio
            volume_colors = []
            for i in range(n):
                if i == 0:
                    volume_colors.append('gray')
                else:
                    if close[i] > close[i-1]:
                        volume_colors.append('green')
                    else:
                        volume_colors.append('red')
            
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

    def check_volume_ema_strategy(self, symbol, interval):
        """Nueva estrategia: Desplome por Volumen + EMA21 con Filtros FTMaverick/Multi-Timeframe"""
        try:
            # Solo para temporalidades específicas
            if interval not in VOLUME_EMA_STRATEGY_TFS:
                return None
            
            cache_key = f"volume_ema_{symbol}_{interval}"
            if cache_key in self.volume_strategy_cache:
                cached_data, timestamp = self.volume_strategy_cache[cache_key]
                if (datetime.now() - timestamp).seconds < 300:
                    return cached_data
            
            # Obtener datos
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular EMA21 de precio
            ema_21 = self.calculate_ema(close, 21)
            
            # Calcular MA21 de volumen
            volume_ma_21 = np.zeros(len(volume))
            for i in range(21, len(volume)):
                volume_ma_21[i] = np.mean(volume[i-20:i+1])
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            current_volume_ma = volume_ma_21[current_idx] if current_idx < len(volume_ma_21) else 0
            current_ema_21 = ema_21[current_idx] if current_idx < len(ema_21) else 0
            
            # Condición A: Volumen y EMA
            volume_condition = current_volume > (current_volume_ma * 2.5) if current_volume_ma > 0 else False
            
            # Condición B: Filtro FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            ftm_condition = not ftm_data['no_trade_zones'][current_idx]
            
            # Condición C: Filtro Multi-Timeframe
            multi_tf_condition = False
            if interval in TIMEFRAME_HIERARCHY:
                hierarchy = TIMEFRAME_HIERARCHY[interval]
                
                # Timeframe Mayor
                mayor_df = self.get_kucoin_data(symbol, hierarchy['mayor'], 50)
                if mayor_df is not None and len(mayor_df) > 20:
                    mayor_close = mayor_df['close'].values
                    mayor_ema = self.calculate_ema(mayor_close, 9)
                    mayor_price = mayor_close[-1]
                    mayor_trend = 'BULLISH' if mayor_price > mayor_ema[-1] else 'BEARISH' if mayor_price < mayor_ema[-1] else 'NEUTRAL'
                    
                    # Timeframe Menor
                    menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
                    if menor_df is not None and len(menor_df) > 10:
                        menor_trend_data = self.calculate_trend_strength_maverick(menor_df['close'].values)
                        menor_signal = menor_trend_data['strength_signals'][-1]
                        
                        # Determinar señal
                        if current_price > current_ema_21:  # Potencial LONG
                            if (mayor_trend in ['BULLISH', 'NEUTRAL'] and 
                                menor_signal in ['STRONG_UP', 'WEAK_UP']):
                                multi_tf_condition = True
                                signal_type = 'LONG'
                        elif current_price < current_ema_21:  # Potencial SHORT
                            if (mayor_trend in ['BEARISH', 'NEUTRAL'] and 
                                menor_signal in ['STRONG_DOWN', 'WEAK_DOWN']):
                                multi_tf_condition = True
                                signal_type = 'SHORT'
            
            # Todas las condiciones deben cumplirse
            if volume_condition and ftm_condition and multi_tf_condition:
                # Calcular niveles de soporte/resistencia
                supports, resistances = self.calculate_support_resistance_levels(
                    df['high'].values, df['low'].values, close
                )
                
                # Calcular entradas/salidas
                levels_data = self.calculate_optimal_entry_exit_v2(
                    df, signal_type, supports, resistances, leverage=15
                )
                
                result = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'current_price': float(current_price),
                    'ema_21': float(current_ema_21),
                    'volume_ratio': float(current_volume / current_volume_ma) if current_volume_ma > 0 else 0,
                    'volume_ma': float(current_volume_ma),
                    'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                    'entry': levels_data['entry'],
                    'support_levels': levels_data['support_levels'],
                    'resistance_levels': levels_data['resistance_levels'],
                    'mayor_trend': mayor_trend,
                    'menor_signal': menor_signal,
                    'strategy': 'VOLUME_EMA_FTM'
                }
                
                self.volume_strategy_cache[cache_key] = (result, datetime.now())
                return result
            
            return None
            
        except Exception as e:
            print(f"Error en check_volume_ema_strategy para {symbol} {interval}: {e}")
            return None

    def evaluate_signal_conditions_v2(self, data, current_idx, interval):
        """Evaluar condiciones de señal V2 con pesos corregidos"""
        # Definir pesos según temporalidad
        weights = {
            '15m': {'30m': {'obligatorio': 55, 'complementarios': 45}},
            '30m': {'30m': {'obligatorio': 55, 'complementarios': 45}},
            '1h': {'30m': {'obligatorio': 55, 'complementarios': 45}},
            '2h': {'30m': {'obligatorio': 55, 'complementarios': 45}},
            '4h': {'30m': {'obligatorio': 55, 'complementarios': 45}},
            '8h': {'30m': {'obligatorio': 55, 'complementarios': 45}},
            '12h': {'0m': {'obligatorio': 55, 'complementarios': 45}},
            '1D': {'0m': {'obligatorio': 55, 'complementarios': 45}},
            '1W': {'0m': {'obligatorio': 55, 'complementarios': 45}}
        }
        
        interval_weights = weights.get(interval, weights['4h'])
        
        conditions = {
            'long': {},
            'short': {}
        }
        
        # Inicializar condiciones
        condition_definitions = {
            # Obligatorias
            'multi_timeframe': {'weight': 30 if interval in MULTI_TF_STRATEGY_TFS else 0, 'type': 'obligatorio'},
            'trend_strength': {'weight': 25 if interval != '1W' else 55, 'type': 'obligatorio'},
            'whale_signal': {'weight': 30 if interval in ['12h', '1D'] else 0, 'type': 'obligatorio'},
            
            # Complementarios
            'ma_cross': {'weight': 10, 'type': 'complementario'},
            'di_cross': {'weight': 10, 'type': 'complementario'},
            'adx_slope': {'weight': 5, 'type': 'complementario'},
            'bollinger_bands': {'weight': 8, 'type': 'complementario'},
            'macd_cross': {'weight': 10, 'type': 'complementario'},
            'volume_anomaly': {'weight': 7, 'type': 'complementario'},
            'rsi_maverick_divergence': {'weight': 8, 'type': 'complementario'},
            'rsi_traditional_divergence': {'weight': 5, 'type': 'complementario'},
            'chart_pattern': {'weight': 5, 'type': 'complementario'},
            'breakout': {'weight': 5, 'type': 'complementario'}
        }
        
        for signal_type in ['long', 'short']:
            for key, definition in condition_definitions.items():
                conditions[signal_type][key] = {
                    'value': False, 
                    'weight': definition['weight'], 
                    'type': definition['type'],
                    'description': self.get_condition_description_v2(key)
                }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        # Obtener valores actuales
        current_price = data['close'][current_idx]
        
        # Condiciones LONG
        if interval in MULTI_TF_STRATEGY_TFS:
            conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_long', False)
        
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        if interval in ['12h', '1D']:
            conditions['long']['whale_signal']['value'] = (
                data['whale_pump'][current_idx] > 20 and
                data['confirmed_buy'][current_idx]
            )
        
        # Condiciones complementarias LONG
        conditions['long']['ma_cross']['value'] = (
            current_idx < len(data['ma_cross_bullish']) and 
            data['ma_cross_bullish'][current_idx]
        )
        
        conditions['long']['di_cross']['value'] = (
            current_idx < len(data['di_cross_bullish']) and 
            data['di_cross_bullish'][current_idx]
        )
        
        conditions['long']['adx_slope']['value'] = (
            current_idx < len(data['adx_slope_positive']) and 
            data['adx_slope_positive'][current_idx]
        )
        
        conditions['long']['bollinger_bands']['value'] = data.get('bollinger_conditions_long', False)
        
        conditions['long']['macd_cross']['value'] = (
            current_idx < len(data['macd_cross_bullish']) and 
            data['macd_cross_bullish'][current_idx]
        )
        
        conditions['long']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly']) and 
            data['volume_anomaly'][current_idx]
        )
        
        conditions['long']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bullish_divergence']) and 
            data['rsi_maverick_bullish_divergence'][current_idx]
        )
        
        conditions['long']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bullish_divergence']) and 
            data['rsi_bullish_divergence'][current_idx]
        )
        
        conditions['long']['chart_pattern']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_flag'][current_idx]
        )
        
        conditions['long']['breakout']['value'] = (
            current_idx < len(data['breakout_up']) and 
            data['breakout_up'][current_idx]
        )
        
        # Condiciones SHORT
        if interval in MULTI_TF_STRATEGY_TFS:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        
        conditions['short']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            not data['no_trade_zones'][current_idx]
        )
        
        if interval in ['12h', '1D']:
            conditions['short']['whale_signal']['value'] = (
                data['whale_dump'][current_idx] > 20 and
                data['confirmed_sell'][current_idx]
            )
        
        # Condiciones complementarias SHORT
        conditions['short']['ma_cross']['value'] = (
            current_idx < len(data['ma_cross_bearish']) and 
            data['ma_cross_bearish'][current_idx]
        )
        
        conditions['short']['di_cross']['value'] = (
            current_idx < len(data['di_cross_bearish']) and 
            data['di_cross_bearish'][current_idx]
        )
        
        conditions['short']['adx_slope']['value'] = (
            current_idx < len(data['adx_slope_positive']) and 
            data['adx_slope_positive'][current_idx]
        )
        
        conditions['short']['bollinger_bands']['value'] = data.get('bollinger_conditions_short', False)
        
        conditions['short']['macd_cross']['value'] = (
            current_idx < len(data['macd_cross_bearish']) and 
            data['macd_cross_bearish'][current_idx]
        )
        
        conditions['short']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly']) and 
            data['volume_anomaly'][current_idx]
        )
        
        conditions['short']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bearish_divergence']) and 
            data['rsi_maverick_bearish_divergence'][current_idx]
        )
        
        conditions['short']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bearish_divergence']) and 
            data['rsi_bearish_divergence'][current_idx]
        )
        
        conditions['short']['chart_pattern']['value'] = (
            data['chart_patterns']['head_shoulders'][current_idx] or
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['bearish_flag'][current_idx]
        )
        
        conditions['short']['breakout']['value'] = (
            current_idx < len(data['breakout_down']) and 
            data['breakout_down'][current_idx]
        )
        
        return conditions

    def get_condition_description_v2(self, condition_key):
        """Obtener descripción de condición V2"""
        descriptions = {
            'multi_timeframe': 'Multi-TF obligatorio',
            'trend_strength': 'Fuerza tendencia Maverick',
            'whale_signal': 'Indicador Ballenas',
            'ma_cross': 'Cruce Medias Móviles (9-21)',
            'di_cross': 'Cruce DMI (+DI/-DI)',
            'adx_slope': 'ADX con pendiente positiva',
            'bollinger_bands': 'Bandas de Bollinger',
            'macd_cross': 'Cruce MACD',
            'volume_anomaly': 'Volumen Anómalo',
            'rsi_maverick_divergence': 'Divergencia RSI Maverick',
            'rsi_traditional_divergence': 'Divergencia RSI Tradicional',
            'chart_pattern': 'Patrón Chartista',
            'breakout': 'Ruptura S/R'
        }
        return descriptions.get(condition_key, condition_key)

    def calculate_signal_score_v2(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal V2"""
        total_obligatory_weight = 0
        achieved_obligatory_weight = 0
        total_complementary_weight = 0
        achieved_complementary_weight = 0
        
        fulfilled_conditions = []
        signal_conditions = conditions.get(signal_type, {})
        
        # Separar obligatorias y complementarias
        obligatory_conditions = []
        complementary_conditions = []
        
        for key, condition in signal_conditions.items():
            if condition['type'] == 'obligatorio' and condition['weight'] > 0:
                obligatory_conditions.append(key)
                total_obligatory_weight += condition['weight']
                if condition['value']:
                    achieved_obligatory_weight += condition['weight']
                    fulfilled_conditions.append(condition['description'])
            elif condition['type'] == 'complementario' and condition['weight'] > 0:
                complementary_conditions.append(key)
                total_complementary_weight += condition['weight']
                if condition['value']:
                    achieved_complementary_weight += condition['weight']
                    fulfilled_conditions.append(condition['description'])
        
        # Verificar que todas las condiciones obligatorias se cumplan
        obligatory_met = all(signal_conditions[cond]['value'] for cond in obligatory_conditions)
        
        if not obligatory_met:
            return 0, []
        
        # Calcular puntuación base
        if total_obligatory_weight > 0:
            obligatory_score = (achieved_obligatory_weight / total_obligatory_weight * 55)
        else:
            obligatory_score = 0
        
        if total_complementary_weight > 0:
            complementary_score = (achieved_complementary_weight / total_complementary_weight * 45)
        else:
            complementary_score = 0
        
        base_score = obligatory_score + complementary_score
        
        # Ajustar según MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:  # short
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0
        
        return min(final_score, 100), fulfilled_conditions

    def generate_multi_tf_strategy_signals(self, symbol, interval, di_period=14, adx_threshold=25, 
                                         sr_period=50, rsi_length=14, bb_multiplier=2.0, leverage=15):
        """Generar señales para estrategia Multi-TF"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular todos los indicadores
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Soporte/resistencia
            supports, resistances = self.calculate_support_resistance_levels(high, low, close)
            
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            
            di_cross_bullish, di_cross_bearish = self.check_di_crossover_extended(plus_di, minus_di)
            adx_slope_positive = self.check_adx_slope(adx)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence_extended(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence_extended(close, rsi_traditional)
            
            breakout_up, breakout_down = self.check_breakout_extended(high, low, close, supports, resistances)
            chart_patterns = self.detect_chart_patterns_extended(high, low, close)
            
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover_extended(ma_9, ma_21)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_bullish, macd_cross_bearish = self.check_macd_crossover_extended(macd, macd_signal)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            bollinger_conditions_long = self.check_bollinger_conditions_corrected(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions_corrected(df, interval, 'SHORT')
            
            # Volumen
            volume_data = self.calculate_volume_anomaly_improved(volume, close)
            
            # Verificar condiciones multi-timeframe
            multi_timeframe_long = self.check_multi_timeframe_obligatory(symbol, interval, 'LONG')
            multi_timeframe_short = self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT')
            
            current_idx = -1
            
            # Preparar datos para análisis
            analysis_data = {
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
                'adx_slope_positive': adx_slope_positive,
                'di_cross_bullish': di_cross_bullish,
                'di_cross_bearish': di_cross_bearish,
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
                'ma_cross_bullish': ma_cross_bullish,
                'ma_cross_bearish': ma_cross_bearish,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'macd_cross_bullish': macd_cross_bullish,
                'macd_cross_bearish': macd_cross_bearish,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'volume_anomaly': volume_data['volume_anomaly'],
                'volume_clusters': volume_data['volume_clusters'],
                'volume_ratio': volume_data['volume_ratio'],
                'volume_ma': volume_data['volume_ma'],
                'volume_colors': volume_data['volume_colors'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short,
                'supports': supports,
                'resistances': resistances
            }
            
            conditions = self.evaluate_signal_conditions_v2(analysis_data, current_idx, interval)
            
            # Calcular condición MA200
            current_ma200 = ma_200[current_idx] if current_idx < len(ma_200) else 0
            current_price = close[current_idx]
            ma200_condition = 'above' if current_price > current_ma200 else 'below'

            long_score, long_conditions = self.calculate_signal_score_v2(conditions, 'long', ma200_condition)
            short_score, short_conditions = self.calculate_signal_score_v2(conditions, 'short', ma200_condition)
            
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
            
            # Calcular niveles de entrada/salida
            levels_data = self.calculate_optimal_entry_exit_v2(
                df, signal_type, supports, resistances, leverage
            )
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': levels_data['support_levels'],
                'resistance_levels': levels_data['resistance_levels'],
                'atr': levels_data['atr'],
                'atr_percentage': levels_data['atr_percentage'],
                'volume': float(volume[current_idx]),
                'volume_ma': float(volume_data['volume_ma'][current_idx]) if current_idx < len(volume_data['volume_ma']) else 0,
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
                    'adx_slope_positive': adx_slope_positive[-50:],
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:],
                    'rsi_maverick_bullish_divergence': rsi_maverick_bullish[-50:],
                    'rsi_maverick_bearish_divergence': rsi_maverick_bearish[-50:],
                    'rsi_bullish_divergence': rsi_bullish[-50:],
                    'rsi_bearish_divergence': rsi_bearish[-50:],
                    'breakout_up': breakout_up[-50:],
                    'breakout_down': breakout_down[-50:],
                    'chart_patterns': {k: v[-50:] for k, v in chart_patterns.items()},
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'ma_cross_bullish': ma_cross_bullish[-50:],
                    'ma_cross_bearish': ma_cross_bearish[-50:],
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'macd_cross_bullish': macd_cross_bullish[-50:],
                    'macd_cross_bearish': macd_cross_bearish[-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'volume_anomaly': volume_data['volume_anomaly'][-50:],
                    'volume_clusters': volume_data['volume_clusters'][-50:],
                    'volume_ratio': volume_data['volume_ratio'][-50:],
                    'volume_ma': volume_data['volume_ma'][-50:],
                    'volume_colors': volume_data['volume_colors'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
                    'colors': trend_strength_data['colors'][-50:]
                }
            }
            
        except Exception as e:
            print(f"Error en generate_multi_tf_strategy_signals para {symbol}: {e}")
            import traceback
            traceback.print_exc()
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
            'support_levels': [],
            'resistance_levels': [],
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

    def generate_scalping_alerts_v2(self):
        """Generar alertas para ambas estrategias"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        # Estrategia 1: Multi-TF (15m-8h)
        for interval in MULTI_TF_STRATEGY_TFS:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:8]:
                try:
                    signal_data = self.generate_multi_tf_strategy_signals(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        alert_key = f"mtf_{symbol}_{interval}_{signal_data['signal']}"
                        if (alert_key not in self.alert_cache or 
                            (datetime.now() - self.alert_cache[alert_key]).seconds > 300):
                            
                            alert = {
                                'symbol': symbol,
                                'interval': interval,
                                'signal': signal_data['signal'],
                                'score': signal_data['signal_score'],
                                'entry': signal_data['entry'],
                                'stop_loss': signal_data['stop_loss'],
                                'take_profit': signal_data['take_profit'],
                                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                                'current_price': signal_data['current_price'],
                                'support_levels': signal_data.get('support_levels', []),
                                'resistance_levels': signal_data.get('resistance_levels', []),
                                'ma200_condition': signal_data.get('ma200_condition', 'below'),
                                'strategy': 'MULTI_TF_WGTA'
                            }
                            
                            alerts.append(alert)
                            self.alert_cache[alert_key] = datetime.now()
                            print(f"Alerta Multi-TF generada: {symbol} {interval} {signal_data['signal']}")
                    
                except Exception as e:
                    print(f"Error generando alerta Multi-TF para {symbol} {interval}: {e}")
                    continue
        
        # Estrategia 2: Volumen + EMA21 (1h-1D)
        for interval in VOLUME_EMA_STRATEGY_TFS:
            check_interval = 300 if interval == '1h' else 420 if interval == '4h' else 600
            
            if (current_time.minute % (check_interval // 60) == 0 and 
                current_time.second < 10):
                
                for symbol in CRYPTO_SYMBOLS[:8]:
                    try:
                        signal_data = self.check_volume_ema_strategy(symbol, interval)
                        
                        if signal_data:
                            alert_key = f"volume_ema_{symbol}_{interval}_{signal_data['signal']}"
                            if (alert_key not in self.alert_cache or 
                                (datetime.now() - self.alert_cache[alert_key]).seconds > 600):
                                
                                alerts.append(signal_data)
                                self.alert_cache[alert_key] = datetime.now()
                                print(f"Alerta Volumen+EMA generada: {symbol} {interval} {signal_data['signal']}")
                        
                    except Exception as e:
                        print(f"Error generando alerta Volumen+EMA para {symbol} {interval}: {e}")
                        continue
        
        return alerts

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert(alert_data):
    """Enviar alerta por Telegram con imagen"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Determinar estrategia
        strategy = alert_data.get('strategy', 'MULTI_TF_WGTA')
        
        if strategy == 'MULTI_TF_WGTA':
            # Mensaje para estrategia Multi-TF
            conditions_text = "\n".join([f"• {cond}" for cond in alert_data.get('fulfilled_conditions', [])])
            
            message = f"""
🚨 {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Score: {alert_data['score']:.1f}%

Precio: ${alert_data['current_price']:.6f}
Entrada: ${alert_data['entry']:.6f}
MA200: {alert_data.get('ma200_condition', 'N/A').upper()}

Condiciones cumplidas:
{conditions_text}
            """
            
            # Generar imagen para Multi-TF
            img_buffer = generate_telegram_image_multi_tf(alert_data)
            
        else:  # VOLUME_EMA_FTM
            # Mensaje para estrategia Volumen+EMA
            message = f"""
🚨 VOL+EMA21 | {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Entrada: ${alert_data['current_price']:.6f} | Vol: {alert_data['volume_ratio']:.1f}x
Filtros: FTMaverick OK | MF: {alert_data['mayor_trend']}/{alert_data['menor_signal']}
EMA21: ${alert_data['ema_21']:.6f}
            """
            
            # Generar imagen para Volumen+EMA
            img_buffer = generate_telegram_image_volume_ema(alert_data)
        
        # Enviar mensaje con imagen
        asyncio.run(bot.send_photo(
            chat_id=TELEGRAM_CHAT_ID,
            photo=img_buffer,
            caption=message
        ))
        print(f"Alerta enviada a Telegram: {alert_data['symbol']} {alert_data['interval']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def generate_telegram_image_multi_tf(alert_data):
    """Generar imagen para Telegram - Estrategia Multi-TF"""
    try:
        # Obtener datos
        symbol = alert_data['symbol']
        interval = alert_data['interval']
        
        signal_data = indicator.generate_multi_tf_strategy_signals(symbol, interval)
        if not signal_data or 'data' not in signal_data or not signal_data['data']:
            return None
        
        # Crear figura con 8 subplots
        fig = plt.figure(figsize=(14, 24))
        
        # 1. Gráfico de Velas
        ax1 = plt.subplot(8, 1, 1)
        dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                else d['timestamp'] for d in signal_data['data']]
        opens = [d['open'] for d in signal_data['data']]
        highs = [d['high'] for d in signal_data['data']]
        lows = [d['low'] for d in signal_data['data']]
        closes = [d['close'] for d in signal_data['data']]
        
        # Graficar velas
        for i in range(len(dates)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
        
        # Bandas de Bollinger transparentes
        if 'indicators' in signal_data:
            bb_upper = signal_data['indicators']['bb_upper']
            bb_lower = signal_data['indicators']['bb_lower']
            if len(bb_upper) == len(dates):
                ax1.fill_between(dates, bb_lower, bb_upper, alpha=0.1, color='orange', label='BB')
        
        # Medias móviles
        if 'indicators' in signal_data:
            ma_9 = signal_data['indicators']['ma_9']
            ma_21 = signal_data['indicators']['ma_21']
            ma_50 = signal_data['indicators']['ma_50']
            if len(ma_9) == len(dates):
                ax1.plot(dates, ma_9, 'orange', linewidth=1, alpha=0.7, label='MA9')
                ax1.plot(dates, ma_21, 'blue', linewidth=1, alpha=0.7, label='MA21')
                ax1.plot(dates, ma_50, 'purple', linewidth=1, alpha=0.7, label='MA50')
        
        # Soporte/resistencia
        support_levels = alert_data.get('support_levels', [])
        resistance_levels = alert_data.get('resistance_levels', [])
        
        for level in support_levels:
            ax1.axhline(y=level, color='green', linestyle='--', alpha=0.5, linewidth=1)
        
        for level in resistance_levels:
            ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_title(f'{symbol} - {interval} - {alert_data["signal"]}', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(8, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx = signal_data['indicators']['adx']
            plus_di = signal_data['indicators']['plus_di']
            minus_di = signal_data['indicators']['minus_di']
            
            if len(adx) == len(dates):
                ax2.plot(dates, adx, 'black', linewidth=2, label='ADX')
                ax2.plot(dates, plus_di, 'green', linewidth=1, label='+DI')
                ax2.plot(dates, minus_di, 'red', linewidth=1, label='-DI')
                ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Volumen con anomalías
        ax3 = plt.subplot(8, 1, 3, sharex=ax1)
        if 'indicators' in signal_data and 'volume_colors' in signal_data['indicators']:
            volume_data = [d['volume'] for d in signal_data['data'][-50:]]
            volume_colors = signal_data['indicators']['volume_colors'][-50:]
            volume_ma = signal_data['indicators']['volume_ma'][-50:]
            
            # Barras de volumen
            ax3.bar(dates, volume_data, color=volume_colors, alpha=0.7, width=0.8)
            
            # Línea de MA de volumen
            if len(volume_ma) == len(dates):
                ax3.plot(dates, volume_ma, 'yellow', linewidth=1.5, label='MA21 Vol')
            
            # Anomalías
            volume_anomaly = signal_data['indicators']['volume_anomaly'][-50:]
            for i, (date, anomaly) in enumerate(zip(dates, volume_anomaly)):
                if anomaly:
                    ax3.plot(date, volume_data[i], 'ro', markersize=8, markeredgewidth=1, markeredgecolor='black')
        
        ax3.set_ylabel('Volumen')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick
        ax4 = plt.subplot(8, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            trend_strength = signal_data['indicators']['trend_strength'][-50:]
            colors = signal_data['indicators']['colors'][-50:]
            
            # Barras de fuerza de tendencia
            for i, (date, strength, color) in enumerate(zip(dates, trend_strength, colors)):
                ax4.bar(date, strength, color=color, alpha=0.7, width=0.8)
            
            # Umbral
            threshold = signal_data['indicators'].get('high_zone_threshold', 5)
            ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        
        ax4.set_ylabel('Fuerza Tendencia')
        ax4.grid(True, alpha=0.3)
        
        # 5. Indicador de Ballenas (solo para 12h y 1D)
        ax5 = plt.subplot(8, 1, 5, sharex=ax1)
        if interval in ['12h', '1D'] and 'indicators' in signal_data:
            whale_pump = signal_data['indicators']['whale_pump'][-50:]
            whale_dump = signal_data['indicators']['whale_dump'][-50:]
            
            # Barras
            ax5.bar(dates, whale_pump, color='green', alpha=0.6, label='Compradoras', width=0.8)
            ax5.bar(dates, whale_dump, color='red', alpha=0.6, label='Vendedoras', width=0.8)
            
            # Señales confirmadas
            confirmed_buy = signal_data['indicators']['confirmed_buy'][-50:]
            confirmed_sell = signal_data['indicators']['confirmed_sell'][-50:]
            
            for i, (date, confirm) in enumerate(zip(dates, confirmed_buy)):
                if confirm:
                    ax5.plot(date, whale_pump[i], 'go', markersize=10, markeredgewidth=2, markeredgecolor='black')
            
            for i, (date, confirm) in enumerate(zip(dates, confirmed_sell)):
                if confirm:
                    ax5.plot(date, whale_dump[i], 'ro', markersize=10, markeredgewidth=2, markeredgecolor='black')
        
        ax5.set_ylabel('Ballenas')
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. RSI Maverick
        ax6 = plt.subplot(8, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_maverick = signal_data['indicators']['rsi_maverick'][-50:]
            
            ax6.plot(dates, rsi_maverick, 'blue', linewidth=2)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, linewidth=1)
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        ax6.set_ylabel('RSI Maverick')
        ax6.grid(True, alpha=0.3)
        
        # 7. RSI Tradicional
        ax7 = plt.subplot(8, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_traditional = signal_data['indicators']['rsi_traditional'][-50:]
            
            ax7.plot(dates, rsi_traditional, 'purple', linewidth=2)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7, linewidth=1)
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        ax7.set_ylabel('RSI Tradicional')
        ax7.grid(True, alpha=0.3)
        
        # 8. MACD
        ax8 = plt.subplot(8, 1, 8, sharex=ax1)
        if 'indicators' in signal_data:
            macd = signal_data['indicators']['macd'][-50:]
            macd_signal = signal_data['indicators']['macd_signal'][-50:]
            macd_histogram = signal_data['indicators']['macd_histogram'][-50:]
            
            ax8.plot(dates, macd, 'blue', linewidth=1, label='MACD')
            ax8.plot(dates, macd_signal, 'red', linewidth=1, label='Señal')
            
            # Histograma como barras
            colors_hist = ['green' if x > 0 else 'red' for x in macd_histogram]
            ax8.bar(dates, macd_histogram, color=colors_hist, alpha=0.6, width=0.8)
            
            ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        ax8.set_ylabel('MACD')
        ax8.legend(loc='upper left', fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=120, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen Multi-TF: {e}")
        return None

def generate_telegram_image_volume_ema(alert_data):
    """Generar imagen para Telegram - Estrategia Volumen+EMA"""
    try:
        # Obtener datos
        symbol = alert_data['symbol']
        interval = alert_data['interval']
        
        df = indicator.get_kucoin_data(symbol, interval, 50)
        if df is None or len(df) < 30:
            return None
        
        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, height_ratios=[3, 1])
        
        dates = df['timestamp'].tail(30).tolist()
        dates = [datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S') if isinstance(d, str) else d for d in dates]
        
        opens = df['open'].tail(30).values
        highs = df['high'].tail(30).values
        lows = df['low'].tail(30).values
        closes = df['close'].tail(30).values
        volumes = df['volume'].tail(30).values
        
        # 1. Gráfico de Velas + EMA21
        # Graficar velas
        for i in range(len(dates)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
        
        # Calcular EMA21
        ema_21 = indicator.calculate_ema(closes, 21)
        ax1.plot(dates, ema_21, 'blue', linewidth=2, label='EMA21')
        
        # Destacar vela actual
        current_color = 'green' if alert_data['signal'] == 'LONG' else 'red'
        ax1.plot(dates[-1], closes[-1], f'{current_color}o', markersize=10, 
                markeredgewidth=2, markeredgecolor='black', label=f'Señal {alert_data["signal"]}')
        
        ax1.set_title(f'{symbol} - {interval} - Señal {alert_data["signal"]} por Volumen+EMA21', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precio')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Volumen + Volume_MA21
        # Calcular MA21 de volumen
        volume_ma_21 = np.zeros(len(volumes))
        for i in range(21, len(volumes)):
            volume_ma_21[i] = np.mean(volumes[i-20:i+1])
        
        # Barras de volumen coloreadas
        volume_colors = []
        for i in range(len(closes)):
            if i == 0:
                volume_colors.append('gray')
            else:
                volume_colors.append('green' if closes[i] > closes[i-1] else 'red')
        
        ax2.bar(dates, volumes, color=volume_colors, alpha=0.7, width=0.8)
        
        # Línea de MA de volumen
        ax2.plot(dates, volume_ma_21, 'yellow', linewidth=2, label='MA21 Vol')
        
        # Destacar volumen actual si es anomalía
        current_volume_ratio = alert_data.get('volume_ratio', 1)
        if current_volume_ratio > 2.5:
            ax2.plot(dates[-1], volumes[-1], 'ro', markersize=10, 
                    markeredgewidth=2, markeredgecolor='black', label=f'Vol {current_volume_ratio:.1f}x')
        
        ax2.set_ylabel('Volumen')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=120, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen Volumen+EMA: {e}")
        return None

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            print("Verificando alertas...")
            
            alerts = indicator.generate_scalping_alerts_v2()
            for alert in alerts:
                send_telegram_alert(alert)
            
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
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        # Determinar qué estrategia usar
        if interval in MULTI_TF_STRATEGY_TFS:
            signal_data = indicator.generate_multi_tf_strategy_signals(
                symbol, interval, di_period, adx_threshold, sr_period, 
                rsi_length, bb_multiplier, leverage
            )
        else:
            # Para otras temporalidades, usar señal básica
            signal_data = indicator.generate_multi_tf_strategy_signals(
                symbol, interval, di_period, adx_threshold, sr_period,
                rsi_length, bb_multiplier, leverage
            )
        
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
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:
            try:
                if interval in MULTI_TF_STRATEGY_TFS:
                    signal_data = indicator.generate_multi_tf_strategy_signals(
                        symbol, interval, di_period, adx_threshold, sr_period,
                        rsi_length, bb_multiplier, leverage
                    )
                else:
                    signal_data = indicator.generate_multi_tf_strategy_signals(
                        symbol, interval, di_period, adx_threshold, sr_period,
                        rsi_length, bb_multiplier, leverage
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
    """Endpoint para datos del scatter plot mejorado"""
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
                if interval in MULTI_TF_STRATEGY_TFS:
                    signal_data = indicator.generate_multi_tf_strategy_signals(symbol, interval, di_period, adx_threshold)
                else:
                    signal_data = indicator.generate_multi_tf_strategy_signals(symbol, interval, di_period, adx_threshold)
                
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones
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
                    
                    # Ajustar según señal
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
    """Endpoint para obtener la clasificación de riesgo"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    """Endpoint para obtener alertas de trading"""
    try:
        alerts = indicator.generate_scalping_alerts_v2()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/volume_ema_signals')
def get_volume_ema_signals():
    """Endpoint para obtener señales de volumen+EMA"""
    try:
        interval = request.args.get('interval', '4h')
        
        signals = []
        for symbol in CRYPTO_SYMBOLS[:10]:
            try:
                signal_data = indicator.check_volume_ema_strategy(symbol, interval)
                if signal_data:
                    signals.append(signal_data)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")
                continue
        
        return jsonify({'signals': signals})
        
    except Exception as e:
        print(f"Error en /api/volume_ema_signals: {e}")
        return jsonify({'signals': []})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        # Obtener datos de señal
        if interval in MULTI_TF_STRATEGY_TFS:
            signal_data = indicator.generate_multi_tf_strategy_signals(symbol, interval)
        else:
            signal_data = indicator.generate_multi_tf_strategy_signals(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        fig = plt.figure(figsize=(14, 18))
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(9, 1, 1)
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
            
            # Bandas de Bollinger
            if 'indicators' in signal_data:
                bb_upper = signal_data['indicators']['bb_upper']
                bb_lower = signal_data['indicators']['bb_lower']
                if len(bb_upper) == len(dates):
                    ax1.fill_between(dates, bb_lower, bb_upper, alpha=0.1, color='orange', label='BB')
            
            # Medias móviles
            if 'indicators' in signal_data:
                ma_9 = signal_data['indicators']['ma_9']
                ma_21 = signal_data['indicators']['ma_21']
                ma_50 = signal_data['indicators']['ma_50']
                if len(ma_9) == len(dates):
                    ax1.plot(dates, ma_9, 'orange', linewidth=1, alpha=0.7, label='MA9')
                    ax1.plot(dates, ma_21, 'blue', linewidth=1, alpha=0.7, label='MA21')
                    ax1.plot(dates, ma_50, 'purple', linewidth=1, alpha=0.7, label='MA50')
            
            # Niveles de entrada/salida
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit']):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
            
            # Soporte/resistencia
            support_levels = signal_data.get('support_levels', [])
            resistance_levels = signal_data.get('resistance_levels', [])
            
            for level in support_levels:
                ax1.axhline(y=level, color='orange', linestyle=':', alpha=0.5, label='Soporte' if level == support_levels[0] else '')
            
            for level in resistance_levels:
                ax1.axhline(y=level, color='purple', linestyle=':', alpha=0.5, label='Resistencia' if level == resistance_levels[0] else '')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: ADX con DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx = signal_data['indicators']['adx']
            plus_di = signal_data['indicators']['plus_di']
            minus_di = signal_data['indicators']['minus_di']
            
            if len(adx) == len(dates):
                ax2.plot(dates, adx, 'white', linewidth=2, label='ADX')
                ax2.plot(dates, plus_di, 'green', linewidth=1, label='+DI')
                ax2.plot(dates, minus_di, 'red', linewidth=1, label='-DI')
                ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Volumen con anomalías
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            volume_data = [d['volume'] for d in signal_data['data']]
            volume_ma = signal_data['indicators']['volume_ma']
            
            # Barras de volumen coloreadas
            volume_colors = signal_data['indicators'].get('volume_colors', ['gray'] * len(volume_data))
            if len(volume_colors) == len(volume_data):
                ax3.bar(dates, volume_data, color=volume_colors, alpha=0.6)
            
            # Línea de MA de volumen
            if len(volume_ma) == len(dates):
                ax3.plot(dates, volume_ma, 'yellow', linewidth=1.5, label='MA21 Vol')
        
        ax3.set_ylabel('Volumen')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Fuerza de Tendencia Maverick
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            if len(trend_strength) == len(dates):
                for i, (date, strength, color) in enumerate(zip(dates, trend_strength, colors)):
                    ax4.bar(date, strength, color=color, alpha=0.7, width=0.8)
                
                # Umbral
                threshold = signal_data['indicators'].get('high_zone_threshold', 5)
                ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7)
                ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        
        ax4.set_ylabel('Fuerza Tendencia')
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: RSI Maverick
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_maverick = signal_data['indicators']['rsi_maverick']
            
            if len(rsi_maverick) == len(dates):
                ax5.plot(dates, rsi_maverick, 'blue', linewidth=2)
                ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
                ax5.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
                ax5.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        
        ax5.set_ylabel('RSI Maverick')
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: RSI Tradicional
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_traditional = signal_data['indicators']['rsi_traditional']
            
            if len(rsi_traditional) == len(dates):
                ax6.plot(dates, rsi_traditional, 'purple', linewidth=2)
                ax6.axhline(y=80, color='red', linestyle='--', alpha=0.7)
                ax6.axhline(y=20, color='green', linestyle='--', alpha=0.7)
                ax6.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        ax6.set_ylabel('RSI Tradicional')
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: MACD
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            macd = signal_data['indicators']['macd']
            macd_signal = signal_data['indicators']['macd_signal']
            macd_histogram = signal_data['indicators']['macd_histogram']
            
            if len(macd) == len(dates):
                ax7.plot(dates, macd, 'blue', linewidth=1, label='MACD')
                ax7.plot(dates, macd_signal, 'red', linewidth=1, label='Señal')
                
                # Histograma
                colors_hist = ['green' if x > 0 else 'red' for x in macd_histogram]
                ax7.bar(dates, macd_histogram, color=colors_hist, alpha=0.6, width=0.8)
                
                ax7.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax7.set_ylabel('MACD')
        ax7.legend(loc='upper left', fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: Indicador de Ballenas (si aplica)
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if interval in ['12h', '1D'] and 'indicators' in signal_data:
            whale_pump = signal_data['indicators']['whale_pump']
            whale_dump = signal_data['indicators']['whale_dump']
            
            if len(whale_pump) == len(dates):
                ax8.bar(dates, whale_pump, color='green', alpha=0.6, label='Compradoras', width=0.8)
                ax8.bar(dates, whale_dump, color='red', alpha=0.6, label='Vendedoras', width=0.8)
        
        ax8.set_ylabel('Ballenas')
        ax8.legend(loc='upper left', fontsize=8)
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
        
        ax9.text(0.1, 0.9, signal_info, transform=ax9.transAxes, fontsize=10,
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
