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

app = Flask(__name__)

# Configuración Telegram
TELEGRAM_BOT_TOKEN = "8007748376:AAHIW8n9b-BtA378g4gF-0-D2mOhn495Q0g"
TELEGRAM_CHAT_ID = "-1003229814161"

# Configuración CoinMarketCap
CMC_API_KEY = "d22df0c59e5e47e0980b89f6eb32ea1b"
CMC_BASE_URL = "https://pro-api.coinmarketcap.com"

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
        self.volume_spike_cache = {}
        self.volume_historical_data = {}
    
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

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con 4 soportes/resistencias"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular 4 niveles de soporte y resistencia
            lookback = 100
            if len(low) >= lookback:
                # Encontrar mínimos locales para soportes
                minima_idx = []
                for i in range(2, len(low)-2):
                    if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
                        minima_idx.append(i)
                
                # Ordenar y tomar los 4 más bajos recientes
                minima_values = [low[i] for i in minima_idx[-20:]]
                minima_values.sort()
                supports = minima_values[:4] if len(minima_values) >= 4 else [np.min(low[-lookback:])] * 4
                
                # Encontrar máximos locales para resistencias
                maxima_idx = []
                for i in range(2, len(high)-2):
                    if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
                        maxima_idx.append(i)
                
                # Ordenar y tomar los 4 más altos recientes
                maxima_values = [high[i] for i in maxima_idx[-20:]]
                maxima_values.sort(reverse=True)
                resistances = maxima_values[:4] if len(maxima_values) >= 4 else [np.max(high[-lookback:])] * 4
            else:
                supports = [np.min(low[-50:])] * 4
                resistances = [np.max(high[-50:])] * 4
            
            # Asegurar que hay suficientes niveles
            while len(supports) < 4:
                supports.append(supports[-1] * 0.98 if supports else current_price * 0.95)
            while len(resistances) < 4:
                resistances.append(resistances[-1] * 1.02 if resistances else current_price * 1.05)
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Entrada en el soporte más cercano por debajo del precio actual
                valid_supports = [s for s in supports if s < current_price]
                entry = valid_supports[-1] if valid_supports else min(supports)
                stop_loss = entry * 0.97
                # Take profit en resistencias progresivas
                take_profit = []
                for i, res in enumerate(resistances):
                    if res > entry and len(take_profit) < 3:
                        take_profit.append(res)
                
                # Si no hay suficientes TPs, calcular basados en ATR
                while len(take_profit) < 3:
                    last_tp = take_profit[-1] if take_profit else entry
                    take_profit.append(last_tp + (current_atr * (1 + len(take_profit) * 0.5)))
                
            else:  # SHORT
                # Entrada en la resistencia más cercana por encima del precio actual
                valid_resistances = [r for r in resistances if r > current_price]
                entry = valid_resistances[0] if valid_resistances else max(resistances)
                stop_loss = entry * 1.03
                # Take profit en soportes progresivos
                take_profit = []
                for i, sup in enumerate(supports):
                    if sup < entry and len(take_profit) < 3:
                        take_profit.append(sup)
                
                # Si no hay suficientes TPs, calcular basados en ATR
                while len(take_profit) < 3:
                    last_tp = take_profit[-1] if take_profit else entry
                    take_profit.append(last_tp - (current_atr * (1 + len(take_profit) * 0.5)))
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profit[:3]],
                'supports': [float(s) for s in supports[:4]],
                'resistances': [float(r) for r in resistances[:4]],
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02, current_price * 1.04, current_price * 1.06],
                'supports': [current_price * 0.95, current_price * 0.93, current_price * 0.91, current_price * 0.89],
                'resistances': [current_price * 1.05, current_price * 1.07, current_price * 1.09, current_price * 1.11],
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
        """Verificar condiciones multi-timeframe obligatorias - CORREGIDO"""
        try:
            # Para temporalidades 12h, 1D, 1W no es obligatorio Multi-Timeframe
            if interval in ['12h', '1D', '1W']:
                return False
                
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False
            
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            # Verificar temporalidad menor
            menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
            if menor_df is not None and len(menor_df) > 10:
                menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                menor_no_trade = not menor_trend['no_trade_zones'][-1]
                
                if signal_type == 'LONG':
                    menor_direction = menor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']
                else:
                    menor_direction = menor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
            else:
                menor_direction = False
                menor_no_trade = False
            
            if signal_type == 'LONG':
                # TF Mayor: Alcista o Neutral
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                # TF Medio: Alcista
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'
                
                return mayor_ok and media_ok and menor_direction and menor_no_trade
                
            elif signal_type == 'SHORT':
                # TF Mayor: Bajista o Neutral
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                # TF Medio: Bajista
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'
                
                return mayor_ok and media_ok and menor_direction and menor_no_trade
            
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

    def detect_divergence(self, price, indicator, lookback=14, confirmation_candles=4):
        """Detectar divergencias entre precio e indicador con confirmación de velas"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-confirmation_candles):
            price_window = price[i-lookback:i+1]
            indicator_window = indicator[i-lookback:i+1]
            
            # Buscar mínimo en precio y máximo en indicador (divergencia alcista)
            if (price[i] < np.min(price_window[:-1]) and 
                indicator[i] > np.max(indicator_window[:-1])):
                # Confirmar en las siguientes velas
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and price[i+j] < price[i]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            bullish_div[i+j] = True
            
            # Buscar máximo en precio y mínimo en indicador (divergencia bajista)
            if (price[i] > np.max(price_window[:-1]) and 
                indicator[i] < np.min(indicator_window[:-1])):
                # Confirmar en las siguientes velas
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and price[i+j] > price[i]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            bearish_div[i+j] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def check_breakout(self, high, low, close, support, resistance, confirmation_candles=1):
        """Detectar rupturas de tendencia con confirmación"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n-confirmation_candles):
            # Ruptura alcista
            if close[i] > resistance[i] and high[i] > high[i-1]:
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and close[i+j] < resistance[i+j]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            breakout_up[i+j] = True
            
            # Ruptura bajista
            if close[i] < support[i] and low[i] < low[i-1]:
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and close[i+j] > support[i+j]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            breakout_down[i+j] = True
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover(self, plus_di, minus_di, lookback=3, confirmation_candles=1):
        """Detectar cruces de +DI y -DI con confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-confirmation_candles):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and plus_di[i+j] <= minus_di[i+j]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            di_cross_bullish[i+j] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and minus_di[i+j] <= plus_di[i+j]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            di_cross_bearish[i+j] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

    def check_ma_crossover(self, ma_fast, ma_slow, confirmation_candles=1):
        """Detectar cruce de medias móviles con confirmación"""
        n = len(ma_fast)
        ma_cross_up = np.zeros(n, dtype=bool)
        ma_cross_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n-confirmation_candles):
            # Cruce alcista: MA rápida cruza por encima de MA lenta
            if (ma_fast[i] > ma_slow[i] and 
                ma_fast[i-1] <= ma_slow[i-1]):
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and ma_fast[i+j] <= ma_slow[i+j]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            ma_cross_up[i+j] = True
            
            # Cruce bajista: MA rápida cruza por debajo de MA lenta
            if (ma_fast[i] < ma_slow[i] and 
                ma_fast[i-1] >= ma_slow[i-1]):
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and ma_fast[i+j] >= ma_slow[i+j]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            ma_cross_down[i+j] = True
        
        return ma_cross_up.tolist(), ma_cross_down.tolist()

    def check_macd_crossover(self, macd, signal, confirmation_candles=1):
        """Detectar cruce de MACD con confirmación"""
        n = len(macd)
        macd_cross_up = np.zeros(n, dtype=bool)
        macd_cross_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n-confirmation_candles):
            # Cruce alcista: MACD cruza por encima de la señal
            if (macd[i] > signal[i] and 
                macd[i-1] <= signal[i-1]):
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and macd[i+j] <= signal[i+j]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            macd_cross_up[i+j] = True
            
            # Cruce bajista: MACD cruza por debajo de la señal
            if (macd[i] < signal[i] and 
                macd[i-1] >= signal[i-1]):
                confirmation = True
                for j in range(1, confirmation_candles+1):
                    if i+j < n and macd[i+j] >= signal[i+j]:
                        confirmation = False
                        break
                if confirmation:
                    for j in range(confirmation_candles+1):
                        if i+j < n:
                            macd_cross_down[i+j] = True
        
        return macd_cross_up.tolist(), macd_cross_down.tolist()

    def check_adx_slope(self, adx, lookback=3):
        """Verificar pendiente positiva del ADX"""
        n = len(adx)
        adx_slope_positive = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            if i >= lookback:
                current_adx = adx[i]
                previous_adx = np.mean(adx[i-lookback:i])
                if current_adx > previous_adx and current_adx > 25:
                    adx_slope_positive[i] = True
        
        return adx_slope_positive.tolist()

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

    def detect_chart_patterns(self, high, low, close, lookback=50, confirmation_candles=7):
        """Detectar patrones de chartismo con confirmación"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback + confirmation_candles, n-confirmation_candles):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            window_close = close[i-lookback:i+1]
            
            # Hombro Cabeza Hombro (simplificado)
            if len(window_high) >= 20:
                max_idx = np.argmax(window_high)
                if (max_idx > 5 and max_idx < len(window_high)-5 and
                    window_high[max_idx-3] < window_high[max_idx] and
                    window_high[max_idx+3] < window_high[max_idx]):
                    # Confirmar en las siguientes velas
                    confirmation = True
                    for j in range(1, confirmation_candles+1):
                        if i+j < n and close[i+j] > window_high[max_idx]:
                            confirmation = False
                            break
                    if confirmation:
                        for j in range(confirmation_candles+1):
                            if i+j < n:
                                patterns['head_shoulders'][i+j] = True
            
            # Doble Techo
            if len(window_high) >= 15:
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 2:
                    last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                    if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                        # Confirmar en las siguientes velas
                        confirmation = True
                        for j in range(1, confirmation_candles+1):
                            if i+j < n and close[i+j] > last_two_peaks[1][1]:
                                confirmation = False
                                break
                        if confirmation:
                            for j in range(confirmation_candles+1):
                                if i+j < n:
                                    patterns['double_top'][i+j] = True
            
            # Doble Fondo
            if len(window_low) >= 15:
                troughs = []
                for j in range(1, len(window_low)-1):
                    if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                        troughs.append((j, window_low[j]))
                
                if len(troughs) >= 2:
                    last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                    if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                        # Confirmar en las siguientes velas
                        confirmation = True
                        for j in range(1, confirmation_candles+1):
                            if i+j < n and close[i+j] < last_two_troughs[1][1]:
                                confirmation = False
                                break
                        if confirmation:
                            for j in range(confirmation_candles+1):
                                if i+j < n:
                                    patterns['double_bottom'][i+j] = True
        
        return patterns

    def calculate_volume_anomaly(self, volume, period=20, std_multiplier=2):
        """Calcular anomalías de volumen - NUEVO INDICADOR"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            
            for i in range(period, n):
                # Media móvil exponencial de volumen
                ema_volume = self.calculate_ema(volume[:i+1], period)
                current_ema = ema_volume[i] if i < len(ema_volume) else volume[i]
                
                # Desviación estándar
                window = volume[max(0, i-period+1):i+1]
                std_volume = np.std(window) if len(window) > 1 else 0
                
                # Ratio volumen actual vs EMA
                if current_ema > 0:
                    volume_ratio[i] = volume[i] / current_ema
                else:
                    volume_ratio[i] = 1
                
                # Detectar anomalía (> 2σ)
                if volume_ratio[i] > 1 + (std_multiplier * (std_volume / current_ema if current_ema > 0 else 0)):
                    volume_anomaly[i] = True
                
                # Detectar clusters (múltiples anomalías en 5-10 periodos)
                if i >= 10:
                    recent_anomalies = volume_anomaly[max(0, i-9):i+1]
                    if np.sum(recent_anomalies) >= 3:  # Al menos 3 anomalías en 10 periodos
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ema': ema_volume.tolist() if 'ema_volume' in locals() else [0] * n
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly: {e}")
            n = len(volume)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ema': [0] * n
            }

    def get_weights_by_timeframe(self, interval):
        """Obtener pesos de los indicadores según la temporalidad - NUEVO SISTEMA"""
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            return {
                'long': {
                    'multi_timeframe': 30,
                    'trend_strength': 25,
                    'ma_crossover': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'bollinger_bands': 8,
                    'macd_crossover': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'multi_timeframe': 30,
                    'trend_strength': 25,
                    'ma_crossover': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'bollinger_bands': 8,
                    'macd_crossover': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                }
            }
        elif interval in ['12h', '1D']:
            return {
                'long': {
                    'whale_signal': 30,
                    'trend_strength': 25,
                    'ma_crossover': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'bollinger_bands': 8,
                    'macd_crossover': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'whale_signal': 30,
                    'trend_strength': 25,
                    'ma_crossover': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'bollinger_bands': 8,
                    'macd_crossover': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                }
            }
        else:  # 1W
            return {
                'long': {
                    'trend_strength': 55,
                    'ma_crossover': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'bollinger_bands': 8,
                    'macd_crossover': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'trend_strength': 55,
                    'ma_crossover': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'bollinger_bands': 8,
                    'macd_crossover': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                }
            }

    def evaluate_signal_conditions_corrected(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con PESOS CORREGIDOS según nuevas reglas"""
        weights = self.get_weights_by_timeframe(interval)
        
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
        
        # Condiciones LONG
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_long', False)
        elif interval in ['12h', '1D']:
            conditions['long']['whale_signal']['value'] = (
                data['whale_pump'][current_idx] > 20 and
                data['confirmed_buy'][current_idx]
            )
        
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        conditions['long']['bollinger_bands']['value'] = data.get('bollinger_conditions_long', False)
        
        # Verificar cruce de medias (solo si está en cruce actual)
        conditions['long']['ma_crossover']['value'] = (
            current_idx < len(data['ma_cross_up']) and 
            data['ma_cross_up'][current_idx]
        )
        
        # Verificar cruce DMI (solo si está en cruce actual)
        conditions['long']['di_crossover']['value'] = (
            current_idx < len(data['di_cross_bullish']) and 
            data['di_cross_bullish'][current_idx]
        )
        
        # Verificar ADX con pendiente positiva
        conditions['long']['adx_slope']['value'] = (
            current_idx < len(data['adx_slope_positive']) and 
            data['adx_slope_positive'][current_idx]
        )
        
        # Verificar cruce MACD (solo si está en cruce actual)
        conditions['long']['macd_crossover']['value'] = (
            current_idx < len(data['macd_cross_up']) and 
            data['macd_cross_up'][current_idx]
        )
        
        conditions['long']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bullish_divergence']) and 
            data['rsi_bullish_divergence'][current_idx]
        )
        
        conditions['long']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bullish_divergence']) and 
            data['rsi_maverick_bullish_divergence'][current_idx]
        )
        
        conditions['long']['chart_pattern']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_flag'][current_idx]
        )
        
        conditions['long']['breakout']['value'] = (
            current_idx < len(data['breakout_up']) and 
            data['breakout_up'][current_idx]
        )
        
        conditions['long']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly']) and 
            data['volume_anomaly'][current_idx] and
            data['volume_clusters'][current_idx]
        )
        
        # Condiciones SHORT
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        elif interval in ['12h', '1D']:
            conditions['short']['whale_signal']['value'] = (
                data['whale_dump'][current_idx] > 20 and
                data['confirmed_sell'][current_idx]
            )
        
        conditions['short']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            not data['no_trade_zones'][current_idx]
        )
        
        conditions['short']['bollinger_bands']['value'] = data.get('bollinger_conditions_short', False)
        
        # Verificar cruce de medias (solo si está en cruce actual)
        conditions['short']['ma_crossover']['value'] = (
            current_idx < len(data['ma_cross_down']) and 
            data['ma_cross_down'][current_idx]
        )
        
        # Verificar cruce DMI (solo si está en cruce actual)
        conditions['short']['di_crossover']['value'] = (
            current_idx < len(data['di_cross_bearish']) and 
            data['di_cross_bearish'][current_idx]
        )
        
        # Verificar ADX con pendiente positiva
        conditions['short']['adx_slope']['value'] = (
            current_idx < len(data['adx_slope_positive']) and 
            data['adx_slope_positive'][current_idx]
        )
        
        # Verificar cruce MACD (solo si está en cruce actual)
        conditions['short']['macd_crossover']['value'] = (
            current_idx < len(data['macd_cross_down']) and 
            data['macd_cross_down'][current_idx]
        )
        
        conditions['short']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bearish_divergence']) and 
            data['rsi_bearish_divergence'][current_idx]
        )
        
        conditions['short']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bearish_divergence']) and 
            data['rsi_maverick_bearish_divergence'][current_idx]
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
        
        conditions['short']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly']) and 
            data['volume_anomaly'][current_idx] and
            data['volume_clusters'][current_idx]
        )
        
        return conditions

    def get_condition_description(self, condition_key):
        """Obtener descripción de condición"""
        descriptions = {
            'multi_timeframe': 'Condiciones Multi-TF obligatorias',
            'trend_strength': 'Fuerza tendencia favorable',
            'whale_signal': 'Señal ballenas confirmada',
            'bollinger_bands': 'Bandas de Bollinger',
            'ma_crossover': 'Cruce Medias Móviles',
            'di_crossover': 'Cruce DMI',
            'adx_slope': 'ADX con pendiente positiva',
            'macd_crossover': 'Cruce MACD',
            'rsi_traditional_divergence': 'RSI Tradicional Divergencia',
            'rsi_maverick_divergence': 'RSI Maverick Divergencia',
            'chart_pattern': 'Patrones Chartistas',
            'breakout': 'Ruptura Confirmada',
            'volume_anomaly': 'Anomalía de Volumen'
        }
        return descriptions.get(condition_key, condition_key)

    def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal basada en condiciones ponderadas - NUEVO SISTEMA"""
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
        
        # Score mínimo ajustado según MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:
            min_score = 65 if ma200_condition == 'below' else 70
        
        if base_score < min_score:
            return 0, []
        
        return min(base_score, 100), fulfilled_conditions

    def calculate_winrate(self, symbol, interval):
        """Calcular winrate basado en datos históricos"""
        try:
            cache_key = f"winrate_{symbol}_{interval}"
            if cache_key in self.winrate_data:
                return self.winrate_data[cache_key]
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return 65.0
            
            total_signals = 0
            successful_signals = 0
            
            for i in range(20, len(df)-5):
                price_change = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
                
                if abs(price_change) > 2:
                    total_signals += 1
                    if price_change > 0:
                        successful_signals += 1
            
            winrate = (successful_signals / total_signals * 100) if total_signals > 0 else 65.0
            winrate = max(50.0, min(85.0, winrate))
            
            self.winrate_data[cache_key] = winrate
            return winrate
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol} {interval}: {e}")
            return 65.0

    def get_cmc_volume_data(self):
        """Obtener datos de volumen de CoinMarketCap"""
        cache_key = "cmc_volume_data"
        current_time = datetime.now()
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (current_time - timestamp).seconds < 300:  # 5 minutos de cache
                return cached_data
        
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': CMC_API_KEY,
        }
        
        try:
            url = f"{CMC_BASE_URL}/v1/cryptocurrency/listings/latest"
            parameters = {
                'start': '1',
                'limit': '100',
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=parameters, timeout=30)
            data = response.json()
            
            volume_data = {}
            for crypto in data['data']:
                symbol = crypto['symbol'] + '-USDT'
                volume_24h = crypto['quote']['USD']['volume_24h']
                percent_change_24h = crypto['quote']['USD']['percent_change_24h']
                current_price = crypto['quote']['USD']['price']
                
                volume_data[symbol] = {
                    'volume_24h': volume_24h,
                    'percent_change_24h': percent_change_24h,
                    'current_price': current_price,
                    'timestamp': current_time
                }
            
            self.cache[cache_key] = (volume_data, current_time)
            
            # Actualizar datos históricos
            for symbol, data in volume_data.items():
                if symbol not in self.volume_historical_data:
                    self.volume_historical_data[symbol] = []
                
                self.volume_historical_data[symbol].append({
                    'volume': data['volume_24h'],
                    'timestamp': current_time
                })
                
                # Mantener solo últimos 7 días
                week_ago = current_time - timedelta(days=7)
                self.volume_historical_data[symbol] = [
                    d for d in self.volume_historical_data[symbol] 
                    if d['timestamp'] > week_ago
                ]
            
            return volume_data
            
        except Exception as e:
            print(f"Error obteniendo datos de CoinMarketCap: {e}")
            return {}

    def detect_cmc_volume_spike(self, symbol):
        """Detectar picos de volumen usando datos de CoinMarketCap"""
        try:
            volume_data = self.get_cmc_volume_data()
            
            if not volume_data or symbol not in volume_data:
                return None
            
            current_data = volume_data[symbol]
            current_volume = current_data['volume_24h']
            percent_change = current_data['percent_change_24h']
            
            # Calcular promedio histórico de 7 días
            if symbol in self.volume_historical_data and len(self.volume_historical_data[symbol]) > 0:
                historical_volumes = [d['volume'] for d in self.volume_historical_data[symbol]]
                historical_avg = np.mean(historical_volumes)
            else:
                # Si no hay histórico, asumir que el volumen actual es normal
                historical_avg = current_volume
            
            # Evitar división por cero
            if historical_avg <= 0:
                return None
            
            # Calcular ratio de volumen
            volume_ratio = current_volume / historical_avg
            
            # Si el volumen es 3 veces mayor al histórico, es un pico
            if volume_ratio >= 3.0:
                # Determinar dirección basada en cambio de precio
                if percent_change > 0:
                    direction = "COMPRA"
                else:
                    direction = "VENTA"
                
                # Formatear volumen en millones
                volume_millions = current_volume / 1000000
                formatted_volume = f"{volume_millions:,.1f}"
                
                # Verificar cache para evitar duplicados
                cache_key = f"volume_spike_{symbol}_{direction}_{int(current_volume/1000000)}"
                if cache_key in self.volume_spike_cache:
                    last_alert_time = self.volume_spike_cache[cache_key]
                    if (datetime.now() - last_alert_time).seconds < 7200:  # 2 horas
                        return None
                
                self.volume_spike_cache[cache_key] = datetime.now()
                
                return {
                    'symbol': symbol,
                    'current_volume': current_volume,
                    'volume_millions': volume_millions,
                    'formatted_volume': formatted_volume,
                    'historical_avg': historical_avg,
                    'volume_ratio': volume_ratio,
                    'percent_change': percent_change,
                    'direction': direction,
                    'current_price': current_data['current_price'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            return None
            
        except Exception as e:
            print(f"Error detectando pico de volumen para {symbol}: {e}")
            return None

    def generate_cmc_volume_alerts(self):
        """Generar alertas de volumen anómalo de CoinMarketCap"""
        alerts = []
        
        for symbol in CRYPTO_SYMBOLS:
            try:
                volume_spike = self.detect_cmc_volume_spike(symbol)
                if volume_spike:
                    alerts.append(volume_spike)
                
                # Pequeña pausa para no sobrecargar la API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error procesando {symbol} para alertas de volumen: {e}")
                continue
        
        return alerts

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - CON NUEVO SISTEMA DE PESOS"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular todos los indicadores
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di, confirmation_candles=1)
            adx_slope_positive = self.check_adx_slope(adx)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick, confirmation_candles=4)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional, confirmation_candles=4)
            
            breakout_up, breakout_down = self.check_breakout(high, low, close, whale_data['support'], whale_data['resistance'], confirmation_candles=1)
            chart_patterns = self.detect_chart_patterns(high, low, close, confirmation_candles=7)
            
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # Cruce de medias
            ma_cross_up, ma_cross_down = self.check_ma_crossover(ma_9, ma_21, confirmation_candles=1)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_up, macd_cross_down = self.check_macd_crossover(macd, macd_signal, confirmation_candles=1)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Verificar condiciones de Bollinger
            bollinger_conditions_long = self.check_bollinger_conditions_corrected(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions_corrected(df, interval, 'SHORT')
            
            # Indicador de volumen
            volume_anomaly_data = self.calculate_volume_anomaly(volume)
            
            current_idx = -1
            
            # Verificar condiciones multi-timeframe obligatorias
            multi_timeframe_long = self.check_multi_timeframe_obligatory(symbol, interval, 'LONG')
            multi_timeframe_short = self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT')
            
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
                'ma_cross_up': ma_cross_up,
                'ma_cross_down': ma_cross_down,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'macd_cross_up': macd_cross_up,
                'macd_cross_down': macd_cross_down,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'volume_anomaly': volume_anomaly_data['volume_anomaly'],
                'volume_clusters': volume_anomaly_data['volume_clusters'],
                'volume_ratio': volume_anomaly_data['volume_ratio'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short
            }
            
            conditions = self.evaluate_signal_conditions_corrected(analysis_data, current_idx, interval, adx_threshold)
            
            # Calcular condición MA200
            current_ma200 = ma_200[current_idx] if current_idx < len(ma_200) else 0
            current_price = close[current_idx]
            ma200_condition = 'above' if current_price > current_ma200 else 'below'

            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', ma200_condition)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', ma200_condition)
            
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= 65:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= 65:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # Calcular winrate
            winrate = self.calculate_winrate(symbol, interval)
            
            current_price = float(close[current_idx])
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # Registrar señal activa si es válida
            if signal_type in ['LONG', 'SHORT'] and signal_score >= 65:
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
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'winrate': float(winrate),
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
                    'adx_slope_positive': adx_slope_positive[-50:],
                    'di_cross_bullish': di_cross_bullish[-50:],
                    'di_cross_bearish': di_cross_bearish[-50:],
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
                    'ma_cross_up': ma_cross_up[-50:],
                    'ma_cross_down': ma_cross_down[-50:],
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'macd_cross_up': macd_cross_up[-50:],
                    'macd_cross_down': macd_cross_down[-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'volume_anomaly': volume_anomaly_data['volume_anomaly'][-50:],
                    'volume_clusters': volume_anomaly_data['volume_clusters'][-50:],
                    'volume_ratio': volume_anomaly_data['volume_ratio'][-50:],
                    'volume_ema': volume_anomaly_data['volume_ema'][-50:],
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
        """Crear señal vacía en caso de error"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'signal': 'NEUTRAL',
            'signal_score': 0,
            'winrate': 65.0,
            'entry': 0,
            'stop_loss': 0,
            'take_profit': [0, 0, 0],
            'supports': [0, 0, 0, 0],
            'resistances': [0, 0, 0, 0],
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
        """Generar alertas de trading"""
        alerts = []
        
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:12]:
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
                            'winrate': signal_data['winrate'],
                            'entry': signal_data['entry'],
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'],
                            'leverage': optimal_leverage,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'supports': signal_data['supports'],
                            'resistances': signal_data['resistances'],
                            'ma200_condition': signal_data.get('ma200_condition', 'below'),
                            'multi_timeframe_ok': signal_data.get('multi_timeframe_ok', False)
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

def generate_telegram_chart_strategy1(alert_data):
    """Generar gráfico para Telegram - Estrategia 1 (Multi-Timeframe)"""
    try:
        symbol = alert_data['symbol']
        interval = alert_data['interval']
        
        # Obtener datos
        signal_data = indicator.generate_signals_improved(symbol, interval)
        if not signal_data or 'data' not in signal_data:
            return None
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(12, 16))
        
        # 1. Gráfico de velas japonesas con Bandas de Bollinger y Medias Móviles
        ax1 = plt.subplot(8, 1, 1)
        if signal_data['data']:
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
            
            # Bandas de Bollinger (transparentes)
            if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
                bb_upper = signal_data['indicators']['bb_upper']
                bb_lower = signal_data['indicators']['bb_lower']
                ax1.fill_between(dates[-len(bb_upper):], bb_lower, bb_upper, 
                               color='orange', alpha=0.1, label='Bollinger Bands')
            
            # Medias móviles
            if 'indicators' in signal_data:
                if 'ma_9' in signal_data['indicators']:
                    ax1.plot(dates[-len(signal_data['indicators']['ma_9']):], 
                           signal_data['indicators']['ma_9'], 'blue', linewidth=1, label='MA 9')
                if 'ma_21' in signal_data['indicators']:
                    ax1.plot(dates[-len(signal_data['indicators']['ma_21']):], 
                           signal_data['indicators']['ma_21'], 'red', linewidth=1, label='MA 21')
                if 'ma_50' in signal_data['indicators']:
                    ax1.plot(dates[-len(signal_data['indicators']['ma_50']):], 
                           signal_data['indicators']['ma_50'], 'green', linewidth=1, label='MA 50')
                if 'ma_200' in signal_data['indicators']:
                    ax1.plot(dates[-len(signal_data['indicators']['ma_200']):], 
                           signal_data['indicators']['ma_200'], 'purple', linewidth=2, label='MA 200')
            
            # Soporte y resistencia
            for i, support in enumerate(alert_data.get('supports', [])[:4]):
                ax1.axhline(y=support, color='blue', linestyle='--', alpha=0.5, 
                          label=f'S{i+1}' if i == 0 else "")
            
            for i, resistance in enumerate(alert_data.get('resistances', [])[:4]):
                ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5,
                          label=f'R{i+1}' if i == 0 else "")
        
        ax1.set_title(f'{symbol} - {interval}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precio')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(8, 1, 2)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax2.plot(adx_dates, signal_data['indicators']['adx'], 
                    'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Indicador de Volumen con Anomalías
        ax3 = plt.subplot(8, 1, 3)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_ratio']):]
            volume_values = signal_data['data'][-len(signal_data['indicators']['volume_ratio']):]
            
            # Colores según compra/venta
            volume_colors = []
            for i, vol in enumerate(volume_values):
                if i > 0:
                    if vol['close'] >= vol['open']:
                        volume_colors.append('green')
                    else:
                        volume_colors.append('red')
                else:
                    volume_colors.append('gray')
            
            ax3.bar(volume_dates, [v['volume'] for v in volume_values], 
                   color=volume_colors, alpha=0.7, label='Volumen')
            
            # Anomalías de volumen
            anomaly_dates = []
            anomaly_values = []
            for i, date in enumerate(volume_dates):
                if signal_data['indicators']['volume_anomaly'][i]:
                    anomaly_dates.append(date)
                    anomaly_values.append(volume_values[i]['volume'])
            
            if anomaly_dates:
                ax3.scatter(anomaly_dates, anomaly_values, color='orange', s=50, 
                           marker='o', label='Anomalías', zorder=5)
        
        ax3.set_ylabel('Volumen')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick (columnas)
        ax4 = plt.subplot(8, 1, 4)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            # Convertir a columnas
            width = (trend_dates[1] - trend_dates[0]).total_seconds() * 0.8 if len(trend_dates) > 1 else 86400 * 0.8
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax4.bar(trend_dates[i], trend_strength[i], color=color, 
                       alpha=0.7, width=width)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7)
                ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        
        ax4.set_ylabel('Fuerza Tendencia')
        ax4.grid(True, alpha=0.3)
        
        # 5. Indicador de Ballenas (solo para 12h y 1D)
        if interval in ['12h', '1D']:
            ax5 = plt.subplot(8, 1, 5)
            if 'indicators' in signal_data:
                whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
                
                # Convertir a columnas
                width = (whale_dates[1] - whale_dates[0]).total_seconds() * 0.4 if len(whale_dates) > 1 else 86400 * 0.4
                ax5.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                       color='green', alpha=0.7, width=width, label='Compra Ballenas')
                ax5.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                       color='red', alpha=0.7, width=width, label='Venta Ballenas')
            
            ax5.set_ylabel('Ballenas')
            ax5.legend(loc='upper left', fontsize=8)
            ax5.grid(True, alpha=0.3)
        
        # 6. RSI Maverick
        ax6 = plt.subplot(8, 1, 6)
        if 'indicators' in signal_data:
            rsi_maverick_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax6.plot(rsi_maverick_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        ax6.set_ylabel('RSI Maverick')
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. RSI Tradicional
        ax7 = plt.subplot(8, 1, 7)
        if 'indicators' in signal_data:
            rsi_traditional_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax7.plot(rsi_traditional_dates, signal_data['indicators']['rsi_traditional'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax7.set_ylabel('RSI Tradicional')
        ax7.legend(loc='upper left', fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 8. MACD con Histograma (columnas)
        ax8 = plt.subplot(8, 1, 8)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            
            # Convertir histograma a columnas
            macd_histogram = signal_data['indicators']['macd_histogram']
            colors = ['green' if x > 0 else 'red' for x in macd_histogram]
            width = (macd_dates[1] - macd_dates[0]).total_seconds() * 0.8 if len(macd_dates) > 1 else 86400 * 0.8
            
            ax8.bar(macd_dates, macd_histogram, color=colors, alpha=0.7, width=width, label='Histograma')
            ax8.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax8.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax8.set_ylabel('MACD')
        ax8.legend(loc='upper left', fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando gráfico para Telegram (estrategia 1): {e}")
        return None

def generate_telegram_chart_strategy2(alert_data):
    """Generar gráfico para Telegram - Estrategia 2 (Volumen Anómalo)"""
    try:
        symbol = alert_data['symbol']
        
        # Obtener datos de KuCoin para el gráfico
        df = indicator.get_kucoin_data(symbol, '1h', 50)
        if df is None or len(df) < 20:
            return None
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(12, 8))
        
        # 1. Gráfico de velas japonesas
        ax1 = plt.subplot(3, 1, 1)
        dates = df['timestamp'].values[-50:]
        opens = df['open'].values[-50:]
        highs = df['high'].values[-50:]
        lows = df['low'].values[-50:]
        closes = df['close'].values[-50:]
        
        # Graficar velas
        for i in range(len(dates)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
        
        # Bandas de Bollinger
        bb_upper, bb_middle, bb_lower = indicator.calculate_bollinger_bands(closes)
        ax1.fill_between(dates[-len(bb_upper):], bb_lower, bb_upper, 
                       color='orange', alpha=0.1, label='Bollinger Bands')
        
        # Medias móviles
        ma_9 = indicator.calculate_sma(closes, 9)
        ma_21 = indicator.calculate_sma(closes, 21)
        ma_50 = indicator.calculate_sma(closes, 50)
        
        ax1.plot(dates[-len(ma_9):], ma_9, 'blue', linewidth=1, label='MA 9')
        ax1.plot(dates[-len(ma_21):], ma_21, 'red', linewidth=1, label='MA 21')
        ax1.plot(dates[-len(ma_50):], ma_50, 'green', linewidth=1, label='MA 50')
        
        ax1.set_title(f'{symbol} - Anomalía de Volumen', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precio')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(3, 1, 2)
        adx, plus_di, minus_di = indicator.calculate_adx(highs, lows, closes)
        adx_dates = dates[-len(adx):]
        
        ax2.plot(adx_dates, adx, 'black', linewidth=2, label='ADX')
        ax2.plot(adx_dates, plus_di, 'green', linewidth=1, label='+DI')
        ax2.plot(adx_dates, minus_di, 'red', linewidth=1, label='-DI')
        ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Gráfico de anomalía de compras/ventas
        ax3 = plt.subplot(3, 1, 3)
        
        # Obtener datos históricos para el símbolo
        if symbol in indicator.volume_historical_data:
            historical_data = indicator.volume_historical_data[symbol]
            if historical_data:
                # Preparar datos para el gráfico
                hist_dates = [d['timestamp'] for d in historical_data]
                hist_volumes = [d['volume'] for d in historical_data]
                
                # Volumen actual
                current_volume = alert_data['current_volume']
                current_time = datetime.now()
                
                # Gráfico de líneas para volumen histórico
                ax3.plot(hist_dates, hist_volumes, 'blue', linewidth=1, label='Volumen Histórico', alpha=0.5)
                
                # Línea para promedio histórico
                hist_avg = alert_data['historical_avg']
                ax3.axhline(y=hist_avg, color='green', linestyle='--', alpha=0.7, 
                          label=f'Promedio: {hist_avg/1000000:,.1f}M')
                
                # Punto para volumen actual
                ax3.scatter([current_time], [current_volume], color='red', s=100, 
                          marker='o', label=f'Actual: {current_volume/1000000:,.1f}M', zorder=5)
                
                # Línea de umbral (3x promedio)
                threshold = hist_avg * 3
                ax3.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                          label=f'Umbral: {threshold/1000000:,.1f}M')
        
        ax3.set_ylabel('Volumen (USD)')
        ax3.set_xlabel('Fecha')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Añadir anotación con dirección
        direction_text = f"DIRECCIÓN: {alert_data['direction']}\n"
        direction_text += f"Cambio 24h: {alert_data['percent_change']:.2f}%\n"
        direction_text += f"Ratio: {alert_data['volume_ratio']:.1f}x promedio"
        
        ax3.text(0.02, 0.98, direction_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando gráfico para Telegram (estrategia 2): {e}")
        return None

async def send_telegram_alert_strategy1(alert_data):
    """Enviar alerta de la estrategia 1 (Multi-Timeframe) por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        
        # Mensaje conciso con condiciones cumplidas
        message = f"""
🚨 ALERTA MULTI-TIMEFRAME - {alert_data['signal']} 🚨

📈 {alert_data['symbol']} ({risk_classification})
⏰ {alert_data['interval']} | Score: {alert_data['score']:.1f}%

💰 Entrada: ${alert_data['entry']:.6f}
📊 Precio: ${alert_data['current_price']:.6f}

✅ Condiciones:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])])}

📈 Apalancamiento: x{alert_data['leverage']}
        """
        
        # Generar imagen
        img_buffer = generate_telegram_chart_strategy1(alert_data)
        
        if img_buffer:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID, 
                photo=img_buffer,
                caption=message
            )
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=message
            )
        
        print(f"Alerta estrategia 1 enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram (estrategia 1): {e}")

async def send_telegram_alert_strategy2(alert_data):
    """Enviar alerta de la estrategia 2 (Volumen Anómalo) por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_data['direction'] == 'COMPRA':
            message = f"""
🚨Alerta de COMPRA Atipica 🚨
Se acaba de ingresar o comprar {alert_data['symbol']} en {alert_data['formatted_volume']} millones de USDT, volumen atípico, revisar LONG
            """
        else:
            message = f"""
🚨Alerta de VENTA Atipica 🚨
Se vendieron {alert_data['formatted_volume']} millones de USDT en {alert_data['symbol']}, volumen atípico, revisar SHORT
            """
        
        # Generar imagen
        img_buffer = generate_telegram_chart_strategy2(alert_data)
        
        if img_buffer:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID, 
                photo=img_buffer,
                caption=message
            )
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=message
            )
        
        print(f"Alerta estrategia 2 enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram (estrategia 2): {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    intraday_intervals = ['15m', '30m', '1h', '2h']
    swing_intervals = ['4h', '8h', '12h', '1D', '1W']
    
    intraday_last_check = datetime.now()
    swing_last_check = datetime.now()
    volume_last_check = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            # Estrategia 1 - Multi-Timeframe
            if (current_time - intraday_last_check).seconds >= 60:
                print("Verificando alertas intradía...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in intraday_intervals:
                        asyncio.run(send_telegram_alert_strategy1(alert))
                
                intraday_last_check = current_time
            
            if (current_time - swing_last_check).seconds >= 300:
                print("Verificando alertas swing...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in swing_intervals:
                        asyncio.run(send_telegram_alert_strategy1(alert))
                
                swing_last_check = current_time
            
            # Estrategia 2 - Volumen Anómalo
            if (current_time - volume_last_check).seconds >= 300:  # Cada 5 minutos
                print("Verificando alertas de volumen anómalo...")
                
                volume_alerts = indicator.generate_cmc_volume_alerts()
                for alert in volume_alerts:
                    asyncio.run(send_telegram_alert_strategy2(alert))
                
                volume_last_check = current_time
            
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
        
        signal_data = indicator.generate_signals_improved(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, volume_filter, leverage
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
                signal_data = indicator.generate_signals_improved(symbol, interval, di_period, adx_threshold)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones basadas en indicadores reales
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
        alerts = indicator.generate_scalping_alerts()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/volume_anomaly_signals')
def get_volume_anomaly_signals():
    """Endpoint para obtener señales de volumen anómalo"""
    try:
        alerts = indicator.generate_cmc_volume_alerts()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/volume_anomaly_signals: {e}")
        return jsonify({'alerts': []})

@app.route('/api/winrate')
def get_winrate():
    """Endpoint para obtener winrate del sistema"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        winrate = indicator.calculate_winrate(symbol, interval)
        return jsonify({'symbol': symbol, 'interval': interval, 'winrate': winrate})
        
    except Exception as e:
        print(f"Error en /api/winrate: {e}")
        return jsonify({'winrate': 65.0})

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
            
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit'][:3]):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
            
            # Mostrar 4 soportes y resistencias
            for i, support in enumerate(signal_data['supports'][:4]):
                ax1.axhline(y=support, color='orange', linestyle=':', alpha=0.5, label=f'S{i+1}' if i == 0 else "")
            
            for i, resistance in enumerate(signal_data['resistances'][:4]):
                ax1.axhline(y=resistance, color='purple', linestyle=':', alpha=0.5, label=f'R{i+1}' if i == 0 else "")
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Ballenas
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
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
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax3.plot(adx_dates, signal_data['indicators']['adx'], 
                    'white', linewidth=2, label='ADX')
            ax3.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax3.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax3.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        ax3.set_ylabel('ADX/DMI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: RSI Tradicional
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi_traditional'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax4.set_ylabel('RSI Tradicional')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: RSI Maverick
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_maverick_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax5.plot(rsi_maverick_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax5.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax5.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        ax5.set_ylabel('RSI Maverick')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: MACD
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax6.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax6.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax6.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6, label='Histograma')
            
            ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax6.set_ylabel('MACD')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: Volumen y Anomalías
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_ratio']):]
            
            # Volumen con colores según compra/venta
            volume_values = signal_data['data'][-len(signal_data['indicators']['volume_ratio']):]
            volume_colors = []
            for i, vol in enumerate(volume_values):
                if i > 0:
                    if vol['close'] >= vol['open']:
                        volume_colors.append('green')
                    else:
                        volume_colors.append('red')
                else:
                    volume_colors.append('gray')
            
            ax7.bar(volume_dates, [v['volume'] for v in volume_values], 
                   color=volume_colors, alpha=0.6, label='Volumen')
            
            # EMA de volumen
            ax7.plot(volume_dates, signal_data['indicators']['volume_ema'], 
                    'yellow', linewidth=1, label='EMA Volumen')
            
            # Anomalías de volumen
            anomaly_dates = []
            anomaly_values = []
            for i, date in enumerate(volume_dates):
                if signal_data['indicators']['volume_anomaly'][i]:
                    anomaly_dates.append(date)
                    anomaly_values.append(volume_values[i]['volume'])
            
            ax7.scatter(anomaly_dates, anomaly_values, color='red', s=50, 
                       label='Anomalías Volumen', zorder=5)
        
        ax7.set_ylabel('Volumen')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: Fuerza de Tendencia Maverick
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax8.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax8.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax8.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax8.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax8.set_ylabel('Fuerza Tendencia %')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # Información de la señal
        ax9 = plt.subplot(9, 1, 9)
        ax9.axis('off')
        
        multi_tf_info = "✅ MULTI-TIMEFRAME: Confirmado" if signal_data.get('multi_timeframe_ok') else "❌ MULTI-TIMEFRAME: No confirmado"
        ma200_info = f"MA200: {'ENCIMA' if signal_data.get('ma200_condition') == 'above' else 'DEBAJO'}"
        winrate_info = f"WINRATE: {signal_data.get('winrate', 65):.1f}%"
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        {winrate_info}
        
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
