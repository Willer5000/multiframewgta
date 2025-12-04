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
        self.volume_alert_cache = {}
    
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
            return remaining_seconds <= (15 * 60 * 0.5)
        elif interval == '30m':
            next_close = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0) + timedelta(minutes=30)
            remaining_seconds = (next_close - current_time).total_seconds()
            return remaining_seconds <= (30 * 60 * 0.5)
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

    def calculate_support_resistance(self, high, low, close, period=50):
        """Calcular 4 soportes y resistencias dinámicas"""
        n = len(close)
        
        # Inicializar arrays para soportes y resistencias
        supports = []
        resistances = []
        
        # Calcular pivots cada 50 periodos
        for i in range(0, n, period):
            if i + period <= n:
                window_high = high[i:i+period]
                window_low = low[i:i+period]
                window_close = close[i:i+period]
                
                # Pivot tradicional
                pivot = (np.max(window_high) + np.min(window_low) + window_close[-1]) / 3
                
                # Calcular niveles de soporte y resistencia
                r1 = (2 * pivot) - np.min(window_low)
                s1 = (2 * pivot) - np.max(window_high)
                r2 = pivot + (np.max(window_high) - np.min(window_low))
                s2 = pivot - (np.max(window_high) - np.min(window_low))
                r3 = np.max(window_high) + (pivot - np.min(window_low))
                s3 = np.min(window_low) - (np.max(window_high) - pivot)
                
                supports.extend([s1, s2, s3])
                resistances.extend([r1, r2, r3])
        
        # Si no hay suficientes niveles, generar algunos basados en máximos/mínimos
        if len(supports) < 4 or len(resistances) < 4:
            # Usar los últimos 100 periodos para calcular niveles
            lookback = min(100, n)
            recent_high = high[-lookback:]
            recent_low = low[-lookback:]
            
            # Encontrar niveles de soporte y resistencia naturales
            resistance_levels = []
            support_levels = []
            
            # Buscar máximos locales para resistencia
            for i in range(2, len(recent_high)-2):
                if (recent_high[i] > recent_high[i-1] and 
                    recent_high[i] > recent_high[i-2] and
                    recent_high[i] > recent_high[i+1] and
                    recent_high[i] > recent_high[i+2]):
                    resistance_levels.append(recent_high[i])
            
            # Buscar mínimos locales para soporte
            for i in range(2, len(recent_low)-2):
                if (recent_low[i] < recent_low[i-1] and 
                    recent_low[i] < recent_low[i-2] and
                    recent_low[i] < recent_low[i+1] and
                    recent_low[i] < recent_low[i+2]):
                    support_levels.append(recent_low[i])
            
            # Ordenar y seleccionar los más relevantes
            resistance_levels.sort(reverse=True)
            support_levels.sort()
            
            if len(supports) < 4:
                supports.extend(support_levels[:4-len(supports)])
            if len(resistances) < 4:
                resistances.extend(resistance_levels[:4-len(resistances)])
        
        # Asegurar al menos 4 valores cada uno
        while len(supports) < 4:
            supports.append(np.min(low[-period:]))
        while len(resistances) < 4:
            resistances.append(np.max(high[-period:]))
        
        # Ordenar y devolver
        supports = sorted(supports)[:4]
        resistances = sorted(resistances)[-4:]
        
        return supports, resistances

    def calculate_optimal_entry_exit_improved(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas mejoradas con 4 S/R"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular 4 soportes y resistencias
            supports, resistances = self.calculate_support_resistance(high, low, close)
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Para LONG: entrada en el soporte más cercano por debajo del precio
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)  # Soporte más cercano
                else:
                    entry = current_price * 0.99  # Fallback
                
                # Stop loss por debajo del siguiente soporte o usando ATR
                if len(supports) > 1:
                    stop_loss = supports[1] * 0.98 if supports[1] < entry else entry - (current_atr * 1.5)
                else:
                    stop_loss = entry - (current_atr * 1.5)
                
                # Take profit en la resistencia más cercana
                valid_resistances = [r for r in resistances if r > entry]
                if valid_resistances:
                    tp1 = min(valid_resistances) * 0.99  # Resistencia más cercana
                else:
                    tp1 = entry + (2 * (entry - stop_loss))
                
            else:  # SHORT
                # Para SHORT: entrada en la resistencia más cercana por encima del precio
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)  # Resistencia más cercana
                else:
                    entry = current_price * 1.01  # Fallback
                
                # Stop loss por encima del siguiente soporte o usando ATR
                if len(resistances) > 1:
                    stop_loss = resistances[1] * 1.02 if resistances[1] > entry else entry + (current_atr * 1.5)
                else:
                    stop_loss = entry + (current_atr * 1.5)
                
                # Take profit en el soporte más cercano
                valid_supports = [s for s in supports if s < entry]
                if valid_supports:
                    tp1 = max(valid_supports) * 1.01  # Soporte más cercano
                else:
                    tp1 = entry - (2 * (stop_loss - entry))
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp1)],
                'supports': [float(s) for s in supports],
                'resistances': [float(r) for r in resistances],
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas mejoradas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'supports': [current_price * 0.95, current_price * 0.92, current_price * 0.89, current_price * 0.86],
                'resistances': [current_price * 1.05, current_price * 1.08, current_price * 1.11, current_price * 1.14],
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
                return True
                
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

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias entre precio e indicador"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        # Detectar divergencias con lookback extendido
        for i in range(lookback, n):
            # Buscar divergencias alcistas (precio baja, indicador sube)
            if i < n - 4:  # Permitir 4 velas posteriores para confirmación
                price_min = np.min(price[i-lookback:i+1])
                indicator_min = np.min(indicator[i-lookback:i+1])
                
                # Verificar si hay divergencia alcista
                if price[i] <= price_min and indicator[i] > indicator_min:
                    # Confirmar en las próximas 4 velas
                    confirmed = True
                    for j in range(1, min(5, n-i)):
                        if indicator[i+j] < indicator[i]:
                            confirmed = False
                            break
                    
                    if confirmed:
                        bullish_div[i] = True
                
                # Buscar divergencias bajistas (precio sube, indicador baja)
                price_max = np.max(price[i-lookback:i+1])
                indicator_max = np.max(indicator[i-lookback:i+1])
                
                if price[i] >= price_max and indicator[i] < indicator_max:
                    # Confirmar en las próximas 4 velas
                    confirmed = True
                    for j in range(1, min(5, n-i)):
                        if indicator[i+j] > indicator[i]:
                            confirmed = False
                            break
                    
                    if confirmed:
                        bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def check_breakout(self, high, low, close, support, resistance):
        """Detectar rupturas de tendencia"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(2, n):
            # Ruptura alcista: cierre por encima de resistencia con volumen
            if close[i] > resistance and close[i-1] <= resistance:
                breakout_up[i] = True
            
            # Ruptura bajista: cierre por debajo de soporte con volumen
            if close[i] < support and close[i-1] >= support:
                breakout_down[i] = True
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI con confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        di_trend_bullish = np.zeros(n, dtype=bool)
        di_trend_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                di_cross_bullish[i] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
            
            # Tendencia alcista: +DI en aumento
            if plus_di[i] > np.mean(plus_di[i-lookback:i]):
                di_trend_bullish[i] = True
            
            # Tendencia bajista: -DI en aumento
            if minus_di[i] > np.mean(minus_di[i-lookback:i]):
                di_trend_bearish[i] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist(), di_trend_bullish.tolist(), di_trend_bearish.tolist()

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

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n):
            if i >= n - 7:  # Requerir 7 velas posteriores para confirmación
                continue
                
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            window_close = close[i-lookback:i+1]
            
            # Doble Techo (confirmación en 7 velas posteriores)
            if len(window_high) >= 15:
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 2:
                    last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                    peak1_idx, peak1_val = last_two_peaks[0]
                    peak2_idx, peak2_val = last_two_peaks[1]
                    
                    # Verificar que los picos estén cerca en precio
                    if abs(peak1_val - peak2_val) / peak1_val < 0.02:
                        # Verificar que después del segundo pico haya caída
                        if i - peak2_idx <= 7:
                            patterns['double_top'][i] = True
            
            # Doble Fondo (confirmación en 7 velas posteriores)
            if len(window_low) >= 15:
                troughs = []
                for j in range(1, len(window_low)-1):
                    if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                        troughs.append((j, window_low[j]))
                
                if len(troughs) >= 2:
                    last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                    trough1_idx, trough1_val = last_two_troughs[0]
                    trough2_idx, trough2_val = last_two_troughs[1]
                    
                    # Verificar que los valles estén cerca en precio
                    if abs(trough1_val - trough2_val) / trough1_val < 0.02:
                        # Verificar que después del segundo valle haya subida
                        if i - trough2_idx <= 7:
                            patterns['double_bottom'][i] = True
            
            # Bandera Alcista
            if i >= 20:
                # Buscar mástil alcista seguido de consolidación
                mast_start = i - 20
                mast_end = i - 10
                flag_start = i - 9
                flag_end = i
                
                mast_high = np.max(high[mast_start:mast_end+1])
                mast_low = np.min(low[mast_start:mast_end+1])
                mast_height = mast_high - mast_low
                
                flag_high = np.max(high[flag_start:flag_end+1])
                flag_low = np.min(low[flag_start:flag_end+1])
                flag_height = flag_high - flag_low
                
                # El mástil debe ser significativo y la bandera debe ser consolidación
                if mast_height > 0 and flag_height > 0:
                    if mast_height > flag_height * 2 and flag_height < mast_height * 0.3:
                        patterns['bullish_flag'][i] = True
            
            # Bandera Bajista
            if i >= 20:
                # Buscar mástil bajista seguido de consolidación
                mast_start = i - 20
                mast_end = i - 10
                flag_start = i - 9
                flag_end = i
                
                mast_high = np.max(high[mast_start:mast_end+1])
                mast_low = np.min(low[mast_start:mast_end+1])
                mast_height = mast_high - mast_low
                
                flag_high = np.max(high[flag_start:flag_end+1])
                flag_low = np.min(low[flag_start:flag_end+1])
                flag_height = flag_high - flag_low
                
                # El mástil debe ser significativo y la bandera debe ser consolidación
                if mast_height > 0 and flag_height > 0:
                    if mast_height > flag_height * 2 and flag_height < mast_height * 0.3:
                        patterns['bearish_flag'][i] = True
        
        return patterns

    def calculate_volume_anomaly_improved(self, df):
        """Calcular anomalías de volumen MEJORADO con colores por compra/venta"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            n = len(volume)
            
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_ema = np.zeros(n)
            volume_colors = ['gray'] * n
            
            # Calcular EMA de volumen
            if n > 0:
                volume_ema = self.calculate_ema(volume, 20)
            
            for i in range(1, n):
                if volume_ema[i] > 0:
                    volume_ratio[i] = volume[i] / volume_ema[i]
                else:
                    volume_ratio[i] = 1
                
                # Detectar anomalía (> 2σ)
                if i >= 20:
                    window = volume[max(0, i-19):i+1]
                    window_mean = np.mean(window)
                    window_std = np.std(window)
                    
                    if window_std > 0 and volume[i] > window_mean + (2 * window_std):
                        volume_anomaly[i] = True
                        
                        # Determinar color basado en dirección del precio
                        if i > 0:
                            if close[i] > close[i-1]:  # Compra
                                volume_colors[i] = 'green'
                            else:  # Venta
                                volume_colors[i] = 'red'
                
                # Detectar clusters (múltiples anomalías consecutivas)
                if i >= 10:
                    recent_anomalies = volume_anomaly[max(0, i-9):i+1]
                    if np.sum(recent_anomalies) >= 2:  # Al menos 2 anomalías en 10 periodos
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ema': volume_ema.tolist(),
                'volume_colors': volume_colors
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly_improved: {e}")
            n = len(df)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ema': [0] * n,
                'volume_colors': ['gray'] * n
            }

    def check_volume_anomaly_signal(self, symbol, interval):
        """Verificar señales basadas en anomalías de volumen"""
        try:
            # Obtener datos de la temporalidad actual
            df_current = self.get_kucoin_data(symbol, interval, 100)
            if df_current is None or len(df_current) < 50:
                return None
            
            # Obtener datos de la temporalidad menor
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy or 'menor' not in hierarchy:
                return None
            
            menor_interval = hierarchy['menor']
            df_menor = self.get_kucoin_data(symbol, menor_interval, 50)
            if df_menor is None or len(df_menor) < 30:
                return None
            
            # Calcular anomalías en temporalidad menor
            volume_analysis = self.calculate_volume_anomaly_improved(df_menor)
            
            # Verificar si hay anomalía o cluster reciente
            current_idx = -1
            has_anomaly = volume_analysis['volume_anomaly'][current_idx]
            has_cluster = volume_analysis['volume_clusters'][current_idx]
            volume_color = volume_analysis['volume_colors'][current_idx]
            
            if not (has_anomaly or has_cluster):
                return None
            
            # Verificar condiciones FTMaverick en temporalidad menor
            menor_trend = self.calculate_trend_strength_maverick(df_menor['close'].values)
            menor_strength = menor_trend['strength_signals'][current_idx]
            menor_no_trade = menor_trend['no_trade_zones'][current_idx]
            
            # Verificar condiciones FTMaverick en temporalidad actual
            current_trend = self.calculate_trend_strength_maverick(df_current['close'].values)
            current_strength = current_trend['strength_signals'][current_idx]
            current_no_trade = current_trend['no_trade_zones'][current_idx]
            
            if current_no_trade or menor_no_trade:
                return None
            
            # Determinar señal basada en color de volumen y tendencias
            signal = None
            conditions = []
            
            if volume_color == 'green':  # Volumen de compra
                if (menor_strength in ['STRONG_UP', 'WEAK_UP'] and 
                    current_strength in ['STRONG_UP', 'WEAK_UP', 'NEUTRAL']):
                    signal = 'LONG'
                    conditions.append("Volumen anómalo de COMPRA en temporalidad menor")
                    conditions.append("FTMaverick: Temporalidad menor con tendencia alcista")
                    conditions.append("FTMaverick: Temporalidad actual Neutral o alcista")
                    conditions.append("Fuera de Zona No Operar en ambas temporalidades")
            
            elif volume_color == 'red':  # Volumen de venta
                if (menor_strength in ['STRONG_DOWN', 'WEAK_DOWN'] and 
                    current_strength in ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL']):
                    signal = 'SHORT'
                    conditions.append("Volumen anómalo de VENTA en temporalidad menor")
                    conditions.append("FTMaverick: Temporalidad menor con tendencia bajista")
                    conditions.append("FTMaverick: Temporalidad actual Neutral o bajista")
                    conditions.append("Fuera de Zona No Operar en ambas temporalidades")
            
            if not signal:
                return None
            
            # Calcular niveles de entrada óptimos
            levels_data = self.calculate_optimal_entry_exit_improved(df_current, signal, 15)
            
            # Obtener clasificación de riesgo
            risk_category = next(
                (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                 if symbol in symbols), 'medio'
            )
            risk_text = {
                'bajo': 'Bajo riesgo',
                'medio': 'Medio riesgo',
                'alto': 'Alto riesgo',
                'memecoins': 'Memecoin'
            }.get(risk_category, 'Medio riesgo')
            
            return {
                'symbol': symbol,
                'interval': interval,
                'menor_interval': menor_interval,
                'signal': signal,
                'risk_category': risk_text,
                'current_price': float(df_current['close'].iloc[-1]),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'conditions': conditions,
                'volume_analysis': {
                    'has_anomaly': has_anomaly,
                    'has_cluster': has_cluster,
                    'volume_color': volume_color,
                    'volume_ratio': volume_analysis['volume_ratio'][current_idx]
                },
                'ftmaverick_analysis': {
                    'current_strength': current_strength,
                    'menor_strength': menor_strength,
                    'current_no_trade': current_no_trade,
                    'menor_no_trade': menor_no_trade
                },
                'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error en check_volume_anomaly_signal para {symbol} {interval}: {e}")
            return None

    def evaluate_signal_conditions_new_weights(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con NUEVOS PESOS"""
        # Definir pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            weights = {
                'long': {
                    'multi_timeframe': 30,      # Obligatorio para estas TF
                    'trend_strength': 25,       # Obligatorio para todas
                    'bollinger_bands': 8,
                    'ma_cross': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'macd_cross': 10,
                    'volume_anomaly': 10,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'multi_timeframe': 30,      # Obligatorio para estas TF
                    'trend_strength': 25,       # Obligatorio para todas
                    'bollinger_bands': 8,
                    'ma_cross': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'macd_cross': 10,
                    'volume_anomaly': 10,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                }
            }
        elif interval in ['12h', '1D']:
            weights = {
                'long': {
                    'whale_signal': 30,         # Obligatorio para 12h, 1D
                    'trend_strength': 25,       # Obligatorio para todas
                    'bollinger_bands': 8,
                    'ma_cross': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'macd_cross': 10,
                    'volume_anomaly': 10,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'whale_signal': 30,         # Obligatorio para 12h, 1D
                    'trend_strength': 25,       # Obligatorio para todas
                    'bollinger_bands': 8,
                    'ma_cross': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'macd_cross': 10,
                    'volume_anomaly': 10,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                }
            }
        else:  # 1W
            weights = {
                'long': {
                    'trend_strength': 55,       # Mayor peso para 1W
                    'bollinger_bands': 8,
                    'ma_cross': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'macd_cross': 10,
                    'volume_anomaly': 10,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'trend_strength': 55,       # Mayor peso para 1W
                    'bollinger_bands': 8,
                    'ma_cross': 10,
                    'di_crossover': 10,
                    'adx_slope': 5,
                    'macd_cross': 10,
                    'volume_anomaly': 10,
                    'rsi_maverick_divergence': 8,
                    'rsi_traditional_divergence': 5,
                    'chart_pattern': 5,
                    'breakout': 5
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
                    'description': self.get_condition_description_new(key)
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
                data['confirmed_buy'][current_idx] and
                self.check_whale_confirmation(data, current_idx, 7)
            )
        
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        conditions['long']['bollinger_bands']['value'] = data.get('bollinger_conditions_long', False)
        
        # MA Cross (9 y 21)
        conditions['long']['ma_cross']['value'] = (
            ma_9 > ma_21 and 
            self.check_ma_cross_confirmation(data, current_idx, 'long')
        )
        
        # DI Crossover (+DI > -DI)
        conditions['long']['di_crossover']['value'] = (
            data['plus_di'][current_idx] > data['minus_di'][current_idx] and
            self.check_di_cross_confirmation(data, current_idx, 'long')
        )
        
        # ADX Slope positiva
        conditions['long']['adx_slope']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            self.check_adx_slope(data, current_idx, 'positive')
        )
        
        # MACD Cross
        conditions['long']['macd_cross']['value'] = (
            data['macd'][current_idx] > data['macd_signal'][current_idx] and
            self.check_macd_cross_confirmation(data, current_idx, 'long')
        )
        
        # Volumen Anomaly
        conditions['long']['volume_anomaly']['value'] = (
            data['volume_anomaly'][current_idx] and
            data['volume_clusters'][current_idx]
        )
        
        # RSI Maverick Divergence (con confirmación 4 velas)
        conditions['long']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bullish_divergence']) and 
            data['rsi_maverick_bullish_divergence'][current_idx] and
            self.check_divergence_confirmation(data, current_idx, 'rsi_maverick', 'bullish', 4)
        )
        
        # RSI Traditional Divergence (con confirmación 4 velas)
        conditions['long']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bullish_divergence']) and 
            data['rsi_bullish_divergence'][current_idx] and
            self.check_divergence_confirmation(data, current_idx, 'rsi_traditional', 'bullish', 4)
        )
        
        # Chart Patterns (con confirmación 7 velas)
        conditions['long']['chart_pattern']['value'] = (
            (data['chart_patterns']['double_bottom'][current_idx] or
             data['chart_patterns']['bullish_flag'][current_idx]) and
            self.check_pattern_confirmation(data, current_idx, 'long', 7)
        )
        
        # Breakout
        conditions['long']['breakout']['value'] = (
            current_idx < len(data['breakout_up']) and 
            data['breakout_up'][current_idx]
        )
        
        # Condiciones SHORT
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        elif interval in ['12h', '1D']:
            conditions['short']['whale_signal']['value'] = (
                data['whale_dump'][current_idx] > 20 and
                data['confirmed_sell'][current_idx] and
                self.check_whale_confirmation(data, current_idx, 7)
            )
        
        conditions['short']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            not data['no_trade_zones'][current_idx]
        )
        
        conditions['short']['bollinger_bands']['value'] = data.get('bollinger_conditions_short', False)
        
        # MA Cross (9 y 21)
        conditions['short']['ma_cross']['value'] = (
            ma_9 < ma_21 and 
            self.check_ma_cross_confirmation(data, current_idx, 'short')
        )
        
        # DI Crossover (-DI > +DI)
        conditions['short']['di_crossover']['value'] = (
            data['minus_di'][current_idx] > data['plus_di'][current_idx] and
            self.check_di_cross_confirmation(data, current_idx, 'short')
        )
        
        # ADX Slope positiva
        conditions['short']['adx_slope']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            self.check_adx_slope(data, current_idx, 'positive')
        )
        
        # MACD Cross
        conditions['short']['macd_cross']['value'] = (
            data['macd'][current_idx] < data['macd_signal'][current_idx] and
            self.check_macd_cross_confirmation(data, current_idx, 'short')
        )
        
        # Volumen Anomaly
        conditions['short']['volume_anomaly']['value'] = (
            data['volume_anomaly'][current_idx] and
            data['volume_clusters'][current_idx]
        )
        
        # RSI Maverick Divergence (con confirmación 4 velas)
        conditions['short']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bearish_divergence']) and 
            data['rsi_maverick_bearish_divergence'][current_idx] and
            self.check_divergence_confirmation(data, current_idx, 'rsi_maverick', 'bearish', 4)
        )
        
        # RSI Traditional Divergence (con confirmación 4 velas)
        conditions['short']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bearish_divergence']) and 
            data['rsi_bearish_divergence'][current_idx] and
            self.check_divergence_confirmation(data, current_idx, 'rsi_traditional', 'bearish', 4)
        )
        
        # Chart Patterns (con confirmación 7 velas)
        conditions['short']['chart_pattern']['value'] = (
            (data['chart_patterns']['head_shoulders'][current_idx] or
             data['chart_patterns']['double_top'][current_idx] or
             data['chart_patterns']['bearish_flag'][current_idx]) and
            self.check_pattern_confirmation(data, current_idx, 'short', 7)
        )
        
        # Breakout
        conditions['short']['breakout']['value'] = (
            current_idx < len(data['breakout_down']) and 
            data['breakout_down'][current_idx]
        )
        
        return conditions

    def get_condition_description_new(self, condition_key):
        """Obtener descripción de condición para nuevos pesos"""
        descriptions = {
            'multi_timeframe': 'Multi-TF obligatorio (30%)',
            'trend_strength': 'Fuerza tendencia favorable (25%)',
            'whale_signal': 'Señal ballenas confirmada (30%)',
            'bollinger_bands': 'Bandas de Bollinger (8%)',
            'ma_cross': 'Cruce MA9/MA21 (10%)',
            'di_crossover': 'Cruce +DI/-DI (10%)',
            'adx_slope': 'ADX pendiente positiva (5%)',
            'macd_cross': 'Cruce MACD (10%)',
            'volume_anomaly': 'Anomalía/Cluster volumen (10%)',
            'rsi_maverick_divergence': 'Divergencia RSI Maverick (8%)',
            'rsi_traditional_divergence': 'Divergencia RSI Tradicional (5%)',
            'chart_pattern': 'Patrón chartista (5%)',
            'breakout': 'Ruptura (5%)'
        }
        return descriptions.get(condition_key, condition_key)

    def check_whale_confirmation(self, data, current_idx, lookback=7):
        """Verificar confirmación de señal de ballenas en velas posteriores"""
        if current_idx >= len(data['whale_pump']) - lookback:
            return False
        
        # Para señal LONG (whale pump)
        if data['confirmed_buy'][current_idx]:
            # Verificar que haya confirmación en las próximas lookback velas
            confirmations = 0
            for i in range(1, min(lookback + 1, len(data['whale_pump']) - current_idx)):
                if data['whale_pump'][current_idx + i] > 15:
                    confirmations += 1
            return confirmations >= 3
        
        # Para señal SHORT (whale dump)
        if data['confirmed_sell'][current_idx]:
            # Verificar que haya confirmación en las próximas lookback velas
            confirmations = 0
            for i in range(1, min(lookback + 1, len(data['whale_dump']) - current_idx)):
                if data['whale_dump'][current_idx + i] > 15:
                    confirmations += 1
            return confirmations >= 3
        
        return False

    def check_ma_cross_confirmation(self, data, current_idx, signal_type):
        """Verificar confirmación de cruce de medias"""
        if current_idx < 3:
            return False
        
        ma_9 = data['ma_9']
        ma_21 = data['ma_21']
        
        if signal_type == 'long':
            # Verificar que MA9 esté por encima de MA21 en las últimas 3 velas
            for i in range(3):
                idx = current_idx - i
                if idx < 0:
                    return False
                if ma_9[idx] <= ma_21[idx]:
                    return False
            return True
        
        else:  # short
            # Verificar que MA9 esté por debajo de MA21 en las últimas 3 velas
            for i in range(3):
                idx = current_idx - i
                if idx < 0:
                    return False
                if ma_9[idx] >= ma_21[idx]:
                    return False
            return True

    def check_di_cross_confirmation(self, data, current_idx, signal_type):
        """Verificar confirmación de cruce DMI"""
        if current_idx < 3:
            return False
        
        plus_di = data['plus_di']
        minus_di = data['minus_di']
        
        if signal_type == 'long':
            # Verificar que +DI > -DI en las últimas 3 velas
            for i in range(3):
                idx = current_idx - i
                if idx < 0:
                    return False
                if plus_di[idx] <= minus_di[idx]:
                    return False
            return True
        
        else:  # short
            # Verificar que -DI > +DI en las últimas 3 velas
            for i in range(3):
                idx = current_idx - i
                if idx < 0:
                    return False
                if minus_di[idx] <= plus_di[idx]:
                    return False
            return True

    def check_adx_slope(self, data, current_idx, slope_type='positive'):
        """Verificar pendiente del ADX"""
        if current_idx < 5:
            return False
        
        adx = data['adx']
        
        if slope_type == 'positive':
            # ADX creciente en las últimas 5 velas
            slopes = []
            for i in range(1, 6):
                if current_idx - i < 0:
                    return False
                if current_idx - i + 1 < 0:
                    return False
                slope = adx[current_idx - i + 1] - adx[current_idx - i]
                slopes.append(slope)
            
            # Al menos 3 pendientes positivas
            positive_slopes = sum(1 for s in slopes if s > 0)
            return positive_slopes >= 3
        
        return False

    def check_macd_cross_confirmation(self, data, current_idx, signal_type):
        """Verificar confirmación de cruce MACD"""
        if current_idx < 3:
            return False
        
        macd = data['macd']
        macd_signal = data['macd_signal']
        
        if signal_type == 'long':
            # Verificar que MACD > Señal en las últimas 3 velas
            for i in range(3):
                idx = current_idx - i
                if idx < 0:
                    return False
                if macd[idx] <= macd_signal[idx]:
                    return False
            return True
        
        else:  # short
            # Verificar que MACD < Señal en las últimas 3 velas
            for i in range(3):
                idx = current_idx - i
                if idx < 0:
                    return False
                if macd[idx] >= macd_signal[idx]:
                    return False
            return True

    def check_divergence_confirmation(self, data, current_idx, indicator_type, divergence_type, lookback=4):
        """Verificar confirmación de divergencia en velas posteriores"""
        if current_idx >= len(data['close']) - lookback:
            return False
        
        close = data['close']
        
        if indicator_type == 'rsi_maverick':
            indicator = data['rsi_maverick']
        else:  # rsi_traditional
            indicator = data['rsi_traditional']
        
        if divergence_type == 'bullish':
            # Después de divergencia alcista, el precio debería subir
            price_increase = 0
            indicator_increase = 0
            
            for i in range(1, min(lookback + 1, len(close) - current_idx)):
                if close[current_idx + i] > close[current_idx]:
                    price_increase += 1
                if indicator[current_idx + i] > indicator[current_idx]:
                    indicator_increase += 1
            
            return price_increase >= 2 or indicator_increase >= 2
        
        else:  # bearish
            # Después de divergencia bajista, el precio debería bajar
            price_decrease = 0
            indicator_decrease = 0
            
            for i in range(1, min(lookback + 1, len(close) - current_idx)):
                if close[current_idx + i] < close[current_idx]:
                    price_decrease += 1
                if indicator[current_idx + i] < indicator[current_idx]:
                    indicator_decrease += 1
            
            return price_decrease >= 2 or indicator_decrease >= 2

    def check_pattern_confirmation(self, data, current_idx, signal_type, lookback=7):
        """Verificar confirmación de patrón chartista en velas posteriores"""
        if current_idx >= len(data['close']) - lookback:
            return False
        
        close = data['close']
        
        if signal_type == 'long':
            # Para patrones alcistas, el precio debería subir
            price_increase = 0
            for i in range(1, min(lookback + 1, len(close) - current_idx)):
                if close[current_idx + i] > close[current_idx]:
                    price_increase += 1
            return price_increase >= 4  # Al menos 4 de 7 velas subiendo
        
        else:  # short
            # Para patrones bajistas, el precio debería bajar
            price_decrease = 0
            for i in range(1, min(lookback + 1, len(close) - current_idx)):
                if close[current_idx + i] < close[current_idx]:
                    price_decrease += 1
            return price_decrease >= 4  # Al menos 4 de 7 velas bajando

    def calculate_signal_score_new_weights(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal con nuevos pesos y reglas MA200"""
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
        
        # Ajustar score mínimo según posición respecto a MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:  # short
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0

        return min(final_score, 100), fulfilled_conditions

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

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - CON NUEVOS PESOS"""
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
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            breakout_up, breakout_down = self.check_breakout(high, low, close, 
                                                           np.min(low[-sr_period:]), 
                                                           np.max(high[-sr_period:]))
            
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Verificar condiciones de Bollinger
            bollinger_conditions_long = self.check_bollinger_conditions_corrected(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions_corrected(df, interval, 'SHORT')
            
            # Nuevo indicador de volumen MEJORADO
            volume_analysis = self.calculate_volume_anomaly_improved(df)
            
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
                'volume_anomaly': volume_analysis['volume_anomaly'],
                'volume_clusters': volume_analysis['volume_clusters'],
                'volume_ratio': volume_analysis['volume_ratio'],
                'volume_colors': volume_analysis['volume_colors'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short
            }
            
            conditions = self.evaluate_signal_conditions_new_weights(analysis_data, current_idx, interval, adx_threshold)
            
            # Calcular condición MA200
            current_ma200 = ma_200[current_idx] if current_idx < len(ma_200) else 0
            current_price = close[current_idx]
            ma200_condition = 'above' if current_price > current_ma200 else 'below'

            long_score, long_conditions = self.calculate_signal_score_new_weights(conditions, 'long', ma200_condition)
            short_score, short_conditions = self.calculate_signal_score_new_weights(conditions, 'short', ma200_condition)
            
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
            levels_data = self.calculate_optimal_entry_exit_improved(df, signal_type, leverage)
            
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
                    'volume_anomaly': volume_analysis['volume_anomaly'][-50:],
                    'volume_clusters': volume_analysis['volume_clusters'][-50:],
                    'volume_ratio': volume_analysis['volume_ratio'][-50:],
                    'volume_ema': volume_analysis['volume_ema'][-50:],
                    'volume_colors': volume_analysis['volume_colors'][-50:],
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
            'take_profit': [0],
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
        """Generar alertas de trading para ambas estrategias"""
        alerts = []
        telegram_intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']
        
        current_time = self.get_bolivia_time()
        
        for interval in telegram_intervals:
            # Estrategia 1: Señales principales
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if should_send_alert:
                for symbol in CRYPTO_SYMBOLS[:12]:
                    try:
                        signal_data = self.generate_signals_improved(symbol, interval)
                        
                        if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                            signal_data['signal_score'] >= 65):
                            
                            risk_category = next(
                                (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                                 if symbol in symbols), 'medio'
                            )
                            
                            risk_text = {
                                'bajo': 'Bajo riesgo',
                                'medio': 'Medio riesgo',
                                'alto': 'Alto riesgo',
                                'memecoins': 'Memecoin'
                            }.get(risk_category, 'Medio riesgo')
                            
                            alert = {
                                'strategy': 'main',
                                'symbol': symbol,
                                'interval': interval,
                                'signal': signal_data['signal'],
                                'score': signal_data['signal_score'],
                                'winrate': signal_data['winrate'],
                                'entry': signal_data['entry'],
                                'stop_loss': signal_data['stop_loss'],
                                'take_profit': signal_data['take_profit'][0],
                                'supports': signal_data['supports'],
                                'resistances': signal_data['resistances'],
                                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                                'risk_category': risk_text,
                                'current_price': signal_data['current_price'],
                                'ma200_condition': signal_data.get('ma200_condition', 'below'),
                                'multi_timeframe_ok': signal_data.get('multi_timeframe_ok', False)
                            }
                            
                            alert_key = f"main_{symbol}_{interval}_{signal_data['signal']}"
                            if (alert_key not in self.alert_cache or 
                                (datetime.now() - self.alert_cache[alert_key]).seconds > 300):
                                
                                alerts.append(alert)
                                self.alert_cache[alert_key] = datetime.now()
                        
                    except Exception as e:
                        print(f"Error generando alerta principal para {symbol} {interval}: {e}")
                        continue
            
            # Estrategia 2: Alertas de volumen anómalo
            volume_check_intervals = {
                '15m': 60,     # Cada 60 segundos
                '30m': 60,     # Cada 60 segundos
                '1h': 300,     # Cada 300 segundos
                '2h': 300,     # Cada 300 segundos
                '4h': 600,     # Cada 600 segundos
                '8h': 600,     # Cada 600 segundos
                '12h': 600,    # Cada 600 segundos
                '1D': 600,     # Cada 600 segundos
                '1W': 0        # No aplica
            }
            
            check_interval = volume_check_intervals.get(interval, 0)
            if check_interval > 0:
                cache_key = f"volume_{symbol}_{interval}"
                last_check = self.volume_alert_cache.get(cache_key)
                
                if (last_check is None or 
                    (datetime.now() - last_check).seconds >= check_interval):
                    
                    for symbol in CRYPTO_SYMBOLS[:8]:
                        try:
                            volume_signal = self.check_volume_anomaly_signal(symbol, interval)
                            
                            if volume_signal:
                                alert = {
                                    'strategy': 'volume_anomaly',
                                    'symbol': volume_signal['symbol'],
                                    'interval': volume_signal['interval'],
                                    'menor_interval': volume_signal['menor_interval'],
                                    'signal': volume_signal['signal'],
                                    'risk_category': volume_signal['risk_category'],
                                    'current_price': volume_signal['current_price'],
                                    'entry': volume_signal['entry'],
                                    'stop_loss': volume_signal['stop_loss'],
                                    'take_profit': volume_signal['take_profit'],
                                    'conditions': volume_signal['conditions'],
                                    'volume_analysis': volume_signal['volume_analysis'],
                                    'ftmaverick_analysis': volume_signal['ftmaverick_analysis'],
                                    'timestamp': volume_signal['timestamp']
                                }
                                
                                alert_key = f"volume_{symbol}_{interval}_{volume_signal['signal']}"
                                if (alert_key not in self.alert_cache or 
                                    (datetime.now() - self.alert_cache[alert_key]).seconds > 300):
                                    
                                    alerts.append(alert)
                                    self.alert_cache[alert_key] = datetime.now()
                                    self.volume_alert_cache[cache_key] = datetime.now()
                        
                        except Exception as e:
                            print(f"Error generando alerta de volumen para {symbol} {interval}: {e}")
                            continue
        
        return alerts

    def generate_telegram_image_main_strategy(self, alert_data):
        """Generar imagen para Telegram de la estrategia principal"""
        try:
            symbol = alert_data['symbol']
            interval = alert_data['interval']
            signal_type = alert_data['signal']
            
            # Obtener datos para los gráficos
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            # Configurar matplotlib para fondo blanco
            plt.style.use('default')
            fig = plt.figure(figsize=(14, 16))
            fig.patch.set_facecolor('white')
            
            # 1. Gráfico de Velas Japonesas con Bollinger y Medias
            ax1 = plt.subplot(8, 1, 1)
            
            dates = [d for d in df['timestamp']]
            if isinstance(dates[0], str):
                dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
            
            closes = df['close'].values
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Graficar velas
            for i in range(len(dates)-50, len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Bandas de Bollinger (transparentes)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes)
            ax1.plot(dates[-50:], bb_upper[-50:], color='orange', alpha=0.3, linewidth=1)
            ax1.plot(dates[-50:], bb_middle[-50:], color='orange', alpha=0.3, linewidth=1)
            ax1.plot(dates[-50:], bb_lower[-50:], color='orange', alpha=0.3, linewidth=1)
            
            # Medias móviles
            ma_9 = self.calculate_sma(closes, 9)
            ma_21 = self.calculate_sma(closes, 21)
            ma_50 = self.calculate_sma(closes, 50)
            
            ax1.plot(dates[-50:], ma_9[-50:], color='blue', linewidth=1, label='MA9')
            ax1.plot(dates[-50:], ma_21[-50:], color='red', linewidth=1, label='MA21')
            ax1.plot(dates[-50:], ma_50[-50:], color='purple', linewidth=1, label='MA50')
            
            # Soportes y resistencias
            supports = alert_data.get('supports', [])
            resistances = alert_data.get('resistances', [])
            
            for i, support in enumerate(supports[:4]):
                ax1.axhline(y=support, color='blue', linestyle='--', alpha=0.5, 
                           label=f'S{i+1}' if i == 0 else '')
            
            for i, resistance in enumerate(resistances[:4]):
                ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5,
                           label=f'R{i+1}' if i == 0 else '')
            
            ax1.set_title(f'{symbol} - {interval} - {signal_type}', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Precio')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. ADX con DMI
            ax2 = plt.subplot(8, 1, 2, sharex=ax1)
            adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes)
            
            ax2.plot(dates[-50:], adx[-50:], 'black', linewidth=2, label='ADX')
            ax2.plot(dates[-50:], plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax2.plot(dates[-50:], minus_di[-50:], 'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Umbral 25')
            
            ax2.set_ylabel('ADX/DMI')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Volumen con Anomalías (como columnas)
            ax3 = plt.subplot(8, 1, 3, sharex=ax1)
            volumes = df['volume'].values
            
            # Colores para barras de volumen
            bar_colors = []
            for i in range(len(dates)-50, len(dates)):
                idx = i
                if idx > 0 and idx < len(closes):
                    bar_colors.append('green' if closes[idx] > closes[idx-1] else 'red')
                else:
                    bar_colors.append('gray')
            
            ax3.bar(dates[-50:], volumes[-50:], color=bar_colors, alpha=0.6, width=0.8)
            
            # Anomalías de volumen
            volume_analysis = self.calculate_volume_anomaly_improved(df)
            anomaly_dates = []
            anomaly_values = []
            
            for i in range(len(dates)-50, len(dates)):
                if volume_analysis['volume_anomaly'][i]:
                    anomaly_dates.append(dates[i])
                    anomaly_values.append(volumes[i])
            
            if anomaly_dates:
                ax3.scatter(anomaly_dates, anomaly_values, color='black', s=50, 
                          marker='o', zorder=5, label='Anomalías')
            
            ax3.set_ylabel('Volumen')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # 4. Fuerza de Tendencia Maverick (como columnas)
            ax4 = plt.subplot(8, 1, 4, sharex=ax1)
            trend_data = self.calculate_trend_strength_maverick(closes)
            trend_strength = trend_data['trend_strength']
            
            # Graficar como columnas
            bar_colors_trend = ['green' if x > 0 else 'red' for x in trend_strength[-50:]]
            ax4.bar(dates[-50:], trend_strength[-50:], color=bar_colors_trend, alpha=0.7, width=0.8)
            
            # Umbral y zona no operar
            threshold = trend_data['high_zone_threshold']
            ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label='Umbral')
            ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            # Marcar zonas no operar
            no_trade_dates = []
            for i in range(len(dates)-50, len(dates)):
                if trend_data['no_trade_zones'][i]:
                    no_trade_dates.append(dates[i])
            
            for date in no_trade_dates:
                ax4.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.legend(loc='upper left', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # 5. Ballenas solo para 12h y 1D
            if interval in ['12h', '1D']:
                ax5 = plt.subplot(8, 1, 5, sharex=ax1)
                whale_data = self.calculate_whale_signals_improved(df)
                
                # Graficar como columnas
                width = 0.4
                dates_whale = dates[-50:]
                whale_pump = whale_data['whale_pump'][-50:]
                whale_dump = whale_data['whale_dump'][-50:]
                
                ax5.bar([d - timedelta(hours=width/2) for d in dates_whale], 
                       whale_pump, width=width, color='green', alpha=0.7, label='Compra')
                ax5.bar([d + timedelta(hours=width/2) for d in dates_whale], 
                       whale_dump, width=width, color='red', alpha=0.7, label='Venta')
                
                ax5.set_ylabel('Ballenas')
                ax5.legend(loc='upper left', fontsize=8)
                ax5.grid(True, alpha=0.3)
            
            # 6. RSI Maverick
            ax6_idx = 6 if interval in ['12h', '1D'] else 5
            ax6 = plt.subplot(8, 1, ax6_idx, sharex=ax1)
            rsi_maverick = self.calculate_rsi_maverick(closes)
            
            ax6.plot(dates[-50:], rsi_maverick[-50:], 'blue', linewidth=2)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            
            ax6.set_ylabel('RSI Maverick')
            ax6.legend(loc='upper left', fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            # 7. RSI Tradicional
            ax7 = plt.subplot(8, 1, ax6_idx + 1, sharex=ax1)
            rsi_traditional = self.calculate_rsi(closes)
            
            ax7.plot(dates[-50:], rsi_traditional[-50:], 'cyan', linewidth=2)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            
            ax7.set_ylabel('RSI Tradicional')
            ax7.legend(loc='upper left', fontsize=8)
            ax7.grid(True, alpha=0.3)
            
            # 8. MACD con Histograma (como columnas)
            ax8 = plt.subplot(8, 1, ax6_idx + 2, sharex=ax1)
            macd, macd_signal, macd_histogram = self.calculate_macd(closes)
            
            ax8.plot(dates[-50:], macd[-50:], 'blue', linewidth=1, label='MACD')
            ax8.plot(dates[-50:], macd_signal[-50:], 'red', linewidth=1, label='Señal')
            
            # Histograma como columnas
            hist_colors = ['green' if x > 0 else 'red' for x in macd_histogram[-50:]]
            ax8.bar(dates[-50:], macd_histogram[-50:], color=hist_colors, alpha=0.6, width=0.8, label='Histograma')
            
            ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax8.set_ylabel('MACD')
            ax8.legend(loc='upper left', fontsize=8)
            ax8.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar imagen en buffer
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando imagen para estrategia principal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_telegram_image_volume_strategy(self, alert_data):
        """Generar imagen para Telegram de la estrategia de volumen"""
        try:
            symbol = alert_data['symbol']
            interval = alert_data['interval']
            menor_interval = alert_data['menor_interval']
            signal_type = alert_data['signal']
            
            # Obtener datos de la temporalidad actual
            df_current = self.get_kucoin_data(symbol, interval, 100)
            if df_current is None or len(df_current) < 50:
                return None
            
            # Obtener datos de la temporalidad menor
            df_menor = self.get_kucoin_data(symbol, menor_interval, 100)
            if df_menor is None or len(df_menor) < 50:
                return None
            
            # Configurar matplotlib para fondo blanco
            plt.style.use('default')
            fig = plt.figure(figsize=(14, 10))
            fig.patch.set_facecolor('white')
            
            # 1. Gráfico de Velas Japonesas (temporalidad actual)
            ax1 = plt.subplot(4, 1, 1)
            
            dates_current = [d for d in df_current['timestamp']]
            if isinstance(dates_current[0], str):
                dates_current = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates_current]
            
            closes_current = df_current['close'].values
            opens_current = df_current['open'].values
            highs_current = df_current['high'].values
            lows_current = df_current['low'].values
            
            # Graficar velas (últimas 50)
            for i in range(len(dates_current)-50, len(dates_current)):
                color = 'green' if closes_current[i] >= opens_current[i] else 'red'
                ax1.plot([dates_current[i], dates_current[i]], [lows_current[i], highs_current[i]], 
                        color='black', linewidth=1)
                ax1.plot([dates_current[i], dates_current[i]], [opens_current[i], closes_current[i]], 
                        color=color, linewidth=3)
            
            # Bandas de Bollinger (transparentes)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes_current)
            ax1.plot(dates_current[-50:], bb_upper[-50:], color='orange', alpha=0.3, linewidth=1)
            ax1.plot(dates_current[-50:], bb_middle[-50:], color='orange', alpha=0.3, linewidth=1)
            ax1.plot(dates_current[-50:], bb_lower[-50:], color='orange', alpha=0.3, linewidth=1)
            
            ax1.set_title(f'{symbol} - {interval} - {signal_type} (Volumen Anómalo)', 
                         fontsize=12, fontweight='bold')
            ax1.set_ylabel('Precio')
            ax1.grid(True, alpha=0.3)
            
            # 2. ADX con DMI (temporalidad actual)
            ax2 = plt.subplot(4, 1, 2, sharex=ax1)
            adx, plus_di, minus_di = self.calculate_adx(highs_current, lows_current, closes_current)
            
            ax2.plot(dates_current[-50:], adx[-50:], 'black', linewidth=2, label='ADX')
            ax2.plot(dates_current[-50:], plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax2.plot(dates_current[-50:], minus_di[-50:], 'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Umbral 25')
            
            ax2.set_ylabel('ADX/DMI')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Volumen con Anomalías (temporalidad menor - como columnas)
            ax3 = plt.subplot(4, 1, 3)
            
            dates_menor = [d for d in df_menor['timestamp']]
            if isinstance(dates_menor[0], str):
                dates_menor = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates_menor]
            
            closes_menor = df_menor['close'].values
            volumes_menor = df_menor['volume'].values
            
            # Calcular anomalías de volumen en temporalidad menor
            volume_analysis = self.calculate_volume_anomaly_improved(df_menor)
            
            # Colores para barras de volumen basados en dirección del precio
            bar_colors = []
            for i in range(len(dates_menor)-50, len(dates_menor)):
                idx = i
                if idx > 0 and idx < len(closes_menor):
                    bar_colors.append('green' if closes_menor[idx] > closes_menor[idx-1] else 'red')
                else:
                    bar_colors.append('gray')
            
            ax3.bar(dates_menor[-50:], volumes_menor[-50:], color=bar_colors, alpha=0.6, width=0.8)
            
            # Marcar anomalías
            anomaly_dates = []
            anomaly_values = []
            cluster_dates = []
            
            for i in range(len(dates_menor)-50, len(dates_menor)):
                if volume_analysis['volume_anomaly'][i]:
                    anomaly_dates.append(dates_menor[i])
                    anomaly_values.append(volumes_menor[i])
                
                if volume_analysis['volume_clusters'][i]:
                    cluster_dates.append(dates_menor[i])
            
            if anomaly_dates:
                ax3.scatter(anomaly_dates, anomaly_values, color='black', s=80,
                          marker='o', zorder=5, label='Anomalías')
            
            if cluster_dates:
                for date in cluster_dates:
                    ax3.axvline(x=date, color='red', alpha=0.5, linewidth=2)
            
            ax3.set_title(f'Volumen {menor_interval} - Anomalías de {"COMPRA" if alert_data["volume_analysis"]["volume_color"] == "green" else "VENTA"}', 
                         fontsize=10)
            ax3.set_ylabel('Volumen')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # 4. Fuerza de Tendencia Maverick (ambas temporalidades)
            ax4 = plt.subplot(4, 1, 4)
            
            # Fuerza de tendencia en temporalidad actual
            trend_current = self.calculate_trend_strength_maverick(closes_current)
            trend_menor = self.calculate_trend_strength_maverick(closes_menor)
            
            # Graficar ambas como columnas lado a lado
            width = 0.4
            
            # Ajustar fechas para visualización
            dates_current_adj = dates_current[-25:]  # Últimas 25 velas
            dates_menor_adj = dates_menor[-25:]      # Últimas 25 velas
            
            ax4.bar([d - timedelta(hours=width/2) for d in dates_current_adj], 
                   trend_current['trend_strength'][-25:], 
                   width=width, color='blue', alpha=0.7, label=interval)
            
            ax4.bar([d + timedelta(hours=width/2) for d in dates_menor_adj], 
                   trend_menor['trend_strength'][-25:], 
                   width=width, color='orange', alpha=0.7, label=menor_interval)
            
            # Umbrales
            threshold_current = trend_current['high_zone_threshold']
            threshold_menor = trend_menor['high_zone_threshold']
            
            ax4.axhline(y=threshold_current, color='blue', linestyle='--', alpha=0.5, 
                       label=f'Umbral {interval}')
            ax4.axhline(y=-threshold_current, color='blue', linestyle='--', alpha=0.5)
            
            ax4.axhline(y=threshold_menor, color='orange', linestyle='--', alpha=0.5,
                       label=f'Umbral {menor_interval}')
            ax4.axhline(y=-threshold_menor, color='orange', linestyle='--', alpha=0.5)
            
            ax4.set_title('Fuerza de Tendencia Maverick - Comparación Temporalidades', fontsize=10)
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.legend(loc='upper left', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar imagen en buffer
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando imagen para estrategia de volumen: {e}")
            import traceback
            traceback.print_exc()
            return None

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert_with_image(alert_data, image_buffer, strategy_type):
    """Enviar alerta por Telegram con imagen"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if strategy_type == 'main':
            # Mensaje para estrategia principal
            conditions_text = "\n".join([f"• {cond}" for cond in alert_data['fulfilled_conditions'][:8]])
            
            message = f"""
🚨 SEÑAL {alert_data['signal']} CONFIRMADA 🚨

Crypto: {alert_data['symbol']} ({alert_data['risk_category']})
Temporalidad: {alert_data['interval']}
Multi-TF: {'✅ CONFIRMADO' if alert_data.get('multi_timeframe_ok', False) else '❌ NO CONFIRMADO'}
Score: {alert_data['score']:.1f}%

Precio actual: ${alert_data['current_price']:.6f}
Entrada recomendada: ${alert_data['entry']:.6f}
Stop Loss: ${alert_data['stop_loss']:.6f}
Take Profit: ${alert_data['take_profit']:.6f}

MA200: {'ENCIMA ✅' if alert_data.get('ma200_condition') == 'above' else 'DEBAJO ⚠️'}

Condiciones cumplidas:
{conditions_text}

Timestamp: {alert_data['timestamp']}
            """
            
        else:  # volume_anomaly
            # Mensaje para estrategia de volumen anómalo
            conditions_text = "\n".join([f"• {cond}" for cond in alert_data['conditions']])
            
            volume_type = "COMPRA" if alert_data['volume_analysis']['volume_color'] == 'green' else "VENTA"
            
            message = f"""
🚨 SEÑAL {alert_data['signal']} VOLUMEN ANÓMALO 🚨

Crypto: {alert_data['symbol']} ({alert_data['risk_category']})
Volumen: Anomalías de {volume_type} en temporalidad menor
Temporalidad actual: {alert_data['interval']}
Temporalidad volumen: {alert_data['menor_interval']}

Precio actual: ${alert_data['current_price']:.6f}
Entrada recomendada: ${alert_data['entry']:.6f}
Stop Loss: ${alert_data['stop_loss']:.6f}
Take Profit: ${alert_data['take_profit']:.6f}

Condiciones FTMaverick y MF:
{conditions_text}

Recomendación: revisar {alert_data['signal']}
Timestamp: {alert_data['timestamp']}
            """
        
        # Enviar imagen
        if image_buffer:
            image_buffer.seek(0)
            asyncio.run(bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=image_buffer,
                caption=message[:1024]  # Telegram limita a 1024 caracteres
            ))
            print(f"Alerta {strategy_type} enviada a Telegram con imagen: {alert_data['symbol']}")
        else:
            # Enviar solo mensaje si no hay imagen
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            ))
            print(f"Alerta {strategy_type} enviada a Telegram sin imagen: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta con imagen a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            print("Verificando alertas...")
            
            # Generar todas las alertas
            alerts = indicator.generate_scalping_alerts()
            
            for alert in alerts:
                try:
                    # Generar imagen según la estrategia
                    image_buffer = None
                    
                    if alert['strategy'] == 'main':
                        image_buffer = indicator.generate_telegram_image_main_strategy(alert)
                    else:  # volume_anomaly
                        image_buffer = indicator.generate_telegram_image_volume_strategy(alert)
                    
                    # Enviar alerta con imagen
                    send_telegram_alert_with_image(alert, image_buffer, alert['strategy'])
                    
                    # Pequeña pausa entre alertas para no sobrecargar Telegram
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error procesando alerta: {e}")
                    continue
            
            # Intervalos de verificación según temporalidad
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
                    
                    risk_category = next(
                        (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                         if symbol in symbols), 'medio'
                    )
                    
                    scatter_data.append({
                        'symbol': symbol,
                        'x': float(buy_pressure),
                        'y': float(sell_pressure),
                        'signal_score': float(signal_data['signal_score']),
                        'current_price': float(signal_data['current_price']),
                        'signal': signal_data['signal'],
                        'risk_category': risk_category
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
        interval = request.args.get('interval', '1h')
        
        volume_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:8]:
            try:
                signal = indicator.check_volume_anomaly_signal(symbol, interval)
                if signal:
                    volume_signals.append(signal)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error procesando volumen anómalo para {symbol}: {e}")
                continue
        
        return jsonify({'volume_signals': volume_signals})
        
    except Exception as e:
        print(f"Error en /api/volume_anomaly_signals: {e}")
        return jsonify({'volume_signals': []})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo CORREGIDO"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Configurar matplotlib
        plt.style.use('default')
        fig = plt.figure(figsize=(14, 18))
        fig.patch.set_facecolor('white')
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(9, 1, 1)
        if signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            opens = [d['open'] for d in signal_data['data']]
            highs = [d['high'] for d in signal_data['data']]
            lows = [d['low'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            
            # Graficar velas (últimas 50)
            start_idx = max(0, len(dates) - 50)
            for i in range(start_idx, len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Bandas de Bollinger
            if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
                bb_upper = signal_data['indicators']['bb_upper'][-50:]
                bb_middle = signal_data['indicators']['bb_middle'][-50:]
                bb_lower = signal_data['indicators']['bb_lower'][-50:]
                
                ax1.plot(dates[start_idx:], bb_upper, color='orange', alpha=0.3, linewidth=1, label='BB Superior')
                ax1.plot(dates[start_idx:], bb_middle, color='orange', alpha=0.3, linewidth=1, label='BB Media')
                ax1.plot(dates[start_idx:], bb_lower, color='orange', alpha=0.3, linewidth=1, label='BB Inferior')
            
            # Medias móviles
            if 'indicators' in signal_data and 'ma_9' in signal_data['indicators']:
                ma_9 = signal_data['indicators']['ma_9'][-50:]
                ma_21 = signal_data['indicators']['ma_21'][-50:]
                ma_50 = signal_data['indicators']['ma_50'][-50:]
                ma_200 = signal_data['indicators']['ma_200'][-50:]
                
                ax1.plot(dates[start_idx:], ma_9, color='blue', linewidth=1, label='MA9')
                ax1.plot(dates[start_idx:], ma_21, color='red', linewidth=1, label='MA21')
                ax1.plot(dates[start_idx:], ma_50, color='purple', linewidth=1, label='MA50')
                ax1.plot(dates[start_idx:], ma_200, color='black', linewidth=2, label='MA200')
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='gold', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal_data['take_profit'][0], color='green', linestyle='--', alpha=0.7, label='Take Profit')
            
            # Soportes y resistencias (4 cada uno)
            supports = signal_data.get('supports', [])
            resistances = signal_data.get('resistances', [])
            
            for i, support in enumerate(supports[:4]):
                ax1.axhline(y=support, color='blue', linestyle=':', alpha=0.5, 
                           label=f'Soporte {i+1}' if i == 0 else '')
            
            for i, resistance in enumerate(resistances[:4]):
                ax1.axhline(y=resistance, color='red', linestyle=':', alpha=0.5,
                           label=f'Resistencia {i+1}' if i == 0 else '')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: ADX con DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[start_idx:]
            adx = signal_data['indicators']['adx'][-50:] if len(signal_data['indicators']['adx']) >= 50 else signal_data['indicators']['adx']
            plus_di = signal_data['indicators']['plus_di'][-50:] if len(signal_data['indicators']['plus_di']) >= 50 else signal_data['indicators']['plus_di']
            minus_di = signal_data['indicators']['minus_di'][-50:] if len(signal_data['indicators']['minus_di']) >= 50 else signal_data['indicators']['minus_di']
            
            ax2.plot(adx_dates, adx, 'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, plus_di, 'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, minus_di, 'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Umbral 25')
        
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Volumen con Anomalías
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[start_idx:]
            volumes = [d['volume'] for d in signal_data['data'][start_idx:]]
            
            # Colores para barras de volumen
            bar_colors = []
            for i in range(start_idx, len(dates)):
                if i > 0 and i < len(closes):
                    bar_colors.append('green' if closes[i] > closes[i-1] else 'red')
                else:
                    bar_colors.append('gray')
            
            ax3.bar(volume_dates, volumes, color=bar_colors, alpha=0.6, width=0.8)
            
            # Anomalías de volumen
            if 'volume_anomaly' in signal_data['indicators']:
                anomaly_dates = []
                anomaly_values = []
                
                for i in range(len(volume_dates)):
                    idx = start_idx + i
                    if idx < len(signal_data['indicators']['volume_anomaly']):
                        if signal_data['indicators']['volume_anomaly'][idx]:
                            anomaly_dates.append(volume_dates[i])
                            anomaly_values.append(volumes[i])
                
                if anomaly_dates:
                    ax3.scatter(anomaly_dates, anomaly_values, color='black', s=50, 
                              marker='o', zorder=5, label='Anomalías')
            
            # EMA de volumen
            if 'volume_ema' in signal_data['indicators']:
                volume_ema = signal_data['indicators']['volume_ema'][-50:] if len(signal_data['indicators']['volume_ema']) >= 50 else signal_data['indicators']['volume_ema']
                ax3.plot(volume_dates, volume_ema, 'yellow', linewidth=1, label='EMA Volumen')
        
        ax3.set_ylabel('Volumen')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Fuerza de Tendencia Maverick
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[start_idx:]
            trend_strength = signal_data['indicators']['trend_strength'][-50:] if len(signal_data['indicators']['trend_strength']) >= 50 else signal_data['indicators']['trend_strength']
            
            # Graficar como columnas
            bar_colors_trend = []
            for val in trend_strength:
                bar_colors_trend.append('green' if val > 0 else 'red')
            
            ax4.bar(trend_dates, trend_strength, color=bar_colors_trend, alpha=0.7, width=0.8)
            
            # Umbral
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label=f'Umbral ({threshold:.1f}%)')
                ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            # Zonas no operar
            if 'no_trade_zones' in signal_data['indicators']:
                no_trade_dates = []
                for i in range(len(trend_dates)):
                    idx = start_idx + i
                    if idx < len(signal_data['indicators']['no_trade_zones']):
                        if signal_data['indicators']['no_trade_zones'][idx]:
                            no_trade_dates.append(trend_dates[i])
                
                for date in no_trade_dates:
                    ax4.axvline(x=date, color='red', alpha=0.3, linewidth=2)
        
        ax4.set_ylabel('Fuerza Tendencia %')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Ballenas
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[start_idx:]
            whale_pump = signal_data['indicators']['whale_pump'][-50:] if len(signal_data['indicators']['whale_pump']) >= 50 else signal_data['indicators']['whale_pump']
            whale_dump = signal_data['indicators']['whale_dump'][-50:] if len(signal_data['indicators']['whale_dump']) >= 50 else signal_data['indicators']['whale_dump']
            
            width = 0.4
            ax5.bar([d - timedelta(hours=width/2) for d in whale_dates], 
                   whale_pump, width=width, color='green', alpha=0.7, label='Compra')
            ax5.bar([d + timedelta(hours=width/2) for d in whale_dates], 
                   whale_dump, width=width, color='red', alpha=0.7, label='Venta')
        
        ax5.set_ylabel('Ballenas')
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: RSI Maverick
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_maverick_dates = dates[start_idx:]
            rsi_maverick = signal_data['indicators']['rsi_maverick'][-50:] if len(signal_data['indicators']['rsi_maverick']) >= 50 else signal_data['indicators']['rsi_maverick']
            
            ax6.plot(rsi_maverick_dates, rsi_maverick, 'blue', linewidth=2)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        
        ax6.set_ylabel('RSI Maverick')
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: RSI Tradicional
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_traditional_dates = dates[start_idx:]
            rsi_traditional = signal_data['indicators']['rsi_traditional'][-50:] if len(signal_data['indicators']['rsi_traditional']) >= 50 else signal_data['indicators']['rsi_traditional']
            
            ax7.plot(rsi_traditional_dates, rsi_traditional, 'cyan', linewidth=2)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        ax7.set_ylabel('RSI Tradicional')
        ax7.legend(loc='upper left', fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: MACD
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[start_idx:]
            macd = signal_data['indicators']['macd'][-50:] if len(signal_data['indicators']['macd']) >= 50 else signal_data['indicators']['macd']
            macd_signal = signal_data['indicators']['macd_signal'][-50:] if len(signal_data['indicators']['macd_signal']) >= 50 else signal_data['indicators']['macd_signal']
            macd_histogram = signal_data['indicators']['macd_histogram'][-50:] if len(signal_data['indicators']['macd_histogram']) >= 50 else signal_data['indicators']['macd_histogram']
            
            ax8.plot(macd_dates, macd, 'blue', linewidth=1, label='MACD')
            ax8.plot(macd_dates, macd_signal, 'red', linewidth=1, label='Señal')
            
            # Histograma como columnas
            hist_colors = ['green' if x > 0 else 'red' for x in macd_histogram]
            ax8.bar(macd_dates, macd_histogram, color=hist_colors, alpha=0.6, width=0.8, label='Histograma')
            
            ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax8.set_ylabel('MACD')
        ax8.legend(loc='upper left', fontsize=8)
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
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])[:6]])}
        """
        
        ax9.text(0.1, 0.9, signal_info, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png', 
                        as_attachment=True, 
                        download_name=f'report_{symbol}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        
    except Exception as e:
        print(f"Error generando reporte: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generando reporte: {str(e)}'}), 500

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
