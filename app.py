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

# Top 10 para estrategia de volumen (excluyendo DOGE)
TOP10_VOLUME_STRATEGY = [
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "LTC-USDT"
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

# Pesos por temporalidad
WEIGHTS_CONFIG = {
    # 15m, 30m, 1h, 2h, 4h, 8h
    'intraday': {
        'multi_timeframe': 30,
        'trend_strength': 25,
        'whale_signal': 0,
        'ma_cross': 10,
        'di_crossover': 10,
        'adx_slope': 5,
        'bollinger_bands': 8,
        'macd_crossover': 10,
        'volume_anomaly': 7,
        'rsi_maverick_div': 8,
        'rsi_traditional_div': 5,
        'chart_pattern': 5,
        'breakout': 5
    },
    # 12h, 1D
    'swing': {
        'multi_timeframe': 0,
        'trend_strength': 25,
        'whale_signal': 30,
        'ma_cross': 10,
        'di_crossover': 10,
        'adx_slope': 5,
        'bollinger_bands': 8,
        'macd_crossover': 10,
        'volume_anomaly': 7,
        'rsi_maverick_div': 8,
        'rsi_traditional_div': 5,
        'chart_pattern': 5,
        'breakout': 5
    },
    # 1W
    'weekly': {
        'multi_timeframe': 0,
        'trend_strength': 55,
        'whale_signal': 0,
        'ma_cross': 10,
        'di_crossover': 10,
        'adx_slope': 5,
        'bollinger_bands': 8,
        'macd_crossover': 10,
        'volume_anomaly': 7,
        'rsi_maverick_div': 8,
        'rsi_traditional_div': 5,
        'chart_pattern': 5,
        'breakout': 5
    }
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_operations = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.signal_tracker = {}  # Para rastrear señales recientes
        
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
        """Calcular Average True Range manualmente"""
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

    def calculate_dynamic_support_resistance(self, high, low, close, period=50, num_levels=6):
        """Calcular soportes y resistencias dinámicos usando pivotes"""
        n = len(close)
        supports = []
        resistances = []
        
        # Calcular pivotes para cada período
        for i in range(period, n, period):
            window_high = high[i-period:i]
            window_low = low[i-period:i]
            window_close = close[i-period:i]
            
            # Pivot point clásico
            pivot = (np.max(window_high) + np.min(window_low) + window_close[-1]) / 3
            
            # Niveles de soporte y resistencia
            r1 = (2 * pivot) - np.min(window_low)
            s1 = (2 * pivot) - np.max(window_high)
            r2 = pivot + (np.max(window_high) - np.min(window_low))
            s2 = pivot - (np.max(window_high) - np.min(window_low))
            r3 = np.max(window_high) + 2 * (pivot - np.min(window_low))
            s3 = np.min(window_low) - 2 * (np.max(window_high) - pivot)
            
            # Añadir niveles únicos
            for level in [s3, s2, s1, pivot, r1, r2, r3]:
                if i == period:
                    if level not in supports and level not in resistances:
                        if level < pivot:
                            supports.append(level)
                        else:
                            resistances.append(level)
                else:
                    # Filtrar niveles cercanos para evitar duplicados
                    min_dist = (np.max(window_high) - np.min(window_low)) * 0.05
                    if level < pivot:
                        existing = any(abs(s - level) < min_dist for s in supports)
                        if not existing:
                            supports.append(level)
                    else:
                        existing = any(abs(r - level) < min_dist for r in resistances)
                        if not existing:
                            resistances.append(level)
        
        # Ordenar y limitar a num_levels
        supports = sorted(supports, reverse=True)[:num_levels]
        resistances = sorted(resistances)[:num_levels]
        
        # Para el período actual, encontrar los más cercanos
        current_price = close[-1]
        current_supports = [s for s in supports if s < current_price]
        current_resistances = [r for r in resistances if r > current_price]
        
        # Asegurar al menos 1 soporte y 1 resistencia
        if not current_supports:
            current_supports = [np.min(close[-period:]) * 0.98]
        if not current_resistances:
            current_resistances = [np.max(close[-period:]) * 1.02]
        
        return current_supports, current_resistances

    def calculate_optimal_entry_exit_with_sr(self, df, signal_type, supports, resistances, leverage=15):
        """Calcular entradas y salidas óptimas con soportes/resistencias dinámicos"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Usar soportes y resistencias dinámicos
            if not supports or not resistances:
                supports = [np.min(low[-50:]) * 0.98]
                resistances = [np.max(high[-50:]) * 1.02]
            
            # Encontrar niveles más relevantes
            relevant_support = max([s for s in supports if s < current_price], default=supports[-1] if supports else current_price * 0.95)
            relevant_resistance = min([r for r in resistances if r > current_price], default=resistances[0] if resistances else current_price * 1.05)
            
            # Detectar rupturas
            is_breakout_up = current_price > relevant_resistance and high[-1] > high[-2]
            is_breakout_down = current_price < relevant_support and low[-1] < low[-2]
            
            if signal_type == 'LONG':
                if is_breakout_up:
                    # Entrada después de ruptura con pullback al soporte más cercano
                    entry = max(relevant_support, current_price - (current_atr * 0.5))
                    stop_loss = max(relevant_support * 0.97, entry - (current_atr * 1.5))
                    
                    # Take profits en resistencias superiores
                    higher_resistances = [r for r in resistances if r > entry]
                    if higher_resistances:
                        take_profit = [min(higher_resistances[:3])]
                    else:
                        take_profit = [entry + (2 * (entry - stop_loss))]
                else:
                    # Entrada en soporte para corrección
                    entry = relevant_support
                    stop_loss = max(relevant_support * 0.97, entry - (current_atr * 2))
                    
                    # Take profit en resistencia
                    take_profit = [relevant_resistance]
                    
            else:  # SHORT
                if is_breakout_down:
                    # Entrada después de ruptura con pullback a resistencia más cercana
                    entry = min(relevant_resistance, current_price + (current_atr * 0.5))
                    stop_loss = min(relevant_resistance * 1.03, entry + (current_atr * 1.5))
                    
                    # Take profits en soportes inferiores
                    lower_supports = [s for s in supports if s < entry]
                    if lower_supports:
                        take_profit = [max(lower_supports[:3])]
                    else:
                        take_profit = [entry - (2 * (stop_loss - entry))]
                else:
                    # Entrada en resistencia para corrección
                    entry = relevant_resistance
                    stop_loss = min(relevant_resistance * 1.03, entry + (current_atr * 2))
                    
                    # Take profit en soporte
                    take_profit = [relevant_support]
            
            # Asegurar ratios de riesgo adecuados
            risk = abs(entry - stop_loss)
            reward = abs(take_profit[0] - entry)
            
            if reward / risk < 1.5:
                # Ajustar take profit para mejor ratio
                if signal_type == 'LONG':
                    take_profit = [entry + (risk * 2)]
                else:
                    take_profit = [entry - (risk * 2)]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profit],
                'supports': [float(s) for s in supports],
                'resistances': [float(r) for r in resistances],
                'atr': float(current_atr),
                'atr_percentage': float(current_atr / current_price),
                'is_breakout': is_breakout_up if signal_type == 'LONG' else is_breakout_down
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'supports': [current_price * 0.95, current_price * 0.92, current_price * 0.89],
                'resistances': [current_price * 1.02, current_price * 1.05, current_price * 1.08],
                'atr': 0.0,
                'atr_percentage': 0.0,
                'is_breakout': False
            }

    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[period-1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        # Rellenar primeros valores
        ema[:period-1] = ema[period-1]
        
        return ema

    def calculate_sma(self, prices, period):
        """Calcular SMA manualmente"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        sma = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
        
        # Rellenar primeros valores
        sma[:period-1] = sma[period-1]
        
        return sma

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger manualmente"""
        if len(prices) < period:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        sma = self.calculate_sma(prices, period)
        std = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            window = prices[i-period+1:i+1]
            std[i] = np.std(window)
        
        # Rellenar primeros valores
        std[:period-1] = std[period-1]
        
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
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        # Primeros periodos
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.zeros(len(prices))
        rsi = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            if avg_losses[i] > 0:
                rs[i] = avg_gains[i] / avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100 if avg_gains[i] > 0 else 50
        
        rsi[:period] = 50
        
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
            
            # Rellenar primeros valores
            if length-1 > 0:
                dev[:length-1] = dev[length-1]
            
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
            
            # Calcular umbral dinámico
            if n >= 50:
                historical_bb_width = bb_width[max(0, n-100):n]
                high_zone_threshold = np.percentile(historical_bb_width, 70)
            else:
                high_zone_threshold = np.percentile(bb_width[bb_width > 0], 70) if len(bb_width[bb_width > 0]) > 0 else 5
            
            no_trade_zones = np.zeros(n, dtype=bool)
            strength_signals = ['NEUTRAL'] * n
            
            for i in range(10, n):
                # Zona de no operar: ancho alto y fuerza decreciente
                if (bb_width[i] > high_zone_threshold and 
                    trend_strength[i] < 0 and 
                    bb_width[i] < np.max(bb_width[max(0, i-10):i])):
                    no_trade_zones[i] = True
                
                # Señales de fuerza
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
                return True
                
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False
            
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            if signal_type == 'LONG':
                # TF Mayor: Alcista o Neutral
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                # TF Media: Alcista
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
                # TF Media: Bajista
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
        """Detectar divergencias entre precio e indicador con duración de 7 velas"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            price_window = price[i-lookback:i+1]
            indicator_window = indicator[i-lookback:i+1]
            
            # Divergencia alcista: precio hace mínimo más bajo, indicador hace mínimo más alto
            if (price[i] < np.min(price_window[:-1]) and 
                indicator[i] > np.min(indicator_window[:-1])):
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace máximo más alto, indicador hace máximo más bajo
            if (price[i] > np.max(price_window[:-1]) and 
                indicator[i] < np.max(indicator_window[:-1])):
                bearish_div[i] = True
        
        # Extender señal por 7 velas
        bullish_extended = bullish_div.copy()
        bearish_extended = bearish_div.copy()
        
        for i in range(n):
            if bullish_div[i]:
                for j in range(1, min(8, n-i)):
                    bullish_extended[i+j] = True
            if bearish_div[i]:
                for j in range(1, min(8, n-i)):
                    bearish_extended[i+j] = True
        
        return bullish_extended.tolist(), bearish_extended.tolist()

    def check_breakout(self, high, low, close, support, resistance):
        """Detectar rupturas de tendencia"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if close[i] > resistance and high[i] > high[i-1]:
                breakout_up[i] = True
            
            if close[i] < support and low[i] < low[i-1]:
                breakout_down[i] = True
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover(self, plus_di, minus_di):
        """Detectar cruces de +DI y -DI"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: +DI cruza por encima de -DI
            if plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]:
                di_cross_bullish[i] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]:
                di_cross_bearish[i] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

    def check_adx_slope(self, adx, period=3):
        """Verificar si ADX tiene pendiente positiva y está por encima de nivel"""
        n = len(adx)
        adx_slope_positive = np.zeros(n, dtype=bool)
        adx_above_level = np.zeros(n, dtype=bool)
        
        for i in range(period, n):
            # Calcular pendiente
            x = np.arange(period)
            y = adx[i-period:i]
            
            if len(y[y > 0]) >= 2:
                slope, _, _, _, _ = stats.linregress(x, y)
                adx_slope_positive[i] = slope > 0
            
            # Verificar si está por encima de 25
            adx_above_level[i] = adx[i] > 25
        
        return adx_slope_positive.tolist(), adx_above_level.tolist()

    def check_ma_crossover(self, ma_fast, ma_slow):
        """Detectar cruce de medias móviles"""
        n = len(ma_fast)
        ma_cross_up = np.zeros(n, dtype=bool)
        ma_cross_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MA rápida cruza por encima de MA lenta
            if ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]:
                ma_cross_up[i] = True
            
            # Cruce bajista: MA rápida cruza por debajo de MA lenta
            if ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]:
                ma_cross_down[i] = True
        
        return ma_cross_up.tolist(), ma_cross_down.tolist()

    def check_macd_crossover(self, macd, signal):
        """Detectar cruce de MACD"""
        n = len(macd)
        macd_cross_up = np.zeros(n, dtype=bool)
        macd_cross_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MACD cruza por encima de la señal
            if macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
                macd_cross_up[i] = True
            
            # Cruce bajista: MACD cruza por debajo de la señal
            if macd[i] < signal[i] and macd[i-1] >= signal[i-1]:
                macd_cross_down[i] = True
        
        return macd_cross_up.tolist(), macd_cross_down.tolist()

    def check_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo con duración de 7 velas"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'triple_top': np.zeros(n, dtype=bool),
            'triple_bottom': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-7):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            
            # Doble techo
            peaks = []
            for j in range(1, len(window_high)-1):
                if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                    peaks.append((j, window_high[j]))
            
            if len(peaks) >= 2:
                last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                    patterns['double_top'][i] = True
            
            # Doble piso
            troughs = []
            for j in range(1, len(window_low)-1):
                if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                    troughs.append((j, window_low[j]))
            
            if len(troughs) >= 2:
                last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                    patterns['double_bottom'][i] = True
        
        # Extender patrones por 7 velas
        for pattern in patterns:
            extended = patterns[pattern].copy()
            for i in range(n):
                if patterns[pattern][i]:
                    for j in range(1, min(8, n-i)):
                        extended[i+j] = True
            patterns[pattern] = extended
        
        return patterns

    def check_volume_anomaly_directional(self, volume, close, period=20):
        """NUEVO: Detectar anomalías de volumen direccionales (compra/venta)"""
        n = len(volume)
        volume_anomaly_buy = np.zeros(n, dtype=bool)
        volume_anomaly_sell = np.zeros(n, dtype=bool)
        volume_clusters = np.zeros(n, dtype=bool)
        volume_ratio = np.zeros(n)
        
        for i in range(period, n):
            # Media móvil de volumen
            volume_ma = np.mean(volume[i-period:i])
            
            # Ratio volumen actual vs media
            if volume_ma > 0:
                volume_ratio[i] = volume[i] / volume_ma
            else:
                volume_ratio[i] = 1
            
            # Detectar anomalía (> 2.5x media)
            if volume_ratio[i] > 2.5:
                # Determinar dirección basada en precio
                if i > 0:
                    price_change = (close[i] - close[i-1]) / close[i-1] * 100
                    if price_change > 0:  # Precio sube con volumen alto = compra
                        volume_anomaly_buy[i] = True
                    elif price_change < 0:  # Precio baja con volumen alto = venta
                        volume_anomaly_sell[i] = True
                    else:
                        # Si no hay cambio de precio, verificar contexto
                        if close[i] > np.mean(close[i-5:i]):
                            volume_anomaly_buy[i] = True
                        else:
                            volume_anomaly_sell[i] = True
            
            # Detectar clusters (múltiples anomalías en 5 periodos)
            if i >= 5:
                recent_buy = volume_anomaly_buy[max(0, i-4):i+1]
                recent_sell = volume_anomaly_sell[max(0, i-4):i+1]
                
                if np.sum(recent_buy) >= 2:
                    volume_clusters[i] = True
                elif np.sum(recent_sell) >= 2:
                    volume_clusters[i] = True
        
        return {
            'volume_anomaly_buy': volume_anomaly_buy.tolist(),
            'volume_anomaly_sell': volume_anomaly_sell.tolist(),
            'volume_clusters': volume_clusters.tolist(),
            'volume_ratio': volume_ratio.tolist(),
            'volume_ma': self.calculate_sma(volume, period).tolist()
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

    def get_weights_for_timeframe(self, interval):
        """Obtener pesos según temporalidad"""
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            return WEIGHTS_CONFIG['intraday']
        elif interval in ['12h', '1D']:
            return WEIGHTS_CONFIG['swing']
        else:  # 1W
            return WEIGHTS_CONFIG['weekly']

    def evaluate_signal_conditions_corrected(self, data, current_idx, interval):
        """Evaluar condiciones de señal con pesos optimizados"""
        weights = self.get_weights_for_timeframe(interval)
        
        conditions = {
            'long': {},
            'short': {}
        }
        
        # Inicializar condiciones
        for condition, weight in weights.items():
            conditions['long'][condition] = {'value': False, 'weight': weight}
            conditions['short'][condition] = {'value': False, 'weight': weight}
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        # Obtener valores actuales
        current_price = data['close'][current_idx]
        ma_9 = data['ma_9'][current_idx] if current_idx < len(data['ma_9']) else 0
        ma_21 = data['ma_21'][current_idx] if current_idx < len(data['ma_21']) else 0
        ma_50 = data['ma_50'][current_idx] if current_idx < len(data['ma_50']) else 0
        ma_200 = data['ma_200'][current_idx] if current_idx < len(data['ma_200']) else 0
        
        # Trackear señales recientes
        signal_key = f"{data.get('symbol', '')}_{interval}"
        if signal_key not in self.signal_tracker:
            self.signal_tracker[signal_key] = {
                'ma_cross': {'long': False, 'short': False, 'count': 0},
                'di_crossover': {'long': False, 'short': False, 'count': 0},
                'macd_crossover': {'long': False, 'short': False, 'count': 0}
            }
        
        tracker = self.signal_tracker[signal_key]
        
        # CONDICIONES LONG
        # 1. Multi-Timeframe (solo para intraday)
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_long', False)
        else:
            conditions['long']['multi_timeframe']['weight'] = 0
        
        # 2. Fuerza de Tendencia Maverick
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        # 3. Señal de Ballenas (solo para 12h, 1D)
        if interval in ['12h', '1D']:
            # Verificar señal de ballena en las últimas 7 velas
            whale_signal = False
            lookback = min(7, current_idx + 1)
            for i in range(lookback):
                idx = current_idx - i
                if idx >= 0 and data['confirmed_buy'][idx]:
                    whale_signal = True
                    break
            conditions['long']['whale_signal']['value'] = whale_signal
        else:
            conditions['long']['whale_signal']['weight'] = 0
        
        # 4. Cruce de Medias (solo en el cruce y 1 vela después)
        ma_cross_up = data['ma_cross_up'][current_idx] if current_idx < len(data['ma_cross_up']) else False
        if ma_cross_up:
            tracker['ma_cross']['long'] = True
            tracker['ma_cross']['count'] = 2  # Dura 2 velas (cruce + 1)
        
        if tracker['ma_cross']['long'] and tracker['ma_cross']['count'] > 0:
            conditions['long']['ma_cross']['value'] = True
            tracker['ma_cross']['count'] -= 1
            if tracker['ma_cross']['count'] == 0:
                tracker['ma_cross']['long'] = False
        
        # 5. Cruce DI (solo en el cruce y 1 vela después)
        di_cross_bullish = data['di_cross_bullish'][current_idx] if current_idx < len(data['di_cross_bullish']) else False
        if di_cross_bullish:
            tracker['di_crossover']['long'] = True
            tracker['di_crossover']['count'] = 2
        
        if tracker['di_crossover']['long'] and tracker['di_crossover']['count'] > 0:
            conditions['long']['di_crossover']['value'] = True
            tracker['di_crossover']['count'] -= 1
            if tracker['di_crossover']['count'] == 0:
                tracker['di_crossover']['long'] = False
        
        # 6. ADX con pendiente positiva
        adx_slope_positive = data['adx_slope_positive'][current_idx] if current_idx < len(data['adx_slope_positive']) else False
        adx_above_level = data['adx_above_level'][current_idx] if current_idx < len(data['adx_above_level']) else False
        conditions['long']['adx_slope']['value'] = adx_slope_positive and adx_above_level
        
        # 7. Bandas de Bollinger
        conditions['long']['bollinger_bands']['value'] = data.get('bollinger_conditions_long', False)
        
        # 8. Cruce MACD (solo en el cruce y 1 vela después)
        macd_cross_up = data['macd_cross_up'][current_idx] if current_idx < len(data['macd_cross_up']) else False
        if macd_cross_up:
            tracker['macd_crossover']['long'] = True
            tracker['macd_crossover']['count'] = 2
        
        if tracker['macd_crossover']['long'] and tracker['macd_crossover']['count'] > 0:
            conditions['long']['macd_crossover']['value'] = True
            tracker['macd_crossover']['count'] -= 1
            if tracker['macd_crossover']['count'] == 0:
                tracker['macd_crossover']['long'] = False
        
        # 9. Volumen Anómalo
        volume_anomaly_buy = data['volume_anomaly_buy'][current_idx] if current_idx < len(data['volume_anomaly_buy']) else False
        volume_clusters = data['volume_clusters'][current_idx] if current_idx < len(data['volume_clusters']) else False
        conditions['long']['volume_anomaly']['value'] = volume_anomaly_buy or volume_clusters
        
        # 10. Divergencia RSI Maverick
        conditions['long']['rsi_maverick_div']['value'] = (
            data['rsi_maverick_bullish_divergence'][current_idx] if 
            current_idx < len(data['rsi_maverick_bullish_divergence']) else False
        )
        
        # 11. Divergencia RSI Tradicional
        conditions['long']['rsi_traditional_div']['value'] = (
            data['rsi_bullish_divergence'][current_idx] if 
            current_idx < len(data['rsi_bullish_divergence']) else False
        )
        
        # 12. Patrones Chartistas
        chart_pattern = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['triple_bottom'][current_idx]
        )
        conditions['long']['chart_pattern']['value'] = chart_pattern
        
        # 13. Rupturas
        conditions['long']['breakout']['value'] = (
            data['breakout_up'][current_idx] if 
            current_idx < len(data['breakout_up']) else False
        )
        
        # CONDICIONES SHORT
        # 1. Multi-Timeframe (solo para intraday)
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        else:
            conditions['short']['multi_timeframe']['weight'] = 0
        
        # 2. Fuerza de Tendencia Maverick
        conditions['short']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            not data['no_trade_zones'][current_idx]
        )
        
        # 3. Señal de Ballenas (solo para 12h, 1D)
        if interval in ['12h', '1D']:
            # Verificar señal de ballena en las últimas 7 velas
            whale_signal = False
            lookback = min(7, current_idx + 1)
            for i in range(lookback):
                idx = current_idx - i
                if idx >= 0 and data['confirmed_sell'][idx]:
                    whale_signal = True
                    break
            conditions['short']['whale_signal']['value'] = whale_signal
        else:
            conditions['short']['whale_signal']['weight'] = 0
        
        # 4. Cruce de Medias (solo en el cruce y 1 vela después)
        ma_cross_down = data['ma_cross_down'][current_idx] if current_idx < len(data['ma_cross_down']) else False
        if ma_cross_down:
            tracker['ma_cross']['short'] = True
            tracker['ma_cross']['count'] = 2
        
        if tracker['ma_cross']['short'] and tracker['ma_cross']['count'] > 0:
            conditions['short']['ma_cross']['value'] = True
            tracker['ma_cross']['count'] -= 1
            if tracker['ma_cross']['count'] == 0:
                tracker['ma_cross']['short'] = False
        
        # 5. Cruce DI (solo en el cruce y 1 vela después)
        di_cross_bearish = data['di_cross_bearish'][current_idx] if current_idx < len(data['di_cross_bearish']) else False
        if di_cross_bearish:
            tracker['di_crossover']['short'] = True
            tracker['di_crossover']['count'] = 2
        
        if tracker['di_crossover']['short'] and tracker['di_crossover']['count'] > 0:
            conditions['short']['di_crossover']['value'] = True
            tracker['di_crossover']['count'] -= 1
            if tracker['di_crossover']['count'] == 0:
                tracker['di_crossover']['short'] = False
        
        # 6. ADX con pendiente positiva
        conditions['short']['adx_slope']['value'] = adx_slope_positive and adx_above_level
        
        # 7. Bandas de Bollinger
        conditions['short']['bollinger_bands']['value'] = data.get('bollinger_conditions_short', False)
        
        # 8. Cruce MACD (solo en el cruce y 1 vela después)
        macd_cross_down = data['macd_cross_down'][current_idx] if current_idx < len(data['macd_cross_down']) else False
        if macd_cross_down:
            tracker['macd_crossover']['short'] = True
            tracker['macd_crossover']['count'] = 2
        
        if tracker['macd_crossover']['short'] and tracker['macd_crossover']['count'] > 0:
            conditions['short']['macd_crossover']['value'] = True
            tracker['macd_crossover']['count'] -= 1
            if tracker['macd_crossover']['count'] == 0:
                tracker['macd_crossover']['short'] = False
        
        # 9. Volumen Anómalo
        volume_anomaly_sell = data['volume_anomaly_sell'][current_idx] if current_idx < len(data['volume_anomaly_sell']) else False
        conditions['short']['volume_anomaly']['value'] = volume_anomaly_sell or volume_clusters
        
        # 10. Divergencia RSI Maverick
        conditions['short']['rsi_maverick_div']['value'] = (
            data['rsi_maverick_bearish_divergence'][current_idx] if 
            current_idx < len(data['rsi_maverick_bearish_divergence']) else False
        )
        
        # 11. Divergencia RSI Tradicional
        conditions['short']['rsi_traditional_div']['value'] = (
            data['rsi_bearish_divergence'][current_idx] if 
            current_idx < len(data['rsi_bearish_divergence']) else False
        )
        
        # 12. Patrones Chartistas
        chart_pattern = (
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['head_shoulders'][current_idx] or
            data['chart_patterns']['triple_top'][current_idx]
        )
        conditions['short']['chart_pattern']['value'] = chart_pattern
        
        # 13. Rupturas
        conditions['short']['breakout']['value'] = (
            data['breakout_down'][current_idx] if 
            current_idx < len(data['breakout_down']) else False
        )
        
        # Actualizar condición MA200
        ma200_condition = 'above' if current_price > ma_200 else 'below'
        
        return conditions, ma200_condition

    def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal con umbrales optimizados"""
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        signal_conditions = conditions.get(signal_type, {})
        
        # Verificar condiciones obligatorias según temporalidad
        obligatory_conditions = []
        for condition, data in signal_conditions.items():
            if data['weight'] >= 25:  # Condiciones con peso >= 25 son obligatorias
                obligatory_conditions.append(condition)
        
        # Verificar que todas las condiciones obligatorias se cumplan
        all_obligatory_met = True
        for cond in obligatory_conditions:
            if not signal_conditions[cond]['value']:
                all_obligatory_met = False
                break
        
        if not all_obligatory_met:
            return 0, []
        
        for condition, data in signal_conditions.items():
            total_weight += data['weight']
            if data['value']:
                achieved_weight += data['weight']
                fulfilled_conditions.append(condition)
        
        if total_weight == 0:
            return 0, []
        
        base_score = (achieved_weight / total_weight * 100)
        
        # Aplicar umbral según MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:  # short
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0

        return min(final_score, 100), fulfilled_conditions

    def get_condition_description(self, condition_key):
        """Obtener descripción de condición para Telegram"""
        descriptions = {
            'multi_timeframe': 'Multi-TF obligatorio',
            'trend_strength': 'Fuerza tendencia Maverick',
            'whale_signal': 'Señal ballenas confirmada',
            'ma_cross': 'Cruce de medias 9/21',
            'di_crossover': 'Cruce DMI (+DI/-DI)',
            'adx_slope': 'ADX con pendiente positiva',
            'bollinger_bands': 'Bandas de Bollinger',
            'macd_crossover': 'Cruce MACD',
            'volume_anomaly': 'Volumen anómalo',
            'rsi_maverick_div': 'Divergencia RSI Maverick',
            'rsi_traditional_div': 'Divergencia RSI Tradicional',
            'chart_pattern': 'Patrón Chartista',
            'breakout': 'Ruptura confirmada'
        }
        return descriptions.get(condition_key, condition_key)

    def get_chart_pattern_name(self, patterns, current_idx):
        """Obtener nombre del patrón chartista activo"""
        if patterns['head_shoulders'][current_idx]:
            return 'Hombro-Cabeza-Hombro'
        elif patterns['double_top'][current_idx]:
            return 'Doble techo'
        elif patterns['double_bottom'][current_idx]:
            return 'Doble piso'
        elif patterns['triple_top'][current_idx]:
            return 'Triple techo'
        elif patterns['triple_bottom'][current_idx]:
            return 'Triple piso'
        return ''

    def check_volume_ema_ftm_signal(self, symbol, interval):
        """Nueva estrategia: Desplome por Volumen + EMA21 con Filtros FTMaverick/Multi-Timeframe"""
        try:
            # Solo para top 10 criptos (excluyendo DOGE)
            if symbol not in TOP10_VOLUME_STRATEGY:
                return None
            
            # Solo para intervalos especificados
            if interval not in ['15m', '30m', '1h', '4h', '12h', '1D']:
                return None
            
            # Verificar timing de ejecución
            current_time = self.get_bolivia_time()
            should_check = False
            
            if interval == '1h':
                # Revisar desde el 50% del tiempo de vela, cada 300 segundos
                next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                elapsed = (next_close - current_time).total_seconds()
                if elapsed <= 1800:  # 50% de 1 hora
                    should_check = True
            elif interval == '4h':
                # Revisar desde el 25% del tiempo de vela, cada 420 segundos
                current_hour = current_time.hour
                next_4h_close = current_time.replace(minute=0, second=0, microsecond=0)
                remainder = current_hour % 4
                if remainder == 0:
                    next_4h_close += timedelta(hours=4)
                else:
                    next_4h_close += timedelta(hours=4 - remainder)
                elapsed = (next_4h_close - current_time).total_seconds()
                if elapsed <= 3600:  # 25% de 4 horas
                    should_check = True
            elif interval in ['12h', '1D']:
                # Revisar desde el 25% del tiempo de vela, cada 600 segundos
                should_check = True
            
            if not should_check:
                return None
            
            # Obtener datos
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular EMA21 de precio
            ema_21 = self.calculate_ema(close, 21)
            
            # Calcular Media Móvil de Volumen (21 periodos)
            volume_ma_21 = self.calculate_sma(volume, 21)
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            current_volume_ma = volume_ma_21[current_idx]
            current_ema = ema_21[current_idx]
            
            # Condición A: Volumen y EMA
            volume_condition = current_volume > (current_volume_ma * 2.5)
            if not volume_condition:
                return None
            
            # Determinar dirección
            if current_price > current_ema:
                signal_type = 'LONG'
            elif current_price < current_ema:
                signal_type = 'SHORT'
            else:
                return None
            
            # Condición B: Filtro FTMaverick (timeframe actual)
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][current_idx]:
                return None
            
            # Condición C: Filtro Multi-Timeframe
            if interval not in TIMEFRAME_HIERARCHY:
                return None
            
            hierarchy = TIMEFRAME_HIERARCHY[interval]
            
            # Timeframe Mayor
            mayor_tf = hierarchy.get('mayor')
            if mayor_tf:
                mayor_df = self.get_kucoin_data(symbol, mayor_tf, 50)
                if mayor_df is not None and len(mayor_df) > 20:
                    mayor_close = mayor_df['close'].values
                    mayor_ftm = self.calculate_trend_strength_maverick(mayor_close)
                    mayor_strength = mayor_ftm['strength_signals'][-1]
                else:
                    mayor_strength = 'NEUTRAL'
            else:
                mayor_strength = 'NEUTRAL'
            
            # Timeframe Menor
            menor_tf = hierarchy.get('menor')
            if menor_tf:
                menor_df = self.get_kucoin_data(symbol, menor_tf, 30)
                if menor_df is not None and len(menor_df) > 10:
                    menor_close = menor_df['close'].values
                    menor_ftm = self.calculate_trend_strength_maverick(menor_close)
                    menor_strength = menor_ftm['strength_signals'][-1]
                else:
                    menor_strength = 'NEUTRAL'
            else:
                menor_strength = 'NEUTRAL'
            
            # Verificar condiciones multi-timeframe
            if signal_type == 'LONG':
                if not (mayor_strength in ['BULLISH', 'NEUTRAL']):
                    return None
                if not (menor_strength in ['STRONG_UP', 'WEAK_UP']):
                    return None
            else:  # SHORT
                if not (mayor_strength in ['BEARISH', 'NEUTRAL']):
                    return None
                if not (menor_strength in ['STRONG_DOWN', 'WEAK_DOWN']):
                    return None
            
            # Si pasa todos los filtros, devolver señal
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'close_price': current_price,
                'ema_21': current_ema,
                'volume_ratio': current_volume / current_volume_ma,
                'mayor_trend': mayor_strength,
                'menor_trend': menor_strength,
                'ftm_no_trade': ftm_data['no_trade_zones'][current_idx]
            }
            
        except Exception as e:
            print(f"Error en check_volume_ema_ftm_signal para {symbol}: {e}")
            return None

    def generate_volume_ema_ftm_alerts(self):
        """Generar alertas para la nueva estrategia de volumen"""
        alerts = []
        
        for symbol in TOP10_VOLUME_STRATEGY:
            for interval in ['15m', '30m', '1h', '4h', '12h', '1D']:
                try:
                    signal = self.check_volume_ema_ftm_signal(symbol, interval)
                    if signal:
                        # Verificar que no sea una alerta duplicada reciente
                        alert_key = f"{symbol}_{interval}_{signal['signal']}"
                        if (alert_key not in self.alert_cache or 
                            (datetime.now() - self.alert_cache[alert_key]).seconds > 300):
                            
                            alerts.append(signal)
                            self.alert_cache[alert_key] = datetime.now()
                except Exception as e:
                    print(f"Error generando alerta volumen para {symbol} {interval}: {e}")
                    continue
        
        return alerts

    def generate_telegram_chart_image_main_strategy(self, symbol, interval, signal_data):
        """Generar imagen para Telegram de la estrategia principal"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calcular todos los indicadores
            whale_data = self.calculate_whale_signals_improved(df)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, 14)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            adx_slope_positive, adx_above_level = self.check_adx_slope(adx)
            
            rsi_maverick = self.calculate_rsi_maverick(close)
            rsi_traditional = self.calculate_rsi(close, 14)
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_cross_up, ma_cross_down = self.check_ma_crossover(ma_9, ma_21)
            
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_up, macd_cross_down = self.check_macd_crossover(macd, macd_signal)
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            supports, resistances = self.calculate_dynamic_support_resistance(high, low, close)
            breakout_up, breakout_down = self.check_breakout(high, low, close, supports[0] if supports else close[-1]*0.95, 
                                                           resistances[0] if resistances else close[-1]*1.05)
            
            chart_patterns = self.check_chart_patterns(high, low, close)
            
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            volume_anomaly_data = self.check_volume_anomaly_directional(volume, close)
            
            # Crear figura con fondo blanco
            plt.style.use('default')
            fig = plt.figure(figsize=(16, 20), facecolor='white')
            
            # 1. Gráfico de velas con Bollinger y medias
            ax1 = plt.subplot(8, 1, 1)
            dates = pd.date_range(end=datetime.now(), periods=len(close), freq=interval)
            
            # Plot velas
            for i in range(len(close)):
                color = 'green' if close[i] >= (close[i-1] if i > 0 else close[i]) else 'red'
                ax1.plot([dates[i], dates[i]], [low[i], high[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [close[i] if close[i] < (close[i-1] if i > 0 else close[i]) else (close[i-1] if i > 0 else close[i]), 
                         close[i] if close[i] >= (close[i-1] if i > 0 else close[i]) else (close[i-1] if i > 0 else close[i])], 
                        color=color, linewidth=3)
            
            # Plot Bollinger Bands (transparentes)
            ax1.plot(dates, bb_upper, color='orange', alpha=0.3, linewidth=1)
            ax1.plot(dates, bb_middle, color='orange', alpha=0.5, linewidth=1)
            ax1.plot(dates, bb_lower, color='orange', alpha=0.3, linewidth=1)
            
            # Plot medias móviles
            ax1.plot(dates, ma_9, color='blue', alpha=0.7, linewidth=1, label='MA9')
            ax1.plot(dates, ma_21, color='red', alpha=0.7, linewidth=1, label='MA21')
            ax1.plot(dates, self.calculate_sma(close, 50), color='purple', alpha=0.7, linewidth=1, label='MA50')
            ax1.plot(dates, self.calculate_sma(close, 200), color='black', alpha=0.7, linewidth=2, label='MA200')
            
            # Plot soportes y resistencias
            for level in supports[-4:]:
                ax1.axhline(y=level, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            for level in resistances[:4]:
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            ax1.set_title(f'{symbol} - {interval} - Velas con Indicadores', fontsize=12)
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. ADX con DMI
            ax2 = plt.subplot(8, 1, 2)
            ax2.plot(dates, adx, color='black', linewidth=2, label='ADX')
            ax2.plot(dates, plus_di, color='green', linewidth=1, label='+DI')
            ax2.plot(dates, minus_di, color='red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax2.set_title('ADX con DMI (+DI/-DI)', fontsize=12)
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Indicador de Volumen con Anomalías
            ax3 = plt.subplot(8, 1, 3)
            colors = ['green' if buy else 'red' if sell else 'gray' 
                     for buy, sell in zip(volume_anomaly_data['volume_anomaly_buy'], 
                                         volume_anomaly_data['volume_anomaly_sell'])]
            ax3.bar(dates, volume, color=colors, alpha=0.7)
            ax3.plot(dates, volume_anomaly_data['volume_ma'], color='yellow', linewidth=1.5, label='MA Volumen')
            ax3.set_title('Indicador de Volumen con Anomalías', fontsize=12)
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # 4. Fuerza de Tendencia Maverick (barras)
            ax4 = plt.subplot(8, 1, 4)
            colors_bars = ['green' if x > 0 else 'red' for x in trend_strength_data['trend_strength']]
            ax4.bar(dates, trend_strength_data['trend_strength'], color=colors_bars, alpha=0.7)
            ax4.axhline(y=trend_strength_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax4.axhline(y=-trend_strength_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax4.set_title('Fuerza de Tendencia Maverick', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            # 5. Indicador de Ballenas (barras)
            ax5 = plt.subplot(8, 1, 5)
            ax5.bar(dates, whale_data['whale_pump'], color='green', alpha=0.7, label='Compradoras')
            ax5.bar(dates, whale_data['whale_dump'], color='red', alpha=0.7, label='Vendedoras')
            ax5.set_title('Indicador Ballenas Compradoras/Vendedoras', fontsize=12)
            ax5.legend(loc='upper left', fontsize=8)
            ax5.grid(True, alpha=0.3)
            
            # 6. RSI Modificado Maverick
            ax6 = plt.subplot(8, 1, 6)
            ax6.plot(dates, rsi_maverick, color='blue', linewidth=2)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            ax6.set_title('RSI Modificado Maverick (%B)', fontsize=12)
            ax6.grid(True, alpha=0.3)
            
            # 7. RSI Tradicional con Divergencias
            ax7 = plt.subplot(8, 1, 7)
            ax7.plot(dates, rsi_traditional, color='purple', linewidth=2)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            ax7.set_title('RSI Tradicional con Divergencias', fontsize=12)
            ax7.grid(True, alpha=0.3)
            
            # 8. MACD con Histograma (barras)
            ax8 = plt.subplot(8, 1, 8)
            ax8.plot(dates, macd, color='blue', linewidth=1, label='MACD')
            ax8.plot(dates, macd_signal, color='red', linewidth=1, label='Señal')
            colors_hist = ['green' if x > 0 else 'red' for x in macd_histogram]
            ax8.bar(dates, macd_histogram, color=colors_hist, alpha=0.5, label='Histograma')
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
            ax8.set_title('MACD con Histograma', fontsize=12)
            ax8.legend(loc='upper left', fontsize=8)
            ax8.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar a buffer
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, facecolor='white')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando imagen Telegram para {symbol}: {e}")
            return None

    def generate_telegram_chart_image_volume_strategy(self, symbol, interval, signal_data):
        """Generar imagen para Telegram de la estrategia de volumen"""
        try:
            df = self.get_kucoin_data(symbol, interval, 50)
            if df is None or len(df) < 30:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calcular EMA21 y Volume MA21
            ema_21 = self.calculate_ema(close, 21)
            volume_ma_21 = self.calculate_sma(volume, 21)
            
            dates = pd.date_range(end=datetime.now(), periods=len(close), freq=interval)
            
            # Crear figura doble
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor='white')
            
            # Gráfico superior: Velas + EMA21
            for i in range(len(close)):
                color = 'green' if close[i] >= (close[i-1] if i > 0 else close[i]) else 'red'
                ax1.plot([dates[i], dates[i]], [low[i], high[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [close[i] if close[i] < (close[i-1] if i > 0 else close[i]) else (close[i-1] if i > 0 else close[i]), 
                         close[i] if close[i] >= (close[i-1] if i > 0 else close[i]) else (close[i-1] if i > 0 else close[i])], 
                        color=color, linewidth=3)
            
            # EMA21
            ax1.plot(dates, ema_21, color='orange', linewidth=2, label='EMA21')
            
            # Destacar vela actual
            signal_color = 'green' if signal_data['signal'] == 'LONG' else 'red'
            ax1.scatter([dates[-1]], [close[-1]], color=signal_color, s=100, 
                       edgecolors='black', zorder=5, label=f'Señal {signal_data["signal"]}')
            
            ax1.set_title(f'{symbol} - {interval} - Señal {signal_data["signal"]} por Volumen+EMA21', fontsize=14)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico inferior: Volumen + Volume MA21
            volume_colors = ['green' if close[i] >= (close[i-1] if i > 0 else close[i]) else 'red' for i in range(len(volume))]
            ax2.bar(dates, volume, color=volume_colors, alpha=0.7)
            ax2.plot(dates, volume_ma_21, color='yellow', linewidth=2, label='MA Volumen 21')
            
            # Destacar volumen actual
            ax2.bar([dates[-1]], [volume[-1]], color=signal_color, alpha=1, edgecolor='black', linewidth=2)
            
            ax2.set_title('Volumen + Media Móvil 21 periodos', fontsize=12)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar a buffer
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, facecolor='white')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando imagen volumen para {symbol}: {e}")
            return None

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - CON PESOS OPTIMIZADOS"""
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
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            adx_slope_positive, adx_above_level = self.check_adx_slope(adx)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            # Medias móviles y cruces
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            ma_cross_up, ma_cross_down = self.check_ma_crossover(ma_9, ma_21)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_up, macd_cross_down = self.check_macd_crossover(macd, macd_signal)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            
            # Verificar condiciones de Bollinger
            current_idx = -1
            current_price = close[current_idx]
            bollinger_conditions_long = current_price <= bb_lower[current_idx] * 1.02
            bollinger_conditions_short = current_price >= bb_upper[current_idx] * 0.98
            
            # Soporte y resistencia dinámicos
            supports, resistances = self.calculate_dynamic_support_resistance(high, low, close, sr_period, 6)
            breakout_up, breakout_down = self.check_breakout(high, low, close, 
                                                           supports[0] if supports else np.min(low[-sr_period:]), 
                                                           resistances[0] if resistances else np.max(high[-sr_period:]))
            
            # Patrones chartistas
            chart_patterns = self.check_chart_patterns(high, low, close)
            
            # Fuerza de tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Volumen anómalo direccional
            volume_anomaly_data = self.check_volume_anomaly_directional(volume, close)
            
            # Verificar condiciones multi-timeframe obligatorias
            multi_timeframe_long = self.check_multi_timeframe_obligatory(symbol, interval, 'LONG')
            multi_timeframe_short = self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT')
            
            # Preparar datos para análisis
            analysis_data = {
                'symbol': symbol,
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
                'adx_above_level': adx_above_level,
                'di_cross_bullish': di_cross_bullish,
                'di_cross_bearish': di_cross_bearish,
                'rsi_maverick': rsi_maverick,
                'rsi_traditional': rsi_traditional,
                'rsi_maverick_bullish_divergence': rsi_maverick_bullish,
                'rsi_maverick_bearish_divergence': rsi_maverick_bearish,
                'rsi_bullish_divergence': rsi_bullish,
                'rsi_bearish_divergence': rsi_bearish,
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
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'chart_patterns': chart_patterns,
                'trend_strength': trend_strength_data['trend_strength'],
                'no_trade_zones': trend_strength_data['no_trade_zones'],
                'trend_strength_signals': trend_strength_data['strength_signals'],
                'volume_anomaly_buy': volume_anomaly_data['volume_anomaly_buy'],
                'volume_anomaly_sell': volume_anomaly_data['volume_anomaly_sell'],
                'volume_clusters': volume_anomaly_data['volume_clusters'],
                'volume_ratio': volume_anomaly_data['volume_ratio'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short
            }
            
            conditions, ma200_condition = self.evaluate_signal_conditions_corrected(analysis_data, current_idx, interval)
            
            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', ma200_condition)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', ma200_condition)
            
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            chart_pattern_name = ''
            
            if long_score >= 65:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
                chart_pattern_name = self.get_chart_pattern_name(chart_patterns, current_idx)
            elif short_score >= 65:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
                chart_pattern_name = self.get_chart_pattern_name(chart_patterns, current_idx)
            
            current_price = float(close[current_idx])
            levels_data = self.calculate_optimal_entry_exit_with_sr(df, signal_type, supports, resistances, leverage)
            
            # Convertir condiciones a descripciones para Telegram
            condition_descriptions = []
            for cond in fulfilled_conditions:
                if cond == 'chart_pattern' and chart_pattern_name:
                    condition_descriptions.append(f'Patrón Chartista: {chart_pattern_name}')
                else:
                    condition_descriptions.append(self.get_condition_description(cond))
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'supports': levels_data['supports'],
                'resistances': levels_data['resistances'],
                'atr': levels_data['atr'],
                'atr_percentage': levels_data['atr_percentage'],
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
                'adx': float(adx[current_idx]),
                'plus_di': float(plus_di[current_idx]),
                'minus_di': float(minus_di[current_idx]),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_maverick': float(rsi_maverick[current_idx]),
                'rsi_traditional': float(rsi_traditional[current_idx]),
                'fulfilled_conditions': condition_descriptions,
                'multi_timeframe_ok': multi_timeframe_long if signal_type == 'LONG' else multi_timeframe_short,
                'ma200_condition': ma200_condition,
                'is_breakout': levels_data['is_breakout'],
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
                    'adx_above_level': adx_above_level[-50:],
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:],
                    'rsi_maverick_bullish_divergence': rsi_maverick_bullish[-50:],
                    'rsi_maverick_bearish_divergence': rsi_maverick_bearish[-50:],
                    'rsi_bullish_divergence': rsi_bullish[-50:],
                    'rsi_bearish_divergence': rsi_bearish[-50:],
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
                    'breakout_up': breakout_up[-50:],
                    'breakout_down': breakout_down[-50:],
                    'volume_anomaly_buy': volume_anomaly_data['volume_anomaly_buy'][-50:],
                    'volume_anomaly_sell': volume_anomaly_data['volume_anomaly_sell'][-50:],
                    'volume_clusters': volume_anomaly_data['volume_clusters'][-50:],
                    'volume_ratio': volume_anomaly_data['volume_ratio'][-50:],
                    'volume_ma': volume_anomaly_data['volume_ma'][-50:],
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
            'entry': 0,
            'stop_loss': 0,
            'take_profit': [0],
            'supports': [],
            'resistances': [],
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
            'is_breakout': False,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de trading para la estrategia principal"""
        alerts = []
        
        for symbol in CRYPTO_SYMBOLS:
            for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        alert_key = f"{symbol}_{interval}_{signal_data['signal']}"
                        if (alert_key not in self.alert_cache or 
                            (datetime.now() - self.alert_cache[alert_key]).seconds > 300):
                            
                            alerts.append(signal_data)
                            self.alert_cache[alert_key] = datetime.now()
                    
                except Exception as e:
                    print(f"Error generando alerta para {symbol} {interval}: {e}")
                    continue
        
        return alerts

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert(alert_data, strategy_type='main'):
    """Enviar alerta por Telegram con imagen"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if strategy_type == 'main':
            # Mensaje para estrategia principal
            message = f"""
🚨 {alert_data['signal']} | {alert_data['symbol']} | {alert_data.get('interval', '')}
Score: {alert_data['signal_score']:.1f}%

Precio: ${alert_data['current_price']:.6f}
Entrada: ${alert_data['entry']:.6f}
MA200: {'ABOVE' if alert_data.get('ma200_condition') == 'above' else 'BELOW'}

Condiciones cumplidas:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])])}
"""
            
            # Generar imagen
            img_buffer = indicator.generate_telegram_chart_image_main_strategy(
                alert_data['symbol'], 
                alert_data.get('interval', '4h'),
                alert_data
            )
            
        else:  # volume strategy
            # Mensaje para estrategia de volumen
            message = f"""
🚨 VOL+EMA21 | {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Entrada: ${alert_data['close_price']:.6f} | Vol: {alert_data['volume_ratio']:.1f}x

Filtros: FTMaverick OK | MF: {alert_data['mayor_trend']}/{alert_data['menor_trend']}
"""
            
            # Generar imagen
            img_buffer = indicator.generate_telegram_chart_image_volume_strategy(
                alert_data['symbol'],
                alert_data['interval'],
                alert_data
            )
        
        if img_buffer:
            # Enviar imagen
            asyncio.run(bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=img_buffer,
                caption=message
            ))
        else:
            # Enviar solo mensaje si no hay imagen
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            ))
        
        print(f"Alerta enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            current_time = datetime.now()
            
            # Estrategia principal cada 60 segundos
            if current_time.second % 60 == 0:
                print("Verificando alertas estrategia principal...")
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    send_telegram_alert(alert, 'main')
            
            # Estrategia de volumen cada 30 segundos
            if current_time.second % 30 == 0:
                print("Verificando alertas estrategia volumen...")
                volume_alerts = indicator.generate_volume_ema_ftm_alerts()
                for alert in volume_alerts:
                    send_telegram_alert(alert, 'volume')
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error en background_alert_checker: {e}")
            time.sleep(10)

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

@app.route('/api/volume_ema_ftm_signals')
def get_volume_ema_ftm_signals():
    """Endpoint para obtener señales de la nueva estrategia de volumen"""
    try:
        alerts = indicator.generate_volume_ema_ftm_alerts()
        return jsonify({'signals': alerts})
    except Exception as e:
        print(f"Error en /api/volume_ema_ftm_signals: {e}")
        return jsonify({'signals': []})

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
        
        # Obtener datos frescos
        df = indicator.get_kucoin_data(symbol, interval, 100)
        if df is None or len(df) < 50:
            return jsonify({'error': 'Datos insuficientes'}), 400
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calcular indicadores para el reporte
        whale_data = indicator.calculate_whale_signals_improved(df)
        adx, plus_di, minus_di = indicator.calculate_adx(high, low, close, 14)
        rsi_maverick = indicator.calculate_rsi_maverick(close)
        rsi_traditional = indicator.calculate_rsi(close, 14)
        ma_9 = indicator.calculate_sma(close, 9)
        ma_21 = indicator.calculate_sma(close, 21)
        ma_50 = indicator.calculate_sma(close, 50)
        ma_200 = indicator.calculate_sma(close, 200)
        macd, macd_signal, macd_histogram = indicator.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = indicator.calculate_bollinger_bands(close)
        trend_strength_data = indicator.calculate_trend_strength_maverick(close)
        volume_anomaly_data = indicator.check_volume_anomaly_directional(volume, close)
        
        # Crear figura
        fig = plt.figure(figsize=(14, 18))
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(9, 1, 1)
        dates = pd.date_range(end=datetime.now(), periods=len(close), freq=interval)
        
        # Plot velas
        for i in range(len(close)):
            color = 'green' if close[i] >= (close[i-1] if i > 0 else close[i]) else 'red'
            ax1.plot([dates[i], dates[i]], [low[i], high[i]], color='black', linewidth=1)
            ax1.plot([dates[i], dates[i]], [close[i] if close[i] < (close[i-1] if i > 0 else close[i]) else (close[i-1] if i > 0 else close[i]), 
                     close[i] if close[i] >= (close[i-1] if i > 0 else close[i]) else (close[i-1] if i > 0 else close[i])], 
                    color=color, linewidth=3)
        
        # Plot niveles
        ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
        ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
        for i, tp in enumerate(signal_data['take_profit']):
            ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
        
        # Plot soportes y resistencias
        for level in signal_data['supports'][-4:]:
            ax1.axhline(y=level, color='orange', linestyle=':', alpha=0.5, label='Soporte')
        for level in signal_data['resistances'][:4]:
            ax1.axhline(y=level, color='purple', linestyle=':', alpha=0.5, label='Resistencia')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Ballenas
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        whale_dates = dates[-len(whale_data['whale_pump']):]
        ax2.bar(whale_dates, whale_data['whale_pump'], color='green', alpha=0.7, label='Compradoras')
        ax2.bar(whale_dates, whale_data['whale_dump'], color='red', alpha=0.7, label='Vendedoras')
        ax2.set_ylabel('Fuerza Ballenas')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: ADX/DMI
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        adx_dates = dates[-len(adx):]
        ax3.plot(adx_dates, adx, 'black', linewidth=2, label='ADX')
        ax3.plot(adx_dates, plus_di, 'green', linewidth=1, label='+DI')
        ax3.plot(adx_dates, minus_di, 'red', linewidth=1, label='-DI')
        ax3.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Umbral 25')
        ax3.set_ylabel('ADX/DMI')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: RSI Tradicional
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        rsi_dates = dates[-len(rsi_traditional):]
        ax4.plot(rsi_dates, rsi_traditional, 'cyan', linewidth=2, label='RSI Tradicional')
        ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
        ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax4.set_ylabel('RSI Tradicional')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: RSI Maverick
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        rsi_maverick_dates = dates[-len(rsi_maverick):]
        ax5.plot(rsi_maverick_dates, rsi_maverick, 'blue', linewidth=2, label='RSI Maverick')
        ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
        ax5.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax5.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        ax5.set_ylabel('RSI Maverick')
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: MACD
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        macd_dates = dates[-len(macd):]
        ax6.plot(macd_dates, macd, 'blue', linewidth=1, label='MACD')
        ax6.plot(macd_dates, macd_signal, 'red', linewidth=1, label='Señal')
        
        colors = ['green' if x > 0 else 'red' for x in macd_histogram]
        ax6.bar(macd_dates, macd_histogram, color=colors, alpha=0.6, label='Histograma')
        
        ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax6.set_ylabel('MACD')
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: Volumen y Anomalías
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        volume_dates = dates[-len(volume_anomaly_data['volume_ratio']):]
        
        # Volumen normal
        ax7.bar(volume_dates, volume[-len(volume_anomaly_data['volume_ratio']):], color='gray', alpha=0.6, label='Volumen')
        
        # EMA de volumen
        ax7.plot(volume_dates, volume_anomaly_data['volume_ma'][-len(volume_anomaly_data['volume_ratio']):], 
                'yellow', linewidth=1, label='EMA Volumen')
        
        # Anomalías de volumen
        anomaly_dates = []
        anomaly_values = []
        for i, date in enumerate(volume_dates):
            if volume_anomaly_data['volume_anomaly_buy'][i] or volume_anomaly_data['volume_anomaly_sell'][i]:
                anomaly_dates.append(date)
                anomaly_values.append(volume[-len(volume_anomaly_data['volume_ratio'])+i])
        
        ax7.scatter(anomaly_dates, anomaly_values, color='red', s=50, label='Anomalías Volumen', zorder=5)
        
        ax7.set_ylabel('Volumen')
        ax7.legend(loc='upper left', fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: Fuerza de Tendencia Maverick
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        trend_dates = dates[-len(trend_strength_data['trend_strength']):]
        trend_strength = trend_strength_data['trend_strength']
        colors = trend_strength_data['colors']
        
        for i in range(len(trend_dates)):
            color = colors[i] if i < len(colors) else 'gray'
            ax8.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
        
        if 'high_zone_threshold' in trend_strength_data:
            threshold = trend_strength_data['high_zone_threshold']
            ax8.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                       label=f'Umbral Alto ({threshold:.1f}%)')
            ax8.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        
        no_trade_zones = trend_strength_data['no_trade_zones']
        for i, date in enumerate(trend_dates):
            if i < len(no_trade_zones) and no_trade_zones[i]:
                ax8.axvline(x=date, color='red', alpha=0.3, linewidth=2)
        
        ax8.set_ylabel('Fuerza Tendencia %')
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
        import traceback
        traceback.print_exc()
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
