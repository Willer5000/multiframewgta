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

# Top 5 criptomonedas para las estrategias
TOP5_CRYPTOS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT"]

# Configuración optimizada - 40 criptomonedas top
CRYPTO_SYMBOLS = [
    # Bajo Riesgo (20) - Top market cap
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
    "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT", "ETC-USDT",
    "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "EOS-USDT",
    
    # Medio Riesgo (10)
    "NEAR-USDT", "AXS-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ONE-USDT", "WAVES-USDT",
    
    # Alto Riesgo (7)
    "APE-USDT", "GMT-USDT", "SAND-USDT", "OP-USDT", "ARB-USDT",
    "MAGIC-USDT", "RNDR-USDT",
    
    # Memecoins (3)
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

# Top 10 para estrategia de volumen (excluyendo Doge)
TOP10_LOW_RISK = [s for s in CRYPTO_RISK_CLASSIFICATION['bajo'] if 'DOGE' not in s][:10]

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

# Configuración de estrategias y temporalidades permitidas
STRATEGY_CONFIG = {
    'TREND_RIDER': {
        'timeframes': ['4h', '8h', '12h', '1D'],
        'description': 'Swing Trading con filtro MA200'
    },
    'MOMENTUM_DIVERGENCE': {
        'timeframes': ['1h', '2h', '4h'],
        'description': 'Divergencias de momentum intraday'
    },
    'BOLLINGER_SQUEEZE': {
        'timeframes': ['15m', '30m', '1h'],
        'description': 'Breakout tras compresión de volatilidad'
    },
    'ADX_POWER_TREND': {
        'timeframes': ['2h', '4h', '8h'],
        'description': 'Tendencias fuertes con ADX >25'
    },
    'MACD_HISTOGRAM_REVERSAL': {
        'timeframes': ['30m', '1h', '2h'],
        'description': 'Reversión con cambio de histograma MACD'
    },
    'VOLUME_SPIKE_MOMENTUM': {
        'timeframes': ['15m', '30m', '1h'],
        'description': 'Momentum confirmado por volumen'
    },
    'DOUBLE_CONFIRMATION_RSI': {
        'timeframes': ['1h', '2h', '4h'],
        'description': 'Confirmación dual RSI clásico y Maverick'
    },
    'TREND_STRENGTH_MAVERICK': {
        'timeframes': ['4h', '8h', '12h'],
        'description': 'Señales fuertes del FTMaverick'
    },
    'WHALE_FOLLOWING': {
        'timeframes': ['12h', '1D'],
        'description': 'Seguimiento de ballenas institucionales'
    },
    'MA_CONVERGENCE_DIVERGENCE': {
        'timeframes': ['2h', '4h', '8h'],
        'description': 'Alineación perfecta de medias móviles'
    },
    'RSI_MAVERICK_EXTREME': {
        'timeframes': ['30m', '1h', '2h'],
        'description': 'Operaciones en extremos del RSI Maverick'
    },
    'VOLUME_PRICE_DIVERGENCE': {
        'timeframes': ['1h', '2h', '4h'],
        'description': 'Divergencia precio-volumen'
    },
    'VOLUME_EMA_FTM': {
        'timeframes': ['15m', '30m', '1h', '4h', '12h', '1D'],
        'description': 'Desplome de volumen con filtros FTMaverick'
    }
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_operations = {}
        self.winrate_data = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.sent_exit_signals = set()
        self.volume_ema_signals = {}
        self.strategy_signals = {}  # Cache para señales de estrategias
        
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
        """Calcular tiempo restante para el cierre de la vela - MEJORADO"""
        interval_minutes = {
            '15m': 15, '30m': 30, '5m': 5, '1h': 60,
            '2h': 120, '4h': 240, '8h': 480, '12h': 720,
            '1D': 1440, '1W': 10080, '1M': 43200
        }
        
        if interval not in interval_minutes:
            return False
            
        total_minutes = interval_minutes[interval]
        
        if interval == '15m':
            check_point = total_minutes * 0.5  # 50%
            check_frequency = 60  # 60 segundos
        elif interval == '30m':
            check_point = total_minutes * 0.5  # 50%
            check_frequency = 120  # 120 segundos
        elif interval == '1h':
            check_point = total_minutes * 0.5  # 50%
            check_frequency = 300  # 300 segundos
        elif interval == '2h':
            check_point = total_minutes * 0.5  # 50%
            check_frequency = 420  # 420 segundos
        elif interval == '4h':
            check_point = total_minutes * 0.25  # 25%
            check_frequency = 420  # 420 segundos
        elif interval == '8h':
            check_point = total_minutes * 0.25  # 25%
            check_frequency = 600  # 600 segundos
        elif interval == '12h':
            check_point = total_minutes * 0.25  # 25%
            check_frequency = 900  # 900 segundos
        elif interval == '1D':
            check_point = total_minutes * 0.25  # 25%
            check_frequency = 3600  # 3600 segundos
        elif interval == '1W':
            check_point = total_minutes * 0.10  # 10%
            check_frequency = 10000  # 10000 segundos
        else:
            return False
            
        current_minute = current_time.minute
        current_hour = current_time.hour
        
        # Calcular minutos transcurridos en la vela actual
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h']:
            elapsed_minutes = current_minute % total_minutes
        elif interval == '1D':
            elapsed_minutes = current_hour * 60 + current_minute
        else:
            elapsed_minutes = 0
            
        return elapsed_minutes >= check_point

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

    def calculate_dynamic_support_resistance(self, high, low, close, num_levels=6):
        """Calcular soportes y resistencias dinámicos"""
        try:
            # Usar pivot points y niveles de Fibonacci
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            r1 = 2 * pivot - low[-1]
            s1 = 2 * pivot - high[-1]
            r2 = pivot + (high[-1] - low[-1])
            s2 = pivot - (high[-1] - low[-1])
            
            # Niveles adicionales basados en máximos/mínimos recientes
            recent_highs = sorted(high[-50:])[-3:]
            recent_lows = sorted(low[-50:])[:3]
            
            support_levels = list(recent_lows) + [s1, s2]
            resistance_levels = list(recent_highs) + [r1, r2]
            
            # Eliminar duplicados y ordenar
            support_levels = sorted(list(set([round(s, 6) for s in support_levels])))[:num_levels]
            resistance_levels = sorted(list(set([round(r, 6) for r in resistance_levels])), reverse=True)[:num_levels]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            print(f"Error calculando soportes/resistencias: {e}")
            # Valores por defecto
            current_price = close[-1]
            support_levels = [current_price * 0.95, current_price * 0.90]
            resistance_levels = [current_price * 1.05, current_price * 1.10]
            return support_levels, resistance_levels

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15, support_levels=None, resistance_levels=None):
        """Calcular entradas y salidas óptimas mejoradas con soportes/resistencias"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular soportes y resistencias si no se proporcionan
            if support_levels is None or resistance_levels is None:
                support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            # Para LONG: entrada en soporte más cercano, stop loss debajo del soporte
            if signal_type == 'LONG':
                # Encontrar soporte más cercano por debajo del precio actual
                valid_supports = [s for s in support_levels if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)  # Soporte más fuerte (más alto)
                else:
                    entry = current_price * 0.995  # Ligera corrección si no hay soportes
                
                # Stop loss debajo del siguiente soporte o usando ATR
                if len(support_levels) > 1:
                    stop_loss = support_levels[1] if len(support_levels) > 1 else entry - (current_atr * 2)
                else:
                    stop_loss = entry - (current_atr * 2)
                
                # Take profits en resistencias
                take_profits = []
                for resistance in resistance_levels[:3]:  # Primeras 3 resistencias
                    if resistance > entry:
                        take_profits.append(resistance)
                
                if not take_profits:
                    take_profits = [entry + (2 * (entry - stop_loss))]
            
            else:  # SHORT
                # Encontrar resistencia más cercana por encima del precio actual
                valid_resistances = [r for r in resistance_levels if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)  # Resistencia más fuerte (más baja)
                else:
                    entry = current_price * 1.005  # Ligera corrección si no hay resistencias
                
                # Stop loss encima de la siguiente resistencia o usando ATR
                if len(resistance_levels) > 1:
                    stop_loss = resistance_levels[1] if len(resistance_levels) > 1 else entry + (current_atr * 2)
                else:
                    stop_loss = entry + (current_atr * 2)
                
                # Take profits en soportes
                take_profits = []
                for support in support_levels[:3]:  # Primeros 3 soportes
                    if support < entry:
                        take_profits.append(support)
                
                if not take_profits:
                    take_profits = [entry - (2 * (stop_loss - entry))]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits[:3]],  # Máximo 3 take profits
                'support_levels': [float(s) for s in support_levels],
                'resistance_levels': [float(r) for r in resistance_levels],
                'atr': float(current_atr),
                'atr_percentage': float(current_atr / current_price)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
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
        """Verificar tendencia en múltiples temporalidades"""
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
        """Verificar condiciones multi-timeframe obligatorias - CORREGIDO"""
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
        
        for i in range(lookback, n-1):
            price_window = price[i-lookback:i+1]
            indicator_window = indicator[i-lookback:i+1]
            
            if (price[i] < np.min(price_window[:-1]) and 
                indicator[i] > np.min(indicator_window[:-1])):
                bullish_div[i] = True
            
            if (price[i] > np.max(price_window[:-1]) and 
                indicator[i] < np.max(indicator_window[:-1])):
                bearish_div[i] = True
        
        # Extender señal por 4 velas para RSI Maverick y 7 para RSI Tradicional
        extended_bullish = bullish_div.copy()
        extended_bearish = bearish_div.copy()
        
        for i in range(n):
            if bullish_div[i]:
                for j in range(1, min(5, n-i)):
                    extended_bullish[i+j] = True
            if bearish_div[i]:
                for j in range(1, min(5, n-i)):
                    extended_bearish[i+j] = True
        
        return extended_bullish.tolist(), extended_bearish.tolist()

    def detect_divergence_traditional(self, price, indicator, lookback=14):
        """Detectar divergencias para RSI Tradicional (7 velas)"""
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
        
        # Extender señal por 7 velas para RSI Tradicional
        extended_bullish = bullish_div.copy()
        extended_bearish = bearish_div.copy()
        
        for i in range(n):
            if bullish_div[i]:
                for j in range(1, min(8, n-i)):
                    extended_bullish[i+j] = True
            if bearish_div[i]:
                for j in range(1, min(8, n-i)):
                    extended_bearish[i+j] = True
        
        return extended_bullish.tolist(), extended_bearish.tolist()

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

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI con confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                di_cross_bullish[i] = True
                # Señal dura 1 vela más
                if i + 1 < n:
                    di_cross_bullish[i+1] = True
            
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
                # Señal dura 1 vela más
                if i + 1 < n:
                    di_cross_bearish[i+1] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

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
            'bearish_flag': np.zeros(n, dtype=bool),
            'pattern_name': [''] * n
        }
        
        for i in range(lookback, n-7):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            window_close = close[i-lookback:i+1]
            
            # Doble Techo
            if len(window_high) >= 15:
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 2:
                    last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                    if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                        patterns['double_top'][i] = True
                        patterns['pattern_name'][i] = 'Doble techo'
                        # Señal dura 7 velas
                        for j in range(1, min(8, n-i)):
                            patterns['double_top'][i+j] = True
                            patterns['pattern_name'][i+j] = 'Doble techo'
            
            # Doble Fondo
            if len(window_low) >= 15:
                troughs = []
                for j in range(1, len(window_low)-1):
                    if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                        troughs.append((j, window_low[j]))
                
                if len(troughs) >= 2:
                    last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                    if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                        patterns['double_bottom'][i] = True
                        patterns['pattern_name'][i] = 'Doble piso'
                        # Señal dura 7 velas
                        for j in range(1, min(8, n-i)):
                            patterns['double_bottom'][i+j] = True
                            patterns['pattern_name'][i+j] = 'Doble piso'
            
            # Hombro Cabeza Hombro (simplificado)
            if len(window_high) >= 20:
                max_idx = np.argmax(window_high)
                if (max_idx > 5 and max_idx < len(window_high)-5 and
                    window_high[max_idx-3] < window_high[max_idx] and
                    window_high[max_idx+3] < window_high[max_idx]):
                    patterns['head_shoulders'][i] = True
                    patterns['pattern_name'][i] = 'Hombro cabeza hombro'
                    # Señal dura 7 velas
                    for j in range(1, min(8, n-i)):
                        patterns['head_shoulders'][i+j] = True
                        patterns['pattern_name'][i+j] = 'Hombro cabeza hombro'
        
        return patterns

    def calculate_volume_anomaly(self, volume, close, period=20, std_multiplier=2):
        """Calcular anomalías de volumen - NUEVA ESTRATEGIA"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_signal = ['NEUTRAL'] * n  # COMPRA, VENTA, NEUTRAL
            
            for i in range(period, n):
                # Media móvil de volumen
                volume_ma = self.calculate_sma(volume[:i+1], period)
                current_volume_ma = volume_ma[i] if i < len(volume_ma) else volume[i]
                
                # Ratio volumen actual vs MA
                if current_volume_ma > 0:
                    volume_ratio[i] = volume[i] / current_volume_ma
                else:
                    volume_ratio[i] = 1
                
                # Detectar anomalía (> 2σ)
                if i >= period * 2:
                    window = volume[max(0, i-period*2):i+1]
                    std_volume = np.std(window) if len(window) > 1 else 0
                    
                    if volume_ratio[i] > 1 + (std_multiplier * (std_volume / current_volume_ma if current_volume_ma > 0 else 0)):
                        volume_anomaly[i] = True
                        
                        # Determinar si es compra o venta basado en dirección del precio
                        if i > 0:
                            price_change = (close[i] - close[i-1]) / close[i-1] * 100
                            if price_change > 0:
                                volume_signal[i] = 'COMPRA'
                            else:
                                volume_signal[i] = 'VENTA'
                
                # Detectar clusters (múltiples anomalías en 5 periodos)
                if i >= 5:
                    recent_anomalies = volume_anomaly[max(0, i-4):i+1]
                    if np.sum(recent_anomalies) >= 2:  # Al menos 2 anomalías en 5 periodos
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ma': volume_ma.tolist() if 'volume_ma' in locals() else [0] * n,
                'volume_signal': volume_signal
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly: {e}")
            n = len(volume)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ma': [0] * n,
                'volume_signal': ['NEUTRAL'] * n
            }

    def check_ma_crossover(self, ma_short, ma_long, lookback=3):
        """Detectar cruce de medias móviles"""
        n = len(ma_short)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MA corta cruza por encima de MA larga
            if (ma_short[i] > ma_long[i] and 
                ma_short[i-1] <= ma_long[i-1]):
                ma_cross_bullish[i] = True
                # Señal dura 1 vela más
                if i + 1 < n:
                    ma_cross_bullish[i+1] = True
            
            # Cruce bajista: MA corta cruza por debajo de MA larga
            if (ma_short[i] < ma_long[i] and 
                ma_short[i-1] >= ma_long[i-1]):
                ma_cross_bearish[i] = True
                # Señal dura 1 vela más
                if i + 1 < n:
                    ma_cross_bearish[i+1] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()

    def check_macd_crossover(self, macd, signal):
        """Detectar cruce de MACD"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MACD cruza por encima de señal
            if (macd[i] > signal[i] and 
                macd[i-1] <= signal[i-1]):
                macd_cross_bullish[i] = True
                # Señal dura 1 vela más
                if i + 1 < n:
                    macd_cross_bullish[i+1] = True
            
            # Cruce bajista: MACD cruza por debajo de señal
            if (macd[i] < signal[i] and 
                macd[i-1] >= signal[i-1]):
                macd_cross_bearish[i] = True
                # Señal dura 1 vela más
                if i + 1 < n:
                    macd_cross_bearish[i+1] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()

    def check_adx_slope(self, adx, period=3):
        """Verificar pendiente positiva del ADX"""
        n = len(adx)
        adx_slope_positive = np.zeros(n, dtype=bool)
        
        for i in range(period, n):
            if adx[i] > 25:  # ADX por encima del nivel
                slope = (adx[i] - adx[i-period]) / period
                if slope > 0:  # Pendiente positiva
                    adx_slope_positive[i] = True
        
        return adx_slope_positive.tolist()

    # =====================================================================
    # NUEVAS ESTRATEGIAS PARA TELEGRAM
    # =====================================================================

    def check_trend_rider_strategy(self, symbol, interval):
        """Estrategia Trend Rider para Telegram"""
        try:
            # Verificar temporalidad permitida
            if interval not in STRATEGY_CONFIG['TREND_RIDER']['timeframes']:
                return None
            
            # Verificar horario para scalping
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 100:
                return None
            
            close = df['close'].values
            
            # Calcular indicadores
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # Obtener datos de temporalidad menor para MACD
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            menor_interval = hierarchy.get('menor')
            
            if menor_interval and menor_interval in ['15m', '30m', '1h', '2h', '4h']:
                menor_df = self.get_kucoin_data(symbol, menor_interval, 50)
                menor_close = menor_df['close'].values if menor_df is not None else close[-50:]
                macd, signal, histogram = self.calculate_macd(menor_close)
            else:
                macd, signal, histogram = self.calculate_macd(close[-50:])
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            # Condiciones para LONG
            current_price = close[-1]
            current_ma50 = ma_50[-1]
            current_ma200 = ma_200[-1]
            
            # LONG: Precio > MA50 > MA200, MACD positivo, FTMaverick no en zona prohibida
            long_conditions = (
                current_price > current_ma50 and
                current_ma50 > current_ma200 and
                macd[-1] > signal[-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][-1]
            )
            
            # SHORT: Precio < MA50 < MA200, MACD negativo, FTMaverick no en zona prohibida
            short_conditions = (
                current_price < current_ma50 and
                current_ma50 < current_ma200 and
                macd[-1] < signal[-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not ftm_data['no_trade_zones'][-1]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles de entrada
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico específico
            chart_buffer = self.generate_trend_rider_chart(symbol, interval, df, ma_50, ma_200, macd, signal, ftm_data, signal_type)
            
            # Preparar datos para Telegram
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'TREND_RIDER',
                'ma50_price': current_ma50,
                'ma200_price': current_ma200,
                'ftm_signal': ftm_data['strength_signals'][-1],
                'no_trade_zone': ftm_data['no_trade_zones'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_trend_rider_strategy para {symbol} {interval}: {e}")
            return None

    def check_momentum_divergence_strategy(self, symbol, interval):
        """Estrategia Momentum Divergence para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['MOMENTUM_DIVERGENCE']['timeframes']:
                return None
            
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular indicadores
            rsi_traditional = self.calculate_rsi(close, 14)
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Detectar divergencias
            rsi_bullish_div, rsi_bearish_div = self.detect_divergence_traditional(close, rsi_traditional)
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            
            # Volume clusters
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Condiciones para LONG: Divergencia alcista en ambos RSI con volumen
            long_conditions = (
                rsi_bullish_div[current_idx] and
                rsi_maverick_bullish[current_idx] and
                volume_data['volume_clusters'][current_idx] and
                volume_data['volume_signal'][current_idx] == 'COMPRA' and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: Divergencia bajista en ambos RSI con volumen
            short_conditions = (
                rsi_bearish_div[current_idx] and
                rsi_maverick_bearish[current_idx] and
                volume_data['volume_clusters'][current_idx] and
                volume_data['volume_signal'][current_idx] == 'VENTA' and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_momentum_divergence_chart(
                symbol, interval, df, rsi_traditional, rsi_maverick, volume_data, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'MOMENTUM_DIVERGENCE',
                'rsi_traditional': rsi_traditional[-1],
                'rsi_maverick': rsi_maverick[-1],
                'volume_ratio': volume_data['volume_ratio'][-1],
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_momentum_divergence_strategy para {symbol} {interval}: {e}")
            return None

    def check_bollinger_squeeze_strategy(self, symbol, interval):
        """Estrategia Bollinger Squeeze Breakout para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['BOLLINGER_SQUEEZE']['timeframes']:
                return None
            
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calcular Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Calcular ADX y DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            # Calcular anomalías de volumen
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            current_idx = -1
            bb_width = (bb_upper[current_idx] - bb_lower[current_idx]) / bb_middle[current_idx]
            
            # Detectar squeeze (bandas estrechas)
            bb_squeeze = bb_width < 0.1  # 10% de ancho relativo
            
            # Condiciones para LONG: Squeeze + breakout alcista con volumen
            long_conditions = (
                bb_squeeze and
                close[current_idx] > bb_upper[current_idx] and
                volume_data['volume_anomaly'][current_idx] and
                volume_data['volume_signal'][current_idx] == 'COMPRA' and
                adx[current_idx] > 25 and
                plus_di[current_idx] > minus_di[current_idx] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: Squeeze + breakout bajista con volumen
            short_conditions = (
                bb_squeeze and
                close[current_idx] < bb_lower[current_idx] and
                volume_data['volume_anomaly'][current_idx] and
                volume_data['volume_signal'][current_idx] == 'VENTA' and
                adx[current_idx] > 25 and
                minus_di[current_idx] > plus_di[current_idx] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_bollinger_squeeze_chart(
                symbol, interval, df, bb_upper, bb_middle, bb_lower, adx, plus_di, minus_di, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'BOLLINGER_SQUEEZE',
                'bb_width': bb_width * 100,  # Porcentaje
                'adx_value': adx[-1],
                'plus_di': plus_di[-1],
                'minus_di': minus_di[-1],
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_bollinger_squeeze_strategy para {symbol} {interval}: {e}")
            return None

    def check_adx_power_trend_strategy(self, symbol, interval):
        """Estrategia ADX Power Trend para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['ADX_POWER_TREND']['timeframes']:
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calcular ADX y DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            # Calcular MA21
            ma_21 = self.calculate_sma(close, 21)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Condiciones para LONG: ADX fuerte, +DI > -DI, precio > MA21
            long_conditions = (
                adx[current_idx] > 25 and
                plus_di[current_idx] > minus_di[current_idx] and
                (plus_di[current_idx] - minus_di[current_idx]) > 5 and
                close[current_idx] > ma_21[current_idx] and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: ADX fuerte, -DI > +DI, precio < MA21
            short_conditions = (
                adx[current_idx] > 25 and
                minus_di[current_idx] > plus_di[current_idx] and
                (minus_di[current_idx] - plus_di[current_idx]) > 5 and
                close[current_idx] < ma_21[current_idx] and
                ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_adx_power_trend_chart(
                symbol, interval, df, ma_21, adx, plus_di, minus_di, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'ADX_POWER_TREND',
                'adx_value': adx[-1],
                'plus_di': plus_di[-1],
                'minus_di': minus_di[-1],
                'di_difference': abs(plus_di[-1] - minus_di[-1]),
                'ma21_price': ma_21[-1],
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_adx_power_trend_strategy para {symbol} {interval}: {e}")
            return None

    def check_macd_histogram_reversal_strategy(self, symbol, interval):
        """Estrategia MACD Histogram Reversal para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['MACD_HISTOGRAM_REVERSAL']['timeframes']:
                return None
            
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # Calcular MACD
            macd, signal, histogram = self.calculate_macd(close)
            
            # Calcular RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Calcular cruce de MA9/MA21
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover(ma_9, ma_21)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            prev_idx = -2
            
            # Condiciones para LONG: Cambio histograma negativo a positivo, cruce MA, RSI no extremo
            long_conditions = (
                histogram[prev_idx] < 0 and histogram[current_idx] > 0 and
                ma_cross_bullish[current_idx] and
                0.3 < rsi_maverick[current_idx] < 0.7 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: Cambio histograma positivo a negativo, cruce MA, RSI no extremo
            short_conditions = (
                histogram[prev_idx] > 0 and histogram[current_idx] < 0 and
                ma_cross_bearish[current_idx] and
                0.3 < rsi_maverick[current_idx] < 0.7 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_macd_histogram_chart(
                symbol, interval, df, ma_9, ma_21, macd, signal, histogram, rsi_maverick, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'MACD_HISTOGRAM_REVERSAL',
                'macd_value': macd[-1],
                'macd_signal': signal[-1],
                'histogram': histogram[-1],
                'rsi_maverick': rsi_maverick[-1],
                'ma_cross': 'BULLISH' if ma_cross_bullish[-1] else 'BEARISH' if ma_cross_bearish[-1] else 'NONE',
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_macd_histogram_reversal_strategy para {symbol} {interval}: {e}")
            return None

    def check_volume_spike_momentum_strategy(self, symbol, interval):
        """Estrategia Volume Spike Momentum para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['VOLUME_SPIKE_MOMENTUM']['timeframes']:
                return None
            
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular MA21
            ma_21 = self.calculate_sma(close, 21)
            
            # Calcular anomalías de volumen
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            # Calcular RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Verificar cluster de volumen (mínimo 2 anomalías en 5 velas)
            volume_cluster = volume_data['volume_clusters'][current_idx]
            volume_signal = volume_data['volume_signal'][current_idx]
            
            # Condiciones para LONG: Cluster de volumen COMPRA, precio > MA21, RSI no extremo
            long_conditions = (
                volume_cluster and
                volume_signal == 'COMPRA' and
                close[current_idx] > ma_21[current_idx] and
                0.3 < rsi_maverick[current_idx] < 0.7 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: Cluster de volumen VENTA, precio < MA21, RSI no extremo
            short_conditions = (
                volume_cluster and
                volume_signal == 'VENTA' and
                close[current_idx] < ma_21[current_idx] and
                0.3 < rsi_maverick[current_idx] < 0.7 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_volume_spike_chart(
                symbol, interval, df, ma_21, volume_data, rsi_maverick, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'VOLUME_SPIKE_MOMENTUM',
                'volume_ratio': volume_data['volume_ratio'][-1],
                'volume_signal': volume_signal,
                'ma21_price': ma_21[-1],
                'rsi_maverick': rsi_maverick[-1],
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_volume_spike_momentum_strategy para {symbol} {interval}: {e}")
            return None

    def check_double_confirmation_rsi_strategy(self, symbol, interval):
        """Estrategia Double Confirmation RSI para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['DOUBLE_CONFIRMATION_RSI']['timeframes']:
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # Calcular ambos RSI
            rsi_traditional = self.calculate_rsi(close, 14)
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Calcular Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            current_price = close[current_idx]
            
            # Condiciones para LONG: Ambos RSI en sobreventa, precio en banda inferior
            long_conditions = (
                rsi_traditional[current_idx] < 30 and
                rsi_maverick[current_idx] < 0.2 and
                current_price <= bb_lower[current_idx] * 1.02 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: Ambos RSI en sobrecompra, precio en banda superior
            short_conditions = (
                rsi_traditional[current_idx] > 70 and
                rsi_maverick[current_idx] > 0.8 and
                current_price >= bb_upper[current_idx] * 0.98 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_double_rsi_chart(
                symbol, interval, df, rsi_traditional, rsi_maverick, bb_upper, bb_middle, bb_lower, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'DOUBLE_CONFIRMATION_RSI',
                'rsi_traditional': rsi_traditional[-1],
                'rsi_maverick': rsi_maverick[-1],
                'bb_position': 'LOWER' if long_conditions else 'UPPER',
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_double_confirmation_rsi_strategy para {symbol} {interval}: {e}")
            return None

    def check_trend_strength_maverick_strategy(self, symbol, interval):
        """Estrategia Trend Strength Maverick para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['TREND_STRENGTH_MAVERICK']['timeframes']:
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            # Calcular MA50
            ma_50 = self.calculate_sma(close, 50)
            
            # Calcular confirmación de volumen
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            current_idx = -1
            current_strength_signal = ftm_data['strength_signals'][current_idx]
            current_no_trade = ftm_data['no_trade_zones'][current_idx]
            
            # Solo operar señales STRONG
            if current_strength_signal == 'STRONG_UP':
                signal_type = 'LONG'
                # Condiciones adicionales para LONG
                conditions_met = (
                    not current_no_trade and
                    close[current_idx] > ma_50[current_idx] and
                    volume_data['volume_ratio'][current_idx] > 1.5
                )
                if not conditions_met:
                    return None
                    
            elif current_strength_signal == 'STRONG_DOWN':
                signal_type = 'SHORT'
                # Condiciones adicionales para SHORT
                conditions_met = (
                    not current_no_trade and
                    close[current_idx] < ma_50[current_idx] and
                    volume_data['volume_ratio'][current_idx] > 1.5
                )
                if not conditions_met:
                    return None
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_trend_strength_chart(
                symbol, interval, df, ftm_data, ma_50, volume_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'TREND_STRENGTH_MAVERICK',
                'ftm_signal': current_strength_signal,
                'trend_strength': ftm_data['trend_strength'][-1],
                'bb_width': ftm_data['bb_width'][-1],
                'ma50_price': ma_50[-1],
                'volume_ratio': volume_data['volume_ratio'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_trend_strength_maverick_strategy para {symbol} {interval}: {e}")
            return None

    def check_whale_following_strategy(self, symbol, interval):
        """Estrategia Whale Following para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['WHALE_FOLLOWING']['timeframes']:
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 100:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calcular señales de ballenas
            whale_data = self.calculate_whale_signals_improved(df)
            
            # Calcular ADX y DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            # Calcular MA200
            ma_200 = self.calculate_sma(close, 200)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Condiciones para LONG: Señal ballena compradora confirmada, ADX >25, +DI > -DI, precio > MA200
            long_conditions = (
                whale_data['confirmed_buy'][current_idx] and
                whale_data['whale_pump'][current_idx] > 20 and
                adx[current_idx] > 25 and
                plus_di[current_idx] > minus_di[current_idx] and  # Condición obligatoria +DI > -DI para LONG
                (plus_di[current_idx] - minus_di[current_idx]) > 5 and
                close[current_idx] > ma_200[current_idx] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: Señal ballena vendedora confirmada, ADX >25, -DI > +DI, precio < MA200
            short_conditions = (
                whale_data['confirmed_sell'][current_idx] and
                whale_data['whale_dump'][current_idx] > 20 and
                adx[current_idx] > 25 and
                minus_di[current_idx] > plus_di[current_idx] and  # Condición obligatoria -DI > +DI para SHORT
                (minus_di[current_idx] - plus_di[current_idx]) > 5 and
                close[current_idx] < ma_200[current_idx] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 10, support_levels, resistance_levels  # Menor apalancamiento para whale trading
            )
            
            # Generar gráfico
            chart_buffer = self.generate_whale_following_chart(
                symbol, interval, df, whale_data, ma_200, adx, plus_di, minus_di, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'WHALE_FOLLOWING',
                'whale_signal': whale_data['whale_pump'][-1] if signal_type == 'LONG' else whale_data['whale_dump'][-1],
                'adx_value': adx[-1],
                'plus_di': plus_di[-1],
                'minus_di': minus_di[-1],
                'di_cross': 'BULLISH' if plus_di[-1] > minus_di[-1] else 'BEARISH',
                'ma200_price': ma_200[-1],
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_whale_following_strategy para {symbol} {interval}: {e}")
            return None

    def check_ma_convergence_divergence_strategy(self, symbol, interval):
        """Estrategia MA Convergence Divergence para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['MA_CONVERGENCE_DIVERGENCE']['timeframes']:
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 100:
                return None
            
            close = df['close'].values
            
            # Calcular todas las medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            
            # Calcular MACD
            macd, signal, histogram = self.calculate_macd(close)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Verificar alineación perfecta
            ma_alignment_bullish = (
                close[current_idx] > ma_9[current_idx] and
                ma_9[current_idx] > ma_21[current_idx] and
                ma_21[current_idx] > ma_50[current_idx]
            )
            
            ma_alignment_bearish = (
                close[current_idx] < ma_9[current_idx] and
                ma_9[current_idx] < ma_21[current_idx] and
                ma_21[current_idx] < ma_50[current_idx]
            )
            
            # Verificar separación mínima (1% del precio)
            price = close[current_idx]
            min_separation = price * 0.01
            
            separation_ok_bullish = (
                (ma_9[current_idx] - ma_21[current_idx]) > min_separation and
                (ma_21[current_idx] - ma_50[current_idx]) > min_separation
            )
            
            separation_ok_bearish = (
                (ma_21[current_idx] - ma_9[current_idx]) > min_separation and
                (ma_50[current_idx] - ma_21[current_idx]) > min_separation
            )
            
            # Condiciones para LONG: Alineación alcista con separación, MACD positivo
            long_conditions = (
                ma_alignment_bullish and
                separation_ok_bullish and
                macd[current_idx] > 0 and
                histogram[current_idx] > 0 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: Alineación bajista con separación, MACD negativo
            short_conditions = (
                ma_alignment_bearish and
                separation_ok_bearish and
                macd[current_idx] < 0 and
                histogram[current_idx] < 0 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_ma_convergence_chart(
                symbol, interval, df, ma_9, ma_21, ma_50, macd, signal, histogram, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': price,
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'MA_CONVERGENCE_DIVERGENCE',
                'ma_alignment': 'BULLISH' if ma_alignment_bullish else 'BEARISH',
                'ma9_price': ma_9[-1],
                'ma21_price': ma_21[-1],
                'ma50_price': ma_50[-1],
                'macd_value': macd[-1],
                'histogram': histogram[-1],
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_ma_convergence_divergence_strategy para {symbol} {interval}: {e}")
            return None

    def check_rsi_maverick_extreme_strategy(self, symbol, interval):
        """Estrategia RSI Maverick Extreme para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['RSI_MAVERICK_EXTREME']['timeframes']:
                return None
            
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # Calcular RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Calcular Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            current_rsi = rsi_maverick[current_idx]
            
            # Verificar extremos sostenidos (mínimo 3 velas consecutivas)
            if len(rsi_maverick) >= 4:
                rsi_extreme_low = all(r < 0.15 for r in rsi_maverick[-3:])
                rsi_extreme_high = all(r > 0.85 for r in rsi_maverick[-3:])
            else:
                rsi_extreme_low = current_rsi < 0.15
                rsi_extreme_high = current_rsi > 0.85
            
            # Condiciones para LONG: RSI extremadamente bajo, precio toca banda inferior
            long_conditions = (
                rsi_extreme_low and
                close[current_idx] <= bb_lower[current_idx] * 1.01 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            # Condiciones para SHORT: RSI extremadamente alto, precio toca banda superior
            short_conditions = (
                rsi_extreme_high and
                close[current_idx] >= bb_upper[current_idx] * 0.99 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 10, support_levels, resistance_levels  # Menor apalancamiento para extremos
            )
            
            # Generar gráfico
            chart_buffer = self.generate_rsi_extreme_chart(
                symbol, interval, df, rsi_maverick, bb_upper, bb_middle, bb_lower, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'RSI_MAVERICK_EXTREME',
                'rsi_maverick': current_rsi,
                'rsi_extreme': 'LOW' if long_conditions else 'HIGH',
                'bb_position': 'LOWER' if long_conditions else 'UPPER',
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_rsi_maverick_extreme_strategy para {symbol} {interval}: {e}")
            return None

    def check_volume_price_divergence_strategy(self, symbol, interval):
        """Estrategia Volume-Price Divergence para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['VOLUME_PRICE_DIVERGENCE']['timeframes']:
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calcular RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            # Analizar divergencia precio-volumen manualmente
            current_idx = -1
            lookback = 10
            
            if len(close) < lookback + 1:
                return None
            
            # Buscar divergencias alcistas (precio nuevo bajo, volumen decreciente)
            recent_lows = low[-lookback:]
            recent_volumes = volume[-lookback:]
            
            min_low_idx = np.argmin(recent_lows)
            min_low = recent_lows[min_low_idx]
            
            # Verificar si el mínimo actual es el más reciente
            if min_low_idx == lookback - 1:
                # Buscar mínimo anterior
                prev_lows = low[-(lookback*2):-lookback]
                if len(prev_lows) > 0:
                    prev_min_low = np.min(prev_lows)
                    prev_min_idx = np.argmin(prev_lows)
                    
                    # Volumen en mínimos actual y anterior
                    volume_at_current_min = volume[current_idx]
                    volume_at_prev_min = volume[-(lookback + prev_min_idx)]
                    
                    # Divergencia alcista: precio hace nuevo bajo pero volumen es menor
                    bullish_divergence = (
                        min_low < prev_min_low and
                        volume_at_current_min < volume_at_prev_min * 0.8  # 20% menos volumen
                    )
                else:
                    bullish_divergence = False
            else:
                bullish_divergence = False
            
            # Buscar divergencias bajistas (precio nuevo alto, volumen decreciente)
            recent_highs = high[-lookback:]
            max_high_idx = np.argmax(recent_highs)
            max_high = recent_highs[max_high_idx]
            
            if max_high_idx == lookback - 1:
                prev_highs = high[-(lookback*2):-lookback]
                if len(prev_highs) > 0:
                    prev_max_high = np.max(prev_highs)
                    prev_max_idx = np.argmax(prev_highs)
                    
                    volume_at_current_max = volume[current_idx]
                    volume_at_prev_max = volume[-(lookback + prev_max_idx)]
                    
                    bearish_divergence = (
                        max_high > prev_max_high and
                        volume_at_current_max < volume_at_prev_max * 0.8
                    )
                else:
                    bearish_divergence = False
            else:
                bearish_divergence = False
            
            # Condiciones adicionales con FTMaverick
            long_conditions = (
                bullish_divergence and
                rsi_maverick[current_idx] < 0.5 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            short_conditions = (
                bearish_divergence and
                rsi_maverick[current_idx] > 0.5 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not ftm_data['no_trade_zones'][current_idx]
            )
            
            if long_conditions:
                signal_type = 'LONG'
            elif short_conditions:
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico
            chart_buffer = self.generate_volume_price_divergence_chart(
                symbol, interval, df, volume, rsi_maverick, ftm_data, signal_type
            )
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'][0],
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'strategy': 'VOLUME_PRICE_DIVERGENCE',
                'divergence_type': 'BULLISH' if bullish_divergence else 'BEARISH',
                'rsi_maverick': rsi_maverick[-1],
                'volume_ratio': volume[-1] / np.mean(volume[-20:]),
                'ftm_signal': ftm_data['strength_signals'][-1]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_volume_price_divergence_strategy para {symbol} {interval}: {e}")
            return None

    def check_volume_ema_ftm_strategy(self, symbol, interval):
        """Estrategia Desplome de Volumen mejorada para Telegram"""
        try:
            if interval not in STRATEGY_CONFIG['VOLUME_EMA_FTM']['timeframes']:
                return None
            
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular EMA21 y Volume MA21
            ema_21 = self.calculate_ema(close, 21)
            volume_ma_21 = self.calculate_sma(volume, 21)
            
            # Calcular FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            current_close = close[current_idx]
            current_volume = volume[current_idx]
            current_volume_ma = volume_ma_21[current_idx]
            current_ema_21 = ema_21[current_idx]
            
            # Nueva condición mejorada: Volumen > 3x MA21 (más estricto)
            volume_condition = current_volume > (current_volume_ma * 3.0)
            
            if not volume_condition:
                return None
            
            # Determinar señal basado en EMA y condiciones adicionales
            if current_close > current_ema_21:
                signal_type = 'LONG'
                # Condiciones adicionales para LONG mejorado
                additional_conditions = (
                    ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                    not ftm_data['no_trade_zones'][current_idx] and
                    current_close > np.mean(close[-20:])  # Precio por encima de media de 20 periodos
                )
                if not additional_conditions:
                    return None
                    
            elif current_close < current_ema_21:
                signal_type = 'SHORT'
                # Condiciones adicionales para SHORT mejorado
                additional_conditions = (
                    ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                    not ftm_data['no_trade_zones'][current_idx] and
                    current_close < np.mean(close[-20:])  # Precio por debajo de media de 20 periodos
                )
                if not additional_conditions:
                    return None
            else:
                return None
            
            # Filtro Multi-Timeframe para temporalidades cortas
            if interval in ['15m', '30m', '1h', '4h']:
                hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                if hierarchy:
                    # Timeframe Mayor
                    mayor_df = self.get_kucoin_data(symbol, hierarchy['mayor'], 50)
                    if mayor_df is not None and len(mayor_df) > 20:
                        mayor_trend = self.check_multi_timeframe_trend(symbol, hierarchy['mayor'])
                        if signal_type == 'LONG':
                            mayor_ok = mayor_trend.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                        else:
                            mayor_ok = mayor_trend.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                    else:
                        mayor_ok = False
                    
                    # Timeframe Menor
                    menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
                    if menor_df is not None and len(menor_df) > 10:
                        menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                        if signal_type == 'LONG':
                            menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']
                        else:
                            menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
                    else:
                        menor_ok = False
                    
                    if not (mayor_ok and menor_ok):
                        return None
            
            # Calcular niveles de entrada/salida
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, 15, support_levels, resistance_levels
            )
            
            # Generar gráfico mejorado
            chart_buffer = self.generate_volume_ema_ftm_chart(
                symbol, interval, df, ema_21, volume_ma_21, ftm_data, signal_type
            )
            
            # Obtener información de multi-timeframe
            if interval in ['15m', '30m', '1h', '4h']:
                tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
                mayor_trend = tf_analysis.get('mayor', 'NEUTRAL')
                menor_trend = "ALCISTA" if signal_type == 'LONG' else "BAJISTA"
            else:
                mayor_trend = "N/A"
                menor_trend = "N/A"
            
            current_time = self.get_bolivia_time()
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_close,
                'volume_ratio': current_volume / current_volume_ma,
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'chart': chart_buffer,
                'entry': levels_data['entry'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'mayor_trend': mayor_trend,
                'menor_trend': menor_trend,
                'strategy': 'VOLUME_EMA_FTM',
                'ema21_price': current_ema_21,
                'ftm_signal': ftm_data['strength_signals'][-1],
                'volume_ma': current_volume_ma
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_volume_ema_ftm_strategy para {symbol} {interval}: {e}")
            return None

    # =====================================================================
    # FUNCIONES PARA GENERAR GRÁFICOS DE ESTRATEGIAS
    # =====================================================================

    def generate_trend_rider_chart(self, symbol, interval, df, ma_50, ma_200, macd, signal, ftm_data, signal_type):
        """Generar gráfico para Trend Rider"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Gráfico 1: Precio con MA50 y MA200
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Velas
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            # Medias móviles
            ax1.plot(dates_matplotlib, ma_50[-50:], label='MA50', color='blue', linewidth=2)
            ax1.plot(dates_matplotlib, ma_200[-50:], label='MA200', color='purple', linewidth=2)
            
            # Destacar vela actual
            ax1.scatter(dates_matplotlib[-1], closes[-1], 
                       color='green' if signal_type == 'LONG' else 'red', 
                       s=100, zorder=5, marker='o')
            
            ax1.set_title(f'{symbol} - {interval} - Trend Rider ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: MACD
            macd_dates = dates_matplotlib[-len(macd[-50:]):]
            ax2.plot(macd_dates, macd[-50:], 'blue', linewidth=1, label='MACD')
            ax2.plot(macd_dates, signal[-50:], 'red', linewidth=1, label='Señal')
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax3.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            # Umbral y zonas no operar
            if 'high_zone_threshold' in ftm_data:
                threshold = ftm_data['high_zone_threshold']
                ax3.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7)
                ax3.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = ftm_data['no_trade_zones'][-50:]
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax3.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax3.set_ylabel('Fuerza Tendencia')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Trend Rider: {e}")
            return None

    def generate_momentum_divergence_chart(self, symbol, interval, df, rsi_traditional, rsi_maverick, volume_data, ftm_data, signal_type):
        """Generar gráfico para Momentum Divergence"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            ax1.set_title(f'{symbol} - {interval} - Momentum Divergence ({signal_type})')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: RSI Tradicional
            rsi_trad_dates = dates_matplotlib[-len(rsi_traditional[-50:]):]
            ax2.plot(rsi_trad_dates, rsi_traditional[-50:], 'cyan', linewidth=2, label='RSI Tradicional')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI Tradicional')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: RSI Maverick
            rsi_mav_dates = dates_matplotlib[-len(rsi_maverick[-50:]):]
            ax3.plot(rsi_mav_dates, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('RSI Maverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: Volumen y FTMaverick
            volume_dates = dates_matplotlib[-len(volume_data['volume_ratio'][-50:]):]
            volumes = df['volume'].iloc[-50:].values
            
            # Colores según señal
            colors = []
            for i, signal in enumerate(volume_data['volume_signal'][-50:]):
                if signal == 'COMPRA':
                    colors.append('green')
                elif signal == 'VENTA':
                    colors.append('red')
                else:
                    colors.append('gray')
            
            ax4.bar(volume_dates, volumes, color=colors, alpha=0.6, label='Volumen')
            ax4.plot(volume_dates, volume_data['volume_ma'][-50:], 'orange', linewidth=2, label='MA Volumen')
            ax4.set_ylabel('Volumen')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Momentum Divergence: {e}")
            return None

    def generate_bollinger_squeeze_chart(self, symbol, interval, df, bb_upper, bb_middle, bb_lower, adx, plus_di, minus_di, ftm_data, signal_type):
        """Generar gráfico para Bollinger Squeeze"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Gráfico 1: Precio con Bollinger Bands
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Velas
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            # Bandas de Bollinger
            ax1.plot(dates_matplotlib, bb_upper[-50:], 'orange', alpha=0.5, linewidth=1, label='BB Superior')
            ax1.plot(dates_matplotlib, bb_middle[-50:], 'orange', alpha=0.5, linewidth=1, label='BB Media')
            ax1.plot(dates_matplotlib, bb_lower[-50:], 'orange', alpha=0.5, linewidth=1, label='BB Inferior')
            ax1.fill_between(dates_matplotlib, bb_lower[-50:], bb_upper[-50:], color='orange', alpha=0.1)
            
            ax1.set_title(f'{symbol} - {interval} - Bollinger Squeeze ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: ADX y DMI
            adx_dates = dates_matplotlib[-len(adx[-50:]):]
            ax2.plot(adx_dates, adx[-50:], 'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, minus_di[-50:], 'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7)
            ax2.set_ylabel('ADX/DMI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax3.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax3.set_ylabel('Fuerza Tendencia')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Bollinger Squeeze: {e}")
            return None

    def generate_adx_power_trend_chart(self, symbol, interval, df, ma_21, adx, plus_di, minus_di, ftm_data, signal_type):
        """Generar gráfico para ADX Power Trend"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Gráfico 1: Precio con MA21
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma_21[-50:], 'blue', linewidth=2, label='MA21')
            ax1.set_title(f'{symbol} - {interval} - ADX Power Trend ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: ADX y DMI
            adx_dates = dates_matplotlib[-len(adx[-50:]):]
            ax2.plot(adx_dates, adx[-50:], 'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, minus_di[-50:], 'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7)
            ax2.set_ylabel('ADX/DMI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax3.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax3.set_ylabel('Fuerza Tendencia')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico ADX Power Trend: {e}")
            return None

    def generate_macd_histogram_chart(self, symbol, interval, df, ma_9, ma_21, macd, signal, histogram, rsi_maverick, ftm_data, signal_type):
        """Generar gráfico para MACD Histogram Reversal"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con MA9 y MA21
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma_9[-50:], 'red', linewidth=1, label='MA9')
            ax1.plot(dates_matplotlib, ma_21[-50:], 'blue', linewidth=2, label='MA21')
            ax1.set_title(f'{symbol} - {interval} - MACD Histogram Reversal ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: MACD
            macd_dates = dates_matplotlib[-len(macd[-50:]):]
            ax2.plot(macd_dates, macd[-50:], 'blue', linewidth=1, label='MACD')
            ax2.plot(macd_dates, signal[-50:], 'red', linewidth=1, label='Señal')
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Histograma MACD
            hist_dates = dates_matplotlib[-len(histogram[-50:]):]
            colors_hist = ['green' if x > 0 else 'red' for x in histogram[-50:]]
            ax3.bar(hist_dates, histogram[-50:], color=colors_hist, alpha=0.6)
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax3.set_ylabel('Histograma MACD')
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: RSI Maverick y FTMaverick
            rsi_dates = dates_matplotlib[-len(rsi_maverick[-50:]):]
            ax4.plot(rsi_dates, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax4.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax4.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax4.set_ylabel('RSI Maverick')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico MACD Histogram: {e}")
            return None

    def generate_volume_spike_chart(self, symbol, interval, df, ma_21, volume_data, rsi_maverick, ftm_data, signal_type):
        """Generar gráfico para Volume Spike Momentum"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con MA21
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma_21[-50:], 'blue', linewidth=2, label='MA21')
            ax1.set_title(f'{symbol} - {interval} - Volume Spike Momentum ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: Volumen
            volume_dates = dates_matplotlib[-len(volume_data['volume_ratio'][-50:]):]
            volumes = df['volume'].iloc[-50:].values
            
            # Colores según señal
            colors = []
            for i, signal in enumerate(volume_data['volume_signal'][-50:]):
                if signal == 'COMPRA':
                    colors.append('green')
                elif signal == 'VENTA':
                    colors.append('red')
                else:
                    colors.append('gray')
            
            ax2.bar(volume_dates, volumes, color=colors, alpha=0.6, label='Volumen')
            ax2.plot(volume_dates, volume_data['volume_ma'][-50:], 'orange', linewidth=2, label='MA Volumen')
            
            # Marcar anomalías
            anomaly_indices = [i for i, anomaly in enumerate(volume_data['volume_anomaly'][-50:]) if anomaly]
            if anomaly_indices:
                anomaly_dates = [volume_dates[i] for i in anomaly_indices]
                anomaly_volumes = [volumes[i] for i in anomaly_indices]
                ax2.scatter(anomaly_dates, anomaly_volumes, color='purple', s=30, marker='x', label='Anomalías')
            
            ax2.set_ylabel('Volumen')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: RSI Maverick
            rsi_dates = dates_matplotlib[-len(rsi_maverick[-50:]):]
            ax3.plot(rsi_dates, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('RSI Maverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors_ftm = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax4.bar(trend_dates[i], trend_strength[i], color=colors_ftm[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Volume Spike: {e}")
            return None

    def generate_double_rsi_chart(self, symbol, interval, df, rsi_traditional, rsi_maverick, bb_upper, bb_middle, bb_lower, ftm_data, signal_type):
        """Generar gráfico para Double Confirmation RSI"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con Bollinger Bands
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            # Bollinger Bands
            ax1.plot(dates_matplotlib, bb_upper[-50:], 'orange', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_middle[-50:], 'orange', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_lower[-50:], 'orange', alpha=0.5, linewidth=1)
            ax1.fill_between(dates_matplotlib, bb_lower[-50:], bb_upper[-50:], color='orange', alpha=0.1)
            
            ax1.set_title(f'{symbol} - {interval} - Double Confirmation RSI ({signal_type})')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: RSI Tradicional
            rsi_trad_dates = dates_matplotlib[-len(rsi_traditional[-50:]):]
            ax2.plot(rsi_trad_dates, rsi_traditional[-50:], 'cyan', linewidth=2, label='RSI Tradicional')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI Tradicional')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: RSI Maverick
            rsi_mav_dates = dates_matplotlib[-len(rsi_maverick[-50:]):]
            ax3.plot(rsi_mav_dates, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('RSI Maverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax4.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Double RSI: {e}")
            return None

    def generate_trend_strength_chart(self, symbol, interval, df, ftm_data, ma_50, volume_data, signal_type):
        """Generar gráfico para Trend Strength Maverick"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con MA50
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma_50[-50:], 'blue', linewidth=2, label='MA50')
            ax1.set_title(f'{symbol} - {interval} - Trend Strength Maverick ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: FTMaverick - Fuerza de Tendencia
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax2.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            # Umbral
            if 'high_zone_threshold' in ftm_data:
                threshold = ftm_data['high_zone_threshold']
                ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7)
                ax2.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            ax2.set_ylabel('Fuerza Tendencia')
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: FTMaverick - Ancho de Bandas
            bb_width_dates = dates_matplotlib[-len(ftm_data['bb_width'][-50:]):]
            ax3.plot(bb_width_dates, ftm_data['bb_width'][-50:], 'purple', linewidth=2, label='Ancho BB %')
            
            # Zonas no operar
            no_trade_zones = ftm_data['no_trade_zones'][-50:]
            for i, date in enumerate(bb_width_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax3.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax3.set_ylabel('Ancho BB %')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: Volumen
            volume_dates = dates_matplotlib[-len(volume_data['volume_ratio'][-50:]):]
            volumes = df['volume'].iloc[-50:].values
            
            ax4.bar(volume_dates, volumes, color='blue', alpha=0.6, label='Volumen')
            ax4.plot(volume_dates, volume_data['volume_ma'][-50:], 'orange', linewidth=2, label='MA Volumen')
            ax4.set_ylabel('Volumen')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Trend Strength: {e}")
            return None

    def generate_whale_following_chart(self, symbol, interval, df, whale_data, ma_200, adx, plus_di, minus_di, ftm_data, signal_type):
        """Generar gráfico para Whale Following"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con MA200
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma_200[-50:], 'purple', linewidth=2, label='MA200')
            ax1.set_title(f'{symbol} - {interval} - Whale Following ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: Señales de Ballenas
            whale_dates = dates_matplotlib[-len(whale_data['whale_pump'][-50:]):]
            ax2.bar(whale_dates, whale_data['whale_pump'][-50:], 
                   color='green', alpha=0.6, label='Ballenas Compradoras')
            ax2.bar(whale_dates, [-x for x in whale_data['whale_dump'][-50:]], 
                   color='red', alpha=0.6, label='Ballenas Vendedoras')
            
            # Marcar señales confirmadas
            confirmed_buy_indices = [i for i, confirmed in enumerate(whale_data['confirmed_buy'][-50:]) if confirmed]
            confirmed_sell_indices = [i for i, confirmed in enumerate(whale_data['confirmed_sell'][-50:]) if confirmed]
            
            if confirmed_buy_indices:
                buy_dates = [whale_dates[i] for i in confirmed_buy_indices]
                buy_values = [whale_data['whale_pump'][-50:][i] for i in confirmed_buy_indices]
                ax2.scatter(buy_dates, buy_values, color='darkgreen', s=50, marker='^', label='Compra Confirmada')
            
            if confirmed_sell_indices:
                sell_dates = [whale_dates[i] for i in confirmed_sell_indices]
                sell_values = [-whale_data['whale_dump'][-50:][i] for i in confirmed_sell_indices]
                ax2.scatter(sell_dates, sell_values, color='darkred', s=50, marker='v', label='Venta Confirmada')
            
            ax2.set_ylabel('Fuerza Ballenas')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: ADX y DMI
            adx_dates = dates_matplotlib[-len(adx[-50:]):]
            ax3.plot(adx_dates, adx[-50:], 'black', linewidth=2, label='ADX')
            ax3.plot(adx_dates, plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax3.plot(adx_dates, minus_di[-50:], 'red', linewidth=1, label='-DI')
            ax3.axhline(y=25, color='yellow', linestyle='--', alpha=0.7)
            ax3.set_ylabel('ADX/DMI')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax4.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Whale Following: {e}")
            return None

    def generate_ma_convergence_chart(self, symbol, interval, df, ma_9, ma_21, ma_50, macd, signal, histogram, ftm_data, signal_type):
        """Generar gráfico para MA Convergence Divergence"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con todas las MAs
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma_9[-50:], 'red', linewidth=1, label='MA9')
            ax1.plot(dates_matplotlib, ma_21[-50:], 'blue', linewidth=2, label='MA21')
            ax1.plot(dates_matplotlib, ma_50[-50:], 'green', linewidth=2, label='MA50')
            ax1.set_title(f'{symbol} - {interval} - MA Convergence ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: MACD
            macd_dates = dates_matplotlib[-len(macd[-50:]):]
            ax2.plot(macd_dates, macd[-50:], 'blue', linewidth=1, label='MACD')
            ax2.plot(macd_dates, signal[-50:], 'red', linewidth=1, label='Señal')
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Histograma MACD
            hist_dates = dates_matplotlib[-len(histogram[-50:]):]
            colors_hist = ['green' if x > 0 else 'red' for x in histogram[-50:]]
            ax3.bar(hist_dates, histogram[-50:], color=colors_hist, alpha=0.6)
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax3.set_ylabel('Histograma MACD')
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax4.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico MA Convergence: {e}")
            return None

    def generate_rsi_extreme_chart(self, symbol, interval, df, rsi_maverick, bb_upper, bb_middle, bb_lower, ftm_data, signal_type):
        """Generar gráfico para RSI Maverick Extreme"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con Bollinger Bands
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            # Bollinger Bands
            ax1.plot(dates_matplotlib, bb_upper[-50:], 'orange', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_middle[-50:], 'orange', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_lower[-50:], 'orange', alpha=0.5, linewidth=1)
            ax1.fill_between(dates_matplotlib, bb_lower[-50:], bb_upper[-50:], color='orange', alpha=0.1)
            
            ax1.set_title(f'{symbol} - {interval} - RSI Maverick Extreme ({signal_type})')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: RSI Maverick
            rsi_dates = dates_matplotlib[-len(rsi_maverick[-50:]):]
            ax2.plot(rsi_dates, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            
            # Zonas extremas
            ax2.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Extremo Alto')
            ax2.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='Extremo Bajo')
            ax2.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            
            # Rellenar zonas extremas
            ax2.fill_between(rsi_dates, 0.85, 1.0, color='red', alpha=0.1)
            ax2.fill_between(rsi_dates, 0.0, 0.15, color='green', alpha=0.1)
            
            ax2.set_ylabel('RSI Maverick')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Historial RSI (últimas 10 velas)
            if len(rsi_maverick) >= 10:
                recent_rsi = rsi_maverick[-10:]
                recent_dates = dates_matplotlib[-10:]
                ax3.bar(recent_dates, recent_rsi, color=['red' if r > 0.85 else 'green' if r < 0.15 else 'blue' for r in recent_rsi])
                ax3.axhline(y=0.85, color='red', linestyle='--', alpha=0.7)
                ax3.axhline(y=0.15, color='green', linestyle='--', alpha=0.7)
                ax3.set_ylabel('RSI (10 velas)')
                ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax4.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico RSI Extreme: {e}")
            return None

    def generate_volume_price_divergence_chart(self, symbol, interval, df, volume, rsi_maverick, ftm_data, signal_type):
        """Generar gráfico para Volume-Price Divergence"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            ax1.set_title(f'{symbol} - {interval} - Volume-Price Divergence ({signal_type})')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: Volumen
            volume_dates = dates_matplotlib[-len(volume[-50:]):]
            volumes = volume[-50:]
            
            # Calcular media móvil de volumen
            volume_ma = self.calculate_sma(volumes, 20)
            
            ax2.bar(volume_dates, volumes, color='blue', alpha=0.6, label='Volumen')
            ax2.plot(volume_dates, volume_ma, 'orange', linewidth=2, label='MA Volumen')
            ax2.set_ylabel('Volumen')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: RSI Maverick
            rsi_dates = dates_matplotlib[-len(rsi_maverick[-50:]):]
            ax3.plot(rsi_dates, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('RSI Maverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax4.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Volume-Price Divergence: {e}")
            return None

    def generate_volume_ema_ftm_chart(self, symbol, interval, df, ema_21, volume_ma_21, ftm_data, signal_type):
        """Generar gráfico mejorado para Volume EMA FTM"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Gráfico 1: Precio con EMA21
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Velas
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], 
                        [open_price, close_price], color=color, linewidth=3)
            
            # EMA21
            ax1.plot(dates_matplotlib, ema_21[-50:], label='EMA21', color='blue', linewidth=2)
            
            # Destacar vela actual
            ax1.scatter(dates_matplotlib[-1], closes[-1], 
                       color='green' if signal_type == 'LONG' else 'red', 
                       s=100, zorder=5, marker='o')
            
            ax1.set_title(f'{symbol} - {interval} - Volume EMA FTM ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico 2: Volumen y Volume MA21
            volume = df['volume'].iloc[-50:].values
            volume_ma = volume_ma_21[-50:]
            
            # Colorear barras según dirección del precio
            colors = []
            for i in range(len(closes)):
                if i == 0:
                    colors.append('gray')
                else:
                    colors.append('green' if closes[i] > closes[i-1] else 'red')
            
            ax2.bar(dates_matplotlib, volume, color=colors, alpha=0.6, label='Volumen')
            ax2.plot(dates_matplotlib, volume_ma, label='Volume MA21', color='orange', linewidth=2)
            
            # Línea de 3x volumen
            ax2.axhline(y=volume_ma[-1] * 3, color='red', linestyle='--', alpha=0.7, label='3x Volumen')
            
            ax2.set_ylabel('Volumen')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: FTMaverick
            trend_dates = dates_matplotlib[-len(ftm_data['trend_strength'][-50:]):]
            trend_strength = ftm_data['trend_strength'][-50:]
            colors_ftm = ftm_data['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax3.bar(trend_dates[i], trend_strength[i], color=colors_ftm[i], alpha=0.7, width=0.8)
            
            # Zonas no operar
            no_trade_zones = ftm_data['no_trade_zones'][-50:]
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax3.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax3.set_ylabel('Fuerza Tendencia')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Volume EMA FTM: {e}")
            return None


# Revisar esta parte

   def evaluate_signal_conditions_corrected(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con PESOS CORREGIDOS"""
        # Definir pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            weights = {
                'long': {
                    'multi_timeframe': 30,
                    'trend_strength': 25,
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
                    'multi_timeframe': 30,
                    'trend_strength': 25,
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
                    'whale_signal': 30,
                    'trend_strength': 25,
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
                    'whale_signal': 30,
                    'trend_strength': 25,
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
                    'trend_strength': 55,
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
                    'trend_strength': 55,
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
        
        # ADX con pendiente positiva
        conditions['long']['adx_dmi']['value'] = (
            data['adx_slope_positive'][current_idx] if current_idx < len(data.get('adx_slope_positive', [])) else False
        )
        
        # Cruce de medias 9 y 21
        conditions['long']['ma_cross']['value'] = (
            data['ma_cross_bullish'][current_idx] if current_idx < len(data.get('ma_cross_bullish', [])) else False
        )
        
        # Cruce de DMI
        conditions['long']['adx_dmi']['value'] = conditions['long']['adx_dmi']['value'] or (
            data['di_cross_bullish'][current_idx] if current_idx < len(data.get('di_cross_bullish', [])) else False
        )
        
        conditions['long']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bullish_divergence']) and 
            data['rsi_bullish_divergence'][current_idx]
        )
        conditions['long']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bullish_divergence']) and 
            data['rsi_maverick_bullish_divergence'][current_idx]
        )
        
        # Cruce MACD
        conditions['long']['macd']['value'] = (
            data['macd_cross_bullish'][current_idx] if current_idx < len(data.get('macd_cross_bullish', [])) else False
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
            current_idx < len(data['volume_clusters']) and 
            data['volume_clusters'][current_idx] and
            data['volume_signal'][current_idx] == 'COMPRA'
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
        
        # ADX con pendiente positiva para SHORT también
        conditions['short']['adx_dmi']['value'] = (
            data['adx_slope_positive'][current_idx] if current_idx < len(data.get('adx_slope_positive', [])) else False
        )
        
        # Cruce de medias 9 y 21 (bajista)
        conditions['short']['ma_cross']['value'] = (
            data['ma_cross_bearish'][current_idx] if current_idx < len(data.get('ma_cross_bearish', [])) else False
        )
        
        # Cruce de DMI (bajista)
        conditions['short']['adx_dmi']['value'] = conditions['short']['adx_dmi']['value'] or (
            data['di_cross_bearish'][current_idx] if current_idx < len(data.get('di_cross_bearish', [])) else False
        )
        
        conditions['short']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bearish_divergence']) and 
            data['rsi_bearish_divergence'][current_idx]
        )
        conditions['short']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bearish_divergence']) and 
            data['rsi_maverick_bearish_divergence'][current_idx]
        )
        
        # Cruce MACD (bajista)
        conditions['short']['macd']['value'] = (
            data['macd_cross_bearish'][current_idx] if current_idx < len(data.get('macd_cross_bearish', [])) else False
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
            current_idx < len(data['volume_clusters']) and 
            data['volume_clusters'][current_idx] and
            data['volume_signal'][current_idx] == 'VENTA'
        )
        
        return conditions

    def get_condition_description(self, condition_key):
        """Obtener descripción de condición"""
        descriptions = {
            'multi_timeframe': 'Multi-TF obligatorio',
            'trend_strength': 'Fuerza tendencia Maverick',
            'whale_signal': 'Señal ballenas confirmada',
            'bollinger_bands': 'Bandas de Bollinger',
            'adx_dmi': 'ADX con pendiente positiva',
            'ma_cross': 'Cruce MA9/MA21',
            'rsi_traditional_divergence': 'Divergencia RSI Tradicional',
            'rsi_maverick_divergence': 'Divergencia RSI Maverick',
            'macd': 'Cruce MACD',
            'chart_pattern': 'Patrón Chartista',
            'breakout': 'Ruptura confirmada',
            'volume_anomaly': 'Volumen Anómalo'
        }
        return descriptions.get(condition_key, condition_key)

 

#Estos son para revisar


def get_chart_pattern_name(self, data, current_idx):
        """Obtener nombre del patrón chartista"""
        if current_idx < len(data.get('chart_pattern_name', [])):
            pattern_name = data['chart_pattern_name'][current_idx]
            if pattern_name:
                return f"Patrón Chartista: {pattern_name}"
        return "Patrón Chartista"

  
    def generate_multiframe_chart(self, symbol, interval, data):
        """Generar gráfico completo para Telegram de estrategia Multiframe"""
        try:
            fig = plt.figure(figsize=(16, 20))
            
            # 1. Gráfico de velas con Bandas de Bollinger (3,1)
            ax1 = plt.subplot(8, 1, 1)
            
            if data['data']:
                dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                        else d['timestamp'] for d in data['data'][-50:]]
                opens = [d['open'] for d in data['data'][-50:]]
                highs = [d['high'] for d in data['data'][-50:]]
                lows = [d['low'] for d in data['data'][-50:]]
                closes = [d['close'] for d in data['data'][-50:]]
                
                dates_matplotlib = mdates.date2num(dates)
                
                # Velas japonesas
                for i in range(len(dates_matplotlib)):
                    color = 'green' if closes[i] >= opens[i] else 'red'
                    ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [lows[i], highs[i]], color='black', linewidth=1)
                    ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [opens[i], closes[i]], color=color, linewidth=3)
                
                # Bandas de Bollinger transparentes
                if 'indicators' in data and 'bb_upper' in data['indicators']:
                    bb_upper = data['indicators']['bb_upper'][-50:]
                    bb_middle = data['indicators']['bb_middle'][-50:]
                    bb_lower = data['indicators']['bb_lower'][-50:]
                    
                    ax1.fill_between(dates_matplotlib, bb_lower, bb_upper, color='orange', alpha=0.1)
                    ax1.plot(dates_matplotlib, bb_middle, color='orange', alpha=0.5, linewidth=1)
                
                # Medias móviles
                if 'indicators' in data:
                    for ma_name, ma_color in [('ma_9', 'red'), ('ma_21', 'blue'), ('ma_50', 'green'), ('ma_200', 'purple')]:
                        if ma_name in data['indicators']:
                            ma_values = data['indicators'][ma_name][-50:]
                            ax1.plot(dates_matplotlib, ma_values, color=ma_color, alpha=0.7, linewidth=1, label=ma_name.upper())
                
                # Soportes y resistencias
                if 'support_levels' in data:
                    for level in data['support_levels'][:3]:  # Primeros 3 soportes
                        ax1.axhline(y=level, color='blue', linestyle='--', alpha=0.5, linewidth=1)
                
                if 'resistance_levels' in data:
                    for level in data['resistance_levels'][:3]:  # Primeras 3 resistencias
                        ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            ax1.set_title(f'{symbol} - {interval} - Gráfico de Velas')
            ax1.legend(loc='upper left', fontsize='small')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # 2. ADX con DMI (3,2)
            ax2 = plt.subplot(8, 1, 2)
            if 'indicators' in data:
                adx_dates = dates_matplotlib[-len(data['indicators']['adx'][-50:]):]
                ax2.plot(adx_dates, data['indicators']['adx'][-50:], 'black', linewidth=2, label='ADX')
                ax2.plot(adx_dates, data['indicators']['plus_di'][-50:], 'green', linewidth=1, label='+DI')
                ax2.plot(adx_dates, data['indicators']['minus_di'][-50:], 'red', linewidth=1, label='-DI')
                ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
            ax2.set_ylabel('ADX/DMI')
            ax2.legend(loc='upper left', fontsize='small')
            ax2.grid(True, alpha=0.3)
            
            # 3. Indicador de Volumen con Anomalías (3,3)
            ax3 = plt.subplot(8, 1, 3)
            if 'indicators' in data and 'volume_anomaly' in data['indicators']:
                volume_dates = dates_matplotlib[-len(data['indicators']['volume_anomaly'][-50:]):]
                volumes = [d['volume'] for d in data['data'][-50:]]
                
                # Colores según señal de volumen
                colors = []
                volume_signal = data['indicators'].get('volume_signal', ['NEUTRAL'] * 50)
                for i, signal in enumerate(volume_signal[-50:]):
                    if signal == 'COMPRA':
                        colors.append('green')
                    elif signal == 'VENTA':
                        colors.append('red')
                    else:
                        colors.append('gray')
                
                ax3.bar(volume_dates, volumes[-50:], color=colors, alpha=0.6, label='Volumen')
                
                # Volume MA
                if 'volume_ma' in data['indicators']:
                    ax3.plot(volume_dates, data['indicators']['volume_ma'][-50:], 'orange', linewidth=2, label='MA Volumen')
                
                # Marcar anomalías
                anomaly_indices = [i for i, anomaly in enumerate(data['indicators']['volume_anomaly'][-50:]) if anomaly]
                if anomaly_indices:
                    anomaly_dates = [volume_dates[i] for i in anomaly_indices]
                    anomaly_volumes = [volumes[-50:][i] for i in anomaly_indices]
                    ax3.scatter(anomaly_dates, anomaly_volumes, color='purple', s=30, marker='x', label='Anomalías')
            
            ax3.set_ylabel('Volumen')
            ax3.legend(loc='upper left', fontsize='small')
            ax3.grid(True, alpha=0.3)
            
            # 4. Fuerza de Tendencia Maverick (4,1)
            ax4 = plt.subplot(8, 1, 4)
            if 'indicators' in data and 'trend_strength' in data['indicators']:
                trend_dates = dates_matplotlib[-len(data['indicators']['trend_strength'][-50:]):]
                trend_strength = data['indicators']['trend_strength'][-50:]
                
                # Barras coloreadas
                colors = data['indicators'].get('colors', ['gray'] * 50)[-50:]
                for i in range(len(trend_dates)):
                    ax4.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
                
                # Línea de umbral
                if 'high_zone_threshold' in data['indicators']:
                    threshold = data['indicators']['high_zone_threshold']
                    ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, linewidth=1)
                    ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            
            # 5. Indicador de Ballenas (solo para 12h y 1D)
            ax5 = plt.subplot(8, 1, 5)
            if interval in ['12h', '1D'] and 'indicators' in data:
                whale_dates = dates_matplotlib[-len(data['indicators']['whale_pump'][-50:]):]
                
                # Barras de ballenas
                ax5.bar(whale_dates, data['indicators']['whale_pump'][-50:], 
                       color='green', alpha=0.6, label='Ballenas Compradoras')
                ax5.bar(whale_dates, [-x for x in data['indicators']['whale_dump'][-50:]], 
                       color='red', alpha=0.6, label='Ballenas Vendedoras')
            
            ax5.set_ylabel('Ballenas')
            ax5.legend(loc='upper left', fontsize='small')
            ax5.grid(True, alpha=0.3)
            
            # 6. RSI Modificado Maverick (6,1)
            ax6 = plt.subplot(8, 1, 6)
            if 'indicators' in data and 'rsi_maverick' in data['indicators']:
                rsi_dates = dates_matplotlib[-len(data['indicators']['rsi_maverick'][-50:]):]
                ax6.plot(rsi_dates, data['indicators']['rsi_maverick'][-50:], 'blue', linewidth=2)
                ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, linewidth=1)
                ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
            ax6.set_ylabel('RSI Maverick')
            ax6.grid(True, alpha=0.3)
            
            # 7. RSI Tradicional (7,1)
            ax7 = plt.subplot(8, 1, 7)
            if 'indicators' in data and 'rsi_traditional' in data['indicators']:
                rsi_trad_dates = dates_matplotlib[-len(data['indicators']['rsi_traditional'][-50:]):]
                ax7.plot(rsi_trad_dates, data['indicators']['rsi_traditional'][-50:], 'cyan', linewidth=2)
                ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7, linewidth=1)
                ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3, linewidth=1)
            ax7.set_ylabel('RSI Tradicional')
            ax7.grid(True, alpha=0.3)
            
            # 8. MACD (8,1)
            ax8 = plt.subplot(8, 1, 8)
            if 'indicators' in data:
                macd_dates = dates_matplotlib[-len(data['indicators']['macd'][-50:]):]
                ax8.plot(macd_dates, data['indicators']['macd'][-50:], 'blue', linewidth=1, label='MACD')
                ax8.plot(macd_dates, data['indicators']['macd_signal'][-50:], 'red', linewidth=1, label='Señal')
                
                # Histograma como barras
                macd_hist = data['indicators']['macd_histogram'][-50:]
                colors_hist = ['green' if x > 0 else 'red' for x in macd_hist]
                ax8.bar(macd_dates, macd_hist, color=colors_hist, alpha=0.6, label='Histograma')
                
                ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            
            ax8.set_ylabel('MACD')
            ax8.legend(loc='upper left', fontsize='small')
            ax8.grid(True, alpha=0.3)
            ax8.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, facecolor='white')
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico multiframe: {e}")
            return None



#Hasta aqui es para revisar



   
     def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal basada en condiciones ponderadas - CORREGIDO"""
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
        
        # Score mínimo ajustado según posición de MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:  # short
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0

        return min(final_score, 100), fulfilled_conditions



        

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

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - CON PESOS CORREGIDOS"""
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
            adx_slope_positive = self.check_adx_slope(adx)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence_traditional(close, rsi_traditional)
            
            breakout_up, breakout_down = self.check_breakout(high, low, close, whale_data['support'], whale_data['resistance'])
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # Cruce de medias
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover(ma_9, ma_21)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_bullish, macd_cross_bearish = self.check_macd_crossover(macd, macd_signal)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Verificar condiciones de Bollinger
            bollinger_conditions_long = self.check_bollinger_conditions_corrected(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions_corrected(df, interval, 'SHORT')
            
            # Indicador de volumen
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            # Verificar condiciones multi-timeframe obligatorias
            multi_timeframe_long = self.check_multi_timeframe_obligatory(symbol, interval, 'LONG')
            multi_timeframe_short = self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT')
            
            # Calcular soportes y resistencias dinámicos
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
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
                'chart_patterns': {
                    'head_shoulders': chart_patterns['head_shoulders'],
                    'double_top': chart_patterns['double_top'],
                    'double_bottom': chart_patterns['double_bottom'],
                    'bullish_flag': chart_patterns['bullish_flag'],
                    'bearish_flag': chart_patterns['bearish_flag']
                },
                'chart_pattern_name': chart_patterns['pattern_name'],
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
                'volume_signal': volume_data['volume_signal'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
            current_idx = -1
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
            
            if long_score >= (65 if ma200_condition == 'above' else 70):
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= (65 if ma200_condition == 'below' else 70):
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # Calcular niveles de trading
            levels_data = self.calculate_optimal_entry_exit(
                df, signal_type, leverage, support_levels, resistance_levels
            )
            
            # Obtener nombre del patrón chartista si aplica
            chart_pattern_desc = ""
            if 'chart_pattern' in [c.split(':')[0] for c in fulfilled_conditions]:
                pattern_name = chart_patterns['pattern_name'][current_idx]
                if pattern_name:
                    chart_pattern_desc = f" ({pattern_name})"
                    # Reemplazar en fulfilled_conditions
                    for i, cond in enumerate(fulfilled_conditions):
                        if 'Patrón Chartista' in cond:
                            fulfilled_conditions[i] = f"Patrón Chartista: {pattern_name}"
            
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
                    'volume_signal': volume_data['volume_signal'][-50:],
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

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert_strategy(alert_data):
    """Enviar alerta por Telegram para estrategias específicas"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Construir mensaje según la estrategia
        strategy_name = alert_data.get('strategy', '')
        signal_type = alert_data.get('signal', '')
        symbol = alert_data.get('symbol', '')
        interval = alert_data.get('interval', '')
        current_price = alert_data.get('current_price', 0)
        entry_price = alert_data.get('entry', 0)
        
        # Construir mensaje base
        message = f"""
🚨 {'LONG' if signal_type == 'LONG' else 'SHORT'} {symbol} en {interval} 🚨
Estrategia: {strategy_name}
Precio actual: ${current_price:.6f} | Entrada: ${entry_price:.6f}
"""
        
        # Añadir información específica según estrategia
        if strategy_name == 'TREND_RIDER':
            message += f"""
Filtros:
- Cruce MACD Temporalidad Menor
- Precio {'>' if signal_type == 'LONG' else '<'} de MA50 Temporalidad Actual
- Precio {'>' if signal_type == 'LONG' else '<'} MA200 Temporalidad Mayor
- Señal FTMaverick: {alert_data.get('ftm_signal', 'N/A')}
"""
            if interval in ['4h', '8h', '12h', '1D']:
                message += "Recomendación: Swing Trading.\n"
            elif interval in ['1h', '2h']:
                message += "Recomendación: Intraday.\n"
                
        elif strategy_name == 'BOLLINGER_SQUEEZE':
            message += f"""
Filtros:
- Compresión Bollinger: {alert_data.get('bb_width', 0):.1f}%
- ADX >25: {'✅' if alert_data.get('adx_value', 0) > 25 else '❌'}
- Breakout con volumen {'COMPRA' if signal_type == 'LONG' else 'VENTA'}
- Señal FTMaverick: {alert_data.get('ftm_signal', 'N/A')}
"""
            if interval in ['15m', '30m', '1h']:
                message += "Recomendación: Scalping/Intraday.\n"
                
        elif strategy_name == 'WHALE_FOLLOWING':
            message += f"""
Filtros:
- Señal Ballenas: {alert_data.get('whale_signal', 0):.1f}
- {'+DI > -DI' if signal_type == 'LONG' else '-DI > +DI'}: ✅
- ADX >25: {'✅' if alert_data.get('adx_value', 0) > 25 else '❌'}
- Precio {'>' if signal_type == 'LONG' else '<'} MA200
- Señal FTMaverick: {alert_data.get('ftm_signal', 'N/A')}
"""
            message += "Recomendación: Swing Trading/Spot.\n"
            
        elif strategy_name == 'VOLUME_EMA_FTM':
            message += f"""
Filtros:
- Volumen: {alert_data.get('volume_ratio', 0):.1f}x MA
- Precio {'>' if signal_type == 'LONG' else '<'} EMA21
- FTMaverick OK
- Multi-Timeframe: {alert_data.get('mayor_trend', 'N/A')}/{alert_data.get('menor_trend', 'N/A')}
"""
            if interval in ['15m', '30m']:
                message += "Recomendación: Scalping.\n"
            elif interval in ['1h', '4h']:
                message += "Recomendación: Intraday.\n"
            elif interval in ['12h', '1D']:
                message += "Recomendación: Swing Trading.\n"
        
        # Añadir hora
        message += f"\n⏰ Hora Bolivia: {alert_data.get('timestamp', '')}"
        
        # Enviar imagen si existe
        if 'chart' in alert_data and alert_data['chart']:
            asyncio.run(bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=alert_data['chart'],
                caption=message
            ))
        else:
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            ))
            
        print(f"Alerta {strategy_name} enviada a Telegram: {symbol} {interval} {signal_type}")
        
    except Exception as e:
        print(f"Error enviando alerta de estrategia a Telegram: {e}")

def background_strategy_checker():
    """Verificador de estrategias en segundo plano para las 5 criptomonedas"""
    strategy_functions = {
        'TREND_RIDER': indicator.check_trend_rider_strategy,
        'MOMENTUM_DIVERGENCE': indicator.check_momentum_divergence_strategy,
        'BOLLINGER_SQUEEZE': indicator.check_bollinger_squeeze_strategy,
        'ADX_POWER_TREND': indicator.check_adx_power_trend_strategy,
        'MACD_HISTOGRAM_REVERSAL': indicator.check_macd_histogram_reversal_strategy,
        'VOLUME_SPIKE_MOMENTUM': indicator.check_volume_spike_momentum_strategy,
        'DOUBLE_CONFIRMATION_RSI': indicator.check_double_confirmation_rsi_strategy,
        'TREND_STRENGTH_MAVERICK': indicator.check_trend_strength_maverick_strategy,
        'WHALE_FOLLOWING': indicator.check_whale_following_strategy,
        'MA_CONVERGENCE_DIVERGENCE': indicator.check_ma_convergence_divergence_strategy,
        'RSI_MAVERICK_EXTREME': indicator.check_rsi_maverick_extreme_strategy,
        'VOLUME_PRICE_DIVERGENCE': indicator.check_volume_price_divergence_strategy,
        'VOLUME_EMA_FTM': indicator.check_volume_ema_ftm_strategy
    }
    
    # Intervalos de verificación específicos para cada timeframe
    timeframe_check_intervals = {
        '15m': {'last_check': datetime.now(), 'interval_seconds': 60},
        '30m': {'last_check': datetime.now(), 'interval_seconds': 120},
        '1h': {'last_check': datetime.now(), 'interval_seconds': 300},
        '2h': {'last_check': datetime.now(), 'interval_seconds': 420},
        '4h': {'last_check': datetime.now(), 'interval_seconds': 420},
        '8h': {'last_check': datetime.now(), 'interval_seconds': 600},
        '12h': {'last_check': datetime.now(), 'interval_seconds': 900},
        '1D': {'last_check': datetime.now(), 'interval_seconds': 3600},
        '1W': {'last_check': datetime.now(), 'interval_seconds': 10000}
    }
    
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar cada timeframe según su intervalo específico
            for interval, check_info in timeframe_check_intervals.items():
                time_since_last_check = (current_time - check_info['last_check']).seconds
                
                if time_since_last_check >= check_info['interval_seconds']:
                    # Verificar si es hora de verificar este timeframe
                    if indicator.calculate_remaining_time(interval, current_time):
                        print(f"Verificando estrategias para timeframe {interval}...")
                        
                        # Para cada criptomoneda top 5
                        for symbol in TOP5_CRYPTOS:
                            # Verificar cada estrategia que opera en este timeframe
                            for strategy_name, strategy_func in strategy_functions.items():
                                # Verificar si la estrategia opera en este timeframe
                                if interval in STRATEGY_CONFIG.get(strategy_name, {}).get('timeframes', []):
                                    try:
                                        # Verificar horario de scalping para timeframes cortos
                                        if interval in ['15m', '30m'] and not indicator.is_scalping_time():
                                            continue
                                        
                                        # Ejecutar la estrategia
                                        signal_data = strategy_func(symbol, interval)
                                        
                                        if signal_data:
                                            # Verificar si ya enviamos esta señal recientemente
                                            cache_key = f"{strategy_name}_{symbol}_{interval}_{signal_data['signal']}"
                                            
                                            if cache_key not in indicator.strategy_signals:
                                                # Enviar alerta
                                                send_telegram_alert_strategy(signal_data)
                                                indicator.strategy_signals[cache_key] = current_time
                                            else:
                                                # Eliminar señales antiguas (más de 1 hora para intraday, 4 horas para swing)
                                                last_sent = indicator.strategy_signals[cache_key]
                                                max_age = 3600 if interval in ['15m', '30m', '1h', '2h'] else 14400
                                                
                                                if (current_time - last_sent).seconds > max_age:
                                                    send_telegram_alert_strategy(signal_data)
                                                    indicator.strategy_signals[cache_key] = current_time
                                            
                                            # Pequeña pausa entre estrategias
                                            time.sleep(0.5)
                                            
                                    except Exception as e:
                                        print(f"Error en estrategia {strategy_name} para {symbol} {interval}: {e}")
                                        continue
                        
                        # Actualizar último check
                        timeframe_check_intervals[interval]['last_check'] = current_time
            
            # Limpiar cache antiguo (señales mayores a 24 horas)
            cache_keys_to_remove = []
            for cache_key, sent_time in indicator.strategy_signals.items():
                if (current_time - sent_time).seconds > 86400:  # 24 horas
                    cache_keys_to_remove.append(cache_key)
            
            for key in cache_keys_to_remove:
                del indicator.strategy_signals[key]
            
            # Pausa general
            time.sleep(10)
            
        except Exception as e:
            print(f"Error en background_strategy_checker: {e}")
            time.sleep(60)

# Iniciar verificador de estrategias en segundo plano
try:
    strategy_thread = Thread(target=background_strategy_checker, daemon=True)
    strategy_thread.start()
    print("Background strategy checker iniciado correctamente para 5 criptomonedas")
except Exception as e:
    print(f"Error iniciando background strategy checker: {e}")

# Las siguientes funciones se mantienen exactamente igual que en tu código original
# Solo se muestran las firmas por brevedad
def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'entry':
            if alert_data.get('strategy') == 'VOL+EMA21':
                # Mensaje para estrategia VOL+EMA21
                message = f"""
🚨 VOL+EMA21 | {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Entrada: ${alert_data['close_price']:.6f} | Vol: {alert_data['volume_ratio']:.1f}x
Filtros: FTMaverick OK | MF: {alert_data['mayor_trend']}/{alert_data['menor_trend']}
"""
                
                # Enviar imagen
                if 'chart' in alert_data and alert_data['chart']:
                    asyncio.run(bot.send_photo(
                        chat_id=TELEGRAM_CHAT_ID,
                        photo=alert_data['chart'],
                        caption=message
                    ))
                else:
                    asyncio.run(bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=message
                    ))
                    
            else:
                # Mensaje para estrategia MULTIFRAME
                message = f"""
🚨 {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Score: {alert_data['score']:.1f}%

Precio: ${alert_data['current_price']:.6f}
Entrada: ${alert_data['entry']:.6f}
MA200: {alert_data['ma200_condition'].upper()}

Condiciones cumplidas:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])])}
"""
                
                # Generar gráfico para MULTIFRAME
                signal_data = indicator.generate_signals_improved(
                    alert_data['symbol'], 
                    alert_data['interval']
                )
                
                if signal_data and signal_data['signal'] != 'NEUTRAL':
                    chart_buffer = indicator.generate_multiframe_chart(
                        alert_data['symbol'],
                        alert_data['interval'],
                        signal_data
                    )
                    
                    if chart_buffer:
                        asyncio.run(bot.send_photo(
                            chat_id=TELEGRAM_CHAT_ID,
                            photo=chart_buffer,
                            caption=message
                        ))
                    else:
                        asyncio.run(bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=message
                        ))
                else:
                    asyncio.run(bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=message
                    ))
            
            print(f"Alerta enviada a Telegram: {alert_data['symbol']} {alert_data['interval']} {alert_data['signal']}")
            
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    intraday_intervals = ['15m', '30m', '1h', '2h']
    swing_intervals = ['4h', '8h', '12h', '1D', '1W']
    
    intraday_last_check = datetime.now()
    swing_last_check = datetime.now()
    volume_ema_last_check = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar señales MULTIFRAME intradía
            if (current_time - intraday_last_check).seconds >= 60:
                print("Verificando alertas MULTIFRAME intradía...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in intraday_intervals:
                        send_telegram_alert(alert, 'entry')
                
                intraday_last_check = current_time
            
            # Verificar señales MULTIFRAME swing
            if (current_time - swing_last_check).seconds >= 300:
                print("Verificando alertas MULTIFRAME swing...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in swing_intervals:
                        send_telegram_alert(alert, 'entry')
                
                swing_last_check = current_time
            
            # Verificar señales VOL+EMA21
            if (current_time - volume_ema_last_check).seconds >= 300:
                print("Verificando alertas VOL+EMA21...")
                
                for symbol in TOP10_LOW_RISK:
                    for interval in ['15m', '30m', '1h', '4h', '12h', '1D']:
                        try:
                            signal = indicator.check_volume_ema_ftm_signal(symbol, interval)
                            if signal:
                                # Verificar si ya enviamos esta señal recientemente
                                signal_key = f"{symbol}_{interval}_{signal['signal']}"
                                if signal_key not in indicator.volume_ema_signals:
                                    send_telegram_alert(signal, 'entry')
                                    indicator.volume_ema_signals[signal_key] = current_time
                                else:
                                    # Eliminar señales antiguas (más de 1 hora)
                                    last_sent = indicator.volume_ema_signals[signal_key]
                                    if (current_time - last_sent).seconds > 3600:
                                        send_telegram_alert(signal, 'entry')
                                        indicator.volume_ema_signals[signal_key] = current_time
                            
                            time.sleep(0.5)  # Pausa para no sobrecargar
                            
                        except Exception as e:
                            print(f"Error verificando VOL+EMA21 para {symbol} {interval}: {e}")
                            continue
                
                volume_ema_last_check = current_time
            
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
            
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit']):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
            
            # Soportes y resistencias
            if 'support_levels' in signal_data:
                for level in signal_data['support_levels'][:3]:
                    ax1.axhline(y=level, color='orange', linestyle=':', alpha=0.5)
            
            if 'resistance_levels' in signal_data:
                for level in signal_data['resistance_levels'][:3]:
                    ax1.axhline(y=level, color='purple', linestyle=':', alpha=0.5)
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
        
        # Gráfico 2: Ballenas
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates_matplotlib[-len(signal_data['indicators']['whale_pump']):]
            ax2.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax2.bar(whale_dates, [-x for x in signal_data['indicators']['whale_dump']], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
        ax2.set_ylabel('Fuerza Ballenas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: ADX/DMI
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates_matplotlib[-len(signal_data['indicators']['adx']):]
            ax3.plot(adx_dates, signal_data['indicators']['adx'], 
                    'black', linewidth=2, label='ADX')
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
            rsi_dates = dates_matplotlib[-len(signal_data['indicators']['rsi_traditional']):]
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
            rsi_maverick_dates = dates_matplotlib[-len(signal_data['indicators']['rsi_maverick']):]
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
            macd_dates = dates_matplotlib[-len(signal_data['indicators']['macd']):]
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
            volume_dates = dates_matplotlib[-len(signal_data['indicators']['volume_ratio']):]
            
            # Colores según señal de volumen
            colors = []
            volume_signal = signal_data['indicators'].get('volume_signal', ['NEUTRAL'] * 50)
            for i, signal in enumerate(volume_signal[-50:]):
                if signal == 'COMPRA':
                    colors.append('green')
                elif signal == 'VENTA':
                    colors.append('red')
                else:
                    colors.append('gray')
            
            # Volumen
            volumes = [d['volume'] for d in signal_data['data'][-50:]]
            ax7.bar(volume_dates, volumes, color=colors, alpha=0.6, label='Volumen')
            
            # MA de volumen
            ax7.plot(volume_dates, signal_data['indicators']['volume_ma'][-50:], 
                    'yellow', linewidth=1, label='MA Volumen')
        
        ax7.set_ylabel('Volumen')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: Fuerza de Tendencia Maverick
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates_matplotlib[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength'][-50:]
            colors = signal_data['indicators']['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax8.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax8.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax8.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = signal_data['indicators']['no_trade_zones'][-50:]
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
        ma200_info = f"MA200: {signal_data.get('ma200_condition', 'below').upper()}"
        
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

