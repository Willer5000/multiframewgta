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

# Configuración optimizada - 5 criptomonedas principales
CRYPTO_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "ADA-USDT", "SOL-USDT", "XRP-USDT"
]

# Clasificación de riesgo para las 5 criptomonedas
CRYPTO_RISK_CLASSIFICATION = {
    "bajo": ["BTC-USDT", "ETH-USDT", "ADA-USDT", "SOL-USDT", "XRP-USDT"]
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

# Configuración de estrategias y temporalidades permitidas
STRATEGY_CONFIG = {
    'TREND_RIDER': {
        'intervals': ['4h', '8h', '12h', '1D'],
        'indicators': ['MA200', 'MA50', 'MACD', 'FTMaverick']
    },
    'MOMENTUM_DIVERGENCE': {
        'intervals': ['1h', '2h', '4h'],
        'indicators': ['RSI_Trad', 'RSI_Mav', 'Volume_Clusters', 'FTMaverick']
    },
    'BOLLINGER_SQUEEZE': {
        'intervals': ['15m', '30m', '1h'],
        'indicators': ['Bollinger', 'Volume', 'ADX', 'FTMaverick']
    },
    'ADX_POWER_TREND': {
        'intervals': ['2h', '4h', '8h'],
        'indicators': ['ADX', 'DMI', 'MA21', 'FTMaverick']
    },
    'MACD_HISTOGRAM_REVERSAL': {
        'intervals': ['30m', '1h', '2h'],
        'indicators': ['MACD_Hist', 'RSI_Mav', 'MA_Crossover', 'FTMaverick']
    },
    'VOLUME_SPIKE_MOMENTUM': {
        'intervals': ['15m', '30m', '1h'],
        'indicators': ['Volume_Clusters', 'MA21', 'RSI_Mav', 'FTMaverick']
    },
    'DOUBLE_CONFIRMATION_RSI': {
        'intervals': ['1h', '2h', '4h'],
        'indicators': ['RSI_Trad', 'RSI_Mav', 'Bollinger', 'FTMaverick']
    },
    'TREND_STRENGTH_MAVERICK': {
        'intervals': ['4h', '8h', '12h'],
        'indicators': ['FTMaverick', 'MA50', 'Volume']
    },
    'WHALE_FOLLOWING': {
        'intervals': ['12h', '1D'],
        'indicators': ['Whale_Signals', 'MA200', 'ADX', 'FTMaverick']
    },
    'MA_CONVERGENCE_DIVERGENCE': {
        'intervals': ['2h', '4h', '8h'],
        'indicators': ['MA_Alignment', 'MACD', 'FTMaverick']
    },
    'RSI_MAVERICK_EXTREME': {
        'intervals': ['30m', '1h', '2h'],
        'indicators': ['RSI_Mav_Extreme', 'Bollinger', 'FTMaverick']
    },
    'VOLUME_PRICE_DIVERGENCE': {
        'intervals': ['1h', '2h', '4h'],
        'indicators': ['Volume_Price_Div', 'RSI_Mav', 'FTMaverick']
    },
    'VOLUME_EMA_STRATEGY': {
        'intervals': ['15m', '30m', '1h', '4h', '12h', '1D'],
        'indicators': ['Volume_EMA', 'MA21', 'FTMaverick', 'Multi_TF']
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
        self.strategy_alerts_sent = {}
        self.check_intervals = {
            '15m': {'start_percent': 50, 'check_seconds': 60},
            '30m': {'start_percent': 50, 'check_seconds': 120},
            '1h': {'start_percent': 50, 'check_seconds': 300},
            '2h': {'start_percent': 50, 'check_seconds': 420},
            '4h': {'start_percent': 25, 'check_seconds': 420},
            '8h': {'start_percent': 25, 'check_seconds': 600},
            '12h': {'start_percent': 25, 'check_seconds': 900},
            '1D': {'start_percent': 25, 'check_seconds': 3600},
            '1W': {'start_percent': 10, 'check_seconds': 10000}
        }
        
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def is_scalping_time(self):
        """Verificar si es horario de scalping"""
        now = self.get_bolivia_time()
        if now.weekday() >= 5:
            return False
        return 4 <= now.hour < 16

    def should_check_interval(self, interval, current_time):
        """Verificar si se debe revisar este intervalo según tiempo de vela"""
        if interval not in self.check_intervals:
            return False
            
        config = self.check_intervals[interval]
        
        # Calcular minutos del intervalo
        interval_minutes = self.interval_to_minutes(interval)
        if not interval_minutes:
            return False
            
        current_minute = current_time.minute
        total_seconds = interval_minutes * 60
        elapsed_seconds = (current_minute % interval_minutes) * 60 + current_time.second
        
        elapsed_percentage = (elapsed_seconds / total_seconds) * 100
        
        return elapsed_percentage >= config['start_percent']

    def interval_to_minutes(self, interval):
        """Convertir intervalo a minutos"""
        interval_map = {
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '8h': 480,
            '12h': 720,
            '1D': 1440,
            '1W': 10080
        }
        return interval_map.get(interval, 0)

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
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            r1 = 2 * pivot - low[-1]
            s1 = 2 * pivot - high[-1]
            r2 = pivot + (high[-1] - low[-1])
            s2 = pivot - (high[-1] - low[-1])
            
            recent_highs = sorted(high[-50:])[-3:]
            recent_lows = sorted(low[-50:])[:3]
            
            support_levels = list(recent_lows) + [s1, s2]
            resistance_levels = list(recent_highs) + [r1, r2]
            
            support_levels = sorted(list(set([round(s, 6) for s in support_levels])))[:num_levels]
            resistance_levels = sorted(list(set([round(r, 6) for r in resistance_levels])), reverse=True)[:num_levels]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            print(f"Error calculando soportes/resistencias: {e}")
            current_price = close[-1]
            support_levels = [current_price * 0.95, current_price * 0.90]
            resistance_levels = [current_price * 1.05, current_price * 1.10]
            return support_levels, resistance_levels

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15, support_levels=None, resistance_levels=None):
        """Calcular entradas y salidas óptimas"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            if support_levels is None or resistance_levels is None:
                support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            if signal_type == 'LONG':
                valid_supports = [s for s in support_levels if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)
                else:
                    entry = current_price * 0.995
                
                if len(support_levels) > 1:
                    stop_loss = support_levels[1] if len(support_levels) > 1 else entry - (current_atr * 2)
                else:
                    stop_loss = entry - (current_atr * 2)
                
                take_profits = []
                for resistance in resistance_levels[:3]:
                    if resistance > entry:
                        take_profits.append(resistance)
                
                if not take_profits:
                    take_profits = [entry + (2 * (entry - stop_loss))]
            
            else:
                valid_resistances = [r for r in resistance_levels if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)
                else:
                    entry = current_price * 1.005
                
                if len(resistance_levels) > 1:
                    stop_loss = resistance_levels[1] if len(resistance_levels) > 1 else entry + (current_atr * 2)
                else:
                    stop_loss = entry + (current_atr * 2)
                
                take_profits = []
                for support in support_levels[:3]:
                    if support < entry:
                        take_profits.append(support)
                
                if not take_profits:
                    take_profits = [entry - (2 * (stop_loss - entry))]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits[:3]],
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
        """Verificar condiciones multi-timeframe obligatorias"""
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
                if i + 1 < n:
                    di_cross_bullish[i+1] = True
            
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
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
                        for j in range(1, min(8, n-i)):
                            patterns['double_top'][i+j] = True
                            patterns['pattern_name'][i+j] = 'Doble techo'
            
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
                        for j in range(1, min(8, n-i)):
                            patterns['double_bottom'][i+j] = True
                            patterns['pattern_name'][i+j] = 'Doble piso'
            
            if len(window_high) >= 20:
                max_idx = np.argmax(window_high)
                if (max_idx > 5 and max_idx < len(window_high)-5 and
                    window_high[max_idx-3] < window_high[max_idx] and
                    window_high[max_idx+3] < window_high[max_idx]):
                    patterns['head_shoulders'][i] = True
                    patterns['pattern_name'][i] = 'Hombro cabeza hombro'
                    for j in range(1, min(8, n-i)):
                        patterns['head_shoulders'][i+j] = True
                        patterns['pattern_name'][i+j] = 'Hombro cabeza hombro'
        
        return patterns

    def calculate_volume_anomaly(self, volume, close, period=20, std_multiplier=2):
        """Calcular anomalías de volumen"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_signal = ['NEUTRAL'] * n
            
            for i in range(period, n):
                volume_ma = self.calculate_sma(volume[:i+1], period)
                current_volume_ma = volume_ma[i] if i < len(volume_ma) else volume[i]
                
                if current_volume_ma > 0:
                    volume_ratio[i] = volume[i] / current_volume_ma
                else:
                    volume_ratio[i] = 1
                
                if i >= period * 2:
                    window = volume[max(0, i-period*2):i+1]
                    std_volume = np.std(window) if len(window) > 1 else 0
                    
                    if volume_ratio[i] > 1 + (std_multiplier * (std_volume / current_volume_ma if current_volume_ma > 0 else 0)):
                        volume_anomaly[i] = True
                        
                        if i > 0:
                            price_change = (close[i] - close[i-1]) / close[i-1] * 100
                            if price_change > 0:
                                volume_signal[i] = 'COMPRA'
                            else:
                                volume_signal[i] = 'VENTA'
                
                if i >= 5:
                    recent_anomalies = volume_anomaly[max(0, i-4):i+1]
                    if np.sum(recent_anomalies) >= 2:
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
            if (ma_short[i] > ma_long[i] and 
                ma_short[i-1] <= ma_long[i-1]):
                ma_cross_bullish[i] = True
                if i + 1 < n:
                    ma_cross_bullish[i+1] = True
            
            if (ma_short[i] < ma_long[i] and 
                ma_short[i-1] >= ma_long[i-1]):
                ma_cross_bearish[i] = True
                if i + 1 < n:
                    ma_cross_bearish[i+1] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()

    def check_macd_crossover(self, macd, signal):
        """Detectar cruce de MACD"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if (macd[i] > signal[i] and 
                macd[i-1] <= signal[i-1]):
                macd_cross_bullish[i] = True
                if i + 1 < n:
                    macd_cross_bullish[i+1] = True
            
            if (macd[i] < signal[i] and 
                macd[i-1] >= signal[i-1]):
                macd_cross_bearish[i] = True
                if i + 1 < n:
                    macd_cross_bearish[i+1] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()

    def check_adx_slope(self, adx, period=3):
        """Verificar pendiente positiva del ADX"""
        n = len(adx)
        adx_slope_positive = np.zeros(n, dtype=bool)
        
        for i in range(period, n):
            if adx[i] > 25:
                slope = (adx[i] - adx[i-period]) / period
                if slope > 0:
                    adx_slope_positive[i] = True
        
        return adx_slope_positive.tolist()

    # =================== ESTRATEGIAS DE TRADING ===================

    def check_trend_rider_strategy(self, symbol, interval):
        """Estrategia Trend Rider: MA200 + MA50 + MACD + FTMaverick"""
        if interval not in STRATEGY_CONFIG['TREND_RIDER']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 100:
                return None
            
            close = df['close'].values
            
            # Calcular indicadores
            ma200 = self.calculate_sma(close, 200)
            ma50 = self.calculate_sma(close, 50)
            macd, macd_signal, macd_hist = self.calculate_macd(close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            current_price = close[current_idx]
            current_ma200 = ma200[current_idx] if len(ma200) > 0 else 0
            current_ma50 = ma50[current_idx] if len(ma50) > 0 else 0
            
            # Verificar LONG
            if (current_price > current_ma200 and 
                current_price > current_ma50 and
                macd[current_idx] > macd_signal[current_idx] and
                macd[current_idx-1] <= macd_signal[current_idx-1] and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                # Verificar multi-timeframe
                if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                    signal_type = 'LONG'
            
            # Verificar SHORT
            elif (current_price < current_ma200 and 
                  current_price < current_ma50 and
                  macd[current_idx] < macd_signal[current_idx] and
                  macd[current_idx-1] >= macd_signal[current_idx-1] and
                  ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                    signal_type = 'SHORT'
            else:
                return None
            
            # Generar niveles
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            
            # Generar gráfico
            chart = self.generate_trend_rider_chart(symbol, interval, df, ma200, ma50, macd, ftm_data, signal_type)
            
            # Obtener información de temporalidades
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'TREND_RIDER',
                'current_price': current_price,
                'entry': levels['entry'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'support_levels': levels['support_levels'][:2],
                'resistance_levels': levels['resistance_levels'][:2],
                'chart': chart,
                'filters': {
                    'ma200_condition': 'above' if current_price > current_ma200 else 'below',
                    'ma50_condition': 'above' if current_price > current_ma50 else 'below',
                    'macd_cross': True,
                    'ftm_signal': ftm_data['strength_signals'][current_idx],
                    'multi_timeframe': tf_analysis
                },
                'recommendation': 'Swing Trading' if interval in ['4h', '8h', '12h'] else 'Intraday'
            }
            
        except Exception as e:
            print(f"Error en check_trend_rider_strategy para {symbol}: {e}")
            return None

    def generate_trend_rider_chart(self, symbol, interval, df, ma200, ma50, macd, ftm_data, signal_type):
        """Generar gráfico para Trend Rider"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1])
            
            # Gráfico de velas con MAs
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            ma200_sub = ma200[-50:]
            ma50_sub = ma50[-50:]
            
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma200_sub, label='MA200', color='purple', linewidth=2)
            ax1.plot(dates_matplotlib, ma50_sub, label='MA50', color='orange', linewidth=2)
            
            ax1.set_title(f'{symbol} - {interval} - Trend Rider ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            
            # Gráfico MACD
            macd_sub = macd[-50:]
            macd_signal = self.calculate_ema(macd_sub, 9)
            macd_hist = macd_sub - macd_signal
            
            ax2.plot(dates_matplotlib, macd_sub, label='MACD', color='blue', linewidth=1)
            ax2.plot(dates_matplotlib, macd_signal, label='Señal', color='red', linewidth=1)
            
            colors_hist = ['green' if x > 0 else 'red' for x in macd_hist]
            ax2.bar(dates_matplotlib, macd_hist, color=colors_hist, alpha=0.6, label='Histograma')
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            threshold = ftm_data['high_zone_threshold']
            ax3.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label='Umbral')
            ax3.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            # Marcar zonas no operar
            no_trade_zones = ftm_data['no_trade_zones'][-50:]
            for i, date in enumerate(dates_matplotlib):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax3.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax3.set_ylabel('FT Maverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Trend Rider: {e}")
            return None

    def check_momentum_divergence_strategy(self, symbol, interval):
        """Estrategia Momentum Divergence"""
        if interval not in STRATEGY_CONFIG['MOMENTUM_DIVERGENCE']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular indicadores
            rsi_trad = self.calculate_rsi(close)
            rsi_mav = self.calculate_rsi_maverick(close)
            volume_data = self.calculate_volume_anomaly(volume, close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Detectar divergencias
            rsi_bullish_div, rsi_bearish_div = self.detect_divergence_traditional(close, rsi_trad)
            rsi_mav_bullish, rsi_mav_bearish = self.detect_divergence(close, rsi_mav)
            
            signal_type = None
            
            # Verificar LONG: divergencia alcista con volumen
            if (rsi_bullish_div[current_idx] and 
                rsi_mav_bullish[current_idx] and
                volume_data['volume_clusters'][current_idx] and
                volume_data['volume_signal'][current_idx] == 'COMPRA' and
                not ftm_data['no_trade_zones'][current_idx] and
                rsi_trad[current_idx] < 40):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                    signal_type = 'LONG'
            
            # Verificar SHORT: divergencia bajista con volumen
            elif (rsi_bearish_div[current_idx] and 
                  rsi_mav_bearish[current_idx] and
                  volume_data['volume_clusters'][current_idx] and
                  volume_data['volume_signal'][current_idx] == 'VENTA' and
                  not ftm_data['no_trade_zones'][current_idx] and
                  rsi_trad[current_idx] > 60):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                    signal_type = 'SHORT'
            else:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_momentum_divergence_chart(symbol, interval, df, rsi_trad, rsi_mav, volume_data, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'MOMENTUM_DIVERGENCE',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'rsi_trad': rsi_trad[current_idx],
                    'rsi_mav': rsi_mav[current_idx],
                    'volume_cluster': True,
                    'volume_signal': volume_data['volume_signal'][current_idx],
                    'ftm_no_trade': not ftm_data['no_trade_zones'][current_idx]
                },
                'recommendation': 'Intraday'
            }
            
        except Exception as e:
            print(f"Error en check_momentum_divergence_strategy para {symbol}: {e}")
            return None

    def generate_momentum_divergence_chart(self, symbol, interval, df, rsi_trad, rsi_mav, volume_data, ftm_data, signal_type):
        """Generar gráfico para Momentum Divergence"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), height_ratios=[2, 1, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Precio
            closes = df['close'].iloc[-50:].values
            ax1.plot(dates_matplotlib, closes, label='Precio', color='black', linewidth=2)
            ax1.set_ylabel('Precio')
            ax1.set_title(f'{symbol} - {interval} - Momentum Divergence ({signal_type})')
            ax1.grid(True, alpha=0.3)
            
            # RSI Tradicional
            rsi_trad_sub = rsi_trad[-50:]
            ax2.plot(dates_matplotlib, rsi_trad_sub, label='RSI Trad', color='blue', linewidth=2)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI Trad')
            ax2.grid(True, alpha=0.3)
            
            # RSI Maverick
            rsi_mav_sub = rsi_mav[-50:]
            ax3.plot(dates_matplotlib, rsi_mav_sub, label='RSI Mav', color='purple', linewidth=2)
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
            ax3.set_ylabel('RSI Mav')
            ax3.grid(True, alpha=0.3)
            
            # Volumen
            volume_sub = df['volume'].iloc[-50:].values
            colors = ['green' if volume_data['volume_signal'][-50:][i] == 'COMPRA' else 
                     'red' if volume_data['volume_signal'][-50:][i] == 'VENTA' else 
                     'gray' for i in range(len(volume_sub))]
            
            ax4.bar(dates_matplotlib, volume_sub, color=colors, alpha=0.6)
            ax4.set_ylabel('Volumen')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Momentum Divergence: {e}")
            return None

    def check_bollinger_squeeze_strategy(self, symbol, interval):
        """Estrategia Bollinger Squeeze Breakout"""
        if interval not in STRATEGY_CONFIG['BOLLINGER_SQUEEZE']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calcular indicadores
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            volume_data = self.calculate_volume_anomaly(volume, close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Calcular squeeze (bandas estrechas)
            bb_width = [(bb_upper[i] - bb_lower[i]) / bb_middle[i] * 100 for i in range(len(close))]
            
            signal_type = None
            
            # Verificar squeeze previo (últimas 10 velas bandas estrechas)
            recent_bb_width = bb_width[-10:]
            avg_bb_width = np.mean(recent_bb_width)
            std_bb_width = np.std(recent_bb_width)
            
            # Squeeze definido como bandas < 2% del precio
            squeeze_condition = avg_bb_width < 2.0
            
            if squeeze_condition:
                # Verificar breakout con volumen
                volume_ratio = volume_data['volume_ratio'][current_idx]
                current_volume = volume[current_idx]
                avg_volume = np.mean(volume[-20:])
                
                # LONG: breakout alcista
                if (close[current_idx] > bb_upper[current_idx] and
                    current_volume > avg_volume * 2.0 and
                    adx[current_idx] > 25 and
                    plus_di[current_idx] > minus_di[current_idx] and
                    not ftm_data['no_trade_zones'][current_idx]):
                    
                    if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                        signal_type = 'LONG'
                
                # SHORT: breakout bajista
                elif (close[current_idx] < bb_lower[current_idx] and
                      current_volume > avg_volume * 2.0 and
                      adx[current_idx] > 25 and
                      minus_di[current_idx] > plus_di[current_idx] and
                      not ftm_data['no_trade_zones'][current_idx]):
                    
                    if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                        signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_bollinger_squeeze_chart(symbol, interval, df, bb_upper, bb_middle, bb_lower, adx, plus_di, minus_di, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'BOLLINGER_SQUEEZE',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'squeeze_detected': squeeze_condition,
                    'volume_ratio': volume_ratio,
                    'adx': adx[current_idx],
                    'dmi_cross': plus_di[current_idx] > minus_di[current_idx] if signal_type == 'LONG' else minus_di[current_idx] > plus_di[current_idx],
                    'ftm_no_trade': not ftm_data['no_trade_zones'][current_idx]
                },
                'recommendation': 'Scalping/Intraday'
            }
            
        except Exception as e:
            print(f"Error en check_bollinger_squeeze_strategy para {symbol}: {e}")
            return None

    def generate_bollinger_squeeze_chart(self, symbol, interval, df, bb_upper, bb_middle, bb_lower, adx, plus_di, minus_di, ftm_data, signal_type):
        """Generar gráfico para Bollinger Squeeze"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de velas con Bollinger Bands
            closes = df['close'].iloc[-50:].values
            bb_upper_sub = bb_upper[-50:]
            bb_middle_sub = bb_middle[-50:]
            bb_lower_sub = bb_lower[-50:]
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, bb_upper_sub, label='BB Superior', color='red', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_middle_sub, label='BB Media', color='blue', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_lower_sub, label='BB Inferior', color='green', alpha=0.5, linewidth=1)
            
            ax1.fill_between(dates_matplotlib, bb_lower_sub, bb_upper_sub, color='gray', alpha=0.1)
            ax1.set_title(f'{symbol} - {interval} - Bollinger Squeeze ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ADX y DMI
            adx_sub = adx[-50:]
            plus_di_sub = plus_di[-50:]
            minus_di_sub = minus_di[-50:]
            
            ax2.plot(dates_matplotlib, adx_sub, label='ADX', color='black', linewidth=2)
            ax2.plot(dates_matplotlib, plus_di_sub, label='+DI', color='green', linewidth=1)
            ax2.plot(dates_matplotlib, minus_di_sub, label='-DI', color='red', linewidth=1)
            ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7)
            ax2.set_ylabel('ADX/DMI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            threshold = ftm_data['high_zone_threshold']
            ax3.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7)
            ax3.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = ftm_data['no_trade_zones'][-50:]
            for i, date in enumerate(dates_matplotlib):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax3.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax3.set_ylabel('FT Maverick')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Bollinger Squeeze: {e}")
            return None

    def check_adx_power_trend_strategy(self, symbol, interval):
        """Estrategia ADX Power Trend"""
        if interval not in STRATEGY_CONFIG['ADX_POWER_TREND']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calcular indicadores
            ma21 = self.calculate_sma(close, 21)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            signal_type = None
            
            # Verificar LONG: ADX fuerte, +DI > -DI, precio > MA21
            if (adx[current_idx] > 30 and
                plus_di[current_idx] > minus_di[current_idx] and
                plus_di[current_idx] - minus_di[current_idx] > 5 and
                close[current_idx] > ma21[current_idx] and
                di_cross_bullish[current_idx] and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                    signal_type = 'LONG'
            
            # Verificar SHORT: ADX fuerte, -DI > +DI, precio < MA21
            elif (adx[current_idx] > 30 and
                  minus_di[current_idx] > plus_di[current_idx] and
                  minus_di[current_idx] - plus_di[current_idx] > 5 and
                  close[current_idx] < ma21[current_idx] and
                  di_cross_bearish[current_idx] and
                  ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                    signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_adx_power_trend_chart(symbol, interval, df, ma21, adx, plus_di, minus_di, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'ADX_POWER_TREND',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'adx': adx[current_idx],
                    'dmi_diff': abs(plus_di[current_idx] - minus_di[current_idx]),
                    'price_vs_ma21': 'above' if close[current_idx] > ma21[current_idx] else 'below',
                    'dmi_cross': True,
                    'ftm_signal': ftm_data['strength_signals'][current_idx]
                },
                'recommendation': 'Swing Trading'
            }
            
        except Exception as e:
            print(f"Error en check_adx_power_trend_strategy para {symbol}: {e}")
            return None

    def generate_adx_power_trend_chart(self, symbol, interval, df, ma21, adx, plus_di, minus_di, ftm_data, signal_type):
        """Generar gráfico para ADX Power Trend"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de velas con MA21
            closes = df['close'].iloc[-50:].values
            ma21_sub = ma21[-50:]
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma21_sub, label='MA21', color='orange', linewidth=2)
            ax1.set_title(f'{symbol} - {interval} - ADX Power Trend ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ADX y DMI
            adx_sub = adx[-50:]
            plus_di_sub = plus_di[-50:]
            minus_di_sub = minus_di[-50:]
            
            ax2.plot(dates_matplotlib, adx_sub, label='ADX', color='black', linewidth=2)
            ax2.plot(dates_matplotlib, plus_di_sub, label='+DI', color='green', linewidth=1)
            ax2.plot(dates_matplotlib, minus_di_sub, label='-DI', color='red', linewidth=1)
            ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7)
            ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.7)
            ax2.set_ylabel('ADX/DMI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            threshold = ftm_data['high_zone_threshold']
            ax3.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7)
            ax3.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            ax3.set_ylabel('FT Maverick')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico ADX Power Trend: {e}")
            return None

    def check_macd_histogram_reversal_strategy(self, symbol, interval):
        """Estrategia MACD Histogram Reversal"""
        if interval not in STRATEGY_CONFIG['MACD_HISTOGRAM_REVERSAL']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # Calcular indicadores
            macd, macd_signal, macd_hist = self.calculate_macd(close)
            rsi_mav = self.calculate_rsi_maverick(close)
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover(ma9, ma21)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            signal_type = None
            
            # Verificar reversión LONG: histograma cambia de negativo a positivo
            if (macd_hist[current_idx] > 0 and
                macd_hist[current_idx-1] < 0 and
                rsi_mav[current_idx] > 0.3 and rsi_mav[current_idx] < 0.7 and
                ma_cross_bullish[current_idx] and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                    signal_type = 'LONG'
            
            # Verificar reversión SHORT: histograma cambia de positivo a negativo
            elif (macd_hist[current_idx] < 0 and
                  macd_hist[current_idx-1] > 0 and
                  rsi_mav[current_idx] > 0.3 and rsi_mav[current_idx] < 0.7 and
                  ma_cross_bearish[current_idx] and
                  ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                    signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_macd_histogram_chart(symbol, interval, df, macd, macd_hist, rsi_mav, ma9, ma21, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'MACD_HISTOGRAM_REVERSAL',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'macd_hist_reversal': True,
                    'rsi_mav': rsi_mav[current_idx],
                    'ma_crossover': True,
                    'ftm_signal': ftm_data['strength_signals'][current_idx]
                },
                'recommendation': 'Intraday'
            }
            
        except Exception as e:
            print(f"Error en check_macd_histogram_reversal_strategy para {symbol}: {e}")
            return None

    def generate_macd_histogram_chart(self, symbol, interval, df, macd, macd_hist, rsi_mav, ma9, ma21, ftm_data, signal_type):
        """Generar gráfico para MACD Histogram Reversal"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), height_ratios=[2, 1, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Precio con MAs
            closes = df['close'].iloc[-50:].values
            ma9_sub = ma9[-50:]
            ma21_sub = ma21[-50:]
            
            ax1.plot(dates_matplotlib, closes, label='Precio', color='black', linewidth=2)
            ax1.plot(dates_matplotlib, ma9_sub, label='MA9', color='red', linewidth=1, alpha=0.7)
            ax1.plot(dates_matplotlib, ma21_sub, label='MA21', color='blue', linewidth=1, alpha=0.7)
            ax1.set_title(f'{symbol} - {interval} - MACD Histogram Reversal ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # MACD Histogram
            macd_hist_sub = macd_hist[-50:]
            colors_hist = ['green' if x > 0 else 'red' for x in macd_hist_sub]
            ax2.bar(dates_matplotlib, macd_hist_sub, color=colors_hist, alpha=0.6)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('MACD Histogram')
            ax2.grid(True, alpha=0.3)
            
            # RSI Maverick
            rsi_mav_sub = rsi_mav[-50:]
            ax3.plot(dates_matplotlib, rsi_mav_sub, label='RSI Maverick', color='purple', linewidth=2)
            ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('RSI Mav')
            ax3.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('FT Maverick')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico MACD Histogram: {e}")
            return None

    def check_volume_spike_momentum_strategy(self, symbol, interval):
        """Estrategia Volume Spike Momentum"""
        if interval not in STRATEGY_CONFIG['VOLUME_SPIKE_MOMENTUM']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular indicadores
            ma21 = self.calculate_sma(close, 21)
            rsi_mav = self.calculate_rsi_maverick(close)
            volume_data = self.calculate_volume_anomaly(volume, close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            signal_type = None
            
            # Verificar cluster de volumen (mínimo 2 anomalías en 5 velas)
            volume_cluster = volume_data['volume_clusters'][current_idx]
            volume_signal = volume_data['volume_signal'][current_idx]
            volume_ratio = volume_data['volume_ratio'][current_idx]
            
            if volume_cluster and volume_ratio > 2.5:
                # LONG: volumen COMPRA, precio > MA21, RSI no extremo
                if (volume_signal == 'COMPRA' and
                    close[current_idx] > ma21[current_idx] and
                    rsi_mav[current_idx] > 0.3 and rsi_mav[current_idx] < 0.7 and
                    ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                    not ftm_data['no_trade_zones'][current_idx]):
                    
                    if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                        signal_type = 'LONG'
                
                # SHORT: volumen VENTA, precio < MA21, RSI no extremo
                elif (volume_signal == 'VENTA' and
                      close[current_idx] < ma21[current_idx] and
                      rsi_mav[current_idx] > 0.3 and rsi_mav[current_idx] < 0.7 and
                      ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                      not ftm_data['no_trade_zones'][current_idx]):
                    
                    if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                        signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_volume_spike_chart(symbol, interval, df, ma21, rsi_mav, volume_data, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'VOLUME_SPIKE_MOMENTUM',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'volume_cluster': True,
                    'volume_ratio': volume_ratio,
                    'volume_signal': volume_signal,
                    'price_vs_ma21': 'above' if close[current_idx] > ma21[current_idx] else 'below',
                    'rsi_mav': rsi_mav[current_idx],
                    'ftm_no_trade': not ftm_data['no_trade_zones'][current_idx]
                },
                'recommendation': 'Scalping/Intraday'
            }
            
        except Exception as e:
            print(f"Error en check_volume_spike_momentum_strategy para {symbol}: {e}")
            return None

    def generate_volume_spike_chart(self, symbol, interval, df, ma21, rsi_mav, volume_data, ftm_data, signal_type):
        """Generar gráfico para Volume Spike Momentum"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), height_ratios=[2, 1, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Precio con MA21
            closes = df['close'].iloc[-50:].values
            ma21_sub = ma21[-50:]
            
            ax1.plot(dates_matplotlib, closes, label='Precio', color='black', linewidth=2)
            ax1.plot(dates_matplotlib, ma21_sub, label='MA21', color='orange', linewidth=2)
            ax1.set_title(f'{symbol} - {interval} - Volume Spike Momentum ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volumen con anomalías
            volume_sub = df['volume'].iloc[-50:].values
            colors = ['green' if volume_data['volume_signal'][-50:][i] == 'COMPRA' else 
                     'red' if volume_data['volume_signal'][-50:][i] == 'VENTA' else 
                     'gray' for i in range(len(volume_sub))]
            
            ax2.bar(dates_matplotlib, volume_sub, color=colors, alpha=0.6, label='Volumen')
            
            # Marcar anomalías
            anomalies = volume_data['volume_anomaly'][-50:]
            for i, anomaly in enumerate(anomalies):
                if anomaly:
                    ax2.scatter(dates_matplotlib[i], volume_sub[i], color='purple', s=30, marker='x')
            
            ax2.set_ylabel('Volumen')
            ax2.grid(True, alpha=0.3)
            
            # RSI Maverick
            rsi_mav_sub = rsi_mav[-50:]
            ax3.plot(dates_matplotlib, rsi_mav_sub, label='RSI Maverick', color='purple', linewidth=2)
            ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
            ax3.set_ylabel('RSI Mav')
            ax3.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors_ftm = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors_ftm[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('FT Maverick')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Volume Spike: {e}")
            return None

    def check_double_confirmation_rsi_strategy(self, symbol, interval):
        """Estrategia Double Confirmation RSI"""
        if interval not in STRATEGY_CONFIG['DOUBLE_CONFIRMATION_RSI']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # Calcular indicadores
            rsi_trad = self.calculate_rsi(close)
            rsi_mav = self.calculate_rsi_maverick(close)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            signal_type = None
            
            # LONG: RSI tradicional sale de sobreventa, RSI Maverick confirma
            if (rsi_trad[current_idx] > 30 and rsi_trad[current_idx-1] <= 30 and
                rsi_mav[current_idx] > 0.2 and
                close[current_idx] <= bb_lower[current_idx] * 1.02 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                    signal_type = 'LONG'
            
            # SHORT: RSI tradicional sale de sobrecompra, RSI Maverick confirma
            elif (rsi_trad[current_idx] < 70 and rsi_trad[current_idx-1] >= 70 and
                  rsi_mav[current_idx] < 0.8 and
                  close[current_idx] >= bb_upper[current_idx] * 0.98 and
                  ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                    signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_double_rsi_chart(symbol, interval, df, rsi_trad, rsi_mav, bb_upper, bb_middle, bb_lower, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'DOUBLE_CONFIRMATION_RSI',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'rsi_trad': rsi_trad[current_idx],
                    'rsi_mav': rsi_mav[current_idx],
                    'bollinger_touch': True,
                    'ftm_signal': ftm_data['strength_signals'][current_idx]
                },
                'recommendation': 'Intraday/Swing'
            }
            
        except Exception as e:
            print(f"Error en check_double_confirmation_rsi_strategy para {symbol}: {e}")
            return None

    def generate_double_rsi_chart(self, symbol, interval, df, rsi_trad, rsi_mav, bb_upper, bb_middle, bb_lower, ftm_data, signal_type):
        """Generar gráfico para Double Confirmation RSI"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), height_ratios=[2, 1, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Precio con Bollinger Bands
            closes = df['close'].iloc[-50:].values
            bb_upper_sub = bb_upper[-50:]
            bb_middle_sub = bb_middle[-50:]
            bb_lower_sub = bb_lower[-50:]
            
            ax1.plot(dates_matplotlib, closes, label='Precio', color='black', linewidth=2)
            ax1.plot(dates_matplotlib, bb_upper_sub, label='BB Sup', color='red', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_middle_sub, label='BB Med', color='blue', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_lower_sub, label='BB Inf', color='green', alpha=0.5, linewidth=1)
            ax1.fill_between(dates_matplotlib, bb_lower_sub, bb_upper_sub, color='gray', alpha=0.1)
            ax1.set_title(f'{symbol} - {interval} - Double Confirmation RSI ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # RSI Tradicional
            rsi_trad_sub = rsi_trad[-50:]
            ax2.plot(dates_matplotlib, rsi_trad_sub, label='RSI Trad', color='blue', linewidth=2)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI Trad')
            ax2.grid(True, alpha=0.3)
            
            # RSI Maverick
            rsi_mav_sub = rsi_mav[-50:]
            ax3.plot(dates_matplotlib, rsi_mav_sub, label='RSI Mav', color='purple', linewidth=2)
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('RSI Mav')
            ax3.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('FT Maverick')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Double RSI: {e}")
            return None

    def check_trend_strength_maverick_strategy(self, symbol, interval):
        """Estrategia Trend Strength Maverick"""
        if interval not in STRATEGY_CONFIG['TREND_STRENGTH_MAVERICK']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular indicadores
            ftm_data = self.calculate_trend_strength_maverick(close)
            ma50 = self.calculate_sma(close, 50)
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            current_idx = -1
            
            signal_type = None
            
            # Solo operar señales STRONG con alineación de precio
            ftm_signal = ftm_data['strength_signals'][current_idx]
            no_trade_zone = ftm_data['no_trade_zones'][current_idx]
            
            if not no_trade_zone:
                # LONG: señal STRONG_UP con precio > MA50 y volumen creciente
                if (ftm_signal == 'STRONG_UP' and
                    close[current_idx] > ma50[current_idx] and
                    volume_data['volume_ratio'][current_idx] > 1.2):
                    
                    if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                        signal_type = 'LONG'
                
                # SHORT: señal STRONG_DOWN con precio < MA50 y volumen creciente
                elif (ftm_signal == 'STRONG_DOWN' and
                      close[current_idx] < ma50[current_idx] and
                      volume_data['volume_ratio'][current_idx] > 1.2):
                    
                    if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                        signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_trend_strength_chart(symbol, interval, df, ftm_data, ma50, volume_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'TREND_STRENGTH_MAVERICK',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'ftm_signal': ftm_signal,
                    'price_vs_ma50': 'above' if close[current_idx] > ma50[current_idx] else 'below',
                    'volume_ratio': volume_data['volume_ratio'][current_idx],
                    'no_trade_zone': no_trade_zone
                },
                'recommendation': 'Swing Trading'
            }
            
        except Exception as e:
            print(f"Error en check_trend_strength_maverick_strategy para {symbol}: {e}")
            return None

    def generate_trend_strength_chart(self, symbol, interval, df, ftm_data, ma50, volume_data, signal_type):
        """Generar gráfico para Trend Strength Maverick"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Precio con MA50
            closes = df['close'].iloc[-50:].values
            ma50_sub = ma50[-50:]
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma50_sub, label='MA50', color='orange', linewidth=2)
            ax1.set_title(f'{symbol} - {interval} - Trend Strength Maverick ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # FTMaverick Strength
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax2.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            threshold = ftm_data['high_zone_threshold']
            ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label=f'Umbral ({threshold:.1f})')
            ax2.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            # Marcar zonas no operar
            no_trade_zones = ftm_data['no_trade_zones'][-50:]
            for i, date in enumerate(dates_matplotlib):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax2.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax2.set_ylabel('Fuerza Tendencia')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Volumen
            volume_sub = df['volume'].iloc[-50:].values
            ax3.bar(dates_matplotlib, volume_sub, color='gray', alpha=0.6)
            ax3.set_ylabel('Volumen')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Trend Strength: {e}")
            return None

    def check_whale_following_strategy(self, symbol, interval):
        """Estrategia Whale Following con condición DMI obligatoria"""
        if interval not in STRATEGY_CONFIG['WHALE_FOLLOWING']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calcular indicadores
            whale_data = self.calculate_whale_signals_improved(df)
            ma200 = self.calculate_sma(close, 200)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Condiciones DMI obligatorias
            dmi_cross_bullish, dmi_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            signal_type = None
            
            # LONG: ballenas comprando con confirmación DMI
            if (whale_data['confirmed_buy'][current_idx] and
                whale_data['whale_pump'][current_idx] > 20 and
                close[current_idx] > ma200[current_idx] and
                adx[current_idx] > 25 and
                dmi_cross_bullish[current_idx] and  # CONDICIÓN OBLIGATORIA
                plus_di[current_idx] > minus_di[current_idx] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                signal_type = 'LONG'
            
            # SHORT: ballenas vendiendo con confirmación DMI
            elif (whale_data['confirmed_sell'][current_idx] and
                  whale_data['whale_dump'][current_idx] > 20 and
                  close[current_idx] < ma200[current_idx] and
                  adx[current_idx] > 25 and
                  dmi_cross_bearish[current_idx] and  # CONDICIÓN OBLIGATORIA
                  minus_di[current_idx] > plus_di[current_idx] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_whale_following_chart(symbol, interval, df, whale_data, ma200, adx, plus_di, minus_di, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'WHALE_FOLLOWING',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'whale_signal': whale_data['whale_pump'][current_idx] if signal_type == 'LONG' else whale_data['whale_dump'][current_idx],
                    'confirmed_action': True,
                    'ma200_condition': 'above' if close[current_idx] > ma200[current_idx] else 'below',
                    'adx': adx[current_idx],
                    'dmi_cross': True,  # Condición obligatoria cumplida
                    'ftm_no_trade': not ftm_data['no_trade_zones'][current_idx]
                },
                'recommendation': 'Swing Trading/Spot'
            }
            
        except Exception as e:
            print(f"Error en check_whale_following_strategy para {symbol}: {e}")
            return None

    def generate_whale_following_chart(self, symbol, interval, df, whale_data, ma200, adx, plus_di, minus_di, ftm_data, signal_type):
        """Generar gráfico para Whale Following"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), height_ratios=[2, 1, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Precio con MA200
            closes = df['close'].iloc[-50:].values
            ma200_sub = ma200[-50:]
            
            ax1.plot(dates_matplotlib, closes, label='Precio', color='black', linewidth=2)
            ax1.plot(dates_matplotlib, ma200_sub, label='MA200', color='purple', linewidth=2)
            ax1.set_title(f'{symbol} - {interval} - Whale Following ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Indicador Ballenas
            whale_pump_sub = whale_data['whale_pump'][-50:]
            whale_dump_sub = whale_data['whale_dump'][-50:]
            
            ax2.bar(dates_matplotlib, whale_pump_sub, color='green', alpha=0.6, label='Compra Ballenas')
            ax2.bar(dates_matplotlib, [-x for x in whale_dump_sub], color='red', alpha=0.6, label='Venta Ballenas')
            ax2.set_ylabel('Fuerza Ballenas')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # ADX y DMI
            adx_sub = adx[-50:]
            plus_di_sub = plus_di[-50:]
            minus_di_sub = minus_di[-50:]
            
            ax3.plot(dates_matplotlib, adx_sub, label='ADX', color='black', linewidth=2)
            ax3.plot(dates_matplotlib, plus_di_sub, label='+DI', color='green', linewidth=1)
            ax3.plot(dates_matplotlib, minus_di_sub, label='-DI', color='red', linewidth=1)
            ax3.axhline(y=25, color='yellow', linestyle='--', alpha=0.7)
            ax3.set_ylabel('ADX/DMI')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.set_ylabel('FT Maverick')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Whale Following: {e}")
            return None

    def check_ma_convergence_divergence_strategy(self, symbol, interval):
        """Estrategia MA Convergence Divergence"""
        if interval not in STRATEGY_CONFIG['MA_CONVERGENCE_DIVERGENCE']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # Calcular medias móviles
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            ma50 = self.calculate_sma(close, 50)
            
            # Calcular otros indicadores
            macd, macd_signal, macd_hist = self.calculate_macd(close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            signal_type = None
            
            # Verificar alineación alcista (convergencia)
            price_ma9 = close[current_idx] > ma9[current_idx]
            ma9_ma21 = ma9[current_idx] > ma21[current_idx]
            ma21_ma50 = ma21[current_idx] > ma50[current_idx]
            
            alineacion_alcista = price_ma9 and ma9_ma21 and ma21_ma50
            
            # Verificar alineación bajista (convergencia)
            price_ma9_bear = close[current_idx] < ma9[current_idx]
            ma9_ma21_bear = ma9[current_idx] < ma21[current_idx]
            ma21_ma50_bear = ma21[current_idx] < ma50[current_idx]
            
            alineacion_bajista = price_ma9_bear and ma9_ma21_bear and ma21_ma50_bear
            
            # Verificar divergencia con MACD
            macd_cross_bullish = macd[current_idx] > macd_signal[current_idx] and macd[current_idx-1] <= macd_signal[current_idx-1]
            macd_cross_bearish = macd[current_idx] < macd_signal[current_idx] and macd[current_idx-1] >= macd_signal[current_idx-1]
            
            if (alineacion_alcista and 
                macd_cross_bullish and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                    signal_type = 'LONG'
            
            elif (alineacion_bajista and 
                  macd_cross_bearish and
                  ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                    signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_ma_convergence_chart(symbol, interval, df, ma9, ma21, ma50, macd, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'MA_CONVERGENCE_DIVERGENCE',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'ma_alignment': 'bullish' if alineacion_alcista else 'bearish',
                    'macd_cross': True,
                    'ftm_signal': ftm_data['strength_signals'][current_idx],
                    'ma_separation': abs(ma9[current_idx] - ma21[current_idx]) / close[current_idx] * 100
                },
                'recommendation': 'Swing Trading'
            }
            
        except Exception as e:
            print(f"Error en check_ma_convergence_divergence_strategy para {symbol}: {e}")
            return None

    def generate_ma_convergence_chart(self, symbol, interval, df, ma9, ma21, ma50, macd, ftm_data, signal_type):
        """Generar gráfico para MA Convergence Divergence"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de velas con MAs
            closes = df['close'].iloc[-50:].values
            ma9_sub = ma9[-50:]
            ma21_sub = ma21[-50:]
            ma50_sub = ma50[-50:]
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma9_sub, label='MA9', color='red', linewidth=1)
            ax1.plot(dates_matplotlib, ma21_sub, label='MA21', color='blue', linewidth=1)
            ax1.plot(dates_matplotlib, ma50_sub, label='MA50', color='green', linewidth=1)
            ax1.set_title(f'{symbol} - {interval} - MA Convergence ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # MACD
            macd_sub = macd[-50:]
            macd_signal = self.calculate_ema(macd_sub, 9)
            
            ax2.plot(dates_matplotlib, macd_sub, label='MACD', color='blue', linewidth=1)
            ax2.plot(dates_matplotlib, macd_signal, label='Señal', color='red', linewidth=1)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax3.set_ylabel('FT Maverick')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico MA Convergence: {e}")
            return None

    def check_rsi_maverick_extreme_strategy(self, symbol, interval):
        """Estrategia RSI Maverick Extreme"""
        if interval not in STRATEGY_CONFIG['RSI_MAVERICK_EXTREME']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # Calcular indicadores
            rsi_mav = self.calculate_rsi_maverick(close)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Verificar extremos de RSI Maverick
            rsi_extreme_bullish = rsi_mav[current_idx] < 0.15
            rsi_extreme_bearish = rsi_mav[current_idx] > 0.85
            
            # Verificar toque de Bollinger Bands
            bb_touch_lower = close[current_idx] <= bb_lower[current_idx] * 1.01
            bb_touch_upper = close[current_idx] >= bb_upper[current_idx] * 0.99
            
            signal_type = None
            
            # LONG: RSI extremo bajista con toque de banda inferior
            if (rsi_extreme_bullish and 
                bb_touch_lower and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                # Verificar al menos 3 velas en zona extrema
                recent_rsi = rsi_mav[-5:]
                if all(r < 0.2 for r in recent_rsi):
                    if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                        signal_type = 'LONG'
            
            # SHORT: RSI extremo alcista con toque de banda superior
            elif (rsi_extreme_bearish and 
                  bb_touch_upper and
                  ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                recent_rsi = rsi_mav[-5:]
                if all(r > 0.8 for r in recent_rsi):
                    if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                        signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_rsi_extreme_chart(symbol, interval, df, rsi_mav, bb_upper, bb_middle, bb_lower, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'RSI_MAVERICK_EXTREME',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'rsi_mav_extreme': rsi_mav[current_idx],
                    'bollinger_touch': True,
                    'ftm_signal': ftm_data['strength_signals'][current_idx],
                    'consecutive_extreme': 5
                },
                'recommendation': 'Intraday'
            }
            
        except Exception as e:
            print(f"Error en check_rsi_maverick_extreme_strategy para {symbol}: {e}")
            return None

    def generate_rsi_extreme_chart(self, symbol, interval, df, rsi_mav, bb_upper, bb_middle, bb_lower, ftm_data, signal_type):
        """Generar gráfico para RSI Maverick Extreme"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Precio con Bollinger Bands
            closes = df['close'].iloc[-50:].values
            bb_upper_sub = bb_upper[-50:]
            bb_middle_sub = bb_middle[-50:]
            bb_lower_sub = bb_lower[-50:]
            
            ax1.plot(dates_matplotlib, closes, label='Precio', color='black', linewidth=2)
            ax1.plot(dates_matplotlib, bb_upper_sub, label='BB Sup', color='red', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_middle_sub, label='BB Med', color='blue', alpha=0.5, linewidth=1)
            ax1.plot(dates_matplotlib, bb_lower_sub, label='BB Inf', color='green', alpha=0.5, linewidth=1)
            ax1.fill_between(dates_matplotlib, bb_lower_sub, bb_upper_sub, color='gray', alpha=0.1)
            ax1.set_title(f'{symbol} - {interval} - RSI Maverick Extreme ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # RSI Maverick con zonas extremas
            rsi_mav_sub = rsi_mav[-50:]
            ax2.plot(dates_matplotlib, rsi_mav_sub, label='RSI Maverick', color='purple', linewidth=2)
            ax2.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Extremo Alcista')
            ax2.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='Extremo Bajista')
            ax2.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax2.fill_between(dates_matplotlib, 0, 0.15, color='green', alpha=0.1)
            ax2.fill_between(dates_matplotlib, 0.85, 1, color='red', alpha=0.1)
            ax2.set_ylabel('RSI Mav')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax3.set_ylabel('FT Maverick')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico RSI Extreme: {e}")
            return None

    def check_volume_price_divergence_strategy(self, symbol, interval):
        """Estrategia Volume-Price Divergence"""
        if interval not in STRATEGY_CONFIG['VOLUME_PRICE_DIVERGENCE']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular indicadores
            rsi_mav = self.calculate_rsi_maverick(close)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            
            # Detectar divergencia precio-volumen
            lookback = 14
            if current_idx >= lookback:
                price_window = close[current_idx-lookback:current_idx+1]
                volume_window = volume[current_idx-lookback:current_idx+1]
                
                # Divergencia alcista: precio hace nuevo mínimo pero volumen decrece
                price_makes_new_low = close[current_idx] < np.min(price_window[:-1])
                volume_decreasing = volume[current_idx] < np.mean(volume_window[:-1]) * 0.8
                
                # Divergencia bajista: precio hace nuevo máximo pero volumen decrece
                price_makes_new_high = close[current_idx] > np.max(price_window[:-1])
                volume_decreasing_high = volume[current_idx] < np.mean(volume_window[:-1]) * 0.8
            
            signal_type = None
            
            # LONG: divergencia alcista precio-volumen
            if (current_idx >= lookback and
                price_makes_new_low and
                volume_decreasing and
                rsi_mav[current_idx] < 0.3 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'LONG'):
                    signal_type = 'LONG'
            
            # SHORT: divergencia bajista precio-volumen
            elif (current_idx >= lookback and
                  price_makes_new_high and
                  volume_decreasing_high and
                  rsi_mav[current_idx] > 0.7 and
                  ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                if self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT'):
                    signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_volume_price_div_chart(symbol, interval, df, close, volume, rsi_mav, ftm_data, signal_type)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'VOLUME_PRICE_DIVERGENCE',
                'current_price': close[current_idx],
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'price_volume_divergence': True,
                    'rsi_mav': rsi_mav[current_idx],
                    'ftm_signal': ftm_data['strength_signals'][current_idx],
                    'lookback_period': lookback
                },
                'recommendation': 'Intraday/Swing'
            }
            
        except Exception as e:
            print(f"Error en check_volume_price_divergence_strategy para {symbol}: {e}")
            return None

    def generate_volume_price_div_chart(self, symbol, interval, df, close, volume, rsi_mav, ftm_data, signal_type):
        """Generar gráfico para Volume-Price Divergence"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Precio
            closes = close[-50:]
            ax1.plot(dates_matplotlib, closes, label='Precio', color='black', linewidth=2)
            ax1.set_ylabel('Precio')
            ax1.set_title(f'{symbol} - {interval} - Volume-Price Divergence ({signal_type})')
            ax1.grid(True, alpha=0.3)
            
            # Volumen
            volume_sub = volume[-50:]
            ax2.bar(dates_matplotlib, volume_sub, color='gray', alpha=0.6)
            ax2.set_ylabel('Volumen')
            ax2.grid(True, alpha=0.3)
            
            # RSI Maverick
            rsi_mav_sub = rsi_mav[-50:]
            ax3.plot(dates_matplotlib, rsi_mav_sub, label='RSI Maverick', color='purple', linewidth=2)
            ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
            ax3.set_ylabel('RSI Mav')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Volume-Price Divergence: {e}")
            return None

    def check_volume_ema_strategy_corrected(self, symbol, interval):
        """Estrategia Desplome de Volumen Corregida"""
        if interval not in STRATEGY_CONFIG['VOLUME_EMA_STRATEGY']['intervals']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular indicadores corregidos
            ema21 = self.calculate_ema(close, 21)
            volume_ma21 = self.calculate_sma(volume, 21)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            current_close = close[current_idx]
            current_volume = volume[current_idx]
            current_volume_ma = volume_ma21[current_idx]
            current_ema21 = ema21[current_idx]
            
            # Condiciones corregidas
            volume_condition = current_volume > (current_volume_ma * 3.0)  # Aumentado a 3x
            if not volume_condition:
                return None
            
            signal_type = None
            
            # Determinar dirección con filtro FTMaverick
            if (current_close > current_ema21 and
                ftm_data['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not ftm_data['no_trade_zones'][current_idx]):
                
                signal_type = 'LONG'
            elif (current_close < current_ema21 and
                  ftm_data['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  not ftm_data['no_trade_zones'][current_idx]):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Filtro Multi-Timeframe adicional para intervalos cortos
            if interval in ['15m', '30m', '1h', '4h']:
                if not self.check_multi_timeframe_obligatory(symbol, interval, signal_type):
                    return None
            
            levels = self.calculate_optimal_entry_exit(df, signal_type)
            chart = self.generate_volume_ema_chart_corrected(symbol, interval, df, ema21, volume_ma21, ftm_data, signal_type)
            
            # Obtener información de multi-timeframe
            if interval in ['15m', '30m', '1h', '4h']:
                tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
                mayor_trend = tf_analysis.get('mayor', 'NEUTRAL')
                menor_trend = "ALCISTA" if signal_type == 'LONG' else "BAJISTA"
            else:
                mayor_trend = "N/A"
                menor_trend = "N/A"
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'strategy': 'VOLUME_EMA_STRATEGY',
                'current_price': current_close,
                'entry': levels['entry'],
                'chart': chart,
                'filters': {
                    'volume_ratio': current_volume / current_volume_ma,
                    'price_vs_ema21': 'above' if current_close > current_ema21 else 'below',
                    'ftm_signal': ftm_data['strength_signals'][current_idx],
                    'multi_timeframe_mayor': mayor_trend,
                    'multi_timeframe_menor': menor_trend
                },
                'recommendation': 'Scalping' if interval in ['15m', '30m'] else 'Intraday/Swing'
            }
            
        except Exception as e:
            print(f"Error en check_volume_ema_strategy_corrected para {symbol}: {e}")
            return None

    def generate_volume_ema_chart_corrected(self, symbol, interval, df, ema21, volume_ma21, ftm_data, signal_type):
        """Generar gráfico para Volume EMA Strategy corregida"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1, 1])
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de velas y EMA21
            closes = df['close'].iloc[-50:].values
            ema21_sub = ema21[-50:]
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ema21_sub, label='EMA21', color='blue', linewidth=2)
            ax1.set_title(f'{symbol} - {interval} - Volume EMA Strategy ({signal_type})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volumen y Volume MA21
            volume_sub = df['volume'].iloc[-50:].values
            volume_ma21_sub = volume_ma21[-50:]
            
            ax2.bar(dates_matplotlib, volume_sub, color='gray', alpha=0.6, label='Volumen')
            ax2.plot(dates_matplotlib, volume_ma21_sub, label='Volume MA21', color='orange', linewidth=2)
            ax2.axhline(y=volume_ma21_sub[-1] * 3, color='red', linestyle='--', alpha=0.7, label='Umbral 3x')
            ax2.set_ylabel('Volumen')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax3.set_ylabel('FT Maverick')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Volume EMA: {e}")
            return None

    # =================== SISTEMA DE ALERTAS ===================

    def check_all_strategies_for_symbol(self, symbol, interval):
        """Revisar todas las estrategias para un símbolo e intervalo"""
        current_time = self.get_bolivia_time()
        
        # Verificar si es hora de scalping para intervalos cortos
        if interval in ['15m', '30m'] and not self.is_scalping_time():
            return []
        
        # Verificar si se debe revisar este intervalo
        if not self.should_check_interval(interval, current_time):
            return []
        
        alerts = []
        
        # Lista de funciones de estrategia
        strategy_functions = [
            ('TREND_RIDER', self.check_trend_rider_strategy),
            ('MOMENTUM_DIVERGENCE', self.check_momentum_divergence_strategy),
            ('BOLLINGER_SQUEEZE', self.check_bollinger_squeeze_strategy),
            ('ADX_POWER_TREND', self.check_adx_power_trend_strategy),
            ('MACD_HISTOGRAM_REVERSAL', self.check_macd_histogram_reversal_strategy),
            ('VOLUME_SPIKE_MOMENTUM', self.check_volume_spike_momentum_strategy),
            ('DOUBLE_CONFIRMATION_RSI', self.check_double_confirmation_rsi_strategy),
            ('TREND_STRENGTH_MAVERICK', self.check_trend_strength_maverick_strategy),
            ('WHALE_FOLLOWING', self.check_whale_following_strategy),
            ('MA_CONVERGENCE_DIVERGENCE', self.check_ma_convergence_divergence_strategy),
            ('RSI_MAVERICK_EXTREME', self.check_rsi_maverick_extreme_strategy),
            ('VOLUME_PRICE_DIVERGENCE', self.check_volume_price_divergence_strategy),
            ('VOLUME_EMA_STRATEGY', self.check_volume_ema_strategy_corrected)
        ]
        
        for strategy_name, strategy_func in strategy_functions:
            try:
                # Verificar si esta estrategia es válida para este intervalo
                if interval in STRATEGY_CONFIG.get(strategy_name, {}).get('intervals', []):
                    signal_data = strategy_func(symbol, interval)
                    
                    if signal_data and 'signal' in signal_data:
                        # Crear clave única para evitar alertas duplicadas
                        alert_key = f"{symbol}_{interval}_{strategy_name}_{signal_data['signal']}"
                        
                        # Verificar si ya enviamos esta alerta recientemente (últimos 30 minutos)
                        if alert_key in self.strategy_alerts_sent:
                            last_sent = self.strategy_alerts_sent[alert_key]
                            if (current_time - last_sent).seconds < 1800:  # 30 minutos
                                continue
                        
                        alerts.append(signal_data)
                        self.strategy_alerts_sent[alert_key] = current_time
                        
            except Exception as e:
                print(f"Error ejecutando estrategia {strategy_name} para {symbol} {interval}: {e}")
                continue
        
        return alerts

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert_strategy(signal_data):
    """Enviar alerta de estrategia por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        symbol = signal_data['symbol']
        interval = signal_data['interval']
        signal_type = signal_data['signal']
        strategy = signal_data['strategy']
        current_price = signal_data['current_price']
        entry = signal_data['entry']
        filters = signal_data.get('filters', {})
        recommendation = signal_data.get('recommendation', '')
        
        # Construir mensaje
        emoji = "🚀" if signal_type == 'LONG' else "📉"
        direction = "LONG" if signal_type == 'LONG' else "SHORT"
        
        message = f"""
{emoji} Alerta {direction} {symbol} en {interval} {emoji}

Estrategia: {strategy.replace('_', ' ').title()}
Precio actual: ${current_price:.6f} | Entrada: ${entry:.6f}

Filtros:"""
        
        # Añadir filtros específicos según estrategia
        if strategy == 'TREND_RIDER':
            message += f"""
- Cruce MACD Temporalidad Menor
- Precio {'>' if filters.get('ma50_condition') == 'above' else '<'} de MA50 Temporalidad Actual
- Precio {'>' if filters.get('ma200_condition') == 'above' else '<'} MA200 Temporalidad Mayor
- Señal FTMaverick: {filters.get('ftm_signal', 'N/A')}"""
        
        elif strategy == 'BOLLINGER_SQUEEZE':
            message += f"""
- Squeeze detectado en Bandas de Bollinger
- Volumen {filters.get('volume_ratio', 0):.1f}x promedio
- ADX: {filters.get('adx', 0):.1f} > 25
- {'+DI > -DI' if filters.get('dmi_cross') else '-DI > +DI'}"""
        
        elif strategy == 'WHALE_FOLLOWING':
            message += f"""
- Señal Ballenas: {filters.get('whale_signal', 0):.1f}
- {'+' if signal_type == 'LONG' else '-'}DI cruzó { '-' if signal_type == 'LONG' else '+'}DI
- ADX: {filters.get('adx', 0):.1f} > 25
- Precio {'>' if filters.get('ma200_condition') == 'above' else '<'} MA200"""
        
        elif strategy == 'VOLUME_EMA_STRATEGY':
            message += f"""
- Volumen: {filters.get('volume_ratio', 0):.1f}x promedio
- Precio {'>' if filters.get('price_vs_ema21') == 'above' else '<'} EMA21
- FTMaverick: {filters.get('ftm_signal', 'N/A')}
- Multi-Timeframe: {filters.get('multi_timeframe_mayor', 'N/A')}/{filters.get('multi_timeframe_menor', 'N/A')}"""
        
        else:
            # Mensaje genérico para otras estrategias
            for key, value in filters.items():
                if isinstance(value, float):
                    message += f"\n- {key.replace('_', ' ').title()}: {value:.2f}"
                elif isinstance(value, bool):
                    if value:
                        message += f"\n- {key.replace('_', ' ').title()}: ✅"
                else:
                    message += f"\n- {key.replace('_', ' ').title()}: {value}"
        
        message += f"\n\nRecomendación: {recommendation}"
        message += f"\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Enviar imagen si existe
        if 'chart' in signal_data and signal_data['chart']:
            asyncio.run(bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=signal_data['chart'],
                caption=message
            ))
        else:
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            ))
        
        print(f"Alerta enviada: {strategy} - {symbol} - {interval} - {signal_type}")
        
    except Exception as e:
        print(f"Error enviando alerta de estrategia a Telegram: {e}")

def background_strategy_checker():
    """Verificador de estrategias en segundo plano"""
    print("Iniciando verificador de estrategias en segundo plano...")
    
    intervals_to_check = list(STRATEGY_CONFIG['TREND_RIDER']['intervals'] + 
                            STRATEGY_CONFIG['MOMENTUM_DIVERGENCE']['intervals'] +
                            STRATEGY_CONFIG['BOLLINGER_SQUEEZE']['intervals'] +
                            STRATEGY_CONFIG['ADX_POWER_TREND']['intervals'])
    
    intervals_to_check = list(set(intervals_to_check))  # Eliminar duplicados
    
    last_check_times = {interval: datetime.now() for interval in intervals_to_check}
    
    while True:
        try:
            current_time = datetime.now()
            
            for interval in intervals_to_check:
                # Calcular segundos desde última revisión
                seconds_since_last = (current_time - last_check_times.get(interval, current_time)).seconds
                
                # Verificar si es hora de revisar este intervalo
                check_config = indicator.check_intervals.get(interval, {'check_seconds': 300})
                if seconds_since_last >= check_config['check_seconds']:
                    
                    print(f"Revisando estrategias para intervalo {interval}...")
                    
                    for symbol in CRYPTO_SYMBOLS:
                        try:
                            alerts = indicator.check_all_strategies_for_symbol(symbol, interval)
                            
                            for alert in alerts:
                                send_telegram_alert_strategy(alert)
                                time.sleep(1)  # Pequeña pausa entre alertas
                            
                            time.sleep(0.5)  # Pausa entre símbolos
                            
                        except Exception as e:
                            print(f"Error revisando {symbol} {interval}: {e}")
                            continue
                    
                    last_check_times[interval] = current_time
                    print(f"Revisión completada para {interval}")
            
            time.sleep(10)  # Revisar cada 10 segundos
            
        except Exception as e:
            print(f"Error en background_strategy_checker: {e}")
            time.sleep(60)

# Iniciar verificador de estrategias en segundo plano
try:
    strategy_thread = Thread(target=background_strategy_checker, daemon=True)
    strategy_thread.start()
    print("Background strategy checker iniciado correctamente")
except Exception as e:
    print(f"Error iniciando background strategy checker: {e}")

# =================== FUNCIONES DEL SISTEMA EXISTENTE ===================

def generate_signals_improved(symbol, interval, di_period=14, adx_threshold=25, 
                             sr_period=50, rsi_length=14, bb_multiplier=2.0, 
                             volume_filter='Todos', leverage=15):
    """GENERACIÓN DE SEÑALES MEJORADA - Función existente del sistema"""
    return indicator.generate_signals_improved(symbol, interval, di_period, adx_threshold, 
                                             sr_period, rsi_length, bb_multiplier, 
                                             volume_filter, leverage)

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
        
        signal_data = generate_signals_improved(
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
        
        for symbol in CRYPTO_SYMBOLS:
            try:
                signal_data = generate_signals_improved(
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
        
        for symbol in CRYPTO_SYMBOLS:
            try:
                signal_data = generate_signals_improved(symbol, interval, di_period, adx_threshold)
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
                        'risk_category': 'bajo'  # Todas son de bajo riesgo
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
