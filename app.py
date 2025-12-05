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

# Top 10 bajo riesgo (excluye Doge para estrategia volumen)
TOP10_LOW_RISK = ["BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
                  "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "LTC-USDT"]

# Clasificación de riesgo optimizada
CRYPTO_RISK_CLASSIFICATION = {
    "bajo": CRYPTO_SYMBOLS[:20],
    "medio": CRYPTO_SYMBOLS[20:30],
    "alto": CRYPTO_SYMBOLS[30:37],
    "memecoins": CRYPTO_SYMBOLS[37:]
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

# Temporalidades para estrategia volumen+EMA21
VOLUME_EMA_STRATEGY_INTERVALS = ['1h', '4h', '12h', '1D']

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_operations = {}
        self.winrate_data = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.sent_exit_signals = set()
        self.volume_ema_signals_sent = set()
        self.divergence_cache = {}
        self.chart_pattern_cache = {}
        
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela - OPTIMIZADO"""
        if interval == '15m':
            minutes = current_time.minute
            remaining = 15 - (minutes % 15)
            return remaining <= 7  # 50% del tiempo
        elif interval == '30m':
            minutes = current_time.minute
            remaining = 30 - (minutes % 30)
            return remaining <= 15
        elif interval == '1h':
            return current_time.minute >= 30
        elif interval == '2h':
            hour = current_time.hour
            remaining = 2 - (hour % 2)
            return remaining <= 1
        elif interval == '4h':
            hour = current_time.hour
            remaining = 4 - (hour % 4)
            return remaining <= 1
        elif interval == '8h':
            hour = current_time.hour
            remaining = 8 - (hour % 8)
            return remaining <= 2
        elif interval == '12h':
            hour = current_time.hour
            if hour < 8:
                remaining = 8 - hour
            elif hour < 20:
                remaining = 20 - hour
            else:
                remaining = 32 - hour
            return remaining <= 3
        elif interval == '1D':
            return current_time.hour >= 12
        elif interval == '1W':
            return current_time.weekday() >= 3
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
                        return self.generate_sample_data(symbol, interval, limit)
                    
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
            print(f"Error obteniendo datos para {symbol} {interval}: {e}")
        
        return self.generate_sample_data(symbol, interval, limit)

    def generate_sample_data(self, symbol, interval, limit):
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

    def calculate_support_resistance_levels(self, high, low, close, period=50, num_levels=6):
        """Calcular niveles dinámicos de soporte y resistencia"""
        n = len(close)
        if n < period:
            return [], []
        
        # Usar pivot points y puntos máximos/mínimos
        pivots_high = []
        pivots_low = []
        
        for i in range(period, n-period):
            window_high = high[i-period:i+period]
            window_low = low[i-period:i+period]
            
            if high[i] == np.max(window_high):
                pivots_high.append(high[i])
            
            if low[i] == np.min(window_low):
                pivots_low.append(low[i])
        
        # Agrupar niveles cercanos
        def cluster_levels(levels, threshold=0.01):
            if not levels:
                return []
            
            levels.sort()
            clusters = []
            current_cluster = [levels[0]]
            
            for price in levels[1:]:
                if abs(price - np.mean(current_cluster)) / np.mean(current_cluster) < threshold:
                    current_cluster.append(price)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [price]
            
            if current_cluster:
                clusters.append(np.mean(current_cluster))
            
            return clusters
        
        supports = cluster_levels(pivots_low)
        resistances = cluster_levels(pivots_high)
        
        # Ordenar y limitar número de niveles
        supports.sort(reverse=True)
        resistances.sort()
        
        current_price = close[-1]
        
        # Filtrar niveles relevantes (cercanos al precio actual)
        relevant_supports = [s for s in supports if s < current_price * 1.5 and s > current_price * 0.5]
        relevant_resistances = [r for r in resistances if r < current_price * 1.5 and r > current_price * 0.5]
        
        # Limitar a num_levels cada uno
        return relevant_supports[:num_levels], relevant_resistances[:num_levels]

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15, num_levels=6):
        """Calcular entradas y salidas óptimas con soportes/resistencias dinámicos"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular soportes y resistencias dinámicos
            supports, resistances = self.calculate_support_resistance_levels(
                high, low, close, period=50, num_levels=num_levels
            )
            
            # Si no hay suficientes niveles, usar mínimos/máximos recientes
            if len(supports) < 2:
                supports = [np.min(low[-50:]), np.min(low[-100:])]
            if len(resistances) < 2:
                resistances = [np.max(high[-50:]), np.max(high[-100:])]
            
            # Ordenar niveles
            supports.sort(reverse=True)  # De mayor a menor
            resistances.sort()  # De menor a mayor
            
            # Encontrar niveles más cercanos
            closest_support = min(supports, key=lambda x: abs(x - current_price)) if supports else current_price * 0.95
            closest_resistance = min(resistances, key=lambda x: abs(x - current_price)) if resistances else current_price * 1.05
            
            if signal_type == 'LONG':
                # Entrada en el soporte más cercano POR DEBAJO del precio actual
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)  # Soporte más alto por debajo del precio
                else:
                    entry = current_price * 0.995  # Ligera corrección
                
                stop_loss = entry - (current_atr * 1.8)
                
                # Take profits en resistencias cercanas
                valid_resistances = [r for r in resistances if r > entry]
                if valid_resistances:
                    take_profit = [valid_resistances[0]]  # Primera resistencia
                    if len(valid_resistances) > 1:
                        take_profit.append(valid_resistances[1])  # Segunda resistencia
                else:
                    take_profit = [entry * 1.02]
                
                # Asegurar relación riesgo/beneficio mínima
                min_tp = entry + (2 * (entry - stop_loss))
                take_profit[0] = max(take_profit[0], min_tp)
                
            else:  # SHORT
                # Entrada en la resistencia más cercana POR ENCIMA del precio actual
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)  # Resistencia más baja por encima del precio
                else:
                    entry = current_price * 1.005  # Ligera corrección
                
                stop_loss = entry + (current_atr * 1.8)
                
                # Take profits en soportes cercanos
                valid_supports = [s for s in supports if s < entry]
                if valid_supports:
                    take_profit = [valid_supports[0]]  # Primer soporte
                    if len(valid_supports) > 1:
                        take_profit.append(valid_supports[1])  # Segundo soporte
                else:
                    take_profit = [entry * 0.98]
                
                # Asegurar relación riesgo/beneficio mínima
                max_tp = entry - (2 * (stop_loss - entry))
                take_profit[0] = min(take_profit[0], max_tp)
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profit],
                'supports': [float(s) for s in supports],
                'resistances': [float(r) for r in resistances],
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
                'supports': [current_price * 0.95, current_price * 0.90],
                'resistances': [current_price * 1.05, current_price * 1.10],
                'atr': 0.0,
                'atr_percentage': 0.0
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
        for i in range(period-1):
            ema[i] = prices[i]
        
        return ema

    def calculate_sma(self, prices, period):
        """Calcular SMA manualmente"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        sma = np.zeros_like(prices)
        for i in range(len(prices)):
            if i >= period - 1:
                sma[i] = np.mean(prices[i-period+1:i+1])
            else:
                sma[i] = np.mean(prices[:i+1])
        
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
                std[i] = np.std(prices[:i+1])
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional manualmente"""
        if len(prices) < period + 1:
            return np.ones_like(prices) * 50
        
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
                rs[i] = 100 if avg_gains[i] > 0 else 1
        
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
            
            colors = []
            for i in range(n):
                if no_trade_zones[i]:
                    colors.append('red')
                elif trend_strength[i] > 0:
                    colors.append('green')
                else:
                    colors.append('red')
            
            return {
                'bb_width': bb_width.tolist(),
                'trend_strength': trend_strength.tolist(),
                'basis': basis.tolist(),
                'upper_band': upper.tolist(),
                'lower_band': lower.tolist(),
                'high_zone_threshold': float(high_zone_threshold),
                'no_trade_zones': no_trade_zones.tolist(),
                'strength_signals': strength_signals,
                'colors': colors
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

    def check_bollinger_conditions(self, df, interval, signal_type):
        """Verificar condiciones de Bandas de Bollinger"""
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
        """Verificar condiciones multi-timeframe obligatorias"""
        try:
            if interval in ['12h', '1D', '1W']:
                return False
                
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

    def detect_divergence(self, price, indicator, lookback=14, signal_duration=7):
        """Detectar divergencias con duración de señal"""
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
        
        # Extender señal por signal_duration velas
        bullish_extended = bullish_div.copy()
        bearish_extended = bearish_div.copy()
        
        for i in range(n):
            if bullish_div[i]:
                for j in range(1, min(signal_duration, n-i)):
                    bullish_extended[i+j] = True
            if bearish_div[i]:
                for j in range(1, min(signal_duration, n-i)):
                    bearish_extended[i+j] = True
        
        return bullish_extended.tolist(), bearish_extended.tolist()

    def check_breakout(self, high, low, close, support, resistance):
        """Detectar rupturas de tendencia"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if close[i] > resistance[i] and high[i] > high[i-1]:
                breakout_up[i] = True
                # Extender señal 1 vela más
                if i + 1 < n:
                    breakout_up[i+1] = True
            
            if close[i] < support[i] and low[i] < low[i-1]:
                breakout_down[i] = True
                # Extender señal 1 vela más
                if i + 1 < n:
                    breakout_down[i+1] = True
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI con confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                di_cross_bullish[i] = True
                # Extender señal 1 vela más
                if i + 1 < n:
                    di_cross_bullish[i+1] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
                # Extender señal 1 vela más
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

    def detect_chart_patterns(self, high, low, close, lookback=50, signal_duration=7):
        """Detectar patrones de chartismo con duración de señal"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-7):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            
            # Hombro Cabeza Hombro
            if len(window_high) >= 20:
                max_idx = np.argmax(window_high)
                if (max_idx > 5 and max_idx < len(window_high)-5 and
                    window_high[max_idx-3] < window_high[max_idx] and
                    window_high[max_idx+3] < window_high[max_idx]):
                    patterns['head_shoulders'][i] = True
            
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
            
            # Bullish Flag (simplificado)
            if i >= 10:
                recent_highs = high[i-10:i+1]
                recent_lows = low[i-10:i+1]
                if (np.max(recent_highs) - np.min(recent_lows)) / np.min(recent_lows) < 0.1:
                    if close[i] > close[i-5]:
                        patterns['bullish_flag'][i] = True
            
            # Bearish Flag (simplificado)
            if i >= 10:
                recent_highs = high[i-10:i+1]
                recent_lows = low[i-10:i+1]
                if (np.max(recent_highs) - np.min(recent_lows)) / np.min(recent_lows) < 0.1:
                    if close[i] < close[i-5]:
                        patterns['bearish_flag'][i] = True
        
        # Extender señal por signal_duration velas
        for pattern_name in patterns:
            pattern_array = patterns[pattern_name]
            extended = pattern_array.copy()
            for i in range(n):
                if pattern_array[i]:
                    for j in range(1, min(signal_duration, n-i)):
                        extended[i+j] = True
            patterns[pattern_name] = extended
        
        return patterns

    def calculate_volume_anomaly_improved(self, volume, close, period=20, std_multiplier=2):
        """Calcular anomalías de volumen con dirección de compra/venta"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_direction = np.zeros(n)  # 1: compra, -1: venta, 0: neutral
            
            ema_volume = self.calculate_ema(volume, period)
            
            for i in range(period, n):
                current_ema = ema_volume[i]
                
                # Ratio volumen actual vs EMA
                if current_ema > 0:
                    volume_ratio[i] = volume[i] / current_ema
                else:
                    volume_ratio[i] = 1
                
                # Detectar anomalía (> 2.5x EMA)
                if volume_ratio[i] > 2.5:
                    volume_anomaly[i] = True
                    
                    # Determinar dirección
                    if i > 0:
                        price_change = close[i] - close[i-1]
                        if price_change > 0:
                            volume_direction[i] = 1  # Compra
                        else:
                            volume_direction[i] = -1  # Venta
                
                # Detectar clusters (múltiples anomalías en 5-10 periodos)
                if i >= 10:
                    recent_anomalies = volume_anomaly[max(0, i-9):i+1]
                    if np.sum(recent_anomalies) >= 3:
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ema': ema_volume.tolist(),
                'volume_direction': volume_direction.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly_improved: {e}")
            n = len(volume)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ema': [0] * n,
                'volume_direction': [0] * n
            }

    def check_ma_crossover(self, ma_fast, ma_slow, lookback=2):
        """Detectar cruce de medias móviles"""
        n = len(ma_fast)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: MA rápida cruza por encima de MA lenta
            if (ma_fast[i] > ma_slow[i] and 
                ma_fast[i-1] <= ma_slow[i-1]):
                ma_cross_bullish[i] = True
                # Extender señal 1 vela más
                if i + 1 < n:
                    ma_cross_bullish[i+1] = True
            
            # Cruce bajista: MA rápida cruza por debajo de MA lenta
            if (ma_fast[i] < ma_slow[i] and 
                ma_fast[i-1] >= ma_slow[i-1]):
                ma_cross_bearish[i] = True
                # Extender señal 1 vela más
                if i + 1 < n:
                    ma_cross_bearish[i+1] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()

    def check_adx_slope(self, adx, lookback=3):
        """Verificar pendiente positiva del ADX"""
        n = len(adx)
        adx_slope_positive = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            if adx[i] > 25:  # ADX sobre nivel
                recent_adx = adx[i-lookback:i+1]
                if len(recent_adx) > 1:
                    slope = (recent_adx[-1] - recent_adx[0]) / lookback
                    if slope > 0:
                        adx_slope_positive[i] = True
        
        return adx_slope_positive.tolist()

    def check_macd_crossover(self, macd_line, macd_signal, lookback=2):
        """Detectar cruce del MACD"""
        n = len(macd_line)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: MACD cruza por encima de señal
            if (macd_line[i] > macd_signal[i] and 
                macd_line[i-1] <= macd_signal[i-1]):
                macd_cross_bullish[i] = True
                # Extender señal 1 vela más
                if i + 1 < n:
                    macd_cross_bullish[i+1] = True
            
            # Cruce bajista: MACD cruza por debajo de señal
            if (macd_line[i] < macd_signal[i] and 
                macd_line[i-1] >= macd_signal[i-1]):
                macd_cross_bearish[i] = True
                # Extender señal 1 vela más
                if i + 1 < n:
                    macd_cross_bearish[i+1] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()

    def evaluate_signal_conditions_optimized(self, data, current_idx, interval):
        """Evaluar condiciones de señal con pesos optimizados"""
        weights = self.get_weights_by_timeframe(interval)
        
        conditions = {
            'long': {},
            'short': {}
        }
        
        # Inicializar condiciones
        for signal_type in ['long', 'short']:
            for key, weight_info in weights[signal_type].items():
                conditions[signal_type][key] = {
                    'value': False, 
                    'weight': weight_info['weight'], 
                    'description': weight_info['description']
                }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        # Condiciones LONG
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_long', False)
        
        if interval in ['12h', '1D']:
            conditions['long']['whale_signal']['value'] = (
                data['whale_pump'][current_idx] > 20 and
                data['confirmed_buy'][current_idx]
            )
        
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        # Indicadores complementarios
        conditions['long']['ma_cross']['value'] = data.get('ma_cross_bullish', [False])[current_idx]
        conditions['long']['di_cross']['value'] = data.get('di_cross_bullish', [False])[current_idx]
        conditions['long']['adx_slope']['value'] = data.get('adx_slope_positive', [False])[current_idx]
        conditions['long']['bollinger_bands']['value'] = data.get('bollinger_conditions_long', False)
        conditions['long']['macd_cross']['value'] = data.get('macd_cross_bullish', [False])[current_idx]
        conditions['long']['volume_anomaly']['value'] = (
            data['volume_anomaly'][current_idx] and 
            data['volume_direction'][current_idx] == 1
        )
        conditions['long']['rsi_maverick_divergence']['value'] = data.get('rsi_maverick_bullish_divergence', [False])[current_idx]
        conditions['long']['rsi_traditional_divergence']['value'] = data.get('rsi_bullish_divergence', [False])[current_idx]
        conditions['long']['chart_pattern']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_flag'][current_idx]
        )
        conditions['long']['breakout']['value'] = data.get('breakout_up', [False])[current_idx]
        
        # Condiciones SHORT
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        
        if interval in ['12h', '1D']:
            conditions['short']['whale_signal']['value'] = (
                data['whale_dump'][current_idx] > 20 and
                data['confirmed_sell'][current_idx]
            )
        
        conditions['short']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            not data['no_trade_zones'][current_idx]
        )
        
        # Indicadores complementarios
        conditions['short']['ma_cross']['value'] = data.get('ma_cross_bearish', [False])[current_idx]
        conditions['short']['di_cross']['value'] = data.get('di_cross_bearish', [False])[current_idx]
        conditions['short']['adx_slope']['value'] = data.get('adx_slope_positive', [False])[current_idx]
        conditions['short']['bollinger_bands']['value'] = data.get('bollinger_conditions_short', False)
        conditions['short']['macd_cross']['value'] = data.get('macd_cross_bearish', [False])[current_idx]
        conditions['short']['volume_anomaly']['value'] = (
            data['volume_anomaly'][current_idx] and 
            data['volume_direction'][current_idx] == -1
        )
        conditions['short']['rsi_maverick_divergence']['value'] = data.get('rsi_maverick_bearish_divergence', [False])[current_idx]
        conditions['short']['rsi_traditional_divergence']['value'] = data.get('rsi_bearish_divergence', [False])[current_idx]
        conditions['short']['chart_pattern']['value'] = (
            data['chart_patterns']['head_shoulders'][current_idx] or
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['bearish_flag'][current_idx]
        )
        conditions['short']['breakout']['value'] = data.get('breakout_down', [False])[current_idx]
        
        return conditions

    def get_weights_by_timeframe(self, interval):
        """Obtener pesos según temporalidad"""
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            return {
                'long': {
                    'multi_timeframe': {'weight': 30, 'description': 'Multi-TF obligatorio'},
                    'trend_strength': {'weight': 25, 'description': 'Fuerza tendencia Maverick'},
                    'ma_cross': {'weight': 10, 'description': 'Cruce Medias Móviles'},
                    'di_cross': {'weight': 10, 'description': 'Cruce DMI'},
                    'adx_slope': {'weight': 5, 'description': 'ADX pendiente positiva'},
                    'bollinger_bands': {'weight': 8, 'description': 'Bandas de Bollinger'},
                    'macd_cross': {'weight': 10, 'description': 'Cruce MACD'},
                    'volume_anomaly': {'weight': 7, 'description': 'Volumen Anómalo'},
                    'rsi_maverick_divergence': {'weight': 8, 'description': 'Divergencia RSI Maverick'},
                    'rsi_traditional_divergence': {'weight': 5, 'description': 'Divergencia RSI Tradicional'},
                    'chart_pattern': {'weight': 5, 'description': 'Patrón Chartista'},
                    'breakout': {'weight': 5, 'description': 'Ruptura'}
                },
                'short': {
                    'multi_timeframe': {'weight': 30, 'description': 'Multi-TF obligatorio'},
                    'trend_strength': {'weight': 25, 'description': 'Fuerza tendencia Maverick'},
                    'ma_cross': {'weight': 10, 'description': 'Cruce Medias Móviles'},
                    'di_cross': {'weight': 10, 'description': 'Cruce DMI'},
                    'adx_slope': {'weight': 5, 'description': 'ADX pendiente positiva'},
                    'bollinger_bands': {'weight': 8, 'description': 'Bandas de Bollinger'},
                    'macd_cross': {'weight': 10, 'description': 'Cruce MACD'},
                    'volume_anomaly': {'weight': 7, 'description': 'Volumen Anómalo'},
                    'rsi_maverick_divergence': {'weight': 8, 'description': 'Divergencia RSI Maverick'},
                    'rsi_traditional_divergence': {'weight': 5, 'description': 'Divergencia RSI Tradicional'},
                    'chart_pattern': {'weight': 5, 'description': 'Patrón Chartista'},
                    'breakout': {'weight': 5, 'description': 'Ruptura'}
                }
            }
        elif interval in ['12h', '1D']:
            return {
                'long': {
                    'whale_signal': {'weight': 30, 'description': 'Señal Ballenas'},
                    'trend_strength': {'weight': 25, 'description': 'Fuerza tendencia Maverick'},
                    'ma_cross': {'weight': 10, 'description': 'Cruce Medias Móviles'},
                    'di_cross': {'weight': 10, 'description': 'Cruce DMI'},
                    'adx_slope': {'weight': 5, 'description': 'ADX pendiente positiva'},
                    'bollinger_bands': {'weight': 8, 'description': 'Bandas de Bollinger'},
                    'macd_cross': {'weight': 10, 'description': 'Cruce MACD'},
                    'volume_anomaly': {'weight': 7, 'description': 'Volumen Anómalo'},
                    'rsi_maverick_divergence': {'weight': 8, 'description': 'Divergencia RSI Maverick'},
                    'rsi_traditional_divergence': {'weight': 5, 'description': 'Divergencia RSI Tradicional'},
                    'chart_pattern': {'weight': 5, 'description': 'Patrón Chartista'},
                    'breakout': {'weight': 5, 'description': 'Ruptura'}
                },
                'short': {
                    'whale_signal': {'weight': 30, 'description': 'Señal Ballenas'},
                    'trend_strength': {'weight': 25, 'description': 'Fuerza tendencia Maverick'},
                    'ma_cross': {'weight': 10, 'description': 'Cruce Medias Móviles'},
                    'di_cross': {'weight': 10, 'description': 'Cruce DMI'},
                    'adx_slope': {'weight': 5, 'description': 'ADX pendiente positiva'},
                    'bollinger_bands': {'weight': 8, 'description': 'Bandas de Bollinger'},
                    'macd_cross': {'weight': 10, 'description': 'Cruce MACD'},
                    'volume_anomaly': {'weight': 7, 'description': 'Volumen Anómalo'},
                    'rsi_maverick_divergence': {'weight': 8, 'description': 'Divergencia RSI Maverick'},
                    'rsi_traditional_divergence': {'weight': 5, 'description': 'Divergencia RSI Tradicional'},
                    'chart_pattern': {'weight': 5, 'description': 'Patrón Chartista'},
                    'breakout': {'weight': 5, 'description': 'Ruptura'}
                }
            }
        else:  # 1W
            return {
                'long': {
                    'trend_strength': {'weight': 55, 'description': 'Fuerza tendencia Maverick'},
                    'ma_cross': {'weight': 10, 'description': 'Cruce Medias Móviles'},
                    'di_cross': {'weight': 10, 'description': 'Cruce DMI'},
                    'adx_slope': {'weight': 5, 'description': 'ADX pendiente positiva'},
                    'bollinger_bands': {'weight': 8, 'description': 'Bandas de Bollinger'},
                    'macd_cross': {'weight': 10, 'description': 'Cruce MACD'},
                    'volume_anomaly': {'weight': 7, 'description': 'Volumen Anómalo'},
                    'rsi_maverick_divergence': {'weight': 8, 'description': 'Divergencia RSI Maverick'},
                    'rsi_traditional_divergence': {'weight': 5, 'description': 'Divergencia RSI Tradicional'},
                    'chart_pattern': {'weight': 5, 'description': 'Patrón Chartista'},
                    'breakout': {'weight': 5, 'description': 'Ruptura'}
                },
                'short': {
                    'trend_strength': {'weight': 55, 'description': 'Fuerza tendencia Maverick'},
                    'ma_cross': {'weight': 10, 'description': 'Cruce Medias Móviles'},
                    'di_cross': {'weight': 10, 'description': 'Cruce DMI'},
                    'adx_slope': {'weight': 5, 'description': 'ADX pendiente positiva'},
                    'bollinger_bands': {'weight': 8, 'description': 'Bandas de Bollinger'},
                    'macd_cross': {'weight': 10, 'description': 'Cruce MACD'},
                    'volume_anomaly': {'weight': 7, 'description': 'Volumen Anómalo'},
                    'rsi_maverick_divergence': {'weight': 8, 'description': 'Divergencia RSI Maverick'},
                    'rsi_traditional_divergence': {'weight': 5, 'description': 'Divergencia RSI Tradicional'},
                    'chart_pattern': {'weight': 5, 'description': 'Patrón Chartista'},
                    'breakout': {'weight': 5, 'description': 'Ruptura'}
                }
            }

    def calculate_signal_score_optimized(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal optimizada"""
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        signal_conditions = conditions.get(signal_type, {})
        
        # Verificar condiciones obligatorias
        obligatory_keys = []
        for key, condition in signal_conditions.items():
            if condition['weight'] >= 25:  # Condiciones con peso >= 25 son obligatorias
                obligatory_keys.append(key)
        
        # Verificar que todas las condiciones obligatorias se cumplan
        all_obligatory_met = all(signal_conditions[cond]['value'] for cond in obligatory_keys)
        
        if not all_obligatory_met:
            return 0, []
        
        # Calcular score
        for key, condition in signal_conditions.items():
            total_weight += condition['weight']
            if condition['value']:
                achieved_weight += condition['weight']
                fulfilled_conditions.append(condition['description'])
        
        if total_weight == 0:
            return 0, []
        
        base_score = (achieved_weight / total_weight * 100)
        
        # Ajustar score mínimo según MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0
        final_score = min(final_score, 100)
        
        return final_score, fulfilled_conditions

    def check_volume_ema_ftm_signal(self, symbol, interval):
        """Nueva estrategia: Desplome por Volumen + EMA21 con Filtros FTMaverick/Multi-Timeframe"""
        try:
            if interval not in VOLUME_EMA_STRATEGY_INTERVALS:
                return None
            
            if symbol not in TOP10_LOW_RISK:
                return None
            
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular EMA21 de precio
            ema_21 = self.calculate_ema(close, 21)
            
            # Calcular MA21 de volumen
            volume_ma_21 = self.calculate_sma(volume, 21)
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            current_volume_ma = volume_ma_21[current_idx]
            current_ema = ema_21[current_idx]
            
            # Condición A: Volumen y EMA
            volume_condition = current_volume > (current_volume_ma * 2.5)
            
            # Determinar señal basada en precio vs EMA21
            signal_type = None
            if volume_condition:
                if current_price > current_ema:
                    signal_type = 'LONG'
                elif current_price < current_ema:
                    signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            # Condición B: Filtro FTMaverick (Timeframe Actual)
            ftm_current = self.calculate_trend_strength_maverick(close)
            ftm_condition = not ftm_current['no_trade_zones'][-1]
            
            if not ftm_condition:
                return None
            
            # Condición C: Filtro Multi-Timeframe
            if interval in ['1h', '4h']:
                hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                
                # TF Mayor
                mayor_df = self.get_kucoin_data(symbol, hierarchy.get('mayor', '4h'), 50)
                if mayor_df is not None and len(mayor_df) > 20:
                    mayor_trend = self.calculate_trend_strength_maverick(mayor_df['close'].values)
                    mayor_signal = mayor_trend['strength_signals'][-1]
                    
                    if signal_type == 'LONG':
                        mayor_condition = mayor_signal in ['STRONG_UP', 'WEAK_UP', 'NEUTRAL']
                    else:
                        mayor_condition = mayor_signal in ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL']
                else:
                    mayor_condition = False
                
                # TF Menor
                menor_df = self.get_kucoin_data(symbol, hierarchy.get('menor', '30m'), 30)
                if menor_df is not None and len(menor_df) > 10:
                    menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                    menor_signal = menor_trend['strength_signals'][-1]
                    
                    if signal_type == 'LONG':
                        menor_condition = menor_signal in ['STRONG_UP', 'WEAK_UP']
                    else:
                        menor_condition = menor_signal in ['STRONG_DOWN', 'WEAK_DOWN']
                else:
                    menor_condition = False
                
                multi_tf_condition = mayor_condition and menor_condition
            else:
                multi_tf_condition = True
            
            if not multi_tf_condition:
                return None
            
            # Todas las condiciones cumplidas
            volume_ratio = current_volume / current_volume_ma if current_volume_ma > 0 else 1
            
            # Obtener información de tendencia para el mensaje
            mayor_trend_text = 'ALCISTA' if signal_type == 'LONG' else 'BAJISTA'
            menor_trend_text = 'ALCISTA' if signal_type == 'LONG' else 'BAJISTA'
            
            signal_key = f"volume_ema_{symbol}_{interval}_{signal_type}_{int(time.time())}"
            if signal_key in self.volume_ema_signals_sent:
                return None
            
            self.volume_ema_signals_sent.add(signal_key)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'price': current_price,
                'ema_21': current_ema,
                'volume_ratio': volume_ratio,
                'volume': current_volume,
                'volume_ma_21': current_volume_ma,
                'mayor_trend': mayor_trend_text,
                'menor_trend': menor_trend_text,
                'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error en check_volume_ema_ftm_signal para {symbol} {interval}: {e}")
            return None

    def generate_volume_ema_telegram_image(self, signal_data):
        """Generar imagen para Telegram de estrategia Volumen+EMA21"""
        try:
            symbol = signal_data['symbol']
            interval = signal_data['interval']
            
            df = self.get_kucoin_data(symbol, interval, 50)
            if df is None or len(df) < 30:
                return None
            
            fig = plt.figure(figsize=(12, 10))
            
            # Gráfico 1: Velas + EMA21
            ax1 = plt.subplot(2, 1, 1)
            
            dates = df['timestamp'].values
            if isinstance(dates[0], pd.Timestamp):
                dates = [d.to_pydatetime() for d in dates]
            
            # Graficar velas
            for i in range(len(df)):
                color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
                ax1.plot([dates[i], dates[i]], [df['low'].iloc[i], df['high'].iloc[i]], 
                        color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [df['open'].iloc[i], df['close'].iloc[i]], 
                        color=color, linewidth=3)
            
            # EMA21
            close = df['close'].values
            ema_21 = self.calculate_ema(close, 21)
            ax1.plot(dates, ema_21, 'blue', linewidth=2, label='EMA 21')
            
            # Destacar vela actual
            current_color = 'green' if signal_data['signal'] == 'LONG' else 'red'
            ax1.scatter([dates[-1]], [df['close'].iloc[-1]], 
                       color=current_color, s=100, zorder=5, 
                       edgecolor='black', linewidth=2)
            
            ax1.set_title(f"{symbol} - {interval} - Señal {signal_data['signal']} por Volumen+EMA21", 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Precio (USDT)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Volumen + Volume_MA21
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            
            volume = df['volume'].values
            volume_ma_21 = self.calculate_sma(volume, 21)
            
            # Barras de volumen
            bar_colors = []
            for i in range(len(df)):
                if i == len(df) - 1:
                    bar_colors.append(current_color)
                else:
                    bar_colors.append('green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red')
            
            ax2.bar(dates, volume, color=bar_colors, alpha=0.7)
            ax2.plot(dates, volume_ma_21, 'orange', linewidth=2, label='MA Volumen 21')
            
            # Destacar volumen actual
            ax2.scatter([dates[-1]], [volume[-1]], 
                       color=current_color, s=50, zorder=5, 
                       edgecolor='black', linewidth=1)
            
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
            print(f"Error generando imagen Volume+EMA: {e}")
            return None

    def generate_telegram_chart_image(self, symbol, interval, signal_type):
        """Generar imagen completa para Telegram con todos los indicadores"""
        try:
            df = self.get_kucoin_data(symbol, interval, 50)
            if df is None or len(df) < 30:
                return None
            
            fig = plt.figure(figsize=(14, 18))
            
            # 1. Gráfico de Velas con Bandas Bollinger y Medias Móviles
            ax1 = plt.subplot(8, 1, 1)
            
            dates = df['timestamp'].values
            if isinstance(dates[0], pd.Timestamp):
                dates = [d.to_pydatetime() for d in dates]
            
            # Graficar velas
            for i in range(len(df)):
                color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
                ax1.plot([dates[i], dates[i]], [df['low'].iloc[i], df['high'].iloc[i]], 
                        color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [df['open'].iloc[i], df['close'].iloc[i]], 
                        color=color, linewidth=3)
            
            # Bandas Bollinger (transparentes)
            close = df['close'].values
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            ax1.fill_between(dates, bb_lower, bb_upper, alpha=0.2, color='orange', label='BB')
            ax1.plot(dates, bb_middle, 'orange', linewidth=1, alpha=0.5)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            ax1.plot(dates, ma_9, 'yellow', linewidth=1, label='MA9')
            ax1.plot(dates, ma_21, 'cyan', linewidth=1, label='MA21')
            ax1.plot(dates, ma_50, 'magenta', linewidth=1, label='MA50')
            ax1.plot(dates, ma_200, 'white', linewidth=2, label='MA200')
            
            ax1.set_title(f'{symbol} - {interval} - Velas con Indicadores', fontsize=12)
            ax1.set_ylabel('Precio')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. ADX con DMI
            ax2 = plt.subplot(8, 1, 2, sharex=ax1)
            high = df['high'].values
            low = df['low'].values
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            ax2.plot(dates, adx, 'black', linewidth=2, label='ADX')
            ax2.plot(dates, plus_di, 'green', linewidth=1, label='+DI')
            ax2.plot(dates, minus_di, 'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7)
            
            ax2.set_ylabel('ADX/DMI')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Indicador de Volumen con Anomalías
            ax3 = plt.subplot(8, 1, 3, sharex=ax1)
            volume = df['volume'].values
            volume_data = self.calculate_volume_anomaly_improved(volume, close)
            
            bar_colors = []
            for i in range(len(df)):
                if volume_data['volume_anomaly'][i]:
                    if volume_data['volume_direction'][i] == 1:
                        bar_colors.append('green')
                    else:
                        bar_colors.append('red')
                else:
                    bar_colors.append('gray')
            
            ax3.bar(dates, volume, color=bar_colors, alpha=0.7)
            ax3.plot(dates, volume_data['volume_ema'], 'yellow', linewidth=1, label='EMA Vol')
            
            ax3.set_ylabel('Volumen')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # 4. Fuerza de Tendencia Maverick (barras)
            ax4 = plt.subplot(8, 1, 4, sharex=ax1)
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            bar_colors_ftm = []
            for i in range(len(dates)):
                if ftm_data['no_trade_zones'][i]:
                    bar_colors_ftm.append('red')
                elif ftm_data['trend_strength'][i] > 0:
                    bar_colors_ftm.append('green')
                else:
                    bar_colors_ftm.append('red')
            
            ax4.bar(dates, ftm_data['trend_strength'], color=bar_colors_ftm, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            
            ax4.set_ylabel('FT Maverick')
            ax4.grid(True, alpha=0.3)
            
            # 5. Indicador de Ballenas (solo para 12h y 1D)
            ax5 = plt.subplot(8, 1, 5, sharex=ax1)
            if interval in ['12h', '1D']:
                whale_data = self.calculate_whale_signals_improved(df)
                ax5.bar(dates, whale_data['whale_pump'], color='green', alpha=0.7, label='Compra')
                ax5.bar(dates, whale_data['whale_dump'], color='red', alpha=0.7, label='Venta')
                ax5.set_ylabel('Ballenas')
                ax5.legend(loc='upper left', fontsize=8)
            else:
                ax5.text(0.5, 0.5, 'Indicador Ballenas: Solo 12h/1D', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_ylabel('Ballenas')
            
            ax5.grid(True, alpha=0.3)
            
            # 6. RSI Modificado Maverick
            ax6 = plt.subplot(8, 1, 6, sharex=ax1)
            rsi_maverick = self.calculate_rsi_maverick(close)
            ax6.plot(dates, rsi_maverick, 'blue', linewidth=1)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax6.set_ylabel('RSI Maverick')
            ax6.grid(True, alpha=0.3)
            
            # 7. RSI Tradicional
            ax7 = plt.subplot(8, 1, 7, sharex=ax1)
            rsi_traditional = self.calculate_rsi(close)
            ax7.plot(dates, rsi_traditional, 'cyan', linewidth=1)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax7.set_ylabel('RSI Trad')
            ax7.grid(True, alpha=0.3)
            
            # 8. MACD con Histograma (barras)
            ax8 = plt.subplot(8, 1, 8, sharex=ax1)
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            
            bar_colors_macd = []
            for val in macd_histogram:
                bar_colors_macd.append('green' if val >= 0 else 'red')
            
            ax8.bar(dates, macd_histogram, color=bar_colors_macd, alpha=0.7)
            ax8.plot(dates, macd_line, 'blue', linewidth=1, label='MACD')
            ax8.plot(dates, macd_signal, 'red', linewidth=1, label='Señal')
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
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
            print(f"Error generando imagen Telegram: {e}")
            return None

    def generate_signals_optimized(self, symbol, interval, di_period=14, adx_threshold=25, 
                                 sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """Generación de señales optimizada con nuevos pesos"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular todos los indicadores
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Indicadores principales
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            adx_slope_positive = self.check_adx_slope(adx)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            breakout_up, breakout_down = self.check_breakout(high, low, close, whale_data['support'], whale_data['resistance'])
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover(ma_9, ma_21)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_bullish, macd_cross_bearish = self.check_macd_crossover(macd_line, macd_signal)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            bollinger_conditions_long = self.check_bollinger_conditions(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions(df, interval, 'SHORT')
            
            # Volumen anómalo
            volume_anomaly_data = self.calculate_volume_anomaly_improved(volume, close)
            
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
                'macd': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'macd_cross_bullish': macd_cross_bullish,
                'macd_cross_bearish': macd_cross_bearish,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'volume_anomaly': volume_anomaly_data['volume_anomaly'],
                'volume_clusters': volume_anomaly_data['volume_clusters'],
                'volume_ratio': volume_anomaly_data['volume_ratio'],
                'volume_direction': volume_anomaly_data['volume_direction'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short
            }
            
            conditions = self.evaluate_signal_conditions_optimized(analysis_data, current_idx, interval)
            
            # Calcular condición MA200
            current_ma200 = ma_200[current_idx] if current_idx < len(ma_200) else 0
            current_price = close[current_idx]
            ma200_condition = 'above' if current_price > current_ma200 else 'below'

            long_score, long_conditions = self.calculate_signal_score_optimized(conditions, 'long', ma200_condition)
            short_score, short_conditions = self.calculate_signal_score_optimized(conditions, 'short', ma200_condition)
            
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
            
            # Calcular niveles óptimos
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # Obtener patrón chartista específico si existe
            chart_pattern_text = ''
            if signal_type == 'LONG':
                if chart_patterns['double_bottom'][current_idx]:
                    chart_pattern_text = 'Doble fondo'
                elif chart_patterns['bullish_flag'][current_idx]:
                    chart_pattern_text = 'Bandera alcista'
            elif signal_type == 'SHORT':
                if chart_patterns['head_shoulders'][current_idx]:
                    chart_pattern_text = 'Hombro cabeza hombro'
                elif chart_patterns['double_top'][current_idx]:
                    chart_pattern_text = 'Doble techo'
                elif chart_patterns['bearish_flag'][current_idx]:
                    chart_pattern_text = 'Bandera bajista'
            
            if chart_pattern_text:
                # Reemplazar en fulfilled_conditions
                for i, cond in enumerate(fulfilled_conditions):
                    if cond == 'Patrón Chartista':
                        fulfilled_conditions[i] = f'Patrón Chartista: {chart_pattern_text}'
            
            # Registrar señal
            if signal_type in ['LONG', 'SHORT'] and signal_score >= 65:
                signal_key = f"{symbol}_{interval}_{signal_type}_{int(time.time())}"
                self.active_operations[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': levels_data['entry'],
                    'timestamp': self.get_bolivia_time(),
                    'score': signal_score,
                    'conditions': fulfilled_conditions
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
                    'adx_slope_positive': adx_slope_positive[-50:],
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
                    'macd': macd_line[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'macd_cross_bullish': macd_cross_bullish[-50:],
                    'macd_cross_bearish': macd_cross_bearish[-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'volume_anomaly': volume_anomaly_data['volume_anomaly'][-50:],
                    'volume_clusters': volume_anomaly_data['volume_clusters'][-50:],
                    'volume_ratio': volume_anomaly_data['volume_ratio'][-50:],
                    'volume_direction': volume_anomaly_data['volume_direction'][-50:],
                    'volume_ema': volume_anomaly_data['volume_ema'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold']
                }
            }
            
        except Exception as e:
            print(f"Error en generate_signals_optimized para {symbol}: {e}")
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
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts_optimized(self):
        """Generar alertas de trading optimizadas"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:15]:
                try:
                    signal_data = self.generate_signals_optimized(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        # Solo enviar si no se ha enviado recientemente
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

def send_telegram_alert_optimized(alert_data, strategy_type='multiframe'):
    """Enviar alerta por Telegram optimizada"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if strategy_type == 'multiframe':
            # Mensaje para estrategia Multi-TF
            conditions_text = '\n'.join([f'• {cond}' for cond in alert_data.get('fulfilled_conditions', [])])
            
            message = f"""
🚨 {alert_data['signal']} | {alert_data['symbol']} | {alert_data.get('interval', 'N/A')}
Score: {alert_data['signal_score']:.1f}%

Precio: ${alert_data.get('current_price', 0):.6f}
Entrada: ${alert_data.get('entry', 0):.6f}
MA200: {'ABOVE' if alert_data.get('ma200_condition') == 'above' else 'BELOW'}

Condiciones cumplidas:
{conditions_text}
            """
            
            # Generar imagen
            img_buffer = indicator.generate_telegram_chart_image(
                alert_data['symbol'], 
                alert_data.get('interval', '4h'),
                alert_data['signal']
            )
            
        else:
            # Mensaje para estrategia Volumen+EMA21
            message = f"""
🚨 VOL+EMA21 | {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Entrada: ${alert_data['price']:.6f} | Vol: {alert_data['volume_ratio']:.1f}x
Filtros: FTMaverick OK | MF: {alert_data['mayor_trend']}/{alert_data['menor_trend']}
            """
            
            # Generar imagen específica
            img_buffer = indicator.generate_volume_ema_telegram_image(alert_data)
        
        if img_buffer:
            asyncio.run(bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=img_buffer,
                caption=message
            ))
            print(f"Alerta {strategy_type} enviada a Telegram: {alert_data['symbol']}")
        else:
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            ))
            print(f"Alerta {strategy_type} enviada (sin imagen): {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker_optimized():
    """Verificador de alertas en segundo plano optimizado"""
    check_intervals = {
        '1h': 300,   # 5 minutos
        '4h': 420,   # 7 minutos
        '12h': 600,  # 10 minutos
        '1D': 600    # 10 minutos
    }
    
    last_checks = {interval: datetime.now() for interval in check_intervals.keys()}
    last_multiframe_check = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            # Estrategia Multi-TF (cada 60 segundos)
            if (current_time - last_multiframe_check).seconds >= 60:
                print("Verificando alertas Multi-TF...")
                
                alerts = indicator.generate_scalping_alerts_optimized()
                for alert in alerts:
                    send_telegram_alert_optimized(alert, 'multiframe')
                
                last_multiframe_check = current_time
            
            # Estrategia Volumen+EMA21
            for interval, check_interval in check_intervals.items():
                if (current_time - last_checks[interval]).seconds >= check_interval:
                    print(f"Verificando estrategia Volumen+EMA21 en {interval}...")
                    
                    for symbol in TOP10_LOW_RISK:
                        try:
                            signal = indicator.check_volume_ema_ftm_signal(symbol, interval)
                            if signal:
                                send_telegram_alert_optimized(signal, 'volume_ema')
                        except Exception as e:
                            print(f"Error verificando {symbol} {interval}: {e}")
                            continue
                    
                    last_checks[interval] = current_time
            
            time.sleep(10)
            
        except Exception as e:
            print(f"Error en background_alert_checker_optimized: {e}")
            time.sleep(60)

# Iniciar verificador de alertas en segundo plano
try:
    alert_thread = Thread(target=background_alert_checker_optimized, daemon=True)
    alert_thread.start()
    print("Background alert checker optimizado iniciado correctamente")
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
        
        signal_data = indicator.generate_signals_optimized(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, volume_filter, leverage
        )
        
        # Convertir numpy arrays a listas
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
        
        for symbol in CRYPTO_SYMBOLS[:15]:
            try:
                signal_data = indicator.generate_signals_optimized(
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

@app.route('/api/scatter_data')
def get_scatter_data():
    """Endpoint para datos del scatter plot"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        
        scatter_data = []
        
        # Analizar símbolos de cada categoría
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:5])
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_optimized(symbol, interval, di_period, adx_threshold)
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
        print(f"Error en /api/scatter_data: {e}")
        return jsonify([])

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    """Endpoint para obtener la clasificación de riesgo"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    """Endpoint para obtener alertas de trading"""
    try:
        alerts = indicator.generate_scalping_alerts_optimized()
        return jsonify({'alerts': alerts[:5]})  # Limitar a 5 alertas
        
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
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_optimized(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Crear figura con todos los gráficos
        fig = plt.figure(figsize=(14, 20))
        
        if signal_data['data']:
            dates = []
            for d in signal_data['data']:
                if isinstance(d['timestamp'], str):
                    dates.append(datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S'))
                else:
                    dates.append(d['timestamp'])
            
            closes = [d['close'] for d in signal_data['data']]
            opens = [d['open'] for d in signal_data['data']]
            highs = [d['high'] for d in signal_data['data']]
            lows = [d['low'] for d in signal_data['data']]
            volumes = [d['volume'] for d in signal_data['data']]
            
            # 1. Gráfico de Velas
            ax1 = plt.subplot(9, 1, 1)
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Soporte y resistencia
            for support in signal_data.get('supports', []):
                ax1.axhline(y=support, color='blue', linestyle='--', alpha=0.5)
            for resistance in signal_data.get('resistances', []):
                ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5)
            
            ax1.set_title(f'{symbol} - Reporte Técnico ({interval})', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Precio (USDT)')
            ax1.grid(True, alpha=0.3)
            
            # 2. ADX con DMI
            ax2 = plt.subplot(9, 1, 2, sharex=ax1)
            if 'indicators' in signal_data:
                adx_dates = dates[-len(signal_data['indicators']['adx']):]
                ax2.plot(adx_dates, signal_data['indicators']['adx'], 
                        'black', linewidth=2, label='ADX')
                ax2.plot(adx_dates, signal_data['indicators']['plus_di'], 
                        'green', linewidth=1, label='+DI')
                ax2.plot(adx_dates, signal_data['indicators']['minus_di'], 
                        'red', linewidth=1, label='-DI')
            ax2.set_ylabel('ADX/DMI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Volumen con Anomalías
            ax3 = plt.subplot(9, 1, 3, sharex=ax1)
            if 'indicators' in signal_data:
                volume_dates = dates[-len(signal_data['indicators']['volume_anomaly']):]
                
                # Barras de volumen coloreadas
                bar_colors = []
                for i in range(len(volume_dates)):
                    if signal_data['indicators']['volume_anomaly'][i]:
                        if signal_data['indicators']['volume_direction'][i] == 1:
                            bar_colors.append('green')
                        else:
                            bar_colors.append('red')
                    else:
                        bar_colors.append('gray')
                
                ax3.bar(volume_dates, volumes[-len(volume_dates):], color=bar_colors, alpha=0.7)
                ax3.plot(volume_dates, signal_data['indicators']['volume_ema'], 
                        'yellow', linewidth=1, label='EMA Volumen')
            
            ax3.set_ylabel('Volumen')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Fuerza de Tendencia Maverick
            ax4 = plt.subplot(9, 1, 4, sharex=ax1)
            if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
                trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
                
                bar_colors_ftm = []
                for i in range(len(trend_dates)):
                    if signal_data['indicators']['no_trade_zones'][i]:
                        bar_colors_ftm.append('red')
                    elif signal_data['indicators']['trend_strength'][i] > 0:
                        bar_colors_ftm.append('green')
                    else:
                        bar_colors_ftm.append('red')
                
                ax4.bar(trend_dates, signal_data['indicators']['trend_strength'], 
                       color=bar_colors_ftm, alpha=0.7)
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax4.set_ylabel('FT Maverick')
            ax4.grid(True, alpha=0.3)
            
            # 5. Indicador de Ballenas
            ax5 = plt.subplot(9, 1, 5, sharex=ax1)
            if 'indicators' in signal_data:
                whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
                ax5.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                       color='green', alpha=0.7, label='Compra')
                ax5.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                       color='red', alpha=0.7, label='Venta')
            ax5.set_ylabel('Ballenas')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. RSI Maverick
            ax6 = plt.subplot(9, 1, 6, sharex=ax1)
            if 'indicators' in signal_data:
                rsi_m_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
                ax6.plot(rsi_m_dates, signal_data['indicators']['rsi_maverick'], 
                        'blue', linewidth=1)
                ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
                ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax6.set_ylabel('RSI Maverick')
            ax6.grid(True, alpha=0.3)
            
            # 7. RSI Tradicional
            ax7 = plt.subplot(9, 1, 7, sharex=ax1)
            if 'indicators' in signal_data:
                rsi_t_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
                ax7.plot(rsi_t_dates, signal_data['indicators']['rsi_traditional'], 
                        'cyan', linewidth=1)
                ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7)
                ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            ax7.set_ylabel('RSI Trad')
            ax7.grid(True, alpha=0.3)
            
            # 8. MACD
            ax8 = plt.subplot(9, 1, 8, sharex=ax1)
            if 'indicators' in signal_data:
                macd_dates = dates[-len(signal_data['indicators']['macd']):]
                ax8.plot(macd_dates, signal_data['indicators']['macd'], 
                        'blue', linewidth=1, label='MACD')
                ax8.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                        'red', linewidth=1, label='Señal')
                
                # Histograma como barras
                bar_colors_macd = []
                for val in signal_data['indicators']['macd_histogram']:
                    bar_colors_macd.append('green' if val >= 0 else 'red')
                
                ax8.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                       color=bar_colors_macd, alpha=0.7, label='Histograma')
            
            ax8.set_ylabel('MACD')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            
            # 9. Información de la señal
            ax9 = plt.subplot(9, 1, 9)
            ax9.axis('off')
            
            conditions_text = '\n'.join([f'• {cond}' for cond in signal_data.get('fulfilled_conditions', [])])
            
            signal_info = f"""
            SEÑAL: {signal_data['signal']}
            SCORE: {signal_data['signal_score']:.1f}%
            
            PRECIO: ${signal_data['current_price']:.6f}
            ENTRADA: ${signal_data['entry']:.6f}
            STOP LOSS: ${signal_data['stop_loss']:.6f}
            TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
            
            MA200: {'ENCIMA' if signal_data.get('ma200_condition') == 'above' else 'DEBAJO'}
            
            CONDICIONES CUMPLIDAS:
            {conditions_text}
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
