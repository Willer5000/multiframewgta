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
        self.volume_alert_cache = set()  # Para evitar duplicados de alertas de volumen
        self.sent_volume_alerts = set()  # Para alertas de volumen enviadas
    
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela - OPTIMIZADO"""
        interval_map = {
            '15m': (15, 0.5, 60),     # 15min, 50%, 60s revisión
            '30m': (30, 0.5, 60),     # 30min, 50%, 60s
            '1h': (60, 0.5, 300),     # 1h, 50%, 300s
            '2h': (120, 0.5, 300),    # 2h, 50%, 300s
            '4h': (240, 0.25, 600),   # 4h, 25%, 600s
            '8h': (480, 0.25, 600),   # 8h, 25%, 600s
            '12h': (720, 0.25, 600),  # 12h, 25%, 600s
            '1D': (1440, 0.25, 600),  # 1D, 25%, 600s
            '1W': (10080, 0.1, 1800)  # 1W, 10%, 1800s
        }
        
        if interval not in interval_map:
            return False
            
        interval_minutes, threshold_percent, check_interval = interval_map[interval]
        
        # Calcular tiempo transcurrido en la vela actual
        if interval == '1W':
            # Para semanas, usar inicio de semana (lunes)
            week_start = current_time - timedelta(days=current_time.weekday())
            elapsed = (current_time - week_start).total_seconds()
            total_seconds = 7 * 24 * 60 * 60
        elif interval == '1D':
            # Para días, usar inicio del día
            day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            elapsed = (current_time - day_start).total_seconds()
            total_seconds = 24 * 60 * 60
        else:
            # Para intervalos intradía
            total_seconds = interval_minutes * 60
            elapsed = (current_time.minute * 60 + current_time.second) % total_seconds
        
        # Verificar si estamos en el porcentaje umbral antes del cierre
        remaining_percent = 1 - (elapsed / total_seconds)
        return remaining_percent <= threshold_percent

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
                        return self.generate_sample_data(limit, interval, symbol)
                    
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
        
        return self.generate_sample_data(limit, interval, symbol)

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
        """Calcular entradas y salidas óptimas mejoradas"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            support_1 = np.min(low[-50:])
            resistance_1 = np.max(high[-50:])

            if signal_type == 'LONG':
                entry = min(current_price, support_1 * 1.01)
                stop_loss = max(support_1 * 0.97, entry - (current_atr * 1.8))
                tp1 = resistance_1 * 0.98
                
                min_tp = entry + (2 * (entry - stop_loss))
                tp1 = max(tp1, min_tp)
                
            else:
                entry = max(current_price, resistance_1 * 0.99)
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
                'atr_percentage': float(current_atr / current_price if current_price > 0 else 0)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support': current_price * 0.95,
                'resistance': current_price * 1.05,
                'atr': current_price * 0.02,
                'atr_percentage': 0.02
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
                'resistance': current_resistance.tolist()
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
                'resistance': df['high'].values.tolist()
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
        
        return bullish_div.tolist(), bearish_div.tolist()

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
            
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
        
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
            'double_bottom': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-7):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            
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
        
        return patterns

    def calculate_volume_anomaly_improved(self, volume, close, open_prices, period=20, std_multiplier=2):
        """Calcular anomalías de volumen mejorado - con detección de compra/venta"""
        try:
            n = len(volume)
            volume_anomaly_buy = np.zeros(n, dtype=bool)
            volume_anomaly_sell = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            
            for i in range(period, n):
                ema_volume = self.calculate_ema(volume[:i+1], period)
                current_ema = ema_volume[i] if i < len(ema_volume) else volume[i]
                
                window = volume[max(0, i-period+1):i+1]
                std_volume = np.std(window) if len(window) > 1 else 0
                
                if current_ema > 0:
                    volume_ratio[i] = volume[i] / current_ema
                else:
                    volume_ratio[i] = 1
                
                # Detectar anomalía (> 2σ)
                if volume_ratio[i] > 1 + (std_multiplier * (std_volume / current_ema if current_ema > 0 else 0)):
                    # Determinar si es compra o venta basado en el precio
                    if close[i] > open_prices[i]:
                        volume_anomaly_buy[i] = True  # Volumen de compra
                    else:
                        volume_anomaly_sell[i] = True  # Volumen de venta
                
                # Detectar clusters (múltiples anomalías consecutivas)
                if i >= 5:
                    recent_buy = volume_anomaly_buy[max(0, i-4):i+1]
                    recent_sell = volume_anomaly_sell[max(0, i-4):i+1]
                    
                    if np.sum(recent_buy) >= 1 or np.sum(recent_sell) >= 1:
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly_buy': volume_anomaly_buy.tolist(),
                'volume_anomaly_sell': volume_anomaly_sell.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ema': ema_volume.tolist() if 'ema_volume' in locals() else [0] * n
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly_improved: {e}")
            n = len(volume)
            return {
                'volume_anomaly_buy': [False] * n,
                'volume_anomaly_sell': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ema': [0] * n
            }

    def evaluate_signal_conditions_corrected(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con PESOS CORREGIDOS"""
        weights = {
            'long': {
                'multi_timeframe': 30,
                'trend_strength': 25,
                'volume_anomaly': 20,
                'bollinger_bands': 10,
                'adx_dmi': 10,
                'rsi_traditional_divergence': 5,
                'rsi_maverick_divergence': 5,
                'macd': 5,
                'chart_pattern': 5,
                'breakout': 5
            },
            'short': {
                'multi_timeframe': 30,
                'trend_strength': 25,
                'volume_anomaly': 20,
                'bollinger_bands': 10,
                'adx_dmi': 10,
                'rsi_traditional_divergence': 5,
                'rsi_maverick_divergence': 5,
                'macd': 5,
                'chart_pattern': 5,
                'breakout': 5
            }
        }
        
        conditions = {
            'long': {},
            'short': {}
        }
        
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
        
        # Condiciones LONG
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_long', False)
        
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        conditions['long']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly_buy']) and 
            (data['volume_anomaly_buy'][current_idx] or data['volume_clusters'][current_idx])
        )
        
        conditions['long']['bollinger_bands']['value'] = data.get('bollinger_conditions_long', False)
        conditions['long']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['plus_di'][current_idx] > data['minus_di'][current_idx]
        )
        
        if current_idx < len(data['rsi_bullish_divergence']):
            conditions['long']['rsi_traditional_divergence']['value'] = data['rsi_bullish_divergence'][current_idx]
        
        if current_idx < len(data['rsi_maverick_bullish_divergence']):
            conditions['long']['rsi_maverick_divergence']['value'] = data['rsi_maverick_bullish_divergence'][current_idx]
        
        conditions['long']['macd']['value'] = (
            data['macd'][current_idx] > data['macd_signal'][current_idx] and
            data['macd_histogram'][current_idx] > 0
        )
        
        if data['chart_patterns']['double_bottom'][current_idx]:
            conditions['long']['chart_pattern']['value'] = True
        
        if current_idx < len(data['breakout_up']):
            conditions['long']['breakout']['value'] = data['breakout_up'][current_idx]
        
        # Condiciones SHORT
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        
        conditions['short']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            not data['no_trade_zones'][current_idx]
        )
        
        conditions['short']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly_sell']) and 
            (data['volume_anomaly_sell'][current_idx] or data['volume_clusters'][current_idx])
        )
        
        conditions['short']['bollinger_bands']['value'] = data.get('bollinger_conditions_short', False)
        conditions['short']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['minus_di'][current_idx] > data['plus_di'][current_idx]
        )
        
        if current_idx < len(data['rsi_bearish_divergence']):
            conditions['short']['rsi_traditional_divergence']['value'] = data['rsi_bearish_divergence'][current_idx]
        
        if current_idx < len(data['rsi_maverick_bearish_divergence']):
            conditions['short']['rsi_maverick_divergence']['value'] = data['rsi_maverick_bearish_divergence'][current_idx]
        
        conditions['short']['macd']['value'] = (
            data['macd'][current_idx] < data['macd_signal'][current_idx] and
            data['macd_histogram'][current_idx] < 0
        )
        
        if data['chart_patterns']['head_shoulders'][current_idx] or data['chart_patterns']['double_top'][current_idx]:
            conditions['short']['chart_pattern']['value'] = True
        
        if current_idx < len(data['breakout_down']):
            conditions['short']['breakout']['value'] = data['breakout_down'][current_idx]
        
        return conditions

    def get_condition_description(self, condition_key):
        """Obtener descripción de condición"""
        descriptions = {
            'multi_timeframe': 'Condiciones Multi-TF obligatorias',
            'trend_strength': 'Fuerza tendencia favorable',
            'volume_anomaly': 'Anomalía/Cluster de Volumen',
            'bollinger_bands': 'Bandas de Bollinger',
            'adx_dmi': 'ADX + DMI',
            'rsi_traditional_divergence': 'RSI Tradicional Divergencia',
            'rsi_maverick_divergence': 'RSI Maverick Divergencia',
            'macd': 'MACD',
            'chart_pattern': 'Patrones Chart',
            'breakout': 'Breakout'
        }
        return descriptions.get(condition_key, condition_key)

    def calculate_signal_score(self, conditions, signal_type):
        """Calcular puntuación de señal basada en condiciones ponderadas"""
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
        
        base_score = (achieved_weight / total_weight * 100)
        
        min_score = 65
        final_score = base_score if base_score >= min_score else 0

        return min(final_score, 100), fulfilled_conditions

    def generate_volume_anomaly_signals(self):
        """Generar señales basadas en volumen anómalo para Telegram"""
        signals = []
        current_time = self.get_bolivia_time()
        
        for interval in TIMEFRAME_HIERARCHY.keys():
            if not self.calculate_remaining_time(interval, current_time):
                continue
                
            for symbol in CRYPTO_SYMBOLS[:8]:  # Limitar para no sobrecargar
                try:
                    # Obtener datos de temporalidad actual
                    df_current = self.get_kucoin_data(symbol, interval, 50)
                    if df_current is None or len(df_current) < 30:
                        continue
                    
                    # Obtener datos de temporalidad menor
                    hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                    if not hierarchy or 'menor' not in hierarchy:
                        continue
                        
                    menor_interval = hierarchy['menor']
                    df_menor = self.get_kucoin_data(symbol, menor_interval, 30)
                    if df_menor is None or len(df_menor) < 20:
                        continue
                    
                    # Calcular volumen anómalo en temporalidad menor
                    volume_data = self.calculate_volume_anomaly_improved(
                        df_menor['volume'].values,
                        df_menor['close'].values,
                        df_menor['open'].values
                    )
                    
                    current_idx = -1
                    
                    # Verificar condiciones para LONG
                    if volume_data['volume_anomaly_buy'][current_idx] or volume_data['volume_clusters'][current_idx]:
                        # Verificar condiciones FTMaverick y Multi-Timeframe
                        menor_trend = self.calculate_trend_strength_maverick(df_menor['close'].values)
                        current_trend = self.calculate_trend_strength_maverick(df_current['close'].values)
                        
                        # Condiciones para LONG
                        if (not menor_trend['no_trade_zones'][current_idx] and
                            not current_trend['no_trade_zones'][current_idx] and
                            menor_trend['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                            current_trend['strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP', 'NEUTRAL']):
                            
                            risk_category = self.get_risk_category(symbol)
                            signal_id = f"VOL_{symbol}_{interval}_LONG_{int(time.time())}"
                            
                            if signal_id not in self.volume_alert_cache:
                                signals.append({
                                    'type': 'LONG',
                                    'symbol': symbol,
                                    'risk_category': risk_category,
                                    'volume_type': 'anomalos de compra' if volume_data['volume_anomaly_buy'][current_idx] else 'cluster de volumenes',
                                    'interval': interval,
                                    'menor_interval': menor_interval,
                                    'conditions': [
                                        'Fuera de Zona No Operar',
                                        f'Temporalidad menor ({menor_interval}) con tendencia alcista',
                                        f'Temporalidad actual ({interval}) tendencia Neutral o alcista'
                                    ],
                                    'signal_id': signal_id
                                })
                                self.volume_alert_cache.add(signal_id)
                    
                    # Verificar condiciones para SHORT
                    if volume_data['volume_anomaly_sell'][current_idx] or volume_data['volume_clusters'][current_idx]:
                        # Verificar condiciones FTMaverick y Multi-Timeframe
                        menor_trend = self.calculate_trend_strength_maverick(df_menor['close'].values)
                        current_trend = self.calculate_trend_strength_maverick(df_current['close'].values)
                        
                        # Condiciones para SHORT
                        if (not menor_trend['no_trade_zones'][current_idx] and
                            not current_trend['no_trade_zones'][current_idx] and
                            menor_trend['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                            current_trend['strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL']):
                            
                            risk_category = self.get_risk_category(symbol)
                            signal_id = f"VOL_{symbol}_{interval}_SHORT_{int(time.time())}"
                            
                            if signal_id not in self.volume_alert_cache:
                                signals.append({
                                    'type': 'SHORT',
                                    'symbol': symbol,
                                    'risk_category': risk_category,
                                    'volume_type': 'anomalos de venta' if volume_data['volume_anomaly_sell'][current_idx] else 'cluster de volumenes',
                                    'interval': interval,
                                    'menor_interval': menor_interval,
                                    'conditions': [
                                        'Fuera de Zona No Operar',
                                        f'Temporalidad menor ({menor_interval}) con tendencia bajista',
                                        f'Temporalidad actual ({interval}) tendencia Neutral o bajista'
                                    ],
                                    'signal_id': signal_id
                                })
                                self.volume_alert_cache.add(signal_id)
                    
                except Exception as e:
                    print(f"Error generando señal de volumen para {symbol} {interval}: {e}")
                    continue
        
        return signals

    def get_risk_category(self, symbol):
        """Obtener categoría de riesgo de una criptomoneda"""
        for category, symbols in CRYPTO_RISK_CLASSIFICATION.items():
            if symbol in symbols:
                if category == 'bajo':
                    return 'Bajo riesgo'
                elif category == 'medio':
                    return 'Medio riesgo'
                elif category == 'alto':
                    return 'Alto riesgo'
                elif category == 'memecoins':
                    return 'Memecoin'
        return 'Medio riesgo'

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
            open_prices = df['open'].values
            
            # Calcular indicadores
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
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
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Verificar condiciones de Bollinger
            bollinger_conditions_long = self.check_bollinger_conditions_corrected(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions_corrected(df, interval, 'SHORT')
            
            # Nuevo indicador de volumen mejorado
            volume_data = self.calculate_volume_anomaly_improved(volume, close, open_prices)
            
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
                'volume_anomaly_buy': volume_data['volume_anomaly_buy'],
                'volume_anomaly_sell': volume_data['volume_anomaly_sell'],
                'volume_clusters': volume_data['volume_clusters'],
                'volume_ratio': volume_data['volume_ratio'],
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

            long_score, long_conditions = self.calculate_signal_score(conditions, 'long')
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short')
            
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            # Ajustar score mínimo según posición respecto a MA200
            min_score_long = 65 if ma200_condition == 'above' else 70
            min_score_short = 65 if ma200_condition == 'below' else 70
            
            if long_score >= min_score_long and long_score > short_score:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= min_score_short and short_score > long_score:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # Calcular niveles de trading
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
                'winrate': float(65.0),  # Winrate fijo para simplificar
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
                    'volume_anomaly_buy': volume_data['volume_anomaly_buy'][-50:],
                    'volume_anomaly_sell': volume_data['volume_anomaly_sell'][-50:],
                    'volume_clusters': volume_data['volume_clusters'][-50:],
                    'volume_ratio': volume_data['volume_ratio'][-50:],
                    'volume_ema': volume_data['volume_ema'][-50:],
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
        
        for interval in TIMEFRAME_HIERARCHY.keys():
            if not self.calculate_remaining_time(interval, current_time):
                continue
                
            for symbol in CRYPTO_SYMBOLS[:12]:
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        risk_category = self.get_risk_category(symbol)
                        
                        alert = {
                            'symbol': symbol,
                            'interval': interval,
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'winrate': signal_data['winrate'],
                            'entry': signal_data['entry'],
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'][0],
                            'leverage': 15,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'support': signal_data['support'],
                            'resistance': signal_data['resistance'],
                            'ma200_condition': signal_data.get('ma200_condition', 'below')
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
    """Enviar alerta por Telegram - VERSIÓN SIMPLIFICADA"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'entry':
            message = f"""
🚨 SEÑAL CONFIRMADA - MULTI-TIMEFRAME WGTA PRO 🚨

Crypto: {alert_data['symbol']} ({alert_data.get('risk_category', 'Medio riesgo')})
Señal: {alert_data['signal']}
Temporalidad: {alert_data['interval']}
Score: {alert_data['score']:.1f}%

Precio: {alert_data['current_price']:.6f}
Entrada: {alert_data['entry']:.6f}
Stop Loss: {alert_data['stop_loss']:.6f}
Take Profit: {alert_data['take_profit']:.6f}

Condiciones cumplidas:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])])}
"""
            
        elif alert_type == 'volume':
            message = f"""
📊 ALERTA VOLUMEN ANÓMALO - MULTI-TIMEFRAME WGTA PRO 📊

Crypto: {alert_data['symbol']} ({alert_data['risk_category']})
Señal: {alert_data['type']}
Volumen: {alert_data['volume_type']} en temporalidad menor
Temporalidad: {alert_data['interval']}

Condiciones FTMaverick y MF:
{chr(10).join(['• ' + cond for cond in alert_data['conditions']])}

Recomendación: revisar {alert_data['type']}
"""
        
        report_url = f"https://multiframewgta.onrender.com/api/generate_report?symbol={alert_data['symbol']}&interval={alert_data['interval']}&leverage={alert_data.get('leverage', 15)}"
        
        keyboard = [[telegram.InlineKeyboardButton("📊 Descargar Reporte", url=report_url)]]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        asyncio.run(bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=message,
            reply_markup=reply_markup
        ))
        print(f"Alerta {alert_type} enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            current_time = datetime.now()
            
            # Generar alertas de volumen anómalo
            volume_signals = indicator.generate_volume_anomaly_signals()
            for signal in volume_signals:
                signal_id = signal['signal_id']
                if signal_id not in indicator.sent_volume_alerts:
                    send_telegram_alert(signal, 'volume')
                    indicator.sent_volume_alerts.add(signal_id)
            
            # Generar alertas de señales confirmadas
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                alert_key = f"{alert['symbol']}_{alert['interval']}_{alert['signal']}"
                if alert_key not in indicator.alert_cache:
                    send_telegram_alert(alert, 'entry')
                    indicator.alert_cache[alert_key] = current_time
            
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
                        'risk_category': indicator.get_risk_category(symbol)
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
        
        # Preparar datos para el gráfico
        if signal_data['data']:
            dates = [datetime.strptime(str(d['timestamp']), '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            opens = [d['open'] for d in signal_data['data']]
            highs = [d['high'] for d in signal_data['data']]
            lows = [d['low'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            volumes = [d['volume'] for d in signal_data['data']]
        else:
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
        
        fig = plt.figure(figsize=(14, 18))
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(9, 1, 1)
        if dates and len(dates) > 0:
            for i in range(len(dates)):
                if i < len(opens) and i < len(closes) and i < len(highs) and i < len(lows):
                    color = 'green' if closes[i] >= opens[i] else 'red'
                    ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                    ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            if signal_data['entry'] > 0:
                ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            if signal_data['stop_loss'] > 0:
                ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            if signal_data['take_profit'] and len(signal_data['take_profit']) > 0:
                for i, tp in enumerate(signal_data['take_profit']):
                    ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
            
            if signal_data['support'] > 0:
                ax1.axhline(y=signal_data['support'], color='orange', linestyle=':', alpha=0.5, label='Soporte')
            if signal_data['resistance'] > 0:
                ax1.axhline(y=signal_data['resistance'], color='purple', linestyle=':', alpha=0.5, label='Resistencia')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        if dates:
            ax1.set_xlim([dates[0], dates[-1]])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: ADX/DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data and 'adx' in signal_data['indicators']:
            adx_data = signal_data['indicators']['adx']
            plus_di_data = signal_data['indicators']['plus_di']
            minus_di_data = signal_data['indicators']['minus_di']
            
            if len(dates) >= len(adx_data):
                adx_dates = dates[-len(adx_data):]
                ax2.plot(adx_dates, adx_data, 'white', linewidth=2, label='ADX')
                ax2.plot(adx_dates, plus_di_data, 'green', linewidth=1, label='+DI')
                ax2.plot(adx_dates, minus_di_data, 'red', linewidth=1, label='-DI')
                ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        
        ax2.set_ylabel('ADX/DMI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Volumen y Anomalías
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if volumes and len(volumes) > 0:
            if len(dates) >= len(volumes):
                volume_dates = dates[-len(volumes):]
                
                # Colores del volumen basados en precio
                volume_colors = []
                for i in range(len(volume_dates)):
                    if i < len(closes) and i < len(opens):
                        color = 'green' if closes[i] >= opens[i] else 'red'
                        volume_colors.append(color)
                    else:
                        volume_colors.append('gray')
                
                ax3.bar(volume_dates, volumes, color=volume_colors, alpha=0.6, label='Volumen')
                
                if 'indicators' in signal_data and 'volume_ema' in signal_data['indicators']:
                    volume_ema = signal_data['indicators']['volume_ema']
                    if len(volume_dates) >= len(volume_ema):
                        ema_dates = volume_dates[-len(volume_ema):]
                        ax3.plot(ema_dates, volume_ema, 'yellow', linewidth=1, label='EMA Volumen')
        
        ax3.set_ylabel('Volumen')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Fuerza de Tendencia Maverick
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            if len(dates) >= len(trend_strength):
                trend_dates = dates[-len(trend_strength):]
                for i in range(len(trend_dates)):
                    if i < len(colors):
                        color = colors[i]
                    else:
                        color = 'gray'
                    ax4.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
        
        ax4.set_ylabel('Fuerza Tendencia %')
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Ballenas
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data and 'whale_pump' in signal_data['indicators']:
            whale_pump = signal_data['indicators']['whale_pump']
            whale_dump = signal_data['indicators']['whale_dump']
            
            if len(dates) >= len(whale_pump):
                whale_dates = dates[-len(whale_pump):]
                ax5.bar(whale_dates, whale_pump, color='green', alpha=0.7, label='Ballenas Compradoras')
                ax5.bar(whale_dates, whale_dump, color='red', alpha=0.7, label='Ballenas Vendedoras')
        
        ax5.set_ylabel('Fuerza Ballenas')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: RSI Maverick
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_maverick' in signal_data['indicators']:
            rsi_maverick = signal_data['indicators']['rsi_maverick']
            
            if len(dates) >= len(rsi_maverick):
                rsi_dates = dates[-len(rsi_maverick):]
                ax6.plot(rsi_dates, rsi_maverick, 'blue', linewidth=2, label='RSI Maverick')
                ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
                ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
                ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        
        ax6.set_ylabel('RSI Maverick')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: RSI Tradicional
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_traditional' in signal_data['indicators']:
            rsi_traditional = signal_data['indicators']['rsi_traditional']
            
            if len(dates) >= len(rsi_traditional):
                rsi_dates = dates[-len(rsi_traditional):]
                ax7.plot(rsi_dates, rsi_traditional, 'cyan', linewidth=2, label='RSI Tradicional')
                ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
                ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
                ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        ax7.set_ylabel('RSI Tradicional')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: MACD
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if 'indicators' in signal_data and 'macd' in signal_data['indicators']:
            macd_data = signal_data['indicators']['macd']
            macd_signal_data = signal_data['indicators']['macd_signal']
            macd_histogram = signal_data['indicators']['macd_histogram']
            
            if len(dates) >= len(macd_data):
                macd_dates = dates[-len(macd_data):]
                ax8.plot(macd_dates, macd_data, 'blue', linewidth=1, label='MACD')
                ax8.plot(macd_dates, macd_signal_data, 'red', linewidth=1, label='Señal')
                
                colors = ['green' if x > 0 else 'red' for x in macd_histogram]
                ax8.bar(macd_dates, macd_histogram, color=colors, alpha=0.6, label='Histograma')
                
                ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax8.set_ylabel('MACD')
        ax8.legend()
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

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
