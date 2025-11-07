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
TELEGRAM_BOT_TOKEN = "8449034982:AAFzXDKCtg51OH45SERkg-QkEBAHFv3z-n4"
TELEGRAM_CHAT_ID = "-1003127538942"

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

# Mapeo de temporalidades para análisis multi-timeframe
TIMEFRAME_HIERARCHY = {
    '15m': {'mayor': '1h', 'media': '30m', 'menor': '5m'},
    '30m': {'mayor': '2h', 'media': '1h', 'menor': '15m'},
    '1h': {'mayor': '4h', 'media': '2h', 'menor': '30m'},
    '2h': {'mayor': '8h', 'media': '4h', 'menor': '1h'},
    '4h': {'mayor': '12h', 'media': '8h', 'menor': '2h'},
    '8h': {'mayor': '1D', 'media': '12h', 'menor': '4h'},
    '12h': {'mayor': '1D', 'media': '8h', 'menor': '4h'},
    '1D': {'mayor': '3D', 'media': '1D', 'menor': '12h'}
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_signals = {}
        self.btc_dominance_data = None
        self.last_btc_update = None
        self.bolivia_tz = pytz.timezone('America/La_Paz')
    
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
        try:
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
        except Exception as e:
            print(f"Error calculando tiempo restante: {e}")
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
                '1D': '1day', '3D': '3day'
            }
            
            kucoin_interval = interval_map.get(interval, '1hour')
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol.replace('-', '')}&type={kucoin_interval}"
            
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
        try:
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
        except Exception as e:
            print(f"Error generando datos de muestra: {e}")
            return pd.DataFrame()

    def calculate_atr(self, high, low, close, period=14):
        """Calcular Average True Range"""
        try:
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
        except Exception as e:
            print(f"Error calculando ATR: {e}")
            return np.zeros_like(high)

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas mejoradas"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1] if len(close) > 0 else 0
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            support_1 = np.min(low[-20:]) if len(low) >= 20 else np.min(low)
            resistance_1 = np.max(high[-20:]) if len(high) >= 20 else np.max(high)
            
            atr_percentage = current_atr / current_price if current_price > 0 else 0

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
            current_price = float(df['close'].iloc[-1]) if len(df) > 0 else 0
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support': current_price * 0.95,
                'resistance': current_price * 1.05,
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente con validación"""
        try:
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
        except Exception as e:
            print(f"Error calculando EMA: {e}")
            return np.zeros_like(prices)

    def calculate_sma(self, prices, period):
        """Calcular SMA manualmente con validación"""
        try:
            if len(prices) == 0 or period <= 0:
                return np.zeros_like(prices)
                
            sma = np.zeros_like(prices)
            for i in range(len(prices)):
                start_idx = max(0, i - period + 1)
                window = prices[start_idx:i+1]
                valid_values = window[~np.isnan(window)]
                sma[i] = np.mean(valid_values) if len(valid_values) > 0 else (prices[i] if i < len(prices) and not np.isnan(prices[i]) else 0)
            
            return sma
        except Exception as e:
            print(f"Error calculando SMA: {e}")
            return np.zeros_like(prices)

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger manualmente - MÉTODO AÑADIDO"""
        try:
            if len(prices) < period:
                return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
            
            sma = self.calculate_sma(prices, period)
            std = np.zeros_like(prices)
            
            for i in range(len(prices)):
                if i >= period - 1:
                    window = prices[i-period+1:i+1]
                    valid_window = window[~np.isnan(window)]
                    if len(valid_window) > 0:
                        std[i] = np.std(valid_window)
                    else:
                        std[i] = 0
                else:
                    std[i] = 0
            
            upper = sma + (std * multiplier)
            lower = sma - (std * multiplier)
            
            return upper, sma, lower
        except Exception as e:
            print(f"Error calculando Bollinger Bands: {e}")
            n = len(prices)
            return np.zeros(n), np.zeros(n), np.zeros(n)

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional manualmente"""
        try:
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
        except Exception as e:
            print(f"Error calculando RSI: {e}")
            return np.zeros_like(prices)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcular MACD manualmente"""
        try:
            if len(prices) < slow:
                return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
            
            ema_fast = self.calculate_ema(prices, fast)
            ema_slow = self.calculate_ema(prices, slow)
            
            macd_line = ema_fast - ema_slow
            signal_line = self.calculate_ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except Exception as e:
            print(f"Error calculando MACD: {e}")
            n = len(prices)
            return np.zeros(n), np.zeros(n), np.zeros(n)

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick"""
        try:
            n = len(close)
            
            basis = self.calculate_sma(close, length)
            dev = np.zeros(n)
            
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                valid_window = window[~np.isnan(window)]
                dev[i] = np.std(valid_window) if len(valid_window) > 1 else 0
            
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
                
                current_ma_9 = ma_9[-1] if len(ma_9) > 0 else 0
                current_ma_21 = ma_21[-1] if len(ma_21) > 0 else 0
                current_price = close[-1]
                
                if current_price > current_ma_9 and current_ma_9 > current_ma_21:
                    results[tf_type] = 'BULLISH'
                elif current_price < current_ma_9 and current_ma_9 < current_ma_21:
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}

    def apply_trend_strength_filter(self, signal_type, trend_strength_data, current_idx):
        """Aplicar filtro de fuerza de tendencia a las señales"""
        try:
            if current_idx < 0 or current_idx >= len(trend_strength_data['strength_signals']):
                return True
            
            current_signal = trend_strength_data['strength_signals'][current_idx]
            current_no_trade = trend_strength_data['no_trade_zones'][current_idx]
            
            if current_no_trade:
                return False
            
            if signal_type == 'LONG':
                return current_signal in ['STRONG_UP', 'WEAK_UP']
            elif signal_type == 'SHORT':
                return current_signal in ['STRONG_DOWN', 'WEAK_DOWN']
            
            return True
        except Exception as e:
            print(f"Error aplicando filtro de tendencia: {e}")
            return True

    def calculate_whale_signals_improved(self, df, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=20, signal_threshold=25, 
                                       sell_signal_threshold=20):
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
            
            sma_20 = self.calculate_sma(close, 20)
            sma_50 = self.calculate_sma(close, 50)
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100 if close[i-1] > 0 else 0
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
                
                if (whale_dump_smooth[i] > sell_signal_threshold and 
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
                'support': df['low'].values.tolist() if len(df) > 0 else [0] * n,
                'resistance': df['high'].values.tolist() if len(df) > 0 else [0] * n,
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

    def detect_divergence(self, price, indicator, lookback=10):
        """Detectar divergencias entre precio e indicador"""
        try:
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
        except Exception as e:
            print(f"Error detectando divergencias: {e}")
            n = len(price)
            return [False] * n, [False] * n

    def check_breakout(self, high, low, close, support, resistance):
        """Detectar rupturas de tendencia"""
        try:
            n = len(close)
            breakout_up = np.zeros(n, dtype=bool)
            breakout_down = np.zeros(n, dtype=bool)
            
            for i in range(1, n):
                if close[i] > resistance[i] and high[i] > high[i-1]:
                    breakout_up[i] = True
                
                if close[i] < support[i] and low[i] < low[i-1]:
                    breakout_down[i] = True
            
            return breakout_up.tolist(), breakout_down.tolist()
        except Exception as e:
            print(f"Error detectando breakouts: {e}")
            n = len(close)
            return [False] * n, [False] * n

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI con confirmación"""
        try:
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
        except Exception as e:
            print(f"Error detectando cruces DI: {e}")
            n = len(plus_di)
            return [False] * n, [False] * n, [False] * n, [False] * n

    def calculate_adx(self, high, low, close, period=14):
        """Calcular ADX, +DI, -DI manualmente con validación"""
        try:
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
        except Exception as e:
            print(f"Error calculando ADX: {e}")
            n = len(high)
            return np.zeros(n), np.zeros(n), np.zeros(n)

    def evaluate_signal_conditions_improved(self, data, current_idx, adx_threshold=20):
        """Evaluar condiciones de señal con lógica corregida"""
        try:
            conditions = {
                'long': {
                    'whale_pump': {'value': False, 'weight': 25, 'description': 'Ballena compradora activa'},
                    'di_cross_bullish': {'value': False, 'weight': 25, 'description': '+DI cruza -DI positivamente'},
                    'di_trend_bullish': {'value': False, 'weight': 15, 'description': '+DI con tendencia positiva'},
                    'divergence': {'value': False, 'weight': 15, 'description': 'Divergencia alcista en RSI Maverick'},
                    'breakout': {'value': False, 'weight': 10, 'description': 'Ruptura de tendencia alcista'},
                    'volume_anomaly': {'value': False, 'weight': 10, 'description': 'Volumen superior al promedio'},
                    'adx': {'value': False, 'weight': 5, 'description': f'ADX > {adx_threshold}'},
                    'trend_strength': {'value': False, 'weight': 5, 'description': 'Fuerza de tendencia favorable'}
                },
                'short': {
                    'whale_dump': {'value': False, 'weight': 25, 'description': 'Ballena vendedora activa'},
                    'di_cross_bearish': {'value': False, 'weight': 25, 'description': '-DI cruza +DI positivamente'},
                    'di_trend_bearish': {'value': False, 'weight': 15, 'description': '-DI con tendencia positiva'},
                    'divergence': {'value': False, 'weight': 15, 'description': 'Divergencia bajista en RSI Maverick'},
                    'breakout': {'value': False, 'weight': 10, 'description': 'Ruptura de tendencia bajista'},
                    'volume_anomaly': {'value': False, 'weight': 10, 'description': 'Volumen superior al promedio'},
                    'adx': {'value': False, 'weight': 5, 'description': f'ADX > {adx_threshold}'},
                    'trend_strength': {'value': False, 'weight': 5, 'description': 'Fuerza de tendencia favorable'}
                }
            }
            
            if current_idx < 0:
                current_idx = len(data['whale_pump']) + current_idx if 'whale_pump' in data else -1
            
            if current_idx < 0 or current_idx >= len(data.get('whale_pump', [])):
                return conditions
            
            conditions['long']['whale_pump']['value'] = data['whale_pump'][current_idx] > 15
            
            conditions['long']['di_cross_bullish']['value'] = (
                data['di_cross_bullish'][current_idx] or 
                (current_idx > 0 and data['di_cross_bullish'][current_idx-1]) or
                (current_idx > 1 and data['di_cross_bullish'][current_idx-2])
            )
            
            conditions['long']['di_trend_bullish']['value'] = (
                current_idx >= 3 and data['di_trend_bullish'][current_idx]
            )
            
            conditions['long']['divergence']['value'] = False
            if current_idx < len(data['bullish_divergence']):
                lookback_start = max(0, current_idx - 6)
                for i in range(lookback_start, current_idx + 1):
                    if i < len(data['bullish_divergence']) and data['bullish_divergence'][i]:
                        conditions['long']['divergence']['value'] = True
                        break
            
            conditions['long']['breakout']['value'] = (
                current_idx < len(data['breakout_up']) and 
                data['breakout_up'][current_idx]
            )
            
            conditions['long']['volume_anomaly']['value'] = (
                current_idx < len(data['volume_anomaly']) and 
                data['volume_anomaly'][current_idx]
            )
            
            conditions['long']['adx']['value'] = (
                current_idx < len(data['adx']) and 
                data['adx'][current_idx] > adx_threshold
            )
            
            conditions['long']['trend_strength']['value'] = (
                current_idx < len(data['trend_strength_signals']) and 
                data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
                not data['no_trade_zones'][current_idx]
            )
            
            conditions['short']['whale_dump']['value'] = data['whale_dump'][current_idx] > 18
            
            conditions['short']['di_cross_bearish']['value'] = (
                data['di_cross_bearish'][current_idx] or 
                (current_idx > 0 and data['di_cross_bearish'][current_idx-1]) or
                (current_idx > 1 and data['di_cross_bearish'][current_idx-2])
            )
            
            conditions['short']['di_trend_bearish']['value'] = (
                current_idx >= 3 and data['di_trend_bearish'][current_idx]
            )
            
            conditions['short']['divergence']['value'] = False
            if current_idx < len(data['bearish_divergence']):
                lookback_start = max(0, current_idx - 6)
                for i in range(lookback_start, current_idx + 1):
                    if i < len(data['bearish_divergence']) and data['bearish_divergence'][i]:
                        conditions['short']['divergence']['value'] = True
                        break
            
            conditions['short']['breakout']['value'] = (
                current_idx < len(data['breakout_down']) and 
                data['breakout_down'][current_idx]
            )
            
            conditions['short']['volume_anomaly']['value'] = (
                current_idx < len(data['volume_anomaly']) and 
                data['volume_anomaly'][current_idx]
            )
            
            conditions['short']['adx']['value'] = (
                current_idx < len(data['adx']) and 
                data['adx'][current_idx] > adx_threshold
            )
            
            conditions['short']['trend_strength']['value'] = (
                current_idx < len(data['trend_strength_signals']) and 
                data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                not data['no_trade_zones'][current_idx]
            )
            
            return conditions
        except Exception as e:
            print(f"Error evaluando condiciones de señal: {e}")
            return {
                'long': {k: {'value': False, 'weight': v['weight'], 'description': v['description']} for k, v in {
                    'whale_pump': {'weight': 25, 'description': 'Ballena compradora activa'},
                    'di_cross_bullish': {'weight': 25, 'description': '+DI cruza -DI positivamente'},
                    'di_trend_bullish': {'weight': 15, 'description': '+DI con tendencia positiva'},
                    'divergence': {'weight': 15, 'description': 'Divergencia alcista en RSI Maverick'},
                    'breakout': {'weight': 10, 'description': 'Ruptura de tendencia alcista'},
                    'volume_anomaly': {'weight': 10, 'description': 'Volumen superior al promedio'},
                    'adx': {'weight': 5, 'description': f'ADX > {adx_threshold}'},
                    'trend_strength': {'weight': 5, 'description': 'Fuerza de tendencia favorable'}
                }.items()},
                'short': {k: {'value': False, 'weight': v['weight'], 'description': v['description']} for k, v in {
                    'whale_dump': {'weight': 25, 'description': 'Ballena vendedora activa'},
                    'di_cross_bearish': {'weight': 25, 'description': '-DI cruza +DI positivamente'},
                    'di_trend_bearish': {'weight': 15, 'description': '-DI con tendencia positiva'},
                    'divergence': {'weight': 15, 'description': 'Divergencia bajista en RSI Maverick'},
                    'breakout': {'weight': 10, 'description': 'Ruptura de tendencia bajista'},
                    'volume_anomaly': {'weight': 10, 'description': 'Volumen superior al promedio'},
                    'adx': {'weight': 5, 'description': f'ADX > {adx_threshold}'},
                    'trend_strength': {'weight': 5, 'description': 'Fuerza de tendencia favorable'}
                }.items()}
            }

    def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal basada en condiciones ponderadas"""
        try:
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
            
            if signal_type == 'long':
                if ma200_condition == 'above':
                    min_score = 70
                else:
                    min_score = 75
            else:
                if ma200_condition == 'above':
                    min_score = 85
                else:
                    min_score = 80

            if score < min_score:
                score = 0

            return min(score, 100), fulfilled_conditions
        except Exception as e:
            print(f"Error calculando score de señal: {e}")
            return 0, []

    def check_multi_timeframe_conditions(self, symbol, interval, signal_type):
        """Verificar condiciones multi-timeframe para entrada"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return True
            
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            if signal_type == 'LONG':
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                media_ok = tf_analysis.get('media', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                
                menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
                if menor_df is not None and len(menor_df) > 10:
                    menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                    menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']
                else:
                    menor_ok = True
                
                return mayor_ok and media_ok and menor_ok
                
            elif signal_type == 'SHORT':
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                media_ok = tf_analysis.get('media', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                
                menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
                if menor_df is not None and len(menor_df) > 10:
                    menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                    menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
                else:
                    menor_ok = True
                
                return mayor_ok and media_ok and menor_ok
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones multi-timeframe: {e}")
            return True

    def generate_exit_signals(self):
        """Generar señales de salida para operaciones activas"""
        exit_alerts = []
        current_time = self.get_bolivia_time()
        
        for signal_key, signal_data in list(self.active_signals.items()):
            try:
                symbol = signal_data['symbol']
                interval = signal_data['interval']
                signal_type = signal_data['signal']
                entry_price = signal_data['entry_price']
                
                df = self.get_kucoin_data(symbol, interval, 20)
                if df is None or len(df) < 10:
                    continue
                
                current_price = float(df['close'].iloc[-1])
                current_trend = self.calculate_trend_strength_maverick(df['close'].values)
                current_strength = current_trend['strength_signals'][-1]
                current_no_trade = current_trend['no_trade_zones'][-1]
                
                exit_reason = None
                
                if signal_type == 'LONG' and current_strength in ['WEAK_UP', 'STRONG_DOWN', 'WEAK_DOWN']:
                    exit_reason = "Fuerza de tendencia desfavorable"
                elif signal_type == 'SHORT' and current_strength in ['WEAK_DOWN', 'STRONG_UP', 'WEAK_UP']:
                    exit_reason = "Fuerza de tendencia desfavorable"
                
                elif current_no_trade:
                    exit_reason = "Zona de NO OPERAR activa"
                
                else:
                    hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                    if hierarchy.get('menor'):
                        menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 10)
                        if menor_df is not None and len(menor_df) > 5:
                            menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                            menor_strength = menor_trend['strength_signals'][-1]
                            
                            if signal_type == 'LONG' and menor_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                                exit_reason = "Cambio de tendencia en temporalidad menor"
                            elif signal_type == 'SHORT' and menor_strength in ['STRONG_UP', 'WEAK_UP']:
                                exit_reason = "Cambio de tendencia en temporalidad menor"
                
                if exit_reason:
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
                        'trend_strength': current_strength,
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    exit_alerts.append(exit_alert)
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

    def generate_signals_professional(self, symbol, interval, di_period=14, adx_threshold=20, 
                                    sr_period=50, rsi_length=20, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES PROFESIONAL - VERSIÓN CORREGIDA"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 30:
                return self._create_empty_signal(symbol)
            
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(
                df['high'].values, df['low'].values, df['close'].values, di_period
            )
            
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi_maverick = self.calculate_rsi_maverick(
                df['close'].values, rsi_length, bb_multiplier
            )
            
            bullish_div, bearish_div = self.detect_divergence(
                df['close'].values, rsi_maverick
            )
            
            breakout_up, breakout_down = self.check_breakout(
                df['high'].values, df['low'].values, df['close'].values,
                whale_data['support'], whale_data['resistance']
            )
            
            trend_strength_data = self.calculate_trend_strength_maverick(
                df['close'].values, length=20, mult=2.0
            )
            
            current_idx = -1
            
            min_length = min(
                len(whale_data['whale_pump']),
                len(whale_data['whale_dump']), 
                len(adx), len(plus_di), len(minus_di),
                len(di_cross_bullish), len(di_cross_bearish),
                len(di_trend_bullish), len(di_trend_bearish),
                len(rsi_maverick), len(bullish_div), len(bearish_div),
                len(breakout_up), len(breakout_down),
                len(whale_data['volume_anomaly']),
                len(trend_strength_data['bb_width']),
                len(trend_strength_data['no_trade_zones']),
                len(trend_strength_data['strength_signals'])
            )
            
            analysis_data = {
                'whale_pump': whale_data['whale_pump'][:min_length],
                'whale_dump': whale_data['whale_dump'][:min_length],
                'plus_di': plus_di[:min_length],
                'minus_di': minus_di[:min_length],
                'adx': adx[:min_length],
                'di_cross_bullish': di_cross_bullish[:min_length],
                'di_cross_bearish': di_cross_bearish[:min_length],
                'di_trend_bullish': di_trend_bullish[:min_length],
                'di_trend_bearish': di_trend_bearish[:min_length],
                'bullish_divergence': bullish_div[:min_length],
                'bearish_divergence': bearish_div[:min_length],
                'breakout_up': breakout_up[:min_length],
                'breakout_down': breakout_down[:min_length],
                'volume_anomaly': whale_data['volume_anomaly'][:min_length],
                'trend_strength': trend_strength_data['trend_strength'][:min_length],
                'no_trade_zones': trend_strength_data['no_trade_zones'][:min_length],
                'trend_strength_signals': trend_strength_data['strength_signals'][:min_length]
            }
            
            if current_idx < -min_length or current_idx >= min_length:
                current_idx = -1
            
            conditions = self.evaluate_signal_conditions_improved(analysis_data, current_idx, adx_threshold)
            
            ma200 = self.calculate_sma(df['close'].values, 200)
            current_ma200 = ma200[current_idx] if current_idx < len(ma200) else 0
            current_price = df['close'].iloc[current_idx]
            ma200_condition = 'above' if current_price > current_ma200 else 'below'

            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', ma200_condition)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', ma200_condition)
            
            signal_type = 'NEUTRAL'
            signal_score = 0
            trend_strength_filter = True
            multi_timeframe_ok = True
            
            if long_score >= 70:
                multi_timeframe_ok = self.check_multi_timeframe_conditions(symbol, interval, 'LONG')
                if multi_timeframe_ok:
                    trend_strength_filter = self.apply_trend_strength_filter('LONG', trend_strength_data, current_idx)
                    if trend_strength_filter:
                        signal_type = 'LONG'
                        signal_score = long_score
                        fulfilled_conditions = long_conditions
                    else:
                        signal_type = 'NEUTRAL'
                        signal_score = 0
                        fulfilled_conditions = []
                else:
                    signal_type = 'NEUTRAL'
                    signal_score = 0
                    fulfilled_conditions = []
            elif short_score >= 80:
                multi_timeframe_ok = self.check_multi_timeframe_conditions(symbol, interval, 'SHORT')
                if multi_timeframe_ok:
                    trend_strength_filter = self.apply_trend_strength_filter('SHORT', trend_strength_data, current_idx)
                    if trend_strength_filter:
                        signal_type = 'SHORT'
                        signal_score = short_score
                        fulfilled_conditions = short_conditions
                    else:
                        signal_type = 'NEUTRAL'
                        signal_score = 0
                        fulfilled_conditions = []
                else:
                    signal_type = 'NEUTRAL'
                    signal_score = 0
                    fulfilled_conditions = []
            else:
                fulfilled_conditions = []
            
            current_price = float(df['close'].iloc[current_idx])
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            if signal_type in ['LONG', 'SHORT'] and signal_score >= 70:
                signal_key = f"{symbol}_{interval}_{signal_type}"
                self.active_signals[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': levels_data['entry'],
                    'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                    'score': signal_score
                }
            
            ma_9 = self.calculate_sma(df['close'].values, 9)
            ma_21 = self.calculate_sma(df['close'].values, 21)
            ma_50 = self.calculate_sma(df['close'].values, 50)
            ma_200 = self.calculate_sma(df['close'].values, 200)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'].values)
            rsi = self.calculate_rsi(df['close'].values)
            macd, macd_signal, macd_histogram = self.calculate_macd(df['close'].values)
            
            result_data = {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'liquidation_long': float(levels_data['entry'] - (levels_data['entry'] / leverage)),
                'liquidation_short': float(levels_data['entry'] + (levels_data['entry'] / leverage)),
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
                'trend_strength_filter': trend_strength_filter,
                'multi_timeframe_ok': multi_timeframe_ok,
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'di_cross_bullish': di_cross_bullish[-50:],
                    'di_cross_bearish': di_cross_bearish[-50:],
                    'rsi_maverick': rsi_maverick[-50:],
                    'bullish_divergence': bullish_div[-50:],
                    'bearish_divergence': bearish_div[-50:],
                    'breakout_up': breakout_up[-50:],
                    'breakout_down': breakout_down[-50:],
                    'support': whale_data['support'][-50:],
                    'resistance': whale_data['resistance'][-50:],
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'rsi': rsi[-50:].tolist(),
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
                    'colors': trend_strength_data['colors'][-50:]
                }
            }
            
            return result_data
            
        except Exception as e:
            print(f"Error en generate_signals_professional para {symbol}: {e}")
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
            'liquidation_long': 0,
            'liquidation_short': 0,
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
            'trend_strength_filter': True,
            'multi_timeframe_ok': True,
            'no_trade_zone': False,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de scalping con filtros mejorados"""
        alerts = []
        telegram_intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D']
        
        current_time = self.get_bolivia_time()
        
        for interval in telegram_intervals:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:10]:
                try:
                    signal_data = self.generate_signals_professional(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65 and
                        signal_data['multi_timeframe_ok'] and
                        signal_data['trend_strength_filter']):
                        
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
                            'entry': signal_data['entry'],
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'][0],
                            'leverage': optimal_leverage,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'trend_strength': signal_data.get('trend_strength_signal', 'NEUTRAL'),
                            'support': signal_data['support'],
                            'resistance': signal_data['resistance']
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

def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram simplificada"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        
        if alert_type == 'entry':
            if alert_data['signal'] == 'LONG':
                stop_explanation = f"Por debajo del soporte en ${alert_data['support']:.6f}"
            else:
                stop_explanation = f"Por encima de la resistencia en ${alert_data['resistance']:.6f}"
            
            conditions_text = ""
            if alert_data.get('fulfilled_conditions'):
                conditions_text = "\n✅ Condiciones Cumplidas:\n• " + "\n• ".join(alert_data['fulfilled_conditions'][:3])
            
            message = f"""
🚨 ALERTA DE TRADING - Whale Hunter WGTA 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%

Precio actual: {alert_data.get('current_price', alert_data['entry']):.6f}

💪 Fuerza de Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}
💰 Entrada: ${alert_data['entry']:.6f}

🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
(Explicación: {stop_explanation})

📈 Apalancamiento: x{alert_data['leverage']}
{conditions_text}

📊 Revisa la señal en: https://ballenasscalpistas.onrender.com/
            """
            
        else:
            pnl_text = f"📊 P&L: {alert_data['pnl_percent']:+.2f}%"
            
            message = f"""
🚨 ALERTA DE SALIDA - Whale Hunter WGTA 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
{pnl_text}

💪 Fuerza de Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

📊 Observación: {alert_data['reason']}
            """
        
        report_url = f"https://ballenasscalpistas.onrender.com/api/generate_report?symbol={alert_data['symbol']}&interval={alert_data['interval']}&leverage={alert_data.get('leverage', 15)}"
        
        keyboard = [[telegram.InlineKeyboardButton("📊 Descargar Reporte Completo", url=report_url)]]
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
    """Verificador de alertas en segundo plano mejorado"""
    intraday_intervals = ['15m', '30m', '1h', '2h']
    swing_intervals = ['4h', '8h', '12h', '1D']
    
    intraday_last_check = datetime.now()
    swing_last_check = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            if (current_time - intraday_last_check).seconds >= 60:
                print("Verificando alertas intradía...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in intraday_intervals:
                        send_telegram_alert(alert, 'entry')
                
                exit_alerts = indicator.generate_exit_signals()
                for alert in exit_alerts:
                    send_telegram_alert(alert, 'exit')
                
                intraday_last_check = current_time
            
            if (current_time - swing_last_check).seconds >= 300:
                print("Verificando alertas swing...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in swing_intervals:
                        send_telegram_alert(alert, 'entry')
                
                exit_alerts = indicator.generate_exit_signals()
                for alert in exit_alerts:
                    send_telegram_alert(alert, 'exit')
                
                swing_last_check = current_time
            
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
    """Endpoint para obtener señales de trading CORREGIDO"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 20))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 20))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_professional(
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
        adx_threshold = int(request.args.get('adx_threshold', 20))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 20))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:
            try:
                signal_data = indicator.generate_signals_professional(
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
        adx_threshold = int(request.args.get('adx_threshold', 20))
        
        scatter_data = []
        
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:3])
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_professional(symbol, interval, di_period, adx_threshold)
                if signal_data and signal_data['current_price'] > 0:
                    
                    buy_pressure = min(100, max(0,
                        (signal_data['whale_pump'] / 100 * 30) +
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 25 +
                        (signal_data['rsi_maverick'] * 15) +
                        (1 if signal_data['adx'] > adx_threshold else 0) * 10 +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 20)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (signal_data['whale_dump'] / 100 * 30) +
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 25 +
                        ((1 - signal_data['rsi_maverick']) * 15) +
                        (1 if signal_data['adx'] > adx_threshold else 0) * 10 +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 20)
                    ))
                    
                    if signal_data['signal'] == 'LONG':
                        buy_pressure = max(buy_pressure, 70)
                        sell_pressure = min(sell_pressure, 30)
                    elif signal_data['signal'] == 'SHORT':
                        sell_pressure = max(sell_pressure, 70)
                        buy_pressure = min(buy_pressure, 30)
                    
                    scatter_data.append({
                        'symbol': symbol,
                        'x': float(buy_pressure),
                        'y': float(sell_pressure),
                        'volume': float(signal_data['volume']),
                        'signal_score': float(signal_data['signal_score']),
                        'current_price': float(signal_data['current_price']),
                        'signal': signal_data['signal'],
                        'buy_pressure': float(buy_pressure),
                        'sell_pressure': float(sell_pressure),
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
    """Endpoint para obtener la clasificación de riesgo de las criptomonedas"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    """Endpoint para obtener alertas de scalping"""
    try:
        alerts = indicator.generate_scalping_alerts()
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

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_professional(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        fig = plt.figure(figsize=(12, 14))
        
        ax1 = plt.subplot(6, 1, 1)
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
        
        ax2 = plt.subplot(6, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax2.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax2.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
        ax2.set_ylabel('Fuerza Ballenas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(6, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax3.plot(adx_dates, signal_data['indicators']['adx'], 
                    'white', linewidth=2, label='ADX')
            ax3.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax3.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
        ax3.set_ylabel('ADX/DMI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(6, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax4.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax4.set_ylabel('RSI Maverick')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(6, 1, 5, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax5.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax5.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax5.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax5.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax5.set_ylabel('Fuerza Tendencia %')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(6, 1, 6)
        ax6.axis('off')
        
        trend_strength_info = f"FUERZA TENDENCIA: {signal_data.get('trend_strength_signal', 'NEUTRAL')}"
        if signal_data.get('no_trade_zone'):
            trend_strength_info += " - ⚠️ ZONA DE NO OPERAR"
        
        multi_tf_info = "✅ MULTI-TIMEFRAME: Favorable" if signal_data.get('multi_timeframe_ok', True) else "❌ MULTI-TIMEFRAME: Desfavorable"
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        
        {trend_strength_info}
        {multi_tf_info}
        
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT 1: ${signal_data['take_profit'][0]:.6f}
        
        APALANCAMIENTO: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
        """
        
        ax6.text(0.1, 0.9, signal_info, transform=ax6.transAxes, fontsize=10,
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
        'day_of_week': current_time.strftime('%A'),
        'is_scalping_time': indicator.is_scalping_time(),
        'timezone': 'America/La_Paz'
    })

@app.route('/api/fear_greed_index')
def get_fear_greed_index():
    """Endpoint para obtener el índice de miedo y codicia"""
    try:
        return jsonify({
            'value': 65,
            'sentiment': 'Codicia',
            'color': 'success',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        print(f"Error obteniendo índice miedo/codicia: {e}")
        return jsonify({
            'value': 50,
            'sentiment': 'Neutral',
            'color': 'warning',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

@app.route('/api/market_recommendations')
def get_market_recommendations():
    """Endpoint para obtener recomendaciones de mercado"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        return jsonify({
            'recommendation': 'Mercado en tendencia lateral. Esperar confirmación de ruptura para posicionarse.',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        print(f"Error obteniendo recomendaciones: {e}")
        return jsonify({
            'recommendation': 'Análisis no disponible temporalmente.',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
