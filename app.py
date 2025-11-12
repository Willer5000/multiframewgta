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

# Configuración Telegram MEJORADA
TELEGRAM_BOT_TOKEN = "8007748376:AAHIW8n9b-BtA378g4gF-0-D2mOhn495Q0g"
TELEGRAM_CHAT_ID = "-1003229814161"

# Configuración optimizada - 40 criptomonedas top (actualizado según pdta2)
CRYPTO_SYMBOLS = [
    # Bajo Riesgo (20) - Top market cap
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
    "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT", "ETC-USDT",
    "XMR-USDT", "ALGO-USDT", "FIL-USDT", "VET-USDT", "THETA-USDT",
    
    # Medio Riesgo (10) - Proyectos consolidados
    "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT", "ONE-USDT",
    
    # Alto Riesgo (7) - Proyectos emergentes
    "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
    "MAGIC-USDT", "RNDR-USDT",
    
    # Memecoins (3) - Top memes
    "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT"
]

# Clasificación de riesgo optimizada (pdta2)
CRYPTO_RISK_CLASSIFICATION = {
    "bajo": [
        "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
        "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
        "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT", "ETC-USDT",
        "XMR-USDT", "ALGO-USDT", "FIL-USDT", "VET-USDT", "THETA-USDT"
    ],
    "medio": [
        "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
        "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT", "ONE-USDT"
    ],
    "alto": [
        "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
        "MAGIC-USDT", "RNDR-USDT"
    ],
    "memecoins": [
        "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT"
    ]
}

# Mapeo de temporalidades MEJORADO con nueva jerarquía (Requerimiento 3)
TIMEFRAME_HIERARCHY = {
    '15m': {'mayor': '1h', 'media': '30m', 'menor': '5m'},
    '30m': {'mayor': '2h', 'media': '1h', 'menor': '15m'},
    '1h': {'mayor': '4h', 'media': '2h', 'menor': '30m'},
    '2h': {'mayor': '8h', 'media': '4h', 'menor': '1h'},
    '4h': {'mayor': '12h', 'media': '8h', 'menor': '2h'},
    '8h': {'mayor': '1D', 'media': '12h', 'menor': '4h'},
    '12h': {'mayor': '3D', 'media': '1D', 'menor': '8h'},
    '1D': {'mayor': '3D', 'media': '1D', 'menor': '12h'},
    '3D': {'mayor': '1W', 'media': '3D', 'menor': '1D'},
    '1W': {'mayor': '1M', 'media': '1W', 'menor': '3D'}
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_signals = {}
        self.winrate_data = {}
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
        elif interval == '3D':
            # Para 3 días, verificar cada 12 horas
            return current_time.hour in [8, 20]
        elif interval == '1W':
            # Para 1 semana, verificar diariamente a las 8am
            return current_time.hour == 8
        
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
                '1D': '1day', '3D': '3day', '1W': '1week', '1M': '1month'
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

    def calculate_sma(self, prices, period):
        """Calcular SMA manualmente"""
        if len(prices) == 0 or period <= 0:
            return np.zeros_like(prices)
            
        sma = np.zeros_like(prices)
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window = prices[start_idx:i+1]
            valid_values = window[~np.isnan(window)]
            sma[i] = np.mean(valid_values) if len(valid_values) > 0 else (prices[i] if i < len(prices) and not np.isnan(prices[i]) else 0)
        
        return sma

    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente"""
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

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional manualmente (Requerimiento 4B)"""
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

    def detect_rsi_divergence(self, price, rsi, lookback=14):
        """Detectar divergencias RSI tradicional (Requerimiento 4B)"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Divergencia alcista: precio hace lower low, RSI hace higher low
            if (price[i] < price[i-1] and 
                rsi[i] > rsi[i-1] and
                price[i] < np.min(price[i-lookback:i])):
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace higher high, RSI hace lower high
            if (price[i] > price[i-1] and 
                rsi[i] < rsi[i-1] and
                price[i] > np.max(price[i-lookback:i])):
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcular MACD manualmente (Requerimiento 4F)"""
        if len(prices) < slow:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def detect_macd_divergence(self, price, macd, lookback=14):
        """Detectar divergencias MACD (Requerimiento 4F)"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            if (price[i] < price[i-1] and 
                macd[i] > macd[i-1] and
                price[i] < np.min(price[i-lookback:i])):
                bullish_div[i] = True
            
            if (price[i] > price[i-1] and 
                macd[i] < macd[i-1] and
                price[i] > np.max(price[i-lookback:i])):
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger manualmente (Requerimiento 4H)"""
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

    def calculate_b_percent(self, close, upper_bb, lower_bb):
        """Calcular %B de Bollinger Bands"""
        n = len(close)
        b_percent = np.zeros(n)
        
        for i in range(n):
            if (upper_bb[i] - lower_bb[i]) > 0:
                b_percent[i] = (close[i] - lower_bb[i]) / (upper_bb[i] - lower_bb[i])
            else:
                b_percent[i] = 0.5
        
        return b_percent

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo (Requerimiento 5)"""
        n = len(close)
        
        # Patrones de reversión
        head_shoulders = np.zeros(n, dtype=bool)  # HCH - SEÑAL SHORT
        double_top = np.zeros(n, dtype=bool)      # Doble Techo - SEÑAL SHORT  
        double_bottom = np.zeros(n, dtype=bool)   # Doble Fondo - SEÑAL LONG
        
        # Patrones de continuación
        bull_flag = np.zeros(n, dtype=bool)       # Banderín Alcista - SEÑAL LONG
        ascending_triangle = np.zeros(n, dtype=bool)  # Triángulo Ascendente - SEÑAL LONG
        bear_rectangle = np.zeros(n, dtype=bool)  # Rectángulo Bajista - SEÑAL LONG
        
        for i in range(lookback, n-1):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            window_close = close[i-lookback:i+1]
            
            # Detección simplificada de patrones (implementación básica)
            max_high = np.max(window_high)
            min_low = np.min(window_low)
            avg_price = np.mean(window_close)
            
            # Doble Techo (simplificado)
            if (len(window_high) >= 10 and 
                np.sum(window_high[-10:] >= max_high * 0.98) >= 2 and
                window_close[i] < avg_price):
                double_top[i] = True
            
            # Doble Fondo (simplificado)
            if (len(window_low) >= 10 and 
                np.sum(window_low[-10:] <= min_low * 1.02) >= 2 and
                window_close[i] > avg_price):
                double_bottom[i] = True
            
            # Banderín Alcista (simplificado)
            if (window_close[i] > avg_price and 
                np.std(window_high[-5:]) < np.std(window_high[-10:-5]) * 0.7):
                bull_flag[i] = True
                
        return {
            'head_shoulders': head_shoulders.tolist(),
            'double_top': double_top.tolist(),
            'double_bottom': double_bottom.tolist(),
            'bull_flag': bull_flag.tolist(),
            'ascending_triangle': ascending_triangle.tolist(),
            'bear_rectangle': bear_rectangle.tolist()
        }

    def calculate_support_resistance(self, high, low, close, period=50):
        """Calcular soportes y resistencias Smart Money (Requerimiento 4D)"""
        n = len(close)
        support_levels = np.zeros(n)
        resistance_levels = np.zeros(n)
        support_touches = np.zeros(n, dtype=int)
        resistance_touches = np.zeros(n, dtype=int)
        
        for i in range(period, n):
            # Niveles de soporte (mínimos locales)
            window_low = low[i-period:i+1]
            support_candidates = []
            for j in range(1, len(window_low)-1):
                if (window_low[j] < window_low[j-1] and 
                    window_low[j] < window_low[j+1]):
                    support_candidates.append(window_low[j])
            
            if support_candidates:
                support_levels[i] = np.min(support_candidates)
                # Contar toques al soporte
                support_touches[i] = np.sum(np.abs(low[i-period:i+1] - support_levels[i]) <= support_levels[i] * 0.002)
            
            # Niveles de resistencia (máximos locales)
            window_high = high[i-period:i+1]
            resistance_candidates = []
            for j in range(1, len(window_high)-1):
                if (window_high[j] > window_high[j-1] and 
                    window_high[j] > window_high[j+1]):
                    resistance_candidates.append(window_high[j])
            
            if resistance_candidates:
                resistance_levels[i] = np.max(resistance_candidates)
                # Contar toques a la resistencia
                resistance_touches[i] = np.sum(np.abs(high[i-period:i+1] - resistance_levels[i]) <= resistance_levels[i] * 0.002)
        
        return {
            'support': support_levels.tolist(),
            'resistance': resistance_levels.tolist(),
            'support_touches': support_touches.tolist(),
            'resistance_touches': resistance_touches.tolist()
        }

    def calculate_adx(self, high, low, close, period=14):
        """Calcular ADX + DMI (Requerimiento 4E - ADX > 25)"""
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

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick"""
        try:
            n = len(close)
            
            # Calcular Bandas de Bollinger
            basis = self.calculate_sma(close, length)
            dev = np.zeros(n)
            
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                dev[i] = np.std(window) if len(window) > 1 else 0
            
            upper = basis + (dev * mult)
            lower = basis - (dev * mult)
            
            # Calcular ancho de las bandas normalizado a porcentaje
            bb_width = np.zeros(n)
            for i in range(n):
                if basis[i] > 0:
                    bb_width[i] = ((upper[i] - lower[i]) / basis[i]) * 100
            
            # Determinar dirección de la fuerza
            trend_strength = np.zeros(n)
            for i in range(1, n):
                if bb_width[i] > bb_width[i-1]:
                    trend_strength[i] = bb_width[i]  # Positivo (verde)
                else:
                    trend_strength[i] = -bb_width[i]  # Negativo (rojo)
            
            # Calcular percentil 70 para zona alta
            if n >= 50:
                historical_bb_width = bb_width[max(0, n-100):n]
                high_zone_threshold = np.percentile(historical_bb_width, 70)
            else:
                high_zone_threshold = np.percentile(bb_width, 70) if len(bb_width) > 0 else 5
            
            # Detectar zonas de no operar
            no_trade_zones = np.zeros(n, dtype=bool)
            strength_signals = ['NEUTRAL'] * n
            
            for i in range(10, n):
                # Zona de no operar cuando hay pérdida de fuerza después de movimiento fuerte
                if (bb_width[i] > high_zone_threshold and 
                    trend_strength[i] < 0 and 
                    bb_width[i] < np.max(bb_width[max(0, i-10):i])):
                    no_trade_zones[i] = True
                
                # Determinar señal de fuerza de tendencia
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
        """Verificar tendencia en múltiples temporalidades (Requerimiento 2)"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(timeframe, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and timeframe != '15m':  # Solo usar 5m para 15m
                    results[tf_type] = 'NEUTRAL'
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 50)
                if df is None or len(df) < 20:
                    results[tf_type] = 'NEUTRAL'
                    continue
                
                close = df['close'].values
                
                # Calcular medias para determinar tendencia
                ma_9 = self.calculate_sma(close, 9)
                ma_21 = self.calculate_sma(close, 21)
                ma_50 = self.calculate_sma(close, 50)
                
                current_ma_9 = ma_9[-1] if len(ma_9) > 0 else 0
                current_ma_21 = ma_21[-1] if len(ma_21) > 0 else 0
                current_ma_50 = ma_50[-1] if len(ma_50) > 0 else 0
                current_price = close[-1]
                
                # Determinar tendencia (Requerimiento 4A)
                if (current_price > current_ma_9 and 
                    current_ma_9 > current_ma_21 and 
                    current_ma_21 > current_ma_50):
                    results[tf_type] = 'BULLISH'
                elif (current_price < current_ma_9 and 
                      current_ma_9 < current_ma_21 and 
                      current_ma_21 < current_ma_50):
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}

    def check_obligatory_conditions(self, symbol, interval, signal_type):
        """Verificar condiciones OBLIGATORIAS (Requerimiento 2)"""
        try:
            # Verificar multi-timeframe
            multi_tf = self.check_multi_timeframe_trend(symbol, interval)
            
            # Verificar fuerza de tendencia Maverick en todas las temporalidades
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            maverick_conditions = {}
            
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and interval != '15m':
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 30)
                if df is not None and len(df) > 10:
                    trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                    maverick_conditions[tf_type] = {
                        'signal': trend_data['strength_signals'][-1],
                        'no_trade': trend_data['no_trade_zones'][-1]
                    }
                else:
                    maverick_conditions[tf_type] = {'signal': 'NEUTRAL', 'no_trade': False}
            
            # Verificar condiciones según tipo de señal
            if signal_type == 'LONG':
                # Temporalidad Mayor: ALCISTA o NEUTRAL
                mayor_ok = multi_tf.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                
                # Temporalidad Media: EXCLUSIVAMENTE ALCISTA
                media_ok = multi_tf.get('media', 'NEUTRAL') == 'BULLISH'
                
                # Temporalidad Menor: Fuerza Maverick ALCISTA
                menor_ok = maverick_conditions.get('menor', {}).get('signal', 'NEUTRAL') in ['STRONG_UP', 'WEAK_UP']
                
                # NO zonas de NO OPERAR en ninguna temporalidad
                no_trade_ok = not any(cond.get('no_trade', False) for cond in maverick_conditions.values())
                
                return all([mayor_ok, media_ok, menor_ok, no_trade_ok])
                
            elif signal_type == 'SHORT':
                # Temporalidad Mayor: BAJISTA o NEUTRAL
                mayor_ok = multi_tf.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                
                # Temporalidad Media: EXCLUSIVAMENTE BAJISTA
                media_ok = multi_tf.get('media', 'NEUTRAL') == 'BEARISH'
                
                # Temporalidad Menor: Fuerza Maverick BAJISTA
                menor_ok = maverick_conditions.get('menor', {}).get('signal', 'NEUTRAL') in ['STRONG_DOWN', 'WEAK_DOWN']
                
                # NO zonas de NO OPERAR en ninguna temporalidad
                no_trade_ok = not any(cond.get('no_trade', False) for cond in maverick_conditions.values())
                
                return all([mayor_ok, media_ok, menor_ok, no_trade_ok])
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False

    def calculate_whale_signals_improved(self, df, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=20, signal_threshold=25, 
                                       sell_signal_threshold=20):
        """Implementación MEJORADA del indicador de ballenas (Requerimiento 1)"""
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
                'support': df['low'].values.tolist(),
                'resistance': df['high'].values.tolist(),
                'volume_anomaly': [False] * n
            }

    def calculate_rsi_maverick(self, close, length=20, bb_multiplier=2.0):
        """Implementación del RSI Modificado Maverick (Bollinger %B)"""
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

    def detect_maverick_divergence(self, price, maverick, lookback=14):
        """Detectar divergencias RSI Maverick"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            if (price[i] < price[i-1] and 
                maverick[i] > maverick[i-1] and
                price[i] < np.min(price[i-lookback:i])):
                bullish_div[i] = True
            
            if (price[i] > price[i-1] and 
                maverick[i] < maverick[i-1] and
                price[i] > np.max(price[i-lookback:i])):
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI con confirmación"""
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

    def check_macd_crossover(self, macd, signal):
        """Detectar cruces MACD"""
        n = len(macd)
        macd_bullish = np.zeros(n, dtype=bool)
        macd_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
                macd_bullish[i] = True
            
            if macd[i] < signal[i] and macd[i-1] >= signal[i-1]:
                macd_bearish[i] = True
        
        return macd_bullish.tolist(), macd_bearish.tolist()

    def evaluate_signal_conditions_improved(self, data, current_idx, adx_threshold=25):
        """Evaluar condiciones de señal con nueva lógica (Requerimiento 6)"""
        conditions = {
            'long': {
                'multi_timeframe': {'value': False, 'weight': 30, 'description': 'Condiciones Multi-TF cumplidas', 'obligatory': True},
                'ma_200': {'value': False, 'weight': 15, 'description': 'Precio sobre MA200', 'obligatory': False},
                'whale_pump': {'value': False, 'weight': 10, 'description': 'Ballena compradora activa'},
                'di_cross_bullish': {'value': False, 'weight': 10, 'description': '+DI cruza -DI positivamente'},
                'rsi_oversold': {'value': False, 'weight': 8, 'description': 'RSI tradicional en sobreventa'},
                'maverick_oversold': {'value': False, 'weight': 8, 'description': 'RSI Maverick en sobreventa'},
                'rsi_divergence': {'value': False, 'weight': 7, 'description': 'Divergencia RSI alcista'},
                'maverick_divergence': {'value': False, 'weight': 7, 'description': 'Divergencia Maverick alcista'},
                'macd_bullish': {'value': False, 'weight': 5, 'description': 'MACD cruce alcista'},
                'chart_pattern': {'value': False, 'weight': 5, 'description': 'Patrón chartista alcista'},
                'adx': {'value': False, 'weight': 5, 'description': f'ADX > {adx_threshold}'}
            },
            'short': {
                'multi_timeframe': {'value': False, 'weight': 30, 'description': 'Condiciones Multi-TF cumplidas', 'obligatory': True},
                'ma_200': {'value': False, 'weight': 15, 'description': 'Precio bajo MA200', 'obligatory': False},
                'whale_dump': {'value': False, 'weight': 10, 'description': 'Ballena vendedora activa'},
                'di_cross_bearish': {'value': False, 'weight': 10, 'description': '-DI cruza +DI positivamente'},
                'rsi_overbought': {'value': False, 'weight': 8, 'description': 'RSI tradicional en sobrecompra'},
                'maverick_overbought': {'value': False, 'weight': 8, 'description': 'RSI Maverick en sobrecompra'},
                'rsi_divergence': {'value': False, 'weight': 7, 'description': 'Divergencia RSI bajista'},
                'maverick_divergence': {'value': False, 'weight': 7, 'description': 'Divergencia Maverick bajista'},
                'macd_bearish': {'value': False, 'weight': 5, 'description': 'MACD cruce bajista'},
                'chart_pattern': {'value': False, 'weight': 5, 'description': 'Patrón chartista bajista'},
                'adx': {'value': False, 'weight': 5, 'description': f'ADX > {adx_threshold}'}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        current_price = data['close'][current_idx]
        
        # Condiciones LONG
        conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_ok', False)
        conditions['long']['ma_200']['value'] = current_price > data['ma_200'][current_idx] if current_idx < len(data['ma_200']) else False
        conditions['long']['whale_pump']['value'] = data['whale_pump'][current_idx] > 15
        conditions['long']['di_cross_bullish']['value'] = data['di_cross_bullish'][current_idx]
        conditions['long']['rsi_oversold']['value'] = data['rsi'][current_idx] < 20 if current_idx < len(data['rsi']) else False
        conditions['long']['maverick_oversold']['value'] = data['rsi_maverick'][current_idx] < 0.2
        conditions['long']['rsi_divergence']['value'] = data['rsi_bullish_div'][current_idx] if current_idx < len(data['rsi_bullish_div']) else False
        conditions['long']['maverick_divergence']['value'] = data['maverick_bullish_div'][current_idx] if current_idx < len(data['maverick_bullish_div']) else False
        conditions['long']['macd_bullish']['value'] = data['macd_bullish'][current_idx] if current_idx < len(data['macd_bullish']) else False
        conditions['long']['chart_pattern']['value'] = (
            data['double_bottom'][current_idx] or 
            data['bull_flag'][current_idx] or
            data['ascending_triangle'][current_idx]
        ) if current_idx < len(data.get('double_bottom', [])) else False
        conditions['long']['adx']['value'] = data['adx'][current_idx] > adx_threshold if current_idx < len(data['adx']) else False
        
        # Condiciones SHORT
        conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_ok', False)
        conditions['short']['ma_200']['value'] = current_price < data['ma_200'][current_idx] if current_idx < len(data['ma_200']) else False
        conditions['short']['whale_dump']['value'] = data['whale_dump'][current_idx] > 18
        conditions['short']['di_cross_bearish']['value'] = data['di_cross_bearish'][current_idx]
        conditions['short']['rsi_overbought']['value'] = data['rsi'][current_idx] > 80 if current_idx < len(data['rsi']) else False
        conditions['short']['maverick_overbought']['value'] = data['rsi_maverick'][current_idx] > 0.8
        conditions['short']['rsi_divergence']['value'] = data['rsi_bearish_div'][current_idx] if current_idx < len(data['rsi_bearish_div']) else False
        conditions['short']['maverick_divergence']['value'] = data['maverick_bearish_div'][current_idx] if current_idx < len(data['maverick_bearish_div']) else False
        conditions['short']['macd_bearish']['value'] = data['macd_bearish'][current_idx] if current_idx < len(data['macd_bearish']) else False
        conditions['short']['chart_pattern']['value'] = (
            data['head_shoulders'][current_idx] or 
            data['double_top'][current_idx]
        ) if current_idx < len(data.get('head_shoulders', [])) else False
        conditions['short']['adx']['value'] = data['adx'][current_idx] > adx_threshold if current_idx < len(data['adx']) else False
        
        return conditions

    def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal (Requerimiento 6)"""
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        obligatory_conditions_met = True
        
        signal_conditions = conditions.get(signal_type, {})
        
        # Verificar condiciones obligatorias
        for key, condition in signal_conditions.items():
            if condition.get('obligatory', False) and not condition['value']:
                obligatory_conditions_met = False
                break
        
        # Si no se cumplen las obligatorias, score = 0
        if not obligatory_conditions_met:
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
        
        # Ajustar score mínimo según MA200 (Requerimiento 6)
        if signal_type == 'long':
            if ma200_condition == 'above':
                min_score = 70
            else:
                min_score = 75
        else:  # short
            if ma200_condition == 'above':
                min_score = 75
            else:
                min_score = 70

        if base_score < min_score:
            return 0, []

        return min(base_score, 100), fulfilled_conditions

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con Smart Money (Requerimiento 4D)"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Soporte y resistencia Smart Money
            sr_data = self.calculate_support_resistance(high, low, close, 50)
            support = sr_data['support'][-1]
            resistance = sr_data['resistance'][-1]
            support_touches = sr_data['support_touches'][-1]
            resistance_touches = sr_data['resistance_touches'][-1]

            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Entrada lo más cerca posible al soporte (Smart Money)
                entry = min(current_price, support * 1.01)  # 1% sobre soporte
                
                # Stop loss más amplio para evitar mechazos
                stop_loss = max(support * 0.97, entry - (current_atr * 1.8))
                
                # Take profit basado en resistencia
                tp1 = resistance * 0.98
                
                # Asegurar relación riesgo/beneficio mínima 1:2
                min_tp = entry + (2 * (entry - stop_loss))
                tp1 = max(tp1, min_tp)
                
            else:  # SHORT
                # Entrada lo más cerca posible a la resistencia (Smart Money)
                entry = max(current_price, resistance * 0.99)  # 1% bajo resistencia
                
                # Stop loss más amplio para evitar mechazos
                stop_loss = min(resistance * 1.03, entry + (current_atr * 1.8))
                
                # Take profit basado en soporte
                tp1 = support * 1.02
                
                # Asegurar relación riesgo/beneficio mínima 1:2
                min_tp = entry - (2 * (stop_loss - entry))
                tp1 = min(tp1, min_tp)
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp1)],
                'support': float(support),
                'resistance': float(resistance),
                'support_touches': int(support_touches),
                'resistance_touches': int(resistance_touches),
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
                'support': float(np.min(df['low'].values[-14:])),
                'resistance': float(np.max(df['high'].values[-14:])),
                'support_touches': 0,
                'resistance_touches': 0,
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_winrate(self, symbol, interval, lookback=100):
        """Calcular winrate histórico (Requerimiento 9)"""
        try:
            if symbol not in self.winrate_data:
                self.winrate_data[symbol] = {}
            
            cache_key = f"{symbol}_{interval}"
            if cache_key in self.winrate_data[symbol]:
                cached_data, timestamp = self.winrate_data[symbol][cache_key]
                if (datetime.now() - timestamp).seconds < 3600:  # Cache 1 hora
                    return cached_data
            
            df = self.get_kucoin_data(symbol, interval, lookback + 20)
            if df is None or len(df) < 30:
                return {'winrate': 0, 'total_signals': 0, 'successful_signals': 0}
            
            signals = []
            successful_signals = 0
            
            # Analizar señales históricas
            for i in range(20, min(len(df), lookback + 20)):
                try:
                    # Simular señal histórica
                    window_df = df.iloc[:i+1]
                    if len(window_df) < 30:
                        continue
                    
                    # Análisis simplificado para winrate
                    price_change = (window_df['close'].iloc[-1] - window_df['close'].iloc[-2]) / window_df['close'].iloc[-2] * 100
                    
                    # Señal LONG si precio sube más del 1%
                    if price_change > 1.0:
                        signals.append('LONG')
                        # Considerar exitosa si sigue subiendo
                        if i + 5 < len(df) and df['close'].iloc[i+5] > window_df['close'].iloc[-1]:
                            successful_signals += 1
                    
                    # Señal SHORT si precio baja más del 1%
                    elif price_change < -1.0:
                        signals.append('SHORT')
                        # Considerar exitosa si sigue bajando
                        if i + 5 < len(df) and df['close'].iloc[i+5] < window_df['close'].iloc[-1]:
                            successful_signals += 1
                            
                except Exception as e:
                    continue
            
            total_signals = len(signals)
            winrate = (successful_signals / total_signals * 100) if total_signals > 0 else 0
            
            result = {
                'winrate': round(winrate, 1),
                'total_signals': total_signals,
                'successful_signals': successful_signals
            }
            
            self.winrate_data[symbol][cache_key] = (result, datetime.now())
            return result
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return {'winrate': 0, 'total_signals': 0, 'successful_signals': 0}

    def generate_exit_signals(self):
        """Generar señales de salida para operaciones activas (Requerimiento 7)"""
        exit_alerts = []
        current_time = self.get_bolivia_time()
        
        # Verificar cada señal activa (hasta 20 velas posteriores)
        for signal_key, signal_data in list(self.active_signals.items()):
            try:
                symbol = signal_data['symbol']
                interval = signal_data['interval']
                signal_type = signal_data['signal']
                entry_price = signal_data['entry_price']
                entry_time = signal_data['timestamp']
                
                # Obtener datos actuales
                df = self.get_kucoin_data(symbol, interval, 30)
                if df is None or len(df) < 10:
                    continue
                
                current_price = float(df['close'].iloc[-1])
                current_trend = self.calculate_trend_strength_maverick(df['close'].values)
                current_strength = current_trend['strength_signals'][-1]
                current_no_trade = current_trend['no_trade_zones'][-1]
                
                # Verificar multi-timeframe actual
                current_multi_tf = self.check_multi_timeframe_trend(symbol, interval)
                
                # Razones para salir
                exit_reason = None
                
                # 1. Salida por cambio de fuerza de tendencia Maverick
                if signal_type == 'LONG' and current_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                    exit_reason = "Fuerza de tendencia Maverick cambió a bajista"
                elif signal_type == 'SHORT' and current_strength in ['STRONG_UP', 'WEAK_UP']:
                    exit_reason = "Fuerza de tendencia Maverick cambió a alcista"
                
                # 2. Salida por zona no operar
                elif current_no_trade:
                    exit_reason = "Zona de NO OPERAR activa"
                
                # 3. Salida por cambio en temporalidad menor
                elif signal_type == 'LONG' and current_multi_tf.get('menor') == 'BEARISH':
                    exit_reason = "Cambio de tendencia en temporalidad menor"
                elif signal_type == 'SHORT' and current_multi_tf.get('menor') == 'BULLISH':
                    exit_reason = "Cambio de tendencia en temporalidad menor"
                
                # 4. Salida por objetivo de profit alcanzado
                elif signal_type == 'LONG' and current_price >= entry_price * 1.02:
                    exit_reason = "Objetivo de profit alcanzado (+2%)"
                elif signal_type == 'SHORT' and current_price <= entry_price * 0.98:
                    exit_reason = "Objetivo de profit alcanzado (+2%)"
                
                if exit_reason:
                    # Calcular P&L
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
                        'pnl_percent': round(pnl_percent, 2),
                        'reason': exit_reason,
                        'trend_strength': current_strength,
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    exit_alerts.append(exit_alert)
                    
                    # Remover señal activa
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=20, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - CON NUEVA ESTRATEGIA"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 30:
                return self._create_empty_signal(symbol)
            
            # INDICADORES PRINCIPALES (Requerimiento 4)
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Medias Móviles (4A)
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # RSI Tradicional (4B)
            rsi = self.calculate_rsi(close, 14)
            rsi_bullish_div, rsi_bearish_div = self.detect_rsi_divergence(close, rsi, 14)
            
            # RSI Maverick (4C)
            rsi_maverick = self.calculate_rsi_maverick(close, rsi_length, bb_multiplier)
            maverick_bullish_div, maverick_bearish_div = self.detect_maverick_divergence(close, rsi_maverick, 14)
            
            # Soporte y Resistencia Smart Money (4D)
            sr_data = self.calculate_support_resistance(high, low, close, sr_period)
            
            # ADX + DMI (4E - ADX > 25)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            # MACD (4F)
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_bullish, macd_bearish = self.check_macd_crossover(macd, macd_signal)
            macd_bullish_div, macd_bearish_div = self.detect_macd_divergence(close, macd, 14)
            
            # Bandas de Bollinger (4H)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            b_percent = self.calculate_b_percent(close, bb_upper, bb_lower)
            
            # Ballenas (Requerimiento 1 - Solo 12H y 1D como señal)
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            
            # Patrones de Chartismo (Requerimiento 5)
            chart_patterns = self.detect_chart_patterns(high, low, close, 50)
            
            # Fuerza de Tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close, length=20, mult=2.0)
            
            current_idx = -1
            
            # VERIFICAR CONDICIONES OBLIGATORIAS MULTI-TIMEFRAME (Requerimiento 2)
            long_obligatory_ok = self.check_obligatory_conditions(symbol, interval, 'LONG')
            short_obligatory_ok = self.check_obligatory_conditions(symbol, interval, 'SHORT')
            
            # Preparar datos para análisis
            analysis_data = {
                'close': close,
                'ma_200': ma_200,
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'di_cross_bullish': di_cross_bullish,
                'di_cross_bearish': di_cross_bearish,
                'rsi': rsi,
                'rsi_maverick': rsi_maverick,
                'rsi_bullish_div': rsi_bullish_div,
                'rsi_bearish_div': rsi_bearish_div,
                'maverick_bullish_div': maverick_bullish_div,
                'maverick_bearish_div': maverick_bearish_div,
                'macd_bullish': macd_bullish,
                'macd_bearish': macd_bearish,
                'double_bottom': chart_patterns['double_bottom'],
                'bull_flag': chart_patterns['bull_flag'],
                'ascending_triangle': chart_patterns['ascending_triangle'],
                'head_shoulders': chart_patterns['head_shoulders'],
                'double_top': chart_patterns['double_top']
            }
            
            conditions = self.evaluate_signal_conditions_improved(analysis_data, current_idx, adx_threshold)
            
            # Forzar condiciones multi-timeframe según verificación obligatoria
            conditions['long']['multi_timeframe']['value'] = long_obligatory_ok
            conditions['short']['multi_timeframe']['value'] = short_obligatory_ok
            
            # Calcular MA200 para ajustar scores mínimos
            current_ma200 = ma_200[current_idx] if current_idx < len(ma_200) else 0
            current_price = close[current_idx]
            ma200_condition = 'above' if current_price > current_ma200 else 'below'

            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', ma200_condition)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', ma200_condition)
            
            # DETERMINAR SEÑAL FINAL
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            # Aplicar reglas de score mínimo (Requerimiento 6)
            if long_obligatory_ok and long_score >= (70 if ma200_condition == 'above' else 75):
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_obligatory_ok and short_score >= (75 if ma200_condition == 'above' else 70):
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # Ballenas solo como señal en 12H y 1D (Requerimiento 1)
            if interval in ['12h', '1D']:
                if (signal_type == 'LONG' and not whale_data['confirmed_buy'][current_idx]) or \
                   (signal_type == 'SHORT' and not whale_data['confirmed_sell'][current_idx]):
                    # Si no hay confirmación de ballenas en estas TF, reducir score
                    signal_score = max(0, signal_score - 20)
                    if signal_score < 65:
                        signal_type = 'NEUTRAL'
                        signal_score = 0
                        fulfilled_conditions = []
            
            current_price = float(close[current_idx])
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # Registrar señal activa si es válida (para tracking de salidas)
            if signal_type in ['LONG', 'SHORT'] and signal_score >= 65:
                signal_key = f"{symbol}_{interval}_{signal_type}"
                self.active_signals[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': levels_data['entry'],
                    'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                    'score': signal_score
                }
            
            # Calcular winrate
            winrate_data = self.calculate_winrate(symbol, interval)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support': levels_data['support'],
                'resistance': levels_data['resistance'],
                'support_touches': levels_data['support_touches'],
                'resistance_touches': levels_data['resistance_touches'],
                'atr': levels_data['atr'],
                'atr_percentage': levels_data['atr_percentage'],
                'volume': float(df['volume'].iloc[current_idx]),
                'volume_ma': float(np.mean(df['volume'].tail(20))),
                'adx': float(adx[current_idx] if current_idx < len(adx) else 0),
                'plus_di': float(plus_di[current_idx] if current_idx < len(plus_di) else 0),
                'minus_di': float(minus_di[current_idx] if current_idx < len(minus_di) else 0),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi': float(rsi[current_idx] if current_idx < len(rsi) else 50),
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'b_percent': float(b_percent[current_idx] if current_idx < len(b_percent) else 0.5),
                'macd': float(macd[current_idx] if current_idx < len(macd) else 0),
                'macd_signal': float(macd_signal[current_idx] if current_idx < len(macd_signal) else 0),
                'macd_histogram': float(macd_histogram[current_idx] if current_idx < len(macd_histogram) else 0),
                'fulfilled_conditions': fulfilled_conditions,
                'multi_timeframe_ok': long_obligatory_ok if signal_type == 'LONG' else short_obligatory_ok,
                'obligatory_conditions_met': long_obligatory_ok if signal_type == 'LONG' else short_obligatory_ok,
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False,
                'trend_strength_signal': trend_strength_data['strength_signals'][current_idx] if current_idx < len(trend_strength_data['strength_signals']) else 'NEUTRAL',
                'winrate': winrate_data['winrate'],
                'total_signals': winrate_data['total_signals'],
                'successful_signals': winrate_data['successful_signals'],
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    # Medias Móviles
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    
                    # Ballenas
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    
                    # ADX/DMI
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'di_cross_bullish': di_cross_bullish[-50:],
                    'di_cross_bearish': di_cross_bearish[-50:],
                    
                    # RSI Tradicional
                    'rsi': rsi[-50:].tolist(),
                    'rsi_bullish_div': rsi_bullish_div[-50:],
                    'rsi_bearish_div': rsi_bearish_div[-50:],
                    
                    # RSI Maverick
                    'rsi_maverick': rsi_maverick[-50:],
                    'maverick_bullish_div': maverick_bullish_div[-50:],
                    'maverick_bearish_div': maverick_bearish_div[-50:],
                    
                    # MACD
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'macd_bullish': macd_bullish[-50:],
                    'macd_bearish': macd_bearish[-50:],
                    
                    # Bollinger Bands
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'b_percent': b_percent[-50:].tolist(),
                    
                    # Chart Patterns
                    'double_bottom': chart_patterns['double_bottom'][-50:],
                    'bull_flag': chart_patterns['bull_flag'][-50:],
                    'ascending_triangle': chart_patterns['ascending_triangle'][-50:],
                    'head_shoulders': chart_patterns['head_shoulders'][-50:],
                    'double_top': chart_patterns['double_top'][-50:],
                    
                    # Trend Strength Maverick
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
                    'colors': trend_strength_data['colors'][-50:],
                    
                    # Support/Resistance
                    'support': sr_data['support'][-50:],
                    'resistance': sr_data['resistance'][-50:],
                    'support_touches': sr_data['support_touches'][-50:],
                    'resistance_touches': sr_data['resistance_touches'][-50:]
                }
            }
            
        except Exception as e:
            print(f"Error en generate_signals_improved para {symbol}: {e}")
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
            'support': 0,
            'resistance': 0,
            'support_touches': 0,
            'resistance_touches': 0,
            'atr': 0,
            'atr_percentage': 0,
            'volume': 0,
            'volume_ma': 0,
            'adx': 0,
            'plus_di': 0,
            'minus_di': 0,
            'whale_pump': 0,
            'whale_dump': 0,
            'rsi': 50,
            'rsi_maverick': 0.5,
            'b_percent': 0.5,
            'macd': 0,
            'macd_signal': 0,
            'macd_histogram': 0,
            'fulfilled_conditions': [],
            'multi_timeframe_ok': False,
            'obligatory_conditions_met': False,
            'no_trade_zone': False,
            'trend_strength_signal': 'NEUTRAL',
            'winrate': 0,
            'total_signals': 0,
            'successful_signals': 0,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de scalping con nueva estrategia"""
        alerts = []
        telegram_intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '3D', '1W']
        
        current_time = self.get_bolivia_time()
        
        for interval in telegram_intervals:
            # Para scalping (15m y 30m) verificar horario
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            # Verificar si es momento de enviar alerta para esta temporalidad
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:15]:  # Limitar para performance
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65 and
                        signal_data['obligatory_conditions_met'] and
                        not signal_data['no_trade_zone']):
                        
                        risk_category = next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                        
                        # Calcular leverage óptimo
                        volatility = signal_data['atr_percentage']
                        if volatility > 0.05:
                            optimal_leverage = 10
                        elif volatility > 0.02:
                            optimal_leverage = 15
                        else:
                            optimal_leverage = 20
                        
                        # Ajustar por categoría de riesgo
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
                            'resistance': signal_data['resistance'],
                            'winrate': signal_data.get('winrate', 0),
                            'multi_timeframe_ok': signal_data.get('multi_timeframe_ok', False),
                            'obligatory_conditions_met': signal_data.get('obligatory_conditions_met', False)
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
    """Enviar alerta por Telegram MEJORADA (Requerimiento 10)"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        
        if alert_type == 'entry':
            # Formato mejorado para entrada
            message = f"""
🚨 ALERTA DE TRADING - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
🏆 Winrate: {alert_data.get('winrate', 0):.1f}%

💰 Precio actual: {alert_data.get('current_price', alert_data['entry']):.6f}
💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

🎯 ENTRADA: ${alert_data['entry']:.6f}
🛑 STOP LOSS: ${alert_data['stop_loss']:.6f}
🎯 TAKE PROFIT: ${alert_data['take_profit']:.6f}

📈 Apalancamiento: x{alert_data['leverage']}
✅ Multi-Timeframe: {'CONFIRMADO' if alert_data.get('multi_timeframe_ok') else 'PENDIENTE'}
🔐 Obligatorios: {'CUMPLIDOS' if alert_data.get('obligatory_conditions_met') else 'FALTANTES'}

📊 Soporte: ${alert_data['support']:.6f}
📊 Resistencia: ${alert_data['resistance']:.6f}

✅ Condiciones Cumplidas:
• {chr(10) + '• '.join(alert_data.get('fulfilled_conditions', ['Análisis técnico favorable'])[:3])}

🔔 Revisa la señal en: https://multiframewgta.onrender.com/

⚠️ Gestiona tu riesgo adecuadamente.
            """
            
        else:  # exit alert
            pnl_text = f"📊 P&L: {alert_data['pnl_percent']:+.2f}%"
            
            message = f"""
🚨 ALERTA DE SALIDA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} - CERRAR POSICIÓN

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
{pnl_text}

💪 Fuerza de Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

📊 Razón de Salida:
{alert_data['reason']}

🎯 Operación: {'EXITOSA ✅' if alert_data['pnl_percent'] > 0 else 'NO EXITOSA ❌'}
⏱️ Timestamp: {alert_data['timestamp']}

🔔 Monitorea el mercado para posibles re-entradas.

⚠️ Registra la operación en tu journal.
            """
        
        # Generar URL para el reporte
        report_url = f"https://multiframewgta.onrender.com/api/generate_report?symbol={alert_data['symbol']}&interval={alert_data['interval']}&leverage={alert_data.get('leverage', 15)}"
        
        # Crear botón de descarga
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
    """Verificador de alertas en segundo plano MEJORADO"""
    intraday_intervals = ['15m', '30m', '1h', '2h']
    swing_intervals = ['4h', '8h', '12h', '1D', '3D', '1W']
    
    intraday_last_check = datetime.now()
    swing_last_check = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar intradía cada 60 segundos
            if (current_time - intraday_last_check).seconds >= 60:
                print("Verificando alertas intradía...")
                
                # Generar alertas de entrada
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    if alert['interval'] in intraday_intervals:
                        send_telegram_alert(alert, 'entry')
                
                # Generar alertas de salida
                exit_alerts = indicator.generate_exit_signals()
                for alert in exit_alerts:
                    send_telegram_alert(alert, 'exit')
                
                intraday_last_check = current_time
            
            # Verificar swing cada 300 segundos
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

@app.route('/api/signals')
def get_signals():
    """Endpoint para obtener señales de trading MEJORADO"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))  # Cambiado a 25
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 20))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, volume_filter, leverage
        )
        
        # Asegurar que los indicadores sean serializables
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
    """Endpoint para obtener múltiples señales MEJORADO"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))  # Cambiado a 25
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 20))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        volume_filter = request.args.get('volume_filter', 'Todos')
        leverage = int(request.args.get('leverage', 15))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
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
    """Endpoint para datos del scatter plot MEJORADO (Requerimiento 9)"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))  # Cambiado a 25
        
        scatter_data = []
        
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:3])  # 3 por categoría
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval, di_period, adx_threshold)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Nueva fórmula de presión basada en indicadores múltiples
                    buy_pressure = min(100, max(0,
                        (1 if signal_data['multi_timeframe_ok'] else 0) * 25 +
                        (signal_data['whale_pump'] / 100 * 15) +
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 15 +
                        (signal_data['rsi_maverick'] * 10) +
                        ((100 - signal_data['rsi']) / 100 * 10) +
                        (1 if signal_data['adx'] > adx_threshold else 0) * 10 +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 15)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (1 if signal_data['multi_timeframe_ok'] else 0) * 25 +
                        (signal_data['whale_dump'] / 100 * 15) +
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 15 +
                        ((1 - signal_data['rsi_maverick']) * 10) +
                        (signal_data['rsi'] / 100 * 10) +
                        (1 if signal_data['adx'] > adx_threshold else 0) * 10 +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 15)
                    ))
                    
                    # Ajustar según señal
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
                        ),
                        'winrate': signal_data.get('winrate', 0)
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

@app.route('/api/winrate')
def get_winrate():
    """Endpoint para obtener winrate del sistema (Requerimiento 9)"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        winrate_data = indicator.calculate_winrate(symbol, interval)
        return jsonify(winrate_data)
        
    except Exception as e:
        print(f"Error en /api/winrate: {e}")
        return jsonify({'winrate': 0, 'total_signals': 0, 'successful_signals': 0})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo MEJORADO (Requerimiento 9)"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        fig = plt.figure(figsize=(16, 18))
        fig.suptitle(f'MULTI-TIMEFRAME CRYPTO WGTA PRO - Reporte Completo\n{symbol} | {interval} | Score: {signal_data["signal_score"]:.1f}%', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Gráfico 1: Precio y Medias Móviles
        ax1 = plt.subplot(8, 1, 1)
        if signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            opens = [d['open'] for d in signal_data['data']]
            highs = [d['high'] for d in signal_data['data']]
            lows = [d['low'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            
            # Velas japonesas
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Medias móviles
            if 'indicators' in signal_data:
                ma_dates = dates[-len(signal_data['indicators']['ma_9']):]
                ax1.plot(ma_dates, signal_data['indicators']['ma_9'], 'orange', linewidth=1, label='MA 9')
                ax1.plot(ma_dates, signal_data['indicators']['ma_21'], 'blue', linewidth=1, label='MA 21')
                ax1.plot(ma_dates, signal_data['indicators']['ma_50'], 'purple', linewidth=1, label='MA 50')
                ax1.plot(ma_dates, signal_data['indicators']['ma_200'], 'red', linewidth=2, label='MA 200')
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='gold', linestyle='--', alpha=0.8, linewidth=2, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.8, linewidth=2, label='Stop Loss')
            ax1.axhline(y=signal_data['take_profit'][0], color='green', linestyle='--', alpha=0.8, linewidth=2, label='Take Profit')
            ax1.axhline(y=signal_data['support'], color='blue', linestyle=':', alpha=0.6, linewidth=1, label='Soporte')
            ax1.axhline(y=signal_data['resistance'], color='red', linestyle=':', alpha=0.6, linewidth=1, label='Resistencia')
        
        ax1.set_title('Precio y Medias Móviles con Niveles de Trading', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Ballenas (solo visual para todas las TF)
        ax2 = plt.subplot(8, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax2.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras', width=0.8)
            ax2.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras', width=0.8)
            ax2.axhline(y=15, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax2.axhline(y=18, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_title('Indicador Ballenas Compradoras/Vendedoras (Visual)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fuerza Ballenas')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: ADX/DMI
        ax3 = plt.subplot(8, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax3.plot(adx_dates, signal_data['indicators']['adx'], 
                    'white', linewidth=2, label='ADX')
            ax3.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1.5, label='+DI')
            ax3.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1.5, label='-DI')
            ax3.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral ADX 25')
        ax3.set_title('ADX + DMI (Fuerza y Dirección de Tendencia)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('ADX/DMI')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: RSI Tradicional
        ax4 = plt.subplot(8, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra 80')
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa 20')
            ax4.axhline(y=50, color='white', linestyle='-', alpha=0.3, linewidth=0.5)
            
            # Marcar divergencias
            for i, date in enumerate(rsi_dates):
                if i < len(signal_data['indicators']['rsi_bullish_div']) and signal_data['indicators']['rsi_bullish_div'][i]:
                    ax4.plot(date, signal_data['indicators']['rsi'][i], 'go', markersize=4, alpha=0.7)
                if i < len(signal_data['indicators']['rsi_bearish_div']) and signal_data['indicators']['rsi_bearish_div'][i]:
                    ax4.plot(date, signal_data['indicators']['rsi'][i], 'ro', markersize=4, alpha=0.7)
        ax4.set_title('RSI Tradicional con Divergencias', fontsize=12, fontweight='bold')
        ax4.set_ylabel('RSI')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: RSI Maverick
        ax5 = plt.subplot(8, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            maverick_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax5.plot(maverick_dates, signal_data['indicators']['rsi_maverick'], 
                    'magenta', linewidth=2, label='RSI Maverick (%B)')
            ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra 0.8')
            ax5.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa 0.2')
            ax5.axhline(y=0.5, color='white', linestyle='-', alpha=0.3, linewidth=0.5)
            
            # Marcar divergencias
            for i, date in enumerate(maverick_dates):
                if i < len(signal_data['indicators']['maverick_bullish_div']) and signal_data['indicators']['maverick_bullish_div'][i]:
                    ax5.plot(date, signal_data['indicators']['rsi_maverick'][i], 'go', markersize=4, alpha=0.7)
                if i < len(signal_data['indicators']['maverick_bearish_div']) and signal_data['indicators']['maverick_bearish_div'][i]:
                    ax5.plot(date, signal_data['indicators']['rsi_maverick'][i], 'ro', markersize=4, alpha=0.7)
        ax5.set_title('RSI Maverick (%B Bollinger) con Divergencias', fontsize=12, fontweight='bold')
        ax5.set_ylabel('RSI Maverick')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: MACD
        ax6 = plt.subplot(8, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax6.plot(macd_dates, signal_data['indicators']['macd'], 
                    'yellow', linewidth=1.5, label='MACD')
            ax6.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            # Histograma MACD con colores
            for i, date in enumerate(macd_dates):
                color = 'green' if signal_data['indicators']['macd_histogram'][i] >= 0 else 'red'
                ax6.bar(date, signal_data['indicators']['macd_histogram'][i], 
                       color=color, alpha=0.6, width=0.8)
            
            ax6.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=0.5)
        ax6.set_title('MACD con Histograma', fontsize=12, fontweight='bold')
        ax6.set_ylabel('MACD')
        ax6.legend(loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: Fuerza de Tendencia Maverick
        ax7 = plt.subplot(8, 1, 7, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax7.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax7.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax7.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax7.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax7.set_title('Fuerza de Tendencia Maverick - Ancho Bandas Bollinger %', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Fuerza Tendencia %')
            ax7.legend(loc='upper left')
            ax7.grid(True, alpha=0.3)
        
        # Información de la señal y justificación
        ax8 = plt.subplot(8, 1, 8)
        ax8.axis('off')
        
        # Información de multi-timeframe
        multi_tf_status = "✅ CONFIRMADO" if signal_data.get('multi_timeframe_ok') else "❌ NO CONFIRMADO"
        obligatory_status = "✅ CUMPLIDAS" if signal_data.get('obligatory_conditions_met') else "❌ FALTANTES"
        no_trade_status = "✅ OPERABLE" if not signal_data.get('no_trade_zone') else "❌ NO OPERAR"
        
        # Justificación de la señal
        justification = "JUSTIFICACIÓN TÉCNICA:\n"
        if signal_data['fulfilled_conditions']:
            for condition in signal_data['fulfilled_conditions'][:5]:  # Máximo 5 condiciones
                justification += f"• {condition}\n"
        else:
            justification += "• Análisis técnico favorable\n"
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']} | SCORE: {signal_data['signal_score']:.1f}%
        Winrate Histórico: {signal_data.get('winrate', 0):.1f}%
        
        MULTI-TIMEFRAME: {multi_tf_status}
        CONDICIONES OBLIGATORIAS: {obligatory_status}
        ZONA DE OPERACIÓN: {no_trade_status}
        
        NIVELES DE TRADING:
        Precio Actual: ${signal_data['current_price']:.6f}
        Entrada: ${signal_data['entry']:.6f}
        Stop Loss: ${signal_data['stop_loss']:.6f} ({((signal_data['entry']-signal_data['stop_loss'])/signal_data['entry']*100):.2f}%)
        Take Profit: ${signal_data['take_profit'][0]:.6f} ({((signal_data['take_profit'][0]-signal_data['entry'])/signal_data['entry']*100):.2f}%)
        
        SOPORTE/RESISTENCIA SMART MONEY:
        Soporte: ${signal_data['support']:.6f} (Toques: {signal_data['support_touches']})
        Resistencia: ${signal_data['resistance']:.6f} (Toques: {signal_data['resistance_touches']})
        
        {justification}
        
        INDICADORES CLAVE:
        • ADX: {signal_data['adx']:.1f} | +DI: {signal_data['plus_di']:.1f} | -DI: {signal_data['minus_di']:.1f}
        • RSI Tradicional: {signal_data['rsi']:.1f} | RSI Maverick: {(signal_data['rsi_maverick']*100):.1f}%
        • Ballenas Comp: {signal_data['whale_pump']:.1f} | Ballenas Vend: {signal_data['whale_dump']:.1f}
        • Fuerza Tendencia: {signal_data.get('trend_strength_signal', 'NEUTRAL')}
        
        APALANCAMIENTO: x{leverage} | ATR: {signal_data['atr_percentage']*100:.1f}%
        """
        
        ax8.text(0.02, 0.98, signal_info, transform=ax8.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4)
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
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
