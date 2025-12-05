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
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.sent_volume_alerts = set()
        self.last_volume_check = {}
        
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela"""
        if interval == '1h':
            next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            remaining_seconds = (next_close - current_time).total_seconds()
            return remaining_seconds <= (60 * 60 * 0.5)
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
                '1D': '1day', '1W': '1week'
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
                trend_data = self.calculate_trend_strength_maverick(close)
                
                current_signal = trend_data['strength_signals'][-1]
                current_no_trade = trend_data['no_trade_zones'][-1]
                
                if current_no_trade:
                    results[tf_type] = 'NO_TRADE'
                elif current_signal in ['STRONG_UP', 'WEAK_UP']:
                    results[tf_type] = 'BULLISH'
                elif current_signal in ['STRONG_DOWN', 'WEAK_DOWN']:
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
                # TF Mayor: Alcista, Neutral o NO_TRADE (pero no Bajista)
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL', 'NO_TRADE']
                
                # TF Media: Alcista
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'
                
                # TF Menor: Alcista y sin zona no operar
                menor_ok = tf_analysis.get('menor', 'NEUTRAL') == 'BULLISH'
                
                return mayor_ok and media_ok and menor_ok
                
            elif signal_type == 'SHORT':
                # TF Mayor: Bajista, Neutral o NO_TRADE (pero no Alcista)
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL', 'NO_TRADE']
                
                # TF Media: Bajista
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'
                
                # TF Menor: Bajista y sin zona no operar
                menor_ok = tf_analysis.get('menor', 'NEUTRAL') == 'BEARISH'
                
                return mayor_ok and media_ok and menor_ok
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones multi-timeframe obligatorias: {e}")
            return False
    
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
    
    def detect_divergence(self, price, indicator, lookback=14, confirmation_bars=4):
        """Detectar divergencias entre precio e indicador con confirmación"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Divergencia alcista: precio hace mínimos más bajos, indicador hace mínimos más altos
            price_lows = price[i-lookback:i+1]
            indicator_lows = indicator[i-lookback:i+1]
            
            if (np.argmin(price_lows) == len(price_lows)-1 and  # Último es el mínimo
                np.argmin(indicator_lows) != len(indicator_lows)-1 and  # Indicador no hizo mínimo
                indicator[i] > np.min(indicator_lows[:-1])):  # Indicador subiendo
                
                # Confirmar en las próximas N velas
                confirm = True
                for j in range(1, min(confirmation_bars, n-i)):
                    if price[i+j] < price[i] or indicator[i+j] < indicator[i]:
                        confirm = False
                        break
                
                if confirm:
                    for j in range(confirmation_bars):
                        if i+j < n:
                            bullish_div[i+j] = True
            
            # Divergencia bajista: precio hace máximos más altos, indicador hace máximos más bajos
            price_highs = price[i-lookback:i+1]
            indicator_highs = indicator[i-lookback:i+1]
            
            if (np.argmax(price_highs) == len(price_highs)-1 and  # Último es el máximo
                np.argmax(indicator_highs) != len(indicator_highs)-1 and  # Indicador no hizo máximo
                indicator[i] < np.max(indicator_highs[:-1])):  # Indicador bajando
                
                # Confirmar en las próximas N velas
                confirm = True
                for j in range(1, min(confirmation_bars, n-i)):
                    if price[i+j] > price[i] or indicator[i+j] > indicator[i]:
                        confirm = False
                        break
                
                if confirm:
                    for j in range(confirmation_bars):
                        if i+j < n:
                            bearish_div[i+j] = True
        
        return bullish_div.tolist(), bearish_div.tolist()
    
    def detect_breakout(self, high, low, close, support, resistance, confirmation_bars=1):
        """Detectar rupturas de soporte/resistencia con confirmación"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n-confirmation_bars):
            # Ruptura alcista: cierre por encima de resistencia
            if close[i] > resistance[i] and all(close[i+j] > resistance[i] for j in range(1, confirmation_bars+1)):
                breakout_up[i] = True
                for j in range(1, confirmation_bars+1):
                    if i+j < n:
                        breakout_up[i+j] = True
            
            # Ruptura bajista: cierre por debajo de soporte
            if close[i] < support[i] and all(close[i+j] < support[i] for j in range(1, confirmation_bars+1)):
                breakout_down[i] = True
                for j in range(1, confirmation_bars+1):
                    if i+j < n:
                        breakout_down[i+j] = True
        
        return breakout_up.tolist(), breakout_down.tolist()
    
    def detect_ma_crossover(self, ma_fast, ma_slow, confirmation_bars=1):
        """Detectar cruce de medias móviles con confirmación"""
        n = len(ma_fast)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n-confirmation_bars):
            # Cruce alcista: MA rápida cruza por encima de MA lenta
            if (ma_fast[i-1] <= ma_slow[i-1] and 
                ma_fast[i] > ma_slow[i] and
                all(ma_fast[i+j] > ma_slow[i+j] for j in range(1, confirmation_bars+1))):
                
                ma_cross_bullish[i] = True
                for j in range(1, confirmation_bars+1):
                    if i+j < n:
                        ma_cross_bullish[i+j] = True
            
            # Cruce bajista: MA rápida cruza por debajo de MA lenta
            if (ma_fast[i-1] >= ma_slow[i-1] and 
                ma_fast[i] < ma_slow[i] and
                all(ma_fast[i+j] < ma_slow[i+j] for j in range(1, confirmation_bars+1))):
                
                ma_cross_bearish[i] = True
                for j in range(1, confirmation_bars+1):
                    if i+j < n:
                        ma_cross_bearish[i+j] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()
    
    def detect_dmi_crossover(self, plus_di, minus_di, confirmation_bars=1):
        """Detectar cruce de DMI con confirmación"""
        n = len(plus_di)
        dmi_cross_bullish = np.zeros(n, dtype=bool)
        dmi_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n-confirmation_bars):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i-1] <= minus_di[i-1] and 
                plus_di[i] > minus_di[i] and
                all(plus_di[i+j] > minus_di[i+j] for j in range(1, confirmation_bars+1))):
                
                dmi_cross_bullish[i] = True
                for j in range(1, confirmation_bars+1):
                    if i+j < n:
                        dmi_cross_bullish[i+j] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i-1] <= plus_di[i-1] and 
                minus_di[i] > plus_di[i] and
                all(minus_di[i+j] > plus_di[i+j] for j in range(1, confirmation_bars+1))):
                
                dmi_cross_bearish[i] = True
                for j in range(1, confirmation_bars+1):
                    if i+j < n:
                        dmi_cross_bearish[i+j] = True
        
        return dmi_cross_bullish.tolist(), dmi_cross_bearish.tolist()
    
    def detect_macd_crossover(self, macd, signal, confirmation_bars=1):
        """Detectar cruce de MACD con confirmación"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n-confirmation_bars):
            # Cruce alcista: MACD cruza por encima de la señal
            if (macd[i-1] <= signal[i-1] and 
                macd[i] > signal[i] and
                all(macd[i+j] > signal[i+j] for j in range(1, confirmation_bars+1))):
                
                macd_cross_bullish[i] = True
                for j in range(1, confirmation_bars+1):
                    if i+j < n:
                        macd_cross_bullish[i+j] = True
            
            # Cruce bajista: MACD cruza por debajo de la señal
            if (macd[i-1] >= signal[i-1] and 
                macd[i] < signal[i] and
                all(macd[i+j] < signal[i+j] for j in range(1, confirmation_bars+1))):
                
                macd_cross_bearish[i] = True
                for j in range(1, confirmation_bars+1):
                    if i+j < n:
                        macd_cross_bearish[i+j] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()
    
    def detect_chart_patterns(self, high, low, close, confirmation_bars=7):
        """Detectar patrones de chartismo con confirmación"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        lookback = min(50, n)
        
        for i in range(lookback, n-confirmation_bars):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            
            # Doble Techo (simplificado)
            peaks = []
            for j in range(1, len(window_high)-1):
                if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                    peaks.append((j, window_high[j]))
            
            if len(peaks) >= 2:
                last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                    patterns['double_top'][i] = True
                    for j in range(1, confirmation_bars+1):
                        if i+j < n:
                            patterns['double_top'][i+j] = True
            
            # Doble Fondo (simplificado)
            troughs = []
            for j in range(1, len(window_low)-1):
                if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                    troughs.append((j, window_low[j]))
            
            if len(troughs) >= 2:
                last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                    patterns['double_bottom'][i] = True
                    for j in range(1, confirmation_bars+1):
                        if i+j < n:
                            patterns['double_bottom'][i+j] = True
        
        return patterns
    
    def detect_volume_anomaly(self, volume, period=20, threshold=2.0, cluster_min=1):
        """Detectar anomalías de volumen con clústeres"""
        n = len(volume)
        volume_anomaly = np.zeros(n, dtype=bool)
        volume_clusters = np.zeros(n, dtype=bool)
        volume_ema = self.calculate_ema(volume, period)
        
        for i in range(period, n):
            if volume_ema[i] > 0:
                volume_ratio = volume[i] / volume_ema[i]
                if volume_ratio > threshold:
                    volume_anomaly[i] = True
        
        # Detectar clústeres (múltiples anomalías cercanas)
        for i in range(period, n):
            if i >= cluster_min:
                recent_anomalies = volume_anomaly[max(0, i-cluster_min+1):i+1]
                if np.sum(recent_anomalies) >= cluster_min:
                    volume_clusters[i] = True
        
        return {
            'volume_anomaly': volume_anomaly.tolist(),
            'volume_clusters': volume_clusters.tolist(),
            'volume_ratio': (volume / volume_ema).tolist(),
            'volume_ema': volume_ema.tolist()
        }
    
    def calculate_whale_signals(self, df, lookback=20, confirmation_bars=7):
        """Calcular señales de ballenas con confirmación"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            
            n = len(close)
            whale_pump = np.zeros(n)
            whale_dump = np.zeros(n)
            confirmed_buy = np.zeros(n, dtype=bool)
            confirmed_sell = np.zeros(n, dtype=bool)
            
            volume_ema = self.calculate_ema(volume, 20)
            
            for i in range(lookback, n):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                # Ballenas compradoras: volumen alto con precio en mínimos
                if volume_ratio > 2.0 and close[i] <= np.min(low[max(0, i-5):i+1]) * 1.01:
                    whale_pump[i] = min(100, volume_ratio * 20)
                
                # Ballenas vendedoras: volumen alto con precio en máximos
                if volume_ratio > 2.0 and close[i] >= np.max(high[max(0, i-5):i+1]) * 0.99:
                    whale_dump[i] = min(100, volume_ratio * 20)
            
            # Confirmar señales en las próximas N velas
            for i in range(lookback, n-confirmation_bars):
                if whale_pump[i] > 25:
                    confirm = True
                    for j in range(1, confirmation_bars+1):
                        if i+j < n and close[i+j] < close[i]:
                            confirm = False
                            break
                    if confirm:
                        for j in range(confirmation_bars+1):
                            if i+j < n:
                                confirmed_buy[i+j] = True
                
                if whale_dump[i] > 25:
                    confirm = True
                    for j in range(1, confirmation_bars+1):
                        if i+j < n and close[i+j] > close[i]:
                            confirm = False
                            break
                    if confirm:
                        for j in range(confirmation_bars+1):
                            if i+j < n:
                                confirmed_sell[i+j] = True
            
            return {
                'whale_pump': whale_pump.tolist(),
                'whale_dump': whale_dump.tolist(),
                'confirmed_buy': confirmed_buy.tolist(),
                'confirmed_sell': confirmed_sell.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'confirmed_buy': [False] * n,
                'confirmed_sell': [False] * n
            }
    
    def calculate_support_resistance(self, high, low, close, period=50, num_levels=4):
        """Calcular soportes y resistencias dinámicos"""
        n = len(close)
        supports = []
        resistances = []
        
        # Usar pivotes para detectar niveles importantes
        for i in range(period, n, period):
            window_high = high[max(0, i-period):i]
            window_low = low[max(0, i-period):i]
            
            if len(window_high) > 0 and len(window_low) > 0:
                # Resistencia: máximos locales
                resistance_levels = []
                for j in range(2, len(window_high)-2):
                    if (window_high[j] > window_high[j-1] and 
                        window_high[j] > window_high[j-2] and
                        window_high[j] > window_high[j+1] and
                        window_high[j] > window_high[j+2]):
                        resistance_levels.append(window_high[j])
                
                # Soporte: mínimos locales
                support_levels = []
                for j in range(2, len(window_low)-2):
                    if (window_low[j] < window_low[j-1] and 
                        window_low[j] < window_low[j-2] and
                        window_low[j] < window_low[j+1] and
                        window_low[j] < window_low[j+2]):
                        support_levels.append(window_low[j])
                
                # Tomar los niveles más significativos
                if resistance_levels:
                    resistances.extend(sorted(resistance_levels, reverse=True)[:num_levels])
                if support_levels:
                    supports.extend(sorted(support_levels)[:num_levels])
        
        # Eliminar duplicados y ordenar
        supports = sorted(list(set(supports)))
        resistances = sorted(list(set(resistances)))
        
        # Asegurar al menos 4 niveles
        while len(supports) < 4:
            if supports:
                supports.append(supports[-1] * 0.99)
            else:
                supports.append(np.min(low) * 0.95)
        
        while len(resistances) < 4:
            if resistances:
                resistances.append(resistances[-1] * 1.01)
            else:
                resistances.append(np.max(high) * 1.05)
        
        return supports[:6], resistances[:6]  # Máximo 6 niveles
    
    def calculate_optimal_entry_exit(self, df, signal_type, supports, resistances, leverage=15):
        """Calcular entradas y salidas óptimas cerca de soportes/resistencias"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            
            if signal_type == 'LONG':
                # Entrada: soporte más cercano por debajo del precio actual
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)
                else:
                    entry = current_price * 0.99
                
                # Stop Loss: soporte inferior
                if len(supports) > 1:
                    stop_loss = supports[0] if supports[0] < entry else entry * 0.97
                else:
                    stop_loss = entry * 0.97
                
                # Take Profits: resistencias por encima
                valid_resistances = [r for r in resistances if r > entry]
                take_profit = []
                if valid_resistances:
                    for i, r in enumerate(valid_resistances[:3]):  # Máximo 3 TP
                        take_profit.append(r)
                else:
                    take_profit.append(entry * 1.03)
                
            else:  # SHORT
                # Entrada: resistencia más cercana por encima del precio actual
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)
                else:
                    entry = current_price * 1.01
                
                # Stop Loss: resistencia superior
                if len(resistances) > 1:
                    stop_loss = resistances[-1] if resistances[-1] > entry else entry * 1.03
                else:
                    stop_loss = entry * 1.03
                
                # Take Profits: soportes por debajo
                valid_supports = [s for s in supports if s < entry]
                take_profit = []
                if valid_supports:
                    for i, s in enumerate(valid_supports[:3]):  # Máximo 3 TP
                        take_profit.append(s)
                else:
                    take_profit.append(entry * 0.97)
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profit],
                'supports': [float(s) for s in supports],
                'resistances': [float(r) for r in resistances]
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.97,
                'take_profit': [current_price * 1.03],
                'supports': [current_price * 0.95, current_price * 0.93],
                'resistances': [current_price * 1.05, current_price * 1.07]
            }
    
    def evaluate_signal_conditions(self, data, current_idx, interval):
        """Evaluar condiciones de señal con pesos corregidos"""
        conditions = {
            'long': {},
            'short': {}
        }
        
        # Definir pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            # Estrategia Multi-Temporalidad
            weights_long = {
                'multi_timeframe': 30,  # Obligatorio
                'trend_strength': 25,   # Obligatorio
                'ma_crossover': 10,     # Complementario
                'dmi_crossover': 10,    # Complementario
                'adx_rising': 5,        # Complementario
                'bollinger_bands': 8,   # Complementario
                'macd_crossover': 10,   # Complementario
                'volume_anomaly': 7,    # Complementario
                'rsi_maverick_div': 8,  # Complementario
                'rsi_traditional_div': 5, # Complementario
                'chart_patterns': 5,    # Complementario
                'breakout': 5           # Complementario
            }
            weights_short = weights_long.copy()
            
        elif interval in ['12h', '1D']:
            # Estrategia con ballenas
            weights_long = {
                'whale_signal': 30,     # Obligatorio
                'trend_strength': 25,   # Obligatorio
                'ma_crossover': 10,     # Complementario
                'dmi_crossover': 10,    # Complementario
                'adx_rising': 5,        # Complementario
                'bollinger_bands': 8,   # Complementario
                'macd_crossover': 10,   # Complementario
                'volume_anomaly': 7,    # Complementario
                'rsi_maverick_div': 8,  # Complementario
                'rsi_traditional_div': 5, # Complementario
                'chart_patterns': 5,    # Complementario
                'breakout': 5           # Complementario
            }
            weights_short = weights_long.copy()
            
        else:  # 1W
            # Solo tendencia fuerte
            weights_long = {
                'trend_strength': 55,   # Obligatorio
                'ma_crossover': 10,     # Complementario
                'dmi_crossover': 10,    # Complementario
                'adx_rising': 5,        # Complementario
                'bollinger_bands': 8,   # Complementario
                'macd_crossover': 10,   # Complementario
                'volume_anomaly': 7,    # Complementario
                'rsi_maverick_div': 8,  # Complementario
                'rsi_traditional_div': 5, # Complementario
                'chart_patterns': 5,    # Complementario
                'breakout': 5           # Complementario
            }
            weights_short = weights_long.copy()
        
        # Inicializar condiciones
        for signal_type in ['long', 'short']:
            weights = weights_long if signal_type == 'long' else weights_short
            for key, weight in weights.items():
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
        ma_200 = data['ma_200'][current_idx] if current_idx < len(data['ma_200']) else 0
        
        # Condiciones LONG
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_long', False)
        elif interval in ['12h', '1D']:
            conditions['long']['whale_signal']['value'] = (
                data['confirmed_buy'][current_idx] if current_idx < len(data['confirmed_buy']) else False
            )
        
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        # Condiciones complementarias LONG
        conditions['long']['ma_crossover']['value'] = data.get('ma_cross_bullish', [False]*len(data['close']))[current_idx]
        conditions['long']['dmi_crossover']['value'] = data.get('dmi_cross_bullish', [False]*len(data['close']))[current_idx]
        conditions['long']['adx_rising']['value'] = (
            data['adx'][current_idx] > 25 and
            current_idx > 0 and
            data['adx'][current_idx] > data['adx'][current_idx-1]
        )
        conditions['long']['bollinger_bands']['value'] = data.get('bollinger_long', False)
        conditions['long']['macd_crossover']['value'] = data.get('macd_cross_bullish', [False]*len(data['close']))[current_idx]
        conditions['long']['volume_anomaly']['value'] = (
            data['volume_clusters'][current_idx] if current_idx < len(data['volume_clusters']) else False
        )
        conditions['long']['rsi_maverick_div']['value'] = (
            data['rsi_maverick_bullish_divergence'][current_idx] if current_idx < len(data['rsi_maverick_bullish_divergence']) else False
        )
        conditions['long']['rsi_traditional_div']['value'] = (
            data['rsi_bullish_divergence'][current_idx] if current_idx < len(data['rsi_bullish_divergence']) else False
        )
        conditions['long']['chart_patterns']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_flag'][current_idx]
        )
        conditions['long']['breakout']['value'] = (
            data['breakout_up'][current_idx] if current_idx < len(data['breakout_up']) else False
        )
        
        # Condiciones SHORT
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        elif interval in ['12h', '1D']:
            conditions['short']['whale_signal']['value'] = (
                data['confirmed_sell'][current_idx] if current_idx < len(data['confirmed_sell']) else False
            )
        
        conditions['short']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            not data['no_trade_zones'][current_idx]
        )
        
        # Condiciones complementarias SHORT
        conditions['short']['ma_crossover']['value'] = data.get('ma_cross_bearish', [False]*len(data['close']))[current_idx]
        conditions['short']['dmi_crossover']['value'] = data.get('dmi_cross_bearish', [False]*len(data['close']))[current_idx]
        conditions['short']['adx_rising']['value'] = (
            data['adx'][current_idx] > 25 and
            current_idx > 0 and
            data['adx'][current_idx] > data['adx'][current_idx-1]
        )
        conditions['short']['bollinger_bands']['value'] = data.get('bollinger_short', False)
        conditions['short']['macd_crossover']['value'] = data.get('macd_cross_bearish', [False]*len(data['close']))[current_idx]
        conditions['short']['volume_anomaly']['value'] = (
            data['volume_clusters'][current_idx] if current_idx < len(data['volume_clusters']) else False
        )
        conditions['short']['rsi_maverick_div']['value'] = (
            data['rsi_maverick_bearish_divergence'][current_idx] if current_idx < len(data['rsi_maverick_bearish_divergence']) else False
        )
        conditions['short']['rsi_traditional_div']['value'] = (
            data['rsi_bearish_divergence'][current_idx] if current_idx < len(data['rsi_bearish_divergence']) else False
        )
        conditions['short']['chart_patterns']['value'] = (
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['head_shoulders'][current_idx] or
            data['chart_patterns']['bearish_flag'][current_idx]
        )
        conditions['short']['breakout']['value'] = (
            data['breakout_down'][current_idx] if current_idx < len(data['breakout_down']) else False
        )
        
        return conditions
    
    def get_condition_description(self, condition_key):
        """Obtener descripción de condición"""
        descriptions = {
            'multi_timeframe': 'Multi-TF confirmado',
            'trend_strength': 'Fuerza tendencia favorable',
            'whale_signal': 'Señal ballenas confirmada',
            'ma_crossover': 'Cruce MA9/MA21',
            'dmi_crossover': 'Cruce DMI',
            'adx_rising': 'ADX con pendiente positiva',
            'bollinger_bands': 'Bandas de Bollinger',
            'macd_crossover': 'Cruce MACD',
            'volume_anomaly': 'Clúster volumen anómalo',
            'rsi_maverick_div': 'Divergencia RSI Maverick',
            'rsi_traditional_div': 'Divergencia RSI Tradicional',
            'chart_patterns': 'Patrón chartista',
            'breakout': 'Ruptura confirmada'
        }
        return descriptions.get(condition_key, condition_key)
    
    def calculate_signal_score(self, conditions, signal_type):
        """Calcular puntuación de señal basada en condiciones ponderadas"""
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        signal_conditions = conditions.get(signal_type, {})
        
        # Verificar condiciones obligatorias
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
        final_score = base_score if base_score >= 65 else 0

        return min(final_score, 100), fulfilled_conditions
    
    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """Generación de señales mejorada"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular todos los indicadores
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Indicadores básicos
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Detección de condiciones
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            ma_cross_bullish, ma_cross_bearish = self.detect_ma_crossover(ma_9, ma_21)
            dmi_cross_bullish, dmi_cross_bearish = self.detect_dmi_crossover(plus_di, minus_di)
            macd_cross_bullish, macd_cross_bearish = self.detect_macd_crossover(macd, macd_signal)
            
            # Divergencias
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick, confirmation_bars=4)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional, confirmation_bars=4)
            
            # Soporte/Resistencia
            supports, resistances = self.calculate_support_resistance(high, low, close, sr_period)
            
            # Breakouts
            current_support = np.min(supports) if supports else np.min(low[-50:])
            current_resistance = np.max(resistances) if resistances else np.max(high[-50:])
            breakout_up, breakout_down = self.detect_breakout(high, low, close, 
                                                             np.full_like(close, current_support),
                                                             np.full_like(close, current_resistance))
            
            # Patrones chartistas
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # Volumen
            volume_data = self.detect_volume_anomaly(volume)
            
            # Ballenas (solo para 12h y 1D)
            whale_data = self.calculate_whale_signals(df) if interval in ['12h', '1D'] else {
                'whale_pump': [0]*len(close),
                'whale_dump': [0]*len(close),
                'confirmed_buy': [False]*len(close),
                'confirmed_sell': [False]*len(close)
            }
            
            # Fuerza de tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Condiciones Bollinger
            bollinger_long = False
            bollinger_short = False
            if len(close) > 0:
                if close[-1] <= bb_lower[-1] * 1.02:
                    bollinger_long = True
                elif close[-1] >= bb_upper[-1] * 0.98:
                    bollinger_short = True
            
            # Multi-timeframe
            multi_timeframe_long = self.check_multi_timeframe_obligatory(symbol, interval, 'LONG')
            multi_timeframe_short = self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT')
            
            # Preparar datos para análisis
            analysis_data = {
                'close': close,
                'high': high,
                'low': low,
                'volume': volume,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'rsi_maverick': rsi_maverick,
                'rsi_traditional': rsi_traditional,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'ma_cross_bullish': ma_cross_bullish,
                'ma_cross_bearish': ma_cross_bearish,
                'dmi_cross_bullish': dmi_cross_bullish,
                'dmi_cross_bearish': dmi_cross_bearish,
                'macd_cross_bullish': macd_cross_bullish,
                'macd_cross_bearish': macd_cross_bearish,
                'rsi_maverick_bullish_divergence': rsi_maverick_bullish,
                'rsi_maverick_bearish_divergence': rsi_maverick_bearish,
                'rsi_bullish_divergence': rsi_bullish,
                'rsi_bearish_divergence': rsi_bearish,
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'chart_patterns': chart_patterns,
                'volume_anomaly': volume_data['volume_anomaly'],
                'volume_clusters': volume_data['volume_clusters'],
                'volume_ratio': volume_data['volume_ratio'],
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'confirmed_buy': whale_data['confirmed_buy'],
                'confirmed_sell': whale_data['confirmed_sell'],
                'trend_strength': trend_strength_data['trend_strength'],
                'no_trade_zones': trend_strength_data['no_trade_zones'],
                'trend_strength_signals': trend_strength_data['strength_signals'],
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_long': bollinger_long,
                'bollinger_short': bollinger_short
            }
            
            current_idx = -1
            conditions = self.evaluate_signal_conditions(analysis_data, current_idx, interval)
            
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
            
            if long_score >= min_score_long:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= min_score_short:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # Calcular niveles óptimos
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, supports, resistances, leverage)
            
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
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
                'adx': float(adx[current_idx] if current_idx < len(adx) else 0),
                'plus_di': float(plus_di[current_idx] if current_idx < len(plus_di) else 0),
                'minus_di': float(minus_di[current_idx] if current_idx < len(minus_di) else 0),
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'rsi_traditional': float(rsi_traditional[current_idx] if current_idx < len(rsi_traditional) else 50),
                'fulfilled_conditions': fulfilled_conditions,
                'multi_timeframe_ok': multi_timeframe_long if signal_type == 'LONG' else multi_timeframe_short,
                'ma200_condition': ma200_condition,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:],
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'volume_anomaly': volume_data['volume_anomaly'][-50:],
                    'volume_clusters': volume_data['volume_clusters'][-50:],
                    'volume_ratio': volume_data['volume_ratio'][-50:],
                    'volume_ema': volume_data['volume_ema'][-50:],
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'confirmed_buy': whale_data['confirmed_buy'][-50:],
                    'confirmed_sell': whale_data['confirmed_sell'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'colors': trend_strength_data['colors'][-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist()
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
            'volume': 0,
            'volume_ma': 0,
            'adx': 0,
            'plus_di': 0,
            'minus_di': 0,
            'rsi_maverick': 0.5,
            'rsi_traditional': 50,
            'fulfilled_conditions': [],
            'multi_timeframe_ok': False,
            'ma200_condition': 'below',
            'data': [],
            'indicators': {}
        }
    
    def check_volume_ema_ftm_signal(self, symbol, interval):
        """Nueva estrategia: Desplome por Volumen + EMA21 con Filtros FTMaverick"""
        try:
            # Solo para temporalidades específicas
            if interval not in ['1h', '4h', '12h', '1D']:
                return None
            
            # Verificar tiempo de ejecución
            current_time = self.get_bolivia_time()
            check_key = f"{symbol}_{interval}"
            
            if check_key in self.last_volume_check:
                last_check = self.last_volume_check[check_key]
                elapsed = (current_time - last_check).total_seconds()
                
                if interval == '1h' and elapsed < 300:
                    return None
                elif interval == '4h' and elapsed < 420:
                    return None
                elif interval in ['12h', '1D'] and elapsed < 600:
                    return None
            
            self.last_volume_check[check_key] = current_time
            
            # Obtener datos
            df = self.get_kucoin_data(symbol, interval, 50)
            if df is None or len(df) < 30:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular EMA21 y Volume MA21
            ema_21 = self.calculate_ema(close, 21)
            volume_ma_21 = self.calculate_ema(volume, 21)
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            current_ema_21 = ema_21[current_idx]
            current_volume_ma_21 = volume_ma_21[current_idx]
            
            # Condición A: Volumen y EMA
            volume_condition = current_volume > (current_volume_ma_21 * 2.5)
            
            if not volume_condition:
                return None
            
            # Determinar señal potencial
            if current_price > current_ema_21:
                potential_signal = 'LONG'
                price_condition = current_price > current_ema_21
            else:
                potential_signal = 'SHORT'
                price_condition = current_price < current_ema_21
            
            if not price_condition:
                return None
            
            # Condición B: Filtro FTMaverick
            trend_data = self.calculate_trend_strength_maverick(close)
            ftm_condition = not trend_data['no_trade_zones'][current_idx]
            
            if not ftm_condition:
                return None
            
            # Condición C: Filtro Multi-Timeframe
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            multi_tf_condition = True
            
            if hierarchy:
                tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
                
                if potential_signal == 'LONG':
                    mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                    menor_ok = tf_analysis.get('menor', 'NEUTRAL') == 'BULLISH'
                    multi_tf_condition = mayor_ok and menor_ok
                else:
                    mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                    menor_ok = tf_analysis.get('menor', 'NEUTRAL') == 'BEARISH'
                    multi_tf_condition = mayor_ok and menor_ok
            
            if not multi_tf_condition:
                return None
            
            # Todas las condiciones cumplidas - Señal CONFIRMADA
            volume_ratio = current_volume / current_volume_ma_21 if current_volume_ma_21 > 0 else 1
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': potential_signal,
                'price': float(current_price),
                'volume_ratio': float(volume_ratio),
                'ema_21': float(current_ema_21),
                'volume_ma_21': float(current_volume_ma_21),
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'ftm_ok': ftm_condition,
                'multi_tf_trend': tf_analysis if hierarchy else {}
            }
            
        except Exception as e:
            print(f"Error en check_volume_ema_ftm_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_volume_ema_signals(self):
        """Generar señales de la estrategia de volumen"""
        signals = []
        intervals = ['1h', '4h', '12h', '1D']
        
        for interval in intervals:
            for symbol in CRYPTO_SYMBOLS[:15]:  # Limitar para no sobrecargar
                try:
                    signal = self.check_volume_ema_ftm_signal(symbol, interval)
                    
                    if signal:
                        signal_key = f"{symbol}_{interval}_{signal['signal']}"
                        
                        if signal_key not in self.sent_volume_alerts:
                            signals.append(signal)
                            self.sent_volume_alerts.add(signal_key)
                            print(f"Señal volumen generada: {symbol} {interval} {signal['signal']}")
                    
                except Exception as e:
                    print(f"Error procesando {symbol} {interval}: {e}")
                    continue
        
        return signals

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert(alert_data, strategy_type='multiframe'):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if strategy_type == 'multiframe':
            # Estrategia Multi-Temporalidad
            conditions_text = "\n".join([f"• {cond}" for cond in alert_data.get('fulfilled_conditions', [])])
            
            message = f"""
🚨 MULTI-TF | {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Entrada: ${alert_data['entry']:.6f} | SL: ${alert_data['stop_loss']:.6f}
Score: {alert_data['signal_score']:.1f}% | MA200: {alert_data['ma200_condition'].upper()}

Condiciones:
{conditions_text}
            """
            
        else:
            # Estrategia Volumen + EMA21
            mayor_trend = alert_data['multi_tf_trend'].get('mayor', 'NEUTRAL')
            menor_trend = alert_data['multi_tf_trend'].get('menor', 'NEUTRAL')
            
            message = f"""
🚨 VOL+EMA21 | {alert_data['signal']} | {alert_data['symbol']} | {alert_data['interval']}
Entrada: ${alert_data['price']:.6f} | Vol: {alert_data['volume_ratio']:.1f}x
Filtros: FTMaverick OK | MF: {mayor_trend}/{menor_trend}
            """
        
        # Generar y enviar imagen
        try:
            if strategy_type == 'multiframe':
                img_buffer = generate_telegram_chart_multiframe(alert_data)
            else:
                img_buffer = generate_telegram_chart_volume(alert_data)
            
            if img_buffer:
                asyncio.run(bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=img_buffer,
                    caption=message[:1024]  # Limitar tamaño del caption
                ))
                print(f"Alerta {strategy_type} enviada a Telegram con imagen: {alert_data['symbol']}")
            else:
                asyncio.run(bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message
                ))
                print(f"Alerta {strategy_type} enviada a Telegram sin imagen: {alert_data['symbol']}")
                
        except Exception as img_error:
            print(f"Error enviando imagen a Telegram: {img_error}")
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            ))
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def generate_telegram_chart_multiframe(signal_data):
    """Generar gráfico para Telegram - Estrategia Multi-Temporalidad"""
    try:
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('white')
        
        # 1. Gráfico de Velas con Bollinger y Medias
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
            
            # Bandas de Bollinger (transparentes)
            if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
                bb_dates = dates[-len(signal_data['indicators']['bb_upper']):]
                ax1.fill_between(bb_dates, 
                               signal_data['indicators']['bb_upper'],
                               signal_data['indicators']['bb_lower'],
                               color='blue', alpha=0.1)
            
            # Medias móviles
            if 'indicators' in signal_data:
                ma_dates = dates[-len(signal_data['indicators']['ma_9']):]
                ax1.plot(ma_dates, signal_data['indicators']['ma_9'], 'orange', linewidth=1, label='MA9')
                ax1.plot(ma_dates, signal_data['indicators']['ma_21'], 'blue', linewidth=1, label='MA21')
                ax1.plot(ma_dates, signal_data['indicators']['ma_50'], 'purple', linewidth=2, label='MA50')
                ax1.plot(ma_dates, signal_data['indicators']['ma_200'], 'black', linewidth=3, label='MA200')
            
            # Soporte/Resistencia
            for support in signal_data.get('supports', []):
                ax1.axhline(y=support, color='green', linestyle='--', alpha=0.5)
            
            for resistance in signal_data.get('resistances', []):
                ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5)
        
        ax1.set_title(f'{signal_data["symbol"]} - {signal_data["interval"]}', fontsize=12)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(8, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax2.plot(adx_dates, signal_data['indicators']['adx'], 
                    'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Volumen con anomalías
        ax3 = plt.subplot(8, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_ratio']):]
            volume_values = [d['volume'] for d in signal_data['data'][-len(signal_data['indicators']['volume_ratio']):]]
            
            # Barras de volumen
            colors = ['green' if signal_data['data'][-len(signal_data['indicators']['volume_ratio'])+i]['close'] >= 
                     signal_data['data'][-len(signal_data['indicators']['volume_ratio'])+i]['open'] 
                     else 'red' for i in range(len(volume_values))]
            
            ax3.bar(volume_dates, volume_values, color=colors, alpha=0.7)
            
            # EMA de volumen
            ax3.plot(volume_dates, signal_data['indicators']['volume_ema'], 
                    'orange', linewidth=1, label='EMA Vol')
            
            # Anomalías
            anomaly_dates = []
            anomaly_values = []
            for i, date in enumerate(volume_dates):
                if signal_data['indicators']['volume_anomaly'][i]:
                    anomaly_dates.append(date)
                    anomaly_values.append(volume_values[i])
            
            if anomaly_dates:
                ax3.scatter(anomaly_dates, anomaly_values, color='red', s=30, 
                           label='Anomalías', zorder=5)
        
        ax3.set_ylabel('Volumen')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick (barras)
        ax4 = plt.subplot(8, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            
            colors = ['green' if x > 0 else 'red' for x in trend_strength]
            ax4.bar(trend_dates, trend_strength, color=colors, alpha=0.7)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7)
                ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        
        ax4.set_ylabel('Fuerza Tendencia')
        ax4.grid(True, alpha=0.3)
        
        # 5. Ballenas (barras) - solo si hay datos
        ax5 = plt.subplot(8, 1, 5, sharex=ax1)
        if 'indicators' in signal_data and signal_data['interval'] in ['12h', '1D']:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax5.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.5, label='Compra')
            ax5.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.5, label='Venta')
            ax5.set_ylabel('Ballenas')
            ax5.legend(loc='upper left', fontsize=8)
            ax5.grid(True, alpha=0.3)
        
        # 6. RSI Maverick
        ax6 = plt.subplot(8, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_m_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax6.plot(rsi_m_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=1)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax6.set_ylabel('RSI Maverick')
            ax6.set_ylim(0, 1)
            ax6.grid(True, alpha=0.3)
        
        # 7. RSI Tradicional
        ax7 = plt.subplot(8, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_t_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax7.plot(rsi_t_dates, signal_data['indicators']['rsi_traditional'], 
                    'purple', linewidth=1)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.5)
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax7.set_ylabel('RSI Trad')
            ax7.set_ylim(0, 100)
            ax7.grid(True, alpha=0.3)
        
        # 8. MACD (barras)
        ax8 = plt.subplot(8, 1, 8, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            
            # Histograma como barras
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax8.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6)
            
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax8.set_ylabel('MACD')
            ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando gráfico multiframe: {e}")
        return None

def generate_telegram_chart_volume(signal_data):
    """Generar gráfico para Telegram - Estrategia Volumen"""
    try:
        # Obtener datos adicionales
        df = indicator.get_kucoin_data(signal_data['symbol'], signal_data['interval'], 50)
        if df is None or len(df) < 30:
            return None
        
        close = df['close'].values
        volume = df['volume'].values
        
        # Calcular EMA21 y Volume MA21
        ema_21 = indicator.calculate_ema(close, 21)
        volume_ma_21 = indicator.calculate_ema(volume, 21)
        
        dates = df['timestamp'].tail(50).tolist()
        dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') if isinstance(d, str) else d for d in dates]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.patch.set_facecolor('white')
        
        # Gráfico superior: Velas + EMA21
        opens = df['open'].tail(50).values
        highs = df['high'].tail(50).values
        lows = df['low'].tail(50).values
        closes = df['close'].tail(50).values
        
        for i in range(len(dates)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
        
        # EMA21
        ax1.plot(dates[-len(ema_21):], ema_21[-50:], 'orange', linewidth=2, label='EMA21')
        
        # Destacar vela actual
        if signal_data['signal'] == 'LONG':
            ax1.scatter([dates[-1]], [closes[-1]], color='green', s=100, marker='o', 
                       edgecolors='black', linewidth=2, zorder=5)
        else:
            ax1.scatter([dates[-1]], [closes[-1]], color='red', s=100, marker='o', 
                       edgecolors='black', linewidth=2, zorder=5)
        
        ax1.set_title(f'{signal_data["symbol"]} - {signal_data["interval"]} - Señal {signal_data["signal"]} por Volumen+EMA21')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico inferior: Volumen + Volume MA21
        volume_values = volume[-50:]
        colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(volume_values))]
        
        ax2.bar(dates, volume_values, color=colors, alpha=0.7)
        ax2.plot(dates[-len(volume_ma_21):], volume_ma_21[-50:], 
                'orange', linewidth=2, label='MA21 Volumen')
        
        # Destacar volumen actual
        if signal_data['signal'] == 'LONG':
            ax2.bar([dates[-1]], [volume_values[-1]], color='green', alpha=1.0)
        else:
            ax2.bar([dates[-1]], [volume_values[-1]], color='red', alpha=1.0)
        
        ax2.set_ylabel('Volumen')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando gráfico volumen: {e}")
        return None

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            current_time = datetime.now()
            
            # Estrategia 1: Multi-Temporalidad (todas las temporalidades)
            print("Verificando señales Multi-TF...")
            for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
                for symbol in CRYPTO_SYMBOLS[:12]:  # Limitar para no sobrecargar
                    try:
                        signal_data = indicator.generate_signals_improved(symbol, interval)
                        
                        if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                            signal_data['signal_score'] >= 65):
                            
                            # Evitar duplicados recientes
                            alert_key = f"multiframe_{symbol}_{interval}_{signal_data['signal']}"
                            if alert_key not in indicator.alert_cache:
                                send_telegram_alert(signal_data, 'multiframe')
                                indicator.alert_cache[alert_key] = current_time
                                print(f"Señal Multi-TF enviada: {symbol} {interval} {signal_data['signal']}")
                        
                    except Exception as e:
                        print(f"Error procesando {symbol} {interval}: {e}")
                        continue
            
            # Estrategia 2: Volumen + EMA21
            print("Verificando señales Volumen+EMA21...")
            volume_signals = indicator.generate_volume_ema_signals()
            for signal in volume_signals:
                try:
                    alert_key = f"volume_{signal['symbol']}_{signal['interval']}_{signal['signal']}"
                    if alert_key not in indicator.alert_cache:
                        send_telegram_alert(signal, 'volume')
                        indicator.alert_cache[alert_key] = current_time
                        print(f"Señal Volumen enviada: {signal['symbol']} {signal['interval']} {signal['signal']}")
                except Exception as e:
                    print(f"Error enviando señal volumen: {e}")
                    continue
            
            # Limpiar cache antiguo (más de 1 hora)
            old_keys = [k for k, v in indicator.alert_cache.items() if (current_time - v).seconds > 3600]
            for key in old_keys:
                del indicator.alert_cache[key]
            
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
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 30 +
                        (signal_data['rsi_maverick'] * 20) +
                        (1 if signal_data['adx'] > 25 else 0) * 20 +
                        (min(1, signal_data['volume'] / max(1, signal_data['volume_ma'])) * 30)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 30 +
                        ((1 - signal_data['rsi_maverick']) * 20) +
                        (1 if signal_data['adx'] > 25 else 0) * 20 +
                        (min(1, signal_data['volume'] / max(1, signal_data['volume_ma'])) * 30)
                    ))
                    
                    # Ajustar según señal
                    if signal_data['signal'] == 'LONG':
                        buy_pressure = min(100, buy_pressure * 1.2)
                        sell_pressure = max(0, sell_pressure * 0.8)
                    elif signal_data['signal'] == 'SHORT':
                        sell_pressure = min(100, sell_pressure * 1.2)
                        buy_pressure = max(0, buy_pressure * 0.8)
                    
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

@app.route('/api/volume_ema_signals')
def get_volume_ema_signals():
    """Endpoint para obtener señales de volumen"""
    try:
        signals = indicator.generate_volume_ema_signals()
        return jsonify({'signals': signals})
        
    except Exception as e:
        print(f"Error en /api/volume_ema_signals: {e}")
        return jsonify({'signals': []})

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
            
            # Velas japonesas
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Medias móviles
            if 'indicators' in signal_data:
                ma_dates = dates[-len(signal_data['indicators']['ma_9']):]
                ax1.plot(ma_dates, signal_data['indicators']['ma_9'], 'orange', linewidth=1, label='MA9', alpha=0.7)
                ax1.plot(ma_dates, signal_data['indicators']['ma_21'], 'blue', linewidth=1, label='MA21', alpha=0.7)
                ax1.plot(ma_dates, signal_data['indicators']['ma_50'], 'purple', linewidth=1, label='MA50', alpha=0.7)
                ax1.plot(ma_dates, signal_data['indicators']['ma_200'], 'black', linewidth=2, label='MA200', alpha=0.7)
            
            # Bandas de Bollinger
            if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
                bb_dates = dates[-len(signal_data['indicators']['bb_upper']):]
                ax1.fill_between(bb_dates, 
                               signal_data['indicators']['bb_upper'],
                               signal_data['indicators']['bb_lower'],
                               color='gray', alpha=0.2)
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            
            for i, tp in enumerate(signal_data['take_profit']):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
            
            # Soporte/Resistencia
            for support in signal_data.get('supports', []):
                ax1.axhline(y=support, color='green', linestyle=':', alpha=0.5)
            
            for resistance in signal_data.get('resistances', []):
                ax1.axhline(y=resistance, color='red', linestyle=':', alpha=0.5)
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: ADX/DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax2.plot(adx_dates, signal_data['indicators']['adx'], 
                    'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Umbral 25')
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Volumen
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_ratio']):]
            volume_values = [d['volume'] for d in signal_data['data'][-len(signal_data['indicators']['volume_ratio']):]]
            
            # Barras de volumen coloreadas
            colors = ['green' if signal_data['data'][-len(signal_data['indicators']['volume_ratio'])+i]['close'] >= 
                     signal_data['data'][-len(signal_data['indicators']['volume_ratio'])+i]['open'] 
                     else 'red' for i in range(len(volume_values))]
            
            ax3.bar(volume_dates, volume_values, color=colors, alpha=0.7, label='Volumen')
            
            # EMA de volumen
            ax3.plot(volume_dates, signal_data['indicators']['volume_ema'], 
                    'orange', linewidth=2, label='EMA Volumen')
            
            # Anomalías
            anomaly_dates = []
            anomaly_values = []
            for i, date in enumerate(volume_dates):
                if signal_data['indicators']['volume_anomaly'][i]:
                    anomaly_dates.append(date)
                    anomaly_values.append(volume_values[i])
            
            if anomaly_dates:
                ax3.scatter(anomaly_dates, anomaly_values, color='red', s=50, 
                           label='Anomalías', zorder=5)
        ax3.set_ylabel('Volumen')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Fuerza de Tendencia (barras)
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax4.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax4.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
        ax4.set_ylabel('Fuerza Tendencia %')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Ballenas (barras)
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax5.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.6, label='Compradoras')
            ax5.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.6, label='Vendedoras')
            
            # Señales confirmadas
            buy_dates = []
            buy_values = []
            sell_dates = []
            sell_values = []
            
            for i, date in enumerate(whale_dates):
                if i < len(signal_data['indicators']['confirmed_buy']) and signal_data['indicators']['confirmed_buy'][i]:
                    buy_dates.append(date)
                    buy_values.append(signal_data['indicators']['whale_pump'][i])
                if i < len(signal_data['indicators']['confirmed_sell']) and signal_data['indicators']['confirmed_sell'][i]:
                    sell_dates.append(date)
                    sell_values.append(signal_data['indicators']['whale_dump'][i])
            
            if buy_dates:
                ax5.scatter(buy_dates, buy_values, color='lime', s=80, marker='^', 
                           label='Compra Confirmada', zorder=5)
            if sell_dates:
                ax5.scatter(sell_dates, sell_values, color='darkred', s=80, marker='v', 
                           label='Venta Confirmada', zorder=5)
            
        ax5.set_ylabel('Ballenas')
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: RSI Maverick
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_m_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax6.plot(rsi_m_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        ax6.set_ylabel('RSI Maverick')
        ax6.set_ylim(0, 1)
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: RSI Tradicional
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_t_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax7.plot(rsi_t_dates, signal_data['indicators']['rsi_traditional'], 
                    'purple', linewidth=2, label='RSI Tradicional')
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax7.set_ylabel('RSI Tradicional')
        ax7.set_ylim(0, 100)
        ax7.legend(loc='upper left', fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: MACD
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax8.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax8.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            # Histograma como barras
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax8.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6, label='Histograma')
            
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax8.set_ylabel('MACD')
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
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
        """
        
        ax9.text(0.1, 0.9, signal_info, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, facecolor='white', bbox_inches='tight')
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
