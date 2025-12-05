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
        self.sent_volume_ema_signals = set()
        self.sent_multi_signals = set()
        
        # Cache para indicadores complementarios (7 velas)
        self.complementary_cache = {
            'ma_cross': {},
            'dmi_cross': {},
            'macd_cross': {},
            'rsi_maverick_div': {},
            'rsi_trad_div': {},
            'chart_pattern': {},
            'breakout': {}
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

    def calculate_support_resistance(self, high, low, close, period=50):
        """Calcular soportes y resistencias dinámicos"""
        try:
            n = len(close)
            pivot_points = []
            
            # Encontrar pivots (máximos y mínimos locales)
            for i in range(period, n - period):
                if high[i] == np.max(high[i-period:i+period+1]):
                    pivot_points.append(('resistance', i, high[i]))
                if low[i] == np.min(low[i-period:i+period+1]):
                    pivot_points.append(('support', i, low[i]))
            
            # Agrupar pivots cercanos
            support_levels = []
            resistance_levels = []
            
            for pivot_type, idx, value in pivot_points:
                if pivot_type == 'support':
                    found = False
                    for level in support_levels:
                        if abs(value - level['value']) / level['value'] < 0.02:  # 2% de tolerancia
                            level['values'].append(value)
                            level['indices'].append(idx)
                            found = True
                            break
                    if not found:
                        support_levels.append({
                            'value': value,
                            'values': [value],
                            'indices': [idx],
                            'strength': 1
                        })
                else:
                    found = False
                    for level in resistance_levels:
                        if abs(value - level['value']) / level['value'] < 0.02:
                            level['values'].append(value)
                            level['indices'].append(idx)
                            found = True
                            break
                    if not found:
                        resistance_levels.append({
                            'value': value,
                            'values': [value],
                            'indices': [idx],
                            'strength': 1
                        })
            
            # Calcular fortaleza basada en toques
            for level in support_levels:
                level['strength'] = len(level['values'])
                level['value'] = np.mean(level['values'])
            
            for level in resistance_levels:
                level['strength'] = len(level['values'])
                level['value'] = np.mean(level['values'])
            
            # Ordenar por fortaleza y tomar los más fuertes (4-6 niveles)
            support_levels.sort(key=lambda x: (-x['strength'], -x['indices'][-1]))
            resistance_levels.sort(key=lambda x: (-x['strength'], -x['indices'][-1]))
            
            # Seleccionar 4-6 niveles más relevantes
            supports = [s['value'] for s in support_levels[:min(4, len(support_levels))]]
            resistances = [r['value'] for r in resistance_levels[:min(4, len(resistance_levels))]]
            
            # Asegurar al menos 4 niveles
            if len(supports) < 4:
                current_low = np.min(low[-period:])
                for i in range(4 - len(supports)):
                    supports.append(current_low * (1 - 0.01 * (i + 1)))
            
            if len(resistances) < 4:
                current_high = np.max(high[-period:])
                for i in range(4 - len(resistances)):
                    resistances.append(current_high * (1 + 0.01 * (i + 1)))
            
            # Ordenar
            supports.sort()
            resistances.sort()
            
            return supports, resistances
            
        except Exception as e:
            print(f"Error calculando soportes/resistencias: {e}")
            current_price = close[-1] if len(close) > 0 else 100
            supports = [current_price * 0.95, current_price * 0.90, current_price * 0.85, current_price * 0.80]
            resistances = [current_price * 1.05, current_price * 1.10, current_price * 1.15, current_price * 1.20]
            return supports, resistances

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15, supports=None, resistances=None):
        """Calcular entradas y salidas óptimas con soportes/resistencias"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular soportes y resistencias si no se proporcionan
            if supports is None or resistances is None:
                supports, resistances = self.calculate_support_resistance(high, low, close)
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Entrada en el soporte más cercano por debajo del precio actual
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)  # Soporte más fuerte más cercano
                else:
                    entry = min(supports)  # Si no hay soportes por debajo, tomar el más bajo
                
                # Stop loss debajo del siguiente soporte
                sorted_supports = sorted(supports)
                entry_idx = np.searchsorted(sorted_supports, entry)
                if entry_idx > 0:
                    stop_loss = sorted_supports[entry_idx - 1] * 0.99
                else:
                    stop_loss = entry * 0.97
                
                # Take profit en resistencias superiores
                take_profits = []
                for res in sorted(resistances):
                    if res > entry:
                        take_profits.append(res)
                        if len(take_profits) >= 2:  # Máximo 2 TPs
                            break
                
                if not take_profits:
                    take_profits = [entry * 1.03]
                
            else:  # SHORT
                # Entrada en la resistencia más cercana por encima del precio actual
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)  # Resistencia más cercana
                else:
                    entry = max(resistances)  # Si no hay resistencias por encima, tomar la más alta
                
                # Stop loss encima de la siguiente resistencia
                sorted_resistances = sorted(resistances)
                entry_idx = np.searchsorted(sorted_resistances, entry)
                if entry_idx < len(sorted_resistances) - 1:
                    stop_loss = sorted_resistances[entry_idx + 1] * 1.01
                else:
                    stop_loss = entry * 1.03
                
                # Take profit en soportes inferiores
                take_profits = []
                for sup in sorted(supports, reverse=True):
                    if sup < entry:
                        take_profits.append(sup)
                        if len(take_profits) >= 2:  # Máximo 2 TPs
                            break
                
                if not take_profits:
                    take_profits = [entry * 0.97]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits],
                'supports': [float(s) for s in supports],
                'resistances': [float(r) for r in resistances],
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
                'supports': [current_price * 0.95, current_price * 0.90, current_price * 0.85, current_price * 0.80],
                'resistances': [current_price * 1.05, current_price * 1.10, current_price * 1.15, current_price * 1.20],
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

    def check_bollinger_conditions(self, df, interval, signal_type):
        """Verificar condiciones de Bandas de Bollinger"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            avg_volume = np.mean(volume[-20:])
            
            # Condiciones para LONG
            if signal_type == 'LONG':
                # Precio toca o está cerca de la banda inferior
                touch_lower = current_price <= bb_lower[current_idx] * 1.02
                # Precio rompe la banda media hacia arriba
                break_middle = current_price > bb_middle[current_idx]
                # Precio rebota desde la banda inferior
                bounce_lower = (current_price > bb_lower[current_idx] and 
                               close[current_idx-1] <= bb_lower[current_idx-1] * 1.01)
                
                return touch_lower or break_middle or bounce_lower
                
            else:  # SHORT
                # Precio toca o está cerca de la banda superior
                touch_upper = current_price >= bb_upper[current_idx] * 0.98
                # Precio rompe la banda media hacia abajo
                break_middle = current_price < bb_middle[current_idx]
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

    def detect_divergence(self, price, indicator, lookback=14, confirmation_period=4):
        """Detectar divergencias con confirmación de 7 velas"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-confirmation_period):
            price_window = price[i-lookback:i+confirmation_period]
            indicator_window = indicator[i-lookback:i+confirmation_period]
            
            # Buscar divergencia alcista
            price_lows = []
            indicator_lows = []
            
            for j in range(confirmation_period):
                current_idx = i + j
                if current_idx >= n:
                    break
                    
                # Buscar mínimo en precio
                price_min_idx = np.argmin(price_window[:lookback])
                indicator_value = indicator_window[price_min_idx]
                
                if (price[current_idx] < price_window[price_min_idx] and 
                    indicator[current_idx] > indicator_value):
                    bullish_div[current_idx] = True
                    break
            
            # Buscar divergencia bajista
            for j in range(confirmation_period):
                current_idx = i + j
                if current_idx >= n:
                    break
                    
                # Buscar máximo en precio
                price_max_idx = np.argmax(price_window[:lookback])
                indicator_value = indicator_window[price_max_idx]
                
                if (price[current_idx] > price_window[price_max_idx] and 
                    indicator[current_idx] < indicator_value):
                    bearish_div[current_idx] = True
                    break
        
        # Mantener señal activa por 7 velas
        for i in range(n):
            if i >= 7:
                if np.any(bullish_div[i-7:i]):
                    bullish_div[i] = True
                if np.any(bearish_div[i-7:i]):
                    bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def check_breakout(self, high, low, close, supports, resistances):
        """Detectar rupturas de soportes/resistencias"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            for resistance in resistances:
                if close[i] > resistance and high[i] > high[i-1]:
                    breakout_up[i] = True
                    break
            
            for support in supports:
                if close[i] < support and low[i] < low[i-1]:
                    breakout_down[i] = True
                    break
        
        # Mantener señal activa por 1 vela
        for i in range(1, n):
            if breakout_up[i-1]:
                breakout_up[i] = True
            if breakout_down[i-1]:
                breakout_down[i] = True
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI con confirmación de 1 vela"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                di_cross_bullish[i] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
        
        # Mantener señal activa por 1 vela
        for i in range(1, n):
            if di_cross_bullish[i-1]:
                di_cross_bullish[i] = True
            if di_cross_bearish[i-1]:
                di_cross_bearish[i] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

    def check_ma_crossover(self, ma_short, ma_long):
        """Detectar cruce de medias móviles con confirmación de 1 vela"""
        n = len(ma_short)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MA corta cruza por encima de MA larga
            if (ma_short[i] > ma_long[i] and 
                ma_short[i-1] <= ma_long[i-1]):
                ma_cross_bullish[i] = True
            
            # Cruce bajista: MA corta cruza por debajo de MA larga
            if (ma_short[i] < ma_long[i] and 
                ma_short[i-1] >= ma_long[i-1]):
                ma_cross_bearish[i] = True
        
        # Mantener señal activa por 1 vela
        for i in range(1, n):
            if ma_cross_bullish[i-1]:
                ma_cross_bullish[i] = True
            if ma_cross_bearish[i-1]:
                ma_cross_bearish[i] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()

    def check_macd_crossover(self, macd, signal):
        """Detectar cruce de MACD con confirmación de 1 vela"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MACD cruza por encima de señal
            if (macd[i] > signal[i] and 
                macd[i-1] <= signal[i-1]):
                macd_cross_bullish[i] = True
            
            # Cruce bajista: MACD cruza por debajo de señal
            if (macd[i] < signal[i] and 
                macd[i-1] >= signal[i-1]):
                macd_cross_bearish[i] = True
        
        # Mantener señal activa por 1 vela
        for i in range(1, n):
            if macd_cross_bullish[i-1]:
                macd_cross_bullish[i] = True
            if macd_cross_bearish[i-1]:
                macd_cross_bearish[i] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()

    def check_adx_slope(self, adx, period=3):
        """Verificar pendiente positiva del ADX"""
        n = len(adx)
        adx_slope_positive = np.zeros(n, dtype=bool)
        
        for i in range(period, n):
            if i >= period:
                current_adx = adx[i]
                previous_adx = np.mean(adx[i-period:i])
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

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo con confirmación de 7 velas"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'triple_top': np.zeros(n, dtype=bool),
            'triple_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-7):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            
            # Doble Techo
            peaks = []
            for j in range(1, len(window_high)-1):
                if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                    peaks.append((j, window_high[j]))
            
            if len(peaks) >= 2:
                last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                    patterns['double_top'][i] = True
            
            # Doble Fondo
            troughs = []
            for j in range(1, len(window_low)-1):
                if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                    troughs.append((j, window_low[j]))
            
            if len(troughs) >= 2:
                last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                    patterns['double_bottom'][i] = True
            
            # Bandera Alcista (simplificada)
            if i > 20:
                prev_high = np.max(high[i-20:i-10])
                prev_low = np.min(low[i-20:i-10])
                current_range = np.max(high[i-10:i+1]) - np.min(low[i-10:i+1])
                
                if (prev_high - prev_low) > current_range * 3:
                    patterns['bullish_flag'][i] = True
            
            # Bandera Bajista (simplificada)
            if i > 20:
                prev_high = np.max(high[i-20:i-10])
                prev_low = np.min(low[i-20:i-10])
                current_range = np.max(high[i-10:i+1]) - np.min(low[i-10:i+1])
                
                if (prev_high - prev_low) > current_range * 3:
                    patterns['bearish_flag'][i] = True
        
        # Mantener señal activa por 7 velas
        for pattern_name in patterns:
            pattern_array = patterns[pattern_name]
            for i in range(n):
                if i >= 7:
                    if np.any(pattern_array[i-7:i]):
                        pattern_array[i] = True
        
        return patterns

    def calculate_volume_anomaly_improved(self, volume, close, period=20, std_multiplier=2):
        """Calcular anomalías de volumen mejoradas (compra/venta)"""
        try:
            n = len(volume)
            volume_anomaly_buy = np.zeros(n, dtype=bool)
            volume_anomaly_sell = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_ma_21 = self.calculate_sma(volume, 21)
            
            for i in range(period, n):
                # Media móvil de volumen
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
                
                # Determinar si es compra o venta basado en precio de cierre
                if i > 0:
                    price_change = (close[i] - close[i-1]) / close[i-1] * 100
                    
                    # Anomalía de COMPRA: volumen alto + precio subiendo
                    if (volume_ratio[i] > 1 + (std_multiplier * (std_volume / current_ema if current_ema > 0 else 0)) and
                        price_change > 0.5):
                        volume_anomaly_buy[i] = True
                    
                    # Anomalía de VENTA: volumen alto + precio bajando
                    elif (volume_ratio[i] > 1 + (std_multiplier * (std_volume / current_ema if current_ema > 0 else 0)) and
                          price_change < -0.5):
                        volume_anomaly_sell[i] = True
                
                # Detectar clusters (múltiples anomalías en 5-10 periodos)
                if i >= 10:
                    recent_buy_anomalies = volume_anomaly_buy[max(0, i-9):i+1]
                    recent_sell_anomalies = volume_anomaly_sell[max(0, i-9):i+1]
                    
                    if np.sum(recent_buy_anomalies) >= 3 or np.sum(recent_sell_anomalies) >= 3:
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly_buy': volume_anomaly_buy.tolist(),
                'volume_anomaly_sell': volume_anomaly_sell.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ma_21': volume_ma_21.tolist(),
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
                'volume_ma_21': [0] * n,
                'volume_ema': [0] * n
            }

    def get_complementary_indicator_value(self, indicator_key, symbol, interval, current_idx):
        """Obtener valor de indicador complementario con cache de 7 velas"""
        cache_key = f"{symbol}_{interval}_{indicator_key}"
        
        if cache_key in self.complementary_cache[indicator_key]:
            cached_data, timestamp = self.complementary_cache[indicator_key][cache_key]
            if (datetime.now() - timestamp).seconds < 300:  # 5 minutos de cache
                return cached_data.get(current_idx, False)
        
        # Si no hay cache o expiró, retornar False
        return False

    def evaluate_signal_conditions_optimized(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con PESOS OPTIMIZADOS"""
        # Definir pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            weights = {
                'long': {
                    'multi_timeframe': 30,  # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                },
                'short': {
                    'multi_timeframe': 30,  # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
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
                    'whale_signal': 30,     # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                },
                'short': {
                    'whale_signal': 30,     # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
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
                    'trend_strength': 55,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                },
                'short': {
                    'trend_strength': 55,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
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
        
        # Cruce DMI (+DI > -DI)
        conditions['long']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['plus_di'][current_idx] > data['minus_di'][current_idx] and
            data['di_cross_bullish'][current_idx]
        )
        
        # Cruce Medias Móviles
        conditions['long']['ma_cross']['value'] = data['ma_cross_bullish'][current_idx]
        
        # Divergencias RSI
        conditions['long']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bullish_divergence']) and 
            data['rsi_bullish_divergence'][current_idx]
        )
        
        conditions['long']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bullish_divergence']) and 
            data['rsi_maverick_bullish_divergence'][current_idx]
        )
        
        # Cruce MACD
        conditions['long']['macd']['value'] = data['macd_cross_bullish'][current_idx]
        
        # Patrones Chartistas
        conditions['long']['chart_pattern']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_flag'][current_idx] or
            data['chart_patterns']['triple_bottom'][current_idx]
        )
        
        # Rupturas
        conditions['long']['breakout']['value'] = (
            current_idx < len(data['breakout_up']) and 
            data['breakout_up'][current_idx]
        )
        
        # Volumen Anómalo (compra)
        conditions['long']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly_buy']) and 
            data['volume_anomaly_buy'][current_idx]
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
        
        # Cruce DMI (-DI > +DI)
        conditions['short']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['minus_di'][current_idx] > data['plus_di'][current_idx] and
            data['di_cross_bearish'][current_idx]
        )
        
        # Cruce Medias Móviles
        conditions['short']['ma_cross']['value'] = data['ma_cross_bearish'][current_idx]
        
        # Divergencias RSI
        conditions['short']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bearish_divergence']) and 
            data['rsi_bearish_divergence'][current_idx]
        )
        
        conditions['short']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bearish_divergence']) and 
            data['rsi_maverick_bearish_divergence'][current_idx]
        )
        
        # Cruce MACD
        conditions['short']['macd']['value'] = data['macd_cross_bearish'][current_idx]
        
        # Patrones Chartistas
        conditions['short']['chart_pattern']['value'] = (
            data['chart_patterns']['head_shoulders'][current_idx] or
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['triple_top'][current_idx] or
            data['chart_patterns']['bearish_flag'][current_idx]
        )
        
        # Rupturas
        conditions['short']['breakout']['value'] = (
            current_idx < len(data['breakout_down']) and 
            data['breakout_down'][current_idx]
        )
        
        # Volumen Anómalo (venta)
        conditions['short']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly_sell']) and 
            data['volume_anomaly_sell'][current_idx]
        )
        
        return conditions

    def get_condition_description(self, condition_key):
        """Obtener descripción de condición"""
        descriptions = {
            'multi_timeframe': 'Multi-TF Confirmado',
            'trend_strength': 'Fuerza Tendencia Favorable',
            'whale_signal': 'Señal Ballenas Confirmada',
            'bollinger_bands': 'Bandas de Bollinger',
            'adx_dmi': 'Cruce DMI + ADX > 25',
            'ma_cross': 'Cruce MA9/MA21',
            'rsi_traditional_divergence': 'Divergencia RSI Tradicional',
            'rsi_maverick_divergence': 'Divergencia RSI Maverick',
            'macd': 'Cruce MACD',
            'chart_pattern': 'Patrón Chartista',
            'breakout': 'Ruptura S/R',
            'volume_anomaly': 'Volumen Anómalo'
        }
        return descriptions.get(condition_key, condition_key)

    def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal basada en condiciones ponderadas"""
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

    # NUEVA ESTRATEGIA: Desplome por Volumen + EMA21
    def check_volume_ema_ftm_signal(self, symbol, interval):
        """Verificar señal de Desplome por Volumen + EMA21 con filtros FTMaverick/Multi-Timeframe"""
        try:
            # Solo para temporalidades 1H, 4H, 12H, 1D
            if interval not in ['1h', '4h', '12h', '1D']:
                return None
            
            # Verificar timing de ejecución
            current_time = self.get_bolivia_time()
            should_check = False
            
            if interval == '1h':
                # Revisar desde el 50% del tiempo de vela, cada 300 segundos
                next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                remaining_seconds = (next_close - current_time).total_seconds()
                if remaining_seconds <= 1800:  # 50% de 3600
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
                remaining_seconds = (next_4h_close - current_time).total_seconds()
                if remaining_seconds <= 3600:  # 25% de 14400
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
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            
            # Calcular EMA21 y Volume MA21
            ema_21 = self.calculate_ema(close, 21)
            volume_ma_21 = self.calculate_sma(volume, 21)
            
            current_ema_21 = ema_21[current_idx] if current_idx < len(ema_21) else 0
            current_volume_ma_21 = volume_ma_21[current_idx] if current_idx < len(volume_ma_21) else 0
            
            # Condición A: Volumen y EMA
            volume_ratio = current_volume / current_volume_ma_21 if current_volume_ma_21 > 0 else 1
            condition_a_long = (volume_ratio > 2.5 and current_price > current_ema_21)
            condition_a_short = (volume_ratio > 2.5 and current_price < current_ema_21)
            
            if not condition_a_long and not condition_a_short:
                return None
            
            # Condición B: Filtro FTMaverick (Timeframe Actual)
            ftm_current = self.calculate_trend_strength_maverick(close)
            condition_b = not ftm_current['no_trade_zones'][current_idx]
            
            if not condition_b:
                return None
            
            # Condición C: Filtro Multi-Timeframe
            condition_c_long = False
            condition_c_short = False
            
            if interval in ['1h', '4h']:
                hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                
                # Timeframe Mayor
                if hierarchy.get('mayor'):
                    df_mayor = self.get_kucoin_data(symbol, hierarchy['mayor'], 50)
                    if df_mayor is not None and len(df_mayor) > 20:
                        mayor_trend = self.check_multi_timeframe_trend(symbol, interval)
                        mayor_condition = mayor_trend.get('mayor', 'NEUTRAL')
                        
                        # Timeframe Menor
                        if hierarchy.get('menor'):
                            df_menor = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
                            if df_menor is not None and len(df_menor) > 10:
                                menor_ftm = self.calculate_trend_strength_maverick(df_menor['close'].values)
                                menor_signal = menor_ftm['strength_signals'][-1]
                                
                                if condition_a_long:
                                    condition_c_long = (
                                        mayor_condition in ['BULLISH', 'NEUTRAL'] and
                                        menor_signal in ['STRONG_UP', 'WEAK_UP']
                                    )
                                
                                if condition_a_short:
                                    condition_c_short = (
                                        mayor_condition in ['BEARISH', 'NEUTRAL'] and
                                        menor_signal in ['STRONG_DOWN', 'WEAK_DOWN']
                                    )
            
            # Para 12h y 1D no hay multi-timeframe obligatorio
            elif interval in ['12h', '1D']:
                condition_c_long = condition_a_long
                condition_c_short = condition_a_short
            
            # Determinar señal
            signal_type = None
            if condition_a_long and condition_b and condition_c_long:
                signal_type = 'LONG'
            elif condition_a_short and condition_b and condition_c_short:
                signal_type = 'SHORT'
            
            if not signal_type:
                return None
            
            # Calcular niveles
            supports, resistances = self.calculate_support_resistance(
                df['high'].values, df['low'].values, close
            )
            
            if signal_type == 'LONG':
                # Entrada en soporte más cercano
                valid_supports = [s for s in supports if s < current_price]
                entry = max(valid_supports) if valid_supports else current_price * 0.99
            else:  # SHORT
                # Entrada en resistencia más cercana
                valid_resistances = [r for r in resistances if r > current_price]
                entry = min(valid_resistances) if valid_resistances else current_price * 1.01
            
            # Generar ID único
            signal_id = f"VOL_EMA21_{symbol}_{interval}_{signal_type}_{int(time.time())}"
            
            # Verificar si ya se envió esta señal recientemente
            if signal_id in self.sent_volume_ema_signals:
                return None
            
            return {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'entry': float(entry),
                'current_price': float(current_price),
                'volume_ratio': float(volume_ratio),
                'ema_21': float(current_ema_21),
                'volume_ma_21': float(current_volume_ma_21),
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'signal_id': signal_id,
                'supports': [float(s) for s in supports],
                'resistances': [float(r) for r in resistances]
            }
            
        except Exception as e:
            print(f"Error en check_volume_ema_ftm_signal para {symbol} {interval}: {e}")
            return None

    def generate_telegram_chart_multi_strategy(self, signal_data, strategy_type='multi'):
        """Generar gráfico para Telegram de la estrategia Multi-Timeframe"""
        try:
            symbol = signal_data['symbol']
            interval = signal_data['interval']
            signal_type = signal_data['signal']
            
            # Obtener datos
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calcular todos los indicadores necesarios
            # 1. Velas japonesas con Bollinger y MA
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # 2. ADX con DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            # 3. Volumen con anomalías
            volume_data = self.calculate_volume_anomaly_improved(volume, close)
            
            # 4. Fuerza de Tendencia Maverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            # 5. Ballenas (solo para 12h y 1D)
            whale_data = None
            if interval in ['12h', '1D']:
                whale_data = self.calculate_whale_signals_improved(df)
            
            # 6. RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # 7. RSI Tradicional
            rsi_trad = self.calculate_rsi(close)
            
            # 8. MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Crear figura con 8 subplots
            fig = plt.figure(figsize=(14, 24))
            
            # 1. Gráfico de Velas (con Bollinger transparente y MAs)
            ax1 = plt.subplot(8, 1, 1)
            dates = pd.date_range(end=datetime.now(), periods=len(close), freq=interval)
            
            # Dibujar velas
            for i in range(len(dates)):
                color = 'green' if close[i] >= (df['open'].values[i] if i < len(df) else close[i]) else 'red'
                ax1.plot([dates[i], dates[i]], [low[i], high[i]], color='black', linewidth=0.5)
                ax1.plot([dates[i], dates[i]], 
                        [df['open'].values[i] if i < len(df) else close[i], close[i]], 
                        color=color, linewidth=2)
            
            # Bandas de Bollinger transparentes
            ax1.fill_between(dates[-50:], bb_upper[-50:], bb_lower[-50:], 
                           alpha=0.1, color='gray', label='Bollinger Bands')
            
            # Medias móviles
            ax1.plot(dates[-50:], ma_9[-50:], 'orange', linewidth=1, label='MA9')
            ax1.plot(dates[-50:], ma_21[-50:], 'blue', linewidth=1, label='MA21')
            ax1.plot(dates[-50:], ma_50[-50:], 'purple', linewidth=1, label='MA50')
            ax1.plot(dates[-50:], ma_200[-50:], 'black', linewidth=2, label='MA200')
            
            # Soporte/Resistencias
            if 'supports' in signal_data:
                for support in signal_data['supports'][:4]:
                    ax1.axhline(y=support, color='blue', linestyle='--', alpha=0.5, linewidth=0.5)
            
            if 'resistances' in signal_data:
                for resistance in signal_data['resistances'][:4]:
                    ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
            
            ax1.set_title(f'{symbol} - {interval} - Velas Japonesas', fontsize=10)
            ax1.set_ylabel('Precio')
            ax1.legend(fontsize=6, loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. ADX con DMI (ADX negro, +DI verde, -DI rojo)
            ax2 = plt.subplot(8, 1, 2, sharex=ax1)
            ax2.plot(dates[-50:], adx[-50:], 'black', linewidth=2, label='ADX')
            ax2.plot(dates[-50:], plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax2.plot(dates[-50:], minus_di[-50:], 'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('ADX/DMI')
            ax2.legend(fontsize=6, loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # 3. Volumen con Anomalías (barras)
            ax3 = plt.subplot(8, 1, 3, sharex=ax1)
            # Barras de volumen
            volume_colors = []
            for i in range(len(dates[-50:])):
                idx = -50 + i
                if idx >= 0 and idx < len(volume_data['volume_anomaly_buy']):
                    if volume_data['volume_anomaly_buy'][idx]:
                        volume_colors.append('green')
                    elif volume_data['volume_anomaly_sell'][idx]:
                        volume_colors.append('red')
                    else:
                        volume_colors.append('gray')
                else:
                    volume_colors.append('gray')
            
            ax3.bar(dates[-50:], volume[-50:], color=volume_colors, alpha=0.6, width=0.8)
            ax3.plot(dates[-50:], volume_data['volume_ma_21'][-50:], 'orange', linewidth=1, label='MA21 Vol')
            ax3.set_ylabel('Volumen')
            ax3.legend(fontsize=6, loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            # 4. Fuerza de Tendencia Maverick (barras)
            ax4 = plt.subplot(8, 1, 4, sharex=ax1)
            ftm_colors = ftm_data['colors'][-50:]
            ax4.bar(dates[-50:], ftm_data['trend_strength'][-50:], color=ftm_colors, alpha=0.7, width=0.8)
            ax4.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.set_ylabel('FT Maverick %')
            ax4.grid(True, alpha=0.3)
            
            # 5. Ballenas Compradoras/Vendedoras (solo para 12h, 1D)
            if whale_data and interval in ['12h', '1D']:
                ax5 = plt.subplot(8, 1, 5, sharex=ax1)
                ax5.bar(dates[-50:], whale_data['whale_pump'][-50:], color='green', alpha=0.6, width=0.8, label='Compra')
                ax5.bar(dates[-50:], whale_data['whale_dump'][-50:], color='red', alpha=0.6, width=0.8, label='Venta')
                ax5.set_ylabel('Ballenas')
                ax5.legend(fontsize=6, loc='upper left')
                ax5.grid(True, alpha=0.3)
            else:
                # Espacio vacío
                ax5 = plt.subplot(8, 1, 5, sharex=ax1)
                ax5.text(0.5, 0.5, 'Indicador Ballenas: Solo para 12H y 1D', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax5.transAxes, fontsize=8)
                ax5.set_ylabel('Ballenas')
                ax5.grid(True, alpha=0.3)
            
            # 6. RSI Maverick
            ax6 = plt.subplot(8, 1, 6, sharex=ax1)
            ax6.plot(dates[-50:], rsi_maverick[-50:], 'blue', linewidth=1)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax6.set_ylabel('RSI Maverick')
            ax6.set_ylim(0, 1)
            ax6.grid(True, alpha=0.3)
            
            # 7. RSI Tradicional
            ax7 = plt.subplot(8, 1, 7, sharex=ax1)
            ax7.plot(dates[-50:], rsi_trad[-50:], 'purple', linewidth=1)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.5)
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax7.set_ylabel('RSI Trad')
            ax7.set_ylim(0, 100)
            ax7.grid(True, alpha=0.3)
            
            # 8. MACD con Histograma (barras)
            ax8 = plt.subplot(8, 1, 8, sharex=ax1)
            ax8.plot(dates[-50:], macd[-50:], 'blue', linewidth=1, label='MACD')
            ax8.plot(dates[-50:], macd_signal[-50:], 'red', linewidth=1, label='Señal')
            
            # Histograma como barras
            macd_colors = ['green' if x > 0 else 'red' for x in macd_histogram[-50:]]
            ax8.bar(dates[-50:], macd_histogram[-50:], color=macd_colors, alpha=0.6, width=0.8)
            
            ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax8.set_ylabel('MACD')
            ax8.set_xlabel('Fecha/Hora')
            ax8.legend(fontsize=6, loc='upper left')
            ax8.grid(True, alpha=0.3)
            
            plt.suptitle(f'{symbol} - {interval} - Señal {signal_type} (Multi-Timeframe)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # Guardar imagen
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando gráfico multi-strategy para {signal_data['symbol']}: {e}")
            return None

    def generate_telegram_chart_volume_ema(self, signal_data):
        """Generar gráfico para Telegram de la estrategia Volumen+EMA21"""
        try:
            symbol = signal_data['symbol']
            interval = signal_data['interval']
            signal_type = signal_data['signal']
            
            # Obtener datos
            df = self.get_kucoin_data(symbol, interval, 50)
            if df is None or len(df) < 30:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Calcular EMA21 y Volume MA21
            ema_21 = self.calculate_ema(close, 21)
            volume_ma_21 = self.calculate_sma(volume, 21)
            
            dates = pd.date_range(end=datetime.now(), periods=len(close), freq=interval)
            
            # Crear figura doble
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Gráfico superior: Velas + EMA21
            current_idx = -1
            
            # Dibujar velas
            for i in range(max(0, len(dates)-30), len(dates)):
                if i < len(df):
                    color = 'green' if close[i] >= df['open'].values[i] else 'red'
                    ax1.plot([dates[i], dates[i]], 
                            [df['low'].values[i], df['high'].values[i]], 
                            color='black', linewidth=0.5)
                    ax1.plot([dates[i], dates[i]], 
                            [df['open'].values[i], close[i]], 
                            color=color, linewidth=2)
            
            # EMA21
            ax1.plot(dates[-30:], ema_21[-30:], 'blue', linewidth=2, label='EMA21')
            
            # Destacar vela actual
            marker_color = 'green' if signal_type == 'LONG' else 'red'
            ax1.scatter(dates[current_idx], close[current_idx], 
                       color=marker_color, s=100, marker='o', 
                       edgecolors='black', linewidth=2, zorder=5,
                       label=f'Señal {signal_type}')
            
            ax1.set_title(f'{symbol} - {interval} - Señal {signal_type} por Volumen+EMA21', 
                         fontsize=12, fontweight='bold')
            ax1.set_ylabel('Precio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico inferior: Volumen + Volume_MA21
            volume_colors = []
            for i in range(max(0, len(dates)-30), len(dates)):
                if i == current_idx:
                    volume_colors.append(marker_color)
                else:
                    volume_colors.append('gray')
            
            ax2.bar(dates[-30:], volume[-30:], color=volume_colors, alpha=0.6, width=0.8)
            ax2.plot(dates[-30:], volume_ma_21[-30:], 'orange', linewidth=2, label='MA21 Volumen')
            
            # Línea de ratio 2.5x
            current_volume_ma = volume_ma_21[current_idx] if current_idx < len(volume_ma_21) else 0
            ax2.axhline(y=current_volume_ma * 2.5, color='red', linestyle='--', 
                       alpha=0.5, label='Umbral 2.5x')
            
            ax2.set_ylabel('Volumen')
            ax2.set_xlabel('Fecha/Hora')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar imagen
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando gráfico volume+ema para {signal_data['symbol']}: {e}")
            return None

    def generate_multi_strategy_signals(self):
        """Generar señales de la estrategia Multi-Timeframe"""
        signals = []
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
            for symbol in CRYPTO_SYMBOLS[:15]:  # Limitar para no sobrecargar
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        # Generar ID único
                        signal_id = f"MULTI_{symbol}_{interval}_{signal_data['signal']}_{int(time.time())}"
                        
                        # Verificar si ya se envió esta señal recientemente
                        if signal_id in self.sent_multi_signals:
                            continue
                        
                        # Obtener condiciones cumplidas
                        conditions_text = ", ".join(signal_data.get('fulfilled_conditions', []))
                        
                        signal_info = {
                            'symbol': symbol,
                            'interval': interval,
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'entry': signal_data['entry'],
                            'current_price': signal_data['current_price'],
                            'conditions': conditions_text,
                            'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                            'signal_id': signal_id,
                            'supports': signal_data.get('supports', []),
                            'resistances': signal_data.get('resistances', []),
                            'strategy': 'multi'
                        }
                        
                        signals.append(signal_info)
                        self.sent_multi_signals.add(signal_id)
                        
                        print(f"Señal MULTI generada: {symbol} {interval} {signal_data['signal']} Score: {signal_data['signal_score']}%")
                    
                except Exception as e:
                    print(f"Error generando señal MULTI para {symbol} {interval}: {e}")
                    continue
        
        return signals

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
            
            # Calcular soportes y resistencias dinámicos
            supports, resistances = self.calculate_support_resistance(high, low, close, sr_period)
            
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            breakout_up, breakout_down = self.check_breakout(high, low, close, supports, resistances)
            
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
            
            # ADX pendiente
            adx_slope_positive = self.check_adx_slope(adx)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Verificar condiciones de Bollinger
            bollinger_conditions_long = self.check_bollinger_conditions(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions(df, interval, 'SHORT')
            
            # Volumen anómalo mejorado
            volume_anomaly_data = self.calculate_volume_anomaly_improved(volume, close)
            
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
                'adx_slope_positive': adx_slope_positive,
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
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'macd_cross_bullish': macd_cross_bullish,
                'macd_cross_bearish': macd_cross_bearish,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'volume_anomaly_buy': volume_anomaly_data['volume_anomaly_buy'],
                'volume_anomaly_sell': volume_anomaly_data['volume_anomaly_sell'],
                'volume_clusters': volume_anomaly_data['volume_clusters'],
                'volume_ratio': volume_anomaly_data['volume_ratio'],
                'volume_ma_21': volume_anomaly_data['volume_ma_21'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short
            }
            
            conditions = self.evaluate_signal_conditions_optimized(analysis_data, current_idx, interval, adx_threshold)
            
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
            
            # Calcular niveles con soportes/resistencias dinámicos
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage, supports, resistances)
            
            current_price = float(close[current_idx])
            
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
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'macd_cross_bullish': macd_cross_bullish[-50:],
                    'macd_cross_bearish': macd_cross_bearish[-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'volume_anomaly_buy': volume_anomaly_data['volume_anomaly_buy'][-50:],
                    'volume_anomaly_sell': volume_anomaly_data['volume_anomaly_sell'][-50:],
                    'volume_clusters': volume_anomaly_data['volume_clusters'][-50:],
                    'volume_ratio': volume_anomaly_data['volume_ratio'][-50:],
                    'volume_ma_21': volume_anomaly_data['volume_ma_21'][-50:],
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

def send_telegram_alert(signal_data, chart_buffer, strategy_type='multi'):
    """Enviar alerta por Telegram con imagen"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        symbol = signal_data['symbol']
        interval = signal_data['interval']
        signal_type = signal_data['signal']
        
        risk_classification = get_risk_classification(symbol)
        
        if strategy_type == 'multi':
            # Mensaje para estrategia Multi-Timeframe
            message = f"""
🚨 MULTI-TIMEFRAME | {signal_type} | {symbol} | {interval}
Riesgo: {risk_classification}

🎯 Entrada: ${signal_data['entry']:.6f}
💰 Actual: ${signal_data['current_price']:.6f}
📊 Score: {signal_data['score']:.1f}%

✅ Condiciones: {signal_data.get('conditions', 'Confirmada')}

🕐 {signal_data['timestamp']}
"""
        else:
            # Mensaje para estrategia Volumen+EMA21
            message = f"""
🚨 VOL+EMA21 | {signal_type} | {symbol} | {interval}
Riesgo: {risk_classification}

🎯 Entrada: ${signal_data['entry']:.6f}
💰 Actual: ${signal_data['current_price']:.6f}
📈 Vol: {signal_data['volume_ratio']:.1f}x MA21

✅ Filtros: FTMaverick OK | MF Confirmado

🕐 {signal_data['timestamp']}
"""
        
        # Enviar mensaje con imagen
        asyncio.run(bot.send_photo(
            chat_id=TELEGRAM_CHAT_ID,
            photo=chart_buffer,
            caption=message
        ))
        
        print(f"Alerta {strategy_type} enviada a Telegram: {symbol} {interval} {signal_type}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano para ambas estrategias"""
    while True:
        try:
            print("Verificando alertas...")
            
            # ESTRATEGIA 1: Multi-Timeframe
            multi_signals = indicator.generate_multi_strategy_signals()
            for signal in multi_signals:
                # Generar gráfico
                chart_buffer = indicator.generate_telegram_chart_multi_strategy(signal, 'multi')
                if chart_buffer:
                    # Enviar alerta
                    send_telegram_alert(signal, chart_buffer, 'multi')
                    # Registrar señal enviada
                    indicator.sent_multi_signals.add(signal['signal_id'])
            
            # ESTRATEGIA 2: Volumen + EMA21
            for interval in ['1h', '4h', '12h', '1D']:
                for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para no sobrecargar
                    try:
                        signal = indicator.check_volume_ema_ftm_signal(symbol, interval)
                        if signal:
                            # Generar gráfico
                            chart_buffer = indicator.generate_telegram_chart_volume_ema(signal)
                            if chart_buffer:
                                # Enviar alerta
                                send_telegram_alert(signal, chart_buffer, 'volume_ema')
                                # Registrar señal enviada
                                indicator.sent_volume_ema_signals.add(signal['signal_id'])
                    except Exception as e:
                        print(f"Error verificando señal VOL+EMA21 para {symbol} {interval}: {e}")
                        continue
            
            # Esperar según estrategia
            time.sleep(300)  # 5 minutos entre verificaciones
            
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
    """Endpoint para obtener alertas de trading (solo Multi-Timeframe)"""
    try:
        alerts = []
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
            for symbol in CRYPTO_SYMBOLS[:8]:
                try:
                    signal_data = indicator.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        risk_category = next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                        
                        alert = {
                            'symbol': symbol,
                            'interval': interval,
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'entry': signal_data['entry'],
                            'current_price': signal_data['current_price'],
                            'timestamp': indicator.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category
                        }
                        
                        alerts.append(alert)
                    
                except Exception as e:
                    print(f"Error generando alerta para {symbol} {interval}: {e}")
                    continue
        
        return jsonify({'alerts': alerts[:10]})  # Limitar a 10 alertas
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/volume_ema_signals')
def get_volume_ema_signals():
    """Endpoint para obtener señales de Volumen+EMA21"""
    try:
        signals = []
        
        for interval in ['1h', '4h', '12h', '1D']:
            for symbol in CRYPTO_SYMBOLS[:8]:
                try:
                    signal = indicator.check_volume_ema_ftm_signal(symbol, interval)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    print(f"Error obteniendo señal VOL+EMA21 para {symbol} {interval}: {e}")
                    continue
        
        return jsonify({'signals': signals})
        
    except Exception as e:
        print(f"Error en /api/volume_ema_signals: {e}")
        return jsonify({'signals': []})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
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
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit']):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
            
            # Soporte/Resistencias (4-6 niveles)
            if 'supports' in signal_data:
                for i, support in enumerate(signal_data['supports'][:4]):
                    ax1.axhline(y=support, color='orange', linestyle=':', alpha=0.5, label=f'Soporte {i+1}')
            
            if 'resistances' in signal_data:
                for i, resistance in enumerate(signal_data['resistances'][:4]):
                    ax1.axhline(y=resistance, color='purple', linestyle=':', alpha=0.5, label=f'Resistencia {i+1}')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: ADX con DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
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
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Volumen con anomalías
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_ratio']):]
            
            # Barras de volumen coloreadas
            volume_colors = []
            for i in range(len(volume_dates)):
                if i < len(signal_data['indicators']['volume_anomaly_buy']):
                    if signal_data['indicators']['volume_anomaly_buy'][i]:
                        volume_colors.append('green')
                    elif signal_data['indicators']['volume_anomaly_sell'][i]:
                        volume_colors.append('red')
                    else:
                        volume_colors.append('gray')
                else:
                    volume_colors.append('gray')
            
            ax3.bar(volume_dates, 
                   [signal_data['data'][-len(signal_data['indicators']['volume_ratio'])+i]['volume'] 
                    for i in range(len(volume_dates))], 
                   color=volume_colors, alpha=0.6, label='Volumen')
            
            # MA21 de volumen
            ax3.plot(volume_dates, signal_data['indicators']['volume_ma_21'], 
                    'orange', linewidth=1, label='MA21 Volumen')
        
        ax3.set_ylabel('Volumen')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Fuerza de Tendencia Maverick
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
            
            ax4.set_ylabel('FT Maverick %')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Ballenas (solo si hay datos)
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data and 'whale_pump' in signal_data['indicators']:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax5.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.6, label='Compradoras')
            ax5.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.6, label='Vendedoras')
        ax5.set_ylabel('Ballenas')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: RSI Maverick
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_maverick_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax6.plot(rsi_maverick_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Sobrecompra')
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Sobreventa')
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        ax6.set_ylabel('RSI Maverick')
        ax6.set_ylim(0, 1)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: RSI Tradicional
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_trad_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax7.plot(rsi_trad_dates, signal_data['indicators']['rsi_traditional'], 
                    'purple', linewidth=2, label='RSI Tradicional')
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.5)
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax7.set_ylabel('RSI Tradicional')
        ax7.set_ylim(0, 100)
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: MACD
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax8.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax8.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax8.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6, label='Histograma')
            
            ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax8.set_ylabel('MACD')
        ax8.set_xlabel('Fecha/Hora')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        # Información de la señal
        ax9 = plt.subplot(9, 1, 9)
        ax9.axis('off')
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
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
