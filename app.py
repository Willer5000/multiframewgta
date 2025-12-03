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
        self.volume_alerts_sent = set()
        self.bolivia_tz = pytz.timezone('America/La_Paz')
    
    def get_bolivia_time(self):
        return datetime.now(self.bolivia_tz)
    
    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela - OPTIMIZADO"""
        if interval == '15m':
            next_close = current_time.replace(minute=current_time.minute // 15 * 15, second=0, microsecond=0) + timedelta(minutes=15)
            remaining_seconds = (next_close - current_time).total_seconds()
            return remaining_seconds <= (15 * 60 * 0.5)  # 50%
        elif interval == '30m':
            next_close = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0) + timedelta(minutes=30)
            remaining_seconds = (next_close - current_time).total_seconds()
            return remaining_seconds <= (30 * 60 * 0.5)  # 50%
        elif interval == '1h':
            next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            remaining_seconds = (next_close - current_time).total_seconds()
            return remaining_seconds <= (60 * 60 * 0.5)  # 50%
        elif interval == '2h':
            current_hour = current_time.hour
            next_2h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour % 2 == 0:
                next_2h_close += timedelta(hours=2)
            else:
                next_2h_close += timedelta(hours=1)
            remaining_seconds = (next_2h_close - current_time).total_seconds()
            return remaining_seconds <= (120 * 60 * 0.5)  # 50%
        elif interval == '4h':
            current_hour = current_time.hour
            next_4h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 4
            if remainder == 0:
                next_4h_close += timedelta(hours=4)
            else:
                next_4h_close += timedelta(hours=4 - remainder)
            remaining_seconds = (next_4h_close - current_time).total_seconds()
            return remaining_seconds <= (240 * 60 * 0.25)  # 25%
        elif interval == '8h':
            current_hour = current_time.hour
            next_8h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 8
            if remainder == 0:
                next_8h_close += timedelta(hours=8)
            else:
                next_8h_close += timedelta(hours=8 - remainder)
            remaining_seconds = (next_8h_close - current_time).total_seconds()
            return remaining_seconds <= (480 * 60 * 0.25)  # 25%
        elif interval == '12h':
            current_hour = current_time.hour
            next_12h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour < 8:
                next_12h_close = next_12h_close.replace(hour=20)
            else:
                next_12h_close = next_12h_close.replace(hour=8) + timedelta(days=1)
            remaining_seconds = (next_12h_close - current_time).total_seconds()
            return remaining_seconds <= (720 * 60 * 0.25)  # 25%
        elif interval == '1D':
            tomorrow_8pm = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
            if current_time.hour >= 20:
                tomorrow_8pm += timedelta(days=1)
            remaining_seconds = (tomorrow_8pm - current_time).total_seconds()
            return remaining_seconds <= (1440 * 60 * 0.25)  # 25%
        elif interval == '1W':
            return False  # No aplica según requisitos
        
        return False

    def get_kucoin_data(self, symbol, interval, limit=100):
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
                
        except Exception as e:
            print(f"Error obteniendo datos de KuCoin para {symbol} {interval}: {e}")
        
        df = self.generate_sample_data(limit, interval, symbol)
        self.cache[cache_key] = (df, datetime.now())
        return df

    def generate_sample_data(self, limit, interval, symbol):
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

    def detect_support_resistance(self, high, low, close, lookback=50, num_levels=6):
        """Detectar múltiples niveles de soporte y resistencia"""
        try:
            n = len(close)
            if n < lookback:
                return [], []
            
            # Usar pivotes para detectar máximos y mínimos locales
            supports = []
            resistances = []
            
            # Detectar máximos locales (resistencia)
            for i in range(5, n-5):
                if high[i] == max(high[i-5:i+6]):
                    resistances.append({
                        'price': high[i],
                        'strength': 1,
                        'index': i
                    })
            
            # Detectar mínimos locales (soporte)
            for i in range(5, n-5):
                if low[i] == min(low[i-5:i+6]):
                    supports.append({
                        'price': low[i],
                        'strength': 1,
                        'index': i
                    })
            
            # Agrupar niveles cercanos
            def group_levels(levels, threshold=0.005):
                if not levels:
                    return []
                
                levels.sort(key=lambda x: x['price'])
                grouped = []
                current_group = [levels[0]]
                
                for level in levels[1:]:
                    if abs(level['price'] - current_group[-1]['price']) / current_group[-1]['price'] < threshold:
                        current_group.append(level)
                    else:
                        avg_price = np.mean([l['price'] for l in current_group])
                        strength = len(current_group)
                        grouped.append({
                            'price': avg_price,
                            'strength': strength,
                            'count': len(current_group)
                        })
                        current_group = [level]
                
                if current_group:
                    avg_price = np.mean([l['price'] for l in current_group])
                    strength = len(current_group)
                    grouped.append({
                        'price': avg_price,
                        'strength': strength,
                        'count': len(current_group)
                    })
                
                return sorted(grouped, key=lambda x: x['price'])
            
            grouped_supports = group_levels(supports)
            grouped_resistances = group_levels(resistances)
            
            # Seleccionar los mejores niveles (4-6 cada uno)
            min_levels = min(4, len(grouped_supports), len(grouped_resistances))
            selected_supports = sorted(grouped_supports, key=lambda x: x['strength'], reverse=True)[:min_levels]
            selected_resistances = sorted(grouped_resistances, key=lambda x: x['strength'], reverse=True)[:min_levels]
            
            # Asegurar al menos 4 líneas totales
            total_levels = len(selected_supports) + len(selected_resistances)
            if total_levels < 4:
                # Añadir niveles basados en medias móviles
                current_price = close[-1]
                ma_50 = self.calculate_sma(close, 50)[-1] if n >= 50 else current_price
                ma_100 = self.calculate_sma(close, 100)[-1] if n >= 100 else current_price
                
                additional_levels = []
                for ma, name in [(ma_50, "MA50"), (ma_100, "MA100")]:
                    if ma not in [s['price'] for s in selected_supports + selected_resistances]:
                        additional_levels.append({
                            'price': ma,
                            'strength': 1,
                            'count': 1,
                            'type': name
                        })
                
                # Distribuir niveles adicionales como soporte/resistencia
                for i, level in enumerate(additional_levels):
                    if len(selected_supports) <= len(selected_resistances):
                        selected_supports.append(level)
                    else:
                        selected_resistances.append(level)
            
            return selected_supports, selected_resistances
            
        except Exception as e:
            print(f"Error detectando soportes/resistencias: {e}")
            current_price = close[-1] if len(close) > 0 else 0
            return [
                {'price': current_price * 0.95, 'strength': 1, 'count': 1, 'type': 'default'}
            ], [
                {'price': current_price * 1.05, 'strength': 1, 'count': 1, 'type': 'default'}
            ]

    def calculate_optimal_entry_exit_improved(self, df, signal_type, supports, resistances, leverage=15):
        """Calcular entradas y salidas óptimas mejoradas con múltiples S/R"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Ordenar soportes y resistencias
            supports_sorted = sorted(supports, key=lambda x: x['price'])
            resistances_sorted = sorted(resistances, key=lambda x: x['price'])
            
            if signal_type == 'LONG':
                # Entrada en el soporte más cercano POR DEBAJO del precio actual
                valid_supports = [s for s in supports_sorted if s['price'] < current_price]
                if valid_supports:
                    # Tomar el soporte más alto (más cercano al precio actual)
                    entry_support = max(valid_supports, key=lambda x: x['price'])
                    entry = entry_support['price']
                else:
                    # Si no hay soportes por debajo, usar un soporte calculado
                    entry = current_price * 0.99
                
                # Stop loss debajo del soporte más bajo
                if supports_sorted:
                    lowest_support = min(supports_sorted, key=lambda x: x['price'])
                    stop_loss = lowest_support['price'] * 0.97
                else:
                    stop_loss = entry - (current_atr * 1.8)
                
                # Take profits en resistencias
                take_profits = []
                if resistances_sorted:
                    # TP1: Primera resistencia
                    tp1 = resistances_sorted[0]['price'] * 0.98
                    min_tp = entry + (2 * (entry - stop_loss))
                    tp1 = max(tp1, min_tp)
                    take_profits.append(tp1)
                    
                    # TP2: Segunda resistencia si existe
                    if len(resistances_sorted) > 1:
                        tp2 = resistances_sorted[1]['price'] * 0.98
                        take_profits.append(tp2)
                
                # Si no hay resistencias, usar ATR
                if not take_profits:
                    take_profits = [entry + (current_atr * 3)]
                
                # Soporte principal (más cercano)
                if supports_sorted:
                    main_support = max([s for s in supports_sorted if s['price'] < current_price], 
                                      key=lambda x: x['price'], default=supports_sorted[0])
                    main_support_price = main_support['price']
                else:
                    main_support_price = entry * 0.95
                
                # Resistencia principal (más cercana)
                if resistances_sorted:
                    main_resistance = min([r for r in resistances_sorted if r['price'] > current_price], 
                                         key=lambda x: x['price'], default=resistances_sorted[-1])
                    main_resistance_price = main_resistance['price']
                else:
                    main_resistance_price = entry * 1.05
                
            else:  # SHORT
                # Entrada en la resistencia más cercana POR ENCIMA del precio actual
                valid_resistances = [r for r in resistances_sorted if r['price'] > current_price]
                if valid_resistances:
                    # Tomar la resistencia más baja (más cercana al precio actual)
                    entry_resistance = min(valid_resistances, key=lambda x: x['price'])
                    entry = entry_resistance['price']
                else:
                    entry = current_price * 1.01
                
                # Stop loss encima de la resistencia más alta
                if resistances_sorted:
                    highest_resistance = max(resistances_sorted, key=lambda x: x['price'])
                    stop_loss = highest_resistance['price'] * 1.03
                else:
                    stop_loss = entry + (current_atr * 1.8)
                
                # Take profits en soportes
                take_profits = []
                if supports_sorted:
                    # TP1: Primer soporte
                    tp1 = supports_sorted[-1]['price'] * 1.02
                    min_tp = entry - (2 * (stop_loss - entry))
                    tp1 = min(tp1, min_tp)
                    take_profits.append(tp1)
                    
                    # TP2: Segundo soporte si existe
                    if len(supports_sorted) > 1:
                        tp2 = supports_sorted[-2]['price'] * 1.02
                        take_profits.append(tp2)
                
                # Si no hay soportes, usar ATR
                if not take_profits:
                    take_profits = [entry - (current_atr * 3)]
                
                # Resistencia principal (más cercana)
                if resistances_sorted:
                    main_resistance = min([r for r in resistances_sorted if r['price'] > current_price], 
                                         key=lambda x: x['price'], default=resistances_sorted[0])
                    main_resistance_price = main_resistance['price']
                else:
                    main_resistance_price = entry * 1.05
                
                # Soporte principal (más cercano)
                if supports_sorted:
                    main_support = max([s for s in supports_sorted if s['price'] < current_price], 
                                      key=lambda x: x['price'], default=supports_sorted[-1])
                    main_support_price = main_support['price']
                else:
                    main_support_price = entry * 0.95
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits],
                'support': float(main_support_price),
                'resistance': float(main_resistance_price),
                'all_supports': supports_sorted,
                'all_resistances': resistances_sorted,
                'atr': float(current_atr),
                'atr_percentage': float(current_atr / current_price) if current_price > 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas mejoradas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support': current_price * 0.95,
                'resistance': current_price * 1.05,
                'all_supports': [],
                'all_resistances': [],
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_ema(self, prices, period):
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
        if len(prices) < slow:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
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

    def calculate_adx(self, high, low, close, period=14):
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
            window_close = close[i-lookback:i+1]
            
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

    def calculate_volume_anomaly_improved(self, close, volume, period=20, std_multiplier=2):
        """Calcular anomalías de volumen MEJORADO - Distingue compra/venta"""
        try:
            n = len(volume)
            volume_anomaly_buy = np.zeros(n, dtype=bool)
            volume_anomaly_sell = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_colors = ['gray'] * n
            
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
                    # Determinar si es compra (verde) o venta (roja) basado en precio
                    if i > 0:
                        if close[i] > close[i-1]:  # Precio subió -> compra anómala
                            volume_anomaly_buy[i] = True
                            volume_colors[i] = 'green'
                        elif close[i] < close[i-1]:  # Precio bajó -> venta anómala
                            volume_anomaly_sell[i] = True
                            volume_colors[i] = 'red'
                        else:
                            volume_colors[i] = 'gray'
                
                # Detectar clusters (múltiples anomalías en 5 periodos)
                if i >= 5:
                    recent_buy_anomalies = volume_anomaly_buy[max(0, i-4):i+1]
                    recent_sell_anomalies = volume_anomaly_sell[max(0, i-4):i+1]
                    if np.sum(recent_buy_anomalies) >= 3 or np.sum(recent_sell_anomalies) >= 3:
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly_buy': volume_anomaly_buy.tolist(),
                'volume_anomaly_sell': volume_anomaly_sell.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ema': ema_volume.tolist() if 'ema_volume' in locals() else [0] * n,
                'volume_colors': volume_colors
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly_improved: {e}")
            n = len(volume)
            return {
                'volume_anomaly_buy': [False] * n,
                'volume_anomaly_sell': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ema': [0] * n,
                'volume_colors': ['gray'] * n
            }

    def check_volume_anomaly_conditions(self, symbol, interval):
        """Verificar condiciones de volumen anómalo para nueva estrategia"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if 'menor' not in hierarchy:
                return None
            
            # Obtener datos de temporalidad menor
            menor_interval = hierarchy['menor']
            df_menor = self.get_kucoin_data(symbol, menor_interval, 50)
            
            if df_menor is None or len(df_menor) < 20:
                return None
            
            # Calcular anomalías de volumen en temporalidad menor
            volume_data = self.calculate_volume_anomaly_improved(
                df_menor['close'].values,
                df_menor['volume'].values
            )
            
            # Verificar si hay anomalía o cluster
            has_buy_anomaly = volume_data['volume_anomaly_buy'][-1]
            has_sell_anomaly = volume_data['volume_anomaly_sell'][-1]
            has_cluster = volume_data['volume_clusters'][-1]
            
            if not (has_buy_anomaly or has_sell_anomaly or has_cluster):
                return None
            
            # Obtener datos de temporalidad actual
            df_current = self.get_kucoin_data(symbol, interval, 50)
            if df_current is None or len(df_current) < 20:
                return None
            
            # Condiciones FTMaverick
            current_trend = self.calculate_trend_strength_maverick(df_current['close'].values)
            current_no_trade = current_trend['no_trade_zones'][-1]
            current_strength = current_trend['strength_signals'][-1]
            
            menor_trend = self.calculate_trend_strength_maverick(df_menor['close'].values)
            menor_no_trade = menor_trend['no_trade_zones'][-1]
            menor_strength = menor_trend['strength_signals'][-1]
            
            # Determinar señal
            signal = None
            conditions = []
            
            if (has_buy_anomaly or (has_cluster and volume_data['volume_anomaly_buy'][-1])):
                if not current_no_trade and not menor_no_trade:
                    conditions.append("Fuera de Zona No Operar")
                    
                    if menor_strength in ['STRONG_UP', 'WEAK_UP']:
                        conditions.append("Temporalidad menor con tendencia alcista")
                    
                    if current_strength in ['STRONG_UP', 'WEAK_UP', 'NEUTRAL']:
                        conditions.append("Temporalidad actual tendencia Neutral o alcista")
                    
                    if len(conditions) == 3:  # Todas las condiciones
                        signal = 'LONG'
            
            elif (has_sell_anomaly or (has_cluster and volume_data['volume_anomaly_sell'][-1])):
                if not current_no_trade and not menor_no_trade:
                    conditions.append("Fuera de Zona No Operar")
                    
                    if menor_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                        conditions.append("Temporalidad menor con tendencia bajista")
                    
                    if current_strength in ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL']:
                        conditions.append("Temporalidad actual tendencia Neutral o bajista")
                    
                    if len(conditions) == 3:
                        signal = 'SHORT'
            
            if signal:
                risk_category = next(
                    (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                     if symbol in symbols), 'medio'
                )
                
                anomaly_type = "compra" if has_buy_anomaly else "venta"
                cluster_text = " (cluster)" if has_cluster else ""
                
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'risk_category': risk_category,
                    'anomaly_type': anomaly_type,
                    'cluster': has_cluster,
                    'interval': interval,
                    'menor_interval': menor_interval,
                    'conditions': conditions,
                    'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            return None
            
        except Exception as e:
            print(f"Error en check_volume_anomaly_conditions para {symbol}: {e}")
            return None

    def evaluate_signal_conditions(self, data, current_idx, interval, adx_threshold=25):
        weights = {
            'long': {
                'multi_timeframe': 30,
                'trend_strength': 25,
                'bollinger_bands': 10,
                'adx_dmi': 10,
                'ma_cross': 10,
                'rsi_traditional_divergence': 10,
                'rsi_maverick_divergence': 10,
                'macd': 5,
                'chart_pattern': 5,
                'breakout': 5,
                'volume_anomaly': 10
            },
            'short': {
                'multi_timeframe': 30,
                'trend_strength': 25,
                'bollinger_bands': 10,
                'adx_dmi': 10,
                'ma_cross': 10,
                'rsi_traditional_divergence': 10,
                'rsi_maverick_divergence': 10,
                'macd': 5,
                'chart_pattern': 5,
                'breakout': 5,
                'volume_anomaly': 10
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
        conditions['long']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['plus_di'][current_idx] > data['minus_di'][current_idx]
        )
        conditions['long']['ma_cross']['value'] = (
            current_price > ma_9 and ma_9 > ma_21 and ma_21 > ma_50
        )
        conditions['long']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bullish_divergence']) and 
            data['rsi_bullish_divergence'][current_idx]
        )
        conditions['long']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bullish_divergence']) and 
            data['rsi_maverick_bullish_divergence'][current_idx]
        )
        conditions['long']['macd']['value'] = (
            data['macd'][current_idx] > data['macd_signal'][current_idx] and
            data['macd_histogram'][current_idx] > 0
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
        conditions['short']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['minus_di'][current_idx] > data['plus_di'][current_idx]
        )
        conditions['short']['ma_cross']['value'] = (
            current_price < ma_9 and ma_9 < ma_21 and ma_21 < ma_50
        )
        conditions['short']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bearish_divergence']) and 
            data['rsi_bearish_divergence'][current_idx]
        )
        conditions['short']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bearish_divergence']) and 
            data['rsi_maverick_bearish_divergence'][current_idx]
        )
        conditions['short']['macd']['value'] = (
            data['macd'][current_idx] < data['macd_signal'][current_idx] and
            data['macd_histogram'][current_idx] < 0
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
            current_idx < len(data['volume_anomaly_sell']) and 
            data['volume_anomaly_sell'][current_idx]
        )
        
        return conditions

    def get_condition_description(self, condition_key):
        descriptions = {
            'multi_timeframe': 'Condiciones Multi-TF obligatorias',
            'trend_strength': 'Fuerza tendencia favorable',
            'whale_signal': 'Señal ballenas confirmada',
            'bollinger_bands': 'Bandas de Bollinger',
            'adx_dmi': 'ADX + DMI',
            'ma_cross': 'MA Cross (9-21-50)',
            'rsi_traditional_divergence': 'RSI Tradicional Divergence',
            'rsi_maverick_divergence': 'RSI Maverick Divergence',
            'macd': 'MACD',
            'chart_pattern': 'Chart Patterns',
            'breakout': 'Breakout Confirmation',
            'volume_anomaly': 'Anomalía de Volumen'
        }
        return descriptions.get(condition_key, condition_key)

    def calculate_signal_score(self, conditions, signal_type):
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        signal_conditions = conditions.get(signal_type, {})
        
        obligatory_conditions = []
        for key, condition in signal_conditions.items():
            if condition['weight'] >= 25:
                obligatory_conditions.append(key)
        
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
        min_score = 65
        final_score = base_score if base_score >= min_score else 0

        return min(final_score, 100), fulfilled_conditions

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Detectar soportes y resistencias
            supports, resistances = self.detect_support_resistance(high, low, close, sr_period, num_levels=6)
            
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            breakout_up, breakout_down = self.check_breakout(high, low, close, whale_data['support'], whale_data['resistance'])
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            volume_anomaly_data = self.calculate_volume_anomaly_improved(close, volume)
            
            current_idx = -1
            
            multi_timeframe_long = self.check_multi_timeframe_obligatory(symbol, interval, 'LONG')
            multi_timeframe_short = self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT')
            
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
                'volume_anomaly_buy': volume_anomaly_data['volume_anomaly_buy'],
                'volume_anomaly_sell': volume_anomaly_data['volume_anomaly_sell'],
                'volume_clusters': volume_anomaly_data['volume_clusters'],
                'volume_ratio': volume_anomaly_data['volume_ratio'],
                'volume_colors': volume_anomaly_data['volume_colors'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short
            }
            
            conditions = self.evaluate_signal_conditions(analysis_data, current_idx, interval, adx_threshold)
            
            current_ma200 = ma_200[current_idx] if current_idx < len(ma_200) else 0
            current_price = close[current_idx]
            ma200_condition = 'above' if current_price > current_ma200 else 'below'

            long_score, long_conditions = self.calculate_signal_score(conditions, 'long')
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short')
            
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
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
            
            levels_data = self.calculate_optimal_entry_exit_improved(df, signal_type, supports, resistances, leverage)
            
            current_time = self.get_bolivia_time()
            
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
                'all_supports': supports,
                'all_resistances': resistances,
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
                    'volume_anomaly_buy': volume_anomaly_data['volume_anomaly_buy'][-50:],
                    'volume_anomaly_sell': volume_anomaly_data['volume_anomaly_sell'][-50:],
                    'volume_clusters': volume_anomaly_data['volume_clusters'][-50:],
                    'volume_ratio': volume_anomaly_data['volume_ratio'][-50:],
                    'volume_ema': volume_anomaly_data['volume_ema'][-50:],
                    'volume_colors': volume_anomaly_data['volume_colors'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
                    'colors': trend_strength_data['colors'][-50:]
                },
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error en generate_signals_improved para {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_signal(symbol)

    def _create_empty_signal(self, symbol):
        current_time = self.get_bolivia_time()
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
            'all_supports': [],
            'all_resistances': [],
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
            'indicators': {},
            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def generate_volume_anomaly_signals(self):
        """Generar señales basadas en volumen anómalo en temporalidad menor"""
        volume_alerts = []
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D']:
            # Verificar si es momento de enviar alerta
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
            
            for symbol in CRYPTO_SYMBOLS[:15]:  # Limitar para no sobrecargar
                try:
                    signal_data = self.check_volume_anomaly_conditions(symbol, interval)
                    
                    if signal_data:
                        alert_key = f"{symbol}_{interval}_{signal_data['signal']}_{current_time.strftime('%Y%m%d_%H')}"
                        
                        if alert_key not in self.volume_alerts_sent:
                            volume_alerts.append(signal_data)
                            self.volume_alerts_sent.add(alert_key)
                            print(f"Alerta volumen anómalo generada: {symbol} {interval} {signal_data['signal']}")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error generando alerta de volumen para {symbol} {interval}: {e}")
                    continue
        
        return volume_alerts

    def generate_scalping_alerts(self):
        """Generar alertas de trading principales"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:12]:
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
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
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'][0],
                            'leverage': 15,
                            'timestamp': signal_data['timestamp'],
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'support': signal_data['support'],
                            'resistance': signal_data['resistance'],
                            'ma200_condition': signal_data.get('ma200_condition', 'below'),
                            'multi_timeframe_ok': signal_data.get('multi_timeframe_ok', False),
                            'signal_data': signal_data  # Agregar datos completos para gráficos
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

async def send_telegram_alert(alert_data, alert_type='entry', image_path=None):
    """Enviar alerta por Telegram con imagen"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        
        if alert_type == 'entry':
            message = f"""
{alert_data['signal']}
Crypto: {alert_data['symbol']} ({risk_classification})
Score: {alert_data['score']:.1f}%
Precio: ${alert_data['current_price']:.6f}
Entrada: ${alert_data['entry']:.6f}
Stop Loss: ${alert_data['stop_loss']:.6f}
Take Profit: ${alert_data['take_profit']:.6f}
Multi-TF: {'✅' if alert_data.get('multi_timeframe_ok', False) else '❌'}
MA200: {'ENCIMA' if alert_data.get('ma200_condition') == 'above' else 'DEBAJO'}

Condiciones:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])])}
            """
            
        elif alert_type == 'volume':
            anomaly_type = alert_data.get('anomaly_type', 'compra')
            cluster_text = " (cluster)" if alert_data.get('cluster') else ""
            conditions_text = chr(10).join(['• ' + cond for cond in alert_data.get('conditions', [])])
            
            message = f"""
{alert_data['signal']}
Crypto: {alert_data['symbol']} ({alert_data['risk_category']})
Volumen: Volúmenes Anómalos de {anomaly_type}{cluster_text} en temporalidad Menor
Temporalidad: {alert_data['interval']}

Condiciones FTMaverick y MF:
{conditions_text}
            """
        
        # Limpiar el mensaje
        message = message.strip()
        
        if image_path:
            with open(image_path, 'rb') as img:
                await bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=img,
                    caption=message
                )
            print(f"Alerta {alert_type} con imagen enviada a Telegram: {alert_data['symbol']}")
        else:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=message
            )
            print(f"Alerta {alert_type} enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    check_intervals = {
        '15m': 60,    # Cada 60 segundos
        '30m': 60,    # Cada 60 segundos
        '1h': 300,    # Cada 300 segundos
        '2h': 300,    # Cada 300 segundos
        '4h': 600,    # Cada 600 segundos
        '8h': 600,    # Cada 600 segundos
        '12h': 600,   # Cada 600 segundos
        '1D': 600,    # Cada 600 segundos
        '1W': 3600    # Cada hora
    }
    
    last_checks = {interval: datetime.now() for interval in check_intervals.keys()}
    
    while True:
        try:
            current_time = datetime.now()
            
            for interval, check_seconds in check_intervals.items():
                if (current_time - last_checks[interval]).seconds >= check_seconds:
                    print(f"Verificando alertas {interval}...")
                    
                    # Generar alertas principales
                    alerts = indicator.generate_scalping_alerts()
                    for alert in alerts:
                        if alert['interval'] == interval:
                            # Generar imagen para la alerta
                            try:
                                img_buffer = generate_telegram_image(alert, 'entry')
                                if img_buffer:
                                    # Guardar temporalmente
                                    temp_path = f"temp_{alert['symbol']}_{interval}_{int(time.time())}.png"
                                    with open(temp_path, 'wb') as f:
                                        f.write(img_buffer.getvalue())
                                    
                                    asyncio.run(send_telegram_alert(alert, 'entry', temp_path))
                                    
                                    # Eliminar archivo temporal
                                    try:
                                        os.remove(temp_path)
                                    except:
                                        pass
                                else:
                                    asyncio.run(send_telegram_alert(alert, 'entry'))
                            except Exception as e:
                                print(f"Error enviando alerta principal: {e}")
                    
                    # Generar alertas de volumen anómalo
                    volume_alerts = indicator.generate_volume_anomaly_signals()
                    for alert in volume_alerts:
                        if alert['interval'] == interval:
                            try:
                                img_buffer = generate_telegram_image(alert, 'volume')
                                if img_buffer:
                                    temp_path = f"temp_volume_{alert['symbol']}_{interval}_{int(time.time())}.png"
                                    with open(temp_path, 'wb') as f:
                                        f.write(img_buffer.getvalue())
                                    
                                    asyncio.run(send_telegram_alert(alert, 'volume', temp_path))
                                    
                                    try:
                                        os.remove(temp_path)
                                    except:
                                        pass
                                else:
                                    asyncio.run(send_telegram_alert(alert, 'volume'))
                            except Exception as e:
                                print(f"Error enviando alerta de volumen: {e}")
                    
                    last_checks[interval] = current_time
            
            time.sleep(10)
            
        except Exception as e:
            print(f"Error en background_alert_checker: {e}")
            time.sleep(60)

def generate_telegram_image(alert_data, alert_type='entry'):
    """Generar imagen para Telegram"""
    try:
        if alert_type == 'entry':
            return generate_entry_telegram_image(alert_data)
        else:
            return generate_volume_telegram_image(alert_data)
    except Exception as e:
        print(f"Error generando imagen para Telegram: {e}")
        return None

def generate_entry_telegram_image(alert_data):
    """Generar imagen para alertas principales con 8 gráficos de indicadores"""
    try:
        # Obtener datos completos de la señal
        if 'signal_data' in alert_data:
            signal_data = alert_data['signal_data']
        else:
            signal_data = indicator.generate_signals_improved(
                alert_data['symbol'], 
                alert_data['interval']
            )
        
        if not signal_data or 'data' not in signal_data or not signal_data['data']:
            return None
        
        # Crear figura con 9 subgráficos (8 indicadores + 1 info)
        fig = plt.figure(figsize=(14, 18))
        fig.patch.set_facecolor('white')
        
        # Obtener datos para gráficos
        df_data = signal_data['data']
        dates = [pd.to_datetime(d['timestamp']) for d in df_data]
        close_prices = [d['close'] for d in df_data]
        open_prices = [d['open'] for d in df_data]
        high_prices = [d['high'] for d in df_data]
        low_prices = [d['low'] for d in df_data]
        volumes = [d['volume'] for d in df_data]
        
        # 1. Gráfico de Velas Japonesas con Bandas de Bollinger
        ax1 = plt.subplot(9, 1, 1)
        
        # Dibujar velas
        for i in range(len(dates)):
            color = 'green' if close_prices[i] >= open_prices[i] else 'red'
            ax1.plot([dates[i], dates[i]], [low_prices[i], high_prices[i]], 
                    color='black', linewidth=0.5)
            ax1.plot([dates[i], dates[i]], [open_prices[i], close_prices[i]], 
                    color=color, linewidth=2)
        
        # Bandas de Bollinger (transparentes)
        if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
            bb_upper = signal_data['indicators']['bb_upper']
            bb_lower = signal_data['indicators']['bb_lower']
            ax1.plot(dates[-len(bb_upper):], bb_upper, 'blue', alpha=0.3, linewidth=1)
            ax1.plot(dates[-len(bb_lower):], bb_lower, 'blue', alpha=0.3, linewidth=1)
            ax1.fill_between(dates[-len(bb_upper):], bb_upper, bb_lower, alpha=0.1, color='blue')
        
        ax1.set_title(f'{alert_data["symbol"]} - {alert_data["interval"]} - Señal {alert_data["signal"]}', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precio')
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx_data = signal_data['indicators']['adx']
            plus_di_data = signal_data['indicators']['plus_di']
            minus_di_data = signal_data['indicators']['minus_di']
            
            ax2.plot(dates[-len(adx_data):], adx_data, 'black', linewidth=1.5, label='ADX')
            ax2.plot(dates[-len(plus_di_data):], plus_di_data, 'green', linewidth=1, label='+DI')
            ax2.plot(dates[-len(minus_di_data):], minus_di_data, 'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7)
        
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Volumen con Anomalías y Clusters
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        
        # Barras de volumen
        for i in range(len(dates)):
            color = 'green' if close_prices[i] >= open_prices[i] else 'red'
            ax3.bar(dates[i], volumes[i], color=color, alpha=0.7, width=0.8)
        
        # Anomalías de volumen
        if 'indicators' in signal_data and 'volume_anomaly_buy' in signal_data['indicators']:
            volume_buy = signal_data['indicators']['volume_anomaly_buy']
            volume_sell = signal_data['indicators']['volume_anomaly_sell']
            
            for i, date in enumerate(dates[-len(volume_buy):]):
                if volume_buy[i]:
                    ax3.axvspan(date, date, alpha=0.3, color='green', ymin=0.8, ymax=1.0)
                if volume_sell[i]:
                    ax3.axvspan(date, date, alpha=0.3, color='red', ymin=0.8, ymax=1.0)
        
        ax3.set_ylabel('Volumen')
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_data = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i, date in enumerate(dates[-len(trend_data):]):
                color = colors[i] if i < len(colors) else 'gray'
                ax4.bar(date, trend_data[i], color=color, alpha=0.7, width=0.8)
        
        ax4.set_ylabel('Fuerza Tendencia')
        ax4.grid(True, alpha=0.3)
        
        # 5. Indicador de Ballenas (solo para 12h y 1D)
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if alert_data['interval'] in ['12h', '1D'] and 'indicators' in signal_data:
            whale_pump = signal_data['indicators']['whale_pump']
            whale_dump = signal_data['indicators']['whale_dump']
            
            ax5.bar(dates[-len(whale_pump):], whale_pump, color='green', alpha=0.7, 
                   label='Ballenas Compra', width=0.8)
            ax5.bar(dates[-len(whale_dump):], whale_dump, color='red', alpha=0.7, 
                   label='Ballenas Venta', width=0.8)
        else:
            ax5.text(0.5, 0.5, 'Indicador Ballenas solo para 12H/1D', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        ax5.set_ylabel('Ballenas')
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. RSI Maverick Modificado
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_maverick' in signal_data['indicators']:
            rsi_mav = signal_data['indicators']['rsi_maverick']
            ax6.plot(dates[-len(rsi_mav):], rsi_mav, 'purple', linewidth=1.5)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        
        ax6.set_ylabel('RSI Maverick')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
        
        # 7. RSI Tradicional con Divergencias
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_traditional' in signal_data['indicators']:
            rsi_trad = signal_data['indicators']['rsi_traditional']
            ax7.plot(dates[-len(rsi_trad):], rsi_trad, 'cyan', linewidth=1.5)
            ax7.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        ax7.set_ylabel('RSI Tradicional')
        ax7.set_ylim(0, 100)
        ax7.grid(True, alpha=0.3)
        
        # 8. MACD con Histograma
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if 'indicators' in signal_data and 'macd' in signal_data['indicators']:
            macd_line = signal_data['indicators']['macd']
            macd_signal = signal_data['indicators']['macd_signal']
            macd_hist = signal_data['indicators']['macd_histogram']
            
            ax8.plot(dates[-len(macd_line):], macd_line, 'blue', linewidth=1, label='MACD')
            ax8.plot(dates[-len(macd_signal):], macd_signal, 'red', linewidth=1, label='Señal')
            
            # Histograma con colores
            for i, date in enumerate(dates[-len(macd_hist):]):
                color = 'green' if macd_hist[i] >= 0 else 'red'
                ax8.bar(date, macd_hist[i], color=color, alpha=0.6, width=0.8)
            
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax8.set_ylabel('MACD')
        ax8.legend(loc='upper right', fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        # 9. Información de la Señal
        ax9 = plt.subplot(9, 1, 9)
        ax9.axis('off')
        
        info_text = f"""
SEÑAL: {alert_data['signal']}
SCORE: {alert_data['score']:.1f}%
CRYPTO: {alert_data['symbol']}
RIESGO: {alert_data['risk_category']}
TEMPORALIDAD: {alert_data['interval']}

PRECIO: ${alert_data['current_price']:.6f}
ENTRADA: ${alert_data['entry']:.6f}
STOP LOSS: ${alert_data['stop_loss']:.6f}
TAKE PROFIT: ${alert_data['take_profit']:.6f}

MULTI-TF: {'✅' if alert_data.get('multi_timeframe_ok', False) else '❌'}
MA200: {'ENCIMA' if alert_data.get('ma200_condition') == 'above' else 'DEBAJO'}

CONDICIONES:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])][:5])}
"""
        
        ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen de entrada: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_volume_telegram_image(alert_data):
    """Generar imagen para alertas de volumen anómalo con 4 gráficos"""
    try:
        # Obtener datos para la temporalidad actual
        df_current = indicator.get_kucoin_data(alert_data['symbol'], alert_data['interval'], 50)
        
        # Obtener datos para la temporalidad menor
        df_menor = indicator.get_kucoin_data(alert_data['symbol'], alert_data['menor_interval'], 50)
        
        if df_current is None or df_menor is None:
            return None
        
        # Crear figura con 5 subgráficos (4 indicadores + 1 info)
        fig = plt.figure(figsize=(14, 12))
        fig.patch.set_facecolor('white')
        
        # Datos para gráficos principales
        dates_current = df_current['timestamp'].tail(30).tolist()
        close_current = df_current['close'].tail(30).values
        open_current = df_current['open'].tail(30).values
        high_current = df_current['high'].tail(30).values
        low_current = df_current['low'].tail(30).values
        
        dates_menor = df_menor['timestamp'].tail(50).tolist()
        close_menor = df_menor['close'].tail(50).values
        volume_menor = df_menor['volume'].tail(50).values
        
        # 1. Gráfico de Velas Japonesas con Bandas de Bollinger (Temporalidad Actual)
        ax1 = plt.subplot(5, 1, 1)
        
        # Dibujar velas
        for i in range(len(dates_current)):
            color = 'green' if close_current[i] >= open_current[i] else 'red'
            ax1.plot([dates_current[i], dates_current[i]], [low_current[i], high_current[i]], 
                    color='black', linewidth=0.5)
            ax1.plot([dates_current[i], dates_current[i]], [open_current[i], close_current[i]], 
                    color=color, linewidth=2)
        
        # Bandas de Bollinger
        bb_upper, bb_middle, bb_lower = indicator.calculate_bollinger_bands(close_current, 20, 2)
        if len(bb_upper) > 0:
            ax1.plot(dates_current[-len(bb_upper):], bb_upper, 'blue', alpha=0.3, linewidth=1)
            ax1.plot(dates_current[-len(bb_lower):], bb_lower, 'blue', alpha=0.3, linewidth=1)
            ax1.fill_between(dates_current[-len(bb_upper):], bb_upper, bb_lower, alpha=0.1, color='blue')
        
        ax1.set_title(f'{alert_data["symbol"]} - {alert_data["interval"]} - Volumen Anómalo {alert_data["signal"]}', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precio')
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI (Temporalidad Actual)
        ax2 = plt.subplot(5, 1, 2, sharex=ax1)
        
        # Calcular ADX y DMI
        if len(close_current) >= 14:
            adx, plus_di, minus_di = indicator.calculate_adx(high_current, low_current, close_current, 14)
            if len(adx) > 0:
                ax2.plot(dates_current[-len(adx):], adx, 'black', linewidth=1.5, label='ADX')
                ax2.plot(dates_current[-len(plus_di):], plus_di, 'green', linewidth=1, label='+DI')
                ax2.plot(dates_current[-len(minus_di):], minus_di, 'red', linewidth=1, label='-DI')
                ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7)
        
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Volumen con Anomalías y Clusters (Temporalidad Menor)
        ax3 = plt.subplot(5, 1, 3)
        
        # Calcular anomalías de volumen
        volume_data = indicator.calculate_volume_anomaly_improved(close_menor, volume_menor)
        
        # Barras de volumen
        bar_colors = []
        for i in range(len(dates_menor)):
            if i > 0:
                color = 'green' if close_menor[i] >= close_menor[i-1] else 'red'
            else:
                color = 'gray'
            bar_colors.append(color)
            ax3.bar(dates_menor[i], volume_menor[i], color=color, alpha=0.7, width=0.8)
        
        # Marcar anomalías
        if 'volume_anomaly_buy' in volume_data and 'volume_anomaly_sell' in volume_data:
            anomaly_buy = volume_data['volume_anomaly_buy']
            anomaly_sell = volume_data['volume_anomaly_sell']
            
            for i, date in enumerate(dates_menor[-len(anomaly_buy):]):
                if anomaly_buy[i]:
                    ax3.axvspan(date, date, alpha=0.3, color='green', ymin=0.8, ymax=1.0)
                if anomaly_sell[i]:
                    ax3.axvspan(date, date, alpha=0.3, color='red', ymin=0.8, ymax=1.0)
        
        ax3.set_title(f'Volumen - Temporalidad Menor ({alert_data["menor_interval"]})', fontsize=10)
        ax3.set_ylabel('Volumen')
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick (Temporalidad Actual)
        ax4 = plt.subplot(5, 1, 4, sharex=ax1)
        
        trend_data = indicator.calculate_trend_strength_maverick(close_current)
        if 'trend_strength' in trend_data:
            trend_strength = trend_data['trend_strength']
            colors = trend_data['colors']
            
            for i, date in enumerate(dates_current[-len(trend_strength):]):
                color = colors[i] if i < len(colors) else 'gray'
                ax4.bar(date, trend_strength[i], color=color, alpha=0.7, width=0.8)
        
        ax4.set_ylabel('Fuerza Tendencia')
        ax4.grid(True, alpha=0.3)
        
        # 5. Información de la Señal de Volumen
        ax5 = plt.subplot(5, 1, 5)
        ax5.axis('off')
        
        anomaly_type = alert_data.get('anomaly_type', 'compra')
        cluster_text = " (CLUSTER)" if alert_data.get('cluster') else ""
        
        info_text = f"""
SEÑAL: {alert_data['signal']}
CRYPTO: {alert_data['symbol']}
RIESGO: {alert_data['risk_category']}

ALERTA: Volumen Anómalo de {anomaly_type}{cluster_text}
TEMPORALIDAD: {alert_data['interval']}
TEMPORALIDAD MENOR: {alert_data['menor_interval']}

CONDICIONES FTMaverick y MF:
{chr(10).join(['• ' + cond for cond in alert_data.get('conditions', [])])}

RECOMENDACIÓN: Revisar {alert_data['signal']}
        """
        
        ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen de volumen: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    try:
        alerts = indicator.generate_scalping_alerts()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/volume_anomaly_signals')
def get_volume_anomaly_signals():
    try:
        alerts = indicator.generate_volume_anomaly_signals()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/volume_anomaly_signals: {e}")
        return jsonify({'alerts': []})

@app.route('/api/generate_report')
def generate_report():
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
            
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Líneas de soporte y resistencia
            if 'all_supports' in signal_data:
                for support in signal_data['all_supports']:
                    ax1.axhline(y=support['price'], color='orange', linestyle=':', alpha=0.5, 
                               label=f"Soporte {support['price']:.2f}")
            
            if 'all_resistances' in signal_data:
                for resistance in signal_data['all_resistances']:
                    ax1.axhline(y=resistance['price'], color='purple', linestyle=':', alpha=0.5,
                               label=f"Resistencia {resistance['price']:.2f}")
            
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit']):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: ADX/DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax2.plot(adx_dates, signal_data['indicators']['adx'], 
                    'white', linewidth=2, label='ADX')
            ax2.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax2.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        ax2.set_ylabel('ADX/DMI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Volumen con anomalías
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_ratio']):]
            volumes = [d['volume'] for d in signal_data['data'][-len(signal_data['indicators']['volume_ratio']):]]
            
            # Barras de volumen con colores
            for i, date in enumerate(volume_dates):
                color = signal_data['indicators']['volume_colors'][i] if i < len(signal_data['indicators']['volume_colors']) else 'gray'
                ax3.bar(date, volumes[i], color=color, alpha=0.7, width=0.8)
            
            # EMA de volumen
            ax3.plot(volume_dates, signal_data['indicators']['volume_ema'], 
                    'yellow', linewidth=1, label='EMA Volumen')
        
        ax3.set_ylabel('Volumen')
        ax3.legend()
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
        
        ax4.set_ylabel('Fuerza Tendencia %')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Ballenas
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax5.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax5.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
        ax5.set_ylabel('Fuerza Ballenas')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: RSI Maverick
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_maverick_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax6.plot(rsi_maverick_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        ax6.set_ylabel('RSI Maverick')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: RSI Tradicional
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax7.plot(rsi_dates, signal_data['indicators']['rsi_traditional'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax7.set_ylabel('RSI Tradicional')
        ax7.legend()
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
        return jsonify({'error': 'Error generando reporte'}), 500

@app.route('/api/bolivia_time')
def get_bolivia_time():
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d'),
        'timezone': 'America/La_Paz'
    })


# Añadir esta ruta si no existe (actual)
@app.route('/manual')
def manual():
    return render_template('manual.html')

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
