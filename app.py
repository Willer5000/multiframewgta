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
    "XMR-USDT", "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT",
    
    # Medio Riesgo (10) - Proyectos consolidados
    "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT", "IOTA-USDT",
    
    # Alto Riesgo (7) - Proyectos emergentes
    "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
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
        "XMR-USDT", "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT"
    ],
    "medio": [
        "NEAR-USDT", "FTM-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
        "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ZIL-USDT", "IOTA-USDT"
    ],
    "alto": [
        "APE-USDT", "GMT-USDT", "GAL-USDT", "OP-USDT", "ARB-USDT",
        "MAGIC-USDT", "RNDR-USDT"
    ],
    "memecoins": [
        "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT"
    ]
}

# Mapeo de temporalidades para análisis multi-timeframe MEJORADO
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
        self.initialize_winrate_tracking()
    
    def initialize_winrate_tracking(self):
        """Inicializar tracking de winrate"""
        for symbol in CRYPTO_SYMBOLS:
            self.winrate_data[symbol] = {
                'total_signals': 0,
                'winning_signals': 0,
                'history': []
            }
    
    def calculate_winrate(self, symbol):
        """Calcular winrate para un símbolo"""
        if symbol not in self.winrate_data:
            return 0
        data = self.winrate_data[symbol]
        if data['total_signals'] == 0:
            return 0
        return (data['winning_signals'] / data['total_signals']) * 100
    
    def get_overall_winrate(self):
        """Calcular winrate general del sistema"""
        total_signals = 0
        winning_signals = 0
        
        for symbol_data in self.winrate_data.values():
            total_signals += symbol_data['total_signals']
            winning_signals += symbol_data['winning_signals']
        
        if total_signals == 0:
            return 0
        return (winning_signals / total_signals) * 100
    
    def update_winrate(self, symbol, signal_type, entry_price, exit_price, timeframe):
        """Actualizar winrate basado en resultado real"""
        if signal_type == 'LONG':
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            is_win = pnl_percent > 0
        else:  # SHORT
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100
            is_win = pnl_percent > 0
        
        if symbol not in self.winrate_data:
            self.winrate_data[symbol] = {
                'total_signals': 0,
                'winning_signals': 0,
                'history': []
            }
        
        self.winrate_data[symbol]['total_signals'] += 1
        if is_win:
            self.winrate_data[symbol]['winning_signals'] += 1
        
        # Mantener solo últimos 200 registros por performance
        self.winrate_data[symbol]['history'].append({
            'timestamp': datetime.now(),
            'signal_type': signal_type,
            'timeframe': timeframe,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_percent': pnl_percent,
            'is_win': is_win
        })
        
        if len(self.winrate_data[symbol]['history']) > 200:
            self.winrate_data[symbol]['history'] = self.winrate_data[symbol]['history'][-200:]

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
            # Para 3D, simplificar lógica
            return True
        elif interval == '1W':
            return True
        
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

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con soportes/resistencias smart money"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Soporte y resistencia smart money con lookback de 50 periodos
            support_1 = np.min(low[-50:])
            resistance_1 = np.max(high[-50:])
            
            # Encontrar niveles de soporte/resistencia más cercanos (smart money)
            if signal_type == 'LONG':
                # Para LONG: entrada lo más cerca posible del soporte
                potential_entries = [price for price in close[-20:] if price <= support_1 * 1.02]
                if potential_entries:
                    entry = min(potential_entries)
                else:
                    entry = min(current_price, support_1 * 1.01)
                
                stop_loss = max(support_1 * 0.97, entry - (current_atr * 1.8))
                tp1 = resistance_1 * 0.98
                
            else:  # SHORT
                # Para SHORT: entrada lo más cerca posible de la resistencia
                potential_entries = [price for price in close[-20:] if price >= resistance_1 * 0.98]
                if potential_entries:
                    entry = max(potential_entries)
                else:
                    entry = max(current_price, resistance_1 * 0.99)
                
                stop_loss = min(resistance_1 * 1.03, entry + (current_atr * 1.8))
                tp1 = support_1 * 1.02
            
            atr_percentage = current_atr / current_price

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
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support': float(np.min(df['low'].values[-50:])),
                'resistance': float(np.max(df['high'].values[-50:])),
                'atr': 0.0,
                'atr_percentage': 0.0
            }

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

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),  # HCH - SEÑAL SHORT
            'double_top': np.zeros(n, dtype=bool),      # Doble Techo - SEÑAL SHORT
            'double_bottom': np.zeros(n, dtype=bool),   # Doble Fondo - SEÑAL LONG
            'rising_wedge': np.zeros(n, dtype=bool),    # Cuña Ascendente - SEÑAL SHORT
            'falling_wedge': np.zeros(n, dtype=bool),   # Cuña Descendente - SEÑAL LONG
            'bull_flag': np.zeros(n, dtype=bool),       # Banderín Alcista - SEÑAL LONG
            'asc_triangle': np.zeros(n, dtype=bool),    # Triángulo Ascendente - SEÑAL LONG
            'bear_rectangle': np.zeros(n, dtype=bool)   # Rectángulo Bajista - SEÑAL LONG
        }
        
        for i in range(lookback, n-7):
            # Head & Shoulders (simplificado)
            window_high = high[i-lookback:i]
            if len(window_high) >= 10:
                peaks = []
                for j in range(1, len(window_high)-1):
                    if (window_high[j] > window_high[j-1] and 
                        window_high[j] > window_high[j+1]):
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 3:
                    peaks.sort(key=lambda x: x[1], reverse=True)
                    if (peaks[0][1] > peaks[1][1] * 1.02 and 
                        peaks[0][1] > peaks[2][1] * 1.02 and
                        abs(peaks[1][1] - peaks[2][1]) < peaks[1][1] * 0.01):
                        patterns['head_shoulders'][i] = True
            
            # Double Top
            if (high[i] >= np.max(high[i-20:i]) and 
                high[i-5] >= np.max(high[i-25:i-5]) * 0.98 and
                abs(high[i] - high[i-5]) < high[i] * 0.02):
                patterns['double_top'][i] = True
            
            # Double Bottom
            if (low[i] <= np.min(low[i-20:i]) and 
                low[i-5] <= np.min(low[i-25:i-5]) * 1.02 and
                abs(low[i] - low[i-5]) < low[i] * 0.02):
                patterns['double_bottom'][i] = True
        
        return patterns

    def calculate_squeeze_momentum(self, high, low, close, length=20):
        """Calcular Squeeze Momentum"""
        n = len(close)
        
        # Calcular Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, length, 2)
        
        # Calcular Keltner Channel
        tr = self.calculate_atr(high, low, close, length)
        kc_upper = bb_middle + (tr * 1.5)
        kc_lower = bb_middle - (tr * 1.5)
        
        # Detectar squeeze
        squeeze_on = np.zeros(n, dtype=bool)
        squeeze_off = np.zeros(n, dtype=bool)
        momentum = np.zeros(n)
        
        for i in range(n):
            # Squeeze ON cuando BB dentro de KC
            if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
                squeeze_on[i] = True
            # Squeeze OFF cuando BB fuera de KC
            elif bb_upper[i] > kc_upper[i] or bb_lower[i] < kc_lower[i]:
                squeeze_off[i] = True
            
            # Momentum basado en cambio de precio
            if i > 0:
                momentum[i] = close[i] - close[i-1]
        
        return {
            'squeeze_on': squeeze_on.tolist(),
            'squeeze_off': squeeze_off.tolist(),
            'momentum': momentum.tolist(),
            'bb_upper': bb_upper.tolist(),
            'bb_lower': bb_lower.tolist(),
            'kc_upper': kc_upper.tolist(),
            'kc_lower': kc_lower.tolist()
        }

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
        """Verificar tendencia en múltiples temporalidades MEJORADO"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(timeframe, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:  # No usar 5m y 1M para análisis
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
                
                # Determinar tendencia con múltiples condiciones
                bullish_conditions = 0
                bearish_conditions = 0
                
                if current_price > current_ma_9:
                    bullish_conditions += 1
                else:
                    bearish_conditions += 1
                    
                if current_ma_9 > current_ma_21:
                    bullish_conditions += 1
                else:
                    bearish_conditions += 1
                    
                if current_price > current_ma_50:
                    bullish_conditions += 1
                else:
                    bearish_conditions += 1
                
                if bullish_conditions >= 2:
                    results[tf_type] = 'BULLISH'
                elif bearish_conditions >= 2:
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}

    def check_multi_timeframe_obligatory_conditions(self, symbol, interval, signal_type):
        """Verificar condiciones OBLIGATORIAS multi-timeframe"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False  # Sin jerarquía, no operar
            
            # Obtener análisis de todas las temporalidades
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            # Verificar fuerza de tendencia Maverick en todas las TF
            maverick_conditions_met = True
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 30)
                if df is not None and len(df) > 10:
                    trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                    current_signal = trend_data['strength_signals'][-1]
                    current_no_trade = trend_data['no_trade_zones'][-1]
                    
                    # OBLIGATORIO: Sin zonas de NO OPERAR
                    if current_no_trade:
                        maverick_conditions_met = False
                        break
                    
                    # OBLIGATORIO: Fuerza de tendencia alineada
                    if signal_type == 'LONG':
                        if current_signal not in ['STRONG_UP', 'WEAK_UP']:
                            maverick_conditions_met = False
                            break
                    else:  # SHORT
                        if current_signal not in ['STRONG_DOWN', 'WEAK_DOWN']:
                            maverick_conditions_met = False
                            break
            
            if not maverick_conditions_met:
                return False
            
            # OBLIGATORIO: Condiciones de tendencia por temporalidad
            if signal_type == 'LONG':
                # Mayor: ALCISTA o NEUTRAL
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                # Media: EXCLUSIVAMENTE ALCISTA
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'
                # Menor: Confirmación señal + fuerza alcista
                menor_ok = tf_analysis.get('menor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                
                return mayor_ok and media_ok and menor_ok
                
            elif signal_type == 'SHORT':
                # Mayor: BAJISTA o NEUTRAL
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                # Media: EXCLUSIVAMENTE BAJISTA
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'
                # Menor: Confirmación señal + fuerza bajista
                menor_ok = tf_analysis.get('menor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                
                return mayor_ok and media_ok and menor_ok
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias multi-timeframe: {e}")
            return False

    def calculate_whale_signals_improved(self, df, interval, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=50, signal_threshold=25, 
                                       sell_signal_threshold=20):
        """Implementación MEJORADA del indicador de ballenas con exclusividad 12H/1D"""
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
            
            # PARA 12H y 1D: Señal completa
            # PARA otras TF: Señal visible pero no obligatoria
            is_whale_obligatory = interval in ['12h', '1D']
            
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
                'volume_anomaly': (volume > np.mean(volume) * min_volume_multiplier).tolist(),
                'is_obligatory': is_whale_obligatory
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
                'volume_anomaly': [False] * n,
                'is_obligatory': False
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

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias entre precio e indicador"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Divergencia alcista: precio hace lower low, indicador hace higher low
            price_low = min(price[i-lookback:i])
            indicator_low = min(indicator[i-lookback:i])
            
            if (price[i] < price_low and 
                indicator[i] > indicator_low and
                price[i] < np.min(price[i-7:i])):
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace higher high, indicador hace lower high
            price_high = max(price[i-lookback:i])
            indicator_high = max(indicator[i-lookback:i])
            
            if (price[i] > price_high and 
                indicator[i] < indicator_high and
                price[i] > np.max(price[i-7:i])):
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

    def evaluate_signal_conditions_improved(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con lógica MEJORADA y pesos actualizados"""
        conditions = {
            'long': {
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación Medias Móviles (MA9, MA21, MA50, MA200)'},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI Tradicional + Divergencias'},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick + Divergencias'},
                'smart_money_levels': {'value': False, 'weight': 20, 'description': 'Soportes/Resistencias Smart Money'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} + DMI favorable'},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD + Histograma favorable'},
                'squeeze_momentum': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable'},
                'bollinger_bands': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger favorables'},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrones Chartismo favorables'}
            },
            'short': {
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación Medias Móviles (MA9, MA21, MA50, MA200)'},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI Tradicional + Divergencias'},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick + Divergencias'},
                'smart_money_levels': {'value': False, 'weight': 20, 'description': 'Soportes/Resistencias Smart Money'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} + DMI favorable'},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD + Histograma favorable'},
                'squeeze_momentum': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable'},
                'bollinger_bands': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger favorables'},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrones Chartismo favorables'}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        current_price = data['close'][current_idx]
        
        # 1. MOVING AVERAGES (15%)
        ma_9 = data['ma_9'][current_idx] if current_idx < len(data['ma_9']) else 0
        ma_21 = data['ma_21'][current_idx] if current_idx < len(data['ma_21']) else 0
        ma_50 = data['ma_50'][current_idx] if current_idx < len(data['ma_50']) else 0
        ma_200 = data['ma_200'][current_idx] if current_idx < len(data['ma_200']) else 0
        
        conditions['long']['moving_averages']['value'] = (
            current_price > ma_9 and ma_9 > ma_21 and ma_21 > ma_50
        )
        conditions['short']['moving_averages']['value'] = (
            current_price < ma_9 and ma_9 < ma_21 and ma_21 < ma_50
        )
        
        # 2. RSI TRADITIONAL + DIVERGENCES (15%)
        rsi_traditional = data['rsi_traditional'][current_idx] if current_idx < len(data['rsi_traditional']) else 50
        conditions['long']['rsi_traditional']['value'] = (
            rsi_traditional < 70 and  # No sobrecomprado
            (data['bullish_div_rsi'][current_idx] if current_idx < len(data['bullish_div_rsi']) else False)
        )
        conditions['short']['rsi_traditional']['value'] = (
            rsi_traditional > 30 and  # No sobrevendido
            (data['bearish_div_rsi'][current_idx] if current_idx < len(data['bearish_div_rsi']) else False)
        )
        
        # 3. RSI MAVERICK + DIVERGENCES (15%)
        rsi_maverick = data['rsi_maverick'][current_idx] if current_idx < len(data['rsi_maverick']) else 0.5
        conditions['long']['rsi_maverick']['value'] = (
            rsi_maverick < 0.8 and  # No sobrecomprado
            (data['bullish_div_maverick'][current_idx] if current_idx < len(data['bullish_div_maverick']) else False)
        )
        conditions['short']['rsi_maverick']['value'] = (
            rsi_maverick > 0.2 and  # No sobrevendido
            (data['bearish_div_maverick'][current_idx] if current_idx < len(data['bearish_div_maverick']) else False)
        )
        
        # 4. SMART MONEY LEVELS (20%)
        support = data['support'][current_idx] if current_idx < len(data['support']) else 0
        resistance = data['resistance'][current_idx] if current_idx < len(data['resistance']) else 0
        
        conditions['long']['smart_money_levels']['value'] = (
            current_price <= support * 1.02  # Precio cerca del soporte
        )
        conditions['short']['smart_money_levels']['value'] = (
            current_price >= resistance * 0.98  # Precio cerca de la resistencia
        )
        
        # 5. ADX + DMI (10%)
        adx = data['adx'][current_idx] if current_idx < len(data['adx']) else 0
        plus_di = data['plus_di'][current_idx] if current_idx < len(data['plus_di']) else 0
        minus_di = data['minus_di'][current_idx] if current_idx < len(data['minus_di']) else 0
        
        conditions['long']['adx_dmi']['value'] = (
            adx > adx_threshold and plus_di > minus_di
        )
        conditions['short']['adx_dmi']['value'] = (
            adx > adx_threshold and minus_di > plus_di
        )
        
        # 6. MACD (10%)
        macd_histogram = data['macd_histogram'][current_idx] if current_idx < len(data['macd_histogram']) else 0
        conditions['long']['macd']['value'] = macd_histogram > 0
        conditions['short']['macd']['value'] = macd_histogram < 0
        
        # 7. SQUEEZE MOMENTUM (10%)
        squeeze_momentum = data['squeeze_momentum'][current_idx] if current_idx < len(data['squeeze_momentum']) else 0
        conditions['long']['squeeze_momentum']['value'] = squeeze_momentum > 0
        conditions['short']['squeeze_momentum']['value'] = squeeze_momentum < 0
        
        # 8. BOLLINGER BANDS (5%)
        bb_position = data['bb_position'][current_idx] if current_idx < len(data['bb_position']) else 0.5
        conditions['long']['bollinger_bands']['value'] = bb_position < 0.8  # No en banda superior
        conditions['short']['bollinger_bands']['value'] = bb_position > 0.2  # No en banda inferior
        
        # 9. CHART PATTERNS (15%)
        conditions['long']['chart_patterns']['value'] = (
            data['chart_patterns_long'][current_idx] if current_idx < len(data['chart_patterns_long']) else False
        )
        conditions['short']['chart_patterns']['value'] = (
            data['chart_patterns_short'][current_idx] if current_idx < len(data['chart_patterns_short']) else False
        )
        
        return conditions

    def calculate_signal_score(self, conditions, signal_type, obligatory_conditions_met):
        """Calcular puntuación de señal con multiplicador de obligatorios"""
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
        
        # APLICAR MULTIPLICADOR DE OBLIGATORIOS
        base_score = (achieved_weight / total_weight * 100)
        
        if obligatory_conditions_met:
            final_score = base_score
        else:
            final_score = 0  # Si no se cumplen obligatorios, score = 0

        return min(final_score, 100), fulfilled_conditions

    def generate_exit_signals(self):
        """Generar señales de salida para operaciones activas MEJORADO"""
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
                
                # Obtener datos actuales (máximo 20 velas posteriores)
                df = self.get_kucoin_data(symbol, interval, 40)  # Obtener más datos para análisis
                if df is None or len(df) < 10:
                    continue
                
                # Encontrar índice de entrada aproximado
                entry_idx = -1
                for i in range(len(df)):
                    if abs(float(df['close'].iloc[i]) - entry_price) / entry_price < 0.01:
                        entry_idx = i
                        break
                
                if entry_idx == -1:
                    entry_idx = len(df) - 20  # Asumir entrada hace 20 velas
                
                current_idx = len(df) - 1
                candles_since_entry = current_idx - entry_idx
                
                # Solo verificar señales de salida para operaciones con al menos 1 vela desde entrada
                if candles_since_entry < 1:
                    continue
                
                current_price = float(df['close'].iloc[current_idx])
                
                # Verificar condiciones de salida
                exit_reason = None
                
                # 1. Verificar fuerza de tendencia Maverick
                trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                current_strength = trend_data['strength_signals'][current_idx]
                current_no_trade = trend_data['no_trade_zones'][current_idx]
                
                if signal_type == 'LONG':
                    if current_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                        exit_reason = "Cambio de fuerza de tendencia a bajista"
                    elif current_no_trade:
                        exit_reason = "Activación de zona NO OPERAR"
                else:  # SHORT
                    if current_strength in ['STRONG_UP', 'WEAK_UP']:
                        exit_reason = "Cambio de fuerza de tendencia a alcista"
                    elif current_no_trade:
                        exit_reason = "Activación de zona NO OPERAR"
                
                # 2. Verificar ruptura de niveles clave
                support = np.min(df['low'].values[-50:])
                resistance = np.max(df['high'].values[-50:])
                
                if signal_type == 'LONG' and current_price < support * 0.98:
                    exit_reason = "Ruptura de soporte clave"
                elif signal_type == 'SHORT' and current_price > resistance * 1.02:
                    exit_reason = "Ruptura de resistencia clave"
                
                # 3. Verificar cambio en temporalidad menor
                hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                if hierarchy.get('menor') and exit_reason is None:
                    menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 15)
                    if menor_df is not None and len(menor_df) > 5:
                        menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                        menor_strength = menor_trend['strength_signals'][-1]
                        
                        if signal_type == 'LONG' and menor_strength in ['STRONG_DOWN', 'WEAK_DOWN']:
                            exit_reason = "Cambio de tendencia en temporalidad menor"
                        elif signal_type == 'SHORT' and menor_strength in ['STRONG_UP', 'WEAK_UP']:
                            exit_reason = "Cambio de tendencia en temporalidad menor"
                
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
                        'pnl_percent': pnl_percent,
                        'reason': exit_reason,
                        'trend_strength': current_strength,
                        'candles_since_entry': candles_since_entry,
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    exit_alerts.append(exit_alert)
                    
                    # Actualizar winrate
                    self.update_winrate(symbol, signal_type, entry_price, current_price, interval)
                    
                    # Remover señal activa
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=20, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - CON SISTEMA COMPLETO DE INDICADORES"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # 1. INDICADOR BALLENAS (con exclusividad 12H/1D)
            whale_data = self.calculate_whale_signals_improved(df, interval, support_resistance_lookback=sr_period)
            
            # 2. ADX + DMI
            adx, plus_di, minus_di = self.calculate_adx(
                df['high'].values, df['low'].values, df['close'].values, di_period
            )
            
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            # 3. RSI MAVERICK
            rsi_maverick = self.calculate_rsi_maverick(
                df['close'].values, rsi_length, bb_multiplier
            )
            
            # 4. RSI TRADICIONAL
            rsi_traditional = self.calculate_rsi(df['close'].values, 14)
            
            # 5. DIVERGENCIAS
            bullish_div_maverick, bearish_div_maverick = self.detect_divergence(
                df['close'].values, rsi_maverick
            )
            
            bullish_div_rsi, bearish_div_rsi = self.detect_divergence(
                df['close'].values, rsi_traditional
            )
            
            # 6. RUPTURAS
            breakout_up, breakout_down = self.check_breakout(
                df['high'].values, df['low'].values, df['close'].values,
                whale_data['support'], whale_data['resistance']
            )
            
            # 7. FUERZA DE TENDENCIA MAVERICK
            trend_strength_data = self.calculate_trend_strength_maverick(
                df['close'].values, length=20, mult=2.0
            )
            
            # 8. MEDIAS MÓVILES
            ma_9 = self.calculate_sma(df['close'].values, 9)
            ma_21 = self.calculate_sma(df['close'].values, 21)
            ma_50 = self.calculate_sma(df['close'].values, 50)
            ma_200 = self.calculate_sma(df['close'].values, 200)
            
            # 9. MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(df['close'].values)
            
            # 10. SQUEEZE MOMENTUM
            squeeze_data = self.calculate_squeeze_momentum(
                df['high'].values, df['low'].values, df['close'].values
            )
            
            # 11. BANDAS DE BOLLINGER
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'].values)
            bb_position = np.zeros(len(df['close'].values))
            for i in range(len(bb_position)):
                if (bb_upper[i] - bb_lower[i]) > 0:
                    bb_position[i] = (df['close'].values[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
                else:
                    bb_position[i] = 0.5
            
            # 12. PATRONES DE CHARTISMO
            chart_patterns = self.detect_chart_patterns(
                df['high'].values, df['low'].values, df['close'].values
            )
            
            current_idx = -1
            
            # Preparar datos para análisis
            analysis_data = {
                'close': df['close'].values,
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'rsi_traditional': rsi_traditional,
                'rsi_maverick': rsi_maverick,
                'bullish_div_rsi': bullish_div_rsi,
                'bearish_div_rsi': bearish_div_rsi,
                'bullish_div_maverick': bullish_div_maverick,
                'bearish_div_maverick': bearish_div_maverick,
                'support': whale_data['support'],
                'resistance': whale_data['resistance'],
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'macd_histogram': macd_histogram,
                'squeeze_momentum': squeeze_data['momentum'],
                'bb_position': bb_position,
                'chart_patterns_long': chart_patterns['double_bottom'],
                'chart_patterns_short': chart_patterns['double_top']
            }
            
            conditions = self.evaluate_signal_conditions_improved(analysis_data, current_idx, interval, adx_threshold)
            
            # VERIFICAR OBLIGATORIOS MULTI-TIMEFRAME
            obligatory_long_met = self.check_multi_timeframe_obligatory_conditions(symbol, interval, 'LONG')
            obligatory_short_met = self.check_multi_timeframe_obligatory_conditions(symbol, interval, 'SHORT')
            
            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', obligatory_long_met)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', obligatory_short_met)
            
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= 70:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= 70:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            current_price = float(df['close'].iloc[current_idx])
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # Registrar señal activa si es válida
            if signal_type in ['LONG', 'SHORT'] and signal_score >= 70:
                signal_key = f"{symbol}_{interval}_{signal_type}_{current_time.timestamp()}"
                self.active_signals[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': levels_data['entry'],
                    'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                    'score': signal_score,
                    'winrate': self.calculate_winrate(symbol)
                }
            
            return {
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
                'rsi_traditional': float(rsi_traditional[current_idx] if current_idx < len(rsi_traditional) else 50),
                'fulfilled_conditions': fulfilled_conditions,
                'obligatory_conditions_met': obligatory_long_met if signal_type == 'LONG' else obligatory_short_met,
                'trend_strength_signal': trend_strength_data['strength_signals'][current_idx] if current_idx < len(trend_strength_data['strength_signals']) else 'NEUTRAL',
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False,
                'winrate': self.calculate_winrate(symbol),
                'overall_winrate': self.get_overall_winrate(),
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:],
                    'bullish_div_maverick': bullish_div_maverick[-50:],
                    'bearish_div_maverick': bearish_div_maverick[-50:],
                    'bullish_div_rsi': bullish_div_rsi[-50:],
                    'bearish_div_rsi': bearish_div_rsi[-50:],
                    'support': whale_data['support'][-50:],
                    'resistance': whale_data['resistance'][-50:],
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['momentum'][-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'bb_position': bb_position[-50:].tolist(),
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
                    'colors': trend_strength_data['colors'][-50:],
                    'chart_patterns_long': chart_patterns['double_bottom'][-50:],
                    'chart_patterns_short': chart_patterns['double_top'][-50:]
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
            'rsi_traditional': 50,
            'fulfilled_conditions': [],
            'obligatory_conditions_met': False,
            'trend_strength_signal': 'NEUTRAL',
            'no_trade_zone': False,
            'winrate': 0,
            'overall_winrate': 0,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de scalping con filtros mejorados"""
        alerts = []
        telegram_intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '3D', '1W']
        
        current_time = self.get_bolivia_time()
        
        for interval in telegram_intervals:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:15]:
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 70 and
                        signal_data['obligatory_conditions_met']):
                        
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
                            'resistance': signal_data['resistance'],
                            'winrate': signal_data.get('winrate', 0),
                            'overall_winrate': signal_data.get('overall_winrate', 0),
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
    """Enviar alerta por Telegram MEJORADA"""
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
            
            # Información de winrate
            winrate_info = f"📊 Winrate: {alert_data.get('winrate', 0):.1f}% (Global: {alert_data.get('overall_winrate', 0):.1f}%)"
            
            message = f"""
🚨 ALERTA DE TRADING - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%

{winrate_info}

💰 Precio actual: {alert_data.get('current_price', alert_data['entry']):.6f}
💪 Fuerza de Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

🎯 ENTRADA: ${alert_data['entry']:.6f}
🛑 STOP LOSS: ${alert_data['stop_loss']:.6f}
(Explicación: {stop_explanation})

📈 Apalancamiento: x{alert_data['leverage']}
{conditions_text}

✅ OBLIGATORIOS MULTI-TF: {'CUMPLIDOS' if alert_data.get('obligatory_conditions_met', False) else 'NO CUMPLIDOS'}

📊 Revisa la señal en: https://ballenasscalpistas.onrender.com/
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

📊 Observación: {alert_data['reason']}
⏱️ Velas desde entrada: {alert_data.get('candles_since_entry', 'N/A')}
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
    """Verificador de alertas en segundo plano MEJORADO"""
    intraday_intervals = ['15m', '30m', '1h', '2h']
    swing_intervals = ['4h', '8h', '12h', '1D', '3D', '1W']
    
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
    """Endpoint para obtener señales de trading MEJORADO"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 20))
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
    """Endpoint para obtener múltiples señales MEJORADO"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 20))
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
                
                if signal_data and signal_data['signal'] != 'NEUTRAL' and signal_data['signal_score'] >= 70:
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
            'total_signals': len(all_signals),
            'overall_winrate': indicator.get_overall_winrate()
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
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:3])
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval, di_period, adx_threshold)
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
    """Endpoint para obtener winrate del sistema"""
    try:
        overall_winrate = indicator.get_overall_winrate()
        symbol_winrates = {}
        
        for symbol in CRYPTO_SYMBOLS[:10]:
            symbol_winrates[symbol] = indicator.calculate_winrate(symbol)
        
        return jsonify({
            'overall_winrate': overall_winrate,
            'symbol_winrates': symbol_winrates,
            'total_signals_analyzed': sum(data['total_signals'] for data in indicator.winrate_data.values())
        })
        
    except Exception as e:
        print(f"Error en /api/winrate: {e}")
        return jsonify({'overall_winrate': 0, 'symbol_winrates': {}})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo MEJORADO"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        fig = plt.figure(figsize=(14, 16))
        fig.suptitle(f'REPORTE ESTRATÉGICO - {symbol} ({interval})\nMULTI-TIMEFRAME CRYPTO WGTA PRO', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Gráfico 1: Precio y niveles clave
        ax1 = plt.subplot(8, 1, 1)
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
            
            ax1.axhline(y=signal_data['support'], color='orange', linestyle=':', alpha=0.5, label='Soporte')
            ax1.axhline(y=signal_data['resistance'], color='purple', linestyle=':', alpha=0.5, label='Resistencia')
        
        ax1.set_title('PRECIO Y NIVELES SMART MONEY', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Ballenas (visible en todas las TF)
        ax2 = plt.subplot(8, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax2.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax2.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
            
            # Marcar si es obligatorio para esta TF
            if interval in ['12h', '1D']:
                ax2.text(0.02, 0.95, 'OBLIGATORIO 12H/1D', transform=ax2.transAxes, 
                        fontsize=10, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        ax2.set_ylabel('Fuerza Ballenas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: ADX/DMI
        ax3 = plt.subplot(8, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax3.plot(adx_dates, signal_data['indicators']['adx'], 
                    'white', linewidth=2, label='ADX')
            ax3.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax3.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax3.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral ADX 25')
        ax3.set_ylabel('ADX/DMI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: RSI Tradicional + Maverick
        ax4 = plt.subplot(8, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi_traditional'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax4.plot(rsi_dates, [x * 100 for x in signal_data['indicators']['rsi_maverick']], 
                    'magenta', linewidth=2, label='RSI Maverick (%B × 100)')
            ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax4.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax4.set_ylabel('RSI Comparativo')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: MACD
        ax5 = plt.subplot(8, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax5.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax5.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            # Histograma con colores
            histogram = signal_data['indicators']['macd_histogram']
            colors = ['green' if x > 0 else 'red' for x in histogram]
            ax5.bar(macd_dates, histogram, color=colors, alpha=0.6, label='Histograma')
            
            ax5.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        ax5.set_ylabel('MACD')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Squeeze Momentum
        ax6 = plt.subplot(8, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            squeeze_dates = dates[-len(signal_data['indicators']['squeeze_momentum']):]
            momentum = signal_data['indicators']['squeeze_momentum']
            colors = ['green' if x > 0 else 'red' for x in momentum]
            ax6.bar(squeeze_dates, momentum, color=colors, alpha=0.7, label='Squeeze Momentum')
            
            ax6.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            
            # Marcar squeeze on/off
            squeeze_on = signal_data['indicators']['squeeze_on']
            squeeze_off = signal_data['indicators']['squeeze_off']
            
            for i, date in enumerate(squeeze_dates):
                if i < len(squeeze_on) and squeeze_on[i]:
                    ax6.axvline(x=date, color='yellow', alpha=0.3, linewidth=1)
                if i < len(squeeze_off) and squeeze_off[i]:
                    ax6.axvline(x=date, color='orange', alpha=0.3, linewidth=1)
        ax6.set_ylabel('Squeeze')
        ax6.legend()
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
                    ax7.axvline(x=date, color='red', alpha=0.5, linewidth=2)
            
            ax7.set_ylabel('Fuerza Tendencia %')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Información de la señal MEJORADA
        ax8 = plt.subplot(8, 1, 8)
        ax8.axis('off')
        
        # Información de obligatorios
        obligatory_status = "✅ CUMPLIDOS" if signal_data.get('obligatory_conditions_met', False) else "❌ NO CUMPLIDOS"
        
        # Información de winrate
        winrate_info = f"Winrate {symbol}: {signal_data.get('winrate', 0):.1f}% | Global: {signal_data.get('overall_winrate', 0):.1f}%"
        
        # Estado de ballenas
        whale_status = "OBLIGATORIO" if interval in ['12h', '1D'] else "VISIBLE"
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']} | SCORE: {signal_data['signal_score']:.1f}%
        {winrate_info}
        
        OBLIGATORIOS MULTI-TIMEFRAME: {obligatory_status}
        INDICADOR BALLENAS: {whale_status} (12H/1D)
        FUERZA TENDENCIA: {signal_data.get('trend_strength_signal', 'NEUTRAL')}
        ZONA NO OPERAR: {'✅ NO' if not signal_data.get('no_trade_zone', False) else '❌ ACTIVA'}
        
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT 1: ${signal_data['take_profit'][0]:.6f}
        
        APALANCAMIENTO: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', ['Ninguna'])[:5]])}
        """
        
        ax8.text(0.1, 0.9, signal_info, transform=ax8.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
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
        'timezone': 'America/La_Paz',
        'is_scalping_time': indicator.is_scalping_time(),
        'day_of_week': current_time.strftime('%A')
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
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'winrate': indicator.get_overall_winrate(),
        'active_signals': len(indicator.active_signals)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
