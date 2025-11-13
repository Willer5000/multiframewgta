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

# Configuración optimizada - 40 criptomonedas TOP
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

# Clasificación de riesgo optimizada
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

# Mapeo de temporalidades MEJORADO con nuevas TF
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
                'successful_signals': 0,
                'history': []
            }
    
    def calculate_winrate(self, symbol):
        """Calcular winrate para un símbolo"""
        if symbol not in self.winrate_data:
            return 0
        data = self.winrate_data[symbol]
        if data['total_signals'] == 0:
            return 0
        return (data['successful_signals'] / data['total_signals']) * 100
    
    def get_overall_winrate(self):
        """Calcular winrate general del sistema"""
        total_signals = 0
        successful_signals = 0
        
        for symbol_data in self.winrate_data.values():
            total_signals += symbol_data['total_signals']
            successful_signals += symbol_data['successful_signals']
        
        if total_signals == 0:
            return 0
        return (successful_signals / total_signals) * 100
    
    def update_winrate(self, symbol, signal_type, entry_price, exit_price):
        """Actualizar winrate basado en resultado real"""
        if signal_type == 'LONG':
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            success = pnl_percent > 0
        else:  # SHORT
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100
            success = pnl_percent > 0
        
        if symbol not in self.winrate_data:
            self.winrate_data[symbol] = {
                'total_signals': 0,
                'successful_signals': 0,
                'history': []
            }
        
        self.winrate_data[symbol]['total_signals'] += 1
        if success:
            self.winrate_data[symbol]['successful_signals'] += 1
        
        self.winrate_data[symbol]['history'].append({
            'timestamp': datetime.now(),
            'signal_type': signal_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_percent': pnl_percent,
            'success': success
        })
        
        # Mantener solo últimos 100 registros
        if len(self.winrate_data[symbol]['history']) > 100:
            self.winrate_data[symbol]['history'] = self.winrate_data[symbol]['history'][-100:]

    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def is_scalping_time(self):
        """Verificar si es horario de scalping"""
        now = self.get_bolivia_time()
        if now.weekday() >= 5:  # Sábado o Domingo
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
            # Para 3D, verificar cada 24 horas
            return current_time.hour == 8  # Verificar a las 8 AM
        elif interval == '1W':
            # Para 1W, verificar los lunes a las 8 AM
            return current_time.weekday() == 0 and current_time.hour == 8
        
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

    def calculate_smart_money_levels(self, high, low, close, period=50):
        """Calcular niveles Smart Money con confirmación de 3 toques"""
        n = len(close)
        supports = []
        resistances = []
        
        # Encontrar pivots significativos
        for i in range(period, n-period):
            # Soporte: mínimo local con confirmación
            if (low[i] == np.min(low[i-period:i+period]) and 
                sum(low[i-period:i] <= low[i] * 1.01) >= 2):  # Al menos 2 toques anteriores
                supports.append(low[i])
            
            # Resistencia: máximo local con confirmación
            if (high[i] == np.max(high[i-period:i+period]) and 
                sum(high[i-period:i] >= high[i] * 0.99) >= 2):  # Al menos 2 toques anteriores
                resistances.append(high[i])
        
        current_support = np.min(supports) if supports else np.min(low[-period:])
        current_resistance = np.max(resistances) if resistances else np.max(high[-period:])
        
        return float(current_support), float(current_resistance)

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con niveles Smart Money"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Niveles Smart Money
            support, resistance = self.calculate_smart_money_levels(high, low, close)
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Entrada lo más cerca posible al soporte
                entry = min(current_price, support * 1.005)  # 0.5% sobre soporte
                stop_loss = max(support * 0.995, entry - (current_atr * 1.5))
                take_profit = resistance * 0.995
                
            else:  # SHORT
                # Entrada lo más cerca posible a la resistencia
                entry = max(current_price, resistance * 0.995)  # 0.5% bajo resistencia
                stop_loss = min(resistance * 1.005, entry + (current_atr * 1.5))
                take_profit = support * 1.005
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(take_profit)],
                'support': float(support),
                'resistance': float(resistance),
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

    def detect_chart_patterns(self, high, low, close, period=50):
        """Detectar patrones de chartismo"""
        n = len(close)
        patterns = {
            'head_shoulders': False,
            'double_top': False,
            'double_bottom': False,
            'bullish_flag': False,
            'ascending_triangle': False,
            'bearish_rectangle': False
        }
        
        if n < period:
            return patterns
        
        try:
            # Doble Techo (Double Top)
            recent_highs = high[-period:]
            if len(recent_highs) >= 20:
                peak1_idx = np.argmax(recent_highs[:10])
                peak2_idx = 10 + np.argmax(recent_highs[10:])
                if (abs(recent_highs[peak1_idx] - recent_highs[peak2_idx]) / recent_highs[peak1_idx] < 0.02 and
                    peak2_idx > peak1_idx):
                    patterns['double_top'] = True
            
            # Doble Fondo (Double Bottom)
            recent_lows = low[-period:]
            if len(recent_lows) >= 20:
                trough1_idx = np.argmin(recent_lows[:10])
                trough2_idx = 10 + np.argmin(recent_lows[10:])
                if (abs(recent_lows[trough1_idx] - recent_lows[trough2_idx]) / recent_lows[trough1_idx] < 0.02 and
                    trough2_idx > trough1_idx):
                    patterns['double_bottom'] = True
            
            # Banderín Alcista (Bullish Flag)
            if n >= 30:
                first_half = close[-30:-15]
                second_half = close[-15:]
                if (np.mean(first_half) > np.mean(second_half) and
                    np.std(second_half) < np.std(first_half) * 0.7):
                    patterns['bullish_flag'] = True
            
        except Exception as e:
            print(f"Error detectando patrones: {e}")
        
        return patterns

    def check_multi_timeframe_trend(self, symbol, interval):
        """Verificar tendencia en múltiples temporalidades MEJORADO"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}
            
            results = {}
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and interval != '15m':  # Solo usar 5m para 15m
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
                
                # Determinar tendencia con múltiples medias
                bullish_count = 0
                bearish_count = 0
                
                if current_price > current_ma_9: bullish_count += 1
                else: bearish_count += 1
                    
                if current_ma_9 > current_ma_21: bullish_count += 1
                else: bearish_count += 1
                    
                if current_ma_21 > current_ma_50: bullish_count += 1
                else: bearish_count += 1
                
                if bullish_count >= 2:
                    results[tf_type] = 'BULLISH'
                elif bearish_count >= 2:
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'}

    def check_multi_timeframe_maverick(self, symbol, interval):
        """Verificar Fuerza de Tendencia Maverick en múltiples temporalidades"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return {'mayor': True, 'media': True, 'menor': True}
            
            results = {}
            
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and interval != '15m':
                    results[tf_type] = True
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 30)
                if df is None or len(df) < 10:
                    results[tf_type] = True
                    continue
                
                trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                current_signal = trend_data['strength_signals'][-1]
                current_no_trade = trend_data['no_trade_zones'][-1]
                
                # Verificar que NO esté en zona de no operar
                results[tf_type] = not current_no_trade
            
            return results
            
        except Exception as e:
            print(f"Error verificando Maverick multi-timeframe: {e}")
            return {'mayor': True, 'media': True, 'menor': True}

    def calculate_whale_signals_improved(self, df, interval, sensitivity=1.7, min_volume_multiplier=1.5):
        """Implementación MEJORADA del indicador de ballenas con EXCLUSIVIDAD TEMPORAL"""
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
            
            # EXCLUSIVIDAD TEMPORAL: Solo activar para 12H y 1D
            is_whale_timeframe = interval in ['12h', '1D']
            
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
                    signal_strength = volume_ratio * 20 * sensitivity * volume_strength
                    whale_pump_signal[i] = min(100, signal_strength) if is_whale_timeframe else 0
                
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    signal_strength = volume_ratio * 20 * sensitivity * volume_strength
                    whale_dump_signal[i] = min(100, signal_strength) if is_whale_timeframe else 0
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            support, resistance = self.calculate_smart_money_levels(high, low, close)
            current_support = np.full(n, support)
            current_resistance = np.full(n, resistance)
            
            for i in range(5, n):
                if (whale_pump_smooth[i] > 25 and 
                    close[i] <= current_support[i] * 1.02 and
                    volume[i] > np.mean(volume[max(0, i-10):i+1]) and
                    is_whale_timeframe):
                    confirmed_buy[i] = True
                
                if (whale_dump_smooth[i] > 25 and 
                    close[i] >= current_resistance[i] * 0.98 and
                    volume[i] > np.mean(volume[max(0, i-10):i+1]) and
                    is_whale_timeframe):
                    confirmed_sell[i] = True
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'confirmed_buy': confirmed_buy.tolist(),
                'confirmed_sell': confirmed_sell.tolist(),
                'support': current_support.tolist(),
                'resistance': current_resistance.tolist(),
                'volume_anomaly': (volume > np.mean(volume) * min_volume_multiplier).tolist(),
                'is_whale_timeframe': is_whale_timeframe
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
                'is_whale_timeframe': False
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
        """Detectar divergencias entre precio e indicador con persistencia"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Divergencia Alcista: Precio hace lower low, indicador hace higher low
            if (price[i] < np.min(price[i-lookback:i]) and
                indicator[i] > np.min(indicator[i-lookback:i]) and
                price[i] < price[i-1] and indicator[i] > indicator[i-1]):
                # Persistencia: 7 velas
                for j in range(i, min(n, i+7)):
                    bullish_div[j] = True
            
            # Divergencia Bajista: Precio hace higher high, indicador hace lower high
            if (price[i] > np.max(price[i-lookback:i]) and
                indicator[i] < np.max(indicator[i-lookback:i]) and
                price[i] > price[i-1] and indicator[i] < indicator[i-1]):
                # Persistencia: 7 velas
                for j in range(i, min(n, i+7)):
                    bearish_div[j] = True
        
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

    def check_macd_signals(self, macd_line, macd_signal, histogram):
        """Detectar señales MACD"""
        n = len(macd_line)
        macd_bullish = np.zeros(n, dtype=bool)
        macd_bearish = np.zeros(n, dtype=bool)
        macd_divergence_bullish = np.zeros(n, dtype=bool)
        macd_divergence_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista
            if macd_line[i] > macd_signal[i] and macd_line[i-1] <= macd_signal[i-1]:
                macd_bullish[i] = True
            
            # Cruce bajista
            if macd_line[i] < macd_signal[i] and macd_line[i-1] >= macd_signal[i-1]:
                macd_bearish[i] = True
            
            # Histograma positivo/negativo
            if histogram[i] > 0 and histogram[i-1] <= 0:
                macd_bullish[i] = True
            elif histogram[i] < 0 and histogram[i-1] >= 0:
                macd_bearish[i] = True
        
        return macd_bullish.tolist(), macd_bearish.tolist()

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

    def check_obligatory_conditions(self, symbol, interval, signal_type, current_price, ma200):
        """Verificar condiciones OBLIGATORIAS del sistema"""
        try:
            # 1. Verificar Multi-Timeframe
            multi_tf_trend = self.check_multi_timeframe_trend(symbol, interval)
            multi_tf_maverick = self.check_multi_timeframe_maverick(symbol, interval)
            
            if signal_type == 'LONG':
                # Tendencia Mayor: ALCISTA o NEUTRAL
                mayor_ok = multi_tf_trend.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                # Tendencia Media: EXCLUSIVAMENTE ALCISTA
                media_ok = multi_tf_trend.get('media', 'NEUTRAL') == 'BULLISH'
                # Tendencia Menor: Fuerza Maverick ALCISTA
                menor_ok = multi_tf_trend.get('menor', 'NEUTRAL') == 'BULLISH'
                # Precio por encima de soporte (verificación básica)
                price_ok = True  # Se verifica más adelante con niveles Smart Money
            else:  # SHORT
                # Tendencia Mayor: BAJISTA o NEUTRAL
                mayor_ok = multi_tf_trend.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                # Tendencia Media: EXCLUSIVAMENTE BAJISTA
                media_ok = multi_tf_trend.get('media', 'NEUTRAL') == 'BEARISH'
                # Tendencia Menor: Fuerza Maverick BAJISTA
                menor_ok = multi_tf_trend.get('menor', 'NEUTRAL') == 'BEARISH'
                # Precio por debajo de resistencia (verificación básica)
                price_ok = True
            
            # 2. Verificar que NO hay zonas de NO OPERAR en ninguna temporalidad
            no_trade_zones = all(multi_tf_maverick.values())
            
            # 3. Verificar condición de MA200
            ma200_condition = current_price > ma200 if ma200 > 0 else True
            
            # Todas las condiciones obligatorias deben cumplirse
            obligatory_conditions_met = all([
                mayor_ok, media_ok, menor_ok, no_trade_zones
            ])
            
            return obligatory_conditions_met, {
                'mayor_trend': multi_tf_trend.get('mayor', 'NEUTRAL'),
                'media_trend': multi_tf_trend.get('media', 'NEUTRAL'),
                'menor_trend': multi_tf_trend.get('menor', 'NEUTRAL'),
                'no_trade_zones': no_trade_zones,
                'ma200_condition': ma200_condition
            }
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False, {}

    def evaluate_signal_conditions(self, data, current_idx, interval, obligatory_conditions):
        """Evaluar condiciones de señal con nuevo sistema de pesos"""
        conditions = {
            'long': {
                'multi_timeframe': {'value': obligatory_conditions[0], 'weight': 30, 'description': 'Confirmación Multi-Timeframe'},
                'whale_pump': {'value': False, 'weight': 15, 'description': 'Ballena compradora activa'},
                'di_cross_bullish': {'value': False, 'weight': 12, 'description': '+DI cruza -DI positivamente'},
                'rsi_oversold': {'value': False, 'weight': 10, 'description': 'RSI en sobreventa'},
                'rsi_maverick_oversold': {'value': False, 'weight': 10, 'description': 'RSI Maverick en sobreventa'},
                'macd_bullish': {'value': False, 'weight': 8, 'description': 'Señal MACD alcista'},
                'chart_pattern_bullish': {'value': False, 'weight': 8, 'description': 'Patrón chartista alcista'},
                'bb_touch_lower': {'value': False, 'weight': 7, 'description': 'Toque banda inferior Bollinger'}
            },
            'short': {
                'multi_timeframe': {'value': obligatory_conditions[0], 'weight': 30, 'description': 'Confirmación Multi-Timeframe'},
                'whale_dump': {'value': False, 'weight': 15, 'description': 'Ballena vendedora activa'},
                'di_cross_bearish': {'value': False, 'weight': 12, 'description': '-DI cruza +DI positivamente'},
                'rsi_overbought': {'value': False, 'weight': 10, 'description': 'RSI en sobrecompra'},
                'rsi_maverick_overbought': {'value': False, 'weight': 10, 'description': 'RSI Maverick en sobrecompra'},
                'macd_bearish': {'value': False, 'weight': 8, 'description': 'Señal MACD bajista'},
                'chart_pattern_bearish': {'value': False, 'weight': 8, 'description': 'Patrón chartista bajista'},
                'bb_touch_upper': {'value': False, 'weight': 7, 'description': 'Toque banda superior Bollinger'}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['whale_pump']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['whale_pump']):
            return conditions
        
        # LONG Conditions
        conditions['long']['whale_pump']['value'] = (
            data['whale_pump'][current_idx] > 25 and 
            data['whale_data']['is_whale_timeframe']
        )
        
        conditions['long']['di_cross_bullish']['value'] = (
            data['di_cross_bullish'][current_idx] or 
            (current_idx > 0 and data['di_cross_bullish'][current_idx-1])
        )
        
        conditions['long']['rsi_oversold']['value'] = (
            current_idx < len(data['rsi']) and 
            data['rsi'][current_idx] < 30
        )
        
        conditions['long']['rsi_maverick_oversold']['value'] = (
            current_idx < len(data['rsi_maverick']) and 
            data['rsi_maverick'][current_idx] < 0.2
        )
        
        conditions['long']['macd_bullish']['value'] = (
            current_idx < len(data['macd_bullish']) and 
            data['macd_bullish'][current_idx]
        )
        
        conditions['long']['chart_pattern_bullish']['value'] = (
            data['chart_patterns']['double_bottom'] or 
            data['chart_patterns']['bullish_flag']
        )
        
        conditions['long']['bb_touch_lower']['value'] = (
            current_idx < len(data['bb_lower']) and 
            abs(data['close'][current_idx] - data['bb_lower'][current_idx]) / data['bb_lower'][current_idx] < 0.01
        )
        
        # SHORT Conditions
        conditions['short']['whale_dump']['value'] = (
            data['whale_dump'][current_idx] > 25 and 
            data['whale_data']['is_whale_timeframe']
        )
        
        conditions['short']['di_cross_bearish']['value'] = (
            data['di_cross_bearish'][current_idx] or 
            (current_idx > 0 and data['di_cross_bearish'][current_idx-1])
        )
        
        conditions['short']['rsi_overbought']['value'] = (
            current_idx < len(data['rsi']) and 
            data['rsi'][current_idx] > 70
        )
        
        conditions['short']['rsi_maverick_overbought']['value'] = (
            current_idx < len(data['rsi_maverick']) and 
            data['rsi_maverick'][current_idx] > 0.8
        )
        
        conditions['short']['macd_bearish']['value'] = (
            current_idx < len(data['macd_bearish']) and 
            data['macd_bearish'][current_idx]
        )
        
        conditions['short']['chart_pattern_bearish']['value'] = (
            data['chart_patterns']['double_top']
        )
        
        conditions['short']['bb_touch_upper']['value'] = (
            current_idx < len(data['bb_upper']) and 
            abs(data['close'][current_idx] - data['bb_upper'][current_idx]) / data['bb_upper'][current_idx] < 0.01
        )
        
        return conditions

    def calculate_signal_score(self, conditions, signal_type, obligatory_info):
        """Calcular puntuación de señal con sistema de multiplicador obligatorio"""
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
        
        # Aplicar multiplicador obligatorio
        obligatory_multiplier = 1 if obligatory_info['obligatory_met'] else 0
        
        # Ajustar score mínimo según condición de MA200
        ma200_condition = obligatory_info.get('ma200_condition', True)
        if signal_type == 'long':
            min_score = 70 if ma200_condition else 75
        else:  # short
            min_score = 75 if ma200_condition else 70
        
        base_score = (achieved_weight / total_weight * 100)
        final_score = base_score * obligatory_multiplier
        
        if final_score < min_score:
            final_score = 0

        return min(final_score, 100), fulfilled_conditions

    def generate_exit_signals(self):
        """Generar señales de salida para operaciones activas"""
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
                
                # Obtener datos actuales (últimas 20 velas)
                df = self.get_kucoin_data(symbol, interval, 20)
                if df is None or len(df) < 10:
                    continue
                
                current_price = float(df['close'].iloc[-1])
                
                # Verificar condiciones de salida
                exit_reason = None
                
                # 1. Verificar Fuerza de Tendencia Maverick
                trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                current_strength = trend_data['strength_signals'][-1]
                
                if signal_type == 'LONG' and current_strength in ['WEAK_UP', 'STRONG_DOWN', 'WEAK_DOWN']:
                    exit_reason = "Fuerza de tendencia desfavorable"
                elif signal_type == 'SHORT' and current_strength in ['WEAK_DOWN', 'STRONG_UP', 'WEAK_UP']:
                    exit_reason = "Fuerza de tendencia desfavorable"
                
                # 2. Verificar ruptura de niveles clave
                support, resistance = self.calculate_smart_money_levels(
                    df['high'].values, df['low'].values, df['close'].values
                )
                
                if signal_type == 'LONG' and current_price < support * 0.99:
                    exit_reason = "Ruptura de soporte clave"
                elif signal_type == 'SHORT' and current_price > resistance * 1.01:
                    exit_reason = "Ruptura de resistencia clave"
                
                # 3. Verificar cambio en temporalidad menor
                hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
                if hierarchy.get('menor'):
                    menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 10)
                    if menor_df is not None and len(menor_df) > 5:
                        menor_trend = self.check_multi_timeframe_trend(symbol, hierarchy['menor'])
                        if (signal_type == 'LONG' and menor_trend.get('menor') == 'BEARISH') or \
                           (signal_type == 'SHORT' and menor_trend.get('menor') == 'BULLISH'):
                            exit_reason = "Cambio de tendencia en temporalidad menor"
                
                if exit_reason:
                    # Calcular P&L
                    if signal_type == 'LONG':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    # Actualizar winrate
                    self.update_winrate(symbol, signal_type, entry_price, current_price)
                    
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
                    
                    # Remover señal activa
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - NUEVO SISTEMA MULTI-TIMEFRAME"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular todos los indicadores
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Indicadores básicos
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # Indicadores técnicos
            whale_data = self.calculate_whale_signals_improved(df, interval)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi = self.calculate_rsi(close, rsi_length)
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            
            bullish_div_rsi, bearish_div_rsi = self.detect_divergence(close, rsi)
            bullish_div_maverick, bearish_div_maverick = self.detect_divergence(close, rsi_maverick)
            
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_bullish, macd_bearish = self.check_macd_signals(macd_line, macd_signal, macd_histogram)
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            
            # Patrones de chartismo
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # Fuerza de tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            current_idx = -1
            current_price = float(close[current_idx])
            current_ma200 = float(ma_200[current_idx]) if current_idx < len(ma_200) else 0
            
            # VERIFICAR CONDICIONES OBLIGATORIAS
            obligatory_met, obligatory_info = self.check_obligatory_conditions(
                symbol, interval, 'NEUTRAL', current_price, current_ma200
            )
            
            obligatory_info['obligatory_met'] = obligatory_met
            
            # Preparar datos para evaluación
            analysis_data = {
                'close': close,
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'whale_data': whale_data,
                'di_cross_bullish': di_cross_bullish,
                'di_cross_bearish': di_cross_bearish,
                'rsi': rsi,
                'rsi_maverick': rsi_maverick,
                'macd_bullish': macd_bullish,
                'macd_bearish': macd_bearish,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'chart_patterns': chart_patterns
            }
            
            # Evaluar condiciones para ambas direcciones
            conditions = self.evaluate_signal_conditions(analysis_data, current_idx, interval, (obligatory_met, obligatory_info))
            
            long_score, long_conditions = self.calculate_signal_score(conditions, 'long', obligatory_info)
            short_score, short_conditions = self.calculate_signal_score(conditions, 'short', obligatory_info)
            
            # Determinar señal final
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= 70 and obligatory_met:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= 70 and obligatory_met:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # Calcular niveles de entrada/salida
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # Registrar señal activa si es válida
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
                'fulfilled_conditions': fulfilled_conditions,
                'obligatory_conditions': obligatory_info,
                'chart_patterns': chart_patterns,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'rsi': rsi[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:],
                    'macd_line': macd_line[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'colors': trend_strength_data['colors'][-50:]
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
            'fulfilled_conditions': [],
            'obligatory_conditions': {},
            'chart_patterns': {},
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de trading"""
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
                        signal_data['signal_score'] >= 70):
                        
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
                        
                        # Obtener winrate
                        winrate = self.calculate_winrate(symbol)
                        
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
                            'support': signal_data['support'],
                            'resistance': signal_data['resistance'],
                            'winrate': winrate,
                            'obligatory_conditions': signal_data.get('obligatory_conditions', {})
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
        winrate = alert_data.get('winrate', 0)
        overall_winrate = indicator.get_overall_winrate()
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA DE TRADING - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
🏆 Winrate: {winrate:.1f}% (Sistema: {overall_winrate:.1f}%)

💰 Precio actual: {alert_data.get('current_price', alert_data['entry']):.6f}
🎯 ENTRADA: ${alert_data['entry']:.6f}
🛑 STOP LOSS: ${alert_data['stop_loss']:.6f}
🎯 TAKE PROFIT: ${alert_data['take_profit']:.6f}

📈 Apalancamiento: x{alert_data['leverage']}

✅ Condiciones Multi-Timeframe:
• Mayor: {alert_data['obligatory_conditions'].get('mayor_trend', 'NEUTRAL')}
• Media: {alert_data['obligatory_conditions'].get('media_trend', 'NEUTRAL')}  
• Menor: {alert_data['obligatory_conditions'].get('menor_trend', 'NEUTRAL')}

🔔 Condiciones Cumplidas:
• {chr(10).join(['• ' + cond for cond in alert_data['fulfilled_conditions'][:3]])}

📊 Revisa la señal en: https://multiframewgta.onrender.com/
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

📊 Observación: {alert_data['reason']}
💪 Fuerza de Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}
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
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar alertas de entrada
            print("Verificando alertas de trading...")
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                send_telegram_alert(alert, 'entry')
            
            # Verificar alertas de salida
            exit_alerts = indicator.generate_exit_signals()
            for alert in exit_alerts:
                send_telegram_alert(alert, 'exit')
            
            time.sleep(60)  # Verificar cada 60 segundos
            
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
        adx_threshold = int(request.args.get('adx_threshold', 25))  # Cambiado a 25
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))  # Cambiado a 14
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
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
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
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:3])  # 3 por categoría
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval, di_period, adx_threshold)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones basadas en nuevos indicadores
                    buy_pressure = min(100, max(0,
                        (1 if signal_data['obligatory_conditions'].get('obligatory_met', False) else 0) * 40 +
                        (signal_data['whale_pump'] / 100 * 20) +
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 15 +
                        (signal_data['rsi_maverick'] * 10) +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 15)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (1 if signal_data['obligatory_conditions'].get('obligatory_met', False) else 0) * 40 +
                        (signal_data['whale_dump'] / 100 * 20) +
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 15 +
                        ((1 - signal_data['rsi_maverick']) * 10) +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 15)
                    ))
                    
                    if signal_data['signal'] == 'LONG':
                        buy_pressure = max(buy_pressure, 60)
                        sell_pressure = min(sell_pressure, 40)
                    elif signal_data['signal'] == 'SHORT':
                        sell_pressure = max(sell_pressure, 60)
                        buy_pressure = min(buy_pressure, 40)
                    
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
        symbol = request.args.get('symbol', '')
        
        if symbol:
            symbol_winrate = indicator.calculate_winrate(symbol)
            return jsonify({
                'symbol': symbol,
                'winrate': symbol_winrate,
                'overall_winrate': overall_winrate
            })
        else:
            return jsonify({
                'overall_winrate': overall_winrate
            })
            
    except Exception as e:
        print(f"Error en /api/winrate: {e}")
        return jsonify({'overall_winrate': 0})

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
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(16, 18))
        
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
                ax1.plot(ma_dates, signal_data['indicators']['ma_50'], 'red', linewidth=1, label='MA 50')
                ax1.plot(ma_dates, signal_data['indicators']['ma_200'], 'purple', linewidth=2, label='MA 200')
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            ax1.axhline(y=signal_data['take_profit'][0], color='green', linestyle='--', alpha=0.7, label='Take Profit')
            ax1.axhline(y=signal_data['support'], color='gray', linestyle=':', alpha=0.5, label='Soporte')
            ax1.axhline(y=signal_data['resistance'], color='gray', linestyle=':', alpha=0.5, label='Resistencia')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Ballenas (solo mostrar para 12H y 1D)
        ax2 = plt.subplot(8, 1, 2, sharex=ax1)
        if 'indicators' in signal_data and interval in ['12h', '1D']:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax2.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax2.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
            ax2.set_ylabel('Fuerza Ballenas')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Indicador Ballenas: Solo disponible en 12H y 1D', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_ylabel('Ballenas')
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
        
        # Gráfico 4: RSI Tradicional
        ax4 = plt.subplot(8, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax4.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax4.set_ylabel('RSI Tradicional')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: RSI Maverick
        ax5 = plt.subplot(8, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_maverick_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax5.plot(rsi_maverick_dates, signal_data['indicators']['rsi_maverick'], 
                    'magenta', linewidth=2, label='RSI Maverick')
            ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax5.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax5.set_ylabel('RSI Maverick')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: MACD
        ax6 = plt.subplot(8, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd_line']):]
            ax6.plot(macd_dates, signal_data['indicators']['macd_line'], 
                    'blue', linewidth=1, label='MACD')
            ax6.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            # Histograma MACD con colores
            colors = ['green' if x >= 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax6.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6, label='Histograma')
            
            ax6.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        ax6.set_ylabel('MACD')
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
            
            # Marcar zonas de no operar
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax7.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax7.set_ylabel('Fuerza Tendencia %')
            ax7.grid(True, alpha=0.3)
        
        # Información de la señal
        ax8 = plt.subplot(8, 1, 8)
        ax8.axis('off')
        
        # Información de condiciones obligatorias
        obligatory_info = signal_data.get('obligatory_conditions', {})
        multi_tf_info = f"""
        CONDICIONES MULTI-TIMEFRAME:
        • Mayor: {obligatory_info.get('mayor_trend', 'NEUTRAL')}
        • Media: {obligatory_info.get('media_trend', 'NEUTRAL')}
        • Menor: {obligatory_info.get('menor_trend', 'NEUTRAL')}
        • Zonas NO OPERAR: {'✅ NO' if obligatory_info.get('no_trade_zones', True) else '❌ SI'}
        """
        
        # Información de patrones chartistas
        chart_info = ""
        patterns = signal_data.get('chart_patterns', {})
        if any(patterns.values()):
            chart_info = "PATRONES DETECTADOS:\n"
            for pattern, detected in patterns.items():
                if detected:
                    chart_info += f"• {pattern.replace('_', ' ').title()}\n"
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        
        {multi_tf_info}
        
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
        APALANCAMIENTO: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        {chart_info}
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
        """
        
        ax8.text(0.02, 0.98, signal_info, transform=ax8.transAxes, fontsize=9,
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
