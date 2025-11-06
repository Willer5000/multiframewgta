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
from collections import deque
import sqlite3
import traceback

app = Flask(__name__)

# Nueva configuración Telegram
TELEGRAM_BOT_TOKEN = "8007748376:AAHIW8n9b-BtA378g4gF-0-D2mOhn495Q0g"
TELEGRAM_CHAT_ID = "-1003229814161"

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
    "DOGE-USDT", "SHIB-USDT", "FLOKI-USDT", "PEPE2-USDT", "BONK-USDT"
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
        "DOGE-USDT", "SHIB-USDT", "FLOKI-USDT", "PEPE2-USDT", "BONK-USDT"
    ]
}

# JERARQUÍA TEMPORAL COMPLETA Y OBLIGATORIA
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

# UMBRALES DE SCORE MEJORADOS
SCORE_THRESHOLDS = {
    'LONG': 75,
    'SHORT': 80
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_signals = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        
        # SISTEMA DE TRACKING MEJORADO
        self.operation_history = deque(maxlen=1000)
        self.winrate_data = {
            'total_operations': 0,
            'successful_operations': 0,
            'winrate': 0.0,
            'strategy_performance': {}
        }
        
        # Inicializar base de datos en memoria para tracking
        self.init_tracking_db()

    def init_tracking_db(self):
        """Inicializar base de datos para tracking de operaciones"""
        try:
            self.conn = sqlite3.connect(':memory:', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    signal_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    status TEXT,
                    timestamp DATETIME,
                    score REAL,
                    conditions TEXT
                )
            ''')
            self.conn.commit()
        except Exception as e:
            print(f"Error inicializando base de datos: {e}")

    def track_operation(self, symbol, timeframe, signal_type, entry_price, score, conditions):
        """Registrar nueva operación en el sistema de tracking"""
        try:
            self.cursor.execute('''
                INSERT INTO operations (symbol, timeframe, signal_type, entry_price, score, conditions, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timeframe, signal_type, entry_price, score, json.dumps(conditions), datetime.now(), 'active'))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            print(f"Error registrando operación: {e}")
            return None

    def update_operation_outcome(self, operation_id, exit_price, pnl, status):
        """Actualizar resultado de operación"""
        try:
            self.cursor.execute('''
                UPDATE operations SET exit_price = ?, pnl = ?, status = ?
                WHERE id = ?
            ''', (exit_price, pnl, status, operation_id))
            self.conn.commit()
            
            # Actualizar winrate global
            self.calculate_winrate()
            
        except Exception as e:
            print(f"Error actualizando operación: {e}")

    def calculate_winrate(self):
        """Calcular winrate global del sistema"""
        try:
            self.cursor.execute('''
                SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) 
                FROM operations WHERE status = 'closed'
            ''')
            result = self.cursor.fetchone()
            
            total_operations = result[0] or 0
            successful_operations = result[1] or 0
            
            if total_operations > 0:
                winrate = (successful_operations / total_operations) * 100
            else:
                winrate = 0.0
            
            self.winrate_data = {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'winrate': winrate
            }
            
            return winrate
            
        except Exception as e:
            print(f"Error calculando winrate: {e}")
            return 0.0

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

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick basado en ancho de Bandas de Bollinger"""
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

    def calculate_squeeze_momentum(self, high, low, close, length=20, mult=2.0, length_kc=20, mult_kc=1.5):
        """Calcular Squeeze Momentum (LazyBear)"""
        try:
            n = len(close)
            
            # Bollinger Bands
            basis = self.calculate_sma(close, length)
            dev = np.zeros(n)
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                dev[i] = np.std(window) if len(window) > 1 else 0
            upper_bb = basis + (dev * mult)
            lower_bb = basis - (dev * mult)
            
            # Keltner Channel
            ma = self.calculate_sma(close, length_kc)
            range_ma = self.calculate_sma(high - low, length_kc)
            upper_kc = ma + range_ma * mult_kc
            lower_kc = ma - range_ma * mult_kc
            
            # Squeeze conditions
            squeeze_on = np.zeros(n, dtype=bool)
            squeeze_off = np.zeros(n, dtype=bool)
            
            for i in range(n):
                if i >= length_kc:
                    if (lower_bb[i] > lower_kc[i]) and (upper_bb[i] < upper_kc[i]):
                        squeeze_on[i] = True
                    else:
                        squeeze_off[i] = True
            
            # Momentum (simplificado)
            momentum = np.zeros(n)
            for i in range(1, n):
                if close[i] > close[i-1]:
                    momentum[i] = 1
                elif close[i] < close[i-1]:
                    momentum[i] = -1
                else:
                    momentum[i] = 0
            
            return {
                'squeeze_on': squeeze_on.tolist(),
                'squeeze_off': squeeze_off.tolist(),
                'squeeze_momentum': momentum.tolist(),
                'upper_bb': upper_bb.tolist(),
                'lower_bb': lower_bb.tolist(),
                'upper_kc': upper_kc.tolist(),
                'lower_kc': lower_kc.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_squeeze_momentum: {e}")
            n = len(close)
            return {
                'squeeze_on': [False] * n,
                'squeeze_off': [False] * n,
                'squeeze_momentum': [0] * n,
                'upper_bb': [0] * n,
                'lower_bb': [0] * n,
                'upper_kc': [0] * n,
                'lower_kc': [0] * n
            }

    def check_multi_timeframe_trend(self, symbol, timeframe):
        """Verificar tendencia en múltiples temporalidades - OBLIGATORIO"""
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
                
                # Calcular fuerza de tendencia Maverick
                trend_data = self.calculate_trend_strength_maverick(close)
                current_signal = trend_data['strength_signals'][-1] if len(trend_data['strength_signals']) > 0 else 'NEUTRAL'
                current_no_trade = trend_data['no_trade_zones'][-1] if len(trend_data['no_trade_zones']) > 0 else False
                
                # Determinar tendencia basada en fuerza Maverick
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

    def check_mandatory_conditions(self, symbol, interval, signal_type):
        """Verificar condiciones obligatorias - SI NO SE CUMPLEN, SCORE = 0"""
        try:
            # Obtener análisis multi-timeframe
            multi_tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            if signal_type == 'LONG':
                # OBLIGATORIO PARA LONG: Mayor ALCISTA/NEUTRAL, Media EXCLUSIVAMENTE ALCISTA, Menor ALCISTA
                mayor_ok = multi_tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                media_ok = multi_tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'  # EXCLUSIVAMENTE ALCISTA
                menor_ok = multi_tf_analysis.get('menor', 'NEUTRAL') == 'BULLISH'  # EXCLUSIVAMENTE ALCISTA
                
                # Verificar que NO HAYA zonas de NO OPERAR en ninguna temporalidad
                no_trade_zones = any(value == 'NO_TRADE' for value in multi_tf_analysis.values())
                
                return mayor_ok and media_ok and menor_ok and not no_trade_zones
                
            elif signal_type == 'SHORT':
                # OBLIGATORIO PARA SHORT: Mayor BAJISTA/NEUTRAL, Media EXCLUSIVAMENTE BAJISTA, Menor BAJISTA
                mayor_ok = multi_tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                media_ok = multi_tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'  # EXCLUSIVAMENTE BAJISTA
                menor_ok = multi_tf_analysis.get('menor', 'NEUTRAL') == 'BEARISH'  # EXCLUSIVAMENTE BAJISTA
                
                # Verificar que NO HAYA zonas de NO OPERAR en ninguna temporalidad
                no_trade_zones = any(value == 'NO_TRADE' for value in multi_tf_analysis.values())
                
                return mayor_ok and media_ok and menor_ok and not no_trade_zones
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias: {e}")
            return False

    def calculate_whale_signals_corrected(self, df, interval):
        """Implementación CORRECTA del Cazador de Ballenas - SOLO para 12H y 1D"""
        try:
            # SOLO aplicar en 12H y 1D
            if interval not in ['12h', '1D']:
                n = len(df)
                return {
                    'whale_pump': [0] * n,
                    'whale_dump': [0] * n,
                    'whale_active': [False] * n
                }
            
            close = df['close'].values
            low = df['low'].values
            high = df['high'].values
            volume = df['volume'].values
            
            n = len(close)
            
            whale_pump_signal = np.zeros(n)
            whale_dump_signal = np.zeros(n)
            whale_active = np.zeros(n, dtype=bool)
            
            # Lógica específica para Ballenas en 12H/1D
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                # Detectar acumulación (pump) en zonas de soporte
                if (volume_ratio > 2.0 and  # Volumen muy alto
                    close[i] <= np.min(low[max(0, i-10):i+1]) * 1.02):  # Precio cerca de mínimos
                    whale_pump_signal[i] = min(100, volume_ratio * 25)
                    whale_active[i] = True
                
                # Detectar distribución (dump) en zonas de resistencia
                if (volume_ratio > 2.0 and  # Volumen muy alto
                    close[i] >= np.max(high[max(0, i-10):i+1]) * 0.98):  # Precio cerca de máximos
                    whale_dump_signal[i] = min(100, volume_ratio * 25)
                    whale_active[i] = True
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'whale_active': whale_active.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_corrected: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'whale_active': [False] * n
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
            price_low = price[i] < np.min(price[i-lookback:i])
            indicator_high = indicator[i] > np.max(indicator[i-lookback:i])
            
            if price_low and indicator_high:
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace higher high, indicador hace lower high
            price_high = price[i] > np.max(price[i-lookback:i])
            indicator_low = indicator[i] < np.min(indicator[i-lookback:i])
            
            if price_high and indicator_low:
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def calculate_support_resistance(self, high, low, close, period=20):
        """Calcular soportes y resistencias dinámicos"""
        n = len(close)
        support = np.zeros(n)
        resistance = np.zeros(n)
        
        for i in range(period, n):
            support[i] = np.min(low[i-period:i+1])
            resistance[i] = np.max(high[i-period:i+1])
        
        return support.tolist(), resistance.tolist()

    def calculate_signal_score_professional(self, conditions, signal_type, interval):
        """Calcular score profesional con ponderaciones mejoradas"""
        try:
            # PONDERACIONES MEJORADAS
            weights = {
                'mandatory': 0.60,  # 60% condiciones obligatorias
                'complementary': 0.40,  # 40% indicadores complementarios
                'whale_bonus': 0.15   # 15% bonus ballenas (solo 12H, 1D)
            }
            
            # Verificar condiciones obligatorias primero
            mandatory_score = 1.0 if conditions['mandatory_conditions'] else 0.0
            
            if mandatory_score == 0:
                return 0.0, []
            
            # Calcular score complementario
            complementary_indicators = [
                ('rsi_maverick_divergence', 0.10),
                ('rsi_traditional_divergence', 0.10),
                ('adx_dmi', 0.05),
                ('macd', 0.05),
                ('squeeze_momentum', 0.05),
                ('moving_averages', 0.05)
            ]
            
            complementary_score = 0.0
            fulfilled_conditions = []
            
            for indicator, weight in complementary_indicators:
                if conditions.get(indicator, False):
                    complementary_score += weight
                    fulfilled_conditions.append(indicator.replace('_', ' ').title())
            
            # Bonus ballenas (solo para 12H y 1D)
            whale_bonus = 0.0
            if interval in ['12h', '1D'] and conditions.get('whale_confirmation', False):
                whale_bonus = weights['whale_bonus']
                fulfilled_conditions.append("Whale Confirmation")
            
            # Score final
            final_score = (
                mandatory_score * weights['mandatory'] +
                complementary_score * weights['complementary'] +
                whale_bonus
            ) * 100
            
            # Aplicar umbral mínimo
            min_threshold = SCORE_THRESHOLDS.get(signal_type, 75)
            if final_score < min_threshold:
                return 0.0, []
            
            return min(final_score, 100), fulfilled_conditions
            
        except Exception as e:
            print(f"Error calculando score profesional: {e}")
            return 0.0, []

    def generate_signals_professional(self, symbol, interval, di_period=14, adx_threshold=20, 
                                   sr_period=50, rsi_length=20, bb_multiplier=2.0, 
                                   volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES PROFESIONAL - CON OBLIGATORIEDADES MULTI-TIMEFRAME"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # OBTENER DATOS DE INDICADORES
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            current_idx = -1
            current_price = float(close[current_idx])
            
            # INDICADORES OBLIGATORIOS
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            support, resistance = self.calculate_support_resistance(high, low, close, sr_period)
            
            # INDICADORES COMPLEMENTARIOS
            rsi_maverick = self.calculate_rsi_maverick(close, rsi_length, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, 14)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            squeeze_data = self.calculate_squeeze_momentum(high, low, close)
            
            # DETECCIÓN DE DIVERGENCIAS
            rsi_maverick_bullish_div, rsi_maverick_bearish_div = self.detect_divergence(close, rsi_maverick)
            rsi_traditional_bullish_div, rsi_traditional_bearish_div = self.detect_divergence(close, rsi_traditional)
            
            # INDICADOR BALLENAS (SOLO 12H, 1D)
            whale_data = self.calculate_whale_signals_corrected(df, interval)
            
            # EVALUAR CONDICIONES PARA CADA TIPO DE SEÑAL
            long_conditions = {
                'mandatory_conditions': self.check_mandatory_conditions(symbol, interval, 'LONG'),
                'rsi_maverick_divergence': rsi_maverick_bullish_div[current_idx] if current_idx < len(rsi_maverick_bullish_div) else False,
                'rsi_traditional_divergence': rsi_traditional_bullish_div[current_idx] if current_idx < len(rsi_traditional_bullish_div) else False,
                'adx_dmi': (adx[current_idx] > adx_threshold if current_idx < len(adx) else False) and 
                          (plus_di[current_idx] > minus_di[current_idx] if current_idx < len(plus_di) else False),
                'macd': macd[current_idx] > macd_signal[current_idx] if current_idx < len(macd) else False,
                'squeeze_momentum': squeeze_data['squeeze_momentum'][current_idx] > 0 if current_idx < len(squeeze_data['squeeze_momentum']) else False,
                'moving_averages': (current_price > self.calculate_sma(close, 50)[current_idx] if current_idx < len(close) else False) and
                                 (self.calculate_sma(close, 50)[current_idx] > self.calculate_sma(close, 200)[current_idx] if current_idx < len(close) else False),
                'whale_confirmation': whale_data['whale_active'][current_idx] if current_idx < len(whale_data['whale_active']) else False,
                'price_above_support': current_price > support[current_idx] if current_idx < len(support) else False
            }
            
            short_conditions = {
                'mandatory_conditions': self.check_mandatory_conditions(symbol, interval, 'SHORT'),
                'rsi_maverick_divergence': rsi_maverick_bearish_div[current_idx] if current_idx < len(rsi_maverick_bearish_div) else False,
                'rsi_traditional_divergence': rsi_traditional_bearish_div[current_idx] if current_idx < len(rsi_traditional_bearish_div) else False,
                'adx_dmi': (adx[current_idx] > adx_threshold if current_idx < len(adx) else False) and 
                          (minus_di[current_idx] > plus_di[current_idx] if current_idx < len(minus_di) else False),
                'macd': macd[current_idx] < macd_signal[current_idx] if current_idx < len(macd) else False,
                'squeeze_momentum': squeeze_data['squeeze_momentum'][current_idx] < 0 if current_idx < len(squeeze_data['squeeze_momentum']) else False,
                'moving_averages': (current_price < self.calculate_sma(close, 50)[current_idx] if current_idx < len(close) else False) and
                                 (self.calculate_sma(close, 50)[current_idx] < self.calculate_sma(close, 200)[current_idx] if current_idx < len(close) else False),
                'whale_confirmation': whale_data['whale_active'][current_idx] if current_idx < len(whale_data['whale_active']) else False,
                'price_below_resistance': current_price < resistance[current_idx] if current_idx < len(resistance) else False
            }
            
            # CALCULAR SCORES
            long_score, long_fulfilled = self.calculate_signal_score_professional(long_conditions, 'LONG', interval)
            short_score, short_fulfilled = self.calculate_signal_score_professional(short_conditions, 'SHORT', interval)
            
            # DETERMINAR SEÑAL FINAL
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= SCORE_THRESHOLDS['LONG']:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_fulfilled
            elif short_score >= SCORE_THRESHOLDS['SHORT']:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_fulfilled
            
            # CALCULAR NIVELES DE ENTRADA/SALIDA
            entry_exit_data = self.calculate_optimal_entry_exit(df, signal_type, leverage, support, resistance)
            
            # REGISTRAR OPERACIÓN SI ES VÁLIDA
            if signal_type in ['LONG', 'SHORT']:
                operation_id = self.track_operation(
                    symbol, interval, signal_type, 
                    entry_exit_data['entry'], signal_score, 
                    fulfilled_conditions
                )
            
            # PREPARAR DATOS PARA VISUALIZACIÓN
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Obtener análisis multi-timeframe para mostrar
            multi_tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': entry_exit_data['entry'],
                'stop_loss': entry_exit_data['stop_loss'],
                'take_profit': entry_exit_data['take_profit'],
                'support': entry_exit_data['support'],
                'resistance': entry_exit_data['resistance'],
                'atr': entry_exit_data['atr'],
                'atr_percentage': entry_exit_data['atr_percentage'],
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
                'adx': float(adx[current_idx] if current_idx < len(adx) else 0),
                'plus_di': float(plus_di[current_idx] if current_idx < len(plus_di) else 0),
                'minus_di': float(minus_di[current_idx] if current_idx < len(minus_di) else 0),
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'rsi_traditional': float(rsi_traditional[current_idx] if current_idx < len(rsi_traditional) else 50),
                'fulfilled_conditions': fulfilled_conditions,
                'multi_timeframe_analysis': multi_tf_analysis,
                'mandatory_conditions_met': long_conditions['mandatory_conditions'] if signal_type == 'LONG' else short_conditions['mandatory_conditions'] if signal_type == 'SHORT' else False,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'colors': trend_strength_data['colors'][-50:],
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'macd': macd[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['squeeze_momentum'][-50:],
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'support': support[-50:],
                    'resistance': resistance[-50:]
                }
            }
            
        except Exception as e:
            print(f"Error en generate_signals_professional para {symbol}: {e}")
            traceback.print_exc()
            return self._create_empty_signal(symbol)

    def calculate_optimal_entry_exit(self, df, signal_type, leverage, support, resistance):
        """Calcular entradas y salidas óptimas"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            
            # Calcular ATR para stops
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            atr_percentage = current_atr / current_price
            
            current_support = support[-1] if len(support) > 0 else np.min(low[-20:])
            current_resistance = resistance[-1] if len(resistance) > 0 else np.max(high[-20:])
            
            if signal_type == 'LONG':
                entry = min(current_price, current_support * 1.01)
                stop_loss = max(current_support * 0.98, entry - (current_atr * 1.5))
                take_profit = [current_resistance * 0.99]
                
                # Asegurar relación riesgo/beneficio mínima 1:2
                min_tp = entry + (2 * (entry - stop_loss))
                take_profit[0] = max(take_profit[0], min_tp)
                
            elif signal_type == 'SHORT':
                entry = max(current_price, current_resistance * 0.99)
                stop_loss = min(current_resistance * 1.02, entry + (current_atr * 1.5))
                take_profit = [current_support * 1.01]
                
                # Asegurar relación riesgo/beneficio mínima 1:2
                min_tp = entry - (2 * (stop_loss - entry))
                take_profit[0] = min(take_profit[0], min_tp)
                
            else:
                entry = current_price
                stop_loss = current_price * 0.98
                take_profit = [current_price * 1.02]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profit],
                'support': float(current_support),
                'resistance': float(current_resistance),
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas: {e}")
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
            'rsi_maverick': 0.5,
            'rsi_traditional': 50,
            'fulfilled_conditions': [],
            'multi_timeframe_analysis': {'mayor': 'NEUTRAL', 'media': 'NEUTRAL', 'menor': 'NEUTRAL'},
            'mandatory_conditions_met': False,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de scalping con nuevo sistema"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D']:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:15]:
                try:
                    signal_data = self.generate_signals_professional(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= SCORE_THRESHOLDS[signal_data['signal']] and
                        signal_data['mandatory_conditions_met']):
                        
                        risk_category = next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                        
                        # Calcular leverage óptimo basado en score y riesgo
                        base_leverage = min(20, 5 + (signal_data['signal_score'] - 70) // 3)
                        risk_factors = {'bajo': 1.0, 'medio': 0.8, 'alto': 0.6, 'memecoins': 0.5}
                        risk_factor = risk_factors.get(risk_category, 0.7)
                        optimal_leverage = max(5, int(base_leverage * risk_factor))
                        
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
                            'multi_timeframe_analysis': signal_data.get('multi_timeframe_analysis', {}),
                            'winrate': self.winrate_data['winrate']
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

    def generate_exit_signals(self):
        """Generar señales de salida para operaciones activas"""
        exit_alerts = []
        current_time = self.get_bolivia_time()
        
        try:
            # Obtener operaciones activas de la base de datos
            self.cursor.execute('''
                SELECT id, symbol, timeframe, signal_type, entry_price, score 
                FROM operations WHERE status = 'active'
            ''')
            active_operations = self.cursor.fetchall()
            
            for op in active_operations:
                op_id, symbol, timeframe, signal_type, entry_price, score = op
                
                try:
                    # Obtener datos actuales
                    signal_data = self.generate_signals_professional(symbol, timeframe)
                    current_price = signal_data['current_price']
                    
                    # Verificar condiciones de salida
                    exit_reason = None
                    
                    # 1. Salida por cambio en condiciones obligatorias
                    if not signal_data['mandatory_conditions_met']:
                        exit_reason = "Pérdida de condiciones obligatorias multi-timeframe"
                    
                    # 2. Salida por cambio de tendencia
                    elif (signal_type == 'LONG' and 
                          signal_data['multi_timeframe_analysis'].get('menor') == 'BEARISH'):
                        exit_reason = "Cambio a tendencia bajista en TF menor"
                    
                    elif (signal_type == 'SHORT' and 
                          signal_data['multi_timeframe_analysis'].get('menor') == 'BULLISH'):
                        exit_reason = "Cambio a tendencia alcista en TF menor"
                    
                    # 3. Salida por zona no operar
                    elif any('NO_TRADE' in str(value) for value in signal_data['multi_timeframe_analysis'].values()):
                        exit_reason = "Activación de zona NO OPERAR"
                    
                    if exit_reason:
                        # Calcular P&L
                        if signal_type == 'LONG':
                            pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        else:
                            pnl_percent = ((entry_price - current_price) / entry_price) * 100
                        
                        # Actualizar operación en base de datos
                        status = 'profit' if pnl_percent > 0 else 'loss'
                        self.update_operation_outcome(op_id, current_price, pnl_percent, status)
                        
                        exit_alert = {
                            'symbol': symbol,
                            'interval': timeframe,
                            'signal': signal_type,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl_percent': pnl_percent,
                            'reason': exit_reason,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        exit_alerts.append(exit_alert)
                        
                except Exception as e:
                    print(f"Error procesando salida para {symbol}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error generando señales de salida: {e}")
        
        return exit_alerts

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
    """Enviar alerta por Telegram con nuevo formato"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        winrate = alert_data.get('winrate', indicator.winrate_data['winrate'])
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA DE TRADING - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
🏆 Winrate Sistema: {winrate:.1f}%

💰 Precio actual: {alert_data['current_price']:.6f}
🎯 ENTRADA: ${alert_data['entry']:.6f}
🛑 STOP LOSS: ${alert_data['stop_loss']:.6f}
🎯 TAKE PROFIT: ${alert_data['take_profit']:.6f}

📈 Apalancamiento: x{alert_data['leverage']}

✅ Confirmación Multi-Timeframe:
   • Mayor: {alert_data['multi_timeframe_analysis'].get('mayor', 'N/A')}
   • Media: {alert_data['multi_timeframe_analysis'].get('media', 'N/A')} 
   • Menor: {alert_data['multi_timeframe_analysis'].get('menor', 'N/A')}

🔔 Condiciones Cumplidas:
• {chr(10) + '• '.join(alert_data['fulfilled_conditions'][:3]) if alert_data['fulfilled_conditions'] else 'Condiciones base cumplidas'}

📊 Sistema profesional con confirmación multi-temporalidad
⚡ Operar solo si todas las condiciones obligatorias se mantienen
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

📊 Razón: {alert_data['reason']}

🏆 Winrate Sistema: {winrate:.1f}%

🔔 Próximo paso: Esperar nueva señal con condiciones óptimas
            """
        
        # Generar URL para el reporte
        report_url = f"https://ballenasscalpistas.onrender.com/api/generate_report?symbol={alert_data['symbol']}&interval={alert_data['interval']}&leverage={alert_data.get('leverage', 15)}"
        
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
            # Generar alertas de entrada
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                send_telegram_alert(alert, 'entry')
            
            # Generar alertas de salida
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

# RUTAS DE LA APLICACIÓN
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint para obtener señales de trading PROFESIONAL"""
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
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
            try:
                signal_data = indicator.generate_signals_professional(
                    symbol, interval, di_period, adx_threshold, sr_period,
                    rsi_length, bb_multiplier, volume_filter, leverage
                )
                
                if signal_data and signal_data['signal'] != 'NEUTRAL':
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
            'long_signals': long_signals[:5],  # Top 5
            'short_signals': short_signals[:5],  # Top 5
            'total_signals': len(all_signals)
        })
        
    except Exception as e:
        print(f"Error en /api/multiple_signals: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/scatter_data_improved')
def get_scatter_data_improved():
    """Endpoint para datos del scatter plot"""
    try:
        interval = request.args.get('interval', '4h')
        
        scatter_data = []
        
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:3])
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_professional(symbol, interval)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones de compra/venta basadas en indicadores
                    buy_pressure = min(100, max(0,
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 40 +
                        (signal_data['rsi_maverick'] * 30) +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 30)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 40 +
                        ((1 - signal_data['rsi_maverick']) * 30) +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 30)
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
        print(f"Error en /api/scatter_data_improved: {e}")
        return jsonify([])

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    """Endpoint para obtener la clasificación de riesgo"""
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

@app.route('/api/winrate_data')
def get_winrate_data():
    """Endpoint para obtener datos de winrate"""
    try:
        winrate = indicator.calculate_winrate()
        return jsonify(indicator.winrate_data)
        
    except Exception as e:
        print(f"Error en /api/winrate_data: {e}")
        return jsonify({'winrate': 0.0, 'total_operations': 0, 'successful_operations': 0})

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
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(12, 16))
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(7, 1, 1)
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
        
        ax1.set_title(f'MULTI-TIMEFRAME CRYPTO WGTA PRO - {symbol} ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Fuerza de Tendencia Maverick
        ax2 = plt.subplot(7, 1, 2, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax2.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            ax2.set_ylabel('Fuerza Tendencia %')
            ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: RSI Maverick
        ax3 = plt.subplot(7, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax3.plot(rsi_dates, signal_data['indicators']['rsi_maverick'], 'blue', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax3.set_ylabel('RSI Maverick')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: ADX/DMI
        ax4 = plt.subplot(7, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax4.plot(adx_dates, signal_data['indicators']['adx'], 'white', linewidth=2, label='ADX')
            ax4.plot(adx_dates, signal_data['indicators']['plus_di'], 'green', linewidth=1, label='+DI')
            ax4.plot(adx_dates, signal_data['indicators']['minus_di'], 'red', linewidth=1, label='-DI')
        ax4.set_ylabel('ADX/DMI')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: MACD
        ax5 = plt.subplot(7, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax5.plot(macd_dates, signal_data['indicators']['macd'], 'blue', linewidth=1, label='MACD')
            ax5.plot(macd_dates, signal_data['indicators']['macd_signal'], 'red', linewidth=1, label='Señal')
            ax5.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']], 
                   alpha=0.6, label='Histograma')
        ax5.set_ylabel('MACD')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Squeeze Momentum
        ax6 = plt.subplot(7, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            squeeze_dates = dates[-len(signal_data['indicators']['squeeze_momentum']):]
            momentum = signal_data['indicators']['squeeze_momentum']
            colors = ['green' if x > 0 else 'red' for x in momentum]
            ax6.bar(squeeze_dates, momentum, color=colors, alpha=0.7, label='Squeeze Momentum')
            ax6.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        ax6.set_ylabel('Squeeze')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Información de la señal
        ax7 = plt.subplot(7, 1, 7)
        ax7.axis('off')
        
        multi_tf = signal_data.get('multi_timeframe_analysis', {})
        winrate = indicator.winrate_data['winrate']
        
        signal_info = f"""
        MULTI-TIMEFRAME CRYPTO WGTA PRO - REPORTE TÉCNICO
        
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        WINRATE SISTEMA: {winrate:.1f}%
        
        CONDICIONES OBLIGATORIAS: {'✅ CUMPLIDAS' if signal_data['mandatory_conditions_met'] else '❌ NO CUMPLIDAS'}
        
        ANÁLISIS MULTI-TIMEFRAME:
        • Mayor: {multi_tf.get('mayor', 'N/A')}
        • Media: {multi_tf.get('media', 'N/A')}
        • Menor: {multi_tf.get('menor', 'N/A')}
        
        NIVELES DE TRADING:
        Precio Actual: ${signal_data['current_price']:.6f}
        Entrada: ${signal_data['entry']:.6f}
        Stop Loss: ${signal_data['stop_loss']:.6f}
        Take Profit: ${signal_data['take_profit'][0]:.6f}
        
        Apalancamiento: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', ['Condiciones base'])[:5]])}
        
        Sistema profesional con confirmación multi-temporalidad
        """
        
        ax7.text(0.1, 0.9, signal_info, transform=ax7.transAxes, fontsize=9,
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
        'timezone': 'America/La_Paz',
        'is_scalping_time': indicator.is_scalping_time()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'winrate': indicator.winrate_data['winrate']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
