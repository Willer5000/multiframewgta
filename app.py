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
import signal
import functools

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message="Timeout"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(error_message)
            
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator

app = Flask(__name__)

# Configuración Telegram MEJORADA
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

# JERARQUÍA TEMPORAL MEJORADA (OBLIGATORIA)
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

# SISTEMA DE TRACKING Y WINRATE
class PerformanceTracker:
    def __init__(self):
        self.operations_history = []
        self.winrate_data = {}
        self.max_history = 200
        
    def add_operation(self, symbol, interval, signal_type, entry_price, exit_price, outcome, timestamp):
        """Agregar operación al historial"""
        operation = {
            'symbol': symbol,
            'interval': interval,
            'signal_type': signal_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'outcome': outcome,  # 'win', 'loss', 'breakeven'
            'timestamp': timestamp,
            'pnl_percent': ((exit_price - entry_price) / entry_price * 100) if signal_type == 'LONG' else ((entry_price - exit_price) / entry_price * 100)
        }
        
        self.operations_history.append(operation)
        
        # Mantener solo el historial máximo
        if len(self.operations_history) > self.max_history:
            self.operations_history = self.operations_history[-self.max_history:]
            
        self._update_winrate()
    
    def _update_winrate(self):
        """Actualizar cálculos de winrate"""
        if not self.operations_history:
            self.winrate_data = {'global_winrate': 0, 'total_operations': 0, 'winning_operations': 0}
            return
            
        winning_ops = [op for op in self.operations_history if op['outcome'] == 'win']
        total_ops = len(self.operations_history)
        
        self.winrate_data = {
            'global_winrate': (len(winning_ops) / total_ops * 100) if total_ops > 0 else 0,
            'total_operations': total_ops,
            'winning_operations': len(winning_ops),
            'avg_win_pct': np.mean([op['pnl_percent'] for op in winning_ops]) if winning_ops else 0,
            'avg_loss_pct': np.mean([abs(op['pnl_percent']) for op in self.operations_history if op['outcome'] == 'loss']) if any(op['outcome'] == 'loss' for op in self.operations_history) else 0
        }
    
    def get_winrate(self):
        """Obtener winrate actual"""
        return self.winrate_data
    
    def get_strategy_performance(self, strategy_filters):
        """Obtener performance por estrategia específica"""
        filtered_ops = self.operations_history
        
        for filter_key, filter_value in strategy_filters.items():
            filtered_ops = [op for op in filtered_ops if op.get(filter_key) == filter_value]
            
        if not filtered_ops:
            return {'winrate': 0, 'total_ops': 0, 'winning_ops': 0}
            
        winning_ops = [op for op in filtered_ops if op['outcome'] == 'win']
        
        return {
            'winrate': (len(winning_ops) / len(filtered_ops) * 100),
            'total_ops': len(filtered_ops),
            'winning_ops': len(winning_ops)
        }

# Instancia global del tracker
performance_tracker = PerformanceTracker()

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_signals = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.performance_tracker = performance_tracker
    
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def is_scalping_time(self):
        """Verificar si es horario de scalping (lunes a viernes de 4am a 4pm hora boliviana)"""
        now = self.get_bolivia_time()
        if now.weekday() >= 5:  # Sábado o Domingo
            return False
        return 4 <= now.hour < 16  # De 4am a 4pm

    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela"""
        if interval == '15m':
            next_close = current_time.replace(minute=current_time.minute // 15 * 15, second=0, microsecond=0) + timedelta(minutes=15)
            return (next_close - current_time).total_seconds() <= 450  # 7.5 minutos (50%)
        elif interval == '30m':
            next_close = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0) + timedelta(minutes=30)
            return (next_close - current_time).total_seconds() <= 900  # 15 minutos (50%)
        elif interval == '1h':
            next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return (next_close - current_time).total_seconds() <= 1800  # 30 minutos (50%)
        elif interval == '2h':
            current_hour = current_time.hour
            next_2h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour % 2 == 0:
                next_2h_close += timedelta(hours=2)
            else:
                next_2h_close += timedelta(hours=1)
            return (next_2h_close - current_time).total_seconds() <= 3600  # 1 hora (50%)
        elif interval == '4h':
            current_hour = current_time.hour
            next_4h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 4
            if remainder == 0:
                next_4h_close += timedelta(hours=4)
            else:
                next_4h_close += timedelta(hours=4 - remainder)
            return (next_4h_close - current_time).total_seconds() <= 7200  # 2 horas (50%)
        elif interval == '8h':
            current_hour = current_time.hour
            next_8h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 8
            if remainder == 0:
                next_8h_close += timedelta(hours=8)
            else:
                next_8h_close += timedelta(hours=8 - remainder)
            return (next_8h_close - current_time).total_seconds() <= 14400  # 4 horas (50%)
        elif interval == '12h':
            current_hour = current_time.hour
            next_12h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour < 8:
                next_12h_close = next_12h_close.replace(hour=20)
            else:
                next_12h_close = next_12h_close.replace(hour=8) + timedelta(days=1)
            return (next_12h_close - current_time).total_seconds() <= 21600  # 6 horas (50%)
        elif interval == '1D':
            tomorrow_8pm = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
            if current_time.hour >= 20:
                tomorrow_8pm += timedelta(days=1)
            return (tomorrow_8pm - current_time).total_seconds() <= 43200  # 12 horas (50%)
        
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
                '1D': '1day', '3D': '3day', '1W': '1week'
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
        np.random.seed(42)
        # Precios realistas basados en el símbolo
        if 'BTC' in symbol:
            base_price = 45000
        elif 'ETH' in symbol:
            base_price = 3000
        elif 'BNB' in symbol:
            base_price = 600
        else:
            base_price = 100
            
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
            
            # Soporte y resistencia con lookback extendido
            support_1 = np.min(low[-20:])
            resistance_1 = np.max(high[-20:])
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                entry = min(current_price, support_1 * 1.02)
                stop_loss = max(support_1 * 0.97, entry - (current_atr * 1.8))
                tp1 = resistance_1 * 0.98
                
                min_tp = entry + (2 * (entry - stop_loss))
                tp1 = max(tp1, min_tp)
                
            else:  # SHORT
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

    def calculate_squeeze_momentum(self, close, high, low, length=20, mult=2):
        """Calcular Squeeze Momentum (LazyBear)"""
        try:
            n = len(close)
            
            # Calcular Bandas de Bollinger
            bb_basis = self.calculate_sma(close, length)
            bb_dev = np.zeros(n)
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                bb_dev[i] = np.std(window) if len(window) > 1 else 0
            bb_upper = bb_basis + (bb_dev * mult)
            bb_lower = bb_basis - (bb_dev * mult)
            
            # Calcular Keltner Channel
            tr = np.zeros(n)
            tr[0] = high[0] - low[0]
            for i in range(1, n):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr[i] = max(tr1, tr2, tr3)
            
            atr = self.calculate_sma(tr, length)
            kc_upper = bb_basis + (atr * mult)
            kc_lower = bb_basis - (atr * mult)
            
            # Determinar Squeeze
            squeeze_on = np.zeros(n, dtype=bool)
            squeeze_off = np.zeros(n, dtype=bool)
            momentum = np.zeros(n)
            
            for i in range(n):
                if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
                    squeeze_on[i] = True
                else:
                    squeeze_off[i] = True
                
                # Calcular momentum simple
                if i >= 1:
                    momentum[i] = close[i] - close[i-1]
            
            return {
                'squeeze_on': squeeze_on.tolist(),
                'squeeze_off': squeeze_off.tolist(),
                'squeeze_momentum': momentum.tolist(),
                'bb_upper': bb_upper.tolist(),
                'bb_lower': bb_lower.tolist(),
                'kc_upper': kc_upper.tolist(),
                'kc_lower': kc_lower.tolist()
            }
            
        except Exception as e:
            print(f"Error calculando Squeeze Momentum: {e}")
            n = len(close)
            return {
                'squeeze_on': [False] * n,
                'squeeze_off': [True] * n,
                'squeeze_momentum': [0] * n,
                'bb_upper': [0] * n,
                'bb_lower': [0] * n,
                'kc_upper': [0] * n,
                'kc_lower': [0] * n
            }

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

    def check_multi_timeframe_trend(self, symbol, timeframe):
        """Verificar tendencia en múltiples temporalidades según jerarquía"""
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
                
                # Calcular tendencia usando Fuerza Maverick
                trend_data = self.calculate_trend_strength_maverick(close)
                current_signal = trend_data['strength_signals'][-1] if trend_data['strength_signals'] else 'NEUTRAL'
                
                # Mapear a tendencia simple
                if current_signal in ['STRONG_UP', 'WEAK_UP']:
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
        """Verificar condiciones obligatorias multi-timeframe"""
        try:
            # Obtener análisis multi-timeframe
            tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            if signal_type == 'LONG':
                # PARA LONG: Mayor alcista/neutral, Media EXCLUSIVAMENTE alcista, Menor alcista
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BULLISH', 'NEUTRAL']
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BULLISH'  # EXCLUSIVAMENTE alcista
                menor_ok = tf_analysis.get('menor', 'NEUTRAL') in ['BULLISH']  # Solo alcista
                
                # Verificar que NO hay zonas de no operar en ninguna temporalidad
                no_trade_zones = False
                for tf_type in ['mayor', 'media', 'menor']:
                    tf_value = TIMEFRAME_HIERARCHY[interval].get(tf_type)
                    if tf_value:
                        df = self.get_kucoin_data(symbol, tf_value, 20)
                        if df is not None and len(df) > 10:
                            trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                            if trend_data['no_trade_zones'][-1]:
                                no_trade_zones = True
                                break
                
                return mayor_ok and media_ok and menor_ok and not no_trade_zones
                
            elif signal_type == 'SHORT':
                # PARA SHORT: Mayor bajista/neutral, Media EXCLUSIVAMENTE bajista, Menor bajista
                mayor_ok = tf_analysis.get('mayor', 'NEUTRAL') in ['BEARISH', 'NEUTRAL']
                media_ok = tf_analysis.get('media', 'NEUTRAL') == 'BEARISH'  # EXCLUSIVAMENTE bajista
                menor_ok = tf_analysis.get('menor', 'NEUTRAL') in ['BEARISH']  # Solo bajista
                
                # Verificar que NO hay zonas de no operar en ninguna temporalidad
                no_trade_zones = False
                for tf_type in ['mayor', 'media', 'menor']:
                    tf_value = TIMEFRAME_HIERARCHY[interval].get(tf_type)
                    if tf_value:
                        df = self.get_kucoin_data(symbol, tf_value, 20)
                        if df is not None and len(df) > 10:
                            trend_data = self.calculate_trend_strength_maverick(df['close'].values)
                            if trend_data['no_trade_zones'][-1]:
                                no_trade_zones = True
                                break
                
                return mayor_ok and media_ok and menor_ok and not no_trade_zones
            
            return False
            
        except Exception as e:
            print(f"Error verificando condiciones obligatorias para {symbol}: {e}")
            return False

    def calculate_whale_signals_corrected(self, df, interval):
        """Implementación CORRECTA del indicador de ballenas - SOLO para 12H y 1D"""
        try:
            # SOLO aplicar en 12H y 1D
            if interval not in ['12h', '1D']:
                n = len(df)
                return {
                    'whale_pump': [0] * n,
                    'whale_dump': [0] * n,
                    'whale_active': [False] * n,
                    'whale_strength': 0
                }
            
            close = df['close'].values
            low = df['low'].values
            high = df['high'].values
            volume = df['volume'].values
            
            n = len(close)
            
            whale_pump_signal = np.zeros(n)
            whale_dump_signal = np.zeros(n)
            whale_active = np.zeros(n, dtype=bool)
            
            # Parámetros específicos para ballenas en timeframe alto
            sensitivity = 2.0
            min_volume_multiplier = 2.0
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                low_5 = np.min(low[max(0, i-5):i+1])
                high_5 = np.max(high[max(0, i-5):i+1])
                
                # Detección de ballenas compradoras
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] < close[i-1] or price_change < -1.0) and
                    low[i] <= low_5 * 1.02):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_pump_signal[i] = min(100, volume_ratio * 25 * sensitivity * volume_strength)
                    whale_active[i] = True
                
                # Detección de ballenas vendedoras
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] > close[i-1] or price_change > 1.0) and
                    high[i] >= high_5 * 0.98):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_dump_signal[i] = min(100, volume_ratio * 25 * sensitivity * volume_strength)
                    whale_active[i] = True
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            # Calcular fuerza actual de ballenas
            current_whale_strength = max(whale_pump_smooth[-1], whale_dump_smooth[-1]) if n > 0 else 0
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'whale_active': whale_active.tolist(),
                'whale_strength': float(current_whale_strength)
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_corrected: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'whale_active': [False] * n,
                'whale_strength': 0
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
        """Detectar divergencias entre precio e indicador (hasta 14 periodos)"""
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

    def calculate_adx(self, high, low, close, period=14):
        """Calcular ADX, +DI, -DI manualmente con validación"""
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

    def evaluate_signal_conditions_professional(self, data, current_idx, interval):
        """Evaluar condiciones de señal con nuevo sistema de scoring profesional"""
        conditions = {
            'long': {
                'mandatory_trend': {'value': False, 'weight': 60, 'description': 'Tendencias multi-TF obligatorias'},
                'rsi_maverick_divergence': {'value': False, 'weight': 10, 'description': 'Divergencia RSI Maverick alcista'},
                'rsi_traditional_divergence': {'value': False, 'weight': 10, 'description': 'Divergencia RSI Tradicional alcista'},
                'adx_dmi': {'value': False, 'weight': 5, 'description': 'ADX > 25 con +DI > -DI'},
                'macd_momentum': {'value': False, 'weight': 5, 'description': 'MACD con momentum alcista'},
                'squeeze_momentum': {'value': False, 'weight': 5, 'description': 'Squeeze Momentum positivo'},
                'moving_averages': {'value': False, 'weight': 5, 'description': 'Alineación medias móviles alcista'},
                'whale_confirmation': {'value': False, 'weight': 15, 'description': 'Confirmación ballenas (12H/1D)'}
            },
            'short': {
                'mandatory_trend': {'value': False, 'weight': 60, 'description': 'Tendencias multi-TF obligatorias'},
                'rsi_maverick_divergence': {'value': False, 'weight': 10, 'description': 'Divergencia RSI Maverick bajista'},
                'rsi_traditional_divergence': {'value': False, 'weight': 10, 'description': 'Divergencia RSI Tradicional bajista'},
                'adx_dmi': {'value': False, 'weight': 5, 'description': 'ADX > 25 con -DI > +DI'},
                'macd_momentum': {'value': False, 'weight': 5, 'description': 'MACD con momentum bajista'},
                'squeeze_momentum': {'value': False, 'weight': 5, 'description': 'Squeeze Momentum negativo'},
                'moving_averages': {'value': False, 'weight': 5, 'description': 'Alineación medias móviles bajista'},
                'whale_confirmation': {'value': False, 'weight': 15, 'description': 'Confirmación ballenas (12H/1D)'}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        # OBLIGATORIO: Condiciones multi-timeframe
        conditions['long']['mandatory_trend']['value'] = data.get('mandatory_conditions_long', False)
        conditions['short']['mandatory_trend']['value'] = data.get('mandatory_conditions_short', False)
        
        # RSI Maverick divergencias
        if current_idx < len(data['bullish_divergence_maverick']):
            lookback_start = max(0, current_idx - 13)  # Hasta 14 periodos
            for i in range(lookback_start, current_idx + 1):
                if i < len(data['bullish_divergence_maverick']) and data['bullish_divergence_maverick'][i]:
                    conditions['long']['rsi_maverick_divergence']['value'] = True
                    break
        
        if current_idx < len(data['bearish_divergence_maverick']):
            lookback_start = max(0, current_idx - 13)
            for i in range(lookback_start, current_idx + 1):
                if i < len(data['bearish_divergence_maverick']) and data['bearish_divergence_maverick'][i]:
                    conditions['short']['rsi_maverick_divergence']['value'] = True
                    break
        
        # RSI Tradicional divergencias
        if current_idx < len(data['bullish_divergence_traditional']):
            lookback_start = max(0, current_idx - 6)  # 7 velas
            for i in range(lookback_start, current_idx + 1):
                if i < len(data['bullish_divergence_traditional']) and data['bullish_divergence_traditional'][i]:
                    conditions['long']['rsi_traditional_divergence']['value'] = True
                    break
        
        if current_idx < len(data['bearish_divergence_traditional']):
            lookback_start = max(0, current_idx - 6)
            for i in range(lookback_start, current_idx + 1):
                if i < len(data['bearish_divergence_traditional']) and data['bearish_divergence_traditional'][i]:
                    conditions['short']['rsi_traditional_divergence']['value'] = True
                    break
        
        # ADX + DMI
        if current_idx < len(data['adx']):
            conditions['long']['adx_dmi']['value'] = (
                data['adx'][current_idx] > 25 and 
                data['plus_di'][current_idx] > data['minus_di'][current_idx]
            )
            conditions['short']['adx_dmi']['value'] = (
                data['adx'][current_idx] > 25 and 
                data['minus_di'][current_idx] > data['plus_di'][current_idx]
            )
        
        # MACD Momentum
        if current_idx < len(data['macd_histogram']):
            conditions['long']['macd_momentum']['value'] = data['macd_histogram'][current_idx] > 0
            conditions['short']['macd_momentum']['value'] = data['macd_histogram'][current_idx] < 0
        
        # Squeeze Momentum
        if current_idx < len(data['squeeze_momentum']):
            conditions['long']['squeeze_momentum']['value'] = data['squeeze_momentum'][current_idx] > 0
            conditions['short']['squeeze_momentum']['value'] = data['squeeze_momentum'][current_idx] < 0
        
        # Moving Averages
        if (current_idx < len(data['ma_50']) and current_idx < len(data['ma_200'])):
            conditions['long']['moving_averages']['value'] = (
                data['close'][current_idx] > data['ma_50'][current_idx] and 
                data['ma_50'][current_idx] > data['ma_200'][current_idx]
            )
            conditions['short']['moving_averages']['value'] = (
                data['close'][current_idx] < data['ma_50'][current_idx] and 
                data['ma_50'][current_idx] < data['ma_200'][current_idx]
            )
        
        # Whale Confirmation (SOLO 12H y 1D)
        if interval in ['12h', '1D'] and current_idx < len(data['whale_active']):
            whale_threshold = 25
            conditions['long']['whale_confirmation']['value'] = (
                data['whale_active'][current_idx] and 
                data['whale_pump'][current_idx] > whale_threshold
            )
            conditions['short']['whale_confirmation']['value'] = (
                data['whale_active'][current_idx] and 
                data['whale_dump'][current_idx] > whale_threshold
            )
        else:
            # En otros timeframes, esta condición no aplica (peso 0)
            conditions['long']['whale_confirmation']['weight'] = 0
            conditions['short']['whale_confirmation']['weight'] = 0
        
        return conditions

    def calculate_signal_score_professional(self, conditions, signal_type):
        """Calcular puntuación de señal con nuevo sistema profesional"""
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
        
        # OBLIGATORIEDAD: Si no se cumplen las condiciones multi-TF, score = 0
        if not signal_conditions.get('mandatory_trend', {}).get('value', False):
            return 0, []
        
        score = (achieved_weight / total_weight * 100)
        
        # Mínimos para operar
        if signal_type == 'long' and score < 75:
            return 0, []
        elif signal_type == 'short' and score < 80:
            return 0, []
        
        return min(score, 100), fulfilled_conditions

    
    
        def generate_signals_professional(self, symbol, interval, di_period=14, adx_threshold=25, 
                                   sr_period=50, rsi_length=20, bb_multiplier=2.0, leverage=15):
        """GENERACIÓN DE SEÑALES PROFESIONAL - SISTEMA COMPLETO MEJORADO"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # OBLIGATORIO: Verificar condiciones multi-timeframe
            mandatory_long = self.check_mandatory_conditions(symbol, interval, 'LONG')
            mandatory_short = self.check_mandatory_conditions(symbol, interval, 'SHORT')
            
            # Indicadores principales
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            # RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close, rsi_length, bb_multiplier)
            bullish_div_maverick, bearish_div_maverick = self.detect_divergence(close, rsi_maverick, 14)
            
            # RSI Tradicional
            rsi_traditional = self.calculate_rsi(close, 14)
            bullish_div_traditional, bearish_div_traditional = self.detect_divergence(close, rsi_traditional, 7)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Squeeze Momentum
            squeeze_data = self.calculate_squeeze_momentum(close, high, low)
            
            # Medias Móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, 2)
            
            # Fuerza de Tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Ballenas (SOLO 12H y 1D)
            whale_data = self.calculate_whale_signals_corrected(df, interval)
            
            current_idx = -1
            
            # Preparar datos para evaluación
            analysis_data = {
                'close': close.tolist() if hasattr(close, 'tolist') else list(close),
                'adx': adx.tolist() if hasattr(adx, 'tolist') else list(adx),
                'plus_di': plus_di.tolist() if hasattr(plus_di, 'tolist') else list(plus_di),
                'minus_di': minus_di.tolist() if hasattr(minus_di, 'tolist') else list(minus_di),
                'bullish_divergence_maverick': bullish_div_maverick,
                'bearish_divergence_maverick': bearish_div_maverick,
                'bullish_divergence_traditional': bullish_div_traditional,
                'bearish_divergence_traditional': bearish_div_traditional,
                'macd_histogram': macd_histogram.tolist() if hasattr(macd_histogram, 'tolist') else list(macd_histogram),
                'squeeze_momentum': squeeze_data['squeeze_momentum'],
                'ma_50': ma_50.tolist() if hasattr(ma_50, 'tolist') else list(ma_50),
                'ma_200': ma_200.tolist() if hasattr(ma_200, 'tolist') else list(ma_200),
                'whale_active': whale_data['whale_active'],
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'mandatory_conditions_long': mandatory_long,
                'mandatory_conditions_short': mandatory_short
            }
            
            conditions = self.evaluate_signal_conditions_professional(analysis_data, current_idx, interval)
            
            long_score, long_conditions = self.calculate_signal_score_professional(conditions, 'long')
            short_score, short_conditions = self.calculate_signal_score_professional(conditions, 'short')
            
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= 75:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= 80:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            current_price = float(close[current_idx])
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # Registrar señal activa si es válida
            if signal_type in ['LONG', 'SHORT']:
                signal_key = f"{symbol}_{interval}_{signal_type}"
                self.active_signals[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': levels_data['entry'],
                    'timestamp': self.get_bolivia_time().strftime("%Y-%m-%d %H:%M:%S"),
                    'score': signal_score
                }
            
            # Convertir todos los arrays numpy a listas para JSON
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
                'whale_strength': whale_data['whale_strength'],
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'rsi_traditional': float(rsi_traditional[current_idx] if current_idx < len(rsi_traditional) else 50),
                'fulfilled_conditions': fulfilled_conditions,
                'trend_strength_signal': trend_strength_data['strength_signals'][current_idx] if current_idx < len(trend_strength_data['strength_signals']) else 'NEUTRAL',
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False,
                'mandatory_conditions_met': mandatory_long if signal_type == 'LONG' else mandatory_short if signal_type == 'SHORT' else False,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist() if hasattr(adx, 'tolist') else list(adx[-50:]),
                    'plus_di': plus_di[-50:].tolist() if hasattr(plus_di, 'tolist') else list(plus_di[-50:]),
                    'minus_di': minus_di[-50:].tolist() if hasattr(minus_di, 'tolist') else list(minus_di[-50:]),
                    'di_cross_bullish': di_cross_bullish[-50:],
                    'di_cross_bearish': di_cross_bearish[-50:],
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:].tolist() if hasattr(rsi_traditional, 'tolist') else list(rsi_traditional[-50:]),
                    'bullish_divergence_maverick': bullish_div_maverick[-50:],
                    'bearish_divergence_maverick': bearish_div_maverick[-50:],
                    'bullish_divergence_traditional': bullish_div_traditional[-50:],
                    'bearish_divergence_traditional': bearish_div_traditional[-50:],
                    'macd': macd[-50:].tolist() if hasattr(macd, 'tolist') else list(macd[-50:]),
                    'macd_signal': macd_signal[-50:].tolist() if hasattr(macd_signal, 'tolist') else list(macd_signal[-50:]),
                    'macd_histogram': macd_histogram[-50:].tolist() if hasattr(macd_histogram, 'tolist') else list(macd_histogram[-50:]),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['squeeze_momentum'][-50:],
                    'ma_9': ma_9[-50:].tolist() if hasattr(ma_9, 'tolist') else list(ma_9[-50:]),
                    'ma_21': ma_21[-50:].tolist() if hasattr(ma_21, 'tolist') else list(ma_21[-50:]),
                    'ma_50': ma_50[-50:].tolist() if hasattr(ma_50, 'tolist') else list(ma_50[-50:]),
                    'ma_200': ma_200[-50:].tolist() if hasattr(ma_200, 'tolist') else list(ma_200[-50:]),
                    'bb_upper': bb_upper[-50:].tolist() if hasattr(bb_upper, 'tolist') else list(bb_upper[-50:]),
                    'bb_middle': bb_middle[-50:].tolist() if hasattr(bb_middle, 'tolist') else list(bb_middle[-50:]),
                    'bb_lower': bb_lower[-50:].tolist() if hasattr(bb_lower, 'tolist') else list(bb_lower[-50:]),
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
                    'colors': trend_strength_data['colors'][-50:]
                }
            }
            
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
            'whale_strength': 0,
            'rsi_maverick': 0.5,
            'rsi_traditional': 50,
            'fulfilled_conditions': [],
            'trend_strength_signal': 'NEUTRAL',
            'no_trade_zone': False,
            'mandatory_conditions_met': False,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de scalping con nuevo sistema"""
        alerts = []
        telegram_intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D']
        
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
                    signal_data = self.generate_signals_professional(symbol, interval)
                    
                    if signal_data['signal'] in ['LONG', 'SHORT'] and signal_data['mandatory_conditions_met']:
                        
                        risk_category = next(
                            (cat for cat, symbols in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols), 'medio'
                        )
                        
                        # Obtener winrate para esta estrategia
                        winrate_info = self.performance_tracker.get_strategy_performance({
                            'symbol': symbol,
                            'interval': interval,
                            'signal_type': signal_data['signal']
                        })
                        
                        # Calcular leverage óptimo basado en score y categoría
                        base_leverage = min(20, max(5, int(signal_data['signal_score'] / 5)))
                        risk_factors = {'bajo': 1.0, 'medio': 0.8, 'alto': 0.6, 'memecoins': 0.5}
                        risk_factor = risk_factors.get(risk_category, 0.7)
                        optimal_leverage = int(base_leverage * risk_factor)
                        
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
                            'winrate': winrate_info.get('winrate', 0),
                            'mandatory_conditions': signal_data['mandatory_conditions_met']
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
        
        for signal_key, signal_data in list(self.active_signals.items()):
            try:
                symbol = signal_data['symbol']
                interval = signal_data['interval']
                signal_type = signal_data['signal']
                entry_price = signal_data['entry_price']
                entry_time = signal_data['timestamp']
                
                # Obtener datos actuales
                current_signal = self.generate_signals_professional(symbol, interval)
                
                if current_signal['signal'] == 'NEUTRAL' or not current_signal['mandatory_conditions_met']:
                    # Condiciones cambiaron, salir
                    current_price = current_signal['current_price']
                    
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
                        'reason': "Condiciones obligatorias ya no se cumplen",
                        'trend_strength': current_signal.get('trend_strength_signal', 'NEUTRAL'),
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    exit_alerts.append(exit_alert)
                    
                    # Registrar en performance tracker
                    outcome = 'win' if pnl_percent > 0 else 'loss' if pnl_percent < 0 else 'breakeven'
                    self.performance_tracker.add_operation(
                        symbol, interval, signal_type, entry_price, current_price, 
                        outcome, current_time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    # Remover señal activa
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram con nuevo formato profesional"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        
        if alert_type == 'entry':
            message = f"""
🚨 MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} 
📊 Score: {alert_data['score']:.1f}%

✅ Condiciones Obligatorias: CUMPLIDAS
💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}
📈 Winrate Estrategia: {alert_data.get('winrate', 0):.1f}%

💰 Precio Actual: ${alert_data['current_price']:.6f}
🎯 Entrada: ${alert_data['entry']:.6f}

🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
🎯 Take Profit: ${alert_data['take_profit']:.6f}

⚡ Apalancamiento: x{alert_data['leverage']}

🔔 Condiciones Cumplidas:
• {chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])[:3]])}

📊 Sistema Multi-Timeframe Confirmado ✅
            """
            
        else:  # exit alert
            pnl_text = f"📊 P&L: {alert_data['pnl_percent']:+.2f}%"
            pnl_emoji = "🟢" if alert_data['pnl_percent'] > 0 else "🔴" if alert_data['pnl_percent'] < 0 else "🟡"
            
            message = f"""
🚨 SEÑAL DE SALIDA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']} - CERRAR

{pnl_emoji} {pnl_text}
💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}

📊 Razón: {alert_data['reason']}
💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

🔔 Sistema de Gestión Activa
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
    """Verificador de alertas en segundo plano mejorado"""
    intraday_intervals = ['15m', '30m', '1h', '2h']
    swing_intervals = ['4h', '8h', '12h', '1D']
    
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


def safe_convert_to_list(self, data):
        """Convertir datos a lista de forma segura para JSON"""
        if data is None:
            return []
        if hasattr(data, 'tolist'):
            return data.tolist()
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]


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
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_professional(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, leverage
        )
        
        # Asegurar que todos los datos sean serializables
        if signal_data and 'data' in signal_data:
            # Convertir cualquier array numpy restante en la data
            for record in signal_data['data']:
                for key, value in record.items():
                    if hasattr(value, 'tolist'):
                        record[key] = value.tolist()
        
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        # Retornar una señal vacía pero válida
        empty_signal = indicator._create_empty_signal(request.args.get('symbol', 'BTC-USDT'))
        return jsonify(empty_signal)
        

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
        leverage = int(request.args.get('leverage', 15))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
            try:
                signal_data = indicator.generate_signals_professional(
                    symbol, interval, di_period, adx_threshold, sr_period,
                    rsi_length, bb_multiplier, leverage
                )
                
                if signal_data and signal_data['signal'] != 'NEUTRAL' and signal_data['mandatory_conditions_met']:
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
@timeout(seconds=8, error_message="Scatter data calculation timeout")
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
                signal_data = indicator.generate_signals_professional(symbol, interval, di_period, adx_threshold)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones de compra/venta mejoradas
                    buy_pressure = min(100, max(0,
                        (1 if signal_data['mandatory_conditions_met'] and signal_data['signal'] == 'LONG' else 0) * 40 +
                        (signal_data['whale_pump'] / 100 * 20) +
                        (signal_data['plus_di'] / 100 * 15) +
                        (signal_data['rsi_maverick'] * 15) +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 10)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (1 if signal_data['mandatory_conditions_met'] and signal_data['signal'] == 'SHORT' else 0) * 40 +
                        (signal_data['whale_dump'] / 100 * 20) +
                        (signal_data['minus_di'] / 100 * 15) +
                        ((1 - signal_data['rsi_maverick']) * 15) +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 10)
                    ))
                    
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
                        'mandatory_conditions': signal_data['mandatory_conditions_met']
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
        winrate_data = performance_tracker.get_winrate()
        return jsonify(winrate_data)
        
    except Exception as e:
        print(f"Error en /api/winrate: {e}")
        return jsonify({'global_winrate': 0, 'total_operations': 0, 'winning_operations': 0})

@app.route('/api/strategy_report')
def get_strategy_report():
    """Endpoint para obtener reporte de estrategias"""
    try:
        # Generar reporte de performance por diferentes estrategias
        strategies = [
            {'name': 'LONG 4H', 'filters': {'interval': '4h', 'signal_type': 'LONG'}},
            {'name': 'SHORT 4H', 'filters': {'interval': '4h', 'signal_type': 'SHORT'}},
            {'name': 'LONG 1D', 'filters': {'interval': '1D', 'signal_type': 'LONG'}},
            {'name': 'SHORT 1D', 'filters': {'interval': '1D', 'signal_type': 'SHORT'}}
        ]
        
        report = {}
        for strategy in strategies:
            performance = performance_tracker.get_strategy_performance(strategy['filters'])
            report[strategy['name']] = performance
        
        return jsonify(report)
        
    except Exception as e:
        print(f"Error en /api/strategy_report: {e}")
        return jsonify({})

@app.route('/api/winrate_data')
def get_winrate_data():
    """Endpoint para obtener datos de winrate (compatibilidad con frontend)"""
    try:
        winrate_data = performance_tracker.get_winrate()
        return jsonify({
            'winrate': winrate_data.get('global_winrate', 0),
            'total_operations': winrate_data.get('total_operations', 0),
            'winning_operations': winrate_data.get('winning_operations', 0),
            'avg_win_pct': winrate_data.get('avg_win_pct', 0),
            'avg_loss_pct': winrate_data.get('avg_loss_pct', 0)
        })
    except Exception as e:
        print(f"Error en /api/winrate_data: {e}")
        return jsonify({
            'winrate': 0, 
            'total_operations': 0, 
            'winning_operations': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0
        })

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
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax2.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax2.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax2.set_ylabel('Fuerza Tendencia %')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: RSI Maverick y Tradicional
        ax3 = plt.subplot(7, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax3.plot(rsi_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax3.plot(rsi_dates, [x/100 for x in signal_data['indicators']['rsi_traditional']], 
                    'orange', linewidth=1, label='RSI Tradicional', alpha=0.7)
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax3.set_ylabel('RSI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: ADX/DMI
        ax4 = plt.subplot(7, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax4.plot(adx_dates, signal_data['indicators']['adx'], 
                    'white', linewidth=2, label='ADX')
            ax4.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax4.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
        ax4.set_ylabel('ADX/DMI')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: MACD
        ax5 = plt.subplot(7, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax5.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax5.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            ax5.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']], 
                   alpha=0.6, label='Histograma')
        ax5.set_ylabel('MACD')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Squeeze Momentum
        ax6 = plt.subplot(7, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'squeeze_momentum' in signal_data['indicators']:
            squeeze_dates = dates[-len(signal_data['indicators']['squeeze_momentum']):]
            squeeze_momentum = signal_data['indicators']['squeeze_momentum']
            
            # Crear histograma para Squeeze Momentum
            colors = ['green' if x > 0 else 'red' for x in squeeze_momentum]
            ax6.bar(squeeze_dates, squeeze_momentum, color=colors, alpha=0.7, width=0.8)
            ax6.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            
            # Marcar períodos de squeeze
            squeeze_on = signal_data['indicators']['squeeze_on']
            for i, date in enumerate(squeeze_dates):
                if i < len(squeeze_on) and squeeze_on[i]:
                    ax6.axvline(x=date, color='yellow', alpha=0.2, linewidth=3)
            
            ax6.set_ylabel('Squeeze Momentum')
            ax6.grid(True, alpha=0.3)
        
        # Información de la señal
        ax7 = plt.subplot(7, 1, 7)
        ax7.axis('off')
        
        # Obtener winrate
        winrate_data = performance_tracker.get_winrate()
        
        signal_info = f"""
        MULTI-TIMEFRAME CRYPTO WGTA PRO
        
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        CONDICIONES OBLIGATORIAS: {'✅ CUMPLIDAS' if signal_data['mandatory_conditions_met'] else '❌ NO CUMPLIDAS'}
        
        WINRATE GLOBAL: {winrate_data['global_winrate']:.1f}%
        OPERACIONES: {winrate_data['total_operations']} (Ganadas: {winrate_data['winning_operations']})
        
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        FUERZA TENDENCIA: {signal_data.get('trend_strength_signal', 'NEUTRAL')}
        {'⚠️ ZONA DE NO OPERAR' if signal_data.get('no_trade_zone') else ''}
        
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
        APALANCAMIENTO: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
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
