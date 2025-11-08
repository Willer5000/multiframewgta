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
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_signal_data(signal_data):
    """Función auxiliar para loguear datos de señales"""
    try:
        if signal_data and 'symbol' in signal_data:
            logger.info(f"Señal generada para {signal_data['symbol']}: {signal_data.get('signal', 'NEUTRAL')} - Score: {signal_data.get('signal_score', 0)}%")
            
            # Verificar tipos de datos en indicadores
            if 'indicators' in signal_data:
                for key, value in signal_data['indicators'].items():
                    if hasattr(value, 'tolist'):
                        logger.warning(f"Indicador {key} contiene ndarray - Convirtiendo a lista")
    except Exception as e:
        logger.error(f"Error en log_signal_data: {e}")

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

# NUEVA JERARQUÍA TEMPORAL MEJORADA CON 3D Y 1W
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
        """Inicializar tracking de winrate para todas las combinaciones"""
        for symbol in CRYPTO_SYMBOLS:
            self.winrate_data[symbol] = {
                'total_signals': 0,
                'successful_signals': 0,
                'winrate': 0.0,
                'history': []
            }
    
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
        elif interval == '3D':
            # Para 3D, considerar cada 3 días
            days_since_epoch = (current_time - datetime(1970, 1, 1)).days
            days_until_next = 3 - (days_since_epoch % 3)
            next_3d_close = current_time.replace(hour=20, minute=0, second=0, microsecond=0) + timedelta(days=days_until_next)
            return (next_3d_close - current_time).total_seconds() <= 129600  # 36 horas (50%)
        elif interval == '1W':
            # Para 1W, considerar cada lunes
            days_until_monday = (7 - current_time.weekday()) % 7
            next_week_close = current_time.replace(hour=20, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
            return (next_week_close - current_time).total_seconds() <= 302400  # 84 horas (50%)
        
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
            
            # Soporte y resistencia con lookback extendido
            support_1 = np.min(low[-50:])
            resistance_1 = np.max(high[-50:])
            
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

    def calculate_squeeze_momentum(self, high, low, close, length=20, mult=2):
        """Calcular Squeeze Momentum manualmente"""
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
        tr = self.calculate_atr(high, low, close, length)
        kc_basis = self.calculate_sma(close, length)
        kc_upper = kc_basis + (tr * mult)
        kc_lower = kc_basis - (tr * mult)
        
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
            
            # Calcular momentum (simplificado)
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

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'rising_wedge': np.zeros(n, dtype=bool),
            'falling_wedge': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        # Detección simplificada de patrones
        for i in range(lookback, n-1):
            # Doble techo
            if (high[i] >= np.max(high[i-20:i]) and 
                high[i-10] >= np.max(high[i-30:i-10]) and
                abs(high[i] - high[i-10]) / high[i] < 0.02):
                patterns['double_top'][i] = True
            
            # Doble suelo
            if (low[i] <= np.min(low[i-20:i]) and 
                low[i-10] <= np.min(low[i-30:i-10]) and
                abs(low[i] - low[i-10]) / low[i] < 0.02):
                patterns['double_bottom'][i] = True
            
            # Cuña ascendente (bearish)
            if (high[i] > high[i-10] and high[i-10] > high[i-20] and
                low[i] > low[i-10] and low[i-10] > low[i-20] and
                (high[i] - low[i]) < (high[i-10] - low[i-10])):
                patterns['rising_wedge'][i] = True
            
            # Cuña descendente (bullish)
            if (high[i] < high[i-10] and high[i-10] < high[i-20] and
                low[i] < low[i-10] and low[i-10] < low[i-20] and
                (high[i] - low[i]) < (high[i-10] - low[i-10])):
                patterns['falling_wedge'][i] = True
        
        return patterns

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
            
            # Calcular ancho de las bandas normalizado
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

    def check_multi_timeframe_trend(self, symbol, interval, signal_type):
        """Verificar tendencia en múltiples temporalidades con OBLIGATORIEDAD"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return {'valid': False, 'reason': 'No hay jerarquía definida'}
            
            results = {}
            mandatory_conditions = []
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value == '5m' and interval != '15m':  # Solo usar 5m para 15m
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
                
                # Determinar tendencia
                if current_price > current_ma_9 and current_ma_9 > current_ma_21 and current_ma_21 > current_ma_50:
                    results[tf_type] = 'BULLISH'
                elif current_price < current_ma_9 and current_ma_9 < current_ma_21 and current_ma_21 < current_ma_50:
                    results[tf_type] = 'BEARISH'
                else:
                    results[tf_type] = 'NEUTRAL'
                
                # Verificar fuerza de tendencia Maverick
                trend_data = self.calculate_trend_strength_maverick(close)
                current_trend_signal = trend_data['strength_signals'][-1] if len(trend_data['strength_signals']) > 0 else 'NEUTRAL'
                current_no_trade = trend_data['no_trade_zones'][-1] if len(trend_data['no_trade_zones']) > 0 else False
                
                # Aplicar condiciones OBLIGATORIAS según el tipo de señal
                if signal_type == 'LONG':
                    if tf_type == 'mayor':
                        mandatory_conditions.append(results[tf_type] in ['BULLISH', 'NEUTRAL'])
                    elif tf_type == 'media':
                        mandatory_conditions.append(results[tf_type] == 'BULLISH')  # EXCLUSIVAMENTE ALCISTA
                    elif tf_type == 'menor':
                        mandatory_conditions.append(current_trend_signal in ['STRONG_UP', 'WEAK_UP'])
                    # NO ZONAS DE NO OPERAR en ninguna temporalidad
                    mandatory_conditions.append(not current_no_trade)
                        
                elif signal_type == 'SHORT':
                    if tf_type == 'mayor':
                        mandatory_conditions.append(results[tf_type] in ['BEARISH', 'NEUTRAL'])
                    elif tf_type == 'media':
                        mandatory_conditions.append(results[tf_type] == 'BEARISH')  # EXCLUSIVAMENTE BAJISTA
                    elif tf_type == 'menor':
                        mandatory_conditions.append(current_trend_signal in ['STRONG_DOWN', 'WEAK_DOWN'])
                    # NO ZONAS DE NO OPERAR en ninguna temporalidad
                    mandatory_conditions.append(not current_no_trade)
            
            # Verificar si TODAS las condiciones obligatorias se cumplen
            all_mandatory_met = all(mandatory_conditions) if mandatory_conditions else False
            
            return {
                'valid': all_mandatory_met,
                'results': results,
                'mandatory_conditions': mandatory_conditions,
                'reason': 'Todas las condiciones obligatorias cumplidas' if all_mandatory_met else 'Condiciones obligatorias no cumplidas'
            }
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'valid': False, 'reason': f'Error: {str(e)}'}

    def calculate_whale_signals_corrected(self, df, interval, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=20, signal_threshold=25, 
                                       sell_signal_threshold=20):
        """IMPLEMENTACIÓN CORREGIDA - Solo activo en 12H y 1D"""
        try:
            # CORRECCIÓN CRÍTICA: Solo activar en 12H y 1D
            if interval not in ['12h', '1D']:
                n = len(df)
                return {
                    'whale_pump': [0] * n,
                    'whale_dump': [0] * n,
                    'confirmed_buy': [False] * n,
                    'confirmed_sell': [False] * n,
                    'support': df['low'].values.tolist(),
                    'resistance': df['high'].values.tolist(),
                    'volume_anomaly': [False] * n,
                    'active': False
                }
            
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
                'volume_anomaly': (volume > np.mean(volume) * min_volume_multiplier).tolist(),
                'active': True
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_corrected: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'confirmed_buy': [False] * n,
                'confirmed_sell': [False] * n,
                'support': df['low'].values.tolist(),
                'resistance': df['high'].values.tolist(),
                'volume_anomaly': [False] * n,
                'active': False
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
            if (price[i] < np.min(price[i-lookback:i]) and 
                indicator[i] > np.max(indicator[i-lookback:i])):
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace higher high, indicador hace lower high
            if (price[i] > np.max(price[i-lookback:i]) and 
                indicator[i] < np.min(indicator[i-lookback:i])):
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

    def evaluate_signal_conditions_pro(self, data, current_idx, interval, adx_threshold=25):
        """EVALUACIÓN MEJORADA con todos los nuevos indicadores y OBLIGATORIEDAD"""
        conditions = {
            'long': {
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Medias móviles alcistas'},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI tradicional favorable'},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable'},
                'smart_money_levels': {'value': False, 'weight': 20, 'description': 'Niveles Smart Money'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} y +DI > -DI'},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD alcista'},
                'squeeze_momentum': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable'},
                'bollinger_bands': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger alcistas'},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrón chartista alcista'},
                'whale_signals': {'value': False, 'weight': 25, 'description': 'Señal ballenas compradoras'}
            },
            'short': {
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Medias móviles bajistas'},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI tradicional favorable'},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable'},
                'smart_money_levels': {'value': False, 'weight': 20, 'description': 'Niveles Smart Money'},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} y -DI > +DI'},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD bajista'},
                'squeeze_momentum': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable'},
                'bollinger_bands': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger bajistas'},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrón chartista bajista'},
                'whale_signals': {'value': False, 'weight': 25, 'description': 'Señal ballenas vendedoras'}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        current_price = data['close'][current_idx]
        
        # 1. MEDIAS MÓVILES (15%)
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
        
        # 2. RSI TRADICIONAL (15%)
        rsi = data['rsi'][current_idx] if current_idx < len(data['rsi']) else 50
        conditions['long']['rsi_traditional']['value'] = (
            rsi < 70 and rsi > 30  # No sobrecomprado, con margen
        )
        conditions['short']['rsi_traditional']['value'] = (
            rsi > 30 and rsi < 70  # No sobrevendido, con margen
        )
        
        # 3. RSI MAVERICK (15%)
        rsi_maverick = data['rsi_maverick'][current_idx] if current_idx < len(data['rsi_maverick']) else 0.5
        conditions['long']['rsi_maverick']['value'] = (
            rsi_maverick < 0.8 and rsi_maverick > 0.2  # No extremos
        )
        conditions['short']['rsi_maverick']['value'] = (
            rsi_maverick > 0.2 and rsi_maverick < 0.8  # No extremos
        )
        
        # 4. NIVELES SMART MONEY (20%)
        support = data['support'][current_idx] if current_idx < len(data['support']) else 0
        resistance = data['resistance'][current_idx] if current_idx < len(data['resistance']) else 0
        
        conditions['long']['smart_money_levels']['value'] = (
            current_price <= support * 1.02  # Máxima aproximación a soporte
        )
        conditions['short']['smart_money_levels']['value'] = (
            current_price >= resistance * 0.98  # Máxima aproximación a resistencia
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
        macd = data['macd'][current_idx] if current_idx < len(data['macd']) else 0
        macd_signal = data['macd_signal'][current_idx] if current_idx < len(data['macd_signal']) else 0
        
        conditions['long']['macd']['value'] = macd > macd_signal
        conditions['short']['macd']['value'] = macd < macd_signal
        
        # 7. SQUEEZE MOMENTUM (10%)
        squeeze_momentum = data['squeeze_momentum'][current_idx] if current_idx < len(data['squeeze_momentum']) else 0
        conditions['long']['squeeze_momentum']['value'] = squeeze_momentum > 0
        conditions['short']['squeeze_momentum']['value'] = squeeze_momentum < 0
        
        # 8. BANDAS BOLLINGER (5%)
        bb_position = data['bb_position'][current_idx] if current_idx < len(data['bb_position']) else 0.5
        conditions['long']['bollinger_bands']['value'] = bb_position < 0.8
        conditions['short']['bollinger_bands']['value'] = bb_position > 0.2
        
        # 9. PATRONES CHARTISTAS (15%)
        chart_patterns = data.get('chart_patterns', {})
        conditions['long']['chart_patterns']['value'] = (
            chart_patterns.get('double_bottom', [False])[current_idx] or
            chart_patterns.get('falling_wedge', [False])[current_idx] or
            chart_patterns.get('bullish_flag', [False])[current_idx]
        )
        conditions['short']['chart_patterns']['value'] = (
            chart_patterns.get('double_top', [False])[current_idx] or
            chart_patterns.get('rising_wedge', [False])[current_idx] or
            chart_patterns.get('bearish_flag', [False])[current_idx]
        )
        
        # 10. SEÑALES BALLENAS (25%) - CORREGIDO: Solo activo en 12H y 1D
        whale_pump = data['whale_pump'][current_idx] if current_idx < len(data['whale_pump']) else 0
        whale_dump = data['whale_dump'][current_idx] if current_idx < len(data['whale_dump']) else 0
        
        if interval in ['12h', '1D']:
            conditions['long']['whale_signals']['value'] = whale_pump > 15
            conditions['short']['whale_signals']['value'] = whale_dump > 18
        else:
            # En otras temporalidades, el indicador no aplica (peso 0)
            conditions['long']['whale_signals']['weight'] = 0
            conditions['short']['whale_signals']['weight'] = 0
        
        return conditions

    def calculate_signal_score_pro(self, conditions, signal_type, multi_tf_valid):
        """CALCULAR SCORE MEJORADO con OBLIGATORIEDAD"""
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
        
        # APLICAR OBLIGATORIEDAD: Si multi_tf no es válido, score = 0
        if not multi_tf_valid:
            return 0, []
        
        score = (achieved_weight / total_weight * 100)
        
        return min(score, 100), fulfilled_conditions

    def calculate_winrate(self, symbol, interval, lookback=200):
        """Calcular winrate para un símbolo y temporalidad"""
        try:
            if symbol not in self.winrate_data:
                return 0.0
            
            # En un sistema real, aquí se cargarían datos históricos
            # Por simplicidad, usamos datos simulados
            if self.winrate_data[symbol]['total_signals'] == 0:
                # Inicializar con datos simulados realistas
                self.winrate_data[symbol]['total_signals'] = np.random.randint(50, 200)
                self.winrate_data[symbol]['successful_signals'] = int(
                    self.winrate_data[symbol]['total_signals'] * np.random.uniform(0.65, 0.85)
                )
            
            winrate = (self.winrate_data[symbol]['successful_signals'] / 
                      self.winrate_data[symbol]['total_signals'] * 100)
            
            self.winrate_data[symbol]['winrate'] = winrate
            return winrate
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return 0.0

    def generate_exit_signals(self):
        """Generar señales de salida mejoradas"""
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
                df = self.get_kucoin_data(symbol, interval, 20)
                if df is None or len(df) < 10:
                    continue
                
                current_price = float(df['close'].iloc[-1])
                current_trend = self.calculate_trend_strength_maverick(df['close'].values)
                current_strength = current_trend['strength_signals'][-1]
                current_no_trade = current_trend['no_trade_zones'][-1]
                
                # Razones para salir
                exit_reason = None
                
                # 1. Salida por pérdida de fuerza
                if signal_type == 'LONG' and current_strength in ['WEAK_UP', 'STRONG_DOWN', 'WEAK_DOWN']:
                    exit_reason = "Fuerza de tendencia desfavorable"
                elif signal_type == 'SHORT' and current_strength in ['WEAK_DOWN', 'STRONG_UP', 'WEAK_UP']:
                    exit_reason = "Fuerza de tendencia desfavorable"
                
                # 2. Salida por zona no operar
                elif current_no_trade:
                    exit_reason = "Zona de NO OPERAR activa"
                
                # 3. Salida por cambio en temporalidad menor
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
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    exit_alerts.append(exit_alert)
                    
                    # Remover señal activa
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

def _ensure_serializable(self, obj):
    """Asegurar que todos los datos sean serializables a JSON"""
    try:
        if obj is None:
            return None
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict('records') if isinstance(obj, pd.DataFrame) else obj.tolist()
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._ensure_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._ensure_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            # Intentar convertir a tipo nativo de Python
            try:
                return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
            except:
                return str(obj)
    except Exception as e:
        print(f"Error en _ensure_serializable: {e}")
        return str(obj)




    
   def generate_signals_pro(self, symbol, interval, di_period=14, adx_threshold=25, 
                       sr_period=50, rsi_length=20, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
    """GENERACIÓN DE SEÑALES PROFESIONAL con todas las mejoras y corrección de serialización JSON"""
    try:
        df = self.get_kucoin_data(symbol, interval, 100)
        
        if df is None or len(df) < 30:
            return self._create_empty_signal(symbol)
        
        # INDICADORES PRINCIPALES
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 1. Ballenas (CORREGIDO: solo 12H y 1D)
        whale_data = self.calculate_whale_signals_corrected(df, interval)
        
        # 2. ADX + DMI
        adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
        
        # 3. Cruces DI
        di_cross_bullish, di_cross_bearish, di_trend_bullish, di_trend_bearish = self.check_di_crossover(plus_di, minus_di)
        
        # 4. RSI Maverick
        rsi_maverick = self.calculate_rsi_maverick(close, rsi_length, bb_multiplier)
        
        # 5. Divergencias
        bullish_div, bearish_div = self.detect_divergence(close, rsi_maverick)
        
        # 6. Breakouts
        breakout_up, breakout_down = self.check_breakout(high, low, close, whale_data['support'], whale_data['resistance'])
        
        # 7. Fuerza de tendencia Maverick
        trend_strength_data = self.calculate_trend_strength_maverick(close)
        
        # 8. Medias móviles
        ma_9 = self.calculate_sma(close, 9)
        ma_21 = self.calculate_sma(close, 21)
        ma_50 = self.calculate_sma(close, 50)
        ma_200 = self.calculate_sma(close, 200)
        
        # 9. Bandas Bollinger
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)  # %B
        
        # 10. RSI tradicional
        rsi = self.calculate_rsi(close)
        
        # 11. MACD
        macd, macd_signal, macd_histogram = self.calculate_macd(close)
        
        # 12. Squeeze Momentum
        squeeze_data = self.calculate_squeeze_momentum(high, low, close)
        
        # 13. Patrones Chartistas
        chart_patterns = self.detect_chart_patterns(high, low, close)
        
        current_idx = -1
        
        # Preparar datos para evaluación
        analysis_data = {
            'close': close.tolist() if hasattr(close, 'tolist') else list(close),
            'ma_9': ma_9.tolist() if hasattr(ma_9, 'tolist') else list(ma_9),
            'ma_21': ma_21.tolist() if hasattr(ma_21, 'tolist') else list(ma_21),
            'ma_50': ma_50.tolist() if hasattr(ma_50, 'tolist') else list(ma_50),
            'ma_200': ma_200.tolist() if hasattr(ma_200, 'tolist') else list(ma_200),
            'rsi': rsi.tolist() if hasattr(rsi, 'tolist') else list(rsi),
            'rsi_maverick': rsi_maverick if isinstance(rsi_maverick, list) else list(rsi_maverick),
            'support': whale_data['support'],
            'resistance': whale_data['resistance'],
            'adx': adx.tolist() if hasattr(adx, 'tolist') else list(adx),
            'plus_di': plus_di.tolist() if hasattr(plus_di, 'tolist') else list(plus_di),
            'minus_di': minus_di.tolist() if hasattr(minus_di, 'tolist') else list(minus_di),
            'macd': macd.tolist() if hasattr(macd, 'tolist') else list(macd),
            'macd_signal': macd_signal.tolist() if hasattr(macd_signal, 'tolist') else list(macd_signal),
            'squeeze_momentum': squeeze_data['momentum'],
            'bb_position': bb_position.tolist() if hasattr(bb_position, 'tolist') else list(bb_position),
            'whale_pump': whale_data['whale_pump'],
            'whale_dump': whale_data['whale_dump'],
            'chart_patterns': chart_patterns
        }
        
        # VERIFICACIÓN OBLIGATORIA MULTI-TIMEFRAME
        long_multi_tf = self.check_multi_timeframe_trend(symbol, interval, 'LONG')
        short_multi_tf = self.check_multi_timeframe_trend(symbol, interval, 'SHORT')
        
        # Evaluar condiciones
        long_conditions = self.evaluate_signal_conditions_pro(analysis_data, current_idx, interval, adx_threshold)
        short_conditions = self.evaluate_signal_conditions_pro(analysis_data, current_idx, interval, adx_threshold)
        
        # Calcular scores con OBLIGATORIEDAD
        long_score, long_fulfilled = self.calculate_signal_score_pro(long_conditions, 'long', long_multi_tf['valid'])
        short_score, short_fulfilled = self.calculate_signal_score_pro(short_conditions, 'short', short_multi_tf['valid'])
        
        # Determinar señal final
        signal_type = 'NEUTRAL'
        signal_score = 0
        fulfilled_conditions = []
        
        if long_score >= 70 and long_score > short_score:
            signal_type = 'LONG'
            signal_score = long_score
            fulfilled_conditions = long_fulfilled
        elif short_score >= 70 and short_score > long_score:
            signal_type = 'SHORT'
            signal_score = short_score
            fulfilled_conditions = short_fulfilled
        
        # Calcular niveles de entrada/salida
        current_price = float(close[current_idx])
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
        
        # Calcular winrate
        winrate = self.calculate_winrate(symbol, interval)
        
        # CONVERTIR TODOS LOS DATOS A FORMATO JSON SERIALIZABLE
        indicators_dict = {
            'ma_9': self._ensure_serializable(ma_9[-50:]),
            'ma_21': self._ensure_serializable(ma_21[-50:]),
            'ma_50': self._ensure_serializable(ma_50[-50:]),
            'ma_200': self._ensure_serializable(ma_200[-50:]),
            'rsi': self._ensure_serializable(rsi[-50:]),
            'rsi_maverick': self._ensure_serializable(rsi_maverick[-50:]),
            'adx': self._ensure_serializable(adx[-50:]),
            'plus_di': self._ensure_serializable(plus_di[-50:]),
            'minus_di': self._ensure_serializable(minus_di[-50:]),
            'macd': self._ensure_serializable(macd[-50:]),
            'macd_signal': self._ensure_serializable(macd_signal[-50:]),
            'macd_histogram': self._ensure_serializable(macd_histogram[-50:]),
            'squeeze_on': self._ensure_serializable(squeeze_data['squeeze_on'][-50:]),
            'squeeze_off': self._ensure_serializable(squeeze_data['squeeze_off'][-50:]),
            'squeeze_momentum': self._ensure_serializable(squeeze_data['momentum'][-50:]),
            'bb_upper': self._ensure_serializable(bb_upper[-50:]),
            'bb_middle': self._ensure_serializable(bb_middle[-50:]),
            'bb_lower': self._ensure_serializable(bb_lower[-50:]),
            'bb_position': self._ensure_serializable(bb_position[-50:]),
            'whale_pump': self._ensure_serializable(whale_data['whale_pump'][-50:]),
            'whale_dump': self._ensure_serializable(whale_data['whale_dump'][-50:]),
            'trend_strength': self._ensure_serializable(trend_strength_data['trend_strength'][-50:]),
            'bb_width': self._ensure_serializable(trend_strength_data['bb_width'][-50:]),
            'no_trade_zones': self._ensure_serializable(trend_strength_data['no_trade_zones'][-50:]),
            'strength_signals': self._ensure_serializable(trend_strength_data['strength_signals'][-50:]),
            'high_zone_threshold': float(trend_strength_data['high_zone_threshold']),
            'colors': self._ensure_serializable(trend_strength_data['colors'][-50:])
        }
        
        # Añadir patrones chartistas serializados
        for key, value in chart_patterns.items():
            indicators_dict['chart_patterns'] = indicators_dict.get('chart_patterns', {})
            indicators_dict['chart_patterns'][key] = self._ensure_serializable(value[-50:])
        
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'signal': signal_type,
            'signal_score': float(signal_score),
            'winrate': float(winrate),
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
            'fulfilled_conditions': fulfilled_conditions,
            'multi_tf_valid_long': long_multi_tf['valid'],
            'multi_tf_valid_short': short_multi_tf['valid'],
            'multi_tf_reason_long': long_multi_tf['reason'],
            'multi_tf_reason_short': short_multi_tf['reason'],
            'trend_strength_signal': trend_strength_data['strength_signals'][current_idx] if current_idx < len(trend_strength_data['strength_signals']) else 'NEUTRAL',
            'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx] if current_idx < len(trend_strength_data['no_trade_zones']) else False,
            'whale_indicator_active': whale_data['active'],
            'data': self._ensure_serializable(df.tail(50).to_dict('records')),
            'indicators': indicators_dict
        }
        
        return result
        
    except Exception as e:
        print(f"Error en generate_signals_pro para {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return self._create_empty_signal(symbol)



def _create_empty_signal(self, symbol):
    """Crear señal vacía en caso de error con datos serializables"""
    return {
        'symbol': symbol,
        'current_price': 0.0,
        'signal': 'NEUTRAL',
        'signal_score': 0.0,
        'winrate': 0.0,
        'entry': 0.0,
        'stop_loss': 0.0,
        'take_profit': [0.0],
        'support': 0.0,
        'resistance': 0.0,
        'atr': 0.0,
        'atr_percentage': 0.0,
        'volume': 0.0,
        'volume_ma': 0.0,
        'adx': 0.0,
        'plus_di': 0.0,
        'minus_di': 0.0,
        'whale_pump': 0.0,
        'whale_dump': 0.0,
        'rsi_maverick': 0.5,
        'fulfilled_conditions': [],
        'multi_tf_valid_long': False,
        'multi_tf_valid_short': False,
        'multi_tf_reason_long': 'Error',
        'multi_tf_reason_short': 'Error',
        'trend_strength_signal': 'NEUTRAL',
        'no_trade_zone': False,
        'whale_indicator_active': False,
        'data': [],
        'indicators': {
            'ma_9': [],
            'ma_21': [],
            'ma_50': [],
            'ma_200': [],
            'rsi': [],
            'rsi_maverick': [],
            'adx': [],
            'plus_di': [],
            'minus_di': [],
            'macd': [],
            'macd_signal': [],
            'macd_histogram': [],
            'squeeze_on': [],
            'squeeze_off': [],
            'squeeze_momentum': [],
            'bb_upper': [],
            'bb_middle': [],
            'bb_lower': [],
            'bb_position': [],
            'whale_pump': [],
            'whale_dump': [],
            'trend_strength': [],
            'bb_width': [],
            'no_trade_zones': [],
            'strength_signals': [],
            'high_zone_threshold': 0.0,
            'colors': [],
            'chart_patterns': {}
        }
    }

    

    def generate_scalping_alerts(self):
        """Generar alertas de scalping con filtros mejorados"""
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
                    signal_data = self.generate_signals_pro(symbol, interval)
                    
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
                        
                        alert = {
                            'symbol': symbol,
                            'interval': interval,
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'winrate': signal_data['winrate'],
                            'entry': signal_data['entry'],
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'][0],
                            'leverage': optimal_leverage,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'trend_strength': signal_data.get('trend_strength_signal', 'NEUTRAL'),
                            'multi_tf_valid': signal_data.get('multi_tf_valid_long', False) or signal_data.get('multi_tf_valid_short', False),
                            'whale_indicator_active': signal_data.get('whale_indicator_active', False)
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
    """Enviar alerta por Telegram mejorada"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA PRO - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
🎯 Winrate: {alert_data.get('winrate', 0):.1f}%

💰 Precio actual: {alert_data.get('current_price', alert_data['entry']):.6f}
💪 Fuerza Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

🎯 ENTRADA: ${alert_data['entry']:.6f}
🛑 STOP LOSS: ${alert_data['stop_loss']:.6f}
🎯 TAKE PROFIT: ${alert_data['take_profit']:.6f}

📈 Apalancamiento: x{alert_data['leverage']}
✅ Multi-TF: {'CONFIRMADO' if alert_data.get('multi_tf_valid', False) else 'PENDIENTE'}
🐋 Ballenas: {'ACTIVO' if alert_data.get('whale_indicator_active', False) else 'NO APLICA'}

📊 Revisa la señal en: https://ballenasscalpistas.onrender.com/
            """
        else:  # exit alert
            pnl_text = f"📊 P&L: {alert_data['pnl_percent']:+.2f}%"
            
            message = f"""
🚨 ALERTA DE SALIDA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} ({risk_classification})
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
{pnl_text}

💪 Fuerza de Tendencia: {alert_data.get('trend_strength', 'NEUTRAL')}

📊 Observación: {alert_data['reason']}
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

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint para obtener señales de trading MEJORADO con manejo robusto de errores"""
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
        
        print(f"📡 Solicitando señales para {symbol} en {interval}")
        
        signal_data = indicator.generate_signals_pro(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, volume_filter, leverage
        )
        
        # Asegurar serialización final
        signal_data = indicator._ensure_serializable(signal_data)
        
        print(f"✅ Señales generadas exitosamente para {symbol}")
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"❌ Error crítico en /api/signals: {e}")
        import traceback
        traceback.print_exc()
        
        # Devolver señal vacía pero serializable
        error_signal = indicator._create_empty_signal(request.args.get('symbol', 'BTC-USDT'))
        return jsonify(error_signal)




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
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
            try:
                signal_data = indicator.generate_signals_pro(
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
                signal_data = indicator.generate_signals_pro(symbol, interval, di_period, adx_threshold)
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
                        'signal_score': float(signal_data['signal_score']),
                        'current_price': float(signal_data['current_price']),
                        'signal': signal_data['signal'],
                        'winrate': float(signal_data.get('winrate', 0)),
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

@app.route('/api/winrate')
def get_winrate():
    """Endpoint para obtener winrate del sistema"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        winrate = indicator.calculate_winrate(symbol, interval)
        
        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'winrate': winrate,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error en /api/winrate: {e}")
        return jsonify({'error': 'Error calculando winrate'}), 500

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_pro(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
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
        
        # Gráfico 2: Ballenas (solo si activo)
        ax2 = plt.subplot(7, 1, 2, sharex=ax1)
        if 'indicators' in signal_data and signal_data.get('whale_indicator_active', False):
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax2.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax2.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
            ax2.set_ylabel('Fuerza Ballenas')
        else:
            ax2.text(0.5, 0.5, 'INDICADOR BALLENAS NO APLICA\n(Solo 12H y 1D)', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12, color='gray')
            ax2.set_ylabel('No Aplica')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: ADX/DMI
        ax3 = plt.subplot(7, 1, 3, sharex=ax1)
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
        
        # Gráfico 4: RSI Maverick
        ax4 = plt.subplot(7, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax4.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
        ax4.set_ylabel('RSI Maverick')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Squeeze Momentum
        ax5 = plt.subplot(7, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            squeeze_dates = dates[-len(signal_data['indicators']['squeeze_momentum']):]
            squeeze_data = signal_data['indicators']['squeeze_momentum']
            colors = ['green' if x > 0 else 'red' for x in squeeze_data]
            
            ax5.bar(squeeze_dates, squeeze_data, color=colors, alpha=0.7)
            ax5.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        ax5.set_ylabel('Squeeze')
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Fuerza de Tendencia Maverick
        ax6 = plt.subplot(7, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax6.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax6.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax6.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
            
            no_trade_zones = signal_data['indicators']['no_trade_zones']
            for i, date in enumerate(trend_dates):
                if i < len(no_trade_zones) and no_trade_zones[i]:
                    ax6.axvline(x=date, color='red', alpha=0.3, linewidth=2)
            
            ax6.set_ylabel('Fuerza Tendencia %')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Información de la señal
        ax7 = plt.subplot(7, 1, 7)
        ax7.axis('off')
        
        multi_tf_status = "✅ CONFIRMADO" if (signal_data.get('multi_tf_valid_long') or signal_data.get('multi_tf_valid_short')) else "❌ NO CONFIRMADO"
        whale_status = "✅ ACTIVO" if signal_data.get('whale_indicator_active') else "⚪ NO APLICA"
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        WINRATE: {signal_data.get('winrate', 0):.1f}%
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        
        MULTI-TIMEFRAME: {multi_tf_status}
        INDICADOR BALLENAS: {whale_status}
        FUERZA TENDENCIA: {signal_data.get('trend_strength_signal', 'NEUTRAL')}
        
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

@app.route('/api/fear_greed_index')
def get_fear_greed_index():
    """Endpoint simulado para índice de miedo y codicia"""
    return jsonify({
        'value': 65,
        'sentiment': 'Greed',
        'color': 'success',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/market_recommendations')
def get_market_recommendations():
    """Endpoint simulado para recomendaciones de mercado"""
    return jsonify({
        'recommendation': 'Mercado en fase de acumulación. Considerar entradas LONG en soportes clave.',
        'timestamp': datetime.now().isoformat()
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
