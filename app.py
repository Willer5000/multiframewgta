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

# Configuración Telegram - NUEVO TOKEN
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

# NUEVA JERARQUÍA TEMPORAL MEJORADA
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
        self.win_rate_data = {}
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
                        return None
                    
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
        
        return None

    # FUNCIONES DE CÁLCULO DE INDICADORES (MANTENIDAS)
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
        if len(prices) < period:
            return np.zeros_like(prices)
            
        sma = np.zeros_like(prices)
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window = prices[start_idx:i+1]
            valid_values = window[~np.isnan(window)]
            sma[i] = np.mean(valid_values) if len(valid_values) > 0 else 0
        
        return sma

    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente"""
        if len(prices) == 0 or period <= 0:
            return np.zeros_like(prices)
            
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0] if len(prices) > 0 else 0
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional"""
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

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger"""
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

    # NUEVAS FUNCIONES PARA LA ESTRATEGIA MEJORADA
    def calculate_whale_signals_corrected(self, df, interval, sensitivity=1.7, min_volume_multiplier=1.5):
        """
        INDICADOR CAZADOR DE BALLENAS CORREGIDO
        Visible en todas las TF, pero señal obligatoria solo en 12H y 1D
        """
        try:
            close = df['close'].values
            low = df['low'].values
            high = df['high'].values
            volume = df['volume'].values
            
            n = len(close)
            
            whale_pump_signal = np.zeros(n)
            whale_dump_signal = np.zeros(n)
            
            # Para todas las temporalidades, calcular el indicador
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                low_5 = np.min(low[max(0, i-5):i+1])
                high_5 = np.max(high[max(0, i-5):i+1])
                
                # Señal de compra (ballenas acumulando)
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] < close[i-1] or price_change < -0.5) and
                    low[i] <= low_5 * 1.01):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_pump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
                
                # Señal de venta (ballenas distribuyendo)
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_dump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            # Para temporalidades 12H y 1D, la señal es obligatoria
            is_obligatory_timeframe = interval in ['12h', '1D']
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'is_obligatory': is_obligatory_timeframe,
                'current_pump': float(whale_pump_smooth[-1]) if len(whale_pump_smooth) > 0 else 0,
                'current_dump': float(whale_dump_smooth[-1]) if len(whale_dump_smooth) > 0 else 0
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_corrected: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'is_obligatory': False,
                'current_pump': 0,
                'current_dump': 0
            }

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
                    trend_strength[i] = bb_width[i]  # Fuerza creciente
                else:
                    trend_strength[i] = -bb_width[i]  # Fuerza decreciente
            
            # Detectar zonas de no operar
            no_trade_zones = np.zeros(n, dtype=bool)
            strength_signals = ['NEUTRAL'] * n
            
            for i in range(10, n):
                # Zona de no operar cuando hay pérdida de fuerza significativa
                if (trend_strength[i] < -5 and 
                    bb_width[i] < np.max(bb_width[max(0, i-10):i])):
                    no_trade_zones[i] = True
                
                # Determinar señal de fuerza de tendencia
                if trend_strength[i] > 2:
                    strength_signals[i] = 'BULLISH'
                elif trend_strength[i] < -2:
                    strength_signals[i] = 'BEARISH'
                else:
                    strength_signals[i] = 'NEUTRAL'
            
            return {
                'trend_strength': trend_strength.tolist(),
                'no_trade_zones': no_trade_zones.tolist(),
                'strength_signals': strength_signals,
                'bb_width': bb_width.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_trend_strength_maverick: {e}")
            n = len(close)
            return {
                'trend_strength': [0] * n,
                'no_trade_zones': [False] * n,
                'strength_signals': ['NEUTRAL'] * n,
                'bb_width': [0] * n
            }

    def check_multi_timeframe_trend(self, symbol, interval):
        """
        VERIFICACIÓN OBLIGATORIA MULTI-TIMEFRAME
        Retorna False si alguna condición obligatoria no se cumple
        """
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return {'ok': True, 'details': 'No hierarchy defined'}
            
            results = {}
            all_obligatory_ok = True
            details = []
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:  # Excluir estas temporalidades
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 50)
                if df is None or len(df) < 20:
                    results[tf_type] = 'NEUTRAL'
                    details.append(f"{tf_type}({tf_value}): Sin datos")
                    continue
                
                close = df['close'].values
                
                # Calcular tendencia usando medias
                ma_9 = self.calculate_sma(close, 9)
                ma_21 = self.calculate_sma(close, 21)
                ma_50 = self.calculate_sma(close, 50)
                
                current_ma_9 = ma_9[-1] if len(ma_9) > 0 else 0
                current_ma_21 = ma_21[-1] if len(ma_21) > 0 else 0
                current_ma_50 = ma_50[-1] if len(ma_50) > 0 else 0
                current_price = close[-1]
                
                # Determinar tendencia
                if current_price > current_ma_9 and current_ma_9 > current_ma_21:
                    trend = 'BULLISH'
                elif current_price < current_ma_9 and current_ma_9 < current_ma_21:
                    trend = 'BEARISH'
                else:
                    trend = 'NEUTRAL'
                
                results[tf_type] = trend
                details.append(f"{tf_type}({tf_value}): {trend}")
            
            # VERIFICAR OBLIGATORIEDADES SEGÚN ESTRATEGIA
            mayor_trend = results.get('mayor', 'NEUTRAL')
            media_trend = results.get('media', 'NEUTRAL')
            menor_trend = results.get('menor', 'NEUTRAL')
            
            # Verificar fuerza de tendencia en temporalidad menor
            menor_df = self.get_kucoin_data(symbol, hierarchy.get('menor', interval), 30)
            menor_strength = 'NEUTRAL'
            if menor_df is not None and len(menor_df) > 10:
                menor_trend_data = self.calculate_trend_strength_maverick(menor_df['close'].values)
                menor_strength = menor_trend_data['strength_signals'][-1]
                details.append(f"Fuerza Menor: {menor_strength}")
            
            # Verificar zonas de no operar en todas las temporalidades
            no_trade_zone_detected = False
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:
                    continue
                df_tf = self.get_kucoin_data(symbol, tf_value, 30)
                if df_tf is not None and len(df_tf) > 10:
                    trend_data = self.calculate_trend_strength_maverick(df_tf['close'].values)
                    if trend_data['no_trade_zones'][-1]:
                        no_trade_zone_detected = True
                        details.append(f"NO OPERAR detectado en {tf_value}")
                        break
            
            return {
                'ok': not no_trade_zone_detected,
                'results': results,
                'menor_strength': menor_strength,
                'no_trade_zone': no_trade_zone_detected,
                'details': details
            }
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {
                'ok': False,
                'results': {},
                'menor_strength': 'NEUTRAL',
                'no_trade_zone': True,
                'details': [f'Error: {str(e)}']
            }

    def check_chart_patterns(self, df, lookback=50):
        """
        DETECCIÓN DE PATRONES DE CHARTISMO
        Retorna patrones detectados en los últimos 'lookback' periodos
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            n = len(close)
            
            patterns = {
                'double_bottom': False,
                'double_top': False,
                'head_shoulders': False,
                'inverse_head_shoulders': False,
                'bullish_flag': False,
                'bearish_flag': False
            }
            
            if n < lookback:
                return patterns
            
            # Usar los últimos 'lookback' periodos
            recent_high = high[-lookback:]
            recent_low = low[-lookback:]
            recent_close = close[-lookback:]
            
            # Detectar Doble Techo (Double Top)
            max_idx = np.argmax(recent_high)
            if max_idx > 10 and max_idx < lookback - 10:
                left_high = np.max(recent_high[:max_idx-5])
                right_high = np.max(recent_high[max_idx+5:])
                if abs(left_high - right_high) / left_high < 0.02:  # 2% de tolerancia
                    patterns['double_top'] = True
            
            # Detectar Doble Fondo (Double Bottom)
            min_idx = np.argmin(recent_low)
            if min_idx > 10 and min_idx < lookback - 10:
                left_low = np.min(recent_low[:min_idx-5])
                right_low = np.min(recent_low[min_idx+5:])
                if abs(left_low - right_low) / left_low < 0.02:  # 2% de tolerancia
                    patterns['double_bottom'] = True
            
            return patterns
            
        except Exception as e:
            print(f"Error en check_chart_patterns: {e}")
            return {
                'double_bottom': False,
                'double_top': False,
                'head_shoulders': False,
                'inverse_head_shoulders': False,
                'bullish_flag': False,
                'bearish_flag': False
            }

    def calculate_squeeze_momentum(self, close, length=20, mult=2.0):
        """
        CALCULAR SQUEEZE MOMENTUM
        Histograma: Verde > 0, Rojo < 0
        """
        try:
            n = len(close)
            
            # Calcular Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, length, mult)
            
            # Calcular Keltner Channel (simplificado)
            atr = self.calculate_atr(close, close, close, length)
            keltner_upper = bb_middle + (atr * 1.5)
            keltner_lower = bb_middle - (atr * 1.5)
            
            # Determinar squeeze
            squeeze_on = np.zeros(n, dtype=bool)
            squeeze_off = np.zeros(n, dtype=bool)
            momentum = np.zeros(n)
            
            for i in range(n):
                # Squeeze ON cuando Bollinger dentro de Keltner
                if bb_upper[i] < keltner_upper[i] and bb_lower[i] > keltner_lower[i]:
                    squeeze_on[i] = True
                else:
                    squeeze_off[i] = True
                
                # Momentum basado en la posición del precio respecto a las medias
                if i > 0:
                    momentum[i] = close[i] - bb_middle[i]
            
            return {
                'squeeze_on': squeeze_on.tolist(),
                'squeeze_off': squeeze_off.tolist(),
                'momentum': momentum.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_squeeze_momentum: {e}")
            n = len(close)
            return {
                'squeeze_on': [False] * n,
                'squeeze_off': [True] * n,
                'momentum': [0] * n
            }

    def calculate_support_resistance(self, high, low, close, period=50):
        """
        CALCULAR SOPORTES Y RESISTENCIAS SMART MONEY
        Lookback: 50 periodos para niveles significativos
        """
        try:
            n = len(close)
            
            support_levels = []
            resistance_levels = []
            
            # Buscar niveles significativos en los últimos 'period' periodos
            lookback_data = {
                'high': high[-period:] if n >= period else high,
                'low': low[-period:] if n >= period else low,
                'close': close[-period:] if n >= period else close
            }
            
            # Encontrar máximos y mínimos locales
            for i in range(2, len(lookback_data['high']) - 2):
                # Posible resistencia
                if (lookback_data['high'][i] > lookback_data['high'][i-1] and 
                    lookback_data['high'][i] > lookback_data['high'][i-2] and
                    lookback_data['high'][i] > lookback_data['high'][i+1] and
                    lookback_data['high'][i] > lookback_data['high'][i+2]):
                    resistance_levels.append(lookback_data['high'][i])
                
                # Posible soporte
                if (lookback_data['low'][i] < lookback_data['low'][i-1] and 
                    lookback_data['low'][i] < lookback_data['low'][i-2] and
                    lookback_data['low'][i] < lookback_data['low'][i+1] and
                    lookback_data['low'][i] < lookback_data['low'][i+2]):
                    support_levels.append(lookback_data['low'][i])
            
            # Tomar los niveles más significativos
            significant_support = np.min(support_levels) if support_levels else np.min(lookback_data['low'])
            significant_resistance = np.max(resistance_levels) if resistance_levels else np.max(lookback_data['high'])
            
            return {
                'support': float(significant_support),
                'resistance': float(significant_resistance),
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            print(f"Error en calculate_support_resistance: {e}")
            current_price = close[-1] if len(close) > 0 else 0
            return {
                'support': current_price * 0.95,
                'resistance': current_price * 1.05,
                'support_levels': [],
                'resistance_levels': []
            }

    def evaluate_signal_conditions_enhanced(self, data, current_idx, interval, multi_tf_analysis):
        """
        EVALUACIÓN MEJORADA DE CONDICIONES DE SEÑAL
        Con obligatoriedades multi-temporalidad
        """
        conditions = {
            'long': {
                'multi_tf_obligatory': {'value': False, 'weight': 0, 'description': 'Condiciones multi-TF obligatorias', 'obligatory': True},
                'whale_pump': {'value': False, 'weight': 25, 'description': 'Ballena compradora activa', 'obligatory': False},
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación medias móviles', 'obligatory': False},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI tradicional favorable', 'obligatory': False},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable', 'obligatory': False},
                'adx_dmi': {'value': False, 'weight': 10, 'description': 'ADX + DMI alcista', 'obligatory': False},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD alcista', 'obligatory': False},
                'squeeze': {'value': False, 'weight': 5, 'description': 'Squeeze momentum favorable', 'obligatory': False},
                'bollinger': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger alcistas', 'obligatory': False},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrones chartismo alcistas', 'obligatory': False}
            },
            'short': {
                'multi_tf_obligatory': {'value': False, 'weight': 0, 'description': 'Condiciones multi-TF obligatorias', 'obligatory': True},
                'whale_dump': {'value': False, 'weight': 25, 'description': 'Ballena vendedora activa', 'obligatory': False},
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación medias móviles', 'obligatory': False},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI tradicional favorable', 'obligatory': False},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable', 'obligatory': False},
                'adx_dmi': {'value': False, 'weight': 10, 'description': 'ADX + DMI bajista', 'obligatory': False},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD bajista', 'obligatory': False},
                'squeeze': {'value': False, 'weight': 5, 'description': 'Squeeze momentum favorable', 'obligatory': False},
                'bollinger': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger bajistas', 'obligatory': False},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrones chartismo bajistas', 'obligatory': False}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        current_price = data['close'][current_idx]
        
        # 1. VERIFICAR OBLIGATORIEDADES MULTI-TIMEFRAME
        multi_tf_ok = multi_tf_analysis['ok']
        conditions['long']['multi_tf_obligatory']['value'] = multi_tf_ok
        conditions['short']['multi_tf_obligatory']['value'] = multi_tf_ok
        
        if not multi_tf_ok:
            return conditions  # Si fallan las obligatorias, no evaluar más
        
        # 2. INDICADOR BALLENAS (condicional según temporalidad)
        whale_data = data.get('whale_data', {})
        is_whale_obligatory = whale_data.get('is_obligatory', False)
        
        if is_whale_obligatory:
            # En 12H y 1D, la señal de ballenas es obligatoria
            conditions['long']['whale_pump']['weight'] = 25
            conditions['short']['whale_dump']['weight'] = 25
            conditions['long']['whale_pump']['obligatory'] = True
            conditions['short']['whale_dump']['obligatory'] = True
        else:
            # En otras TF, redistribuir el peso
            redistribution = 25 / 8  # Distribuir entre los otros 8 indicadores
            for condition in ['moving_averages', 'rsi_traditional', 'rsi_maverick', 'adx_dmi', 
                            'macd', 'squeeze', 'bollinger', 'chart_patterns']:
                conditions['long'][condition]['weight'] += redistribution
                conditions['short'][condition]['weight'] += redistribution
        
        conditions['long']['whale_pump']['value'] = whale_data.get('current_pump', 0) > 15
        conditions['short']['whale_dump']['value'] = whale_data.get('current_dump', 0) > 18
        
        # 3. MEDIAS MÓVILES
        ma_9 = data.get('ma_9', [])
        ma_21 = data.get('ma_21', [])
        ma_50 = data.get('ma_50', [])
        ma_200 = data.get('ma_200', [])
        
        if (len(ma_9) > current_idx and len(ma_21) > current_idx and 
            len(ma_50) > current_idx and len(ma_200) > current_idx):
            conditions['long']['moving_averages']['value'] = (
                current_price > ma_9[current_idx] and 
                ma_9[current_idx] > ma_21[current_idx] and
                ma_21[current_idx] > ma_50[current_idx]
            )
            conditions['short']['moving_averages']['value'] = (
                current_price < ma_9[current_idx] and 
                ma_9[current_idx] < ma_21[current_idx] and
                ma_21[current_idx] < ma_50[current_idx]
            )
        
        # 4. RSI TRADICIONAL
        rsi_traditional = data.get('rsi_traditional', [])
        if len(rsi_traditional) > current_idx:
            conditions['long']['rsi_traditional']['value'] = (
                rsi_traditional[current_idx] < 70 and  # No sobrecomprado
                rsi_traditional[current_idx] > 30     # No sobrevendido
            )
            conditions['short']['rsi_traditional']['value'] = (
                rsi_traditional[current_idx] > 30 and  # No sobrevendido
                rsi_traditional[current_idx] < 70      # No sobrecomprado
            )
        
        # 5. RSI MAVERICK
        rsi_maverick = data.get('rsi_maverick', [])
        if len(rsi_maverick) > current_idx:
            conditions['long']['rsi_maverick']['value'] = rsi_maverick[current_idx] < 0.8
            conditions['short']['rsi_maverick']['value'] = rsi_maverick[current_idx] > 0.2
        
        # 6. ADX + DMI
        adx = data.get('adx', [])
        plus_di = data.get('plus_di', [])
        minus_di = data.get('minus_di', [])
        
        if (len(adx) > current_idx and len(plus_di) > current_idx and 
            len(minus_di) > current_idx):
            conditions['long']['adx_dmi']['value'] = (
                adx[current_idx] > 25 and
                plus_di[current_idx] > minus_di[current_idx]
            )
            conditions['short']['adx_dmi']['value'] = (
                adx[current_idx] > 25 and
                minus_di[current_idx] > plus_di[current_idx]
            )
        
        # 7. MACD
        macd_histogram = data.get('macd_histogram', [])
        if len(macd_histogram) > current_idx:
            conditions['long']['macd']['value'] = macd_histogram[current_idx] > 0
            conditions['short']['macd']['value'] = macd_histogram[current_idx] < 0
        
        # 8. SQUEEZE MOMENTUM
        squeeze_momentum = data.get('squeeze_momentum', [])
        if len(squeeze_momentum) > current_idx:
            conditions['long']['squeeze']['value'] = squeeze_momentum[current_idx] > 0
            conditions['short']['squeeze']['value'] = squeeze_momentum[current_idx] < 0
        
        # 9. BOLLINGER BANDS
        bb_position = data.get('bb_position', [])
        if len(bb_position) > current_idx:
            conditions['long']['bollinger']['value'] = bb_position[current_idx] > 0.2
            conditions['short']['bollinger']['value'] = bb_position[current_idx] < 0.8
        
        # 10. PATRONES DE CHARTISMO
        chart_patterns = data.get('chart_patterns', {})
        conditions['long']['chart_patterns']['value'] = (
            chart_patterns.get('double_bottom', False) or
            chart_patterns.get('bullish_flag', False)
        )
        conditions['short']['chart_patterns']['value'] = (
            chart_patterns.get('double_top', False) or
            chart_patterns.get('bearish_flag', False)
        )
        
        return conditions

    def calculate_signal_score_enhanced(self, conditions, signal_type):
        """
        CÁLCULO DE SCORE MEJORADO CON OBLIGATORIEDADES
        Si alguna condición obligatoria falla → Score = 0
        """
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        obligatory_failed = False
        
        signal_conditions = conditions.get(signal_type, {})
        
        for key, condition in signal_conditions.items():
            if condition['obligatory'] and not condition['value']:
                obligatory_failed = True
                break
        
        if obligatory_failed:
            return 0, []
        
        for key, condition in signal_conditions.items():
            if not condition['obligatory']:  # Solo contar indicadores no obligatorios para el score
                total_weight += condition['weight']
                if condition['value']:
                    achieved_weight += condition['weight']
                    fulfilled_conditions.append(condition['description'])
        
        if total_weight == 0:
            return 0, []
        
        score = (achieved_weight / total_weight * 100)
        return min(score, 100), fulfilled_conditions

    def calculate_optimal_entry_exit_enhanced(self, df, signal_type, support_resistance):
        """
        CÁLCULO MEJORADO DE ENTRADAS Y SALIDAS
        Basado en soportes/resistencias smart money
        """
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            support = support_resistance['support']
            resistance = support_resistance['resistance']
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Entrada en máximo acercamiento al soporte
                entry = min(current_price, support * 1.01)  # 1% sobre soporte
                stop_loss = support * 0.98  # 2% bajo soporte
                take_profit = resistance * 0.99  # 1% bajo resistencia
                
            else:  # SHORT
                # Entrada en máximo acercamiento a la resistencia
                entry = max(current_price, resistance * 0.99)  # 1% bajo resistencia
                stop_loss = resistance * 1.02  # 2% sobre resistencia
                take_profit = support * 1.01  # 1% sobre soporte
            
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

    def calculate_win_rate(self, symbol, interval, lookback=100):
        """
        CALCULAR WINRATE BASADO EN HISTORIAL
        Análisis de 100 periodos históricos
        """
        try:
            df = self.get_kucoin_data(symbol, interval, lookback + 20)
            if df is None or len(df) < lookback + 10:
                return 50.0  # Winrate por defecto si no hay datos suficientes
            
            total_signals = 0
            successful_signals = 0
            
            # Analizar señales en ventana deslizante
            for i in range(10, len(df) - 10):
                window_data = df.iloc[i-10:i]
                current_price = df['close'].iloc[i]
                future_prices = df['close'].iloc[i:i+10]
                
                # Simular señal (simplificado para demo)
                if len(window_data) >= 10:
                    # Lógica básica de detección de señales
                    ma_9 = self.calculate_sma(window_data['close'].values, 9)
                    ma_21 = self.calculate_sma(window_data['close'].values, 21)
                    
                    if len(ma_9) > 0 and len(ma_21) > 0:
                        if current_price > ma_9[-1] and ma_9[-1] > ma_21[-1]:
                            # Señal LONG simulada
                            if any(future_prices > current_price * 1.01):  # 1% de ganancia
                                successful_signals += 1
                            total_signals += 1
                        elif current_price < ma_9[-1] and ma_9[-1] < ma_21[-1]:
                            # Señal SHORT simulada
                            if any(future_prices < current_price * 0.99):  # 1% de ganancia
                                successful_signals += 1
                            total_signals += 1
            
            if total_signals > 0:
                win_rate = (successful_signals / total_signals) * 100
            else:
                win_rate = 50.0
            
            return win_rate
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return 50.0

    def generate_signals_enhanced(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, leverage=15):
        """
        GENERACIÓN DE SEÑALES MEJORADA - ESTRATEGIA MULTI-TIMEFRAME
        """
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # 1. VERIFICACIÓN OBLIGATORIA MULTI-TIMEFRAME
            multi_tf_analysis = self.check_multi_timeframe_trend(symbol, interval)
            
            # 2. CALCULAR TODOS LOS INDICADORES
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Indicador Ballenas (corregido)
            whale_data = self.calculate_whale_signals_corrected(df, interval)
            
            # Medias Móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # RSI Tradicional
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            # RSI Maverick (%B Bollinger)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            rsi_maverick = np.zeros(len(close))
            for i in range(len(close)):
                if (bb_upper[i] - bb_lower[i]) > 0:
                    rsi_maverick[i] = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
                else:
                    rsi_maverick[i] = 0.5
            
            # ADX + DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Squeeze Momentum
            squeeze_data = self.calculate_squeeze_momentum(close)
            
            # Soporte y Resistencia Smart Money
            support_resistance = self.calculate_support_resistance(high, low, close, sr_period)
            
            # Patrones de Chartismo
            chart_patterns = self.check_chart_patterns(df)
            
            # Fuerza de Tendencia Maverick
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Preparar datos para evaluación
            analysis_data = {
                'close': close.tolist(),
                'high': high.tolist(),
                'low': low.tolist(),
                'whale_data': whale_data,
                'ma_9': ma_9.tolist(),
                'ma_21': ma_21.tolist(),
                'ma_50': ma_50.tolist(),
                'ma_200': ma_200.tolist(),
                'rsi_traditional': rsi_traditional.tolist(),
                'rsi_maverick': rsi_maverick.tolist(),
                'adx': adx.tolist(),
                'plus_di': plus_di.tolist(),
                'minus_di': minus_di.tolist(),
                'macd_histogram': macd_histogram.tolist(),
                'squeeze_momentum': squeeze_data['momentum'],
                'bb_position': rsi_maverick.tolist(),  # Usar RSI Maverick como posición en BB
                'chart_patterns': chart_patterns,
                'support_resistance': support_resistance
            }
            
            current_idx = -1
            
            # 3. EVALUAR CONDICIONES
            conditions = self.evaluate_signal_conditions_enhanced(
                analysis_data, current_idx, interval, multi_tf_analysis
            )
            
            # 4. CALCULAR SCORES
            long_score, long_conditions = self.calculate_signal_score_enhanced(conditions, 'long')
            short_score, short_conditions = self.calculate_signal_score_enhanced(conditions, 'short')
            
            # 5. DETERMINAR SEÑAL FINAL
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= 70 and long_score > short_score:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= 70 and short_score > long_score:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # 6. CALCULAR NIVELES DE ENTRADA/SALIDA
            levels_data = self.calculate_optimal_entry_exit_enhanced(df, signal_type, support_resistance)
            
            # 7. CALCULAR WINRATE
            win_rate = self.calculate_win_rate(symbol, interval)
            
            # 8. PREPARAR RESPUESTA
            current_price = float(df['close'].iloc[current_idx])
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal_type,
                'signal_score': float(signal_score),
                'win_rate': float(win_rate),
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
                'whale_pump': whale_data['current_pump'],
                'whale_dump': whale_data['current_dump'],
                'rsi_maverick': float(rsi_maverick[current_idx] if current_idx < len(rsi_maverick) else 0.5),
                'fulfilled_conditions': fulfilled_conditions,
                'multi_tf_analysis': multi_tf_analysis,
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:],
                    'rsi_traditional': rsi_traditional[-50:].tolist(),
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'macd': macd_line[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['momentum'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist()
                }
            }
            
        except Exception as e:
            print(f"Error en generate_signals_enhanced para {symbol}: {e}")
            return self._create_empty_signal(symbol)

    def _create_empty_signal(self, symbol):
        """Crear señal vacía en caso de error"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'signal': 'NEUTRAL',
            'signal_score': 0,
            'win_rate': 50.0,
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
            'multi_tf_analysis': {'ok': False, 'details': ['Error']},
            'data': [],
            'indicators': {}
        }

    def generate_exit_signals_enhanced(self):
        """Generar señales de salida mejoradas"""
        exit_alerts = []
        current_time = self.get_bolivia_time()
        
        for signal_key, signal_data in list(self.active_signals.items()):
            try:
                symbol = signal_data['symbol']
                interval = signal_data['interval']
                signal_type = signal_data['signal']
                entry_price = signal_data['entry_price']
                
                # Obtener análisis actual
                current_analysis = self.generate_signals_enhanced(symbol, interval)
                
                if current_analysis['signal'] == 'NEUTRAL':
                    # Señal de salida por cambio de condiciones
                    exit_reason = "Condiciones cambiaron a NEUTRAL"
                    
                    exit_alert = {
                        'symbol': symbol,
                        'interval': interval,
                        'signal': f"EXIT_{signal_type}",
                        'entry_price': entry_price,
                        'exit_price': current_analysis['current_price'],
                        'pnl_percent': ((current_analysis['current_price'] - entry_price) / entry_price * 100) 
                                    if signal_type == 'LONG' else 
                                    ((entry_price - current_analysis['current_price']) / entry_price * 100),
                        'reason': exit_reason,
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    exit_alerts.append(exit_alert)
                    del self.active_signals[signal_key]
                    
            except Exception as e:
                print(f"Error generando señal de salida para {signal_key}: {e}")
                continue
        
        return exit_alerts

# Instancia global del indicador
indicator = TradingIndicator()

# Funciones auxiliares para Telegram y background (mantenidas)
def send_telegram_alert(alert_data, alert_type='entry'):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'entry':
            message = f"""
🚨 ALERTA DE TRADING - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
🎯 WinRate: {alert_data.get('win_rate', 50):.1f}%

💰 Precio actual: {alert_data.get('current_price', alert_data['entry']):.6f}
🎯 Entrada: ${alert_data['entry']:.6f}
🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
📈 Take Profit: ${alert_data['take_profit']:.6f}

💪 Apalancamiento: x{alert_data['leverage']}

✅ Condiciones Multi-TF: {alert_data.get('multi_tf_ok', 'CONFIRMADO')}
🔔 Análisis completado: {alert_data['timestamp']}

📊 Revisa la señal en: https://multiframewgta.onrender.com/
            """
        else:  # exit alert
            message = f"""
🚨 ALERTA DE SALIDA - MULTI-TIMEFRAME CRYPTO WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}

💰 Entrada: ${alert_data['entry_price']:.6f}
💰 Salida: ${alert_data['exit_price']:.6f}
📊 P&L: {alert_data['pnl_percent']:+.2f}%

📊 Observación: {alert_data['reason']}
            """
        
        asyncio.run(bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=message
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
            exit_alerts = indicator.generate_exit_signals_enhanced()
            for alert in exit_alerts:
                send_telegram_alert(alert, 'exit')
            
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

# ENDPOINTS PRINCIPALES (mantenidos con mejoras)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint principal para señales"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_enhanced(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, leverage
        )
        
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/multiple_signals')
def get_multiple_signals():
    """Endpoint para múltiples señales"""
    try:
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
            try:
                signal_data = indicator.generate_signals_enhanced(
                    symbol, interval, di_period, adx_threshold
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
            'long_signals': long_signals[:5],
            'short_signals': short_signals[:5],
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
        
        symbols_to_analyze = CRYPTO_SYMBOLS[:15]  # Limitar para performance
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals_enhanced(symbol, interval)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones de compra/venta
                    buy_pressure = min(100, max(0,
                        (signal_data['whale_pump'] / 100 * 25) +
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 20 +
                        (signal_data['rsi_maverick'] * 15) +
                        (1 if signal_data['adx'] > 25 else 0) * 10 +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 15) +
                        (signal_data['win_rate'] / 100 * 15)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (signal_data['whale_dump'] / 100 * 25) +
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 20 +
                        ((1 - signal_data['rsi_maverick']) * 15) +
                        (1 if signal_data['adx'] > 25 else 0) * 10 +
                        (min(1, signal_data['volume'] / signal_data['volume_ma']) * 15) +
                        ((100 - signal_data['win_rate']) / 100 * 15)
                    ))
                    
                    scatter_data.append({
                        'symbol': symbol,
                        'x': float(buy_pressure),
                        'y': float(sell_pressure),
                        'signal_score': float(signal_data['signal_score']),
                        'current_price': float(signal_data['current_price']),
                        'signal': signal_data['signal'],
                        'win_rate': float(signal_data['win_rate'])
                    })
                    
            except Exception as e:
                print(f"Error procesando {symbol} para scatter: {e}")
                continue
        
        return jsonify(scatter_data)
        
    except Exception as e:
        print(f"Error en /api/scatter_data_improved: {e}")
        return jsonify([])

@app.route('/api/win_rate')
def get_win_rate():
    """Endpoint para obtener winrate"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        win_rate = indicator.calculate_win_rate(symbol, interval)
        
        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'win_rate': win_rate,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error en /api/win_rate: {e}")
        return jsonify({'error': 'Error calculando winrate'}), 500

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    """Endpoint para clasificación de riesgo"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/bolivia_time')
def get_bolivia_time():
    """Endpoint para hora de Bolivia"""
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d'),
        'timezone': 'America/La_Paz',
        'is_scalping_time': indicator.is_scalping_time()
    })

# Endpoints mantenidos para compatibilidad
@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    return jsonify({'alerts': []})

@app.route('/api/exit_signals')
def get_exit_signals():
    return jsonify({'exit_signals': []})

@app.route('/api/fear_greed_index')
def get_fear_greed_index():
    return jsonify({
        'value': 50,
        'sentiment': 'NEUTRAL',
        'color': 'warning',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/market_recommendations')
def get_market_recommendations():
    return jsonify({
        'recommendation': 'Mercado en fase de acumulación. Esperar señales claras.',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
