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

# Configuración Telegram - NUEVOS DATOS
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
        interval_map = {
            '15m': 450,    # 7.5 minutos (50%)
            '30m': 900,    # 15 minutos (50%)
            '1h': 1800,    # 30 minutos (50%)
            '2h': 3600,    # 1 hora (50%)
            '4h': 7200,    # 2 horas (50%)
            '8h': 14400,   # 4 horas (50%)
            '12h': 21600,  # 6 horas (50%)
            '1D': 43200,   # 12 horas (50%)
            '3D': 129600,  # 36 horas (50%)
            '1W': 302400   # 84 horas (50%)
        }
        
        return interval in interval_map

    def get_kucoin_data(self, symbol, interval, limit=100):
        """Obtener datos reales de KuCoin"""
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
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if candles:
                        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                        df = df.iloc[::-1].reset_index(drop=True)
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df = df.dropna()
                        result = df.tail(limit)
                        self.cache[cache_key] = (result, datetime.now())
                        return result
            
            # Fallback a datos de ejemplo si falla la API
            return self.generate_sample_data(limit, interval, symbol)
            
        except Exception as e:
            print(f"Error obteniendo datos de KuCoin para {symbol}: {e}")
            return self.generate_sample_data(limit, interval, symbol)

    def generate_sample_data(self, limit, interval, symbol):
        """Generar datos de ejemplo realistas basados en precios actuales"""
        try:
            # Obtener precio actual real desde una fuente alternativa
            price_url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol.replace('-', '')}"
            response = requests.get(price_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    current_price = float(data['data']['price'])
                else:
                    current_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
            else:
                current_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
                
        except:
            current_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
        
        # Generar datos más realistas con volatilidad apropiada
        volatility = 0.02 if 'BTC' in symbol or 'ETH' in symbol else 0.04
        returns = np.random.normal(0.001, volatility, limit)
        prices = current_price * (1 + np.cumsum(returns))
        
        data = {
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0.01, 0.01, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, limit))),
            'close': prices,
            'volume': np.random.lognormal(12, 1, limit)  # Volúmenes más realistas
        }
        
        df = pd.DataFrame(data)
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df

    # INDICADORES TÉCNICOS - TODOS MANUALES
    def calculate_sma(self, prices, period):
        """Calcular Media Móvil Simple"""
        if len(prices) < period:
            return np.zeros(len(prices))
        
        sma = np.zeros(len(prices))
        for i in range(len(prices)):
            if i < period - 1:
                sma[i] = np.mean(prices[:i+1])
            else:
                sma[i] = np.mean(prices[i-period+1:i+1])
        return sma

    def calculate_ema(self, prices, period):
        """Calcular Media Móvil Exponencial"""
        if len(prices) == 0 or period <= 0:
            return np.zeros(len(prices))
            
        alpha = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI Tradicional"""
        if len(prices) < period + 1:
            return np.zeros(len(prices))
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        # Primeros valores
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.zeros(len(prices))
        for i in range(len(prices)):
            if avg_losses[i] > 0:
                rs[i] = avg_gains[i] / avg_losses[i]
            else:
                rs[i] = 100 if avg_gains[i] > 0 else 50
        
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcular MACD"""
        if len(prices) < slow:
            return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger"""
        if len(prices) < period:
            return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
        
        sma = self.calculate_sma(prices, period)
        std = np.zeros(len(prices))
        
        for i in range(len(prices)):
            if i >= period - 1:
                window = prices[i-period+1:i+1]
                std[i] = np.std(window)
            else:
                std[i] = 0
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower

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
        
        atr = self.calculate_ema(tr, period)
        return atr

    def calculate_adx(self, high, low, close, period=14):
        """Calcular ADX, +DI, -DI"""
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

    def calculate_squeeze_momentum(self, close, high, low, period=20, bb_multiplier=2, kc_multiplier=1.5):
        """Calcular Squeeze Momentum"""
        n = len(close)
        
        # Bandas de Bollinger
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, period, bb_multiplier)
        
        # Keltner Channel
        typical_price = (high + low + close) / 3
        atr = self.calculate_atr(high, low, close, period)
        kc_upper = self.calculate_ema(typical_price, period) + (atr * kc_multiplier)
        kc_lower = self.calculate_ema(typical_price, period) - (atr * kc_multiplier)
        
        # Detectar squeeze
        squeeze_on = np.zeros(n, dtype=bool)
        squeeze_off = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]:
                squeeze_on[i] = True
            elif bb_lower[i] < kc_lower[i] and bb_upper[i] > kc_upper[i]:
                squeeze_off[i] = True
        
        # Momentum (simplificado)
        momentum = np.zeros(n)
        for i in range(1, n):
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

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias entre precio e indicador"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            # Divergencia alcista: precio hace lower low, indicador hace higher low
            if (price[i] < np.min(price[i-lookback:i]) and 
                indicator[i] > np.min(indicator[i-lookback:i])):
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace higher high, indicador hace lower high
            if (price[i] > np.max(price[i-lookback:i]) and 
                indicator[i] < np.max(indicator[i-lookback:i])):
                bearish_div[i] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones de chartismo básicos"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'wedge': np.zeros(n, dtype=bool)
        }
        
        for i in range(lookback, n-1):
            window_high = high[i-lookback:i+1]
            window_low = low[i-lookback:i+1]
            
            # Detección simplificada de patrones
            max_high = np.max(window_high)
            min_low = np.min(window_low)
            
            # Doble techo (simplificado)
            if (high[i] >= max_high * 0.98 and 
                np.sum(window_high >= max_high * 0.95) >= 2):
                patterns['double_top'][i] = True
            
            # Doble suelo (simplificado)
            if (low[i] <= min_low * 1.02 and 
                np.sum(window_low <= min_low * 1.05) >= 2):
                patterns['double_bottom'][i] = True
        
        return patterns

    # NUEVO: SISTEMA MULTI-TIMEFRAME MEJORADO
    def check_multi_timeframe_trend(self, symbol, interval, signal_type):
        """Verificar condiciones multi-temporalidad OBLIGATORIAS"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return {'valid': True, 'reason': 'No hierarchy defined'}
            
            results = {}
            mandatory_conditions = []
            
            # Verificar cada temporalidad en la jerarquía
            for tf_type, tf_value in hierarchy.items():
                if tf_value in ['5m', '1M']:  # Excluir temporalidades no operativas
                    continue
                    
                df = self.get_kucoin_data(symbol, tf_value, 50)
                if df is None or len(df) < 20:
                    results[tf_type] = 'NEUTRAL'
                    continue
                
                close = df['close'].values
                current_price = close[-1]
                
                # Calcular medias para determinar tendencia
                ma_9 = self.calculate_sma(close, 9)
                ma_21 = self.calculate_sma(close, 21)
                ma_50 = self.calculate_sma(close, 50)
                
                current_ma_9 = ma_9[-1] if len(ma_9) > 0 else 0
                current_ma_21 = ma_21[-1] if len(ma_21) > 0 else 0
                current_ma_50 = ma_50[-1] if len(ma_50) > 0 else 0
                
                # Determinar tendencia
                if current_price > current_ma_9 and current_ma_9 > current_ma_21:
                    trend = 'BULLISH'
                elif current_price < current_ma_9 and current_ma_9 < current_ma_21:
                    trend = 'BEARISH'
                else:
                    trend = 'NEUTRAL'
                
                results[tf_type] = trend
                
                # Verificar fuerza de tendencia Maverick
                trend_strength = self.calculate_trend_strength_maverick(close)
                current_strength = trend_strength['strength_signals'][-1]
                current_no_trade = trend_strength['no_trade_zones'][-1]
                
                # CONDICIONES OBLIGATORIAS
                if signal_type == 'LONG':
                    if tf_type == 'mayor':
                        if trend not in ['BULLISH', 'NEUTRAL']:
                            mandatory_conditions.append(f"Tendencia Mayor no es alcista/neutral: {trend}")
                    elif tf_type == 'media':
                        if trend != 'BULLISH':
                            mandatory_conditions.append(f"Tendencia Media no es EXCLUSIVAMENTE alcista: {trend}")
                    elif tf_type == 'menor':
                        if current_strength not in ['STRONG_UP', 'WEAK_UP']:
                            mandatory_conditions.append(f"Fuerza Tendencia Menor no es alcista: {current_strength}")
                    
                    # Verificar zona NO OPERAR
                    if current_no_trade:
                        mandatory_conditions.append(f"Zona NO OPERAR activa en {tf_type}")
                
                elif signal_type == 'SHORT':
                    if tf_type == 'mayor':
                        if trend not in ['BEARISH', 'NEUTRAL']:
                            mandatory_conditions.append(f"Tendencia Mayor no es bajista/neutral: {trend}")
                    elif tf_type == 'media':
                        if trend != 'BEARISH':
                            mandatory_conditions.append(f"Tendencia Media no es EXCLUSIVAMENTE bajista: {trend}")
                    elif tf_type == 'menor':
                        if current_strength not in ['STRONG_DOWN', 'WEAK_DOWN']:
                            mandatory_conditions.append(f"Fuerza Tendencia Menor no es bajista: {current_strength}")
                    
                    # Verificar zona NO OPERAR
                    if current_no_trade:
                        mandatory_conditions.append(f"Zona NO OPERAR activa en {tf_type}")
            
            is_valid = len(mandatory_conditions) == 0
            return {
                'valid': is_valid,
                'results': results,
                'mandatory_conditions': mandatory_conditions,
                'reason': 'OK' if is_valid else '; '.join(mandatory_conditions)
            }
            
        except Exception as e:
            print(f"Error verificando multi-timeframe para {symbol}: {e}")
            return {'valid': False, 'reason': f'Error: {str(e)}'}

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
            
            # Calcular percentil para zona alta
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
            
            return {
                'bb_width': bb_width.tolist(),
                'trend_strength': trend_strength.tolist(),
                'no_trade_zones': no_trade_zones.tolist(),
                'strength_signals': strength_signals,
                'high_zone_threshold': float(high_zone_threshold)
            }
            
        except Exception as e:
            print(f"Error en calculate_trend_strength_maverick: {e}")
            n = len(close)
            return {
                'bb_width': [0] * n,
                'trend_strength': [0] * n,
                'no_trade_zones': [False] * n,
                'strength_signals': ['NEUTRAL'] * n,
                'high_zone_threshold': 5.0
            }

    def calculate_whale_signals_improved(self, df, interval, sensitivity=1.7, min_volume_multiplier=1.5):
        """Indicador Cazador de Ballenas MEJORADO - Solo activo en 12H y 1D"""
        try:
            # CORRECCIÓN: Solo activar señal obligatoria en 12H y 1D
            if interval not in ['12h', '1D']:
                # Para otras TF, retornar datos pero sin señal obligatoria
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
                
                # Señales de compra (ballenas acumulando)
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] < close[i-1] or price_change < -0.5) and
                    low[i] <= low_5 * 1.01):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_pump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
                
                # Señales de venta (ballenas distribuyendo)
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_dump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            # Soporte y resistencia
            support_resistance_lookback = 20
            current_support = np.array([np.min(low[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            current_resistance = np.array([np.max(high[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            
            for i in range(5, n):
                if (whale_pump_smooth[i] > 25 and  # Umbral más alto para 12H/1D
                    close[i] <= current_support[i] * 1.02 and
                    volume[i] > np.mean(volume[max(0, i-10):i+1])):
                    confirmed_buy[i] = True
                
                if (whale_dump_smooth[i] > 20 and  # Umbral más alto para 12H/1D
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
                'is_obligatory': True
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
                'is_obligatory': interval in ['12h', '1D']
            }

    def calculate_smart_money_levels(self, high, low, close, lookback=50):
        """Calcular niveles Smart Money"""
        n = len(close)
        support_levels = np.zeros(n)
        resistance_levels = np.zeros(n)
        
        for i in range(lookback, n):
            # Soporte: mínimo del lookback
            support_levels[i] = np.min(low[i-lookback:i+1])
            # Resistencia: máximo del lookback
            resistance_levels[i] = np.max(high[i-lookback:i+1])
        
        return support_levels.tolist(), resistance_levels.tolist()

    def evaluate_signal_conditions_improved(self, data, current_idx, interval, adx_threshold=25):
        """Evaluar condiciones de señal con NUEVA ESTRUCTURA"""
        conditions = {
            'long': {
                'multi_timeframe': {'value': False, 'weight': 0, 'description': 'Condiciones Multi-TF obligatorias', 'obligatory': True},
                'whale_pump': {'value': False, 'weight': 25 if interval in ['12h', '1D'] else 0, 'description': 'Ballena compradora activa', 'obligatory': interval in ['12h', '1D']},
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación Medias Móviles', 'obligatory': False},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI Tradicional favorable', 'obligatory': False},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable', 'obligatory': False},
                'smart_money': {'value': False, 'weight': 20, 'description': 'Nivel Smart Money favorable', 'obligatory': False},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} + DMI favorable', 'obligatory': False},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD favorable', 'obligatory': False},
                'squeeze': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable', 'obligatory': False},
                'bollinger': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger favorables', 'obligatory': False},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrón chartismo favorable', 'obligatory': False}
            },
            'short': {
                'multi_timeframe': {'value': False, 'weight': 0, 'description': 'Condiciones Multi-TF obligatorias', 'obligatory': True},
                'whale_dump': {'value': False, 'weight': 25 if interval in ['12h', '1D'] else 0, 'description': 'Ballena vendedora activa', 'obligatory': interval in ['12h', '1D']},
                'moving_averages': {'value': False, 'weight': 15, 'description': 'Alineación Medias Móviles', 'obligatory': False},
                'rsi_traditional': {'value': False, 'weight': 15, 'description': 'RSI Tradicional favorable', 'obligatory': False},
                'rsi_maverick': {'value': False, 'weight': 15, 'description': 'RSI Maverick favorable', 'obligatory': False},
                'smart_money': {'value': False, 'weight': 20, 'description': 'Nivel Smart Money favorable', 'obligatory': False},
                'adx_dmi': {'value': False, 'weight': 10, 'description': f'ADX > {adx_threshold} + DMI favorable', 'obligatory': False},
                'macd': {'value': False, 'weight': 10, 'description': 'MACD favorable', 'obligatory': False},
                'squeeze': {'value': False, 'weight': 10, 'description': 'Squeeze Momentum favorable', 'obligatory': False},
                'bollinger': {'value': False, 'weight': 5, 'description': 'Bandas Bollinger favorables', 'obligatory': False},
                'chart_patterns': {'value': False, 'weight': 15, 'description': 'Patrón chartismo favorable', 'obligatory': False}
            }
        }
        
        if current_idx < 0:
            current_idx = len(data['close']) + current_idx
        
        if current_idx < 0 or current_idx >= len(data['close']):
            return conditions
        
        current_price = data['close'][current_idx]
        
        # Condiciones LONG
        conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_valid', False)
        
        # Ballenas (solo obligatorio en 12H/1D)
        if interval in ['12h', '1D']:
            conditions['long']['whale_pump']['value'] = (
                data['whale_pump'][current_idx] > 25 and 
                data['whale_confirmed_buy'][current_idx]
            )
        
        # Medias Móviles
        conditions['long']['moving_averages']['value'] = (
            current_price > data['ma_9'][current_idx] and
            data['ma_9'][current_idx] > data['ma_21'][current_idx] and
            data['ma_21'][current_idx] > data['ma_50'][current_idx]
        )
        
        # RSI Tradicional
        conditions['long']['rsi_traditional']['value'] = (
            data['rsi_traditional'][current_idx] < 70 and
            data['rsi_traditional_bullish_div'][current_idx]
        )
        
        # RSI Maverick
        conditions['long']['rsi_maverick']['value'] = (
            data['rsi_maverick'][current_idx] < 0.8 and
            data['rsi_maverick_bullish_div'][current_idx]
        )
        
        # Smart Money
        conditions['long']['smart_money']['value'] = (
            current_price <= data['smart_support'][current_idx] * 1.02
        )
        
        # ADX + DMI
        conditions['long']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['plus_di'][current_idx] > data['minus_di'][current_idx]
        )
        
        # MACD
        conditions['long']['macd']['value'] = (
            data['macd_line'][current_idx] > data['macd_signal'][current_idx] and
            data['macd_histogram'][current_idx] > 0
        )
        
        # Squeeze Momentum
        conditions['long']['squeeze']['value'] = (
            data['squeeze_off'][current_idx] and
            data['squeeze_momentum'][current_idx] > 0
        )
        
        # Bandas Bollinger
        conditions['long']['bollinger']['value'] = (
            current_price <= data['bb_lower'][current_idx] * 1.02
        )
        
        # Patrones Chartismo
        conditions['long']['chart_patterns']['value'] = (
            data['chart_double_bottom'][current_idx] or
            data['chart_wedge'][current_idx]
        )
        
        # Condiciones SHORT
        conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_valid', False)
        
        # Ballenas (solo obligatorio en 12H/1D)
        if interval in ['12h', '1D']:
            conditions['short']['whale_dump']['value'] = (
                data['whale_dump'][current_idx] > 20 and 
                data['whale_confirmed_sell'][current_idx]
            )
        
        # Medias Móviles
        conditions['short']['moving_averages']['value'] = (
            current_price < data['ma_9'][current_idx] and
            data['ma_9'][current_idx] < data['ma_21'][current_idx] and
            data['ma_21'][current_idx] < data['ma_50'][current_idx]
        )
        
        # RSI Tradicional
        conditions['short']['rsi_traditional']['value'] = (
            data['rsi_traditional'][current_idx] > 30 and
            data['rsi_traditional_bearish_div'][current_idx]
        )
        
        # RSI Maverick
        conditions['short']['rsi_maverick']['value'] = (
            data['rsi_maverick'][current_idx] > 0.2 and
            data['rsi_maverick_bearish_div'][current_idx]
        )
        
        # Smart Money
        conditions['short']['smart_money']['value'] = (
            current_price >= data['smart_resistance'][current_idx] * 0.98
        )
        
        # ADX + DMI
        conditions['short']['adx_dmi']['value'] = (
            data['adx'][current_idx] > adx_threshold and
            data['minus_di'][current_idx] > data['plus_di'][current_idx]
        )
        
        # MACD
        conditions['short']['macd']['value'] = (
            data['macd_line'][current_idx] < data['macd_signal'][current_idx] and
            data['macd_histogram'][current_idx] < 0
        )
        
        # Squeeze Momentum
        conditions['short']['squeeze']['value'] = (
            data['squeeze_off'][current_idx] and
            data['squeeze_momentum'][current_idx] < 0
        )
        
        # Bandas Bollinger
        conditions['short']['bollinger']['value'] = (
            current_price >= data['bb_upper'][current_idx] * 0.98
        )
        
        # Patrones Chartismo
        conditions['short']['chart_patterns']['value'] = (
            data['chart_double_top'][current_idx] or
            data['chart_head_shoulders'][current_idx]
        )
        
        return conditions

    def calculate_signal_score(self, conditions, signal_type):
        """Calcular puntuación de señal con OBLIGATORIEDAD"""
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        obligatory_failed = False
        
        signal_conditions = conditions.get(signal_type, {})
        
        # Verificar condiciones obligatorias primero
        for key, condition in signal_conditions.items():
            if condition.get('obligatory', False) and not condition['value']:
                obligatory_failed = True
                break
        
        # Si falla alguna obligatoria, score = 0
        if obligatory_failed:
            return 0, []
        
        # Calcular score normal
        for key, condition in signal_conditions.items():
            if not condition.get('obligatory', False):  # No contar obligatorias en el peso
                total_weight += condition['weight']
                if condition['value']:
                    achieved_weight += condition['weight']
                    fulfilled_conditions.append(condition['description'])
        
        if total_weight == 0:
            return 0, []
        
        score = (achieved_weight / total_weight) * 100
        return min(score, 100), fulfilled_conditions

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con Smart Money"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Niveles Smart Money
            smart_support, smart_resistance = self.calculate_smart_money_levels(high, low, close)
            current_support = smart_support[-1] if len(smart_support) > 0 else np.min(low[-20:])
            current_resistance = smart_resistance[-1] if len(smart_resistance) > 0 else np.max(high[-20:])
            
            atr_percentage = current_atr / current_price

            if signal_type == 'LONG':
                # Entrada lo más cerca posible del soporte Smart Money
                entry = min(current_price, current_support * 1.005)  # 0.5% sobre soporte
                stop_loss = max(current_support * 0.98, entry - (current_atr * 1.5))
                
                # Take Profit basado en resistencia Smart Money
                tp1 = current_resistance * 0.995  # 0.5% bajo resistencia
                
                # Asegurar relación riesgo/beneficio mínima 1:2
                min_tp = entry + (2 * (entry - stop_loss))
                tp1 = max(tp1, min_tp)
                
            else:  # SHORT
                # Entrada lo más cerca posible de la resistencia Smart Money
                entry = max(current_price, current_resistance * 0.995)  # 0.5% bajo resistencia
                stop_loss = min(current_resistance * 1.02, entry + (current_atr * 1.5))
                
                # Take Profit basado en soporte Smart Money
                tp1 = current_support * 1.005  # 0.5% sobre soporte
                
                # Asegurar relación riesgo/beneficio mínima 1:2
                min_tp = entry - (2 * (stop_loss - entry))
                tp1 = min(tp1, min_tp)
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp1)],
                'support': float(current_support),
                'resistance': float(current_resistance),
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.05],
                'support': current_price * 0.95,
                'resistance': current_price * 1.05,
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_win_rate(self, symbol, interval, lookback=100):
        """Calcular winrate histórico"""
        try:
            df = self.get_kucoin_data(symbol, interval, lookback + 20)
            if df is None or len(df) < lookback + 10:
                return 0.0
            
            close = df['close'].values
            signals = []
            
            # Simular señales históricas (simplificado)
            for i in range(10, len(close) - 5):
                # Lógica básica de detección de señales
                if close[i] > self.calculate_sma(close, 20)[i] and close[i] < close[i-1]:
                    # LONG signal
                    entry = close[i]
                    exit_price = close[i+5]  # Salida después de 5 velas
                    signals.append(('LONG', entry, exit_price))
                elif close[i] < self.calculate_sma(close, 20)[i] and close[i] > close[i-1]:
                    # SHORT signal  
                    entry = close[i]
                    exit_price = close[i+5]  # Salida después de 5 velas
                    signals.append(('SHORT', entry, exit_price))
            
            if not signals:
                return 0.0
            
            winning_trades = 0
            for signal_type, entry, exit_price in signals:
                if signal_type == 'LONG' and exit_price > entry:
                    winning_trades += 1
                elif signal_type == 'SHORT' and exit_price < entry:
                    winning_trades += 1
            
            win_rate = (winning_trades / len(signals)) * 100
            return win_rate
            
        except Exception as e:
            print(f"Error calculando winrate para {symbol}: {e}")
            return 0.0

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """GENERACIÓN DE SEÑALES MEJORADA - NUEVA ESTRATEGIA"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            close = df['close'].values
            high = df['high'].values  
            low = df['low'].values
            volume = df['volume'].values
            
            # 1. INDICADORES BÁSICOS
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # 2. INDICADORES DE MOMENTUM
            rsi_traditional = self.calculate_rsi(close, 14)
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            
            # 3. INDICADORES DE VOLATILIDAD
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            squeeze_data = self.calculate_squeeze_momentum(close, high, low)
            
            # 4. INDICADORES PERSONALIZADOS
            whale_data = self.calculate_whale_signals_improved(df, interval)
            
            # RSI Maverick (basado en %B Bollinger)
            rsi_maverick = np.zeros(len(close))
            for i in range(len(close)):
                if (bb_upper[i] - bb_lower[i]) > 0:
                    rsi_maverick[i] = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
                else:
                    rsi_maverick[i] = 0.5
            
            # 5. FUERZA DE TENDENCIA MAVERICK
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # 6. NIVELES SMART MONEY
            smart_support, smart_resistance = self.calculate_smart_money_levels(high, low, close, sr_period)
            
            # 7. PATRONES DE CHARTISMO
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # 8. DIVERGENCIAS
            rsi_traditional_bullish_div, rsi_traditional_bearish_div = self.detect_divergence(close, rsi_traditional)
            rsi_maverick_bullish_div, rsi_maverick_bearish_div = self.detect_divergence(close, rsi_maverick)
            
            # 9. VERIFICACIÓN MULTI-TIMEFRAME (OBLIGATORIA)
            current_idx = -1
            
            # Preparar datos para evaluación
            analysis_data = {
                'close': close,
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'whale_confirmed_buy': whale_data['confirmed_buy'],
                'whale_confirmed_sell': whale_data['confirmed_sell'],
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'rsi_traditional': rsi_traditional,
                'rsi_traditional_bullish_div': rsi_traditional_bullish_div,
                'rsi_traditional_bearish_div': rsi_traditional_bearish_div,
                'rsi_maverick': rsi_maverick,
                'rsi_maverick_bullish_div': rsi_maverick_bullish_div,
                'rsi_maverick_bearish_div': rsi_maverick_bearish_div,
                'smart_support': smart_support,
                'smart_resistance': smart_resistance,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'squeeze_off': squeeze_data['squeeze_off'],
                'squeeze_momentum': squeeze_data['momentum'],
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'chart_double_top': chart_patterns['double_top'],
                'chart_double_bottom': chart_patterns['double_bottom'],
                'chart_head_shoulders': chart_patterns['head_shoulders'],
                'chart_wedge': chart_patterns['wedge']
            }
            
            # Evaluar condiciones LONG
            multi_tf_long = self.check_multi_timeframe_trend(symbol, interval, 'LONG')
            analysis_data['multi_timeframe_valid'] = multi_tf_long['valid']
            conditions_long = self.evaluate_signal_conditions_improved(analysis_data, current_idx, interval, adx_threshold)
            long_score, long_conditions = self.calculate_signal_score(conditions_long, 'long')
            
            # Evaluar condiciones SHORT  
            multi_tf_short = self.check_multi_timeframe_trend(symbol, interval, 'SHORT')
            analysis_data['multi_timeframe_valid'] = multi_tf_short['valid']
            conditions_short = self.evaluate_signal_conditions_improved(analysis_data, current_idx, interval, adx_threshold)
            short_score, short_conditions = self.calculate_signal_score(conditions_short, 'short')
            
            # Determinar señal final
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            
            if long_score >= 70 and multi_tf_long['valid']:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_conditions
            elif short_score >= 70 and multi_tf_short['valid']:
                signal_type = 'SHORT'  
                signal_score = short_score
                fulfilled_conditions = short_conditions
            
            # Calcular niveles de entrada/salida
            current_price = float(close[current_idx])
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            # Calcular winrate
            win_rate = self.calculate_win_rate(symbol, interval)
            
            # Preparar datos para el frontend
            result_data = {
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
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
                'adx': float(adx[current_idx]),
                'plus_di': float(plus_di[current_idx]),
                'minus_di': float(minus_di[current_idx]),
                'whale_pump': float(whale_data['whale_pump'][current_idx]),
                'whale_dump': float(whale_data['whale_dump'][current_idx]),
                'rsi_maverick': float(rsi_maverick[current_idx]),
                'rsi_traditional': float(rsi_traditional[current_idx]),
                'win_rate': float(win_rate),
                'fulfilled_conditions': fulfilled_conditions,
                'multi_timeframe_ok': multi_tf_long['valid'] if signal_type == 'LONG' else multi_tf_short['valid'],
                'multi_timeframe_reason': multi_tf_long['reason'] if signal_type == 'LONG' else multi_tf_short['reason'],
                'trend_strength_signal': trend_strength_data['strength_signals'][current_idx],
                'no_trade_zone': trend_strength_data['no_trade_zones'][current_idx],
                'data': df.tail(50).to_dict('records'),
                'indicators': {
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:].tolist(),
                    'rsi_traditional': rsi_traditional[-50:].tolist(),
                    'ma_9': ma_9[-50:].tolist(),
                    'ma_21': ma_21[-50:].tolist(),
                    'ma_50': ma_50[-50:].tolist(),
                    'ma_200': ma_200[-50:].tolist(),
                    'bb_upper': bb_upper[-50:].tolist(),
                    'bb_middle': bb_middle[-50:].tolist(),
                    'bb_lower': bb_lower[-50:].tolist(),
                    'macd_line': macd_line[-50:].tolist(),
                    'macd_signal': macd_signal[-50:].tolist(),
                    'macd_histogram': macd_histogram[-50:].tolist(),
                    'squeeze_on': squeeze_data['squeeze_on'][-50:],
                    'squeeze_off': squeeze_data['squeeze_off'][-50:],
                    'squeeze_momentum': squeeze_data['momentum'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'smart_support': smart_support[-50:],
                    'smart_resistance': smart_resistance[-50:]
                }
            }
            
            return result_data
            
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
            'win_rate': 0,
            'fulfilled_conditions': [],
            'multi_timeframe_ok': False,
            'multi_timeframe_reason': 'Error en análisis',
            'trend_strength_signal': 'NEUTRAL',
            'no_trade_zone': False,
            'data': [],
            'indicators': {}
        }

    def generate_scalping_alerts(self):
        """Generar alertas de scalping"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h']:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            for symbol in CRYPTO_SYMBOLS[:10]:  # Limitar para performance
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 70 and
                        signal_data['multi_timeframe_ok']):
                        
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
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'win_rate': signal_data['win_rate'],
                            'risk_category': risk_category
                        }
                        
                        alerts.append(alert)
                    
                except Exception as e:
                    print(f"Error generando alerta para {symbol}: {e}")
                    continue
        
        return alerts

# Instancia global del indicador
indicator = TradingIndicator()

# Funciones auxiliares
def get_risk_classification(symbol):
    """Obtener la clasificación de riesgo de una criptomoneda"""
    for risk_level, symbols in CRYPTO_RISK_CLASSIFICATION.items():
        if symbol in symbols:
            return risk_level
    return "medio"

def send_telegram_alert(alert_data):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        risk_classification = get_risk_classification(alert_data['symbol'])
        risk_text = {
            'bajo': '🟢 BAJO RIESGO',
            'medio': '🟡 MEDIO RIESGO', 
            'alto': '🔴 ALTO RIESGO',
            'memecoins': '🟣 MEMECOIN'
        }.get(risk_classification, '🟡 MEDIO RIESGO')
        
        message = f"""
🚨 ALERTA DE TRADING - MULTI-TIMEFRAME WGTA PRO 🚨

📈 Crypto: {alert_data['symbol']} {risk_text}
⏰ Temporalidad: {alert_data['interval']}
🎯 Señal: {alert_data['signal']}
📊 Score: {alert_data['score']:.1f}%
📈 WinRate: {alert_data['win_rate']:.1f}%

💰 Precio actual: ${alert_data.get('current_price', alert_data['entry']):.6f}
🎯 Entrada: ${alert_data['entry']:.6f}
🛑 Stop Loss: ${alert_data['stop_loss']:.6f}
🎯 Take Profit: ${alert_data['take_profit']:.6f}

📊 Apalancamiento: x{alert_data['leverage']}

✅ Condiciones Multi-TF: CONFIRMADAS
🔔 Sistema profesional multi-temporalidad

⚠️ Gestiona tu riesgo adecuadamente.
        """
        
        asyncio.run(bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, 
            text=message
        ))
        print(f"Alerta enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar cada 60 segundos
            if True:  # Simplificado para testing
                print("Verificando alertas...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    send_telegram_alert(alert)
            
            time.sleep(60)
            
        except Exception as e:
            print(f"Error en background_alert_checker: {e}")
            time.sleep(60)

# Iniciar verificador de alertas en segundo plano
try:
    alert_thread = Thread(target=background_alert_checker, daemon=True)
    alert_thread.start()
    print("Background alert checker iniciado")
except Exception as e:
    print(f"Error iniciando background alert checker: {e}")

# ENDPOINTS PRINCIPALES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint principal para señales de trading"""
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
                signal_data = indicator.generate_signals_improved(
                    symbol, interval, di_period, adx_threshold
                )
                
                if signal_data and signal_data['signal'] != 'NEUTRAL' and signal_data['signal_score'] >= 70:
                    all_signals.append(signal_data)
                
            except Exception as e:
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
        
        for symbol in CRYPTO_SYMBOLS[:15]:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones de compra/venta
                    buy_pressure = min(100, max(0, signal_data['signal_score']))
                    sell_pressure = min(100, max(0, 100 - signal_data['signal_score']))
                    
                    scatter_data.append({
                        'symbol': symbol,
                        'x': float(buy_pressure),
                        'y': float(sell_pressure),
                        'signal_score': float(signal_data['signal_score']),
                        'current_price': float(signal_data['current_price']),
                        'signal': signal_data['signal'],
                        'risk_category': get_risk_classification(symbol)
                    })
                    
            except Exception as e:
                continue
        
        return jsonify(scatter_data)
        
    except Exception as e:
        print(f"Error en /api/scatter_data_improved: {e}")
        return jsonify([])

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    """Endpoint para clasificación de riesgo"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    """Endpoint para alertas de scalping"""
    try:
        alerts = indicator.generate_scalping_alerts()
        return jsonify({'alerts': alerts})
    except Exception as e:
        return jsonify({'alerts': []})

@app.route('/api/win_rate')
def get_win_rate():
    """Endpoint para winrate"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        win_rate = indicator.calculate_win_rate(symbol, interval)
        return jsonify({'win_rate': win_rate})
    except Exception as e:
        return jsonify({'win_rate': 0})

@app.route('/api/bolivia_time')
def get_bolivia_time():
    """Endpoint para hora de Bolivia"""
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d'),
        'day_of_week': current_time.strftime('%A'),
        'is_scalping_time': indicator.is_scalping_time()
    })

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        # Crear gráfico simple del reporte
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if signal_data['data']:
            dates = [d['timestamp'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            ax.plot(dates, closes, label='Precio')
            ax.axhline(y=signal_data['entry'], color='green', linestyle='--', label='Entrada')
            ax.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', label='Stop Loss')
            ax.axhline(y=signal_data['take_profit'][0], color='blue', linestyle='--', label='Take Profit')
        
        ax.set_title(f'Reporte {symbol} - {interval}')
        ax.legend()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': 'Error generando reporte'}), 500

# Health check y manejo de errores
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
