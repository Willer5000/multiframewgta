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

# 6 criptomonedas principales para estrategias
TOP_CRYPTO_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "XRP-USDT", "XAUT-USDT"]

# Mapeo de temporalidades para análisis multi-timeframe (solo 2h y 8h para análisis)
TIMEFRAME_HIERARCHY = {
    '4h': {'mayor': '8h', 'media': '4h', 'menor': '2h'},
    '12h': {'mayor': '1D', 'media': '12h', 'menor': '4h'},
    '1D': {'mayor': '1W', 'media': '1D', 'menor': '12h'},
    '1W': {'mayor': '1M', 'media': '1W', 'menor': '3D'}
}

# Configuración de estrategias por temporalidad (SOLO 4h, 12h, 1D, 1W)
STRATEGY_TIMEFRAMES = {
    'Momentum Divergence': ['4h', '12h', '1D', '1W'],
    'ADX Power Trend': ['4h', '12h', '1D'],
    'MACD Histogram Reversal': ['4h', '12h', '1D'],
    'Volume Spike Momentum': ['4h', '12h', '1D'],
    'Double Confirmation RSI': ['4h', '12h', '1D'],
    'Trend Strength Maverick': ['4h', '12h', '1D', '1W'],
    'MA Convergence Divergence': ['4h', '12h', '1D'],
    'Volume-Price Divergence': ['4h', '12h', '1D', '1W'],
    'Stochastic RSI Combo': ['4h', '12h', '1D', '1W'],
    'Whale DMI Combo': ['12h', '1D'],
    'Support Resistance Bounce': ['4h', '12h', '1D'],
    'Multi-Timeframe Confirmation': ['4h', '12h', '1D']
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_operations = {}
        self.winrate_data = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.sent_exit_signals = set()
        self.volume_ema_signals = {}
        self.strategy_signals = {}
        
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def is_trading_time(self):
        """Verificar si es horario de trading - 24/7"""
        return True  # Trading 24/7

    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela"""
        interval_seconds = {
            '4h': 14400, '12h': 43200, '1D': 86400, '1W': 604800
        }
        
        seconds = interval_seconds.get(interval, 3600)
        
        # PORCENTAJES DE FORMACIÓN DE VELA
        if interval == '4h':
            percent = 85
        elif interval == '12h':
            percent = 90
        elif interval == '1D':
            percent = 95
        elif interval == '1W':
            percent = 99
        else:
            percent = 50
        
        seconds_passed = current_time.timestamp() % seconds
        return seconds_passed >= (seconds * percent / 100)

    def get_kucoin_data(self, symbol, interval, limit=100):
        """Obtener datos de KuCoin con manejo robusto de errores"""
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < 60:
                    return cached_data
            
            interval_map = {
                '15m': '15min', '30m': '30min', '1h': '1hour', '2h': '2hour',
                '4h': '4hour', '8h': '8hour', '12h': '12hour',
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

    def calculate_dynamic_support_resistance(self, high, low, close, num_levels=6):
        """Calcular soportes y resistencias dinámicos"""
        try:
            # Usar pivot points y niveles de Fibonacci
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            r1 = 2 * pivot - low[-1]
            s1 = 2 * pivot - high[-1]
            r2 = pivot + (high[-1] - low[-1])
            s2 = pivot - (high[-1] - low[-1])
            
            # Niveles adicionales basados en máximos/mínimos recientes
            recent_highs = sorted(high[-50:])[-3:]
            recent_lows = sorted(low[-50:])[:3]
            
            support_levels = list(recent_lows) + [s1, s2]
            resistance_levels = list(recent_highs) + [r1, r2]
            
            # Eliminar duplicados y ordenar
            support_levels = sorted(list(set([round(s, 6) for s in support_levels])))[:num_levels]
            resistance_levels = sorted(list(set([round(r, 6) for r in resistance_levels])), reverse=True)[:num_levels]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            print(f"Error calculando soportes/resistencias: {e}")
            current_price = close[-1]
            support_levels = [current_price * 0.95, current_price * 0.90]
            resistance_levels = [current_price * 1.05, current_price * 1.10]
            return support_levels, resistance_levels

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15, support_levels=None, resistance_levels=None):
        """Calcular entradas y salidas óptimas mejoradas con soportes/resistencias"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular soportes y resistencias si no se proporcionan
            if support_levels is None or resistance_levels is None:
                support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            # Para LONG: entrada en soporte más cercano, stop loss debajo del soporte
            if signal_type == 'LONG':
                # Encontrar soporte más cercano por debajo del precio actual
                valid_supports = [s for s in support_levels if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)  # Soporte más fuerte (más alto)
                else:
                    entry = current_price * 0.995  # Ligera corrección si no hay soportes
                
                # Stop loss debajo del siguiente soporte o usando ATR
                if len(support_levels) > 1:
                    stop_loss = support_levels[1] if len(support_levels) > 1 else entry - (current_atr * 2)
                else:
                    stop_loss = entry - (current_atr * 2)
                
                # Take profits en resistencias
                take_profits = []
                for resistance in resistance_levels[:3]:
                    if resistance > entry:
                        take_profits.append(resistance)
                
                if not take_profits:
                    take_profits = [entry + (2 * (entry - stop_loss))]
            
            else:  # SHORT
                # Encontrar resistencia más cercana por encima del precio actual
                valid_resistances = [r for r in resistance_levels if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)  # Resistencia más fuerte (más baja)
                else:
                    entry = current_price * 1.005  # Ligera corrección si no hay resistencias
                
                # Stop loss encima de la siguiente resistencia o usando ATR
                if len(resistance_levels) > 1:
                    stop_loss = resistance_levels[1] if len(resistance_levels) > 1 else entry + (current_atr * 2)
                else:
                    stop_loss = entry + (current_atr * 2)
                
                # Take profits en soportes
                take_profits = []
                for support in support_levels[:3]:
                    if support < entry:
                        take_profits.append(support)
                
                if not take_profits:
                    take_profits = [entry - (2 * (stop_loss - entry))]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits[:3]],
                'support_levels': [float(s) for s in support_levels],
                'resistance_levels': [float(r) for r in resistance_levels],
                'atr': float(current_atr),
                'atr_percentage': float(current_atr / current_price)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support_levels': [current_price * 0.95, current_price * 0.90],
                'resistance_levels': [current_price * 1.05, current_price * 1.10],
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

    def calculate_stochastic_rsi(self, close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        """Calcular RSI Estocástico"""
        try:
            n = len(close)
            
            # Calcular RSI primero
            rsi = self.calculate_rsi(close, rsi_period)
            
            # Calcular Stochastic del RSI
            stoch_rsi = np.zeros(n)
            k_line = np.zeros(n)
            d_line = np.zeros(n)
            
            for i in range(stoch_period - 1, n):
                start_idx = i - stoch_period + 1
                if start_idx < 0:
                    start_idx = 0
                
                rsi_window = rsi[start_idx:i+1]
                if len(rsi_window) > 0:
                    rsi_low = np.min(rsi_window)
                    rsi_high = np.max(rsi_window)
                    
                    if (rsi_high - rsi_low) > 0:
                        stoch_rsi[i] = 100 * (rsi[i] - rsi_low) / (rsi_high - rsi_low)
                    else:
                        stoch_rsi[i] = 50
                else:
                    stoch_rsi[i] = 50
            
            # Calcular %K (media móvil simple de Stochastic RSI)
            for i in range(k_period - 1, n):
                start_idx = i - k_period + 1
                if start_idx < 0:
                    start_idx = 0
                k_line[i] = np.mean(stoch_rsi[start_idx:i+1])
            
            # Calcular %D (media móvil simple de %K)
            for i in range(k_period + d_period - 2, n):
                start_idx = i - d_period + 1
                if start_idx < 0:
                    start_idx = 0
                d_line[i] = np.mean(k_line[start_idx:i+1])
            
            return {
                'stoch_rsi': stoch_rsi.tolist(),
                'k_line': k_line.tolist(),
                'd_line': d_line.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_stochastic_rsi: {e}")
            n = len(close)
            return {
                'stoch_rsi': [50] * n,
                'k_line': [50] * n,
                'd_line': [50] * n
            }

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

    def calculate_whale_signals_improved(self, df, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=50, signal_threshold=25):
        """Implementación MEJORADA del indicador de ballenas - señal dura 7 velas"""
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
            extended_buy = np.zeros(n, dtype=bool)  # Señal extendida 7 velas
            extended_sell = np.zeros(n, dtype=bool)  # Señal extendida 7 velas
            
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
                    
                    # Extender señal 7 velas hacia adelante
                    for j in range(i, min(n, i + 7)):
                        whale_pump_signal[j] = max(whale_pump_signal[j], whale_pump_signal[i] * 0.8)
                
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_dump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
                    
                    # Extender señal 7 velas hacia adelante
                    for j in range(i, min(n, i + 7)):
                        whale_dump_signal[j] = max(whale_dump_signal[j], whale_dump_signal[i] * 0.8)
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            current_support = np.array([np.min(low[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            current_resistance = np.array([np.max(high[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            
            for i in range(5, n):
                if (whale_pump_smooth[i] > signal_threshold and 
                    close[i] <= current_support[i] * 1.02 and
                    volume[i] > np.mean(volume[max(0, i-10):i+1])):
                    confirmed_buy[i] = True
                    # Extender señal de compra 7 velas
                    for j in range(i, min(n, i + 7)):
                        extended_buy[j] = True
                
                if (whale_dump_smooth[i] > signal_threshold and 
                    close[i] >= current_resistance[i] * 0.98 and
                    volume[i] > np.mean(volume[max(0, i-10):i+1])):
                    confirmed_sell[i] = True
                    # Extender señal de venta 7 velas
                    for j in range(i, min(n, i + 7)):
                        extended_sell[j] = True
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'confirmed_buy': confirmed_buy.tolist(),
                'confirmed_sell': confirmed_sell.tolist(),
                'extended_buy': extended_buy.tolist(),
                'extended_sell': extended_sell.tolist(),
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
                'extended_buy': [False] * n,
                'extended_sell': [False] * n,
                'support': df['low'].values.tolist(),
                'resistance': df['high'].values.tolist(),
                'volume_anomaly': [False] * n
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

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias entre precio e indicador"""
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
        
        # Extender señal por 4 velas
        extended_bullish = bullish_div.copy()
        extended_bearish = bearish_div.copy()
        
        for i in range(n):
            if bullish_div[i]:
                for j in range(1, min(5, n-i)):
                    extended_bullish[i+j] = True
            if bearish_div[i]:
                for j in range(1, min(5, n-i)):
                    extended_bearish[i+j] = True
        
        return extended_bullish.tolist(), extended_bearish.tolist()

    def detect_divergence_traditional(self, price, indicator, lookback=14):
        """Detectar divergencias para RSI Tradicional (7 velas)"""
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
        
        # Extender señal por 7 velas para RSI Tradicional
        extended_bullish = bullish_div.copy()
        extended_bearish = bearish_div.copy()
        
        for i in range(n):
            if bullish_div[i]:
                for j in range(1, min(8, n-i)):
                    extended_bullish[i+j] = True
            if bearish_div[i]:
                for j in range(1, min(8, n-i)):
                    extended_bearish[i+j] = True
        
        return extended_bullish.tolist(), extended_bearish.tolist()

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

    def calculate_volume_anomaly(self, volume, close, period=20, std_multiplier=2):
        """Calcular anomalías de volumen"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_signal = ['NEUTRAL'] * n
            
            for i in range(period, n):
                # Media móvil de volumen
                volume_ma = self.calculate_sma(volume[:i+1], period)
                current_volume_ma = volume_ma[i] if i < len(volume_ma) else volume[i]
                
                # Ratio volumen actual vs MA
                if current_volume_ma > 0:
                    volume_ratio[i] = volume[i] / current_volume_ma
                else:
                    volume_ratio[i] = 1
                
                # Detectar anomalía (> 2σ)
                if i >= period * 2:
                    window = volume[max(0, i-period*2):i+1]
                    std_volume = np.std(window) if len(window) > 1 else 0
                    
                    if volume_ratio[i] > 1 + (std_multiplier * (std_volume / current_volume_ma if current_volume_ma > 0 else 0)):
                        volume_anomaly[i] = True
                        
                        # Determinar si es compra o venta basado en dirección del precio
                        if i > 0:
                            price_change = (close[i] - close[i-1]) / close[i-1] * 100
                            if price_change > 0:
                                volume_signal[i] = 'COMPRA'
                            else:
                                volume_signal[i] = 'VENTA'
                
                # Detectar clusters (múltiples anomalías en 5 periodos)
                if i >= 5:
                    recent_anomalies = volume_anomaly[max(0, i-4):i+1]
                    if np.sum(recent_anomalies) >= 2:
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ma': volume_ma.tolist() if 'volume_ma' in locals() else [0] * n,
                'volume_signal': volume_signal
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly: {e}")
            n = len(volume)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ma': [0] * n,
                'volume_signal': ['NEUTRAL'] * n
            }

    def check_ma_crossover(self, ma_short, ma_long, lookback=3):
        """Detectar cruce de medias móviles"""
        n = len(ma_short)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if (ma_short[i] > ma_long[i] and 
                ma_short[i-1] <= ma_long[i-1]):
                ma_cross_bullish[i] = True
                if i + 1 < n:
                    ma_cross_bullish[i+1] = True
            
            if (ma_short[i] < ma_long[i] and 
                ma_short[i-1] >= ma_long[i-1]):
                ma_cross_bearish[i] = True
                if i + 1 < n:
                    ma_cross_bearish[i+1] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()

    def check_macd_crossover(self, macd, signal):
        """Detectar cruce de MACD"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if (macd[i] > signal[i] and 
                macd[i-1] <= signal[i-1]):
                macd_cross_bullish[i] = True
                if i + 1 < n:
                    macd_cross_bullish[i+1] = True
            
            if (macd[i] < signal[i] and 
                macd[i-1] >= signal[i-1]):
                macd_cross_bearish[i] = True
                if i + 1 < n:
                    macd_cross_bearish[i+1] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de +DI y -DI con confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                di_cross_bullish[i] = True
                # Señal dura 1 vela más
                if i + 1 < n:
                    di_cross_bullish[i+1] = True
            
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
                # Señal dura 1 vela más
                if i + 1 < n:
                    di_cross_bearish[i+1] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

    def check_adx_slope(self, adx, period=3):
        """Verificar pendiente positiva del ADX"""
        n = len(adx)
        adx_slope_positive = np.zeros(n, dtype=bool)
        
        for i in range(period, n):
            if adx[i] > 25:
                slope = (adx[i] - adx[i-period]) / period
                if slope > 0:
                    adx_slope_positive[i] = True
        
        return adx_slope_positive.tolist()

    # ==============================================
    # ESTRATEGIA 1: MOMENTUM DIVERGENCE
    # ==============================================
    def check_momentum_divergence_signal(self, symbol, interval):
        """Estrategia Momentum Divergence para temporalidades mayores"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Momentum Divergence']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calcular indicadores
            rsi_traditional = self.calculate_rsi(close, 14)
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Detectar divergencias
            rsi_bullish, rsi_bearish = self.detect_divergence_traditional(close, rsi_traditional)
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            
            # Volume clusters
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Medias móviles para confirmación
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            
            # Condiciones para LONG (suelo)
            if (rsi_bullish[-1] and rsi_maverick_bullish[-1] and
                volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'COMPRA' and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP'] and
                close[-1] > ma9[-1] and close[-1] > ma21[-1]):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT (techo)
            elif (rsi_bearish[-1] and rsi_maverick_bearish[-1] and
                  volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'VENTA' and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  close[-1] < ma9[-1] and close[-1] < ma21[-1]):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_momentum_divergence_chart(symbol, interval, df, rsi_traditional, 
                                                                  rsi_maverick, volume_data, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'MOMENTUM DIVERGENCE',
                'chart': chart_buffer,
                'filters': [
                    'Divergencia RSI Tradicional confirmada',
                    'Divergencia RSI Maverick confirmada',
                    'Clúster de volumen confirmado',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Precio {" > " if signal_type == "LONG" else " < "} MA9 y MA21'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_momentum_divergence_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_momentum_divergence_chart(self, symbol, interval, df, rsi_traditional, rsi_maverick, 
                                          volume_data, ftm_data, signal_type):
        """Generar gráfico para Momentum Divergence"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.set_title(f'Momentum Divergence - {symbol} - {interval} - Señal {signal_type}')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: RSI Tradicional
            ax2.plot(dates_matplotlib, rsi_traditional[-50:], 'cyan', linewidth=2, label='RSI Tradicional')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI Tradicional')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: RSI Maverick
            ax3.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('RSI Maverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: Volumen
            volume = df['volume'].iloc[-50:].values
            colors_vol = []
            for i, signal in enumerate(volume_data['volume_signal'][-50:]):
                colors_vol.append('green' if signal == 'COMPRA' else 'red' if signal == 'VENTA' else 'gray')
            
            ax4.bar(dates_matplotlib, volume, color=colors_vol, alpha=0.6, label='Volumen')
            ax4.plot(dates_matplotlib, volume_data['volume_ma'][-50:], 'orange', linewidth=1, label='MA Volumen')
            
            # Marcar clusters
            cluster_indices = [i for i, cluster in enumerate(volume_data['volume_clusters'][-50:]) if cluster]
            if cluster_indices:
                for idx in cluster_indices:
                    ax4.scatter(dates_matplotlib[idx], volume[idx], color='purple', s=50, marker='x')
            
            ax4.set_ylabel('Volumen')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Momentum Divergence: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 2: ADX POWER TREND
    # ==============================================
    def check_adx_power_trend_signal(self, symbol, interval):
        """Estrategia ADX Power Trend"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['ADX Power Trend']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # ADX y DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            # MA21
            ma21 = self.calculate_sma(close, 21)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Detectar cruce DMI
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            # Condiciones para LONG
            if (adx[-1] > 30 and adx[-1] > adx[-2] and
                di_cross_bullish[-1] and
                close[-1] > ma21[-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT
            elif (adx[-1] > 30 and adx[-1] > adx[-2] and
                  di_cross_bearish[-1] and
                  close[-1] < ma21[-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_adx_power_chart(symbol, interval, df, adx, plus_di, minus_di, 
                                                        ma21, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'ADX POWER TREND',
                'chart': chart_buffer,
                'filters': [
                    f'ADX > 30 y creciente: {adx[-1]:.1f}',
                    f'{"+DI > -DI" if signal_type == "LONG" else "-DI > +DI"} cruce confirmado',
                    f'Precio {" > " if signal_type == "LONG" else " < "} MA21',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_adx_power_trend_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_adx_power_chart(self, symbol, interval, df, adx, plus_di, minus_di, ma21, ftm_data, signal_type):
        """Generar gráfico para ADX Power Trend"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Gráfico 1: Precio con MA21
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma21[-50:], 'blue', linewidth=2, label='MA21')
            
            ax1.set_title(f'ADX Power Trend - {symbol} - {interval} - Señal {signal_type}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: ADX y DMI
            ax2.plot(dates_matplotlib, adx[-50:], 'black', linewidth=2, label='ADX')
            ax2.plot(dates_matplotlib, plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax2.plot(dates_matplotlib, minus_di[-50:], 'red', linewidth=1, label='-DI')
            ax2.axhline(y=30, color='yellow', linestyle='--', alpha=0.7, label='Umbral 30')
            ax2.set_ylabel('ADX/DMI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            ax3.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax3.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax3.set_ylabel('Fuerza Tendencia')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico ADX Power Trend: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 3: MACD HISTOGRAM REVERSAL
    # ==============================================
    def check_macd_histogram_reversal_signal(self, symbol, interval):
        """Estrategia MACD Histogram Reversal"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['MACD Histogram Reversal']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # MACD
            macd, signal, histogram = self.calculate_macd(close)
            
            # RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # MA9 y MA21
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Detectar reversión del histograma
            hist_reversal_bullish = histogram[-1] > 0 and histogram[-2] < 0 and histogram[-3] < histogram[-2]
            hist_reversal_bearish = histogram[-1] < 0 and histogram[-2] > 0 and histogram[-3] > histogram[-2]
            
            # Cruce MA9/MA21
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover(ma9, ma21)
            
            # Condiciones para LONG
            if (hist_reversal_bullish and
                ma_cross_bullish[-1] and
                0.3 < rsi_maverick[-1] < 0.7 and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT
            elif (hist_reversal_bearish and
                  ma_cross_bearish[-1] and
                  0.3 < rsi_maverick[-1] < 0.7 and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_macd_histogram_chart(symbol, interval, df, macd, signal, histogram, 
                                                             ma9, ma21, rsi_maverick, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'MACD HISTOGRAM REVERSAL',
                'chart': chart_buffer,
                'filters': [
                    f'Reversión histograma MACD confirmada',
                    f'Cruce MA9/MA21 en dirección {signal_type}',
                    f'RSI Maverick neutral: {rsi_maverick[-1]:.2f}',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_macd_histogram_reversal_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_macd_histogram_chart(self, symbol, interval, df, macd, signal, histogram, 
                                     ma9, ma21, rsi_maverick, ftm_data, signal_type):
        """Generar gráfico para MACD Histogram Reversal"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gráfico 1: Precio con MA9 y MA21
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma9[-50:], 'red', linewidth=1, label='MA9')
            ax1.plot(dates_matplotlib, ma21[-50:], 'blue', linewidth=1, label='MA21')
            
            ax1.set_title(f'MACD Histogram Reversal - {symbol} - {interval}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: MACD
            ax2.plot(dates_matplotlib, macd[-50:], 'blue', linewidth=1, label='MACD')
            ax2.plot(dates_matplotlib, signal[-50:], 'red', linewidth=1, label='Señal')
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Histograma MACD
            colors_hist = ['green' if x > 0 else 'red' for x in histogram[-50:]]
            ax3.bar(dates_matplotlib, histogram[-50:], color=colors_hist, alpha=0.6)
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax3.set_ylabel('MACD Histogram')
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: RSI Maverick
            ax4.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=1, label='RSI Maverick')
            ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
            ax4.axhline(y=0.2, color='green', linestyle='--', alpha=0.3)
            ax4.axhline(y=0.5, color='gray', linestyle='-', alpha=0.2)
            
            ax4.set_ylabel('RSI Maverick')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Señal {signal_type} - {symbol} - {interval}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico MACD Histogram: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 4: VOLUME SPIKE MOMENTUM
    # ==============================================
    def check_volume_spike_momentum_signal(self, symbol, interval):
        """Estrategia Volume Spike Momentum"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Volume Spike Momentum']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # Volume anomaly
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            # MA21
            ma21 = self.calculate_sma(close, 21)
            
            # RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Verificar cluster de volumen
            recent_clusters = volume_data['volume_clusters'][-5:]
            has_cluster = sum(recent_clusters) >= 2
            
            if not has_cluster:
                return None
            
            # Determinar señal basada en último volumen signal
            last_signal = volume_data['volume_signal'][-1]
            
            if last_signal == 'COMPRA' and close[-1] > ma21[-1] and 0.3 < rsi_maverick[-1] < 0.7:
                signal_type = 'LONG'
            elif last_signal == 'VENTA' and close[-1] < ma21[-1] and 0.3 < rsi_maverick[-1] < 0.7:
                signal_type = 'SHORT'
            else:
                return None
            
            # Verificar FTMaverick en dirección correcta
            if signal_type == 'LONG' and ftm_data['strength_signals'][-1] not in ['STRONG_UP', 'WEAK_UP']:
                return None
            if signal_type == 'SHORT' and ftm_data['strength_signals'][-1] not in ['STRONG_DOWN', 'WEAK_DOWN']:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_volume_spike_chart(symbol, interval, df, volume_data, ma21, 
                                                           rsi_maverick, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'VOLUME SPIKE MOMENTUM',
                'chart': chart_buffer,
                'filters': [
                    f'Clúster de volumen confirmado ({sum(recent_clusters)} anomalías)',
                    f'Volumen signal: {last_signal}',
                    f'Precio {" > " if signal_type == "LONG" else " < "} MA21',
                    f'RSI Maverick neutral: {rsi_maverick[-1]:.2f}',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_volume_spike_momentum_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_volume_spike_chart(self, symbol, interval, df, volume_data, ma21, rsi_maverick, 
                                   ftm_data, signal_type):
        """Generar gráfico para Volume Spike Momentum"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con MA21
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma21[-50:], 'blue', linewidth=2, label='MA21')
            ax1.set_title(f'Volume Spike Momentum - {symbol} - {interval} - Señal {signal_type}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Volumen
            volume = df['volume'].iloc[-50:].values
            colors_vol = []
            for i, signal in enumerate(volume_data['volume_signal'][-50:]):
                colors_vol.append('green' if signal == 'COMPRA' else 'red' if signal == 'VENTA' else 'gray')
            
            ax2.bar(dates_matplotlib, volume, color=colors_vol, alpha=0.6, label='Volumen')
            ax2.plot(dates_matplotlib, volume_data['volume_ma'][-50:], 'orange', linewidth=1, label='MA Volumen')
            
            ax2.set_ylabel('Volumen')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: RSI Maverick
            ax3.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.3)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.2)
            ax3.set_ylabel('RSI Maverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            ax4.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Volume Spike: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 5: DOUBLE CONFIRMATION RSI
    # ==============================================
    def check_double_confirmation_rsi_signal(self, symbol, interval):
        """Estrategia Double Confirmation RSI"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Double Confirmation RSI']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # RSI Tradicional
            rsi_traditional = self.calculate_rsi(close, 14)
            
            # RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Condiciones para LONG (sobreventa)
            if (rsi_traditional[-1] < 30 and rsi_traditional[-2] >= 30 and
                rsi_maverick[-1] < 0.2 and rsi_maverick[-2] >= 0.2 and
                close[-1] <= bb_lower[-1] * 1.02 and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT (sobrecompra)
            elif (rsi_traditional[-1] > 70 and rsi_traditional[-2] <= 70 and
                  rsi_maverick[-1] > 0.8 and rsi_maverick[-2] <= 0.8 and
                  close[-1] >= bb_upper[-1] * 0.98 and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_double_rsi_chart(symbol, interval, df, rsi_traditional, 
                                                         rsi_maverick, bb_upper, bb_middle, bb_lower, 
                                                         ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'DOUBLE CONFIRMATION RSI',
                'chart': chart_buffer,
                'filters': [
                    f'RSI Tradicional {"sale de sobreventa" if signal_type == "LONG" else "sale de sobrecompra"}',
                    f'RSI Maverick {"sale de sobreventa" if signal_type == "LONG" else "sale de sobrecompra"}',
                    f'Precio cerca banda {"inferior" if signal_type == "LONG" else "superior"} BB',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_double_confirmation_rsi_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_double_rsi_chart(self, symbol, interval, df, rsi_traditional, rsi_maverick, 
                                 bb_upper, bb_middle, bb_lower, ftm_data, signal_type):
        """Generar gráfico para Double Confirmation RSI"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            # Gráfico 1: Precio con Bollinger Bands
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, bb_upper[-50:], 'orange', alpha=0.7, linewidth=1, label='BB Superior')
            ax1.plot(dates_matplotlib, bb_middle[-50:], 'orange', alpha=0.5, linewidth=1, label='BB Media')
            ax1.plot(dates_matplotlib, bb_lower[-50:], 'orange', alpha=0.7, linewidth=1, label='BB Inferior')
            ax1.fill_between(dates_matplotlib, bb_lower[-50:], bb_upper[-50:], color='orange', alpha=0.1)
            
            ax1.set_title(f'Double Confirmation RSI - {symbol} - {interval} - Señal {signal_type}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: RSI Tradicional
            ax2.plot(dates_matplotlib, rsi_traditional[-50:], 'cyan', linewidth=2, label='RSI Tradicional')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI Tradicional')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: RSI Maverick
            ax3.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('RSI Maverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            ax4.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.set_ylabel('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Double RSI: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 6: TREND STRENGTH MAVERICK
    # ==============================================
    def check_trend_strength_maverick_signal(self, symbol, interval):
        """Estrategia Trend Strength Maverick"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Trend Strength Maverick']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            
            # MA50
            ma50 = self.calculate_sma(close, 50)
            
            # Volume confirmation
            volume = df['volume'].values
            volume_ma = self.calculate_sma(volume, 20)
            volume_confirmation = volume[-1] > volume_ma[-1]
            
            # Solo operar señales STRONG
            if ftm_data['strength_signals'][-1] not in ['STRONG_UP', 'STRONG_DOWN']:
                return None
            
            # Evitar No-Trade Zones
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Determinar señal
            if (ftm_data['strength_signals'][-1] == 'STRONG_UP' and
                close[-1] > ma50[-1] and
                volume_confirmation):
                
                signal_type = 'LONG'
                
            elif (ftm_data['strength_signals'][-1] == 'STRONG_DOWN' and
                  close[-1] < ma50[-1] and
                  volume_confirmation):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_trend_strength_chart(symbol, interval, df, ftm_data, ma50, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'TREND STRENGTH MAVERICK',
                'chart': chart_buffer,
                'filters': [
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Precio {" > " if signal_type == "LONG" else " < "} MA50',
                    'Volumen > MA Volumen confirmado',
                    'Zona No-Operar evitada'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_trend_strength_maverick_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_trend_strength_chart(self, symbol, interval, df, ftm_data, ma50, signal_type):
        """Generar gráfico para Trend Strength Maverick"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Gráfico 1: Precio con MA50
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma50[-50:], 'green', linewidth=2, label='MA50')
            ax1.set_title(f'Trend Strength Maverick - {symbol} - {interval} - Señal {signal_type}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Fuerza de Tendencia Maverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            for i in range(len(dates_matplotlib)):
                ax2.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            # Líneas de umbral
            ax2.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', 
                       alpha=0.7, label='Umbral Alto')
            ax2.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            
            ax2.set_ylabel('Fuerza Tendencia %')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Bandas FTMaverick
            ax3.plot(dates_matplotlib, ftm_data['upper_band'][-50:], 'orange', alpha=0.5, linewidth=1, label='Banda Superior')
            ax3.plot(dates_matplotlib, ftm_data['basis'][-50:], 'orange', alpha=0.7, linewidth=1, label='Base')
            ax3.plot(dates_matplotlib, ftm_data['lower_band'][-50:], 'orange', alpha=0.5, linewidth=1, label='Banda Inferior')
            ax3.fill_between(dates_matplotlib, ftm_data['lower_band'][-50:], ftm_data['upper_band'][-50:], 
                           color='orange', alpha=0.1)
            
            # Precio
            ax3.plot(dates_matplotlib, closes, 'blue', alpha=0.7, linewidth=1, label='Precio')
            
            ax3.set_ylabel('Bandas FTMaverick')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Trend Strength: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 7: MA CONVERGENCE DIVERGENCE
    # ==============================================
    def check_ma_convergence_signal(self, symbol, interval):
        """Estrategia MA Convergence Divergence"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['MA Convergence Divergence']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            # Medias móviles
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            ma50 = self.calculate_sma(close, 50)
            
            # MACD
            macd, signal, histogram = self.calculate_macd(close)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Verificar alineación
            ma_aligned_bullish = close[-1] > ma9[-1] > ma21[-1] > ma50[-1]
            ma_aligned_bearish = close[-1] < ma9[-1] < ma21[-1] < ma50[-1]
            
            # Separación mínima entre MAs (1% del precio)
            separation_ok_bullish = (ma9[-1] - ma21[-1]) > close[-1] * 0.01 and (ma21[-1] - ma50[-1]) > close[-1] * 0.01
            separation_ok_bearish = (ma21[-1] - ma9[-1]) > close[-1] * 0.01 and (ma50[-1] - ma21[-1]) > close[-1] * 0.01
            
            # Condiciones para LONG
            if (ma_aligned_bullish and separation_ok_bullish and
                histogram[-1] > 0 and histogram[-2] <= 0 and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT
            elif (ma_aligned_bearish and separation_ok_bearish and
                  histogram[-1] < 0 and histogram[-2] >= 0 and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_ma_convergence_chart(symbol, interval, df, ma9, ma21, ma50, 
                                                             macd, histogram, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'MA CONVERGENCE DIVERGENCE',
                'chart': chart_buffer,
                'filters': [
                    f'Alineación MA9 > MA21 > MA50 confirmada',
                    f'Separación >1% entre medias',
                    f'Histograma MACD positivo para LONG/negativo para SHORT',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_ma_convergence_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_ma_convergence_chart(self, symbol, interval, df, ma9, ma21, ma50, 
                                     macd, histogram, ftm_data, signal_type):
        """Generar gráfico para MA Convergence"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Gráfico 1: Precio con MAs
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma9[-50:], 'red', linewidth=1, label='MA9')
            ax1.plot(dates_matplotlib, ma21[-50:], 'blue', linewidth=1, label='MA21')
            ax1.plot(dates_matplotlib, ma50[-50:], 'green', linewidth=1, label='MA50')
            
            ax1.set_title(f'MA Convergence - {symbol} - {interval} - Señal {signal_type}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: MACD Histogram
            colors_hist = ['green' if x > 0 else 'red' for x in histogram[-50:]]
            ax2.bar(dates_matplotlib, histogram[-50:], color=colors_hist, alpha=0.6)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.set_ylabel('MACD Histogram')
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            ax3.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax3.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax3.set_ylabel('Fuerza Tendencia')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico MA Convergence: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 8: VOLUME-PRICE DIVERGENCE
    # ==============================================
    def check_volume_price_divergence_signal(self, symbol, interval):
        """Estrategia Volume-Price Divergence"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Volume-Price Divergence']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            # RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Detectar divergencia volumen/precio (5 velas)
            lookback = 5
            if len(close) >= lookback + 1:
                # Para LONG: precio hace nuevo bajo pero volumen decrece
                price_lows = close[-lookback-1:-1]
                volume_avgs = volume[-lookback-1:-1]
                
                new_price_low = close[-1] < np.min(price_lows)
                volume_decreasing = volume[-1] < np.mean(volume_avgs) * 0.8
                
                # Para SHORT: precio hace nuevo alto pero volumen decrece
                price_highs = close[-lookback-1:-1]
                new_price_high = close[-1] > np.max(price_highs)
                
                # Divergencia RSI Maverick
                rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_maverick, lookback=lookback)
                
                # Condiciones para LONG
                if (new_price_low and volume_decreasing and rsi_bullish[-1] and
                    ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                    
                    signal_type = 'LONG'
                    
                # Condiciones para SHORT
                elif (new_price_high and volume_decreasing and rsi_bearish[-1] and
                      ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                    
                    signal_type = 'SHORT'
                else:
                    return None
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(
                df['high'].values, df['low'].values, close
            )
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_volume_price_div_chart(symbol, interval, df, volume, 
                                                               rsi_maverick, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'VOLUME-PRICE DIVERGENCE',
                'chart': chart_buffer,
                'filters': [
                    f'Divergencia precio/volumen confirmada',
                    f'Divergencia RSI Maverick confirmada',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_volume_price_divergence_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_volume_price_div_chart(self, symbol, interval, df, volume, rsi_maverick, 
                                       ftm_data, signal_type):
        """Generar gráfico para Volume-Price Divergence"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gráfico 1: Precio
            dates = df['timestamp'].iloc[-20:].values
            closes = df['close'].iloc[-20:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-20+i]
                close_price = df['close'].iloc[-20+i]
                high_price = df['high'].iloc[-20+i]
                low_price = df['low'].iloc[-20+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.set_title(f'Precio - {symbol} - {interval}')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Volumen
            volume_vals = volume[-20:]
            colors_vol = []
            for i in range(len(closes)):
                if i == 0:
                    colors_vol.append('gray')
                else:
                    colors_vol.append('green' if closes[i] > closes[i-1] else 'red')
            
            ax2.bar(dates_matplotlib, volume_vals, color=colors_vol, alpha=0.6)
            ax2.set_title('Volumen')
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: RSI Maverick
            ax3.plot(dates_matplotlib, rsi_maverick[-20:], 'blue', linewidth=2)
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
            ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.3)
            ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.2)
            ax3.set_title('RSI Maverick')
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_strength = ftm_data['trend_strength'][-20:]
            colors = ftm_data['colors'][-20:]
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            ax4.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.set_title('Fuerza Tendencia')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Volume-Price Divergence - Señal {signal_type} - {symbol} - {interval}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Volume-Price Divergence: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 9: STOCHASTIC RSI COMBO
    # ==============================================
    def check_stochastic_rsi_combo_signal(self, symbol, interval):
        """Estrategia Stochastic RSI Combo - Nueva estrategia"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Stochastic RSI Combo']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # RSI Estocástico
            stoch_rsi_data = self.calculate_stochastic_rsi(close)
            stoch_rsi = stoch_rsi_data['stoch_rsi']
            k_line = stoch_rsi_data['k_line']
            d_line = stoch_rsi_data['d_line']
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Soporte/Resistencia
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            # Verificar si está en soporte/resistencia
            current_price = close[-1]
            near_support = any(abs(current_price - s) / s < 0.02 for s in support_levels[:2])
            near_resistance = any(abs(current_price - r) / r < 0.02 for r in resistance_levels[:2])
            
            # Cruce K/D en zonas extremas
            k_d_cross_bullish = k_line[-1] > d_line[-1] and k_line[-2] <= d_line[-2]
            k_d_cross_bearish = k_line[-1] < d_line[-1] and k_line[-2] >= d_line[-2]
            
            # Condiciones para LONG (suelo)
            if (k_d_cross_bullish and stoch_rsi[-1] < 20 and
                near_support and current_price <= bb_lower[-1] * 1.02 and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT (techo)
            elif (k_d_cross_bearish and stoch_rsi[-1] > 80 and
                  near_resistance and current_price >= bb_upper[-1] * 0.98 and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_stochastic_rsi_combo_chart(symbol, interval, df, stoch_rsi_data, 
                                                                   bb_upper, bb_lower, ftm_data, 
                                                                   support_levels, resistance_levels, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'STOCHASTIC RSI COMBO',
                'chart': chart_buffer,
                'filters': [
                    f'RSI Estocástico {"sobreventa" if signal_type == "LONG" else "sobrecompra"}',
                    f'Cruce K/D confirmado',
                    f'Precio cerca {"soporte" if signal_type == "LONG" else "resistencia"}',
                    f'Precio cerca banda {"inferior" if signal_type == "LONG" else "superior"} BB',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_stochastic_rsi_combo_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_stochastic_rsi_combo_chart(self, symbol, interval, df, stoch_rsi_data, 
                                           bb_upper, bb_lower, ftm_data, 
                                           support_levels, resistance_levels, signal_type):
        """Generar gráfico para Stochastic RSI Combo"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gráfico 1: Precio con Bollinger Bands y S/R
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            # Bollinger Bands
            ax1.plot(dates_matplotlib, bb_upper[-50:], 'orange', alpha=0.5, linewidth=1, label='BB Superior')
            ax1.plot(dates_matplotlib, bb_lower[-50:], 'orange', alpha=0.5, linewidth=1, label='BB Inferior')
            
            # Soporte/Resistencia
            for i, level in enumerate(support_levels[:2]):
                ax1.axhline(y=level, color='green', linestyle='--', alpha=0.5, label=f'Soporte {i+1}' if i == 0 else "")
            
            for i, level in enumerate(resistance_levels[:2]):
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5, label=f'Resistencia {i+1}' if i == 0 else "")
            
            ax1.set_title(f'Precio con S/R - {symbol} - {interval}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: RSI Estocástico
            stoch_rsi = stoch_rsi_data['stoch_rsi'][-50:]
            k_line = stoch_rsi_data['k_line'][-50:]
            d_line = stoch_rsi_data['d_line'][-50:]
            
            ax2.plot(dates_matplotlib, stoch_rsi, 'blue', linewidth=1, label='RSI Estocástico')
            ax2.plot(dates_matplotlib, k_line, 'green', linewidth=1, label='%K')
            ax2.plot(dates_matplotlib, d_line, 'red', linewidth=1, label='%D')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.3)
            ax2.set_title('RSI Estocástico')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            for i in range(len(dates_matplotlib)):
                ax3.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            ax3.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax3.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax3.set_title('Fuerza Tendencia Maverick')
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: Histograma divergencias
            price_changes = np.diff(closes)
            volume = df['volume'].iloc[-50:].values
            
            colors_div = []
            for i in range(len(price_changes)):
                if price_changes[i] > 0 and volume[i+1] < volume[i]:
                    colors_div.append('green')  # Divergencia alcista
                elif price_changes[i] < 0 and volume[i+1] > volume[i]:
                    colors_div.append('red')    # Divergencia bajista
                else:
                    colors_div.append('gray')
            
            ax4.bar(dates_matplotlib[1:], price_changes, color=colors_div, alpha=0.6)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title('Divergencias Precio/Volumen')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Stochastic RSI Combo - Señal {signal_type} - {symbol} - {interval}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Stochastic RSI Combo: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 10: WHALE DMI COMBO
    # ==============================================
    def check_whale_dmi_combo_signal(self, symbol, interval):
        """Estrategia Whale DMI Combo - Nueva estrategia (12h, 1D)"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Whale DMI Combo']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Whale signals (con señal extendida 7 velas)
            whale_data = self.calculate_whale_signals_improved(df)
            
            # ADX y DMI
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Medias móviles para tendencia
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            
            # Usar señal extendida de ballenas (7 velas)
            whale_signal_active = whale_data['extended_buy'][-1] or whale_data['extended_sell'][-1]
            
            if not whale_signal_active:
                return None
            
            # Detectar cruce DMI
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            # Condiciones para LONG
            if (whale_data['extended_buy'][-1] and di_cross_bullish[-1] and
                adx[-1] > 25 and close[-1] > ma9[-1] and close[-1] > ma21[-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT
            elif (whale_data['extended_sell'][-1] and di_cross_bearish[-1] and
                  adx[-1] > 25 and close[-1] < ma9[-1] and close[-1] < ma21[-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_whale_dmi_combo_chart(symbol, interval, df, whale_data, 
                                                              adx, plus_di, minus_di, 
                                                              ma9, ma21, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'WHALE DMI COMBO',
                'chart': chart_buffer,
                'filters': [
                    f'Señal ballenas extendida (7 velas) confirmada',
                    f'{"+DI > -DI" if signal_type == "LONG" else "-DI > +DI"} cruce confirmado',
                    f'ADX > 25: {adx[-1]:.1f}',
                    f'Precio {" > " if signal_type == "LONG" else " < "} MA9 y MA21',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_whale_dmi_combo_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_whale_dmi_combo_chart(self, symbol, interval, df, whale_data, adx, plus_di, minus_di, 
                                      ma9, ma21, ftm_data, signal_type):
        """Generar gráfico para Whale DMI Combo"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gráfico 1: Precio con MAs
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.plot(dates_matplotlib, ma9[-50:], 'red', linewidth=1, label='MA9')
            ax1.plot(dates_matplotlib, ma21[-50:], 'blue', linewidth=1, label='MA21')
            ax1.set_title(f'Precio - {symbol} - {interval}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Señales de Ballenas
            whale_pump = whale_data['whale_pump'][-50:]
            whale_dump = whale_data['whale_dump'][-50:]
            extended_buy = whale_data['extended_buy'][-50:]
            extended_sell = whale_data['extended_sell'][-50:]
            
            ax2.bar(dates_matplotlib, whale_pump, color='green', alpha=0.6, label='Ballenas Compradoras')
            ax2.bar(dates_matplotlib, [-x for x in whale_dump], color='red', alpha=0.6, label='Ballenas Vendedoras')
            
            # Marcar señales extendidas
            buy_indices = [i for i, extended in enumerate(extended_buy) if extended]
            sell_indices = [i for i, extended in enumerate(extended_sell) if extended]
            
            if buy_indices:
                for idx in buy_indices:
                    ax2.scatter(dates_matplotlib[idx], whale_pump[idx], 
                              color='darkgreen', s=30, marker='^', label='Compra Extendida' if idx == buy_indices[0] else "")
            
            if sell_indices:
                for idx in sell_indices:
                    ax2.scatter(dates_matplotlib[idx], -whale_dump[idx], 
                              color='darkred', s=30, marker='v', label='Venta Extendida' if idx == sell_indices[0] else "")
            
            ax2.set_title('Señales de Ballenas')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: ADX y DMI
            ax3.plot(dates_matplotlib, adx[-50:], 'black', linewidth=2, label='ADX')
            ax3.plot(dates_matplotlib, plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax3.plot(dates_matplotlib, minus_di[-50:], 'red', linewidth=1, label='-DI')
            ax3.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
            ax3.set_title('ADX/DMI')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            ax4.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.set_title('Fuerza Tendencia Maverick')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Whale DMI Combo - Señal {signal_type} - {symbol} - {interval}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Whale DMI Combo: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 11: SUPPORT RESISTANCE BOUNCE
    # ==============================================
    def check_support_resistance_bounce_signal(self, symbol, interval):
        """Estrategia Support Resistance Bounce - Nueva estrategia"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Support Resistance Bounce']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # RSI Estocástico
            stoch_rsi_data = self.calculate_stochastic_rsi(close)
            stoch_rsi = stoch_rsi_data['stoch_rsi']
            k_line = stoch_rsi_data['k_line']
            d_line = stoch_rsi_data['d_line']
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Soporte/Resistencia dinámicos
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            # Volume anomaly
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Verificar rebote en soporte/resistencia
            current_price = close[-1]
            price_tolerance = 0.02  # 2%
            
            # Buscar soporte más cercano para LONG
            nearest_support = None
            for support in support_levels:
                if support < current_price and abs(current_price - support) / current_price < price_tolerance:
                    nearest_support = support
                    break
            
            # Buscar resistencia más cercana para SHORT
            nearest_resistance = None
            for resistance in resistance_levels:
                if resistance > current_price and abs(current_price - resistance) / current_price < price_tolerance:
                    nearest_resistance = resistance
                    break
            
            # Condiciones para LONG (rebote en soporte)
            if (nearest_support is not None and
                current_price <= nearest_support * 1.01 and  # Precio cerca del soporte
                stoch_rsi[-1] < 30 and  # RSI Estocástico en sobreventa
                k_line[-1] > d_line[-1] and k_line[-2] <= d_line[-2] and  # Cruce K/D alcista
                volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'COMPRA' and
                current_price <= bb_lower[-1] * 1.02 and  # Precio cerca banda inferior BB
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT (rechazo en resistencia)
            elif (nearest_resistance is not None and
                  current_price >= nearest_resistance * 0.99 and  # Precio cerca de la resistencia
                  stoch_rsi[-1] > 70 and  # RSI Estocástico en sobrecompra
                  k_line[-1] < d_line[-1] and k_line[-2] >= d_line[-2] and  # Cruce K/D bajista
                  volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'VENTA' and
                  current_price >= bb_upper[-1] * 0.98 and  # Precio cerca banda superior BB
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_support_resistance_bounce_chart(symbol, interval, df, 
                                                                       stoch_rsi_data, support_levels, 
                                                                       resistance_levels, volume_data, 
                                                                       bb_upper, bb_lower, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'SUPPORT RESISTANCE BOUNCE',
                'chart': chart_buffer,
                'filters': [
                    f'Rebote en {"soporte" if signal_type == "LONG" else "resistencia"} confirmado',
                    f'RSI Estocástico {"sobreventa" if signal_type == "LONG" else "sobrecompra"}',
                    f'Cruce K/D {"alcista" if signal_type == "LONG" else "bajista"} confirmado',
                    f'Clúster de volumen confirmado',
                    f'Precio cerca banda {"inferior" if signal_type == "LONG" else "superior"} BB',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_support_resistance_bounce_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_support_resistance_bounce_chart(self, symbol, interval, df, stoch_rsi_data, 
                                                support_levels, resistance_levels, volume_data, 
                                                bb_upper, bb_lower, ftm_data, signal_type):
        """Generar gráfico para Support Resistance Bounce"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gráfico 1: Precio con S/R y Bollinger Bands
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            # Bollinger Bands
            ax1.plot(dates_matplotlib, bb_upper[-50:], 'orange', alpha=0.5, linewidth=1, label='BB Superior')
            ax1.plot(dates_matplotlib, bb_lower[-50:], 'orange', alpha=0.5, linewidth=1, label='BB Inferior')
            
            # Soportes y resistencias clave
            for level in support_levels[:2]:
                ax1.axhline(y=level, color='green', linestyle='--', alpha=0.7)
            
            for level in resistance_levels[:2]:
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.7)
            
            ax1.set_title(f'Precio con S/R - {symbol} - {interval}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: RSI Estocástico
            stoch_rsi = stoch_rsi_data['stoch_rsi'][-50:]
            k_line = stoch_rsi_data['k_line'][-50:]
            d_line = stoch_rsi_data['d_line'][-50:]
            
            ax2.plot(dates_matplotlib, stoch_rsi, 'blue', linewidth=1, label='RSI Estocástico')
            ax2.plot(dates_matplotlib, k_line, 'green', linewidth=1, label='%K')
            ax2.plot(dates_matplotlib, d_line, 'red', linewidth=1, label='%D')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.3)
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.2)
            ax2.set_title('RSI Estocástico')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Volumen y anomalías
            volume = df['volume'].iloc[-50:].values
            colors_vol = []
            for i, signal in enumerate(volume_data['volume_signal'][-50:]):
                colors_vol.append('green' if signal == 'COMPRA' else 'red' if signal == 'VENTA' else 'gray')
            
            ax3.bar(dates_matplotlib, volume, color=colors_vol, alpha=0.6, label='Volumen')
            ax3.plot(dates_matplotlib, volume_data['volume_ma'][-50:], 'orange', linewidth=1, label='MA Volumen')
            
            # Marcar clusters
            cluster_indices = [i for i, cluster in enumerate(volume_data['volume_clusters'][-50:]) if cluster]
            if cluster_indices:
                for idx in cluster_indices:
                    ax3.scatter(dates_matplotlib[idx], volume[idx], color='purple', s=30, marker='x')
            
            ax3.set_title('Volumen con Anomalías')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_strength = ftm_data['trend_strength'][-50:]
            colors = ftm_data['colors'][-50:]
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            ax4.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.set_title('Fuerza Tendencia Maverick')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Support Resistance Bounce - Señal {signal_type} - {symbol} - {interval}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Support Resistance Bounce: {e}")
            return None
    
    # ==============================================
    # ESTRATEGIA 12: MULTI-TIMEFRAME CONFIRMATION
    # ==============================================
    def check_multi_timeframe_confirmation_signal(self, symbol, interval):
        """Estrategia Multi-Timeframe Confirmation - Nueva estrategia"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Multi-Timeframe Confirmation']:
            return None
        
        try:
            # Obtener datos de temporalidad actual
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # RSI Estocástico en temporalidad actual
            stoch_rsi_data = self.calculate_stochastic_rsi(close)
            stoch_rsi = stoch_rsi_data['stoch_rsi']
            k_line = stoch_rsi_data['k_line']
            d_line = stoch_rsi_data['d_line']
            
            # FTMaverick
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            # Obtener jerarquía de temporalidades
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return None
            
            # Obtener datos de temporalidad menor (2h)
            menor_df = self.get_kucoin_data(symbol, '2h', 50)
            if menor_df is None or len(menor_df) < 20:
                return None
            
            menor_close = menor_df['close'].values
            menor_stoch_rsi_data = self.calculate_stochastic_rsi(menor_close)
            menor_stoch_rsi = menor_stoch_rsi_data['stoch_rsi']
            
            # Obtener datos de temporalidad mayor (8h para 4h, 1D para 12h)
            if interval == '4h':
                mayor_tf = '8h'
            elif interval == '12h':
                mayor_tf = '1D'
            elif interval == '1D':
                mayor_tf = '1W'
            else:
                return None
            
            mayor_df = self.get_kucoin_data(symbol, mayor_tf, 50)
            if mayor_df is None or len(mayor_df) < 20:
                return None
            
            mayor_close = mayor_df['close'].values
            mayor_ma50 = self.calculate_sma(mayor_close, 50)
            
            # Soporte/Resistencia
            support_levels, resistance_levels = self.calculate_dynamic_support_resistance(high, low, close)
            
            # Volume confirmation
            volume = df['volume'].values
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            # Condiciones para LONG (confirmación multi-temporalidad)
            current_price = close[-1]
            
            # Condición temporalidad actual: RSI Estocástico en sobreventa y cruce K/D
            current_condition = (stoch_rsi[-1] < 30 and 
                               k_line[-1] > d_line[-1] and 
                               k_line[-2] <= d_line[-2])
            
            # Condición temporalidad menor: confirmación con RSI Estocástico
            menor_condition = menor_stoch_rsi[-1] < 40
            
            # Condición temporalidad mayor: precio > MA50
            mayor_condition = current_price > mayor_ma50[-1] if len(mayor_ma50) > 0 else True
            
            if (current_condition and menor_condition and mayor_condition and
                volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'COMPRA' and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'LONG'
                
            # Condiciones para SHORT (confirmación multi-temporalidad)
            # Condición temporalidad actual: RSI Estocástico en sobrecompra y cruce K/D
            current_condition_short = (stoch_rsi[-1] > 70 and 
                                     k_line[-1] < d_line[-1] and 
                                     k_line[-2] >= d_line[-2])
            
            # Condición temporalidad menor: confirmación con RSI Estocástico
            menor_condition_short = menor_stoch_rsi[-1] > 60
            
            # Condición temporalidad mayor: precio < MA50
            mayor_condition_short = current_price < mayor_ma50[-1] if len(mayor_ma50) > 0 else True
            
            if (current_condition_short and menor_condition_short and mayor_condition_short and
                volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'VENTA' and
                ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'SHORT'
            else:
                return None
            
            # Calcular niveles
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 15, support_levels, resistance_levels)
            
            # Generar gráfico
            chart_buffer = self.generate_multi_timeframe_confirmation_chart(symbol, interval, df, 
                                                                           stoch_rsi_data, menor_stoch_rsi_data,
                                                                           mayor_ma50, support_levels, 
                                                                           resistance_levels, volume_data, 
                                                                           ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'strategy': 'MULTI-TIMEFRAME CONFIRMATION',
                'chart': chart_buffer,
                'filters': [
                    f'RSI Estocástico {"sobreventa" if signal_type == "LONG" else "sobrecompra"} (TF actual)',
                    f'Cruce K/D {"alcista" if signal_type == "LONG" else "bajista"} confirmado',
                    f'Confirmación RSI Estocástico TF menor',
                    f'Precio {" > " if signal_type == "LONG" else " < "} MA50 TF mayor',
                    f'Clúster de volumen confirmado',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_multi_timeframe_confirmation_signal para {symbol} {interval}: {e}")
            return None
    
    def generate_multi_timeframe_confirmation_chart(self, symbol, interval, df, stoch_rsi_data, 
                                                   menor_stoch_rsi_data, mayor_ma50, 
                                                   support_levels, resistance_levels, 
                                                   volume_data, ftm_data, signal_type):
        """Generar gráfico para Multi-Timeframe Confirmation"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gráfico 1: Precio con S/R y MA50 mayor
            dates = df['timestamp'].iloc[-50:].values
            closes = df['close'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            # Línea de MA50 mayor (aproximada)
            if len(mayor_ma50) > 0:
                ma50_value = mayor_ma50[-1]
                ax1.axhline(y=ma50_value, color='purple', linestyle='--', alpha=0.7, label='MA50 (TF Mayor)')
            
            # Soportes y resistencias clave
            for level in support_levels[:2]:
                ax1.axhline(y=level, color='green', linestyle=':', alpha=0.5)
            
            for level in resistance_levels[:2]:
                ax1.axhline(y=level, color='red', linestyle=':', alpha=0.5)
            
            ax1.set_title(f'Precio - {symbol} - {interval}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: RSI Estocástico TF Actual vs TF Menor
            stoch_rsi = stoch_rsi_data['stoch_rsi'][-50:]
            k_line = stoch_rsi_data['k_line'][-50:]
            d_line = stoch_rsi_data['d_line'][-50:]
            menor_stoch_rsi = menor_stoch_rsi_data['stoch_rsi'][-50:]
            
            # Ajustar longitud si es necesario
            min_len = min(len(stoch_rsi), len(menor_stoch_rsi))
            if min_len < 50:
                stoch_rsi = stoch_rsi[-min_len:]
                k_line = k_line[-min_len:]
                d_line = d_line[-min_len:]
                menor_stoch_rsi = menor_stoch_rsi[-min_len:]
                dates_matplotlib = dates_matplotlib[-min_len:]
            
            ax2.plot(dates_matplotlib, stoch_rsi, 'blue', linewidth=2, label='RSI Estocástico (TF Actual)')
            ax2.plot(dates_matplotlib, k_line, 'green', linewidth=1, label='%K')
            ax2.plot(dates_matplotlib, d_line, 'red', linewidth=1, label='%D')
            ax2.plot(dates_matplotlib, menor_stoch_rsi, 'orange', linewidth=1, label='RSI Estocástico (TF Menor)')
            
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.3)
            ax2.set_title('RSI Estocástico Multi-TF')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: Volumen
            volume = df['volume'].iloc[-len(dates_matplotlib):].values
            colors_vol = []
            volume_signal_slice = volume_data['volume_signal'][-len(dates_matplotlib):]
            
            for i, signal in enumerate(volume_signal_slice):
                colors_vol.append('green' if signal == 'COMPRA' else 'red' if signal == 'VENTA' else 'gray')
            
            ax3.bar(dates_matplotlib, volume, color=colors_vol, alpha=0.6, label='Volumen')
            ax3.plot(dates_matplotlib, volume_data['volume_ma'][-len(dates_matplotlib):], 'orange', linewidth=1, label='MA Volumen')
            ax3.set_title('Volumen con Señales')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: FTMaverick
            trend_strength = ftm_data['trend_strength'][-len(dates_matplotlib):]
            colors = ftm_data['colors'][-len(dates_matplotlib):]
            
            for i in range(len(dates_matplotlib)):
                ax4.bar(dates_matplotlib[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            ax4.axhline(y=ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.axhline(y=-ftm_data['high_zone_threshold'], color='orange', linestyle='--', alpha=0.7)
            ax4.set_title('Fuerza Tendencia Maverick')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multi-Timeframe Confirmation - Señal {signal_type} - {symbol} - {interval}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Multi-Timeframe Confirmation: {e}")
            return None
    
    # ==============================================
    # SISTEMA DE GENERACIÓN DE SEÑALES COMPLETO
    # ==============================================
    def generate_strategy_signals(self):
        """Generar señales para todas las estrategias"""
        all_signals = []
        
        # Intervalos a verificar (SOLO 4h, 12h, 1D, 1W)
        intervals_to_check = ['4h', '12h', '1D', '1W']
        
        current_time = self.get_bolivia_time()
        
        for interval in intervals_to_check:
            # Verificar si es momento de revisar este intervalo
            should_check = self.calculate_remaining_time(interval, current_time)
            if not should_check:
                continue
            
            for symbol in TOP_CRYPTO_SYMBOLS:
                try:
                    # Estrategia 1: Momentum Divergence
                    if interval in STRATEGY_TIMEFRAMES['Momentum Divergence']:
                        signal = self.check_momentum_divergence_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 2: ADX Power Trend
                    if interval in STRATEGY_TIMEFRAMES['ADX Power Trend']:
                        signal = self.check_adx_power_trend_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 3: MACD Histogram Reversal
                    if interval in STRATEGY_TIMEFRAMES['MACD Histogram Reversal']:
                        signal = self.check_macd_histogram_reversal_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 4: Volume Spike Momentum
                    if interval in STRATEGY_TIMEFRAMES['Volume Spike Momentum']:
                        signal = self.check_volume_spike_momentum_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 5: Double Confirmation RSI
                    if interval in STRATEGY_TIMEFRAMES['Double Confirmation RSI']:
                        signal = self.check_double_confirmation_rsi_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 6: Trend Strength Maverick
                    if interval in STRATEGY_TIMEFRAMES['Trend Strength Maverick']:
                        signal = self.check_trend_strength_maverick_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 7: MA Convergence Divergence
                    if interval in STRATEGY_TIMEFRAMES['MA Convergence Divergence']:
                        signal = self.check_ma_convergence_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 8: Volume-Price Divergence
                    if interval in STRATEGY_TIMEFRAMES['Volume-Price Divergence']:
                        signal = self.check_volume_price_divergence_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 9: Stochastic RSI Combo
                    if interval in STRATEGY_TIMEFRAMES['Stochastic RSI Combo']:
                        signal = self.check_stochastic_rsi_combo_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 10: Whale DMI Combo
                    if interval in STRATEGY_TIMEFRAMES['Whale DMI Combo']:
                        signal = self.check_whale_dmi_combo_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 11: Support Resistance Bounce
                    if interval in STRATEGY_TIMEFRAMES['Support Resistance Bounce']:
                        signal = self.check_support_resistance_bounce_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 12: Multi-Timeframe Confirmation
                    if interval in STRATEGY_TIMEFRAMES['Multi-Timeframe Confirmation']:
                        signal = self.check_multi_timeframe_confirmation_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                except Exception as e:
                    print(f"Error generando señales para {symbol} {interval}: {e}")
                    continue
        
        return all_signals

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert(alert_data):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Formatear mensaje para estrategias
        filters_text = '\n'.join(['• ' + f for f in alert_data.get('filters', [])])
        
        # Determinar recomendación según estrategia y temporalidad
        recommendation = "Swing Trading"
        if alert_data['interval'] in ['12h', '1D']:
            recommendation = "Swing Trading / Inversión"
        elif alert_data['interval'] == '1W':
            recommendation = "Inversión Spot Largo Plazo"
        
        message = f"""
🚨 Alerta {alert_data['signal']} {alert_data['symbol']} en {alert_data['interval']} 🚨
Estrategia: {alert_data['strategy']}
Precio actual: ${alert_data['current_price']:.2f} | Entrada: ${alert_data['entry']:.2f}
Stop Loss: ${alert_data['stop_loss']:.2f} | Take Profit: ${alert_data['take_profit'][0]:.2f}
Filtros: 
{filters_text}
Recomendación: {recommendation}.
"""
        
        # Enviar imagen
        if 'chart' in alert_data and alert_data['chart']:
            asyncio.run(bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=alert_data['chart'],
                caption=message
            ))
        else:
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            ))
        
        print(f"Alerta enviada a Telegram: {alert_data['symbol']} {alert_data['interval']} {alert_data['signal']} - {alert_data['strategy']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_strategy_checker():
    """Verificador de estrategias en segundo plano"""
    print("Background strategy checker iniciado...")
    
    # FRECUENCIAS DE REVISIÓN (segundos)
    interval_wait_times = {
        '4h': 660,    # 11 minutos
        '12h': 1380,  # 23 minutos
        '1D': 1380,   # 23 minutos
        '1W': 1920    # 32 minutos
    }
    
    # Última verificación por intervalo
    last_checks = {interval: datetime.now() for interval in interval_wait_times.keys()}
    
    while True:
        try:
            current_time = datetime.now()
            
            for interval, wait_time in interval_wait_times.items():
                # Verificar si es tiempo de revisar este intervalo
                if (current_time - last_checks[interval]).seconds >= wait_time:
                    print(f"Verificando estrategias para intervalo {interval}...")
                    
                    # Verificar si es momento de la vela para este intervalo
                    if indicator.calculate_remaining_time(interval, current_time):
                        # Generar señales para todas las estrategias
                        signals = indicator.generate_strategy_signals()
                        
                        # Filtrar señales para este intervalo
                        interval_signals = [s for s in signals if s['interval'] == interval]
                        
                        # Enviar alertas
                        for signal in interval_signals:
                            signal_key = f"{signal['symbol']}_{signal['interval']}_{signal['strategy']}_{signal['signal']}"
                            
                            # Verificar si ya enviamos esta señal recientemente
                            if signal_key not in indicator.strategy_signals:
                                send_telegram_alert(signal)
                                indicator.strategy_signals[signal_key] = current_time
                            else:
                                # Eliminar señales antiguas (más de 2 horas)
                                last_sent = indicator.strategy_signals[signal_key]
                                if (current_time - last_sent).seconds > 7200:
                                    send_telegram_alert(signal)
                                    indicator.strategy_signals[signal_key] = current_time
                    
                    last_checks[interval] = current_time
            
            time.sleep(10)
            
        except Exception as e:
            print(f"Error en background_strategy_checker: {e}")
            time.sleep(60)

# Iniciar verificador de estrategias en segundo plano
try:
    strategy_thread = Thread(target=background_strategy_checker, daemon=True)
    strategy_thread.start()
    print("Background strategy checker iniciado correctamente")
except Exception as e:
    print(f"Error iniciando background strategy checker: {e}")

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
        
        # Obtener datos básicos
        df = indicator.get_kucoin_data(symbol, interval, 100)
        if df is None or len(df) < 50:
            return jsonify({'error': 'No hay datos disponibles'}), 400
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calcular todos los indicadores necesarios
        whale_data = indicator.calculate_whale_signals_improved(df)
        adx, plus_di, minus_di = indicator.calculate_adx(high, low, close, di_period)
        rsi_traditional = indicator.calculate_rsi(close, rsi_length)
        rsi_maverick = indicator.calculate_rsi_maverick(close)
        stoch_rsi_data = indicator.calculate_stochastic_rsi(close)
        
        # Medias móviles
        ma_9 = indicator.calculate_sma(close, 9)
        ma_21 = indicator.calculate_sma(close, 21)
        ma_50 = indicator.calculate_sma(close, 50)
        ma_200 = indicator.calculate_sma(close, 200)
        
        # MACD
        macd, macd_signal, macd_histogram = indicator.calculate_macd(close)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = indicator.calculate_bollinger_bands(close)
        
        # Volume anomaly
        volume_data = indicator.calculate_volume_anomaly(volume, close)
        
        # FTMaverick
        ftm_data = indicator.calculate_trend_strength_maverick(close)
        
        # Soporte/Resistencia
        support_levels, resistance_levels = indicator.calculate_dynamic_support_resistance(high, low, close)
        
        # Determinar señal basada en múltiples condiciones
        current_price = close[-1]
        signal_type = 'NEUTRAL'
        signal_score = 0
        
        # Calcular score simple basado en indicadores
        score_components = []
        
        # Componente 1: FTMaverick
        if not ftm_data['no_trade_zones'][-1]:
            score_components.append(20)
        
        # Componente 2: RSI Estocástico en zona favorable
        stoch_rsi = stoch_rsi_data['stoch_rsi'][-1]
        if stoch_rsi < 30 or stoch_rsi > 70:
            score_components.append(15)
        
        # Componente 3: Volumen anómalo
        if volume_data['volume_clusters'][-1]:
            score_components.append(15)
        
        # Componente 4: Posición respecto a MA200
        if current_price > ma_200[-1]:
            score_components.append(10)
        
        # Calcular score total
        if score_components:
            signal_score = min(100, sum(score_components))
        
        # Asignar señal si score es suficientemente alto
        if signal_score >= 60:
            # Determinar dirección basada en múltiples factores
            bullish_factors = 0
            bearish_factors = 0
            
            if stoch_rsi < 30:
                bullish_factors += 1
            if stoch_rsi > 70:
                bearish_factors += 1
            if plus_di[-1] > minus_di[-1]:
                bullish_factors += 1
            if minus_di[-1] > plus_di[-1]:
                bearish_factors += 1
            if rsi_maverick[-1] < 0.3:
                bullish_factors += 1
            if rsi_maverick[-1] > 0.7:
                bearish_factors += 1
            
            if bullish_factors > bearish_factors:
                signal_type = 'LONG'
            elif bearish_factors > bullish_factors:
                signal_type = 'SHORT'
        
        # Calcular niveles de trading
        levels_data = indicator.calculate_optimal_entry_exit(
            df, signal_type, leverage, support_levels, resistance_levels
        )
        
        return jsonify({
            'symbol': symbol,
            'current_price': float(current_price),
            'signal': signal_type,
            'signal_score': float(signal_score),
            'entry': levels_data['entry'],
            'stop_loss': levels_data['stop_loss'],
            'take_profit': levels_data['take_profit'],
            'support_levels': levels_data['support_levels'][:3],
            'resistance_levels': levels_data['resistance_levels'][:3],
            'atr': levels_data['atr'],
            'atr_percentage': levels_data['atr_percentage'],
            'volume': float(volume[-1]),
            'volume_ma': float(np.mean(volume[-20:])),
            'adx': float(adx[-1]),
            'plus_di': float(plus_di[-1]),
            'minus_di': float(minus_di[-1]),
            'whale_pump': float(whale_data['whale_pump'][-1]),
            'whale_dump': float(whale_data['whale_dump'][-1]),
            'rsi_maverick': float(rsi_maverick[-1]),
            'rsi_traditional': float(rsi_traditional[-1]),
            'stoch_rsi': float(stoch_rsi),
            'stoch_k': float(stoch_rsi_data['k_line'][-1]),
            'stoch_d': float(stoch_rsi_data['d_line'][-1]),
            'ma200_condition': 'above' if current_price > ma_200[-1] else 'below',
            'data': df.tail(50).to_dict('records'),
            'indicators': {
                'whale_pump': whale_data['whale_pump'][-50:],
                'whale_dump': whale_data['whale_dump'][-50:],
                'adx': adx[-50:].tolist(),
                'plus_di': plus_di[-50:].tolist(),
                'minus_di': minus_di[-50:].tolist(),
                'rsi_traditional': rsi_traditional[-50:],
                'rsi_maverick': rsi_maverick[-50:],
                'stoch_rsi': stoch_rsi_data['stoch_rsi'][-50:],
                'stoch_k': stoch_rsi_data['k_line'][-50:],
                'stoch_d': stoch_rsi_data['d_line'][-50:],
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
                'volume_anomaly': volume_data['volume_anomaly'][-50:],
                'volume_clusters': volume_data['volume_clusters'][-50:],
                'volume_ratio': volume_data['volume_ratio'][-50:],
                'volume_ma': volume_data['volume_ma'][-50:],
                'volume_signal': volume_data['volume_signal'][-50:],
                'trend_strength': ftm_data['trend_strength'][-50:],
                'bb_width': ftm_data['bb_width'][-50:],
                'no_trade_zones': ftm_data['no_trade_zones'][-50:],
                'strength_signals': ftm_data['strength_signals'][-50:],
                'high_zone_threshold': ftm_data['high_zone_threshold'],
                'colors': ftm_data['colors'][-50:]
            }
        })
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/strategy_signals')
def get_strategy_signals():
    """Endpoint para obtener señales de las estrategias"""
    try:
        signals = indicator.generate_strategy_signals()
        return jsonify({'signals': signals})
        
    except Exception as e:
        print(f"Error en /api/strategy_signals: {e}")
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
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        # Obtener datos de la señal
        signal_data_response = get_signals()
        signal_data = signal_data_response.get_json()
        
        if 'error' in signal_data:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Generar gráfico del reporte
        fig = plt.figure(figsize=(14, 18))
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(10, 1, 1)
        if 'data' in signal_data and signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            opens = [d['open'] for d in signal_data['data']]
            highs = [d['high'] for d in signal_data['data']]
            lows = [d['low'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit'][:3]):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
        
        # Gráfico 2: RSI Estocástico
        ax2 = plt.subplot(10, 1, 2, sharex=ax1)
        if 'indicators' in signal_data and 'stoch_rsi' in signal_data['indicators']:
            stoch_dates = dates_matplotlib[-len(signal_data['indicators']['stoch_rsi']):]
            ax2.plot(stoch_dates, signal_data['indicators']['stoch_rsi'], 
                    'blue', linewidth=1, label='RSI Estocástico')
            ax2.plot(stoch_dates, signal_data['indicators']['stoch_k'], 
                    'green', linewidth=1, label='%K')
            ax2.plot(stoch_dates, signal_data['indicators']['stoch_d'], 
                    'red', linewidth=1, label='%D')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.3)
        ax2.set_ylabel('RSI Estocástico')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: ADX/DMI
        ax3 = plt.subplot(10, 1, 3, sharex=ax1)
        if 'indicators' in signal_data and 'adx' in signal_data['indicators']:
            adx_dates = dates_matplotlib[-len(signal_data['indicators']['adx']):]
            ax3.plot(adx_dates, signal_data['indicators']['adx'], 
                    'black', linewidth=2, label='ADX')
            ax3.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax3.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax3.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        ax3.set_ylabel('ADX/DMI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: RSI Tradicional y Maverick
        ax4 = plt.subplot(10, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_traditional' in signal_data['indicators']:
            rsi_dates = dates_matplotlib[-len(signal_data['indicators']['rsi_traditional']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi_traditional'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax4.plot(rsi_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax4.axhline(y=80, color='red', linestyle='--', alpha=0.3)
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.3)
            ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.2)
        ax4.set_ylabel('RSI Trad/Maverick')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: MACD
        ax5 = plt.subplot(10, 1, 5, sharex=ax1)
        if 'indicators' in signal_data and 'macd' in signal_data['indicators']:
            macd_dates = dates_matplotlib[-len(signal_data['indicators']['macd']):]
            ax5.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax5.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax5.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6, label='Histograma')
            
            ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax5.set_ylabel('MACD')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Ballenas
        ax6 = plt.subplot(10, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'whale_pump' in signal_data['indicators']:
            whale_dates = dates_matplotlib[-len(signal_data['indicators']['whale_pump']):]
            ax6.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax6.bar(whale_dates, [-x for x in signal_data['indicators']['whale_dump']], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
        ax6.set_ylabel('Fuerza Ballenas')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: Volumen y Anomalías
        ax7 = plt.subplot(10, 1, 7, sharex=ax1)
        if 'indicators' in signal_data and 'volume_ratio' in signal_data['indicators']:
            volume_dates = dates_matplotlib[-len(signal_data['indicators']['volume_ratio']):]
            
            # Colores según señal de volumen
            colors = []
            volume_signal = signal_data['indicators'].get('volume_signal', ['NEUTRAL'] * 50)
            for i, signal in enumerate(volume_signal[-50:]):
                if signal == 'COMPRA':
                    colors.append('green')
                elif signal == 'VENTA':
                    colors.append('red')
                else:
                    colors.append('gray')
            
            # Volumen
            volumes = [d['volume'] for d in signal_data['data'][-50:]]
            ax7.bar(volume_dates, volumes, color=colors, alpha=0.6, label='Volumen')
            
            # MA de volumen
            ax7.plot(volume_dates, signal_data['indicators']['volume_ma'][-50:], 
                    'yellow', linewidth=1, label='MA Volumen')
        ax7.set_ylabel('Volumen')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: Fuerza de Tendencia Maverick
        ax8 = plt.subplot(10, 1, 8, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates_matplotlib[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength'][-50:]
            colors = signal_data['indicators']['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax8.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax8.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax8.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        ax8.set_ylabel('Fuerza Tendencia %')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Gráfico 9: Bollinger Bands
        ax9 = plt.subplot(10, 1, 9, sharex=ax1)
        if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
            bb_dates = dates_matplotlib[-len(signal_data['indicators']['bb_upper']):]
            ax9.plot(bb_dates, closes[-50:], 'blue', linewidth=1, label='Precio')
            ax9.plot(bb_dates, signal_data['indicators']['bb_upper'][-50:], 
                    'orange', alpha=0.7, linewidth=1, label='BB Superior')
            ax9.plot(bb_dates, signal_data['indicators']['bb_middle'][-50:], 
                    'orange', alpha=0.5, linewidth=1, label='BB Media')
            ax9.plot(bb_dates, signal_data['indicators']['bb_lower'][-50:], 
                    'orange', alpha=0.7, linewidth=1, label='BB Inferior')
            ax9.fill_between(bb_dates, signal_data['indicators']['bb_lower'][-50:], 
                           signal_data['indicators']['bb_upper'][-50:], 
                           color='orange', alpha=0.1)
        ax9.set_ylabel('Bollinger Bands')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # Información de la señal
        ax10 = plt.subplot(10, 1, 10)
        ax10.axis('off')
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
        APALANCAMIENTO: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        INDICADORES:
        RSI Tradicional: {signal_data['rsi_traditional']:.1f}
        RSI Maverick: {signal_data['rsi_maverick']:.2f}
        RSI Estocástico: {signal_data['stoch_rsi']:.1f}
        ADX: {signal_data['adx']:.1f}
        +DI: {signal_data['plus_di']:.1f}
        -DI: {signal_data['minus_di']:.1f}
        """
        
        ax10.text(0.1, 0.9, signal_info, transform=ax10.transAxes, fontsize=10,
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
