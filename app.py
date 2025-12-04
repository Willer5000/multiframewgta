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
import hashlib

app = Flask(__name__)

# Configuración Telegram
TELEGRAM_BOT_TOKEN = "8007748376:AAHIW8n9b-BtA378g4gF-0-D2mOhn495Q0g"
TELEGRAM_CHAT_ID = "-1003229814161"

# Configuración CoinMarketCap
COINMARKETCAP_API_KEY = "d22df0c59e5e47e0980b89f6eb32ea1b"
CMC_CRYPTO_LIST = ["BTC", "ETH", "SOL", "XRP", "ADA"]

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
        self.volume_spike_cache = {}
        self.active_operations = {}
        self.winrate_data = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.cmc_cache = {'data': None, 'timestamp': None}
        self.cmc_volume_history = {symbol: deque(maxlen=7) for symbol in CMC_CRYPTO_LIST}
        
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def get_cmc_volume_data(self):
        """Obtener datos de volumen de CoinMarketCap"""
        try:
            # Verificar cache (60 segundos)
            if (self.cmc_cache['data'] and self.cmc_cache['timestamp'] and 
                (datetime.now() - self.cmc_cache['timestamp']).seconds < 60):
                return self.cmc_cache['data']
            
            headers = {
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY,
            }
            
            url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
            parameters = {
                'start': '1',
                'limit': '100',
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=parameters, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                crypto_data = {}
                
                for crypto in data['data']:
                    symbol = crypto['symbol']
                    if symbol in CMC_CRYPTO_LIST:
                        crypto_data[symbol] = {
                            'volume_24h': crypto['quote']['USD']['volume_24h'],
                            'percent_change_24h': crypto['quote']['USD']['percent_change_24h'],
                            'price': crypto['quote']['USD']['price'],
                            'market_cap': crypto['quote']['USD']['market_cap']
                        }
                        
                        # Actualizar historial de volumen
                        if symbol in self.cmc_volume_history:
                            self.cmc_volume_history[symbol].append(crypto['quote']['USD']['volume_24h'])
                
                self.cmc_cache = {'data': crypto_data, 'timestamp': datetime.now()}
                print(f"Datos CMC obtenidos: {len(crypto_data)} cryptos")
                return crypto_data
            else:
                print(f"Error CMC API: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error obteniendo datos CMC: {e}")
            return {}
    
    def detect_cmc_volume_spike(self):
        """Detectar spikes de volumen en CoinMarketCap"""
        alerts = []
        current_time = datetime.now()
        
        cmc_data = self.get_cmc_volume_data()
        if not cmc_data:
            return alerts
        
        for symbol, data in cmc_data.items():
            try:
                volume_24h = data['volume_24h']
                percent_change = data['percent_change_24h']
                
                # Calcular promedio histórico (últimos 7 días)
                hist_data = list(self.cmc_volume_history.get(symbol, deque()))
                if len(hist_data) >= 3:
                    hist_avg = np.mean(hist_data)
                else:
                    hist_avg = volume_24h * 0.3  # Estimación inicial
                
                # Detectar spike (>300% del promedio histórico)
                if hist_avg > 0 and volume_24h > hist_avg * 3.0:
                    
                    # Determinar dirección
                    if percent_change > 0:
                        alert_type = "COMPRA"
                        action = "revisar LONG"
                    else:
                        alert_type = "VENTA"
                        action = "revisar SHORT"
                    
                    # Formatear monto en millones
                    amount_millions = volume_24h / 1_000_000
                    
                    # Crear alerta única
                    alert_id = f"cmc_volume_{symbol}_{alert_type}_{current_time.strftime('%Y%m%d%H')}"
                    
                    if alert_id not in self.volume_spike_cache:
                        alert = {
                            'symbol': symbol,
                            'alert_type': alert_type,
                            'amount_millions': amount_millions,
                            'percent_change': percent_change,
                            'volume_24h': volume_24h,
                            'hist_avg': hist_avg,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'action': action,
                            'alert_id': alert_id
                        }
                        
                        alerts.append(alert)
                        self.volume_spike_cache[alert_id] = current_time
                        
                        print(f"Spike detectado: {symbol} {alert_type} {amount_millions:.2f}M")
                    
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")
                continue
        
        return alerts
    
    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela"""
        if interval == '15m':
            minutes = current_time.minute
            remaining = 15 - (minutes % 15)
            return remaining <= 3  # Últimos 3 minutos
        elif interval == '30m':
            minutes = current_time.minute
            remaining = 30 - (minutes % 30)
            return remaining <= 6  # Últimos 6 minutos
        elif interval == '1h':
            return current_time.minute >= 55  # Últimos 5 minutos
        elif interval == '2h':
            current_hour = current_time.hour
            remaining = 2 - (current_hour % 2)
            return remaining == 0 and current_time.minute >= 55
        elif interval == '4h':
            current_hour = current_time.hour
            remaining = 4 - (current_hour % 4)
            return remaining == 0 and current_time.minute >= 55
        elif interval == '8h':
            current_hour = current_time.hour
            remaining = 8 - (current_hour % 8)
            return remaining == 0 and current_time.minute >= 55
        elif interval == '12h':
            current_hour = current_time.hour
            if current_hour < 8:
                return current_hour == 7 and current_time.minute >= 55
            else:
                return current_hour == 19 and current_time.minute >= 55
        elif interval == '1D':
            return current_time.hour == 23 and current_time.minute >= 55
        elif interval == '1W':
            return current_time.weekday() == 6 and current_time.hour == 23 and current_time.minute >= 55
        
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
                '1D': '1day', '1W': '1week'
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
        """Generar datos de ejemplo"""
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
        """Calcular 4 niveles de soporte y resistencia"""
        n = len(close)
        supports = []
        resistances = []
        
        # Calcular pivots cada 'period' velas
        for i in range(period, n, period):
            window_high = high[i-period:i]
            window_low = low[i-period:i]
            
            # Encontrar máximos y mínimos locales
            local_max = np.max(window_high)
            local_min = np.min(window_low)
            
            resistances.append(float(local_max))
            supports.append(float(local_min))
        
        # Asegurar al menos 4 niveles
        while len(supports) < 4:
            supports.append(float(np.min(low[-period:])))
        
        while len(resistances) < 4:
            resistances.append(float(np.max(high[-period:])))
        
        # Ordenar y tomar los 4 más cercanos
        current_price = close[-1]
        
        supports.sort(reverse=True)  # De mayor a menor
        resistances.sort()  # De menor a mayor
        
        # Filtrar niveles relevantes
        relevant_supports = [s for s in supports if s < current_price]
        relevant_resistances = [r for r in resistances if r > current_price]
        
        # Tomar hasta 4 niveles
        final_supports = relevant_supports[:4]
        final_resistances = relevant_resistances[:4]
        
        # Si no hay suficientes, usar los más cercanos
        while len(final_supports) < 4:
            if supports:
                final_supports.append(supports[min(len(final_supports), len(supports)-1)])
            else:
                final_supports.append(current_price * 0.95)
        
        while len(final_resistances) < 4:
            if resistances:
                final_resistances.append(resistances[min(len(final_resistances), len(resistances)-1)])
            else:
                final_resistances.append(current_price * 1.05)
        
        return final_supports, final_resistances

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con 4 S/R"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular 4 soportes y resistencias
            supports, resistances = self.calculate_support_resistance(high, low, close)
            
            # Para LONG: entrada en soporte más cercano
            # Para SHORT: entrada en resistencia más cercana
            if signal_type == 'LONG':
                # Buscar soporte más cercano por debajo del precio actual
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)  # Soporte más alto por debajo
                else:
                    entry = min(supports)  # Soporte más bajo disponible
                
                stop_loss = entry - (current_atr * 1.5)
                
                # Take profits en resistencias
                take_profits = []
                for resistance in resistances[:3]:  # Primeras 3 resistencias
                    if resistance > entry:
                        take_profits.append(resistance)
                
                if not take_profits:
                    take_profits = [entry + (2 * (entry - stop_loss))]
                
            else:  # SHORT
                # Buscar resistencia más cercana por encima del precio actual
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)  # Resistencia más baja por encima
                else:
                    entry = max(resistances)  # Resistencia más alta disponible
                
                stop_loss = entry + (current_atr * 1.5)
                
                # Take profits en soportes
                take_profits = []
                for support in supports[:3]:  # Primeros 3 soportes
                    if support < entry:
                        take_profits.append(support)
                
                if not take_profits:
                    take_profits = [entry - (2 * (stop_loss - entry))]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits[:3]],  # Máximo 3 TPs
                'supports': supports,
                'resistances': resistances,
                'atr': float(current_atr),
                'atr_percentage': float(current_atr / current_price)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.05],
                'supports': [current_price * 0.95, current_price * 0.90, current_price * 0.85, current_price * 0.80],
                'resistances': [current_price * 1.05, current_price * 1.10, current_price * 1.15, current_price * 1.20],
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    def calculate_ema(self, prices, period):
        """Calcular EMA"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[period-1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def calculate_sma(self, prices, period):
        """Calcular SMA"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        sma = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
        
        return sma

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger"""
        if len(prices) < period:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        sma = self.calculate_sma(prices, period)
        std = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional"""
        if len(prices) < period + 1:
            return np.zeros_like(prices)
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100 - 100 / (1 + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs = up / down if down != 0 else 0
            rsi[i] = 100 - 100 / (1 + rs)
        
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcular MACD"""
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
            
            # Calcular umbral alto
            if n >= 50:
                historical_bb_width = bb_width[max(0, n-100):n]
                high_zone_threshold = np.percentile(historical_bb_width, 70)
            else:
                high_zone_threshold = np.percentile(bb_width[bb_width > 0], 70) if np.any(bb_width > 0) else 5
            
            # Detectar zonas no operar
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
            
            return {
                'bb_width': bb_width.tolist(),
                'trend_strength': trend_strength.tolist(),
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
                'high_zone_threshold': 5.0,
                'no_trade_zones': [False] * n,
                'strength_signals': ['NEUTRAL'] * n,
                'colors': ['gray'] * n
            }

    def check_multi_timeframe_obligatory(self, symbol, interval, signal_type):
        """Verificar condiciones multi-timeframe obligatorias - CORREGIDO"""
        try:
            # Para temporalidades 12h, 1D, 1W no es obligatorio Multi-Timeframe
            if interval in ['12h', '1D', '1W']:
                return True
                
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False
            
            # TF Mayor: Alcista o Neutral para LONG, Bajista o Neutral para SHORT
            mayor_df = self.get_kucoin_data(symbol, hierarchy.get('mayor', '4h'), 50)
            if mayor_df is not None and len(mayor_df) > 20:
                mayor_trend = self.calculate_trend_strength_maverick(mayor_df['close'].values)
                mayor_signal = mayor_trend['strength_signals'][-1]
                
                if signal_type == 'LONG':
                    mayor_ok = mayor_signal in ['STRONG_UP', 'WEAK_UP', 'NEUTRAL']
                else:
                    mayor_ok = mayor_signal in ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL']
            else:
                mayor_ok = True
            
            # TF Medio: Debe estar en dirección de la señal
            media_df = self.get_kucoin_data(symbol, hierarchy.get('media', '2h'), 30)
            if media_df is not None and len(media_df) > 15:
                media_trend = self.calculate_trend_strength_maverick(media_df['close'].values)
                media_signal = media_trend['strength_signals'][-1]
                
                if signal_type == 'LONG':
                    media_ok = media_signal in ['STRONG_UP', 'WEAK_UP']
                else:
                    media_ok = media_signal in ['STRONG_DOWN', 'WEAK_DOWN']
            else:
                media_ok = False
            
            # TF Menor: Dirección de señal y fuera de zona no operar
            menor_df = self.get_kucoin_data(symbol, hierarchy.get('menor', '30m'), 30)
            if menor_df is not None and len(menor_df) > 10:
                menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                menor_signal = menor_trend['strength_signals'][-1]
                menor_no_trade = not menor_trend['no_trade_zones'][-1]
                
                if signal_type == 'LONG':
                    menor_ok = menor_signal in ['STRONG_UP', 'WEAK_UP'] and menor_no_trade
                else:
                    menor_ok = menor_signal in ['STRONG_DOWN', 'WEAK_DOWN'] and menor_no_trade
            else:
                menor_ok = True
                menor_no_trade = True
            
            return mayor_ok and media_ok and menor_ok and menor_no_trade
            
        except Exception as e:
            print(f"Error verificando multi-timeframe: {e}")
            return False

    def calculate_whale_signals_improved(self, df, sensitivity=1.7, min_volume_multiplier=1.5, 
                                       support_resistance_lookback=50, signal_threshold=25):
        """Indicador de ballenas mejorado"""
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
        """RSI Modificado Maverick"""
        try:
            n = len(close)
            
            basis = self.calculate_sma(close, length)
            dev = np.zeros(n)
            
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                dev[i] = np.std(window) if len(window) > 1 else 0
            
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
        """Detectar divergencias con periodo de confirmación"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            price_window = price[i-lookback:i+1]
            indicator_window = indicator[i-lookback:i+1]
            
            # Divergencia alcista: precio hace mínimo más bajo, indicador hace mínimo más alto
            if (price[i] < np.min(price_window[:-1]) and 
                indicator[i] > np.min(indicator_window[:-1])):
                bullish_div[i] = True
            
            # Divergencia bajista: precio hace máximo más alto, indicador hace máximo más bajo
            if (price[i] > np.max(price_window[:-1]) and 
                indicator[i] < np.max(indicator_window[:-1])):
                bearish_div[i] = True
        
        # Extender señal por confirmation_period velas
        for i in range(n):
            if bullish_div[i]:
                for j in range(1, min(confirmation_period+1, n-i)):
                    if i+j < n:
                        bullish_div[i+j] = True
            
            if bearish_div[i]:
                for j in range(1, min(confirmation_period+1, n-i)):
                    if i+j < n:
                        bearish_div[i+j] = True
        
        return bullish_div.tolist(), bearish_div.tolist()

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

    def detect_chart_patterns(self, high, low, close, confirmation_period=7):
        """Detectar patrones de chartismo con confirmación"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(20, n-7):
            window_high = high[i-20:i+1]
            window_low = low[i-20:i+1]
            window_close = close[i-20:i+1]
            
            # Hombro Cabeza Hombro
            max_idx = np.argmax(window_high)
            if (max_idx > 5 and max_idx < len(window_high)-5 and
                window_high[max_idx-3] < window_high[max_idx] and
                window_high[max_idx+3] < window_high[max_idx]):
                patterns['head_shoulders'][i] = True
            
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
        
        # Extender patrones por confirmation_period
        for pattern in patterns:
            for i in range(n):
                if patterns[pattern][i]:
                    for j in range(1, min(confirmation_period+1, n-i)):
                        if i+j < n:
                            patterns[pattern][i+j] = True
        
        return patterns

    def check_breakout(self, high, low, close, support, resistance):
        """Detectar rupturas con 1 vela de confirmación"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Ruptura alcista: cierre por encima de resistencia
            if close[i] > resistance[i] and high[i] > high[i-1]:
                breakout_up[i] = True
                # Extender 1 vela más
                if i+1 < n:
                    breakout_up[i+1] = True
            
            # Ruptura bajista: cierre por debajo de soporte
            if close[i] < support[i] and low[i] < low[i-1]:
                breakout_down[i] = True
                # Extender 1 vela más
                if i+1 < n:
                    breakout_down[i+1] = True
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover(self, plus_di, minus_di):
        """Detectar cruces de DMI con 1 vela de confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                di_cross_bullish[i] = True
                # Extender 1 vela más
                if i+1 < n:
                    di_cross_bullish[i+1] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
                # Extender 1 vela más
                if i+1 < n:
                    di_cross_bearish[i+1] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

    def check_ma_crossover(self, ma_fast, ma_slow):
        """Detectar cruce de medias con 1 vela de confirmación"""
        n = len(ma_fast)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MA rápida cruza por encima de MA lenta
            if (ma_fast[i] > ma_slow[i] and 
                ma_fast[i-1] <= ma_slow[i-1]):
                ma_cross_bullish[i] = True
                # Extender 1 vela más
                if i+1 < n:
                    ma_cross_bullish[i+1] = True
            
            # Cruce bajista: MA rápida cruza por debajo de MA lenta
            if (ma_fast[i] < ma_slow[i] and 
                ma_fast[i-1] >= ma_slow[i-1]):
                ma_cross_bearish[i] = True
                # Extender 1 vela más
                if i+1 < n:
                    ma_cross_bearish[i+1] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()

    def check_macd_crossover(self, macd, macd_signal):
        """Detectar cruce MACD con 1 vela de confirmación"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MACD cruza por encima de señal
            if (macd[i] > macd_signal[i] and 
                macd[i-1] <= macd_signal[i-1]):
                macd_cross_bullish[i] = True
                # Extender 1 vela más
                if i+1 < n:
                    macd_cross_bullish[i+1] = True
            
            # Cruce bajista: MACD cruza por debajo de señal
            if (macd[i] < macd_signal[i] and 
                macd[i-1] >= macd_signal[i-1]):
                macd_cross_bearish[i] = True
                # Extender 1 vela más
                if i+1 < n:
                    macd_cross_bearish[i+1] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()

    def calculate_volume_anomaly_improved(self, volume, close, period=20, std_multiplier=2):
        """Calcular anomalías de volumen mejorado"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_direction = np.zeros(n)  # 1=compra, -1=venta, 0=neutral
            
            for i in range(period, n):
                # Media móvil de volumen
                ma_volume = np.mean(volume[i-period:i])
                
                # Desviación estándar
                std_volume = np.std(volume[i-period:i]) if period > 1 else 0
                
                # Anomalía: volumen > MA + 2σ
                if volume[i] > ma_volume + (std_multiplier * std_volume):
                    volume_anomaly[i] = True
                    
                    # Determinar dirección basada en precio
                    if i > 0:
                        price_change = (close[i] - close[i-1]) / close[i-1] * 100
                        if price_change > 0:
                            volume_direction[i] = 1  # Compra
                        else:
                            volume_direction[i] = -1  # Venta
            
            # Detectar clusters (múltiples anomalías cercanas)
            for i in range(period, n-5):
                window_anomalies = volume_anomaly[i:i+5]
                if np.sum(window_anomalies) >= 3:
                    volume_clusters[i:i+5] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_direction': volume_direction.tolist(),
                'volume_ma': [np.mean(volume[max(0, i-period):i+1]) for i in range(n)]
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly_improved: {e}")
            n = len(volume)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_direction': [0] * n,
                'volume_ma': [0] * n
            }

    def evaluate_signal_conditions_corrected(self, data, current_idx, interval):
        """Evaluar condiciones de señal con nuevos pesos"""
        # Definir pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            weights = {
                'long': {
                    'multi_timeframe': 30,  # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'ma_cross': 10,
                    'di_cross': 10,
                    'adx_slope': 5,
                    'bollinger': 8,
                    'macd_cross': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_div': 8,
                    'rsi_trad_div': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'multi_timeframe': 30,
                    'trend_strength': 25,
                    'ma_cross': 10,
                    'di_cross': 10,
                    'adx_slope': 5,
                    'bollinger': 8,
                    'macd_cross': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_div': 8,
                    'rsi_trad_div': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                }
            }
        elif interval in ['12h', '1D']:
            weights = {
                'long': {
                    'whale_signal': 30,    # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'ma_cross': 10,
                    'di_cross': 10,
                    'adx_slope': 5,
                    'bollinger': 8,
                    'macd_cross': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_div': 8,
                    'rsi_trad_div': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'whale_signal': 30,
                    'trend_strength': 25,
                    'ma_cross': 10,
                    'di_cross': 10,
                    'adx_slope': 5,
                    'bollinger': 8,
                    'macd_cross': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_div': 8,
                    'rsi_trad_div': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                }
            }
        else:  # 1W
            weights = {
                'long': {
                    'trend_strength': 55,   # Obligatorio
                    'ma_cross': 10,
                    'di_cross': 10,
                    'adx_slope': 5,
                    'bollinger': 8,
                    'macd_cross': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_div': 8,
                    'rsi_trad_div': 5,
                    'chart_pattern': 5,
                    'breakout': 5
                },
                'short': {
                    'trend_strength': 55,
                    'ma_cross': 10,
                    'di_cross': 10,
                    'adx_slope': 5,
                    'bollinger': 8,
                    'macd_cross': 10,
                    'volume_anomaly': 7,
                    'rsi_maverick_div': 8,
                    'rsi_trad_div': 5,
                    'chart_pattern': 5,
                    'breakout': 5
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
        
        # Condiciones LONG
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['long']['multi_timeframe']['value'] = data.get('multi_timeframe_long', False)
        elif interval in ['12h', '1D']:
            conditions['long']['whale_signal']['value'] = (
                data['whale_pump'][current_idx] > 20 and
                data['confirmed_buy'][current_idx]
            )
        
        # Fuerza de tendencia (obligatorio para todos)
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        # Condiciones complementarias
        conditions['long']['ma_cross']['value'] = data.get('ma_cross_bullish', [False])[current_idx]
        conditions['long']['di_cross']['value'] = data.get('di_cross_bullish', [False])[current_idx]
        conditions['long']['adx_slope']['value'] = (
            data['adx'][current_idx] > 25 and
            current_idx > 0 and
            data['adx'][current_idx] > data['adx'][current_idx-1]
        )
        conditions['long']['bollinger']['value'] = data.get('bollinger_touch_lower', [False])[current_idx]
        conditions['long']['macd_cross']['value'] = data.get('macd_cross_bullish', [False])[current_idx]
        conditions['long']['volume_anomaly']['value'] = (
            data['volume_anomaly'][current_idx] and
            data['volume_direction'][current_idx] == 1
        )
        conditions['long']['rsi_maverick_div']['value'] = data.get('rsi_maverick_bullish_divergence', [False])[current_idx]
        conditions['long']['rsi_trad_div']['value'] = data.get('rsi_bullish_divergence', [False])[current_idx]
        conditions['long']['chart_pattern']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_flag'][current_idx]
        )
        conditions['long']['breakout']['value'] = data.get('breakout_up', [False])[current_idx]
        
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
        
        conditions['short']['ma_cross']['value'] = data.get('ma_cross_bearish', [False])[current_idx]
        conditions['short']['di_cross']['value'] = data.get('di_cross_bearish', [False])[current_idx]
        conditions['short']['adx_slope']['value'] = (
            data['adx'][current_idx] > 25 and
            current_idx > 0 and
            data['adx'][current_idx] > data['adx'][current_idx-1]
        )
        conditions['short']['bollinger']['value'] = data.get('bollinger_touch_upper', [False])[current_idx]
        conditions['short']['macd_cross']['value'] = data.get('macd_cross_bearish', [False])[current_idx]
        conditions['short']['volume_anomaly']['value'] = (
            data['volume_anomaly'][current_idx] and
            data['volume_direction'][current_idx] == -1
        )
        conditions['short']['rsi_maverick_div']['value'] = data.get('rsi_maverick_bearish_divergence', [False])[current_idx]
        conditions['short']['rsi_trad_div']['value'] = data.get('rsi_bearish_divergence', [False])[current_idx]
        conditions['short']['chart_pattern']['value'] = (
            data['chart_patterns']['head_shoulders'][current_idx] or
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['bearish_flag'][current_idx]
        )
        conditions['short']['breakout']['value'] = data.get('breakout_down', [False])[current_idx]
        
        return conditions

    def get_condition_description(self, condition_key):
        """Obtener descripción de condición"""
        descriptions = {
            'multi_timeframe': 'Multi-Timeframe',
            'trend_strength': 'Fuerza Tendencia Maverick',
            'whale_signal': 'Señal Ballenas',
            'ma_cross': 'Cruce Medias 9/21',
            'di_cross': 'Cruce DMI',
            'adx_slope': 'ADX Pendiente Positiva',
            'bollinger': 'Bandas Bollinger',
            'macd_cross': 'Cruce MACD',
            'volume_anomaly': 'Volumen Anómalo',
            'rsi_maverick_div': 'Divergencia RSI Maverick',
            'rsi_trad_div': 'Divergencia RSI Tradicional',
            'chart_pattern': 'Patrón Chartista',
            'breakout': 'Ruptura S/R'
        }
        return descriptions.get(condition_key, condition_key)

    def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal"""
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        signal_conditions = conditions.get(signal_type, {})
        
        # Verificar condiciones obligatorias
        obligatory_conditions = []
        for key, condition in signal_conditions.items():
            if condition['weight'] >= 25:  # Condiciones obligatorias
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
        
        # Score mínimo ajustado según MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0
        
        return min(final_score, 100), fulfilled_conditions

    def check_bollinger_conditions(self, close, bb_upper, bb_lower, bb_middle):
        """Verificar condiciones de Bollinger Bands"""
        n = len(close)
        touch_upper = np.zeros(n, dtype=bool)
        touch_lower = np.zeros(n, dtype=bool)
        
        for i in range(n):
            # Toque banda superior (para SHORT)
            if close[i] >= bb_upper[i] * 0.99:
                touch_upper[i] = True
            
            # Toque banda inferior (para LONG)
            if close[i] <= bb_lower[i] * 1.01:
                touch_lower[i] = True
        
        return touch_upper.tolist(), touch_lower.tolist()

    def generate_signals_improved(self, symbol, interval, di_period=14, adx_threshold=25, 
                                sr_period=50, rsi_length=14, bb_multiplier=2.0, volume_filter='Todos', leverage=15):
        """Generación de señales mejorada"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular todos los indicadores
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Indicadores principales
            whale_data = self.calculate_whale_signals_improved(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            breakout_up, breakout_down = self.check_breakout(high, low, close, 
                                                           whale_data['support'], whale_data['resistance'])
            
            chart_patterns = self.detect_chart_patterns(high, low, close)
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover(ma_9, ma_21)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_bullish, macd_cross_bearish = self.check_macd_crossover(macd, macd_signal)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            bollinger_touch_upper, bollinger_touch_lower = self.check_bollinger_conditions(close, bb_upper, bb_lower, bb_middle)
            
            # Volumen anómalo
            volume_anomaly_data = self.calculate_volume_anomaly_improved(volume, close)
            
            current_idx = -1
            
            # Verificar condiciones multi-timeframe
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
                'bollinger_touch_upper': bollinger_touch_upper,
                'bollinger_touch_lower': bollinger_touch_lower,
                'volume_anomaly': volume_anomaly_data['volume_anomaly'],
                'volume_clusters': volume_anomaly_data['volume_clusters'],
                'volume_direction': volume_anomaly_data['volume_direction'],
                'volume_ma': volume_anomaly_data['volume_ma'],
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short
            }
            
            conditions = self.evaluate_signal_conditions_corrected(analysis_data, current_idx, interval)
            
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
            
            # Calcular niveles de trading
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, leverage)
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
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
                    'bollinger_touch_upper': bollinger_touch_upper[-50:],
                    'bollinger_touch_lower': bollinger_touch_lower[-50:],
                    'volume_anomaly': volume_anomaly_data['volume_anomaly'][-50:],
                    'volume_clusters': volume_anomaly_data['volume_clusters'][-50:],
                    'volume_direction': volume_anomaly_data['volume_direction'][-50:],
                    'volume_ma': volume_anomaly_data['volume_ma'][-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
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
        """Crear señal vacía"""
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

    def generate_scalping_alerts(self):
        """Generar alertas de trading"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
            if interval in ['15m', '30m'] and not self.is_scalping_time():
                continue
                
            should_send_alert = self.calculate_remaining_time(interval, current_time)
            
            if not should_send_alert:
                continue
                
            for symbol in CRYPTO_SYMBOLS[:12]:
                try:
                    signal_data = self.generate_signals_improved(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        alert = {
                            'symbol': symbol,
                            'interval': interval,
                            'signal': signal_data['signal'],
                            'score': signal_data['signal_score'],
                            'entry': signal_data['entry'],
                            'current_price': signal_data['current_price'],
                            'supports': signal_data['supports'],
                            'resistances': signal_data['resistances'],
                            'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                            'ma200_condition': signal_data.get('ma200_condition', 'below'),
                            'multi_timeframe_ok': signal_data.get('multi_timeframe_ok', False),
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
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

    def is_scalping_time(self):
        """Verificar si es horario de scalping"""
        now = self.get_bolivia_time()
        if now.weekday() >= 5:
            return False
        return 4 <= now.hour < 16

# Instancia global del indicador
indicator = TradingIndicator()

def generate_telegram_image_strategy1(signal_data):
    """Generar imagen para Telegram (Estrategia 1)"""
    try:
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('white')
        
        # 1. Gráfico de velas
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
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=0.5)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=2)
            
            # Bandas de Bollinger (transparentes)
            if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
                bb_dates = dates[-len(signal_data['indicators']['bb_upper']):]
                ax1.plot(bb_dates, signal_data['indicators']['bb_upper'], 
                        color='orange', alpha=0.3, linewidth=1)
                ax1.plot(bb_dates, signal_data['indicators']['bb_middle'], 
                        color='orange', alpha=0.3, linewidth=1)
                ax1.plot(bb_dates, signal_data['indicators']['bb_lower'], 
                        color='orange', alpha=0.3, linewidth=1)
            
            # Medias móviles
            if 'indicators' in signal_data:
                ma_dates = dates[-len(signal_data['indicators']['ma_9']):]
                ax1.plot(ma_dates, signal_data['indicators']['ma_9'], 
                        color='blue', alpha=0.7, linewidth=1, label='MA9')
                ax1.plot(ma_dates, signal_data['indicators']['ma_21'], 
                        color='red', alpha=0.7, linewidth=1, label='MA21')
                ax1.plot(ma_dates, signal_data['indicators']['ma_50'], 
                        color='purple', alpha=0.7, linewidth=1, label='MA50')
        
        ax1.set_title(f"{signal_data['symbol']} - Velas Japonesas", fontsize=10, fontweight='bold')
        ax1.set_ylabel('Precio')
        ax1.legend(fontsize=6)
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(8, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax2.plot(adx_dates, signal_data['indicators']['adx'], 
                    'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)
        
        # 3. Volumen con anomalías (columnas)
        ax3 = plt.subplot(8, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_anomaly']):]
            
            # Colores según dirección
            colors = []
            for i in range(len(volume_dates)):
                if signal_data['indicators']['volume_direction'][i] == 1:
                    colors.append('green')
                elif signal_data['indicators']['volume_direction'][i] == -1:
                    colors.append('red')
                else:
                    colors.append('gray')
            
            ax3.bar(volume_dates, signal_data['indicators']['volume_ma'], 
                   color=colors, alpha=0.6, width=0.8)
        
        ax3.set_ylabel('Volumen')
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick (columnas)
        ax4 = plt.subplot(8, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_values = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            ax4.bar(trend_dates, trend_values, color=colors, alpha=0.7, width=0.8)
        
        ax4.set_ylabel('Fuerza Tendencia')
        ax4.grid(True, alpha=0.3)
        
        # 5. Indicador Ballenas (solo para 12h y 1D)
        ax5 = plt.subplot(8, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            
            ax5.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.6, width=0.8, label='Compra')
            ax5.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.6, width=0.8, label='Venta')
        
        ax5.set_ylabel('Ballenas')
        ax5.legend(fontsize=6)
        ax5.grid(True, alpha=0.3)
        
        # 6. RSI Maverick
        ax6 = plt.subplot(8, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax6.plot(rsi_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=1)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
        
        ax6.set_ylabel('RSI Maverick')
        ax6.grid(True, alpha=0.3)
        
        # 7. RSI Tradicional
        ax7 = plt.subplot(8, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_trad_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax7.plot(rsi_trad_dates, signal_data['indicators']['rsi_traditional'], 
                    'purple', linewidth=1)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.7)
        
        ax7.set_ylabel('RSI Tradicional')
        ax7.grid(True, alpha=0.3)
        
        # 8. MACD (columnas para histograma)
        ax8 = plt.subplot(8, 1, 8, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd_histogram']):]
            hist_colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            
            ax8.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=hist_colors, alpha=0.6, width=0.8)
        
        ax8.set_ylabel('MACD Hist')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen estrategia 1: {e}")
        return None

def generate_telegram_image_strategy2(alert_data):
    """Generar imagen para Telegram (Estrategia 2 - Volumen atípico)"""
    try:
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Obtener datos históricos para la gráfica
        symbol = alert_data['symbol'] + "-USDT"
        df = indicator.get_kucoin_data(symbol, '1h', 50)
        
        if df is not None and len(df) > 20:
            dates = df['timestamp'].tail(30)
            closes = df['close'].tail(30).values
            volumes = df['volume'].tail(30).values
            
            # 1. Gráfico de velas
            ax1 = plt.subplot(3, 1, 1)
            
            # Simular velas simples
            ax1.plot(dates, closes, 'black', linewidth=1)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = indicator.calculate_bollinger_bands(closes)
            ax1.plot(dates, bb_upper, 'orange', alpha=0.3, linewidth=1)
            ax1.plot(dates, bb_middle, 'orange', alpha=0.3, linewidth=1)
            ax1.plot(dates, bb_lower, 'orange', alpha=0.3, linewidth=1)
            
            # Medias móviles
            ma_9 = indicator.calculate_sma(closes, 9)
            ma_21 = indicator.calculate_sma(closes, 21)
            ax1.plot(dates[-len(ma_9):], ma_9, 'blue', alpha=0.7, linewidth=1)
            ax1.plot(dates[-len(ma_21):], ma_21, 'red', alpha=0.7, linewidth=1)
            
            ax1.set_title(f"{alert_data['symbol']} - Volumen Atípico", fontsize=10, fontweight='bold')
            ax1.set_ylabel('Precio')
            ax1.grid(True, alpha=0.3)
            
            # 2. ADX con DMI
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            if len(closes) > 14:
                high = df['high'].tail(30).values
                low = df['low'].tail(30).values
                adx, plus_di, minus_di = indicator.calculate_adx(high, low, closes)
                
                adx_dates = dates[-len(adx):]
                ax2.plot(adx_dates, adx, 'black', linewidth=2, label='ADX')
                ax2.plot(adx_dates, plus_di, 'green', linewidth=1, label='+DI')
                ax2.plot(adx_dates, minus_di, 'red', linewidth=1, label='-DI')
            
            ax2.set_ylabel('ADX/DMI')
            ax2.legend(fontsize=6)
            ax2.grid(True, alpha=0.3)
            
            # 3. Gráfico de anomalía de compras/ventas
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            
            # Simular datos de anomalía
            anomaly_dates = [dates.iloc[-1]]
            anomaly_values = [alert_data['amount_millions']]
            colors = ['green' if alert_data['alert_type'] == 'COMPRA' else 'red']
            
            ax3.scatter(anomaly_dates, anomaly_values, color=colors, s=100, zorder=5)
            ax3.axhline(y=alert_data['hist_avg']/1_000_000, color='gray', linestyle='--', alpha=0.5)
            
            ax3.set_ylabel('Volumen (Millones USD)')
            ax3.set_title(f"Volumen Atípico: {alert_data['amount_millions']:.2f}M USD", 
                         fontsize=9, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Añadir texto informativo
            text_info = f"Tipo: {alert_data['alert_type']}\n"
            text_info += f"Cambio 24h: {alert_data['percent_change']:.2f}%\n"
            text_info += f"Promedio histórico: {alert_data['hist_avg']/1_000_000:.2f}M"
            
            ax3.text(0.02, 0.98, text_info, transform=ax3.transAxes,
                    fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen estrategia 2: {e}")
        return None

async def send_telegram_alert_with_image(alert_data, alert_type='strategy1'):
    """Enviar alerta a Telegram con imagen"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'strategy1':
            # Generar imagen
            img_buffer = generate_telegram_image_strategy1(alert_data)
            
            if img_buffer:
                # Crear mensaje conciso
                signal_icon = "📈" if alert_data['signal'] == 'LONG' else "📉"
                message = f"""
{signal_icon} *ALERTA DE {alert_data['signal']} - {alert_data['symbol']}*
⏰ Temporalidad: {alert_data['interval']}
📊 Score: {alert_data['score']:.1f}%

💰 Precio Actual: ${alert_data['current_price']:.6f}
🎯 Entrada: ${alert_data['entry']:.6f}

✅ Condiciones cumplidas:
{chr(10).join(['• ' + cond for cond in alert_data.get('fulfilled_conditions', [])])}

📈 MA200: {'ENCIMA ✅' if alert_data.get('ma200_condition') == 'above' else 'DEBAJO ⚠️'}
🌀 Multi-TF: {'CONFIRMADO ✅' if alert_data.get('multi_timeframe_ok') else 'NO CONFIRMADO ❌'}
"""
                
                await bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=img_buffer,
                    caption=message,
                    parse_mode='Markdown'
                )
                print(f"Alerta estrategia 1 enviada: {alert_data['symbol']}")
        
        elif alert_type == 'strategy2':
            # Generar imagen para estrategia 2
            img_buffer = generate_telegram_image_strategy2(alert_data)
            
            if img_buffer:
                # Mensaje específico para volumen atípico
                if alert_data['alert_type'] == 'COMPRA':
                    message = f"""
🚨*Alerta de COMPRA Atípica* 🚨

Se acaba de ingresar o comprar {alert_data['symbol']} en {alert_data['amount_millions']:.2f} millones de USDT, volumen atípico, revisar LONG

📊 Volumen actual: {alert_data['amount_millions']:.2f}M USD
📈 Cambio 24h: {alert_data['percent_change']:.2f}%
⏰ {alert_data['timestamp']}
"""
                else:
                    message = f"""
🚨*Alerta de VENTA Atípica* 🚨

Se vendieron {alert_data['amount_millions']:.2f} millones de USDT en {alert_data['symbol']}, volumen atípico, revisar SHORT

📊 Volumen actual: {alert_data['amount_millions']:.2f}M USD
📉 Cambio 24h: {alert_data['percent_change']:.2f}%
⏰ {alert_data['timestamp']}
"""
                
                await bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=img_buffer,
                    caption=message,
                    parse_mode='Markdown'
                )
                print(f"Alerta estrategia 2 enviada: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            current_time = datetime.now()
            
            # Verificar alertas estrategia 1 cada 60 segundos
            if current_time.second % 60 == 0:
                print("Verificando alertas estrategia 1...")
                
                alerts = indicator.generate_scalping_alerts()
                for alert in alerts:
                    asyncio.run(send_telegram_alert_with_image(alert, 'strategy1'))
                    time.sleep(1)  # Espaciar alertas
            
            # Verificar alertas estrategia 2 cada 300 segundos (5 minutos)
            if current_time.second % 300 == 0:
                print("Verificando alertas estrategia 2...")
                
                volume_alerts = indicator.detect_cmc_volume_spike()
                for alert in volume_alerts:
                    asyncio.run(send_telegram_alert_with_image(alert, 'strategy2'))
                    time.sleep(1)
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error en background_alert_checker: {e}")
            time.sleep(60)

# Iniciar verificador de alertas
try:
    alert_thread = Thread(target=background_alert_checker, daemon=True)
    alert_thread.start()
    print("Background alert checker iniciado")
except Exception as e:
    print(f"Error iniciando alert checker: {e}")

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
        
        # Convertir numpy arrays a listas
        if 'indicators' in signal_data:
            for key in signal_data['indicators']:
                if isinstance(signal_data['indicators'][key], (np.ndarray, np.generic)):
                    signal_data['indicators'][key] = signal_data['indicators'][key].tolist()
                elif isinstance(signal_data['indicators'][key], list):
                    signal_data['indicators'][key] = [bool(x) if isinstance(x, (bool, np.bool_)) else x 
                                                     for x in signal_data['indicators'][key]]
        
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        return jsonify({'error': str(e)}), 500

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
        
        for symbol in CRYPTO_SYMBOLS[:8]:
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
            'long_signals': long_signals[:5],
            'short_signals': short_signals[:5],
            'total_signals': len(all_signals)
        })
        
    except Exception as e:
        print(f"Error en /api/multiple_signals: {e}")
        return jsonify({'error': str(e)}), 500

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
                    
                    # Calcular presiones
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
        return jsonify({'alerts': alerts[:5]})
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/volume_anomaly_signals')
def get_volume_anomaly_signals():
    """Endpoint para señales de volumen atípico"""
    try:
        alerts = indicator.detect_cmc_volume_spike()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/volume_anomaly_signals: {e}")
        return jsonify({'alerts': []})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo - CORREGIDO"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals_improved(symbol, interval, leverage=leverage)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        fig = plt.figure(figsize=(14, 18))
        fig.patch.set_facecolor('#121212')
        
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
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='white', linewidth=0.5)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Mostrar 4 soportes y resistencias
            for i, support in enumerate(signal_data['supports'][:4]):
                ax1.axhline(y=support, color='blue', linestyle='--', alpha=0.5, 
                           label=f'S{i+1}' if i == 0 else '')
            
            for i, resistance in enumerate(signal_data['resistances'][:4]):
                ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5, 
                           label=f'R{i+1}' if i == 0 else '')
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='yellow', linestyle='-', alpha=0.8, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='-', alpha=0.8, label='Stop Loss')
            
            for i, tp in enumerate(signal_data['take_profit']):
                ax1.axhline(y=tp, color='green', linestyle='-', alpha=0.8, label=f'TP{i+1}' if i == 0 else '')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', 
                     fontsize=14, fontweight='bold', color='white')
        ax1.set_ylabel('Precio (USDT)', color='white')
        ax1.tick_params(colors='white')
        ax1.legend(facecolor='#121212', edgecolor='white', labelcolor='white')
        ax1.grid(True, alpha=0.3, color='white')
        ax1.set_facecolor('#121212')
        
        # Gráfico 2: Ballenas
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax2.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Compradoras')
            ax2.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Vendedoras')
        ax2.set_ylabel('Fuerza Ballenas', color='white')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#121212', edgecolor='white', labelcolor='white')
        ax2.grid(True, alpha=0.3, color='white')
        ax2.set_facecolor('#121212')
        
        # Gráfico 3: ADX/DMI
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax3.plot(adx_dates, signal_data['indicators']['adx'], 
                    'white', linewidth=2, label='ADX')
            ax3.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax3.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax3.axhline(y=25, color='yellow', linestyle='--', alpha=0.7)
        ax3.set_ylabel('ADX/DMI', color='white')
        ax3.tick_params(colors='white')
        ax3.legend(facecolor='#121212', edgecolor='white', labelcolor='white')
        ax3.grid(True, alpha=0.3, color='white')
        ax3.set_facecolor('#121212')
        
        # Gráfico 4: RSI Tradicional
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax4.plot(rsi_dates, signal_data['indicators']['rsi_traditional'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax4.set_ylabel('RSI Tradicional', color='white')
        ax4.tick_params(colors='white')
        ax4.legend(facecolor='#121212', edgecolor='white', labelcolor='white')
        ax4.grid(True, alpha=0.3, color='white')
        ax4.set_facecolor('#121212')
        
        # Gráfico 5: RSI Maverick
        ax5 = plt.subplot(9, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_maverick_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax5.plot(rsi_maverick_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax5.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
            ax5.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        ax5.set_ylabel('RSI Maverick', color='white')
        ax5.tick_params(colors='white')
        ax5.legend(facecolor='#121212', edgecolor='white', labelcolor='white')
        ax5.grid(True, alpha=0.3, color='white')
        ax5.set_facecolor('#121212')
        
        # Gráfico 6: MACD
        ax6 = plt.subplot(9, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax6.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax6.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax6.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6, label='Histograma')
            
            ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax6.set_ylabel('MACD', color='white')
        ax6.tick_params(colors='white')
        ax6.legend(facecolor='#121212', edgecolor='white', labelcolor='white')
        ax6.grid(True, alpha=0.3, color='white')
        ax6.set_facecolor('#121212')
        
        # Gráfico 7: Volumen
        ax7 = plt.subplot(9, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_ma']):]
            
            # Colores según dirección
            colors = []
            for i in range(len(volume_dates)):
                if signal_data['indicators']['volume_direction'][i] == 1:
                    colors.append('green')
                elif signal_data['indicators']['volume_direction'][i] == -1:
                    colors.append('red')
                else:
                    colors.append('gray')
            
            ax7.bar(volume_dates, signal_data['indicators']['volume_ma'], 
                   color=colors, alpha=0.6)
            
            # Marcar anomalías
            for i, date in enumerate(volume_dates):
                if signal_data['indicators']['volume_anomaly'][i]:
                    ax7.plot(date, signal_data['indicators']['volume_ma'][i], 
                            'o', color='yellow', markersize=8)
        
        ax7.set_ylabel('Volumen', color='white')
        ax7.tick_params(colors='white')
        ax7.grid(True, alpha=0.3, color='white')
        ax7.set_facecolor('#121212')
        
        # Gráfico 8: Fuerza de Tendencia
        ax8 = plt.subplot(9, 1, 8, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength']
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax8.bar(trend_dates[i], trend_strength[i], color=color, alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax8.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7)
                ax8.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        
        ax8.set_ylabel('Fuerza Tendencia %', color='white')
        ax8.tick_params(colors='white')
        ax8.grid(True, alpha=0.3, color='white')
        ax8.set_facecolor('#121212')
        
        # Información de la señal
        ax9 = plt.subplot(9, 1, 9)
        ax9.axis('off')
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        
        MULTI-TIMEFRAME: {'✅ CONFIRMADO' if signal_data.get('multi_timeframe_ok') else '❌ NO CONFIRMADO'}
        MA200: {'ENCIMA ✅' if signal_data.get('ma200_condition') == 'above' else 'DEBAJO ⚠️'}
        
        PRECIO ACTUAL: ${signal_data['current_price']:.6f}
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${signal_data['take_profit'][0]:.6f}
        
        APALANCAMIENTO: x{leverage}
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])])}
        """
        
        ax9.text(0.1, 0.9, signal_info, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='#121212', edgecolor='none')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png', 
                        as_attachment=True, 
                        download_name=f'report_{symbol}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        
    except Exception as e:
        print(f"Error generando reporte: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generando reporte: {str(e)}'}), 500

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

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
