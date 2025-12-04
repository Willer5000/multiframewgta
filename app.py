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

app = Flask(__name__)

# Configuración Telegram
TELEGRAM_BOT_TOKEN = "8007748376:AAHIW8n9b-BtA378g4gF-0-D2mOhn495Q0g"
TELEGRAM_CHAT_ID = "-1003229814161"

# Configuración CoinMarketCap
CMC_API_KEY = "d22df0c59e5e47e0980b89f6eb32ea1b"

# Configuración optimizada - 40 criptomonedas top
CRYPTO_SYMBOLS = [
    # Bajo Riesgo (20) - Top market cap
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
    "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT", "ETC-USDT",
    "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "EOS-USDT",
    
    # Medio Riesgo (10)
    "NEAR-USDT", "AXS-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ONE-USDT", "WAVES-USDT",
    
    # Alto Riesgo (7)
    "APE-USDT", "GMT-USDT", "SAND-USDT", "OP-USDT", "ARB-USDT",
    "MAGIC-USDT", "RNDR-USDT",
    
    # Memecoins (3)
    "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT"
]

# Clasificación de riesgo
CRYPTO_RISK_CLASSIFICATION = {
    "bajo": CRYPTO_SYMBOLS[:20],
    "medio": CRYPTO_SYMBOLS[20:30],
    "alto": CRYPTO_SYMBOLS[30:37],
    "memecoins": CRYPTO_SYMBOLS[37:]
}

# Mapeo para CoinMarketCap
CMC_SYMBOL_MAP = {
    "BTC-USDT": "BTC",
    "ETH-USDT": "ETH",
    "BNB-USDT": "BNB",
    "SOL-USDT": "SOL",
    "XRP-USDT": "XRP",
    "ADA-USDT": "ADA",
    "AVAX-USDT": "AVAX",
    "DOT-USDT": "DOT",
    "LINK-USDT": "LINK",
    "DOGE-USDT": "DOGE",
    "LTC-USDT": "LTC",
    "BCH-USDT": "BCH",
    "ATOM-USDT": "ATOM",
    "XLM-USDT": "XLM",
    "ETC-USDT": "ETC",
    "FIL-USDT": "FIL",
    "ALGO-USDT": "ALGO",
    "ICP-USDT": "ICP",
    "VET-USDT": "VET",
    "EOS-USDT": "EOS",
    "NEAR-USDT": "NEAR",
    "AXS-USDT": "AXS",
    "EGLD-USDT": "EGLD",
    "HBAR-USDT": "HBAR",
    "GRT-USDT": "GRT",
    "ENJ-USDT": "ENJ",
    "CHZ-USDT": "CHZ",
    "BAT-USDT": "BAT",
    "ONE-USDT": "ONE",
    "WAVES-USDT": "WAVES",
    "APE-USDT": "APE",
    "GMT-USDT": "GMT",
    "SAND-USDT": "SAND",
    "OP-USDT": "OP",
    "ARB-USDT": "ARB",
    "MAGIC-USDT": "MAGIC",
    "RNDR-USDT": "RNDR",
    "SHIB-USDT": "SHIB",
    "PEPE-USDT": "PEPE",
    "FLOKI-USDT": "FLOKI"
}

# Mapeo de temporalidades
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
        self.cmc_cache = {}
        self.cmc_volume_history = {}
        self.cmc_alert_cache = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        
    def get_bolivia_time(self):
        return datetime.now(self.bolivia_tz)
    
    def is_scalping_time(self):
        now = self.get_bolivia_time()
        if now.weekday() >= 5:
            return False
        return 4 <= now.hour < 16

    def calculate_remaining_time(self, interval, current_time):
        if interval == '15m':
            next_close = current_time.replace(minute=current_time.minute // 15 * 15, second=0, microsecond=0) + timedelta(minutes=15)
            return (next_close - current_time).total_seconds() <= (15 * 60 * 0.75)
        elif interval == '30m':
            next_close = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0) + timedelta(minutes=30)
            return (next_close - current_time).total_seconds() <= (30 * 60 * 0.75)
        elif interval == '1h':
            next_close = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return (next_close - current_time).total_seconds() <= (60 * 60 * 0.5)
        elif interval == '2h':
            current_hour = current_time.hour
            next_2h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour % 2 == 0:
                next_2h_close += timedelta(hours=2)
            else:
                next_2h_close += timedelta(hours=1)
            return (next_2h_close - current_time).total_seconds() <= (120 * 60 * 0.5)
        elif interval == '4h':
            current_hour = current_time.hour
            next_4h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 4
            if remainder == 0:
                next_4h_close += timedelta(hours=4)
            else:
                next_4h_close += timedelta(hours=4 - remainder)
            return (next_4h_close - current_time).total_seconds() <= (240 * 60 * 0.25)
        elif interval == '8h':
            current_hour = current_time.hour
            next_8h_close = current_time.replace(minute=0, second=0, microsecond=0)
            remainder = current_hour % 8
            if remainder == 0:
                next_8h_close += timedelta(hours=8)
            else:
                next_8h_close += timedelta(hours=8 - remainder)
            return (next_8h_close - current_time).total_seconds() <= (480 * 60 * 0.25)
        elif interval == '12h':
            current_hour = current_time.hour
            next_12h_close = current_time.replace(minute=0, second=0, microsecond=0)
            if current_hour < 8:
                next_12h_close = next_12h_close.replace(hour=20)
            else:
                next_12h_close = next_12h_close.replace(hour=8) + timedelta(days=1)
            return (next_12h_close - current_time).total_seconds() <= (720 * 60 * 0.25)
        elif interval == '1D':
            tomorrow_8pm = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
            if current_time.hour >= 20:
                tomorrow_8pm += timedelta(days=1)
            return (tomorrow_8pm - current_time).total_seconds() <= (1440 * 60 * 0.25)
        elif interval == '1W':
            days_passed = current_time.weekday()
            next_monday = current_time + timedelta(days=(7 - days_passed))
            next_monday = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            return (next_monday - current_time).total_seconds() <= (10080 * 60 * 0.1)
        
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
                        return self.generate_sample_data(limit, interval, symbol)
                    
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
        
        return self.generate_sample_data(limit, interval, symbol)

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

    def get_cmc_volume_data(self):
        """Obtener datos de volumen de CoinMarketCap"""
        try:
            cache_key = "cmc_volume_data"
            if cache_key in self.cmc_cache:
                cached_data, timestamp = self.cmc_cache[cache_key]
                if (datetime.now() - timestamp).seconds < 300:  # Cache 5 minutos
                    return cached_data
            
            headers = {
                'X-CMC_PRO_API_KEY': CMC_API_KEY,
                'Accept': 'application/json'
            }
            
            # Obtener las primeras 100 criptomonedas
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            params = {
                'start': '1',
                'limit': '100',
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                cmc_data = {}
                
                for crypto in data['data']:
                    symbol = crypto['symbol'] + "-USDT"
                    if symbol in CRYPTO_SYMBOLS:
                        volume_24h = crypto['quote']['USD'].get('volume_24h', 0)
                        percent_change_24h = crypto['quote']['USD'].get('percent_change_24h', 0)
                        price = crypto['quote']['USD'].get('price', 0)
                        
                        # Guardar en historial
                        if symbol not in self.cmc_volume_history:
                            self.cmc_volume_history[symbol] = []
                        
                        self.cmc_volume_history[symbol].append({
                            'timestamp': datetime.now(),
                            'volume': volume_24h,
                            'price': price,
                            'change': percent_change_24h
                        })
                        
                        # Mantener solo últimos 7 días
                        if len(self.cmc_volume_history[symbol]) > 1008:  # 7 días * 24h * 6 (cada 10 min)
                            self.cmc_volume_history[symbol] = self.cmc_volume_history[symbol][-1008:]
                        
                        cmc_data[symbol] = {
                            'volume_24h': volume_24h,
                            'percent_change_24h': percent_change_24h,
                            'price': price,
                            'symbol': crypto['symbol']
                        }
                
                self.cmc_cache[cache_key] = (cmc_data, datetime.now())
                return cmc_data
                
        except Exception as e:
            print(f"Error obteniendo datos de CoinMarketCap: {e}")
        
        return {}

    def detect_cmc_volume_spike(self, symbol, cmc_data):
        """Detectar spikes de volumen en CoinMarketCap"""
        try:
            if symbol not in cmc_data:
                return None
            
            current_data = cmc_data[symbol]
            current_volume = current_data['volume_24h']
            current_change = current_data['percent_change_24h']
            
            # Obtener histórico de 7 días
            if symbol in self.cmc_volume_history and len(self.cmc_volume_history[symbol]) > 10:
                historical_volumes = [d['volume'] for d in self.cmc_volume_history[symbol][:-1]]
                if historical_volumes:
                    avg_volume = np.mean(historical_volumes)
                    std_volume = np.std(historical_volumes)
                    
                    # Detectar spike (300% sobre promedio o 3 desviaciones estándar)
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    if volume_ratio >= 3.0 or (std_volume > 0 and current_volume > avg_volume + (3 * std_volume)):
                        # Determinar dirección
                        if current_change > 0:
                            direction = "COMPRA"
                            signal_type = "LONG"
                        else:
                            direction = "VENTA"
                            signal_type = "SHORT"
                        
                        # Formatear monto en millones
                        amount_millions = current_volume / 1000000
                        
                        return {
                            'symbol': symbol,
                            'direction': direction,
                            'signal_type': signal_type,
                            'volume': current_volume,
                            'amount_formatted': f"{amount_millions:.2f}",
                            'volume_ratio': volume_ratio,
                            'price_change': current_change,
                            'current_price': current_data['price'],
                            'timestamp': datetime.now()
                        }
            
        except Exception as e:
            print(f"Error detectando spike de volumen para {symbol}: {e}")
        
        return None

    def calculate_support_resistance(self, high, low, close, period=50):
        """Calcular 4 niveles de soporte/resistencia"""
        n = len(close)
        supports = []
        resistances = []
        
        # Método de pivot points
        for i in range(period, n, period):
            start_idx = i - period
            end_idx = i
            
            window_high = high[start_idx:end_idx]
            window_low = low[start_idx:end_idx]
            window_close = close[start_idx:end_idx]
            
            if len(window_high) > 0:
                # Pivot Point clásico
                pivot = (np.max(window_high) + np.min(window_low) + window_close[-1]) / 3
                
                # Niveles de soporte y resistencia
                r1 = (2 * pivot) - np.min(window_low)
                s1 = (2 * pivot) - np.max(window_high)
                r2 = pivot + (np.max(window_high) - np.min(window_low))
                s2 = pivot - (np.max(window_high) - np.min(window_low))
                
                # Solo agregar si están en rango razonable
                current_price = close[end_idx-1]
                price_range = np.max(window_high) - np.min(window_low)
                
                levels = [s2, s1, r1, r2]
                for level in levels:
                    if abs(level - current_price) < price_range * 2:
                        if level < current_price:
                            supports.append(level)
                        else:
                            resistances.append(level)
        
        # Ordenar y obtener los 4 más cercanos
        supports.sort(reverse=True)  # Más alto al más bajo
        resistances.sort()  # Más bajo al más alto
        
        # Tomar hasta 4 niveles de cada uno, pero priorizar los más cercanos
        final_supports = supports[:4]
        final_resistances = resistances[:4]
        
        # Si no hay suficientes, usar extensiones de Fibonacci
        if len(final_supports) < 2:
            price_range = np.max(high[-period*2:]) - np.min(low[-period*2:])
            current_price = close[-1]
            
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            for fib in fib_levels:
                level = current_price - (price_range * fib)
                if level > 0:
                    final_supports.append(level)
            
            final_supports = sorted(list(set(final_supports)), reverse=True)[:4]
        
        if len(final_resistances) < 2:
            price_range = np.max(high[-period*2:]) - np.min(low[-period*2:])
            current_price = close[-1]
            
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            for fib in fib_levels:
                level = current_price + (price_range * fib)
                final_resistances.append(level)
            
            final_resistances = sorted(list(set(final_resistances)))[:4]
        
        return final_supports, final_resistances

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=15):
        """Calcular entradas y salidas óptimas con soportes/resistencias"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calcular 4 niveles de soporte/resistencia
            supports, resistances = self.calculate_support_resistance(high, low, close)
            
            if signal_type == 'LONG':
                # Buscar soporte más cercano por encima del precio actual
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)
                else:
                    entry = current_price * 0.995  # Pequeño descuento si no hay soporte
                
                # Stop loss en el siguiente soporte o usando ATR
                if len(supports) > 1 and supports[1] < entry:
                    stop_loss = supports[1]
                else:
                    stop_loss = entry - (current_atr * 1.5)
                
                # Take profit en resistencias
                take_profits = []
                for resistance in resistances[:2]:  # Primeras 2 resistencias
                    if resistance > entry:
                        take_profits.append(resistance)
                
                if not take_profits:
                    take_profits = [entry + (2 * (entry - stop_loss))]
                
            else:  # SHORT
                # Buscar resistencia más cercana por debajo del precio actual
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)
                else:
                    entry = current_price * 1.005  # Pequeño incremento si no hay resistencia
                
                # Stop loss en la siguiente resistencia o usando ATR
                if len(resistances) > 1 and resistances[1] > entry:
                    stop_loss = resistances[1]
                else:
                    stop_loss = entry + (current_atr * 1.5)
                
                # Take profit en soportes
                take_profits = []
                for support in supports[:2]:  # Primeros 2 soportes
                    if support < entry:
                        take_profits.append(support)
                
                if not take_profits:
                    take_profits = [entry - (2 * (stop_loss - entry))]
            
            atr_percentage = current_atr / current_price if current_price > 0 else 0
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits],
                'supports': [float(s) for s in supports[:4]],
                'resistances': [float(r) for r in resistances[:4]],
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.05],
                'supports': [current_price * 0.95, current_price * 0.93, current_price * 0.90, current_price * 0.87],
                'resistances': [current_price * 1.05, current_price * 1.08, current_price * 1.10, current_price * 1.12],
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

    def calculate_ema(self, prices, period):
        """Calcular EMA"""
        if len(prices) == 0 or period <= 0:
            return np.zeros_like(prices)
            
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0] if len(prices) > 0 else 0
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def calculate_sma(self, prices, period):
        """Calcular SMA"""
        if len(prices) == 0 or period <= 0:
            return np.zeros_like(prices)
            
        sma = np.zeros_like(prices)
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window = prices[start_idx:i+1]
            sma[i] = np.mean(window) if len(window) > 0 else prices[i]
        
        return sma

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
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower

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
        
        rsi = 100 - (100 / (1 + rs))
        rsi = np.nan_to_num(rsi, nan=50, posinf=100, neginf=0)
        
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
            
            # Determinar umbral alto
            if n >= 50:
                historical_bb_width = bb_width[max(0, n-100):n]
                high_zone_threshold = np.percentile(historical_bb_width, 70) if len(historical_bb_width) > 0 else 5
            else:
                high_zone_threshold = np.percentile(bb_width, 70) if len(bb_width) > 0 else 5
            
            no_trade_zones = np.zeros(n, dtype=bool)
            strength_signals = ['NEUTRAL'] * n
            
            for i in range(10, n):
                # Zona NO OPERAR: ancho alto y tendencia decreciente
                if (bb_width[i] > high_zone_threshold and 
                    trend_strength[i] < 0):
                    no_trade_zones[i] = True
                
                # Señales de fuerza
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

    def check_bollinger_conditions(self, df, interval, signal_type):
        """Verificar condiciones de Bandas de Bollinger"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            current_idx = -1
            current_price = close[current_idx]
            current_volume = volume[current_idx]
            avg_volume = np.mean(volume[-20:])
            
            if signal_type == 'LONG':
                # Condiciones para LONG
                touch_lower = current_price <= bb_lower[current_idx] * 1.02
                bounce_lower = (current_price > bb_lower[current_idx] and 
                               close[current_idx-1] <= bb_lower[current_idx-1] * 1.01)
                
                return touch_lower or bounce_lower
                
            else:  # SHORT
                # Condiciones para SHORT
                touch_upper = current_price >= bb_upper[current_idx] * 0.98
                rejection_upper = (current_price < bb_upper[current_idx] and 
                                 close[current_idx-1] >= bb_upper[current_idx-1] * 0.99)
                
                return touch_upper or rejection_upper
                
        except Exception as e:
            print(f"Error verificando condiciones Bollinger: {e}")
            return False

    def check_multi_timeframe_obligatory(self, symbol, interval, signal_type):
        """Verificar condiciones multi-timeframe obligatorias"""
        try:
            # Para 12h, 1D, 1W no aplica Multi-Timeframe
            if interval in ['12h', '1D', '1W']:
                return True
                
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return False
            
            # TF Mayor: puede estar en dirección o neutral
            mayor_df = self.get_kucoin_data(symbol, hierarchy['mayor'], 30)
            if mayor_df is not None and len(mayor_df) > 10:
                mayor_trend = self.calculate_trend_strength_maverick(mayor_df['close'].values)
                mayor_ok = mayor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP', 'NEUTRAL'] if signal_type == 'LONG' else mayor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL']
            else:
                mayor_ok = True
            
            # TF Medio: debe estar en dirección de la señal
            media_df = self.get_kucoin_data(symbol, hierarchy['media'], 30)
            if media_df is not None and len(media_df) > 10:
                media_trend = self.calculate_trend_strength_maverick(media_df['close'].values)
                media_ok = media_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP'] if signal_type == 'LONG' else media_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
            else:
                media_ok = True
            
            # TF Menor: dirección de señal y fuera zona NO OPERAR
            menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
            if menor_df is not None and len(menor_df) > 10:
                menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                menor_direction = menor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP'] if signal_type == 'LONG' else menor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
                menor_no_trade = not menor_trend['no_trade_zones'][-1]
                menor_ok = menor_direction and menor_no_trade
            else:
                menor_ok = True
            
            return mayor_ok and media_ok and menor_ok
            
        except Exception as e:
            print(f"Error verificando condiciones multi-timeframe: {e}")
            return False

    def calculate_whale_signals(self, df, sensitivity=1.7, min_volume_multiplier=1.5, 
                               support_resistance_lookback=50, signal_threshold=25):
        """Indicador de ballenas"""
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
                
                # Señal de compra (ballena compradora)
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] < close[i-1] or price_change < -0.5) and
                    low[i] <= low_5 * 1.01):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_pump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
                
                # Señal de venta (ballena vendedora)
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_dump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
            
            # Suavizar señales
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            # Confirmar señales
            current_support = np.array([np.min(low[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            current_resistance = np.array([np.max(high[max(0, i-support_resistance_lookback+1):i+1]) for i in range(n)])
            
            for i in range(5, n):
                if (whale_pump_smooth[i] > signal_threshold and 
                    close[i] <= current_support[i] * 1.02):
                    confirmed_buy[i] = True
                
                if (whale_dump_smooth[i] > signal_threshold and 
                    close[i] >= current_resistance[i] * 0.98):
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
            print(f"Error en calculate_whale_signals: {e}")
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
            
            basis = np.array([np.mean(close[max(0, i-length+1):i+1]) for i in range(n)])
            dev = np.array([np.std(close[max(0, i-length+1):i+1]) for i in range(n)])
            
            upper = basis + (dev * bb_multiplier)
            lower = basis - (dev * bb_multiplier)
            
            b_percent = np.zeros(n)
            for i in range(n):
                if (upper[i] - lower[i]) > 0:
                    b_percent[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
            
            return b_percent.tolist()
            
        except Exception as e:
            print(f"Error en calculate_rsi_maverick: {e}")
            return [0.5] * len(close)

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias"""
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
        
        return bullish_div.tolist(), bearish_div.tolist()

    def check_breakout(self, high, low, close, supports, resistances):
        """Detectar rupturas"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Ruptura alcista
            for resistance in resistances:
                if close[i] > resistance and high[i] > high[i-1]:
                    breakout_up[i] = True
                    break
            
            # Ruptura bajista
            for support in supports:
                if close[i] < support and low[i] < low[i-1]:
                    breakout_down[i] = True
                    break
        
        return breakout_up.tolist(), breakout_down.tolist()

    def check_di_crossover(self, plus_di, minus_di, lookback=3):
        """Detectar cruces de DMI"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1]):
                di_cross_bullish[i] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i] > plus_di[i] and 
                minus_di[i-1] <= plus_di[i-1]):
                di_cross_bearish[i] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

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

    def detect_chart_patterns(self, high, low, close, lookback=50):
        """Detectar patrones chartistas"""
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
            
            # Doble Techo
            if len(window_high) >= 15:
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append((j, window_high[j]))
                
                if len(peaks) >= 2:
                    last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                    if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                        patterns['double_top'][i] = True
            
            # Doble Fondo
            if len(window_low) >= 15:
                troughs = []
                for j in range(1, len(window_low)-1):
                    if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                        troughs.append((j, window_low[j]))
                
                if len(troughs) >= 2:
                    last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                    if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.02:
                        patterns['double_bottom'][i] = True
            
            # Bandera Alcista (simplificada)
            if i > 10:
                # Fuerte movimiento alcista seguido de consolidación
                strong_up_move = (close[i-10] < close[i-5]) and ((close[i-5] - close[i-10]) / close[i-10] > 0.05)
                consolidation = np.std(close[i-5:i+1]) / close[i] < 0.01
                if strong_up_move and consolidation:
                    patterns['bullish_flag'][i] = True
            
            # Bandera Bajista (simplificada)
            if i > 10:
                # Fuerte movimiento bajista seguido de consolidación
                strong_down_move = (close[i-10] > close[i-5]) and ((close[i-10] - close[i-5]) / close[i-10] > 0.05)
                consolidation = np.std(close[i-5:i+1]) / close[i] < 0.01
                if strong_down_move and consolidation:
                    patterns['bearish_flag'][i] = True
        
        return patterns

    def check_volume_anomaly(self, volume, period=20):
        """Verificar anomalías de volumen"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            
            for i in range(period, n):
                window = volume[i-period:i]
                mean_volume = np.mean(window)
                std_volume = np.std(window)
                
                if std_volume > 0:
                    z_score = (volume[i] - mean_volume) / std_volume
                    if z_score > 2:  # Más de 2 desviaciones estándar
                        volume_anomaly[i] = True
            
            return volume_anomaly.tolist()
            
        except Exception as e:
            print(f"Error verificando anomalías de volumen: {e}")
            return [False] * len(volume)

    def check_ma_crossover(self, ma_short, ma_long, lookback=2):
        """Verificar cruce de medias móviles"""
        n = len(ma_short)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: MA corta cruza por encima de MA larga
            if (ma_short[i] > ma_long[i] and 
                ma_short[i-1] <= ma_long[i-1]):
                ma_cross_bullish[i] = True
            
            # Cruce bajista: MA corta cruza por debajo de MA larga
            if (ma_short[i] < ma_long[i] and 
                ma_short[i-1] >= ma_long[i-1]):
                ma_cross_bearish[i] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()

    def check_macd_crossover(self, macd, macd_signal, lookback=2):
        """Verificar cruce de MACD"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # Cruce alcista: MACD cruza por encima de la señal
            if (macd[i] > macd_signal[i] and 
                macd[i-1] <= macd_signal[i-1]):
                macd_cross_bullish[i] = True
            
            # Cruce bajista: MACD cruza por debajo de la señal
            if (macd[i] < macd_signal[i] and 
                macd[i-1] >= macd_signal[i-1]):
                macd_cross_bearish[i] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()

    def check_adx_trend(self, adx, lookback=5, threshold=25):
        """Verificar tendencia ADX"""
        n = len(adx)
        adx_trend_bullish = np.zeros(n, dtype=bool)
        adx_trend_bearish = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n):
            # ADX con pendiente positiva y por encima del umbral
            if adx[i] > threshold and adx[i] > np.mean(adx[i-lookback:i]):
                adx_trend_bullish[i] = True
            
            # ADX con pendiente negativa o por debajo del umbral
            if adx[i] < threshold or adx[i] < np.mean(adx[i-lookback:i]):
                adx_trend_bearish[i] = True
        
        return adx_trend_bullish.tolist(), adx_trend_bearish.tolist()

    def evaluate_signal_conditions(self, data, current_idx, interval):
        """Evaluar condiciones de señal con nuevos pesos"""
        # Definir pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            weights = {
                'long': {
                    'multi_timeframe': 30,  # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                },
                'short': {
                    'multi_timeframe': 30,  # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                }
            }
        elif interval in ['12h', '1D']:
            weights = {
                'long': {
                    'whale_signal': 30,     # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                },
                'short': {
                    'whale_signal': 30,     # Obligatorio
                    'trend_strength': 25,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                }
            }
        else:  # 1W
            weights = {
                'long': {
                    'trend_strength': 55,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
                },
                'short': {
                    'trend_strength': 55,   # Obligatorio
                    'bollinger_bands': 8,
                    'adx_dmi': 10,
                    'ma_cross': 10,
                    'rsi_traditional_divergence': 5,
                    'rsi_maverick_divergence': 8,
                    'macd': 10,
                    'chart_pattern': 5,
                    'breakout': 5,
                    'volume_anomaly': 7
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
        
        # Obtener valores actuales
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
        
        # ADX con DMI (cruce)
        conditions['long']['adx_dmi']['value'] = (
            data['di_cross_bullish'][current_idx] if current_idx < len(data['di_cross_bullish']) else False
        )
        
        # Cruce de medias (9 y 21)
        conditions['long']['ma_cross']['value'] = (
            data['ma_cross_bullish'][current_idx] if current_idx < len(data['ma_cross_bullish']) else False
        )
        
        # Divergencias RSI
        conditions['long']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bullish_divergence']) and 
            data['rsi_bullish_divergence'][current_idx]
        )
        
        conditions['long']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bullish_divergence']) and 
            data['rsi_maverick_bullish_divergence'][current_idx]
        )
        
        # MACD
        conditions['long']['macd']['value'] = (
            data['macd_cross_bullish'][current_idx] if current_idx < len(data['macd_cross_bullish']) else False
        )
        
        # Patrones chartistas
        conditions['long']['chart_pattern']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_flag'][current_idx]
        )
        
        # Breakouts
        conditions['long']['breakout']['value'] = (
            current_idx < len(data['breakout_up']) and 
            data['breakout_up'][current_idx]
        )
        
        # Volumen anómalo
        conditions['long']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly']) and 
            data['volume_anomaly'][current_idx]
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
        
        # ADX con DMI (cruce)
        conditions['short']['adx_dmi']['value'] = (
            data['di_cross_bearish'][current_idx] if current_idx < len(data['di_cross_bearish']) else False
        )
        
        # Cruce de medias (9 y 21)
        conditions['short']['ma_cross']['value'] = (
            data['ma_cross_bearish'][current_idx] if current_idx < len(data['ma_cross_bearish']) else False
        )
        
        # Divergencias RSI
        conditions['short']['rsi_traditional_divergence']['value'] = (
            current_idx < len(data['rsi_bearish_divergence']) and 
            data['rsi_bearish_divergence'][current_idx]
        )
        
        conditions['short']['rsi_maverick_divergence']['value'] = (
            current_idx < len(data['rsi_maverick_bearish_divergence']) and 
            data['rsi_maverick_bearish_divergence'][current_idx]
        )
        
        # MACD
        conditions['short']['macd']['value'] = (
            data['macd_cross_bearish'][current_idx] if current_idx < len(data['macd_cross_bearish']) else False
        )
        
        # Patrones chartistas
        conditions['short']['chart_pattern']['value'] = (
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['bearish_flag'][current_idx]
        )
        
        # Breakouts
        conditions['short']['breakout']['value'] = (
            current_idx < len(data['breakout_down']) and 
            data['breakout_down'][current_idx]
        )
        
        # Volumen anómalo
        conditions['short']['volume_anomaly']['value'] = (
            current_idx < len(data['volume_anomaly']) and 
            data['volume_anomaly'][current_idx]
        )
        
        return conditions

    def get_condition_description(self, condition_key):
        descriptions = {
            'multi_timeframe': 'Multi-Timeframe',
            'trend_strength': 'Fuerza Tendencia Maverick',
            'whale_signal': 'Indicador Ballenas',
            'bollinger_bands': 'Bandas de Bollinger',
            'adx_dmi': 'ADX con DMI',
            'ma_cross': 'Cruce de Medias (9/21)',
            'rsi_traditional_divergence': 'Divergencia RSI Tradicional',
            'rsi_maverick_divergence': 'Divergencia RSI Maverick',
            'macd': 'Cruce MACD',
            'chart_pattern': 'Patrón Chartista',
            'breakout': 'Ruptura',
            'volume_anomaly': 'Volumen Anómalo'
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
            if condition['weight'] >= 25:  # Condiciones con peso >= 25 son obligatorias
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
        
        # Ajustar score mínimo según MA200
        min_score = 65
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0

        return min(final_score, 100), fulfilled_conditions

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
            
            # Indicadores básicos
            whale_data = self.calculate_whale_signals(df, support_resistance_lookback=sr_period)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            # Soporte y resistencia
            supports, resistances = self.calculate_support_resistance(high, low, close)
            breakout_up, breakout_down = self.check_breakout(high, low, close, supports, resistances)
            
            # Patrones chartistas
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # Fuerza de tendencia
            trend_strength_data = self.calculate_trend_strength_maverick(close)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # Cruce de medias
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover(ma_9, ma_21)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_bullish, macd_cross_bearish = self.check_macd_crossover(macd, macd_signal)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            bollinger_conditions_long = self.check_bollinger_conditions(df, interval, 'LONG')
            bollinger_conditions_short = self.check_bollinger_conditions(df, interval, 'SHORT')
            
            # Volumen anómalo
            volume_anomaly = self.check_volume_anomaly(volume)
            
            # ADX tendencia
            adx_trend_bullish, adx_trend_bearish = self.check_adx_trend(adx)
            
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
                'adx_trend_bullish': adx_trend_bullish,
                'adx_trend_bearish': adx_trend_bearish,
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
                'volume_anomaly': volume_anomaly,
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short,
                'bollinger_conditions_long': bollinger_conditions_long,
                'bollinger_conditions_short': bollinger_conditions_short,
                'supports': supports,
                'resistances': resistances
            }
            
            conditions = self.evaluate_signal_conditions(analysis_data, current_idx, interval)
            
            # Condición MA200
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
                'volume': float(volume[current_idx]),
                'volume_ma': float(np.mean(volume[-20:])),
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
                    'adx_trend_bullish': adx_trend_bullish[-50:],
                    'adx_trend_bearish': adx_trend_bearish[-50:],
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
                    'volume_anomaly': volume_anomaly[-50:],
                    'trend_strength': trend_strength_data['trend_strength'][-50:],
                    'bb_width': trend_strength_data['bb_width'][-50:],
                    'no_trade_zones': trend_strength_data['no_trade_zones'][-50:],
                    'strength_signals': trend_strength_data['strength_signals'][-50:],
                    'high_zone_threshold': trend_strength_data['high_zone_threshold'],
                    'colors': trend_strength_data['colors'][-50:],
                    'supports': supports,
                    'resistances': resistances
                }
            }
            
        except Exception as e:
            print(f"Error en generate_signals_improved para {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_signal(symbol)

    def _create_empty_signal(self, symbol):
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
                        
                        alert_key = f"{symbol}_{interval}_{signal_data['signal']}"
                        if (alert_key not in self.alert_cache or 
                            (datetime.now() - self.alert_cache[alert_key]).seconds > 300):
                            
                            alerts.append({
                                'symbol': symbol,
                                'interval': interval,
                                'signal': signal_data['signal'],
                                'score': signal_data['signal_score'],
                                'entry': signal_data['entry'],
                                'current_price': signal_data['current_price'],
                                'fulfilled_conditions': signal_data.get('fulfilled_conditions', []),
                                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                'ma200_condition': signal_data.get('ma200_condition', 'below')
                            })
                            
                            self.alert_cache[alert_key] = datetime.now()
                    
                except Exception as e:
                    print(f"Error generando alerta para {symbol} {interval}: {e}")
                    continue
        
        return alerts

    def generate_cmc_volume_alerts(self):
        """Generar alertas de volumen atípico de CoinMarketCap"""
        alerts = []
        try:
            cmc_data = self.get_cmc_volume_data()
            
            for symbol in CRYPTO_SYMBOLS:
                volume_spike = self.detect_cmc_volume_spike(symbol, cmc_data)
                
                if volume_spike:
                    alert_key = f"cmc_{symbol}_{volume_spike['direction']}"
                    
                    # Evitar duplicados (2 horas)
                    if (alert_key not in self.cmc_alert_cache or 
                        (datetime.now() - self.cmc_alert_cache[alert_key]).seconds > 7200):
                        
                        alerts.append(volume_spike)
                        self.cmc_alert_cache[alert_key] = datetime.now()
                        print(f"Alerta CMC generada: {symbol} {volume_spike['direction']}")
        
        except Exception as e:
            print(f"Error generando alertas CMC: {e}")
        
        return alerts

# Instancia global
indicator = TradingIndicator()

def send_telegram_alert(alert_data, alert_type='multiframe'):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'multiframe':
            # Alerta estrategia 1: Multi-Timeframe
            signal_icon = "📈" if alert_data['signal'] == 'LONG' else "📉"
            ma200_icon = "🔼" if alert_data.get('ma200_condition') == 'above' else "🔽"
            
            conditions_text = "\n".join([f"• {cond}" for cond in alert_data.get('fulfilled_conditions', [])])
            
            message = f"""
{signal_icon} *ALERTA TRADING - MULTI-TIMEFRAME* {signal_icon}

*Par:* {alert_data['symbol']}
*Temporalidad:* {alert_data['interval']}
*Señal:* {alert_data['signal']}
*Score:* {alert_data['score']:.1f}%
*MA200:* {ma200_icon}

*Condiciones Cumplidas:*
{conditions_text}

*Precio Actual:* ${alert_data['current_price']:.6f}
*Entrada Recomendada:* ${alert_data['entry']:.6f}
"""
            
            # Generar y enviar imagen
            img_buffer = generate_telegram_image(alert_data, 'multiframe')
            if img_buffer:
                asyncio.run(bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=img_buffer,
                    caption=message,
                    parse_mode='Markdown'
                ))
            else:
                asyncio.run(bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                ))
            
        else:
            # Alerta estrategia 2: Volumen atípico
            if alert_data['direction'] == "COMPRA":
                message = f"""
🚨*Alerta de COMPRA Atípica*🚨

Se acaba de ingresar o comprar {alert_data['symbol']} en {alert_data['amount_formatted']} millones de USDT, volumen atípico, revisar LONG

*Volumen:* {alert_data['amount_formatted']}M USDT
*Ratio Volumen:* {alert_data['volume_ratio']:.1f}x
*Cambio Precio 24h:* {alert_data['price_change']:+.2f}%
*Precio Actual:* ${alert_data['current_price']:.6f}
"""
            else:
                message = f"""
🚨*Alerta de VENTA Atípica*🚨

Se vendieron {alert_data['amount_formatted']} millones de USDT en {alert_data['symbol']}, volumen atípico, revisar SHORT

*Volumen:* {alert_data['amount_formatted']}M USDT
*Ratio Volumen:* {alert_data['volume_ratio']:.1f}x
*Cambio Precio 24h:* {alert_data['price_change']:+.2f}%
*Precio Actual:* ${alert_data['current_price']:.6f}
"""
            
            # Generar y enviar imagen
            img_buffer = generate_telegram_image(alert_data, 'cmc_volume')
            if img_buffer:
                asyncio.run(bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=img_buffer,
                    caption=message,
                    parse_mode='Markdown'
                ))
            else:
                asyncio.run(bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                ))
        
        print(f"Alerta {alert_type} enviada a Telegram: {alert_data['symbol']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def generate_telegram_image(alert_data, strategy_type):
    """Generar imagen para Telegram"""
    try:
        if strategy_type == 'multiframe':
            return generate_multiframe_telegram_image(alert_data)
        else:
            return generate_cmc_volume_telegram_image(alert_data)
    except Exception as e:
        print(f"Error generando imagen para Telegram: {e}")
        return None

def generate_multiframe_telegram_image(alert_data):
    """Generar imagen para estrategia Multi-Timeframe"""
    try:
        symbol = alert_data['symbol']
        interval = alert_data['interval']
        
        # Obtener datos
        df = indicator.get_kucoin_data(symbol, interval, 100)
        if df is None or len(df) < 50:
            return None
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calcular indicadores
        adx, plus_di, minus_di = indicator.calculate_adx(high, low, close)
        whale_data = indicator.calculate_whale_signals(df)
        trend_strength = indicator.calculate_trend_strength_maverick(close)
        rsi_maverick = indicator.calculate_rsi_maverick(close)
        rsi_traditional = indicator.calculate_rsi(close)
        macd, macd_signal, macd_histogram = indicator.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = indicator.calculate_bollinger_bands(close)
        ma_9 = indicator.calculate_sma(close, 9)
        ma_21 = indicator.calculate_sma(close, 21)
        ma_50 = indicator.calculate_sma(close, 50)
        
        # Crear figura
        fig = plt.figure(figsize=(14, 18))
        fig.patch.set_facecolor('white')
        
        # 1. Gráfico de velas
        ax1 = plt.subplot(9, 1, 1)
        dates = pd.date_range(end=datetime.now(), periods=len(close), freq=interval)
        
        # Bandas de Bollinger transparentes
        ax1.fill_between(dates[-50:], bb_lower[-50:], bb_upper[-50:], 
                        alpha=0.1, color='blue', label='BB')
        
        # Medias móviles
        ax1.plot(dates[-50:], ma_9[-50:], 'orange', linewidth=1, label='MA9')
        ax1.plot(dates[-50:], ma_21[-50:], 'blue', linewidth=1, label='MA21')
        ax1.plot(dates[-50:], ma_50[-50:], 'purple', linewidth=1, label='MA50')
        
        # Velas
        for i in range(-50, 0):
            idx = len(close) + i
            color = 'green' if close[idx] >= df['open'].iloc[idx] else 'red'
            ax1.plot([dates[idx], dates[idx]], [low[idx], high[idx]], color='black', linewidth=1)
            ax1.plot([dates[idx], dates[idx]], [df['open'].iloc[idx], close[idx]], color=color, linewidth=3)
        
        ax1.set_title(f'{symbol} - {interval}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(9, 1, 2, sharex=ax1)
        ax2.plot(dates[-50:], adx[-50:], 'black', linewidth=2, label='ADX')
        ax2.plot(dates[-50:], plus_di[-50:], 'green', linewidth=1, label='+DI')
        ax2.plot(dates[-50:], minus_di[-50:], 'red', linewidth=1, label='-DI')
        ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('ADX/DMI')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Volumen con anomalías
        ax3 = plt.subplot(9, 1, 3, sharex=ax1)
        volume_anomaly = indicator.check_volume_anomaly(volume)
        
        # Barras de volumen
        for i in range(-50, 0):
            idx = len(volume) + i
            color = 'green' if close[idx] >= df['open'].iloc[idx] else 'red'
            ax3.bar(dates[idx], volume[idx], color=color, alpha=0.7, width=0.8)
        
        # Marcadores de anomalías
        anomaly_dates = []
        anomaly_values = []
        for i in range(-50, 0):
            idx = len(volume_anomaly) + i
            if volume_anomaly[idx]:
                anomaly_dates.append(dates[idx])
                anomaly_values.append(volume[idx])
        
        if anomaly_dates:
            ax3.scatter(anomaly_dates, anomaly_values, color='blue', s=50, zorder=5)
        
        ax3.set_ylabel('Volumen')
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick (columnas)
        ax4 = plt.subplot(9, 1, 4, sharex=ax1)
        colors = ['green' if x > 0 else 'red' for x in trend_strength['trend_strength'][-50:]]
        ax4.bar(dates[-50:], trend_strength['trend_strength'][-50:], color=colors, alpha=0.7, width=0.8)
        ax4.axhline(y=0, color='black', linewidth=0.5)
        ax4.set_ylabel('FT Maverick')
        ax4.grid(True, alpha=0.3)
        
        # 5. Ballenas (solo para 12h y 1D)
        if interval in ['12h', '1D']:
            ax5 = plt.subplot(9, 1, 5, sharex=ax1)
            ax5.bar(dates[-50:], whale_data['whale_pump'][-50:], color='green', alpha=0.7, label='Compra', width=0.8)
            ax5.bar(dates[-50:], whale_data['whale_dump'][-50:], color='red', alpha=0.7, label='Venta', width=0.8)
            ax5.set_ylabel('Ballenas')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
        
        # 6. RSI Maverick
        ax6 = plt.subplot(9, 1, 6 if interval not in ['12h', '1D'] else 7, sharex=ax1)
        ax6.plot(dates[-50:], rsi_maverick[-50:], 'blue', linewidth=2)
        ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
        ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
        ax6.set_ylabel('RSI Maverick')
        ax6.grid(True, alpha=0.3)
        
        # 7. RSI Tradicional
        ax7 = plt.subplot(9, 1, 7 if interval not in ['12h', '1D'] else 8, sharex=ax1)
        ax7.plot(dates[-50:], rsi_traditional[-50:], 'purple', linewidth=2)
        ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        ax7.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        ax7.set_ylabel('RSI Tradicional')
        ax7.grid(True, alpha=0.3)
        
        # 8. MACD (columnas para histograma)
        ax8 = plt.subplot(9, 1, 8 if interval not in ['12h', '1D'] else 9, sharex=ax1)
        colors_macd = ['green' if x > 0 else 'red' for x in macd_histogram[-50:]]
        ax8.bar(dates[-50:], macd_histogram[-50:], color=colors_macd, alpha=0.7, width=0.8)
        ax8.plot(dates[-50:], macd[-50:], 'blue', linewidth=1, label='MACD')
        ax8.plot(dates[-50:], macd_signal[-50:], 'orange', linewidth=1, label='Señal')
        ax8.axhline(y=0, color='black', linewidth=0.5)
        ax8.set_ylabel('MACD')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen Multi-Timeframe: {e}")
        return None

def generate_cmc_volume_telegram_image(alert_data):
    """Generar imagen para estrategia CMC Volume"""
    try:
        symbol = alert_data['symbol']
        
        # Obtener datos históricos de volumen
        cmc_data = indicator.get_cmc_volume_data()
        if symbol not in cmc_data:
            return None
        
        # Crear figura
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Gráfico de dispersión de volumen
        ax = plt.subplot(1, 1, 1)
        
        # Simular datos históricos (en producción sería real)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        volumes = np.random.lognormal(20, 1, 30) * 1000000
        
        # Último punto (el spike)
        volumes[-1] = float(alert_data['volume'])
        
        # Colores según dirección
        colors = ['gray'] * 29
        colors.append('green' if alert_data['direction'] == 'COMPRA' else 'red')
        
        ax.scatter(dates, volumes, c=colors, s=100, alpha=0.7)
        
        # Línea de tendencia
        z = np.polyfit(range(len(volumes)), volumes, 1)
        p = np.poly1d(z)
        ax.plot(dates, p(range(len(volumes))), 'blue', linestyle='--', alpha=0.5)
        
        # Destacar spike
        ax.annotate(f"SPIKE!\n{alert_data['amount_formatted']}M USDT", 
                   xy=(dates[-1], volumes[-1]),
                   xytext=(dates[-1] + timedelta(hours=2), volumes[-1] * 1.1),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, fontweight='bold')
        
        ax.set_title(f'Volumen Atípico - {symbol}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Volumen 24h (USDT)')
        ax.set_xlabel('Fecha')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(6,6))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Error generando imagen CMC Volume: {e}")
        return None

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    while True:
        try:
            # Verificar alertas Multi-Timeframe
            print("Verificando alertas Multi-Timeframe...")
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                send_telegram_alert(alert, 'multiframe')
            
            # Verificar alertas CMC Volume (cada 5 minutos)
            print("Verificando alertas CMC Volume...")
            cmc_alerts = indicator.generate_cmc_volume_alerts()
            for alert in cmc_alerts:
                send_telegram_alert(alert, 'cmc_volume')
            
            time.sleep(300)  # Esperar 5 minutos
            
        except Exception as e:
            print(f"Error en background_alert_checker: {e}")
            time.sleep(60)

# Iniciar verificador de alertas
try:
    alert_thread = Thread(target=background_alert_checker, daemon=True)
    alert_thread.start()
    print("Background alert checker iniciado")
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
                if isinstance(signal_data['indicators'][key], np.ndarray):
                    signal_data['indicators'][key] = signal_data['indicators'][key].tolist()
        
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/multiple_signals')
def get_multiple_signals():
    """Endpoint para obtener múltiples señales"""
    try:
        interval = request.args.get('interval', '4h')
        
        long_signals = []
        short_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:8]:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval)
                
                if signal_data['signal'] == 'LONG' and signal_data['signal_score'] >= 65:
                    long_signals.append(signal_data)
                elif signal_data['signal'] == 'SHORT' and signal_data['signal_score'] >= 65:
                    short_signals.append(signal_data)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")
                continue
        
        long_signals.sort(key=lambda x: x['signal_score'], reverse=True)
        short_signals.sort(key=lambda x: x['signal_score'], reverse=True)
        
        return jsonify({
            'long_signals': long_signals[:5],
            'short_signals': short_signals[:5]
        })
        
    except Exception as e:
        print(f"Error en /api/multiple_signals: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/scatter_data_improved')
def get_scatter_data_improved():
    """Endpoint para datos del scatter plot"""
    try:
        interval = request.args.get('interval', '4h')
        
        scatter_data = []
        
        for symbol in CRYPTO_SYMBOLS[:20]:
            try:
                signal_data = indicator.generate_signals_improved(symbol, interval)
                
                if signal_data['current_price'] > 0:
                    # Calcular presiones
                    buy_pressure = min(100, max(0,
                        (signal_data['whale_pump'] / 100 * 25) +
                        (1 if signal_data['plus_di'] > signal_data['minus_di'] else 0) * 20 +
                        (signal_data['rsi_maverick'] * 20) +
                        (1 if signal_data['adx'] > 25 else 0) * 15 +
                        (min(1, signal_data['volume'] / max(1, signal_data['volume_ma'])) * 20)
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (signal_data['whale_dump'] / 100 * 25) +
                        (1 if signal_data['minus_di'] > signal_data['plus_di'] else 0) * 20 +
                        ((1 - signal_data['rsi_maverick']) * 20) +
                        (1 if signal_data['adx'] > 25 else 0) * 15 +
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
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    try:
        alerts = indicator.generate_scalping_alerts()
        return jsonify({'alerts': alerts[:5]})
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/volume_anomaly_signals')
def get_volume_anomaly_signals():
    try:
        alerts = indicator.generate_cmc_volume_alerts()
        return jsonify({'alerts': alerts})
    except Exception as e:
        print(f"Error en /api/volume_anomaly_signals: {e}")
        return jsonify({'alerts': []})

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        signal_data = indicator.generate_signals_improved(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        # Crear figura
        fig = plt.figure(figsize=(14, 16))
        
        # Gráfico 1: Precio con soportes/resistencias
        ax1 = plt.subplot(8, 1, 1)
        if signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data'][-50:]]
            opens = [d['open'] for d in signal_data['data'][-50:]]
            highs = [d['high'] for d in signal_data['data'][-50:]]
            lows = [d['low'] for d in signal_data['data'][-50:]]
            closes = [d['close'] for d in signal_data['data'][-50:]]
            
            # Velas
            for i in range(len(dates)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            # Soportes y resistencias
            for support in signal_data['supports'][:4]:
                ax1.axhline(y=support, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            
            for resistance in signal_data['resistances'][:4]:
                ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='orange', linestyle='-', alpha=0.7, linewidth=2)
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='-', alpha=0.7, linewidth=2)
            for tp in signal_data['take_profit']:
                ax1.axhline(y=tp, color='green', linestyle='-', alpha=0.7, linewidth=1)
        
        ax1.set_title(f'{symbol} - {interval}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: ADX/DMI
        ax2 = plt.subplot(8, 1, 2, sharex=ax1)
        if 'indicators' in signal_data:
            adx_dates = dates[-len(signal_data['indicators']['adx']):]
            ax2.plot(adx_dates, signal_data['indicators']['adx'], 'black', linewidth=2, label='ADX')
            ax2.plot(adx_dates, signal_data['indicators']['plus_di'], 'green', linewidth=1, label='+DI')
            ax2.plot(adx_dates, signal_data['indicators']['minus_di'], 'red', linewidth=1, label='-DI')
        ax2.set_ylabel('ADX/DMI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Volumen
        ax3 = plt.subplot(8, 1, 3, sharex=ax1)
        if 'indicators' in signal_data:
            volume_dates = dates[-len(signal_data['indicators']['volume_anomaly']):]
            volume_values = [d['volume'] for d in signal_data['data'][-len(signal_data['indicators']['volume_anomaly']):]]
            
            for i in range(len(volume_dates)):
                color = 'green' if signal_data['data'][-len(signal_data['indicators']['volume_anomaly'])+i]['close'] >= signal_data['data'][-len(signal_data['indicators']['volume_anomaly'])+i]['open'] else 'red'
                ax3.bar(volume_dates[i], volume_values[i], color=color, alpha=0.7, width=0.8)
        
        ax3.set_ylabel('Volumen')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Fuerza Tendencia Maverick
        ax4 = plt.subplot(8, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates[-len(signal_data['indicators']['trend_strength']):]
            colors = signal_data['indicators']['colors']
            
            for i in range(len(trend_dates)):
                color = colors[i] if i < len(colors) else 'gray'
                ax4.bar(trend_dates[i], signal_data['indicators']['trend_strength'][i], 
                       color=color, alpha=0.7, width=0.8)
        
        ax4.set_ylabel('FT Maverick')
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Ballenas
        ax5 = plt.subplot(8, 1, 5, sharex=ax1)
        if 'indicators' in signal_data:
            whale_dates = dates[-len(signal_data['indicators']['whale_pump']):]
            ax5.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Compra')
            ax5.bar(whale_dates, signal_data['indicators']['whale_dump'], 
                   color='red', alpha=0.7, label='Venta')
        ax5.set_ylabel('Ballenas')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: RSI Maverick
        ax6 = plt.subplot(8, 1, 6, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_dates = dates[-len(signal_data['indicators']['rsi_maverick']):]
            ax6.plot(rsi_dates, signal_data['indicators']['rsi_maverick'], 'blue', linewidth=2)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
        ax6.set_ylabel('RSI Maverick')
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: RSI Tradicional
        ax7 = plt.subplot(8, 1, 7, sharex=ax1)
        if 'indicators' in signal_data:
            rsi_trad_dates = dates[-len(signal_data['indicators']['rsi_traditional']):]
            ax7.plot(rsi_trad_dates, signal_data['indicators']['rsi_traditional'], 'purple', linewidth=2)
            ax7.axhline(y=80, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        ax7.set_ylabel('RSI Tradicional')
        ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: MACD
        ax8 = plt.subplot(8, 1, 8, sharex=ax1)
        if 'indicators' in signal_data:
            macd_dates = dates[-len(signal_data['indicators']['macd']):]
            ax8.plot(macd_dates, signal_data['indicators']['macd'], 'blue', linewidth=1, label='MACD')
            ax8.plot(macd_dates, signal_data['indicators']['macd_signal'], 'red', linewidth=1, label='Señal')
            
            colors_macd = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax8.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors_macd, alpha=0.5, label='Histograma')
        ax8.set_ylabel('MACD')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
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
