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
CMC_CRYPTOS = ["BTC", "ETH", "SOL", "XRP", "ADA"]

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

class CMCVolumeAggregator:
    """Estrategia de Volumen Agregado Total con CoinMarketCap"""
    
    def __init__(self):
        self.api_key = CMC_API_KEY
        self.cache = {}
        self.volume_history = {crypto: [] for crypto in CMC_CRYPTOS}
        self.last_check = {crypto: None for crypto in CMC_CRYPTOS}
        
    def get_cmc_data(self, symbol):
        """Obtener datos de CoinMarketCap"""
        try:
            cache_key = f"cmc_{symbol}"
            now = datetime.now()
            
            # Verificar cache (1 hora)
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (now - timestamp).seconds < 3600:
                    return data
            
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            headers = {
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': self.api_key,
            }
            params = {
                'symbol': symbol,
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and symbol in data['data']:
                    crypto_data = data['data'][symbol]
                    quote = crypto_data['quote']['USD']
                    
                    result = {
                        'symbol': f"{symbol}-USDT",
                        'current_price': quote['price'],
                        'volume_24h': quote['volume_24h'],
                        'percent_change_24h': quote['percent_change_24h'],
                        'timestamp': now
                    }
                    
                    # Actualizar historial de volumen
                    if symbol in self.volume_history:
                        self.volume_history[symbol].append({
                            'timestamp': now,
                            'volume': quote['volume_24h'],
                            'price': quote['price']
                        })
                        
                        # Mantener solo últimas 24 horas
                        cutoff = now - timedelta(hours=24)
                        self.volume_history[symbol] = [
                            entry for entry in self.volume_history[symbol]
                            if entry['timestamp'] > cutoff
                        ]
                    
                    self.cache[cache_key] = (result, now)
                    return result
                    
        except Exception as e:
            print(f"Error obteniendo datos CMC para {symbol}: {e}")
        
        return None
    
    def calculate_volume_sma_24h(self, symbol):
        """Calcular SMA de volumen manualmente (24 horas)"""
        if symbol not in self.volume_history or len(self.volume_history[symbol]) < 2:
            return 0
        
        volumes = [entry['volume'] for entry in self.volume_history[symbol]]
        return np.mean(volumes) if volumes else 0
    
    def check_volume_signal(self, symbol):
        """Verificar señal de volumen agregado"""
        data = self.get_cmc_data(symbol)
        if not data:
            return None
        
        volume_sma = self.calculate_volume_sma_24h(symbol)
        if volume_sma <= 0:
            return None
        
        volume_ratio = data['volume_24h'] / volume_sma
        price_strength = abs(data['percent_change_24h'])
        
        signal = None
        reason = ""
        
        # Condiciones LONG
        if (data['percent_change_24h'] > 0 and 
            volume_ratio >= 2.5 and 
            price_strength < 8):
            signal = 'LONG'
            reason = "Acumulación con alto volumen"
        
        # Condiciones SHORT
        elif (data['percent_change_24h'] < 0 and 
              volume_ratio >= 2.5 and 
              price_strength < 8):
            signal = 'SHORT'
            reason = "Distribución con alto volumen"
        
        if signal:
            return {
                'symbol': data['symbol'],
                'signal': signal,
                'current_price': data['current_price'],
                'volume_ratio': volume_ratio,
                'volume_24h': data['volume_24h'],
                'percent_change_24h': data['percent_change_24h'],
                'price_strength': price_strength,
                'reason': reason,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return None
    
    def generate_volume_chart(self, symbol):
        """Generar gráfico de precio y volumen para Telegram"""
        try:
            if symbol not in self.volume_history or len(self.volume_history[symbol]) < 2:
                return None
            
            history = self.volume_history[symbol]
            
            # Preparar datos
            timestamps = [entry['timestamp'] for entry in history]
            prices = [entry['price'] for entry in history]
            volumes = [entry['volume'] for entry in history]
            
            # Crear figura con doble eje Y
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Precio (eje izquierdo)
            color = 'tab:blue'
            ax1.set_xlabel('Tiempo')
            ax1.set_ylabel('Precio (USD)', color=color)
            ax1.plot(timestamps, prices, color=color, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Volumen (eje derecho)
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Volumen (Millones USD)', color=color)
            ax2.fill_between(timestamps, 0, [v/1_000_000 for v in volumes], 
                            alpha=0.3, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Formatear fechas
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
            
            # Título
            current_data = self.get_cmc_data(symbol)
            if current_data:
                volume_ratio = current_data['volume_24h'] / self.calculate_volume_sma_24h(symbol)
                plt.title(f"{symbol} - Volumen {volume_ratio:.1f}x el promedio", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Guardar imagen
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando gráfico CMC para {symbol}: {e}")
            return None

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.cmc_aggregator = CMCVolumeAggregator()
        self.bolivia_tz = pytz.timezone('America/La_Paz')
    
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def calculate_support_resistance(self, df, num_levels=6, lookback=100):
        """Calcular soportes y resistencias con método de pivotes"""
        try:
            high = df['high'].values[-lookback:]
            low = df['low'].values[-lookback:]
            close = df['close'].values[-lookback:]
            
            # Encontrar pivotes locales
            pivots_high = []
            pivots_low = []
            
            for i in range(2, len(high)-2):
                # Pivote alto
                if (high[i] > high[i-1] and high[i] > high[i-2] and 
                    high[i] > high[i+1] and high[i] > high[i+2]):
                    pivots_high.append(high[i])
                
                # Pivote bajo
                if (low[i] < low[i-1] and low[i] < low[i-2] and 
                    low[i] < low[i+1] and low[i] < low[i+2]):
                    pivots_low.append(low[i])
            
            # Clustering de pivotes
            if pivots_high:
                pivots_high = np.array(pivots_high)
                pivots_low = np.array(pivots_low)
                
                # Usar percentiles para niveles significativos
                resistance_levels = []
                support_levels = []
                
                if len(pivots_high) >= num_levels:
                    percentiles = np.linspace(20, 80, num_levels//2)
                    resistance_levels = [np.percentile(pivots_high, p) for p in percentiles]
                else:
                    resistance_levels = sorted(pivots_high)[-num_levels//2:]
                
                if len(pivots_low) >= num_levels:
                    percentiles = np.linspace(20, 80, num_levels//2)
                    support_levels = [np.percentile(pivots_low, p) for p in percentiles]
                else:
                    support_levels = sorted(pivots_low)[:num_levels//2]
                
                # Combinar y ordenar
                all_levels = sorted(list(set(support_levels + resistance_levels)))
                
                # Asegurar número mínimo de niveles
                if len(all_levels) < 4:
                    # Añadir niveles basados en Fibonacci
                    price_range = high.max() - low.min()
                    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                    for level in fib_levels:
                        all_levels.append(low.min() + price_range * level)
                    all_levels = sorted(list(set(all_levels)))
                
                return all_levels[:num_levels]
            
            return []
            
        except Exception as e:
            print(f"Error calculando soportes/resistencias: {e}")
            return []
    
    def get_kucoin_data(self, symbol, interval, limit=100):
        """Obtener datos de KuCoin con manejo robusto"""
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < 60:
                    return cached_data
            
            interval_map = {
                '15m': '15min', '30m': '30min', '1h': '1hour',
                '2h': '2hour', '4h': '4hour', '8h': '8hour',
                '12h': '12hour', '1D': '1day', '1W': '1week'
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
                    
                    df = df.dropna().tail(limit)
                    self.cache[cache_key] = (df, datetime.now())
                    return df
                
        except Exception as e:
            print(f"Error obteniendo datos de KuCoin para {symbol} {interval}: {e}")
        
        return self.generate_sample_data(limit, interval, symbol)
    
    def calculate_optimal_entry_exit(self, df, signal_type, support_resistance_levels):
        """Calcular entradas óptimas en soportes/resistencias"""
        try:
            current_price = df['close'].iloc[-1]
            high = df['high'].values
            low = df['low'].values
            
            # Encontrar niveles más cercanos
            if signal_type == 'LONG':
                # Para LONG: entrada en el soporte más cercano por debajo
                supports = [level for level in support_resistance_levels if level < current_price]
                if supports:
                    entry = max(supports)  # Soporte más fuerte (más alto)
                else:
                    entry = current_price * 0.995  # Ligero descuento
                
                stop_loss = entry * 0.97  # 3% debajo de entrada
                
                # Take profits en resistencias por encima
                resistances = [level for level in support_resistance_levels if level > entry]
                take_profits = sorted(resistances)[:3]  # 3 primeras resistencias
                if not take_profits:
                    take_profits = [entry * 1.02, entry * 1.04, entry * 1.06]
                    
            else:  # SHORT
                # Para SHORT: entrada en la resistencia más cercana por encima
                resistances = [level for level in support_resistance_levels if level > current_price]
                if resistances:
                    entry = min(resistances)  # Resistencia más débil (más baja)
                else:
                    entry = current_price * 1.005  # Ligero premium
                
                stop_loss = entry * 1.03  # 3% encima de entrada
                
                # Take profits en soportes por debajo
                supports = [level for level in support_resistance_levels if level < entry]
                take_profits = sorted(supports, reverse=True)[:3]  # 3 primeros soportes
                if not take_profits:
                    take_profits = [entry * 0.98, entry * 0.96, entry * 0.94]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits],
                'support_levels': [level for level in support_resistance_levels if level < current_price],
                'resistance_levels': [level for level in support_resistance_levels if level > current_price]
            }
            
        except Exception as e:
            print(f"Error calculando entradas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.97,
                'take_profit': [current_price * 1.02],
                'support_levels': [],
                'resistance_levels': []
            }
    
    def calculate_ema(self, prices, period):
        """Calcular EMA"""
        if len(prices) == 0 or period <= 0:
            return np.zeros_like(prices)
            
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0] if len(prices) > 0 else 0
        
        for i in range(1, len(prices)):
            if np.isnan(prices[i]):
                ema[i] = ema[i-1]
            else:
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
            valid_values = window[~np.isnan(window)]
            sma[i] = np.mean(valid_values) if len(valid_values) > 0 else 0
        
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
            else:
                std[i] = 0
        
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
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
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
            
            # Determinar zonas de no operar
            no_trade_zones = np.zeros(n, dtype=bool)
            for i in range(10, n):
                if (abs(trend_strength[i]) > 20 and 
                    trend_strength[i] * trend_strength[i-1] < 0):
                    no_trade_zones[i] = True
            
            # Señales de fuerza
            strength_signals = ['NEUTRAL'] * n
            for i in range(n):
                if trend_strength[i] > 10:
                    strength_signals[i] = 'STRONG_UP'
                elif trend_strength[i] > 0:
                    strength_signals[i] = 'WEAK_UP'
                elif trend_strength[i] < -10:
                    strength_signals[i] = 'STRONG_DOWN'
                elif trend_strength[i] < 0:
                    strength_signals[i] = 'WEAK_DOWN'
                else:
                    strength_signals[i] = 'NEUTRAL'
            
            return {
                'bb_width': bb_width.tolist(),
                'trend_strength': trend_strength.tolist(),
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
            
            # Verificar TF Menor
            menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
            if menor_df is not None and len(menor_df) > 10:
                menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                
                if signal_type == 'LONG':
                    menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']
                else:
                    menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
                
                menor_no_trade = not menor_trend['no_trade_zones'][-1]
            else:
                menor_ok = False
                menor_no_trade = True
            
            # Verificar TF Media
            media_df = self.get_kucoin_data(symbol, hierarchy['media'], 50)
            if media_df is not None and len(media_df) > 20:
                close_media = media_df['close'].values
                ma_9_media = self.calculate_sma(close_media, 9)
                ma_21_media = self.calculate_sma(close_media, 21)
                
                if signal_type == 'LONG':
                    media_ok = (close_media[-1] > ma_9_media[-1] and 
                               ma_9_media[-1] > ma_21_media[-1])
                else:
                    media_ok = (close_media[-1] < ma_9_media[-1] and 
                               ma_9_media[-1] < ma_21_media[-1])
            else:
                media_ok = False
            
            # Verificar TF Mayor (puede ser neutral)
            mayor_df = self.get_kucoin_data(symbol, hierarchy['mayor'], 50)
            if mayor_df is not None and len(mayor_df) > 20:
                mayor_trend = self.calculate_trend_strength_maverick(mayor_df['close'].values)
                
                if signal_type == 'LONG':
                    mayor_ok = mayor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP', 'NEUTRAL']
                else:
                    mayor_ok = mayor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL']
            else:
                mayor_ok = False
            
            return menor_ok and menor_no_trade and media_ok and mayor_ok
            
        except Exception as e:
            print(f"Error verificando condiciones multi-timeframe: {e}")
            return False
    
    def calculate_whale_signals(self, df, sensitivity=1.5):
        """Indicador de ballenas compradoras/vendedoras"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            n = len(close)
            whale_pump = np.zeros(n)
            whale_dump = np.zeros(n)
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                
                # Señal de compra fuerte (ballenas comprando)
                if volume_ratio > 2.0 and price_change > 0.5:
                    whale_pump[i] = min(100, volume_ratio * 20 * sensitivity)
                
                # Señal de venta fuerte (ballenas vendiendo)
                if volume_ratio > 2.0 and price_change < -0.5:
                    whale_dump[i] = min(100, volume_ratio * 20 * sensitivity)
            
            # Confirmación después de 7 velas
            confirmed_buy = np.zeros(n, dtype=bool)
            confirmed_sell = np.zeros(n, dtype=bool)
            
            for i in range(7, n):
                if (whale_pump[i-7] > 30 and 
                    close[i] > close[i-7] * 1.02):
                    confirmed_buy[i] = True
                
                if (whale_dump[i-7] > 30 and 
                    close[i] < close[i-7] * 0.98):
                    confirmed_sell[i] = True
            
            return {
                'whale_pump': whale_pump.tolist(),
                'whale_dump': whale_dump.tolist(),
                'confirmed_buy': confirmed_buy.tolist(),
                'confirmed_sell': confirmed_sell.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'confirmed_buy': [False] * n,
                'confirmed_sell': [False] * n
            }
    
    def calculate_rsi_maverick(self, close, length=20, bb_multiplier=2.0):
        """RSI Modificado Maverick (%B)"""
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
    
    def detect_divergence(self, price, indicator, lookback=14, confirmation_bars=4):
        """Detectar divergencias con confirmación de 4 velas"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-confirmation_bars):
            # Buscar mínimos en precio
            price_lows = price[i-lookback:i+1]
            indicator_lows = indicator[i-lookback:i+1]
            
            min_price_idx = np.argmin(price_lows)
            min_indicator_idx = np.argmin(indicator_lows)
            
            # Divergencia alcista: precio hace mínimo más bajo, indicador mínimo más alto
            if (min_price_idx > min_indicator_idx and 
                price[i] < price[i-lookback+min_price_idx] and
                indicator[i] > indicator[i-lookback+min_indicator_idx]):
                
                # Confirmar en las próximas 4 velas
                if i + confirmation_bars < n:
                    if all(price[i+j] > price[i] for j in range(1, confirmation_bars+1)):
                        for j in range(confirmation_bars+1):
                            if i+j < n:
                                bullish_div[i+j] = True
            
            # Buscar máximos en precio
            max_price_idx = np.argmax(price_lows)
            max_indicator_idx = np.argmax(indicator_lows)
            
            # Divergencia bajista: precio hace máximo más alto, indicador máximo más bajo
            if (max_price_idx > max_indicator_idx and 
                price[i] > price[i-lookback+max_price_idx] and
                indicator[i] < indicator[i-lookback+max_indicator_idx]):
                
                # Confirmar en las próximas 4 velas
                if i + confirmation_bars < n:
                    if all(price[i+j] < price[i] for j in range(1, confirmation_bars+1)):
                        for j in range(confirmation_bars+1):
                            if i+j < n:
                                bearish_div[i+j] = True
        
        return bullish_div.tolist(), bearish_div.tolist()
    
    def check_ma_crossover(self, ma_9, ma_21, confirmation_bars=1):
        """Detectar cruce de medias con confirmación"""
        n = len(ma_9)
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MA9 cruza por encima de MA21
            if (ma_9[i] > ma_21[i] and ma_9[i-1] <= ma_21[i-1]):
                for j in range(min(confirmation_bars+1, n-i)):
                    ma_cross_bullish[i+j] = True
            
            # Cruce bajista: MA9 cruza por debajo de MA21
            if (ma_9[i] < ma_21[i] and ma_9[i-1] >= ma_21[i-1]):
                for j in range(min(confirmation_bars+1, n-i)):
                    ma_cross_bearish[i+j] = True
        
        return ma_cross_bullish.tolist(), ma_cross_bearish.tolist()
    
    def check_di_crossover(self, plus_di, minus_di, confirmation_bars=1):
        """Detectar cruce de DMI con confirmación"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: +DI cruza por encima de -DI
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]):
                for j in range(min(confirmation_bars+1, n-i)):
                    di_cross_bullish[i+j] = True
            
            # Cruce bajista: -DI cruza por encima de +DI
            if (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]):
                for j in range(min(confirmation_bars+1, n-i)):
                    di_cross_bearish[i+j] = True
        
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
        
        # Pendiente positiva del ADX
        adx_slope_positive = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if adx[i] > adx[i-1] and adx[i] > 25:
                adx_slope_positive[i] = True
        
        return adx, plus_di, minus_di, adx_slope_positive
    
    def check_bollinger_bands(self, close, bb_upper, bb_lower, confirmation_bars=1):
        """Verificar condiciones de Bandas de Bollinger"""
        n = len(close)
        bb_touch_upper = np.zeros(n, dtype=bool)
        bb_touch_lower = np.zeros(n, dtype=bool)
        
        for i in range(n):
            # Precio toca banda superior
            if close[i] >= bb_upper[i] * 0.99:
                for j in range(min(confirmation_bars+1, n-i)):
                    bb_touch_upper[i+j] = True
            
            # Precio toca banda inferior
            if close[i] <= bb_lower[i] * 1.01:
                for j in range(min(confirmation_bars+1, n-i)):
                    bb_touch_lower[i+j] = True
        
        return bb_touch_upper.tolist(), bb_touch_lower.tolist()
    
    def check_macd_crossover(self, macd, macd_signal, confirmation_bars=1):
        """Detectar cruce de MACD con confirmación"""
        n = len(macd)
        macd_cross_bullish = np.zeros(n, dtype=bool)
        macd_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            # Cruce alcista: MACD cruza por encima de señal
            if (macd[i] > macd_signal[i] and macd[i-1] <= macd_signal[i-1]):
                for j in range(min(confirmation_bars+1, n-i)):
                    macd_cross_bullish[i+j] = True
            
            # Cruce bajista: MACD cruza por debajo de señal
            if (macd[i] < macd_signal[i] and macd[i-1] >= macd_signal[i-1]):
                for j in range(min(confirmation_bars+1, n-i)):
                    macd_cross_bearish[i+j] = True
        
        return macd_cross_bullish.tolist(), macd_cross_bearish.tolist()
    
    def check_breakout(self, high, low, close, support_resistance_levels, confirmation_bars=1):
        """Detectar rupturas de soportes/resistencias"""
        n = len(close)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            current_close = close[i]
            
            # Verificar ruptura de resistencia
            for level in support_resistance_levels:
                if (current_close > level and 
                    close[i-1] <= level and 
                    high[i] > level):
                    for j in range(min(confirmation_bars+1, n-i)):
                        breakout_up[i+j] = True
            
            # Verificar ruptura de soporte
            for level in support_resistance_levels:
                if (current_close < level and 
                    close[i-1] >= level and 
                    low[i] < level):
                    for j in range(min(confirmation_bars+1, n-i)):
                        breakout_down[i+j] = True
        
        return breakout_up.tolist(), breakout_down.tolist()
    
    def check_chart_patterns(self, high, low, close, confirmation_bars=7):
        """Detectar patrones chartistas con confirmación"""
        n = len(close)
        patterns = {
            'head_shoulders': np.zeros(n, dtype=bool),
            'double_top': np.zeros(n, dtype=bool),
            'double_bottom': np.zeros(n, dtype=bool),
            'bullish_flag': np.zeros(n, dtype=bool),
            'bearish_flag': np.zeros(n, dtype=bool)
        }
        
        for i in range(20, n-confirmation_bars):
            # Doble techo
            if i >= 10:
                window_high = high[i-10:i+1]
                peaks = []
                for j in range(1, len(window_high)-1):
                    if window_high[j] > window_high[j-1] and window_high[j] > window_high[j+1]:
                        peaks.append(j)
                
                if len(peaks) >= 2:
                    last_two_peaks = peaks[-2:]
                    if (abs(window_high[last_two_peaks[0]] - window_high[last_two_peaks[1]]) / 
                        window_high[last_two_peaks[0]] < 0.02):
                        for j in range(confirmation_bars+1):
                            if i+j < n:
                                patterns['double_top'][i+j] = True
            
            # Doble fondo
            if i >= 10:
                window_low = low[i-10:i+1]
                troughs = []
                for j in range(1, len(window_low)-1):
                    if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                        troughs.append(j)
                
                if len(troughs) >= 2:
                    last_two_troughs = troughs[-2:]
                    if (abs(window_low[last_two_troughs[0]] - window_low[last_two_troughs[1]]) / 
                        window_low[last_two_troughs[0]] < 0.02):
                        for j in range(confirmation_bars+1):
                            if i+j < n:
                                patterns['double_bottom'][i+j] = True
        
        return {k: v.tolist() for k, v in patterns.items()}
    
    def calculate_volume_anomaly(self, volume, close, period=20):
        """Calcular anomalías de volumen (verde=compra, rojo=venta)"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_colors = ['gray'] * n
            
            for i in range(period, n):
                # Media móvil de volumen
                sma_volume = np.mean(volume[max(0, i-period):i])
                
                # Desviación estándar
                std_volume = np.std(volume[max(0, i-period):i]) if i >= period else 0
                
                # Ratio volumen
                volume_ratio = volume[i] / sma_volume if sma_volume > 0 else 1
                
                # Anomalía (volumen > 2σ)
                if volume_ratio > 1 + (2 * std_volume / sma_volume if sma_volume > 0 else 0):
                    volume_anomaly[i] = True
                    
                    # Determinar color basado en cambio de precio
                    if i > 0:
                        price_change = (close[i] - close[i-1]) / close[i-1]
                        if price_change > 0:
                            volume_colors[i] = 'green'  # Compra
                        else:
                            volume_colors[i] = 'red'    # Venta
                
                # Clusters (múltiples anomalías consecutivas)
                if i >= 3 and np.sum(volume_anomaly[i-3:i+1]) >= 3:
                    volume_clusters[i] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_colors': volume_colors,
                'volume_sma': [np.mean(volume[max(0, i-period):i]) if i >= period else volume[i] for i in range(n)]
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly: {e}")
            n = len(volume)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_colors': ['gray'] * n,
                'volume_sma': [0] * n
            }
    
    def evaluate_signal_conditions(self, data, current_idx, interval):
        """Evaluar condiciones de señal con pesos corregidos"""
        conditions = {
            'long': {},
            'short': {}
        }
        
        # Definir pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            weights = {
                'multi_timeframe': 30,
                'trend_strength': 25,
                'whale_signal': 0,
                'ma_cross': 10,
                'di_cross': 10,
                'adx_slope': 5,
                'bollinger_bands': 8,
                'macd_cross': 10,
                'volume_anomaly': 7,
                'rsi_maverick_div': 8,
                'rsi_traditional_div': 5,
                'chart_pattern': 5,
                'breakout': 5
            }
        elif interval in ['12h', '1D']:
            weights = {
                'multi_timeframe': 0,
                'trend_strength': 25,
                'whale_signal': 30,
                'ma_cross': 10,
                'di_cross': 10,
                'adx_slope': 5,
                'bollinger_bands': 8,
                'macd_cross': 10,
                'volume_anomaly': 7,
                'rsi_maverick_div': 8,
                'rsi_traditional_div': 5,
                'chart_pattern': 5,
                'breakout': 5
            }
        else:  # 1W
            weights = {
                'multi_timeframe': 0,
                'trend_strength': 55,
                'whale_signal': 0,
                'ma_cross': 10,
                'di_cross': 10,
                'adx_slope': 5,
                'bollinger_bands': 8,
                'macd_cross': 10,
                'volume_anomaly': 7,
                'rsi_maverick_div': 8,
                'rsi_traditional_div': 5,
                'chart_pattern': 5,
                'breakout': 5
            }
        
        # Inicializar condiciones
        for signal_type in ['long', 'short']:
            for key, weight in weights.items():
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
        
        conditions['long']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_UP', 'WEAK_UP'] and
            not data['no_trade_zones'][current_idx]
        )
        
        if interval in ['12h', '1D']:
            conditions['long']['whale_signal']['value'] = (
                data['confirmed_buy'][current_idx] and
                data['whale_pump'][current_idx] > 30
            )
        
        conditions['long']['ma_cross']['value'] = data['ma_cross_bullish'][current_idx]
        conditions['long']['di_cross']['value'] = data['di_cross_bullish'][current_idx]
        conditions['long']['adx_slope']['value'] = data['adx_slope_positive'][current_idx]
        conditions['long']['bollinger_bands']['value'] = data['bb_touch_lower'][current_idx]
        conditions['long']['macd_cross']['value'] = data['macd_cross_bullish'][current_idx]
        conditions['long']['volume_anomaly']['value'] = (
            data['volume_anomaly'][current_idx] and 
            data['volume_colors'][current_idx] == 'green'
        )
        conditions['long']['rsi_maverick_div']['value'] = data['rsi_maverick_bullish_divergence'][current_idx]
        conditions['long']['rsi_traditional_div']['value'] = data['rsi_bullish_divergence'][current_idx]
        conditions['long']['chart_pattern']['value'] = (
            data['chart_patterns']['double_bottom'][current_idx] or
            data['chart_patterns']['bullish_flag'][current_idx]
        )
        conditions['long']['breakout']['value'] = data['breakout_up'][current_idx]
        
        # Condiciones SHORT
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            conditions['short']['multi_timeframe']['value'] = data.get('multi_timeframe_short', False)
        
        conditions['short']['trend_strength']['value'] = (
            data['trend_strength_signals'][current_idx] in ['STRONG_DOWN', 'WEAK_DOWN'] and
            not data['no_trade_zones'][current_idx]
        )
        
        if interval in ['12h', '1D']:
            conditions['short']['whale_signal']['value'] = (
                data['confirmed_sell'][current_idx] and
                data['whale_dump'][current_idx] > 30
            )
        
        conditions['short']['ma_cross']['value'] = data['ma_cross_bearish'][current_idx]
        conditions['short']['di_cross']['value'] = data['di_cross_bearish'][current_idx]
        conditions['short']['adx_slope']['value'] = data['adx_slope_positive'][current_idx]
        conditions['short']['bollinger_bands']['value'] = data['bb_touch_upper'][current_idx]
        conditions['short']['macd_cross']['value'] = data['macd_cross_bearish'][current_idx]
        conditions['short']['volume_anomaly']['value'] = (
            data['volume_anomaly'][current_idx] and 
            data['volume_colors'][current_idx] == 'red'
        )
        conditions['short']['rsi_maverick_div']['value'] = data['rsi_maverick_bearish_divergence'][current_idx]
        conditions['short']['rsi_traditional_div']['value'] = data['rsi_bearish_divergence'][current_idx]
        conditions['short']['chart_pattern']['value'] = (
            data['chart_patterns']['double_top'][current_idx] or
            data['chart_patterns']['head_shoulders'][current_idx] or
            data['chart_patterns']['bearish_flag'][current_idx]
        )
        conditions['short']['breakout']['value'] = data['breakout_down'][current_idx]
        
        return conditions
    
    def get_condition_description(self, condition_key):
        """Obtener descripción de condición"""
        descriptions = {
            'multi_timeframe': 'Multi-Timeframe confirmado',
            'trend_strength': 'Fuerza de tendencia favorable',
            'whale_signal': 'Señal ballenas confirmada',
            'ma_cross': 'Cruce de medias MA9/MA21',
            'di_cross': 'Cruce DMI (+DI/-DI)',
            'adx_slope': 'ADX con pendiente positiva >25',
            'bollinger_bands': 'Bandas de Bollinger',
            'macd_cross': 'Cruce MACD',
            'volume_anomaly': 'Anomalía de volumen',
            'rsi_maverick_div': 'Divergencia RSI Maverick',
            'rsi_traditional_div': 'Divergencia RSI Tradicional',
            'chart_pattern': 'Patrón chartista',
            'breakout': 'Ruptura de S/R'
        }
        return descriptions.get(condition_key, condition_key)
    
    def calculate_signal_score(self, conditions, signal_type, ma200_condition):
        """Calcular puntuación de señal"""
        total_weight = 0
        achieved_weight = 0
        fulfilled_conditions = []
        
        signal_conditions = conditions.get(signal_type, {})
        
        # Verificar condiciones obligatorias
        obligatory_met = True
        for key, condition in signal_conditions.items():
            if condition['weight'] >= 25:  # Condiciones obligatorias
                if not condition['value']:
                    obligatory_met = False
                else:
                    fulfilled_conditions.append(condition['description'])
        
        if not obligatory_met:
            return 0, []
        
        # Calcular score total
        for key, condition in signal_conditions.items():
            total_weight += condition['weight']
            if condition['value']:
                achieved_weight += condition['weight']
                if condition['weight'] < 25:  # Solo añadir complementarias
                    fulfilled_conditions.append(condition['description'])
        
        if total_weight == 0:
            return 0, []
        
        base_score = (achieved_weight / total_weight * 100)
        
        # Ajustar score mínimo según MA200
        if signal_type == 'long':
            min_score = 65 if ma200_condition == 'above' else 70
        else:
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0
        
        return min(final_score, 100), fulfilled_conditions
    
    def generate_signals(self, symbol, interval, di_period=14, adx_threshold=25, 
                        sr_period=50, rsi_length=14, bb_multiplier=2.0, leverage=15):
        """Generar señales de trading mejoradas"""
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular indicadores
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Soporte y resistencia
            sr_levels = self.calculate_support_resistance(df, num_levels=6)
            
            # Indicadores básicos
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            ma_cross_bullish, ma_cross_bearish = self.check_ma_crossover(ma_9, ma_21)
            
            # ADX y DMI
            adx, plus_di, minus_di, adx_slope_positive = self.calculate_adx(high, low, close, di_period)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, bb_multiplier)
            bb_touch_upper, bb_touch_lower = self.check_bollinger_bands(close, bb_upper, bb_lower)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            macd_cross_bullish, macd_cross_bearish = self.check_macd_crossover(macd, macd_signal)
            
            # RSI
            rsi_traditional = self.calculate_rsi(close, rsi_length)
            rsi_maverick = self.calculate_rsi_maverick(close, 20, bb_multiplier)
            
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            
            # Volumen
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            # Ballenas
            whale_data = self.calculate_whale_signals(df)
            
            # Fuerza de tendencia
            trend_data = self.calculate_trend_strength_maverick(close)
            
            # Patrones chartistas
            chart_patterns = self.check_chart_patterns(high, low, close)
            
            # Breakouts
            breakout_up, breakout_down = self.check_breakout(high, low, close, sr_levels)
            
            # Multi-timeframe
            multi_timeframe_long = self.check_multi_timeframe_obligatory(symbol, interval, 'LONG')
            multi_timeframe_short = self.check_multi_timeframe_obligatory(symbol, interval, 'SHORT')
            
            # Preparar datos para análisis
            analysis_data = {
                'close': close,
                'ma_cross_bullish': ma_cross_bullish,
                'ma_cross_bearish': ma_cross_bearish,
                'di_cross_bullish': di_cross_bullish,
                'di_cross_bearish': di_cross_bearish,
                'adx_slope_positive': adx_slope_positive.tolist(),
                'bb_touch_upper': bb_touch_upper,
                'bb_touch_lower': bb_touch_lower,
                'macd_cross_bullish': macd_cross_bullish,
                'macd_cross_bearish': macd_cross_bearish,
                'volume_anomaly': volume_data['volume_anomaly'],
                'volume_colors': volume_data['volume_colors'],
                'rsi_maverick_bullish_divergence': rsi_maverick_bullish,
                'rsi_maverick_bearish_divergence': rsi_maverick_bearish,
                'rsi_bullish_divergence': rsi_bullish,
                'rsi_bearish_divergence': rsi_bearish,
                'trend_strength_signals': trend_data['strength_signals'],
                'no_trade_zones': trend_data['no_trade_zones'],
                'whale_pump': whale_data['whale_pump'],
                'whale_dump': whale_data['whale_dump'],
                'confirmed_buy': whale_data['confirmed_buy'],
                'confirmed_sell': whale_data['confirmed_sell'],
                'chart_patterns': chart_patterns,
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'multi_timeframe_long': multi_timeframe_long,
                'multi_timeframe_short': multi_timeframe_short
            }
            
            current_idx = -1
            current_price = close[current_idx]
            
            # Condición MA200
            current_ma200 = ma_200[current_idx] if current_idx < len(ma_200) else 0
            ma200_condition = 'above' if current_price > current_ma200 else 'below'
            
            # Evaluar condiciones
            conditions = self.evaluate_signal_conditions(analysis_data, current_idx, interval)
            
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
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, sr_levels)
            
            # Datos para gráficos
            chart_data = df.tail(50).to_dict('records')
            
            # Preparar indicadores para gráficos
            indicators = {
                'ma_9': ma_9[-50:].tolist(),
                'ma_21': ma_21[-50:].tolist(),
                'ma_50': ma_50[-50:].tolist(),
                'ma_200': ma_200[-50:].tolist(),
                'adx': adx[-50:].tolist(),
                'plus_di': plus_di[-50:].tolist(),
                'minus_di': minus_di[-50:].tolist(),
                'di_cross_bullish': di_cross_bullish[-50:],
                'di_cross_bearish': di_cross_bearish[-50:],
                'bb_upper': bb_upper[-50:].tolist(),
                'bb_middle': bb_middle[-50:].tolist(),
                'bb_lower': bb_lower[-50:].tolist(),
                'macd': macd[-50:].tolist(),
                'macd_signal': macd_signal[-50:].tolist(),
                'macd_histogram': macd_histogram[-50:].tolist(),
                'rsi_traditional': rsi_traditional[-50:],
                'rsi_maverick': rsi_maverick[-50:],
                'volume_anomaly': volume_data['volume_anomaly'][-50:],
                'volume_colors': volume_data['volume_colors'][-50:],
                'volume_sma': volume_data['volume_sma'][-50:],
                'whale_pump': whale_data['whale_pump'][-50:],
                'whale_dump': whale_data['whale_dump'][-50:],
                'trend_strength': trend_data['trend_strength'][-50:],
                'trend_colors': trend_data['colors'][-50:],
                'no_trade_zones': trend_data['no_trade_zones'][-50:],
                'support_levels': levels_data['support_levels'],
                'resistance_levels': levels_data['resistance_levels']
            }
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'signal': signal_type,
                'signal_score': float(signal_score),
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'fulfilled_conditions': fulfilled_conditions,
                'multi_timeframe_ok': multi_timeframe_long if signal_type == 'LONG' else multi_timeframe_short,
                'ma200_condition': ma200_condition,
                'data': chart_data,
                'indicators': indicators,
                'support_levels': levels_data['support_levels'],
                'resistance_levels': levels_data['resistance_levels']
            }
            
        except Exception as e:
            print(f"Error generando señales para {symbol}: {e}")
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
            'fulfilled_conditions': [],
            'multi_timeframe_ok': False,
            'ma200_condition': 'below',
            'data': [],
            'indicators': {},
            'support_levels': [],
            'resistance_levels': []
        }
    
    def generate_scalping_alerts(self):
        """Generar alertas de trading"""
        alerts = []
        
        for interval in ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']:
            for symbol in CRYPTO_SYMBOLS[:12]:
                try:
                    signal_data = self.generate_signals(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
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
                            'take_profit': signal_data['take_profit'],
                            'fulfilled_conditions': signal_data['fulfilled_conditions'],
                            'risk_category': risk_category,
                            'current_price': signal_data['current_price'],
                            'ma200_condition': signal_data['ma200_condition'],
                            'multi_timeframe_ok': signal_data['multi_timeframe_ok']
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
    
    def generate_telegram_chart_image(self, signal_data):
        """Generar imagen para Telegram (solo gráficos)"""
        try:
            if not signal_data or not signal_data['data']:
                return None
            
            df_data = signal_data['data']
            indicators = signal_data['indicators']
            
            # Crear figura con múltiples subplots
            fig = plt.figure(figsize=(12, 16))
            
            # 1. Gráfico de velas
            ax1 = plt.subplot(8, 1, 1)
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') 
                    if isinstance(d['timestamp'], str) else d['timestamp'] 
                    for d in df_data]
            
            # Velas japonesas
            for i in range(len(dates)):
                color = 'green' if df_data[i]['close'] >= df_data[i]['open'] else 'red'
                ax1.plot([dates[i], dates[i]], [df_data[i]['low'], df_data[i]['high']], 
                        color='black', linewidth=1)
                ax1.plot([dates[i], dates[i]], [df_data[i]['open'], df_data[i]['close']], 
                        color=color, linewidth=3)
            
            # Bandas de Bollinger (transparentes)
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                ax1.fill_between(dates[-len(indicators['bb_upper']):], 
                                indicators['bb_upper'], indicators['bb_lower'], 
                                alpha=0.1, color='orange')
            
            # Medias móviles
            if 'ma_9' in indicators:
                ax1.plot(dates[-len(indicators['ma_9']):], indicators['ma_9'], 
                        'orange', linewidth=1, alpha=0.7)
            if 'ma_21' in indicators:
                ax1.plot(dates[-len(indicators['ma_21']):], indicators['ma_21'], 
                        'blue', linewidth=1, alpha=0.7)
            if 'ma_200' in indicators:
                ax1.plot(dates[-len(indicators['ma_200']):], indicators['ma_200'], 
                        'purple', linewidth=2, alpha=0.7)
            
            ax1.set_title(f"{signal_data['symbol']} - {signal_data['signal']} ({signal_data['signal_score']:.1f}%)", 
                         fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 2. ADX con DMI
            ax2 = plt.subplot(8, 1, 2)
            if 'adx' in indicators and 'plus_di' in indicators and 'minus_di' in indicators:
                adx_dates = dates[-len(indicators['adx']):]
                ax2.plot(adx_dates, indicators['adx'], 'black', linewidth=2, label='ADX')
                ax2.plot(adx_dates, indicators['plus_di'], 'green', linewidth=1, label='+DI')
                ax2.plot(adx_dates, indicators['minus_di'], 'red', linewidth=1, label='-DI')
                ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Volumen con anomalías
            ax3 = plt.subplot(8, 1, 3)
            if 'volume_colors' in indicators:
                volume_dates = dates[-len(indicators['volume_colors']):]
                volumes = [d['volume'] for d in df_data[-len(indicators['volume_colors']):]]
                
                for i, (date, vol, color) in enumerate(zip(volume_dates, volumes, indicators['volume_colors'])):
                    ax3.bar(date, vol, color=color, alpha=0.7, width=0.8)
            
            if 'volume_sma' in indicators:
                ax3.plot(volume_dates, indicators['volume_sma'], 'yellow', linewidth=1)
            
            ax3.grid(True, alpha=0.3)
            
            # 4. Fuerza de Tendencia Maverick (barras)
            ax4 = plt.subplot(8, 1, 4)
            if 'trend_strength' in indicators and 'trend_colors' in indicators:
                trend_dates = dates[-len(indicators['trend_strength']):]
                for i, (date, strength, color) in enumerate(zip(trend_dates, 
                                                              indicators['trend_strength'], 
                                                              indicators['trend_colors'])):
                    ax4.bar(date, strength, color=color, alpha=0.7, width=0.8)
            ax4.grid(True, alpha=0.3)
            
            # 5. Ballenas (barras) - solo para 12h y 1D
            ax5 = plt.subplot(8, 1, 5)
            if 'whale_pump' in indicators and 'whale_dump' in indicators:
                whale_dates = dates[-len(indicators['whale_pump']):]
                ax5.bar(whale_dates, indicators['whale_pump'], color='green', alpha=0.6, label='Compra')
                ax5.bar(whale_dates, [-x for x in indicators['whale_dump']], color='red', alpha=0.6, label='Venta')
                ax5.legend(loc='upper right', fontsize=8)
            ax5.grid(True, alpha=0.3)
            
            # 6. RSI Maverick
            ax6 = plt.subplot(8, 1, 6)
            if 'rsi_maverick' in indicators:
                rsi_dates = dates[-len(indicators['rsi_maverick']):]
                ax6.plot(rsi_dates, indicators['rsi_maverick'], 'blue', linewidth=2)
                ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
                ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
                ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax6.grid(True, alpha=0.3)
            
            # 7. RSI Tradicional
            ax7 = plt.subplot(8, 1, 7)
            if 'rsi_traditional' in indicators:
                rsi_trad_dates = dates[-len(indicators['rsi_traditional']):]
                ax7.plot(rsi_trad_dates, indicators['rsi_traditional'], 'cyan', linewidth=2)
                ax7.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                ax7.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax7.grid(True, alpha=0.3)
            
            # 8. MACD (barras)
            ax8 = plt.subplot(8, 1, 8)
            if 'macd_histogram' in indicators:
                macd_dates = dates[-len(indicators['macd_histogram']):]
                colors = ['green' if x > 0 else 'red' for x in indicators['macd_histogram']]
                ax8.bar(macd_dates, indicators['macd_histogram'], color=colors, alpha=0.7)
                ax8.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax8.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar imagen
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error generando gráfico para Telegram: {e}")
            return None

# Instancia global
indicator = TradingIndicator()

def send_telegram_alert(alert_data, alert_type='multiframe'):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if alert_type == 'multiframe':
            # Mensaje para estrategia Multi-Timeframe
            conditions_text = "\n".join([f"• {cond}" for cond in alert_data.get('fulfilled_conditions', [])])
            
            message = f"""
🚨 WGTA PRO | {alert_data['signal']} | {alert_data['symbol']}
📊 Score: {alert_data['score']:.1f}%

💰 Precio: ${alert_data['current_price']:.6f}
🎯 Entrada: ${alert_data['entry']:.6f}
🛑 Stop Loss: ${alert_data['stop_loss']:.6f}

✅ Condiciones cumplidas:
{conditions_text}
            """
            
            # Generar imagen
            signal_data = indicator.generate_signals(alert_data['symbol'], alert_data['interval'])
            img_buffer = indicator.generate_telegram_chart_image(signal_data)
            
            if img_buffer:
                asyncio.run(bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=img_buffer,
                    caption=message
                ))
            else:
                asyncio.run(bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message
                ))
                
        elif alert_type == 'volume':
            # Mensaje para estrategia de Volumen Agregado
            message = f"""
🚨 VOL AGREGADO | {alert_data['signal']} | {alert_data['symbol']}
💰 Entrada: ${alert_data['current_price']:.2f}
📈 Volumen: {alert_data['volume_ratio']:.1f}x el promedio
📊 Razón: {alert_data['reason']}
            """
            
            # Generar gráfico CMC
            symbol_cmc = alert_data['symbol'].replace('-USDT', '')
            img_buffer = indicator.cmc_aggregator.generate_volume_chart(symbol_cmc)
            
            if img_buffer:
                asyncio.run(bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=img_buffer,
                    caption=message
                ))
            else:
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
            # Verificar estrategia Multi-Timeframe cada 5 minutos
            alerts = indicator.generate_scalping_alerts()
            for alert in alerts:
                send_telegram_alert(alert, 'multiframe')
            
            # Verificar estrategia Volumen Agregado cada hora
            current_time = datetime.now()
            if current_time.minute == 0:  # Cada hora en punto
                for symbol in CMC_CRYPTOS:
                    volume_signal = indicator.cmc_aggregator.check_volume_signal(symbol)
                    if volume_signal:
                        send_telegram_alert(volume_signal, 'volume')
            
            time.sleep(300)  # 5 minutos
            
        except Exception as e:
            print(f"Error en background_alert_checker: {e}")
            time.sleep(60)

# Iniciar thread de alertas
try:
    alert_thread = Thread(target=background_alert_checker, daemon=True)
    alert_thread.start()
    print("Background alert checker iniciado")
except Exception as e:
    print(f"Error iniciando alert checker: {e}")

# Rutas Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint para señales de trading"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        di_period = int(request.args.get('di_period', 14))
        adx_threshold = int(request.args.get('adx_threshold', 25))
        sr_period = int(request.args.get('sr_period', 50))
        rsi_length = int(request.args.get('rsi_length', 14))
        bb_multiplier = float(request.args.get('bb_multiplier', 2.0))
        leverage = int(request.args.get('leverage', 15))
        
        signal_data = indicator.generate_signals(
            symbol, interval, di_period, adx_threshold, sr_period, 
            rsi_length, bb_multiplier, leverage
        )
        
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/multiple_signals')
def get_multiple_signals():
    """Endpoint para múltiples señales"""
    try:
        interval = request.args.get('interval', '4h')
        
        all_signals = []
        
        for symbol in CRYPTO_SYMBOLS[:10]:
            try:
                signal_data = indicator.generate_signals(symbol, interval)
                
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
            'short_signals': short_signals[:5]
        })
        
    except Exception as e:
        print(f"Error en /api/multiple_signals: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/scatter_data')
def get_scatter_data():
    """Endpoint para datos del scatter plot"""
    try:
        interval = request.args.get('interval', '4h')
        
        scatter_data = []
        
        symbols_to_analyze = []
        for category in ['bajo', 'medio', 'alto', 'memecoins']:
            symbols_to_analyze.extend(CRYPTO_RISK_CLASSIFICATION[category][:5])
        
        for symbol in symbols_to_analyze:
            try:
                signal_data = indicator.generate_signals(symbol, interval)
                if signal_data and signal_data['current_price'] > 0:
                    
                    # Calcular presiones
                    indicators = signal_data.get('indicators', {})
                    
                    buy_pressure = min(100, max(0,
                        (signal_data.get('signal_score', 0) / 100 * 40) +
                        (1 if indicators.get('volume_colors', ['gray'])[-1] == 'green' else 0) * 30 +
                        (1 if signal_data.get('ma200_condition') == 'above' else 0) * 30
                    ))
                    
                    sell_pressure = min(100, max(0,
                        (signal_data.get('signal_score', 0) / 100 * 40) +
                        (1 if indicators.get('volume_colors', ['gray'])[-1] == 'red' else 0) * 30 +
                        (1 if signal_data.get('ma200_condition') == 'below' else 0) * 30
                    ))
                    
                    scatter_data.append({
                        'symbol': symbol,
                        'x': float(buy_pressure),
                        'y': float(sell_pressure),
                        'signal_score': float(signal_data.get('signal_score', 0)),
                        'current_price': float(signal_data.get('current_price', 0)),
                        'signal': signal_data.get('signal', 'NEUTRAL'),
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
        print(f"Error en /api/scatter_data: {e}")
        return jsonify([])

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    """Endpoint para clasificación de riesgo"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/scalping_alerts')
def get_scalping_alerts():
    """Endpoint para alertas de trading"""
    try:
        alerts = indicator.generate_scalping_alerts()
        return jsonify({'alerts': alerts[:5]})  # Limitar a 5 alertas
        
    except Exception as e:
        print(f"Error en /api/scalping_alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/api/cmc_volume_signals')
def get_cmc_volume_signals():
    """Endpoint para señales de volumen agregado"""
    try:
        signals = []
        for symbol in CMC_CRYPTOS:
            signal = indicator.cmc_aggregator.check_volume_signal(symbol)
            if signal:
                signals.append(signal)
        
        return jsonify({'signals': signals})
        
    except Exception as e:
        print(f"Error en /api/cmc_volume_signals: {e}")
        return jsonify({'signals': []})

@app.route('/api/bolivia_time')
def get_bolivia_time():
    """Endpoint para hora de Bolivia"""
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d')
    })

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        signal_data = indicator.generate_signals(symbol, interval)
        
        if not signal_data or not signal_data['data']:
            return jsonify({'error': 'No hay datos para el reporte'}), 400
        
        # Generar imagen (similar a Telegram pero con más datos)
        img_buffer = indicator.generate_telegram_chart_image(signal_data)
        
        if img_buffer:
            return send_file(
                img_buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name=f"report_{symbol}_{interval}.png"
            )
        else:
            return jsonify({'error': 'Error generando imagen'}), 500
            
    except Exception as e:
        print(f"Error generando reporte: {e}")
        return jsonify({'error': 'Error generando reporte'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
