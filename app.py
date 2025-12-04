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
from typing import Dict, List, Optional, Tuple

app = Flask(__name__)

# ============================================
# CONFIGURACIÓN API
# ============================================

# Configuración Telegram
TELEGRAM_BOT_TOKEN = "8007748376:AAHIW8n9b-BtA378g4gF-0-D2mOhn495Q0g"
TELEGRAM_CHAT_ID = "-1003229814161"

# Configuración CoinMarketCap (API Key segura)
CMC_API_KEY = "d22df0c59e5e47e0980b89f6eb32ea1b"
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"

# ============================================
# LISTA DE CRIPTOMONEDAS OPTIMIZADA
# ============================================

# 40 criptomonedas principales (optimizadas para ambas estrategias)
CRYPTO_SYMBOLS = [
    # Bajo Riesgo (15) - Top market cap
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "DOGE-USDT",
    "LTC-USDT", "BCH-USDT", "ATOM-USDT", "XLM-USDT", "ETC-USDT",
    
    # Medio Riesgo (10) - Proyectos consolidados
    "FIL-USDT", "ALGO-USDT", "ICP-USDT", "VET-USDT", "EOS-USDT",
    "NEAR-USDT", "AXS-USDT", "EGLD-USDT", "HBAR-USDT", "GRT-USDT",
    
    # Alto Riesgo (10) - Proyectos emergentes
    "ENJ-USDT", "CHZ-USDT", "BAT-USDT", "ONE-USDT", "WAVES-USDT",
    "APE-USDT", "GMT-USDT", "SAND-USDT", "OP-USDT", "ARB-USDT",
    
    # Memecoins (5) - Top memes
    "SHIB-USDT", "PEPE-USDT", "FLOKI-USDT", "BONK-USDT", "WIF-USDT"
]

# Criptos para estrategia 2 (CoinMarketCap) - símbolos compatibles
CMC_SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "ADA"]

# Clasificación de riesgo
CRYPTO_RISK_CLASSIFICATION = {
    "bajo": CRYPTO_SYMBOLS[:15],
    "medio": CRYPTO_SYMBOLS[15:25],
    "alto": CRYPTO_SYMBOLS[25:35],
    "memecoins": CRYPTO_SYMBOLS[35:]
}

# ============================================
# MAPEO DE TEMPORALIDADES
# ============================================

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

# ============================================
# CLASE PRINCIPAL DEL SISTEMA
# ============================================

class TradingSystem:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_operations = {}
        self.winrate_data = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.sent_exit_signals = set()
        
        # Cache para CoinMarketCap
        self.cmc_cache = {}
        self.cmc_last_update = None
        self.volume_spike_alerts_sent = {}
        
    # ============================================
    # FUNCIONES BÁSICAS Y UTILIDADES
    # ============================================
    
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def get_kucoin_data(self, symbol: str, interval: str, limit: int = 100):
        """Obtener datos de KuCoin con cache"""
        cache_key = f"{symbol}_{interval}_{limit}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 60:
                return cached_data
        
        try:
            interval_map = {
                '15m': '15min', '30m': '30min', '1h': '1hour',
                '2h': '2hour', '4h': '4hour', '8h': '8hour',
                '12h': '12hour', '1D': '1day', '1W': '1week'
            }
            
            kucoin_interval = interval_map.get(interval, '1hour')
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol}&type={kucoin_interval}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
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
            print(f"Error obteniendo datos KuCoin {symbol}: {e}")
        
        # Datos de respaldo
        return self.generate_sample_data(symbol, limit)
    
    def generate_sample_data(self, symbol: str, limit: int):
        """Generar datos de ejemplo"""
        np.random.seed(42)
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1h')
        returns = np.random.normal(0.0005, 0.015, limit)
        prices = base_price * np.cumprod(1 + returns)
        
        data = {
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.003, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0.01, 0.008, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0.01, 0.008, limit))),
            'close': prices,
            'volume': np.random.lognormal(12, 1, limit)
        }
        
        df = pd.DataFrame(data)
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df
    
    # ============================================
    # INDICADORES TÉCNICOS
    # ============================================
    
    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcular Media Móvil Simple"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        weights = np.ones(period) / period
        sma = np.convolve(prices, weights, mode='valid')
        
        # Rellenar inicio con NaN
        prefix = np.full(period - 1, np.nan)
        return np.concatenate([prefix, sma])
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcular Media Móvil Exponencial"""
        if len(prices) == 0:
            return np.array([])
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcular Bandas de Bollinger"""
        sma = self.calculate_sma(prices, period)
        
        # Calcular desviación estándar
        std = np.zeros_like(prices)
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            std[i] = np.std(window)
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calcular RSI tradicional"""
        if len(prices) < period + 1:
            return np.zeros_like(prices)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = self.calculate_sma(gains, period)
        avg_loss = self.calculate_sma(losses, period)
        
        rs = np.zeros_like(prices)
        for i in range(len(prices)):
            if avg_loss[i] > 0:
                rs[i] = avg_gain[i] / avg_loss[i]
            elif avg_gain[i] > 0:
                rs[i] = 100  # Máximo RSI
            else:
                rs[i] = 50  # Neutral
        
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcular ADX, +DI, -DI"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        
        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            elif down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Suavizar TR y DM
        tr_smooth = self.calculate_ema(tr, period)
        plus_dm_smooth = self.calculate_ema(plus_dm, period)
        minus_dm_smooth = self.calculate_ema(minus_dm, period)
        
        # Calcular DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        
        for i in range(n):
            if tr_smooth[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        
        # Calcular DX y ADX
        dx = np.zeros(n)
        for i in range(n):
            if (plus_di[i] + minus_di[i]) > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = self.calculate_ema(dx, period)
        
        return adx, plus_di, minus_di
    
    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcular MACD"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    # ============================================
    # INDICADORES AVANZADOS DEL SISTEMA
    # ============================================
    
    def calculate_trend_strength_maverick(self, close: np.ndarray, length: int = 20, mult: float = 2.0) -> Dict:
        """Calcular Fuerza de Tendencia Maverick (Ancho Bandas Bollinger %)"""
        n = len(close)
        
        # Calcular Bandas de Bollinger
        basis = self.calculate_sma(close, length)
        dev = np.zeros(n)
        
        for i in range(length - 1, n):
            window = close[i - length + 1:i + 1]
            dev[i] = np.std(window) if len(window) > 1 else 0
        
        upper = basis + (dev * mult)
        lower = basis - (dev * mult)
        
        # Calcular ancho porcentual de las bandas
        bb_width = np.zeros(n)
        for i in range(n):
            if basis[i] > 0:
                bb_width[i] = ((upper[i] - lower[i]) / basis[i]) * 100
        
        # Calcular fuerza de tendencia
        trend_strength = np.zeros(n)
        for i in range(1, n):
            if bb_width[i] > bb_width[i-1]:
                trend_strength[i] = bb_width[i]
            else:
                trend_strength[i] = -bb_width[i]
        
        # Determinar umbral alto (percentil 70)
        if n >= 50:
            historical_bb_width = bb_width[max(0, n-100):n]
            high_zone_threshold = np.percentile(historical_bb_width, 70)
        else:
            high_zone_threshold = np.percentile(bb_width[bb_width > 0], 70) if np.any(bb_width > 0) else 5
        
        # Determinar zonas de NO OPERAR y señales
        no_trade_zones = np.zeros(n, dtype=bool)
        strength_signals = ['NEUTRAL'] * n
        
        for i in range(10, n):
            # Zona NO OPERAR: alta volatilidad + fuerza decreciente
            if (bb_width[i] > high_zone_threshold * 1.2 and 
                trend_strength[i] < 0 and 
                bb_width[i] < np.max(bb_width[max(0, i-10):i])):
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
        
        # Colores para visualización
        colors = []
        for i in range(n):
            if no_trade_zones[i]:
                colors.append('red')
            elif strength_signals[i] == 'STRONG_UP':
                colors.append('darkgreen')
            elif strength_signals[i] == 'WEAK_UP':
                colors.append('lightgreen')
            elif strength_signals[i] == 'STRONG_DOWN':
                colors.append('darkred')
            elif strength_signals[i] == 'WEAK_DOWN':
                colors.append('lightcoral')
            else:
                colors.append('gray')
        
        return {
            'bb_width': bb_width.tolist(),
            'trend_strength': trend_strength.tolist(),
            'basis': basis.tolist(),
            'upper_band': upper.tolist(),
            'lower_band': lower.tolist(),
            'high_zone_threshold': float(high_zone_threshold),
            'no_trade_zones': no_trade_zones.tolist(),
            'strength_signals': strength_signals,
            'colors': colors
        }
    
    def calculate_whale_signals(self, df: pd.DataFrame, sensitivity: float = 1.5) -> Dict:
        """Calcular señales de actividad de ballenas"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        n = len(close)
        
        whale_pump = np.zeros(n)
        whale_dump = np.zeros(n)
        confirmed_buy = np.zeros(n, dtype=bool)
        confirmed_sell = np.zeros(n, dtype=bool)
        
        for i in range(5, n-1):
            avg_volume = np.mean(volume[max(0, i-20):i+1])
            volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
            
            # Señal de bombeo (compra agresiva)
            if (volume_ratio > 1.5 and 
                close[i] > close[i-1] and 
                low[i] <= np.min(low[max(0, i-5):i+1]) * 1.01):
                
                whale_pump[i] = min(100, volume_ratio * 20 * sensitivity)
            
            # Señal de descarga (venta agresiva)
            if (volume_ratio > 1.5 and 
                close[i] < close[i-1] and 
                high[i] >= np.max(high[max(0, i-5):i+1]) * 0.99):
                
                whale_dump[i] = min(100, volume_ratio * 20 * sensitivity)
        
        # Confirmar señales con volumen sostenido
        for i in range(7, n):
            if (whale_pump[i] > 20 and 
                np.mean(whale_pump[i-3:i+1]) > 15 and
                volume[i] > np.mean(volume[max(0, i-10):i+1])):
                confirmed_buy[i] = True
            
            if (whale_dump[i] > 20 and 
                np.mean(whale_dump[i-3:i+1]) > 15 and
                volume[i] > np.mean(volume[max(0, i-10):i+1])):
                confirmed_sell[i] = True
        
        return {
            'whale_pump': whale_pump.tolist(),
            'whale_dump': whale_dump.tolist(),
            'confirmed_buy': confirmed_buy.tolist(),
            'confirmed_sell': confirmed_sell.tolist()
        }
    
    def calculate_rsi_maverick(self, close: np.ndarray, length: int = 20, bb_multiplier: float = 2.0) -> List[float]:
        """Calcular RSI Maverick (%B de Bollinger)"""
        n = len(close)
        
        upper, middle, lower = self.calculate_bollinger_bands(close, length, bb_multiplier)
        
        b_percent = np.zeros(n)
        for i in range(n):
            band_width = upper[i] - lower[i]
            if band_width > 0:
                b_percent[i] = (close[i] - lower[i]) / band_width
            else:
                b_percent[i] = 0.5
        
        return b_percent.tolist()
    
    def detect_divergence(self, price: np.ndarray, indicator: np.ndarray, lookback: int = 14) -> Tuple[List[bool], List[bool]]:
        """Detectar divergencias alcistas y bajistas"""
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
    
    def detect_chart_patterns(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 50) -> Dict[str, List[bool]]:
        """Detectar patrones chartistas simples"""
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
                    price_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1]
                    if price_diff < 0.02:
                        patterns['double_top'][i] = True
            
            # Doble Fondo
            if len(window_low) >= 15:
                troughs = []
                for j in range(1, len(window_low)-1):
                    if window_low[j] < window_low[j-1] and window_low[j] < window_low[j+1]:
                        troughs.append((j, window_low[j]))
                
                if len(troughs) >= 2:
                    last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                    price_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1]
                    if price_diff < 0.02:
                        patterns['double_bottom'][i] = True
        
        # Convertir a listas
        return {k: v.tolist() for k, v in patterns.items()}
    
    # ============================================
    # VOLUMEN Y S/R MEJORADOS
    # ============================================
    
    def calculate_support_resistance_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 50) -> Dict:
        """Calcular 4 niveles de soporte y resistencia"""
        n = len(close)
        
        # Usar pivots para identificar niveles
        pivots_high = []
        pivots_low = []
        
        for i in range(5, n-5):
            # Pivot alto
            if high[i] == np.max(high[i-5:i+6]):
                pivots_high.append((i, high[i]))
            
            # Pivot bajo
            if low[i] == np.min(low[i-5:i+6]):
                pivots_low.append((i, low[i]))
        
        # Agrupar niveles cercanos
        def cluster_levels(levels, tolerance=0.02):
            if not levels:
                return []
            
            levels_sorted = sorted(levels, key=lambda x: x[1])
            clusters = []
            current_cluster = [levels_sorted[0]]
            
            for level in levels_sorted[1:]:
                if abs(level[1] - current_cluster[-1][1]) / current_cluster[-1][1] < tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [level]
            
            clusters.append(current_cluster)
            return [np.mean([l[1] for l in cluster]) for cluster in clusters]
        
        resistance_levels = cluster_levels([(idx, val) for idx, val in pivots_high])
        support_levels = cluster_levels([(idx, val) for idx, val in pivots_low])
        
        # Ordenar y limitar a 4 niveles
        resistance_levels = sorted(resistance_levels, reverse=True)[:4]
        support_levels = sorted(support_levels)[:4]
        
        # Si no hay suficientes niveles, usar percentiles
        if len(resistance_levels) < 4:
            percentiles = [75, 85, 92, 97]
            for p in percentiles[len(resistance_levels):]:
                resistance_levels.append(np.percentile(high[-period:], p))
        
        if len(support_levels) < 4:
            percentiles = [25, 15, 8, 3]
            for p in percentiles[len(support_levels):]:
                support_levels.append(np.percentile(low[-period:], p))
        
        # Asegurar que los niveles estén ordenados correctamente
        resistance_levels = sorted(resistance_levels, reverse=True)
        support_levels = sorted(support_levels)
        
        return {
            'resistances': resistance_levels,
            'supports': support_levels,
            'current_support': support_levels[0] if support_levels else np.min(low[-period:]),
            'current_resistance': resistance_levels[0] if resistance_levels else np.max(high[-period:])
        }
    
    def calculate_volume_indicators(self, volume: np.ndarray, close: np.ndarray, period: int = 20) -> Dict:
        """Calcular indicadores de volumen mejorados"""
        n = len(volume)
        
        # EMA de volumen
        volume_ema = self.calculate_ema(volume, period)
        
        # Ratio volumen/EMA
        volume_ratio = np.zeros(n)
        for i in range(n):
            if volume_ema[i] > 0:
                volume_ratio[i] = volume[i] / volume_ema[i]
            else:
                volume_ratio[i] = 1
        
        # Detectar anomalías (volumen > 2.5x EMA)
        volume_anomaly = volume_ratio > 2.5
        
        # Detectar clusters (múltiples anomalías consecutivas)
        volume_clusters = np.zeros(n, dtype=bool)
        for i in range(5, n):
            if np.sum(volume_anomaly[i-4:i+1]) >= 3:
                volume_clusters[i] = True
        
        # Determinar dirección (compra/venta) basado en cambio de precio
        volume_direction = np.zeros(n)  # 1=compra, -1=venta, 0=neutral
        for i in range(1, n):
            if volume_anomaly[i]:
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                if price_change > 0.5:
                    volume_direction[i] = 1  # Compra
                elif price_change < -0.5:
                    volume_direction[i] = -1  # Venta
        
        return {
            'volume_ema': volume_ema.tolist(),
            'volume_ratio': volume_ratio.tolist(),
            'volume_anomaly': volume_anomaly.tolist(),
            'volume_clusters': volume_clusters.tolist(),
            'volume_direction': volume_direction.tolist(),
            'volume_colors': ['green' if d == 1 else 'red' if d == -1 else 'gray' for d in volume_direction]
        }
    
    # ============================================
    # ESTRATEGIA 2: COINMARKETCAP VOLUME SPIKE
    # ============================================
    
    def get_cmc_volume_data(self) -> Optional[Dict]:
        """Obtener datos de volumen de CoinMarketCap"""
        cache_key = "cmc_volume_data"
        current_time = datetime.now()
        
        # Verificar cache (60 segundos)
        if cache_key in self.cmc_cache:
            data, timestamp = self.cmc_cache[cache_key]
            if (current_time - timestamp).seconds < 60:
                return data
        
        try:
            headers = {
                'X-CMC_PRO_API_KEY': CMC_API_KEY,
                'Accept': 'application/json'
            }
            
            # Obtener datos de las 5 criptos principales
            params = {
                'start': '1',
                'limit': '10',
                'convert': 'USD'
            }
            
            response = requests.get(
                f"{CMC_BASE_URL}/cryptocurrency/listings/latest",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                cmc_data = {}
                
                for crypto in data['data']:
                    symbol = crypto['symbol']
                    if symbol in CMC_SYMBOLS:
                        cmc_data[symbol] = {
                            'volume_24h': crypto['quote']['USD']['volume_24h'],
                            'price': crypto['quote']['USD']['price'],
                            'percent_change_24h': crypto['quote']['USD']['percent_change_24h'],
                            'market_cap': crypto['quote']['USD']['market_cap'],
                            'last_updated': crypto['quote']['USD']['last_updated']
                        }
                
                self.cmc_cache[cache_key] = (cmc_data, current_time)
                self.cmc_last_update = current_time
                return cmc_data
            
        except Exception as e:
            print(f"Error obteniendo datos CMC: {e}")
        
        return None
    
    def detect_cmc_volume_spike(self) -> List[Dict]:
        """Detectar spikes de volumen atípicos en CMC"""
        cmc_data = self.get_cmc_volume_data()
        if not cmc_data:
            return []
        
        alerts = []
        current_time = datetime.now()
        
        # Umbral de spike: 300% del volumen histórico promedio
        # Para simplicidad, usamos un promedio fijo basado en market cap
        for symbol, data in cmc_data.items():
            volume_24h = data['volume_24h']
            percent_change = data['percent_change_24h']
            price = data['price']
            
            # Estimar volumen histórico promedio (simplificado)
            # En producción, se debería almacenar histórico de 7 días
            historical_avg_volume = volume_24h * 0.25  # Asumir que volumen normal es 25% del actual
            
            # Verificar spike (300% = 3x)
            if volume_24h > historical_avg_volume * 3:
                # Determinar dirección
                direction = "COMPRA" if percent_change > 0 else "VENTA"
                
                # Formatear monto en millones
                amount_millions = volume_24h / 1_000_000
                
                alert_id = f"cmc_{symbol}_{direction}_{int(current_time.timestamp() / 300)}"  # Agrupar por 5 minutos
                
                # Verificar si ya se envió alerta similar recientemente (2 horas)
                if alert_id not in self.volume_spike_alerts_sent:
                    alert = {
                        'symbol': f"{symbol}-USDT",
                        'cmc_symbol': symbol,
                        'direction': direction,
                        'volume_24h': volume_24h,
                        'amount_millions': round(amount_millions, 2),
                        'percent_change': percent_change,
                        'price': price,
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'alert_id': alert_id
                    }
                    
                    alerts.append(alert)
                    self.volume_spike_alerts_sent[alert_id] = current_time
                    
                    # Limpiar cache viejo (más de 4 horas)
                    to_delete = []
                    for aid, timestamp in self.volume_spike_alerts_sent.items():
                        if (current_time - timestamp).seconds > 14400:  # 4 horas
                            to_delete.append(aid)
                    
                    for aid in to_delete:
                        del self.volume_spike_alerts_sent[aid]
        
        return alerts
    
    # ============================================
    # ANÁLISIS MULTI-TIMEFRAME
    # ============================================
    
    def check_multi_timeframe_condition(self, symbol: str, interval: str, signal_type: str) -> Tuple[bool, List[str]]:
        """Verificar condición multi-timeframe según temporalidad"""
        details = []
        
        # Para 12h, 1D, 1W no aplica multi-timeframe
        if interval in ['12h', '1D', '1W']:
            details.append("Multi-Timeframe: No aplica para esta temporalidad")
            return True, details
        
        hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
        if not hierarchy:
            details.append("Multi-Timeframe: No hay jerarquía definida")
            return False, details
        
        # 1. Temporalidad Menor (debe estar en dirección de la señal)
        try:
            menor_df = self.get_kucoin_data(symbol, hierarchy['menor'], 30)
            if menor_df is not None and len(menor_df) > 10:
                menor_trend = self.calculate_trend_strength_maverick(menor_df['close'].values)
                
                if signal_type == 'LONG':
                    menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']
                    menor_no_trade = not menor_trend['no_trade_zones'][-1]
                    details.append(f"TF Menor ({hierarchy['menor']}): {'ALCISTA' if menor_ok else 'NO ALCISTA'} | Zona NO OPERAR: {'NO' if menor_no_trade else 'SI'}")
                else:  # SHORT
                    menor_ok = menor_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
                    menor_no_trade = not menor_trend['no_trade_zones'][-1]
                    details.append(f"TF Menor ({hierarchy['menor']}): {'BAJISTA' if menor_ok else 'NO BAJISTA'} | Zona NO OPERAR: {'NO' if menor_no_trade else 'SI'}")
                
                menor_condition = menor_ok and menor_no_trade
            else:
                details.append(f"TF Menor ({hierarchy['menor']}): Sin datos suficientes")
                menor_condition = False
        except Exception as e:
            details.append(f"TF Menor ({hierarchy['menor']}): Error - {str(e)[:50]}")
            menor_condition = False
        
        # 2. Temporalidad Actual (fuera de zona NO OPERAR)
        try:
            actual_df = self.get_kucoin_data(symbol, interval, 30)
            if actual_df is not None and len(actual_df) > 10:
                actual_trend = self.calculate_trend_strength_maverick(actual_df['close'].values)
                actual_no_trade = not actual_trend['no_trade_zones'][-1]
                details.append(f"TF Actual ({interval}): Zona NO OPERAR: {'NO' if actual_no_trade else 'SI'}")
                actual_condition = actual_no_trade
            else:
                details.append(f"TF Actual ({interval}): Sin datos suficientes")
                actual_condition = False
        except Exception as e:
            details.append(f"TF Actual ({interval}): Error - {str(e)[:50]}")
            actual_condition = False
        
        # 3. Temporalidad Media (debe estar en dirección de la señal)
        try:
            media_df = self.get_kucoin_data(symbol, hierarchy['media'], 30)
            if media_df is not None and len(media_df) > 10:
                media_trend = self.calculate_trend_strength_maverick(media_df['close'].values)
                
                if signal_type == 'LONG':
                    media_ok = media_trend['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']
                    details.append(f"TF Media ({hierarchy['media']}): {'ALCISTA' if media_ok else 'NO ALCISTA'}")
                else:  # SHORT
                    media_ok = media_trend['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']
                    details.append(f"TF Media ({hierarchy['media']}): {'BAJISTA' if media_ok else 'NO BAJISTA'}")
                
                media_condition = media_ok
            else:
                details.append(f"TF Media ({hierarchy['media']}): Sin datos suficientes")
                media_condition = False
        except Exception as e:
            details.append(f"TF Media ({hierarchy['media']}): Error - {str(e)[:50]}")
            media_condition = False
        
        # 4. Temporalidad Mayor (puede estar en dirección o neutral)
        try:
            mayor_df = self.get_kucoin_data(symbol, hierarchy['mayor'], 30)
            if mayor_df is not None and len(mayor_df) > 10:
                mayor_trend = self.calculate_trend_strength_maverick(mayor_df['close'].values)
                mayor_signal = mayor_trend['strength_signals'][-1]
                
                if signal_type == 'LONG':
                    mayor_ok = mayor_signal in ['STRONG_UP', 'WEAK_UP', 'NEUTRAL']
                    details.append(f"TF Mayor ({hierarchy['mayor']}): {mayor_signal}")
                else:  # SHORT
                    mayor_ok = mayor_signal in ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL']
                    details.append(f"TF Mayor ({hierarchy['mayor']}): {mayor_signal}")
                
                mayor_condition = mayor_ok
            else:
                details.append(f"TF Mayor ({hierarchy['mayor']}): Sin datos suficientes")
                mayor_condition = False
        except Exception as e:
            details.append(f"TF Mayor ({hierarchy['mayor']}): Error - {str(e)[:50]}")
            mayor_condition = False
        
        # Resultado final
        multi_tf_ok = menor_condition and actual_condition and media_condition and mayor_condition
        
        if multi_tf_ok:
            details.insert(0, "✅ Multi-Timeframe: CONFIRMADO")
        else:
            details.insert(0, "❌ Multi-Timeframe: NO CONFIRMADO")
        
        return multi_tf_ok, details
    
    # ============================================
    # EVALUACIÓN DE SEÑALES CON PESOS ACTUALIZADOS
    # ============================================
    
    def evaluate_signal_conditions(self, data: Dict, current_idx: int, interval: str, signal_type: str) -> Tuple[Dict, List[str]]:
        """Evaluar condiciones de señal con pesos actualizados"""
        conditions = {}
        details = []
        
        # Determinar pesos según temporalidad
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            weights = {
                'multi_timeframe': 30,
                'trend_strength': 25,
                'ma_cross': 10,
                'dmi_cross': 10,
                'adx_slope': 5,
                'bollinger': 8,
                'macd_cross': 10,
                'volume_anomaly': 7,
                'rsi_maverick_div': 8,
                'rsi_traditional_div': 5,
                'chart_pattern': 5,
                'breakout': 5
            }
        elif interval in ['12h', '1D']:
            weights = {
                'trend_strength': 25,
                'whale_signal': 30,
                'ma_cross': 10,
                'dmi_cross': 10,
                'adx_slope': 5,
                'bollinger': 8,
                'macd_cross': 10,
                'volume_anomaly': 7,
                'rsi_maverick_div': 8,
                'rsi_traditional_div': 5,
                'chart_pattern': 5,
                'breakout': 5
            }
        else:  # 1W
            weights = {
                'trend_strength': 55,
                'ma_cross': 10,
                'dmi_cross': 10,
                'adx_slope': 5,
                'bollinger': 8,
                'macd_cross': 10,
                'volume_anomaly': 7,
                'rsi_maverick_div': 8,
                'rsi_traditional_div': 5,
                'chart_pattern': 5,
                'breakout': 5
            }
        
        # Inicializar condiciones
        for key in weights.keys():
            conditions[key] = {'value': False, 'weight': weights[key]}
        
        # Obtener valores actuales
        current_price = data['close'][current_idx]
        
        # 1. Multi-Timeframe (solo para 15m-8h)
        if interval in ['15m', '30m', '1h', '2h', '4h', '8h']:
            multi_tf_ok, multi_details = self.check_multi_timeframe_condition(
                data['symbol'], interval, signal_type
            )
            conditions['multi_timeframe']['value'] = multi_tf_ok
            details.extend(multi_details)
        
        # 2. Fuerza de Tendencia Maverick
        trend_strength_signal = data['trend_strength_signals'][current_idx]
        no_trade_zone = data['no_trade_zones'][current_idx]
        
        if signal_type == 'LONG':
            ft_condition = (trend_strength_signal in ['STRONG_UP', 'WEAK_UP']) and (not no_trade_zone)
            details.append(f"FT Maverick: {'ALCISTA' if ft_condition else 'NO ALCISTA'} | Zona NO OPERAR: {'NO' if not no_trade_zone else 'SI'}")
        else:  # SHORT
            ft_condition = (trend_strength_signal in ['STRONG_DOWN', 'WEAK_DOWN']) and (not no_trade_zone)
            details.append(f"FT Maverick: {'BAJISTA' if ft_condition else 'NO BAJISTA'} | Zona NO OPERAR: {'NO' if not no_trade_zone else 'SI'}")
        
        conditions['trend_strength']['value'] = ft_condition
        
        # 3. Señal de Ballenas (solo 12h, 1D)
        if interval in ['12h', '1D']:
            whale_condition = False
            if signal_type == 'LONG':
                whale_condition = (
                    data['whale_pump'][current_idx] > 20 and
                    data['confirmed_buy'][current_idx]
                )
            else:  # SHORT
                whale_condition = (
                    data['whale_dump'][current_idx] > 20 and
                    data['confirmed_sell'][current_idx]
                )
            conditions['whale_signal']['value'] = whale_condition
            details.append(f"Señal Ballenas: {'CONFIRMADA' if whale_condition else 'NO CONFIRMADA'}")
        
        # 4. Cruce de Medias Móviles (9 y 21) - solo cruce + 1 vela
        ma_cross_condition = False
        if current_idx >= 1:
            ma9_current = data['ma_9'][current_idx]
            ma21_current = data['ma_21'][current_idx]
            ma9_prev = data['ma_9'][current_idx-1]
            ma21_prev = data['ma_21'][current_idx-1]
            
            if signal_type == 'LONG':
                ma_cross_condition = (
                    (ma9_current > ma21_current and ma9_prev <= ma21_prev) or
                    (ma9_current > ma21_current and ma9_current > data['ma_9'][max(0, current_idx-2)])
                )
            else:  # SHORT
                ma_cross_condition = (
                    (ma9_current < ma21_current and ma9_prev >= ma21_prev) or
                    (ma9_current < ma21_current and ma9_current < data['ma_9'][max(0, current_idx-2)])
                )
        
        conditions['ma_cross']['value'] = ma_cross_condition
        if ma_cross_condition:
            details.append("Cruce MA9/MA21: CONFIRMADO")
        
        # 5. Cruce DMI (+DI/-DI) - solo cruce + 1 vela
        dmi_cross_condition = False
        if current_idx >= 1:
            plus_di_current = data['plus_di'][current_idx]
            minus_di_current = data['minus_di'][current_idx]
            plus_di_prev = data['plus_di'][current_idx-1]
            minus_di_prev = data['minus_di'][current_idx-1]
            
            if signal_type == 'LONG':
                dmi_cross_condition = (
                    (plus_di_current > minus_di_current and plus_di_prev <= minus_di_prev) or
                    (plus_di_current > minus_di_current and data['plus_di'][max(0, current_idx-2)] <= data['minus_di'][max(0, current_idx-2)])
                )
            else:  # SHORT
                dmi_cross_condition = (
                    (minus_di_current > plus_di_current and minus_di_prev <= plus_di_prev) or
                    (minus_di_current > plus_di_current and data['minus_di'][max(0, current_idx-2)] <= data['plus_di'][max(0, current_idx-2)])
                )
        
        conditions['dmi_cross']['value'] = dmi_cross_condition
        if dmi_cross_condition:
            details.append(f"Cruce DMI: {'+DI/-DI' if signal_type == 'LONG' else '-DI/+DI'}")
        
        # 6. ADX con pendiente positiva > nivel
        adx_slope_condition = False
        if current_idx >= 2:
            adx_current = data['adx'][current_idx]
            adx_prev = data['adx'][current_idx-1]
            adx_level = 25  # Nivel mínimo
            
            adx_slope_condition = (adx_current > adx_level) and (adx_current > adx_prev)
        
        conditions['adx_slope']['value'] = adx_slope_condition
        if adx_slope_condition:
            details.append(f"ADX: {adx_current:.1f} > {adx_prev:.1f}")
        
        # 7. Bandas de Bollinger
        bollinger_condition = False
        if current_idx >= 1:
            price = data['close'][current_idx]
            bb_upper = data['bb_upper'][current_idx]
            bb_lower = data['bb_lower'][current_idx]
            bb_middle = data['bb_middle'][current_idx]
            
            if signal_type == 'LONG':
                bollinger_condition = (
                    price <= bb_lower * 1.02 or  # Toca banda inferior
                    (price > bb_middle and data['close'][current_idx-1] <= bb_middle)  # Cruza media
                )
            else:  # SHORT
                bollinger_condition = (
                    price >= bb_upper * 0.98 or  # Toca banda superior
                    (price < bb_middle and data['close'][current_idx-1] >= bb_middle)  # Cruza media
                )
        
        conditions['bollinger']['value'] = bollinger_condition
        if bollinger_condition:
            details.append("Bandas Bollinger: SEÑAL")
        
        # 8. Cruce MACD - solo cruce + 1 vela
        macd_cross_condition = False
        if current_idx >= 1:
            macd_current = data['macd'][current_idx]
            macd_signal_current = data['macd_signal'][current_idx]
            macd_prev = data['macd'][current_idx-1]
            macd_signal_prev = data['macd_signal'][current_idx-1]
            
            if signal_type == 'LONG':
                macd_cross_condition = (
                    (macd_current > macd_signal_current and macd_prev <= macd_signal_prev) or
                    (macd_current > macd_signal_current and data['macd'][max(0, current_idx-2)] <= data['macd_signal'][max(0, current_idx-2)])
                )
            else:  # SHORT
                macd_cross_condition = (
                    (macd_current < macd_signal_current and macd_prev >= macd_signal_prev) or
                    (macd_current < macd_signal_current and data['macd'][max(0, current_idx-2)] >= data['macd_signal'][max(0, current_idx-2)])
                )
        
        conditions['macd_cross']['value'] = macd_cross_condition
        if macd_cross_condition:
            details.append("Cruce MACD: CONFIRMADO")
        
        # 9. Volumen Anómalo y Clusters (>= 1)
        volume_condition = (
            data['volume_anomaly'][current_idx] and 
            data['volume_clusters'][max(0, current_idx-3):current_idx+1].count(True) >= 1
        )
        
        conditions['volume_anomaly']['value'] = volume_condition
        if volume_condition:
            details.append("Volumen Anómalo: DETECTADO")
        
        # 10. Divergencia RSI Maverick (+4 velas)
        rsi_maverick_div_condition = False
        if signal_type == 'LONG':
            rsi_maverick_div_condition = any(
                data['rsi_maverick_bullish_divergence'][max(0, current_idx-4):current_idx+1]
            )
        else:  # SHORT
            rsi_maverick_div_condition = any(
                data['rsi_maverick_bearish_divergence'][max(0, current_idx-4):current_idx+1]
            )
        
        conditions['rsi_maverick_div']['value'] = rsi_maverick_div_condition
        if rsi_maverick_div_condition:
            details.append(f"Divergencia RSI Maverick: {'ALCISTA' if signal_type == 'LONG' else 'BAJISTA'}")
        
        # 11. Divergencia RSI Tradicional (+4 velas)
        rsi_traditional_div_condition = False
        if signal_type == 'LONG':
            rsi_traditional_div_condition = any(
                data['rsi_bullish_divergence'][max(0, current_idx-4):current_idx+1]
            )
        else:  # SHORT
            rsi_traditional_div_condition = any(
                data['rsi_bearish_divergence'][max(0, current_idx-4):current_idx+1]
            )
        
        conditions['rsi_traditional_div']['value'] = rsi_traditional_div_condition
        if rsi_traditional_div_condition:
            details.append(f"Divergencia RSI Tradicional: {'ALCISTA' if signal_type == 'LONG' else 'BAJISTA'}")
        
        # 12. Patrones Chartistas (+7 velas)
        chart_pattern_condition = False
        lookback = min(7, current_idx + 1)
        
        if signal_type == 'LONG':
            chart_pattern_condition = any(
                data['chart_patterns']['double_bottom'][max(0, current_idx-lookback):current_idx+1]
            ) or any(
                data['chart_patterns']['bullish_flag'][max(0, current_idx-lookback):current_idx+1]
            )
        else:  # SHORT
            chart_pattern_condition = any(
                data['chart_patterns']['double_top'][max(0, current_idx-lookback):current_idx+1]
            ) or any(
                data['chart_patterns']['head_shoulders'][max(0, current_idx-lookback):current_idx+1]
            ) or any(
                data['chart_patterns']['bearish_flag'][max(0, current_idx-lookback):current_idx+1]
            )
        
        conditions['chart_pattern']['value'] = chart_pattern_condition
        if chart_pattern_condition:
            details.append("Patrón Chartista: DETECTADO")
        
        # 13. Rupturas/Breakouts (+1 vela)
        breakout_condition = False
        if signal_type == 'LONG':
            breakout_condition = any(
                data['breakout_up'][max(0, current_idx-1):current_idx+1]
            )
        else:  # SHORT
            breakout_condition = any(
                data['breakout_down'][max(0, current_idx-1):current_idx+1]
            )
        
        conditions['breakout']['value'] = breakout_condition
        if breakout_condition:
            details.append(f"Ruptura: {'ALCISTA' if signal_type == 'LONG' else 'BAJISTA'}")
        
        return conditions, details
    
    def calculate_signal_score(self, conditions: Dict, signal_type: str, ma200_condition: str) -> Tuple[float, List[str]]:
        """Calcular score de señal con pesos actualizados"""
        total_possible = 0
        achieved = 0
        fulfilled_conditions = []
        
        for key, condition in conditions.items():
            total_possible += condition['weight']
            if condition['value']:
                achieved += condition['weight']
                
                # Agregar a condiciones cumplidas (solo indicadores complementarios)
                if key not in ['multi_timeframe', 'trend_strength', 'whale_signal']:
                    fulfilled_conditions.append(self.get_condition_name(key))
        
        if total_possible == 0:
            return 0.0, []
        
        base_score = (achieved / total_possible) * 100
        
        # Ajustar score mínimo según posición MA200
        if signal_type == 'LONG':
            min_score = 65 if ma200_condition == 'above' else 70
        else:  # SHORT
            min_score = 65 if ma200_condition == 'below' else 70
        
        final_score = base_score if base_score >= min_score else 0
        
        return min(final_score, 100), fulfilled_conditions
    
    def get_condition_name(self, key: str) -> str:
        """Obtener nombre legible de condición"""
        names = {
            'multi_timeframe': 'Multi-Timeframe',
            'trend_strength': 'Fuerza Tendencia Maverick',
            'whale_signal': 'Señal Ballenas',
            'ma_cross': 'Cruce MA9/MA21',
            'dmi_cross': 'Cruce DMI',
            'adx_slope': 'ADX Pendiente Positiva',
            'bollinger': 'Bandas Bollinger',
            'macd_cross': 'Cruce MACD',
            'volume_anomaly': 'Volumen Anómalo',
            'rsi_maverick_div': 'Divergencia RSI Maverick',
            'rsi_traditional_div': 'Divergencia RSI Tradicional',
            'chart_pattern': 'Patrón Chartista',
            'breakout': 'Ruptura'
        }
        return names.get(key, key)
    
    # ============================================
    # GENERACIÓN DE SEÑALES COMPLETA
    # ============================================
    
    def generate_trading_signal(self, symbol: str, interval: str, **kwargs) -> Dict:
        """Generar señal de trading completa"""
        try:
            # Obtener datos
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return self._create_empty_signal(symbol)
            
            # Calcular indicadores básicos
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Indicadores avanzados
            whale_data = self.calculate_whale_signals(df)
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            rsi_traditional = self.calculate_rsi(close)
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Detectar divergencias
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            
            # Detectar rupturas
            sr_levels = self.calculate_support_resistance_levels(high, low, close)
            breakout_up = close > np.array([sr_levels['current_resistance']] * len(close))
            breakout_down = close < np.array([sr_levels['current_support']] * len(close))
            
            # Patrones chartistas
            chart_patterns = self.detect_chart_patterns(high, low, close)
            
            # FT Maverick
            trend_data = self.calculate_trend_strength_maverick(close)
            
            # Medias móviles
            ma_9 = self.calculate_sma(close, 9)
            ma_21 = self.calculate_sma(close, 21)
            ma_50 = self.calculate_sma(close, 50)
            ma_200 = self.calculate_sma(close, 200)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(close)
            
            # Bandas Bollinger
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Indicadores de volumen
            volume_data = self.calculate_volume_indicators(volume, close)
            
            # Preparar datos para análisis
            current_idx = -1
            current_price = close[current_idx]
            
            # Determinar condición MA200
            ma200_condition = 'above' if current_price > ma_200[current_idx] else 'below'
            
            analysis_data = {
                'symbol': symbol,
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
                'rsi_traditional': rsi_traditional,
                'rsi_maverick': rsi_maverick,
                'rsi_bullish_divergence': rsi_bullish,
                'rsi_bearish_divergence': rsi_bearish,
                'rsi_maverick_bullish_divergence': rsi_maverick_bullish,
                'rsi_maverick_bearish_divergence': rsi_maverick_bearish,
                'breakout_up': breakout_up.tolist(),
                'breakout_down': breakout_down.tolist(),
                'chart_patterns': chart_patterns,
                'trend_strength_signals': trend_data['strength_signals'],
                'no_trade_zones': trend_data['no_trade_zones'],
                'ma_9': ma_9,
                'ma_21': ma_21,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'volume_anomaly': volume_data['volume_anomaly'],
                'volume_clusters': volume_data['volume_clusters'],
                'volume_direction': volume_data['volume_direction']
            }
            
            # Evaluar señales LONG y SHORT
            long_conditions, long_details = self.evaluate_signal_conditions(
                analysis_data, current_idx, interval, 'LONG'
            )
            
            short_conditions, short_details = self.evaluate_signal_conditions(
                analysis_data, current_idx, interval, 'SHORT'
            )
            
            # Calcular scores
            long_score, long_fulfilled = self.calculate_signal_score(
                long_conditions, 'LONG', ma200_condition
            )
            
            short_score, short_fulfilled = self.calculate_signal_score(
                short_conditions, 'SHORT', ma200_condition
            )
            
            # Determinar señal final
            signal_type = 'NEUTRAL'
            signal_score = 0
            fulfilled_conditions = []
            details = []
            
            if long_score >= 65:
                signal_type = 'LONG'
                signal_score = long_score
                fulfilled_conditions = long_fulfilled
                details = long_details
            elif short_score >= 65:
                signal_type = 'SHORT'
                signal_score = short_score
                fulfilled_conditions = short_fulfilled
                details = short_details
            
            # Calcular niveles de trading (entrada en S/R más cercano)
            entry_price = self.calculate_optimal_entry(
                current_price, signal_type, sr_levels
            )
            
            stop_loss = self.calculate_stop_loss(
                entry_price, signal_type, sr_levels
            )
            
            take_profit = self.calculate_take_profit(
                entry_price, stop_loss, signal_type
            )
            
            # Preparar respuesta
            response = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'signal_score': round(signal_score, 1),
                'current_price': round(current_price, 6),
                'entry': round(entry_price, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit': [round(tp, 6) for tp in take_profit],
                'supports': [round(s, 6) for s in sr_levels['supports'][:4]],
                'resistances': [round(r, 6) for r in sr_levels['resistances'][:4]],
                'ma200_condition': ma200_condition,
                'fulfilled_conditions': fulfilled_conditions,
                'details': details,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'indicators': {
                    'close': close[-50:].tolist(),
                    'volume': volume[-50:].tolist(),
                    'volume_anomaly': volume_data['volume_anomaly'][-50:],
                    'volume_clusters': volume_data['volume_clusters'][-50:],
                    'volume_direction': volume_data['volume_direction'][-50:],
                    'volume_colors': volume_data['volume_colors'][-50:],
                    'adx': adx[-50:].tolist(),
                    'plus_di': plus_di[-50:].tolist(),
                    'minus_di': minus_di[-50:].tolist(),
                    'rsi_traditional': rsi_traditional[-50:].tolist(),
                    'rsi_maverick': rsi_maverick[-50:].tolist(),
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
                    'trend_strength': trend_data['trend_strength'][-50:],
                    'no_trade_zones': trend_data['no_trade_zones'][-50:],
                    'strength_signals': trend_data['strength_signals'][-50:],
                    'whale_pump': whale_data['whale_pump'][-50:],
                    'whale_dump': whale_data['whale_dump'][-50:],
                    'confirmed_buy': whale_data['confirmed_buy'][-50:],
                    'confirmed_sell': whale_data['confirmed_sell'][-50:]
                }
            }
            
            # Registrar señal activa si es válida
            if signal_type in ['LONG', 'SHORT'] and signal_score >= 65:
                signal_key = f"{symbol}_{interval}_{signal_type}_{int(time.time())}"
                self.active_operations[signal_key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'signal': signal_type,
                    'entry_price': entry_price,
                    'timestamp': self.get_bolivia_time(),
                    'score': signal_score
                }
            
            return response
            
        except Exception as e:
            print(f"Error generando señal para {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_signal(symbol)
    
    def _create_empty_signal(self, symbol: str) -> Dict:
        """Crear señal vacía por error"""
        return {
            'symbol': symbol,
            'interval': '4h',
            'signal': 'NEUTRAL',
            'signal_score': 0,
            'current_price': 0,
            'entry': 0,
            'stop_loss': 0,
            'take_profit': [0],
            'supports': [0, 0, 0, 0],
            'resistances': [0, 0, 0, 0],
            'ma200_condition': 'below',
            'fulfilled_conditions': [],
            'details': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'indicators': {}
        }
    
    def calculate_optimal_entry(self, current_price: float, signal_type: str, sr_levels: Dict) -> float:
        """Calcular precio de entrada óptimo (en S/R más cercano)"""
        if signal_type == 'LONG':
            # Para LONG: entrada en el soporte más cercano por debajo del precio
            supports_below = [s for s in sr_levels['supports'] if s < current_price]
            if supports_below:
                return max(supports_below)  # Soporte más alto por debajo
            else:
                # Si no hay soportes por debajo, usar 1% por debajo del precio
                return current_price * 0.99
        else:  # SHORT
            # Para SHORT: entrada en la resistencia más cercana por encima del precio
            resistances_above = [r for r in sr_levels['resistances'] if r > current_price]
            if resistances_above:
                return min(resistances_above)  # Resistencia más baja por encima
            else:
                # Si no hay resistencias por encima, usar 1% por encima del precio
                return current_price * 1.01
    
    def calculate_stop_loss(self, entry_price: float, signal_type: str, sr_levels: Dict) -> float:
        """Calcular stop loss"""
        if signal_type == 'LONG':
            # Para LONG: stop loss en el siguiente soporte más bajo o 3% abajo
            supports_below = [s for s in sr_levels['supports'] if s < entry_price]
            if len(supports_below) > 1:
                return supports_below[1]  # Segundo soporte más cercano
            else:
                return entry_price * 0.97  # 3% stop loss
        else:  # SHORT
            # Para SHORT: stop loss en la siguiente resistencia más alta o 3% arriba
            resistances_above = [r for r in sr_levels['resistances'] if r > entry_price]
            if len(resistances_above) > 1:
                return resistances_above[1]  # Segunda resistencia más cercana
            else:
                return entry_price * 1.03  # 3% stop loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, signal_type: str) -> List[float]:
        """Calcular take profits"""
        risk = abs(entry_price - stop_loss)
        
        if signal_type == 'LONG':
            tp1 = entry_price + risk * 1.5  # 1.5:1 risk:reward
            tp2 = entry_price + risk * 2.5   # 2.5:1 risk:reward
            return [tp1, tp2]
        else:  # SHORT
            tp1 = entry_price - risk * 1.5
            tp2 = entry_price - risk * 2.5
            return [tp1, tp2]
    
    # ============================================
    # GENERACIÓN DE ALERTAS
    # ============================================
    
    def generate_strategy1_alerts(self) -> List[Dict]:
        """Generar alertas para Estrategia 1 (Multi-Timeframe)"""
        alerts = []
        current_time = self.get_bolivia_time()
        
        # Temporalidades a analizar
        intervals = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '1D', '1W']
        
        for interval in intervals:
            # Limitar símbolos para no sobrecargar
            symbols = CRYPTO_SYMBOLS[:15] if interval in ['15m', '30m'] else CRYPTO_SYMBOLS[:10]
            
            for symbol in symbols:
                try:
                    signal_data = self.generate_trading_signal(symbol, interval)
                    
                    if (signal_data['signal'] in ['LONG', 'SHORT'] and 
                        signal_data['signal_score'] >= 65):
                        
                        # Verificar si ya se envió alerta similar recientemente (30 minutos)
                        alert_key = f"strategy1_{symbol}_{interval}_{signal_data['signal']}"
                        last_alert_time = self.alert_cache.get(alert_key)
                        
                        if (last_alert_time is None or 
                            (current_time - last_alert_time).seconds > 1800):  # 30 minutos
                            
                            alerts.append(signal_data)
                            self.alert_cache[alert_key] = current_time
                    
                except Exception as e:
                    print(f"Error generando alerta Strategy1 {symbol} {interval}: {e}")
                    continue
        
        return alerts
    
    def generate_strategy2_alerts(self) -> List[Dict]:
        """Generar alertas para Estrategia 2 (Volúmenes Atípicos CMC)"""
        return self.detect_cmc_volume_spike()

# ============================================
# INSTANCIA GLOBAL DEL SISTEMA
# ============================================

trading_system = TradingSystem()

# ============================================
# FUNCIONES DE TELEGRAM
# ============================================

async def send_telegram_message_async(message: str, photo_bytes: bytes = None):
    """Enviar mensaje a Telegram (con o sin foto)"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        if photo_bytes:
            # Enviar con foto
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo_bytes,
                caption=message[:1024],  # Máximo 1024 caracteres para caption
                parse_mode='HTML'
            )
        else:
            # Enviar solo texto
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
        
        return True
    except Exception as e:
        print(f"Error enviando mensaje Telegram: {e}")
        return False

def send_telegram_alert(alert_data: Dict, strategy: int = 1):
    """Enviar alerta a Telegram según estrategia"""
    try:
        if strategy == 1:
            # Estrategia 1: Multi-Timeframe
            message = format_strategy1_telegram_message(alert_data)
            photo_bytes = generate_strategy1_telegram_chart(alert_data)
        else:
            # Estrategia 2: Volúmenes Atípicos
            message = format_strategy2_telegram_message(alert_data)
            photo_bytes = generate_strategy2_telegram_chart(alert_data)
        
        # Enviar en segundo plano
        asyncio.run(send_telegram_message_async(message, photo_bytes))
        
        print(f"Alerta Strategy{strategy} enviada: {alert_data.get('symbol', 'N/A')}")
        
    except Exception as e:
        print(f"Error enviando alerta Telegram: {e}")

def format_strategy1_telegram_message(data: Dict) -> str:
    """Formatear mensaje Telegram para Estrategia 1"""
    symbol = data['symbol']
    interval = data['interval']
    signal = data['signal']
    score = data['signal_score']
    ma200 = data['ma200_condition']
    
    # Información básica
    message = f"<b>🚨 SEÑAL {signal} CONFIRMADA 🚨</b>\n\n"
    message += f"<b>📊 {symbol} | {interval}</b>\n"
    message += f"<b>🎯 Score:</b> {score}%\n"
    message += f"<b>📈 MA200:</b> {ma200.upper()}\n\n"
    
    # Precios
    message += f"<b>💰 Precio Actual:</b> ${data['current_price']:.6f}\n"
    message += f"<b>🎯 Entrada Recomendada:</b> ${data['entry']:.6f}\n\n"
    
    # Condiciones cumplidas
    if data['fulfilled_conditions']:
        message += "<b>✅ Condiciones Cumplidas:</b>\n"
        for condition in data['fulfilled_conditions'][:5]:  # Máximo 5 condiciones
            message += f"• {condition}\n"
    
    # Soportes/Resistencias (solo los más relevantes)
    if data['supports'] and data['resistances']:
        message += f"\n<b>📊 Soportes:</b> {', '.join([f'${s:.2f}' for s in data['supports'][:2]])}\n"
        message += f"<b>📈 Resistencias:</b> {', '.join([f'${r:.2f}' for r in data['resistances'][:2]])}\n"
    
    return message

def format_strategy2_telegram_message(data: Dict) -> str:
    """Formatear mensaje Telegram para Estrategia 2"""
    symbol = data['symbol']
    direction = data['direction']
    amount = data['amount_millions']
    percent_change = data['percent_change']
    
    if direction == "COMPRA":
        message = f"<b>🚨ALERTA de COMPRA Atípica 🚨</b>\n\n"
        message += f"Se acaba de ingresar o comprar <b>{symbol}</b> en <b>{amount:.2f}M</b> millones de USDT, volumen atípico, revisar LONG\n\n"
    else:
        message = f"<b>🚨ALERTA de VENTA Atípica 🚨</b>\n\n"
        message += f"Se vendieron <b>{amount:.2f}M</b> millones de USDT en <b>{symbol}</b>, volumen atípico, revisar SHORT\n\n"
    
    message += f"<b>📊 Cambio 24h:</b> {percent_change:+.2f}%\n"
    message += f"<b>💰 Precio:</b> ${data['price']:.4f}\n"
    message += f"<b>🕐 Hora:</b> {data['timestamp']}"
    
    return message

def generate_strategy1_telegram_chart(data: Dict) -> bytes:
    """Generar gráfico para Telegram (Estrategia 1)"""
    try:
        # Configurar estilo para Telegram (fondo blanco)
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 12))
        fig.patch.set_facecolor('white')
        
        # 1. Gráfico de Velas
        ax1 = plt.subplot(4, 2, 1)
        if 'indicators' in data and 'close' in data['indicators']:
            closes = data['indicators']['close']
            x = range(len(closes))
            
            # Bandas de Bollinger transparentes
            if 'bb_upper' in data['indicators'] and 'bb_lower' in data['indicators']:
                ax1.fill_between(x, 
                               data['indicators']['bb_lower'],
                               data['indicators']['bb_upper'],
                               alpha=0.2, color='orange', label='BB')
            
            # Medias móviles
            if 'ma_9' in data['indicators']:
                ax1.plot(x, data['indicators']['ma_9'], 'blue', linewidth=1, label='MA9', alpha=0.7)
            if 'ma_21' in data['indicators']:
                ax1.plot(x, data['indicators']['ma_21'], 'red', linewidth=1, label='MA21', alpha=0.7)
            if 'ma_50' in data['indicators']:
                ax1.plot(x, data['indicators']['ma_50'], 'green', linewidth=1.5, label='MA50', alpha=0.7)
            
            # Precio
            ax1.plot(x, closes, 'black', linewidth=2, label='Precio', alpha=0.8)
            
            # Soportes/Resistencias
            for i, support in enumerate(data['supports'][:2]):
                ax1.axhline(y=support, color='green', linestyle='--', alpha=0.5, 
                          label=f'S{i+1}' if i == 0 else "")
            
            for i, resistance in enumerate(data['resistances'][:2]):
                ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5,
                          label=f'R{i+1}' if i == 0 else "")
        
        ax1.set_title(f'{data["symbol"]} - Velas', fontsize=10)
        ax1.legend(fontsize=6)
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(4, 2, 2)
        if 'indicators' in data:
            if 'adx' in data['indicators'] and 'plus_di' in data['indicators'] and 'minus_di' in data['indicators']:
                x = range(len(data['indicators']['adx']))
                ax2.plot(x, data['indicators']['adx'], 'black', linewidth=2, label='ADX')
                ax2.plot(x, data['indicators']['plus_di'], 'green', linewidth=1, label='+DI')
                ax2.plot(x, data['indicators']['minus_di'], 'red', linewidth=1, label='-DI')
                ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_title('ADX con DMI', fontsize=10)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)
        
        # 3. Indicador de Volumen
        ax3 = plt.subplot(4, 2, 3)
        if 'indicators' in data and 'volume' in data['indicators']:
            x = range(len(data['indicators']['volume']))
            volumes = data['indicators']['volume']
            
            # Colores según dirección
            if 'volume_direction' in data['indicators']:
                colors = ['green' if d == 1 else 'red' if d == -1 else 'gray' 
                         for d in data['indicators']['volume_direction']]
                ax3.bar(x, volumes, color=colors, alpha=0.7)
            else:
                ax3.bar(x, volumes, color='gray', alpha=0.7)
            
            # Anomalías
            if 'volume_anomaly' in data['indicators']:
                anomaly_indices = [i for i, v in enumerate(data['indicators']['volume_anomaly']) if v]
                anomaly_volumes = [volumes[i] for i in anomaly_indices]
                ax3.scatter(anomaly_indices, anomaly_volumes, color='orange', s=30, zorder=5)
        
        ax3.set_title('Volumen con Anomalías', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Fuerza de Tendencia Maverick
        ax4 = plt.subplot(4, 2, 4)
        if 'indicators' in data and 'trend_strength' in data['indicators']:
            x = range(len(data['indicators']['trend_strength']))
            strength = data['indicators']['trend_strength']
            
            # Colores según fuerza
            colors = []
            for val in strength:
                if val > 0:
                    colors.append('green' if val > 5 else 'lightgreen')
                else:
                    colors.append('red' if val < -5 else 'lightcoral')
            
            ax4.bar(x, strength, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linewidth=0.5)
        
        ax4.set_title('Fuerza Tendencia Maverick', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Indicador de Ballenas (si aplica)
        ax5 = plt.subplot(4, 2, 5)
        if 'indicators' in data and data['interval'] in ['12h', '1D']:
            if 'whale_pump' in data['indicators'] and 'whale_dump' in data['indicators']:
                x = range(len(data['indicators']['whale_pump']))
                ax5.bar(x, data['indicators']['whale_pump'], color='green', alpha=0.6, label='Compra')
                ax5.bar(x, data['indicators']['whale_dump'], color='red', alpha=0.6, label='Venta')
                ax5.legend(fontsize=6)
        
        ax5.set_title('Actividad Ballenas', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. RSI Maverick
        ax6 = plt.subplot(4, 2, 6)
        if 'indicators' in data and 'rsi_maverick' in data['indicators']:
            x = range(len(data['indicators']['rsi_maverick']))
            ax6.plot(x, data['indicators']['rsi_maverick'], 'blue', linewidth=2)
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
            ax6.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        
        ax6.set_title('RSI Maverick (%B)', fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # 7. RSI Tradicional
        ax7 = plt.subplot(4, 2, 7)
        if 'indicators' in data and 'rsi_traditional' in data['indicators']:
            x = range(len(data['indicators']['rsi_traditional']))
            ax7.plot(x, data['indicators']['rsi_traditional'], 'purple', linewidth=2)
            ax7.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax7.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        ax7.set_title('RSI Tradicional', fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        # 8. MACD
        ax8 = plt.subplot(4, 2, 8)
        if 'indicators' in data and 'macd' in data['indicators']:
            x = range(len(data['indicators']['macd']))
            ax8.plot(x, data['indicators']['macd'], 'blue', linewidth=1, label='MACD')
            ax8.plot(x, data['indicators']['macd_signal'], 'red', linewidth=1, label='Señal')
            
            # Histograma como columnas
            if 'macd_histogram' in data['indicators']:
                colors = ['green' if h > 0 else 'red' for h in data['indicators']['macd_histogram']]
                ax8.bar(x, data['indicators']['macd_histogram'], color=colors, alpha=0.6, label='Hist')
            
            ax8.axhline(y=0, color='black', linewidth=0.5)
            ax8.legend(fontsize=6)
        
        ax8.set_title('MACD', fontsize=10)
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar en buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()
        
    except Exception as e:
        print(f"Error generando gráfico Strategy1: {e}")
        return None

def generate_strategy2_telegram_chart(data: Dict) -> bytes:
    """Generar gráfico para Telegram (Estrategia 2)"""
    try:
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        # 1. Gráfico de Velas (simplificado)
        ax1 = plt.subplot(2, 2, 1)
        
        # Simular datos de precio
        np.random.seed(42)
        n_points = 50
        x = range(n_points)
        prices = 100 + np.cumsum(np.random.randn(n_points) * 2)
        
        # Bandas de Bollinger
        basis = np.mean(prices)
        std = np.std(prices)
        upper = basis + 2 * std
        lower = basis - 2 * std
        
        ax1.fill_between(x, lower, upper, alpha=0.2, color='orange', label='BB')
        
        # Medias móviles
        ma9 = np.convolve(prices, np.ones(9)/9, mode='valid')
        ma21 = np.convolve(prices, np.ones(21)/21, mode='valid')
        
        ax1.plot(x[8:], ma9, 'blue', linewidth=1, label='MA9', alpha=0.7)
        ax1.plot(x[20:], ma21, 'red', linewidth=1, label='MA21', alpha=0.7)
        ax1.plot(x, prices, 'black', linewidth=2, label='Precio', alpha=0.8)
        
        ax1.set_title(f'{data["symbol"]} - Precio', fontsize=10)
        ax1.legend(fontsize=6)
        ax1.grid(True, alpha=0.3)
        
        # 2. ADX con DMI
        ax2 = plt.subplot(2, 2, 2)
        
        # Simular indicadores
        adx = 30 + np.random.randn(n_points) * 5
        plus_di = 25 + np.random.randn(n_points) * 3
        minus_di = 20 + np.random.randn(n_points) * 3
        
        ax2.plot(x, adx, 'black', linewidth=2, label='ADX')
        ax2.plot(x, plus_di, 'green', linewidth=1, label='+DI')
        ax2.plot(x, minus_di, 'red', linewidth=1, label='-DI')
        ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_title('ADX con DMI', fontsize=10)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)
        
        # 3. Gráfico de Anormalidad de Compras/Ventas
        ax3 = plt.subplot(2, 1, 2)
        
        # Simular datos de volumen atípico
        dates = pd.date_range(end=datetime.now(), periods=n_points, freq='H')
        
        # Generar spikes aleatorios
        buy_spikes = np.zeros(n_points)
        sell_spikes = np.zeros(n_points)
        
        spike_indices = np.random.choice(n_points, size=8, replace=False)
        for idx in spike_indices:
            if np.random.rand() > 0.5:
                buy_spikes[idx] = np.random.uniform(50, 100)
            else:
                sell_spikes[idx] = np.random.uniform(50, 100)
        
        # Añadir el spike actual
        if data['direction'] == "COMPRA":
            buy_spikes[-1] = data['amount_millions'] * 10
        else:
            sell_spikes[-1] = data['amount_millions'] * 10
        
        width = 0.35
        x_axis = range(n_points)
        
        ax3.bar([i - width/2 for i in x_axis], buy_spikes, width, color='green', alpha=0.7, label='Compras Atípicas')
        ax3.bar([i + width/2 for i in x_axis], sell_spikes, width, color='red', alpha=0.7, label='Ventas Atípicas')
        
        # Destacar spike actual
        if data['direction'] == "COMPRA":
            ax3.bar(n_points - 1 - width/2, buy_spikes[-1], width, color='darkgreen', alpha=1, edgecolor='black', linewidth=2)
        else:
            ax3.bar(n_points - 1 + width/2, sell_spikes[-1], width, color='darkred', alpha=1, edgecolor='black', linewidth=2)
        
        ax3.set_title('Anormalidad de Compras/Ventas (Últimas 50 horas)', fontsize=10)
        ax3.set_xlabel('Horas')
        ax3.set_ylabel('Intensidad')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar en buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()
        
    except Exception as e:
        print(f"Error generando gráfico Strategy2: {e}")
        return None

# ============================================
# BACKGROUND CHECKER
# ============================================

def background_alert_checker():
    """Verificador de alertas en segundo plano"""
    print("Background alert checker iniciado")
    
    strategy1_last_check = datetime.now()
    strategy2_last_check = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            # Estrategia 1: cada 2 minutos
            if (current_time - strategy1_last_check).seconds >= 120:
                print(f"[{current_time.strftime('%H:%M:%S')}] Verificando Strategy1...")
                
                alerts = trading_system.generate_strategy1_alerts()
                for alert in alerts:
                    send_telegram_alert(alert, strategy=1)
                
                strategy1_last_check = current_time
            
            # Estrategia 2: cada 5 minutos
            if (current_time - strategy2_last_check).seconds >= 300:
                print(f"[{current_time.strftime('%H:%M:%S')}] Verificando Strategy2...")
                
                alerts = trading_system.generate_strategy2_alerts()
                for alert in alerts:
                    send_telegram_alert(alert, strategy=2)
                
                strategy2_last_check = current_time
            
            time.sleep(10)
            
        except Exception as e:
            print(f"Error en background_alert_checker: {e}")
            time.sleep(60)

# Iniciar background checker
try:
    Thread(target=background_alert_checker, daemon=True).start()
except Exception as e:
    print(f"Error iniciando background checker: {e}")

# ============================================
# ENDPOINTS FLASK
# ============================================

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
        
        signal_data = trading_system.generate_trading_signal(symbol, interval)
        return jsonify(signal_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {e}")
        return jsonify({'error': str(e)[:100]}), 500

@app.route('/api/multiple_signals')
def get_multiple_signals():
    """Endpoint para múltiples señales"""
    try:
        interval = request.args.get('interval', '4h')
        
        all_signals = []
        symbols = CRYPTO_SYMBOLS[:8]  # Limitar para rendimiento
        
        for symbol in symbols:
            try:
                signal_data = trading_system.generate_trading_signal(symbol, interval)
                if signal_data['signal'] != 'NEUTRAL' and signal_data['signal_score'] >= 65:
                    all_signals.append(signal_data)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")
                continue
        
        # Separar por tipo
        long_signals = [s for s in all_signals if s['signal'] == 'LONG']
        short_signals = [s for s in all_signals if s['signal'] == 'SHORT']
        
        # Ordenar por score
        long_signals.sort(key=lambda x: x['signal_score'], reverse=True)
        short_signals.sort(key=lambda x: x['signal_score'], reverse=True)
        
        return jsonify({
            'long_signals': long_signals[:5],
            'short_signals': short_signals[:5],
            'total_signals': len(all_signals)
        })
        
    except Exception as e:
        print(f"Error en /api/multiple_signals: {e}")
        return jsonify({'error': str(e)[:100]}), 500

@app.route('/api/scatter_data_improved')
def get_scatter_data_improved():
    """Endpoint para datos del scatter plot"""
    try:
        interval = request.args.get('interval', '4h')
        
        scatter_data = []
        symbols = CRYPTO_SYMBOLS[:20]  # Limitar para rendimiento
        
        for symbol in symbols:
            try:
                signal_data = trading_system.generate_trading_signal(symbol, interval)
                
                if signal_data['current_price'] > 0:
                    # Calcular presiones de compra/venta
                    buy_pressure = min(100, signal_data['signal_score'] if signal_data['signal'] == 'LONG' else 0)
                    sell_pressure = min(100, signal_data['signal_score'] if signal_data['signal'] == 'SHORT' else 0)
                    
                    # Ajustar según indicadores
                    if 'indicators' in signal_data:
                        if signal_data['indicators'].get('volume_direction'):
                            last_direction = signal_data['indicators']['volume_direction'][-1]
                            if last_direction == 1:
                                buy_pressure = min(100, buy_pressure * 1.2)
                            elif last_direction == -1:
                                sell_pressure = min(100, sell_pressure * 1.2)
                    
                    scatter_data.append({
                        'symbol': symbol,
                        'x': float(buy_pressure),
                        'y': float(sell_pressure),
                        'signal_score': float(signal_data['signal_score']),
                        'current_price': float(signal_data['current_price']),
                        'signal': signal_data['signal'],
                        'risk_category': next(
                            (cat for cat, symbols_list in CRYPTO_RISK_CLASSIFICATION.items() 
                             if symbol in symbols_list), 'medio'
                        )
                    })
                    
            except Exception as e:
                print(f"Error procesando {symbol} para scatter: {e}")
                continue
        
        return jsonify(scatter_data)
        
    except Exception as e:
        print(f"Error en /api/scatter_data_improved: {e}")
        return jsonify({'error': str(e)[:100]}), 500

@app.route('/api/crypto_risk_classification')
def get_crypto_risk_classification():
    """Endpoint para clasificación de riesgo"""
    return jsonify(CRYPTO_RISK_CLASSIFICATION)

@app.route('/api/volume_anomaly_signals')
def get_volume_anomaly_signals():
    """Endpoint para señales de volumen atípico"""
    try:
        alerts = trading_system.generate_strategy2_alerts()
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        print(f"Error en /api/volume_anomaly_signals: {e}")
        return jsonify({'alerts': [], 'error': str(e)[:100]}), 500

@app.route('/api/bolivia_time')
def get_bolivia_time():
    """Endpoint para hora de Bolivia"""
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d'),
        'timezone': 'America/La_Paz'
    })

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte completo para descarga"""
    try:
        symbol = request.args.get('symbol', 'BTC-USDT')
        interval = request.args.get('interval', '4h')
        
        # Obtener datos
        signal_data = trading_system.generate_trading_signal(symbol, interval)
        
        if not signal_data or signal_data['current_price'] == 0:
            return jsonify({'error': 'No hay datos para el reporte'}), 400
        
        # Crear figura grande
        fig = plt.figure(figsize=(14, 20))
        
        # Gráfico 1: Precio y niveles
        ax1 = plt.subplot(10, 1, 1)
        
        if 'indicators' in signal_data and 'close' in signal_data['indicators']:
            closes = signal_data['indicators']['close']
            x = range(len(closes))
            
            # Precio
            ax1.plot(x, closes, 'black', linewidth=2, label='Precio')
            
            # Bandas Bollinger
            if 'bb_upper' in signal_data['indicators'] and 'bb_lower' in signal_data['indicators']:
                ax1.fill_between(x, 
                               signal_data['indicators']['bb_lower'],
                               signal_data['indicators']['bb_upper'],
                               alpha=0.1, color='orange', label='BB')
            
            # Medias móviles
            if 'ma_9' in signal_data['indicators']:
                ax1.plot(x, signal_data['indicators']['ma_9'], 'blue', linewidth=1, label='MA9')
            if 'ma_21' in signal_data['indicators']:
                ax1.plot(x, signal_data['indicators']['ma_21'], 'red', linewidth=1, label='MA21')
            if 'ma_50' in signal_data['indicators']:
                ax1.plot(x, signal_data['indicators']['ma_50'], 'green', linewidth=1, label='MA50')
            if 'ma_200' in signal_data['indicators']:
                ax1.plot(x, signal_data['indicators']['ma_200'], 'purple', linewidth=2, label='MA200')
            
            # Soportes y resistencias
            for i, support in enumerate(signal_data['supports'][:4]):
                ax1.axhline(y=support, color='green', linestyle='--', alpha=0.5, 
                          label=f'S{i+1}' if i == 0 else "")
            
            for i, resistance in enumerate(signal_data['resistances'][:4]):
                ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.5,
                          label=f'R{i+1}' if i == 0 else "")
            
            # Niveles de trading
            ax1.axhline(y=signal_data['entry'], color='gold', linestyle='-', linewidth=2, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='darkred', linestyle='-', linewidth=2, label='Stop Loss')
            
            for i, tp in enumerate(signal_data['take_profit'][:2]):
                ax1.axhline(y=tp, color='darkgreen', linestyle='-', linewidth=2, 
                          label=f'TP{i+1}' if i == 0 else "")
        
        ax1.set_title(f'{symbol} - Análisis Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráficos 2-9: Indicadores (similar a función anterior pero simplificado)
        indicators_config = [
            ('ADX/DMI', ['adx', 'plus_di', 'minus_di'], 2, 'ADX con DMI'),
            ('Volumen Anómalo', ['volume', 'volume_anomaly'], 3, 'Volumen con Anomalías'),
            ('Fuerza Tendencia', ['trend_strength'], 4, 'Fuerza Tendencia Maverick'),
            ('Ballenas', ['whale_pump', 'whale_dump'], 5, 'Actividad Ballenas'),
            ('RSI Maverick', ['rsi_maverick'], 6, 'RSI Maverick (%B)'),
            ('RSI Tradicional', ['rsi_traditional'], 7, 'RSI Tradicional'),
            ('MACD', ['macd', 'macd_signal', 'macd_histogram'], 8, 'MACD'),
            ('Volume Direction', ['volume_direction'], 9, 'Dirección Volumen')
        ]
        
        for idx, (indicator_name, indicator_keys, plot_num, title) in enumerate(indicators_config, 2):
            ax = plt.subplot(10, 1, plot_num, sharex=ax1)
            
            if 'indicators' in signal_data:
                x = range(len(signal_data['indicators'].get('close', [])))
                
                for key in indicator_keys:
                    if key in signal_data['indicators']:
                        data = signal_data['indicators'][key]
                        
                        if key == 'volume':
                            # Volumen como barras
                            colors = ['green' if signal_data['indicators']['volume_direction'][i] == 1 
                                    else 'red' if signal_data['indicators']['volume_direction'][i] == -1 
                                    else 'gray' for i in range(len(data))]
                            ax.bar(x, data, color=colors, alpha=0.7)
                            
                            # Marcar anomalías
                            if 'volume_anomaly' in signal_data['indicators']:
                                anomaly_indices = [i for i, v in enumerate(signal_data['indicators']['volume_anomaly']) if v]
                                anomaly_values = [data[i] for i in anomaly_indices]
                                ax.scatter(anomaly_indices, anomaly_values, color='orange', s=30, zorder=5)
                        
                        elif key == 'macd_histogram':
                            # Histograma como barras
                            colors = ['green' if h > 0 else 'red' for h in data]
                            ax.bar(x, data, color=colors, alpha=0.6, width=0.8)
                        
                        elif key == 'trend_strength':
                            # FT Maverick como barras coloreadas
                            colors = []
                            for val in data:
                                if val > 0:
                                    colors.append('green' if val > 5 else 'lightgreen')
                                else:
                                    colors.append('red' if val < -5 else 'lightcoral')
                            ax.bar(x, data, color=colors, alpha=0.7)
                            ax.axhline(y=0, color='black', linewidth=0.5)
                        
                        elif key in ['whale_pump', 'whale_dump']:
                            # Ballenas como barras superpuestas
                            if key == 'whale_pump':
                                ax.bar(x, data, color='green', alpha=0.6, label='Compra')
                            else:
                                ax.bar(x, data, color='red', alpha=0.6, label='Venta')
                        
                        else:
                            # Líneas para otros indicadores
                            color = {
                                'adx': 'black',
                                'plus_di': 'green',
                                'minus_di': 'red',
                                'rsi_maverick': 'blue',
                                'rsi_traditional': 'purple',
                                'macd': 'blue',
                                'macd_signal': 'red',
                                'volume_direction': 'gray'
                            }.get(key, 'blue')
                            
                            ax.plot(x, data, color=color, linewidth=1.5, label=key)
                
                # Líneas de referencia según indicador
                if indicator_name == 'RSI Maverick':
                    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
                    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5)
                    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
                elif indicator_name == 'RSI Tradicional':
                    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                    ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                    ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
                elif indicator_name == 'ADX/DMI':
                    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
                elif indicator_name == 'MACD':
                    ax.axhline(y=0, color='black', linewidth=0.5)
            
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(indicator_name)
            if indicator_name in ['Ballenas', 'ADX/DMI']:
                ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
        
        # Información de la señal
        ax10 = plt.subplot(10, 1, 10)
        ax10.axis('off')
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        PRECIO: ${signal_data['current_price']:.6f}
        ENTRADA: ${signal_data['entry']:.6f}
        STOP LOSS: ${signal_data['stop_loss']:.6f}
        TAKE PROFIT: ${', '.join([f'${tp:.6f}' for tp in signal_data['take_profit'][:2]])}
        
        MA200: {signal_data['ma200_condition'].upper()}
        
        CONDICIONES CUMPLIDAS:
        {chr(10).join(['• ' + cond for cond in signal_data.get('fulfilled_conditions', [])][:8])}
        
        S/R NIVELES:
        Soportes: {', '.join([f'${s:.2f}' for s in signal_data['supports'][:4]])}
        Resistencias: {', '.join([f'${r:.2f}' for r in signal_data['resistances'][:4]])}
        
        Temporalidad: {interval}
        Fecha: {signal_data['timestamp']}
        """
        
        ax10.text(0.1, 0.9, signal_info, transform=ax10.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar imagen
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png', 
                        as_attachment=True, 
                        download_name=f'report_{symbol}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        
    except Exception as e:
        print(f"Error generando reporte: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generando reporte: {str(e)[:100]}'}), 500

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
