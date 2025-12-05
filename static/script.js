// Configuración global
let currentChart = null;
let currentScatterChart = null;
let currentWhaleChart = null;
let currentAdxChart = null;
let currentRsiTraditionalChart = null;
let currentRsiMaverickChart = null;
let currentMacdChart = null;
let currentTrendStrengthChart = null;
let currentVolumeChart = null;
let currentCmcVolumeChart = null;
let currentSymbol = 'BTC-USDT';
let currentData = null;
let allCryptos = [];
let updateInterval = null;

// Inicialización
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    updateCharts();
    startAutoUpdate();
});

function initializeApp() {
    console.log('MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializado');
    loadCryptoRiskClassification();
    updateCalendarInfo();
}

function setupEventListeners() {
    // Configurar event listeners
    document.getElementById('interval-select').addEventListener('change', updateCharts);
    document.getElementById('di-period').addEventListener('change', updateCharts);
    document.getElementById('adx-threshold').addEventListener('change', updateCharts);
    document.getElementById('sr-period').addEventListener('change', updateCharts);
    document.getElementById('rsi-length').addEventListener('change', updateCharts);
    document.getElementById('bb-multiplier').addEventListener('change', updateCharts);
    document.getElementById('leverage').addEventListener('change', updateCharts);
    
    // Configurar buscador
    setupCryptoSearch();
    
    // Configurar controles de indicadores
    setupIndicatorControls();
}

function updateCalendarInfo() {
    // Actualizar información del calendario
    const now = new Date();
    const dayOfWeek = now.toLocaleDateString('es-BO', { weekday: 'long' });
    const hour = now.getHours();
    const isTradingTime = hour >= 4 && hour < 16 && now.getDay() >= 1 && now.getDay() <= 5;
    
    const tradingStatus = isTradingTime ? 
        '<span class="text-success">🟢 ACTIVO</span>' : 
        '<span class="text-danger">🔴 INACTIVO</span>';
    
    document.getElementById('calendar-info').innerHTML = `
        <small class="text-muted">
            📅 ${dayOfWeek.charAt(0).toUpperCase() + dayOfWeek.slice(1)} | Trading Multi-TF: ${tradingStatus} | Horario: 4am-4pm L-V
        </small>
    `;
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
    const cryptoList = document.getElementById('crypto-list');
    
    searchInput.addEventListener('input', function() {
        const filter = this.value.toUpperCase();
        filterCryptoList(filter);
    });
    
    searchInput.addEventListener('click', function(e) {
        e.stopPropagation();
    });
}

function setupIndicatorControls() {
    const indicatorControls = document.querySelectorAll('.indicator-control');
    indicatorControls.forEach(control => {
        control.addEventListener('change', function() {
            updateChartIndicators();
        });
    });
}

function updateChartIndicators() {
    const showMA9 = document.getElementById('show-ma9').checked;
    const showMA21 = document.getElementById('show-ma21').checked;
    const showMA50 = document.getElementById('show-ma50').checked;
    const showMA200 = document.getElementById('show-ma200').checked;
    const showBB = document.getElementById('show-bollinger').checked;
    
    if (currentData && currentChart) {
        renderCandleChart(currentData, {
            showMA9: showMA9,
            showMA21: showMA21,
            showMA50: showMA50,
            showMA200: showMA200,
            showBB: showBB
        });
    }
}

function filterCryptoList(filter) {
    const cryptoList = document.getElementById('crypto-list');
    cryptoList.innerHTML = '';
    
    const filteredCryptos = allCryptos.filter(crypto => 
        crypto.symbol.toUpperCase().includes(filter)
    );
    
    if (filteredCryptos.length === 0) {
        cryptoList.innerHTML = `
            <div class="dropdown-item text-muted text-center">
                No se encontraron resultados
            </div>
        `;
        return;
    }
    
    // Agrupar por categoría
    const categories = {};
    filteredCryptos.forEach(crypto => {
        if (!categories[crypto.category]) {
            categories[crypto.category] = [];
        }
        categories[crypto.category].push(crypto);
    });
    
    // Mostrar por categorías
    Object.keys(categories).forEach(category => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'dropdown-header';
        
        let icon = '🟢';
        let className = 'text-success';
        if (category === 'medio') {
            icon = '🟡';
            className = 'text-warning';
        } else if (category === 'alto') {
            icon = '🔴';
            className = 'text-danger';
        } else if (category === 'memecoins') {
            icon = '🟣';
            className = 'text-info';
        }
        
        categoryDiv.innerHTML = `${icon} ${category.toUpperCase()} RIESGO`;
        categoryDiv.classList.add(className, 'small');
        cryptoList.appendChild(categoryDiv);
        
        categories[category].forEach(crypto => {
            const item = document.createElement('a');
            item.className = 'dropdown-item crypto-item';
            item.href = '#';
            item.innerHTML = crypto.symbol;
            item.addEventListener('click', function(e) {
                e.preventDefault();
                selectCrypto(crypto.symbol);
            });
            cryptoList.appendChild(item);
        });
        
        cryptoList.appendChild(document.createElement('div')).className = 'dropdown-divider';
    });
}

function selectCrypto(symbol) {
    currentSymbol = symbol;
    document.getElementById('selected-crypto').textContent = symbol;
    
    const dropdown = bootstrap.Dropdown.getInstance(document.getElementById('cryptoDropdown'));
    if (dropdown) {
        dropdown.hide();
    }
    
    updateCharts();
}

function loadCryptoRiskClassification() {
    fetch('/api/crypto_risk_classification')
        .then(response => response.json())
        .then(riskData => {
            allCryptos = [];
            Object.keys(riskData).forEach(category => {
                if (Array.isArray(riskData[category])) {
                    riskData[category].forEach(symbol => {
                        allCryptos.push({
                            symbol: symbol,
                            category: category
                        });
                    });
                }
            });
            
            filterCryptoList('');
        })
        .catch(error => {
            console.error('Error cargando clasificación:', error);
            loadBasicCryptoSymbols();
        });
}

function loadBasicCryptoSymbols() {
    const basicSymbols = [
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT', 'XRP-USDT'
    ];
    
    allCryptos = basicSymbols.map(symbol => ({
        symbol: symbol,
        category: 'bajo'
    }));
    
    filterCryptoList('');
}

function showLoadingState() {
    document.getElementById('market-summary').innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Cargando...</span>
            </div>
            <p class="mt-2 mb-0">Analizando mercado...</p>
        </div>
    `;
    
    document.getElementById('signal-analysis').innerHTML = `
        <div class="text-center py-3">
            <div class="spinner-border spinner-border-sm text-info" role="status">
                <span class="visually-hidden">Analizando...</span>
            </div>
            <p class="text-muted mb-0 small">Evaluando condiciones...</p>
        </div>
    `;
}

function startAutoUpdate() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    updateInterval = setInterval(() => {
        if (document.visibilityState === 'visible') {
            console.log('Actualización automática');
            updateCharts();
        }
    }, 90000);
}

function updateCharts() {
    showLoadingState();
    
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const diPeriod = document.getElementById('di-period').value;
    const adxThreshold = document.getElementById('adx-threshold').value;
    const srPeriod = document.getElementById('sr-period').value;
    const rsiLength = document.getElementById('rsi-length').value;
    const bbMultiplier = document.getElementById('bb-multiplier').value;
    const leverage = document.getElementById('leverage').value;
    
    // Actualizar gráficos principales
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    
    // Actualizar gráfico de dispersión
    updateScatterChart(interval);
    
    // Actualizar señales múltiples
    updateMultipleSignals(interval);
    
    // Actualizar señales de volumen
    updateVolumeSignals();
    
    // Actualizar alertas
    updateScalpingAlerts();
}

function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage) {
    const url = `/api/signals?symbol=${symbol}&interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            currentData = data;
            renderCandleChart(data);
            renderAdxChart(data);
            renderTrendStrengthChart(data);
            renderWhaleChart(data);
            renderRsiMaverickChart(data);
            renderRsiTraditionalChart(data);
            renderMacdChart(data);
            renderVolumeChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Error al cargar datos: ' + error.message);
            showSampleData(symbol);
        });
}

function showSampleData(symbol) {
    const sampleData = {
        symbol: symbol,
        current_price: 50000,
        signal: 'NEUTRAL',
        signal_score: 0,
        entry: 50000,
        stop_loss: 48000,
        take_profit: [52000],
        fulfilled_conditions: []
    };
    
    updateMarketSummary(sampleData);
    updateSignalAnalysis(sampleData);
}

function renderCandleChart(data, indicatorOptions = {}) {
    const chartElement = document.getElementById('candle-chart');
    
    if (!data.data || data.data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <h5>No hay datos disponibles</h5>
                <p>No se pudieron cargar los datos para el gráfico.</p>
                <button class="btn btn-sm btn-primary mt-2" onclick="updateCharts()">Reintentar</button>
            </div>
        `;
        return;
    }

    const dates = data.data.map(d => new Date(d.timestamp));
    const opens = data.data.map(d => parseFloat(d.open));
    const highs = data.data.map(d => parseFloat(d.high));
    const lows = data.data.map(d => parseFloat(d.low));
    const closes = data.data.map(d => parseFloat(d.close));
    
    // Traza de velas japonesas
    const candlestickTrace = {
        type: 'candlestick',
        x: dates,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        increasing: {line: {color: '#00C853'}, fillcolor: '#00C853'},
        decreasing: {line: {color: '#FF1744'}, fillcolor: '#FF1744'},
        name: 'Precio'
    };
    
    const traces = [candlestickTrace];
    
    // Añadir soportes y resistencias
    if (data.support_levels && data.resistance_levels) {
        // Soportes
        data.support_levels.forEach((level, index) => {
            traces.push({
                type: 'scatter',
                x: [dates[0], dates[dates.length - 1]],
                y: [level, level],
                mode: 'lines',
                line: {color: 'blue', dash: 'dash', width: 1},
                name: `Soporte ${index + 1}`
            });
        });
        
        // Resistencias
        data.resistance_levels.forEach((level, index) => {
            traces.push({
                type: 'scatter',
                x: [dates[0], dates[dates.length - 1]],
                y: [level, level],
                mode: 'lines',
                line: {color: 'red', dash: 'dash', width: 1},
                name: `Resistencia ${index + 1}`
            });
        });
    }
    
    // Añadir niveles de entrada y stop loss
    if (data.entry && data.stop_loss) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.entry, data.entry],
            mode: 'lines',
            line: {color: '#FFD700', dash: 'solid', width: 2},
            name: 'Entrada'
        });
        
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.stop_loss, data.stop_loss],
            mode: 'lines',
            line: {color: '#FF0000', dash: 'solid', width: 2},
            name: 'Stop Loss'
        });
    }
    
    // Añadir take profits
    if (data.take_profit && Array.isArray(data.take_profit)) {
        data.take_profit.forEach((tp, index) => {
            traces.push({
                type: 'scatter',
                x: [dates[0], dates[dates.length - 1]],
                y: [tp, tp],
                mode: 'lines',
                line: {color: '#00FF00', dash: 'dash', width: 1.5},
                name: `TP${index + 1}`
            });
        });
    }
    
    // Añadir indicadores informativos
    if (data.indicators) {
        const options = indicatorOptions || {
            showMA9: false,
            showMA21: false,
            showMA50: false,
            showMA200: false,
            showBB: false
        };
        
        // Medias móviles
        if (options.showMA9 && data.indicators.ma_9) {
            traces.push({
                type: 'scatter',
                x: dates.slice(-data.indicators.ma_9.length),
                y: data.indicators.ma_9,
                mode: 'lines',
                line: {color: '#FF9800', width: 1},
                name: 'MA 9'
            });
        }
        
        if (options.showMA21 && data.indicators.ma_21) {
            traces.push({
                type: 'scatter',
                x: dates.slice(-data.indicators.ma_21.length),
                y: data.indicators.ma_21,
                mode: 'lines',
                line: {color: '#2196F3', width: 1},
                name: 'MA 21'
            });
        }
        
        if (options.showMA50 && data.indicators.ma_50) {
            traces.push({
                type: 'scatter',
                x: dates.slice(-data.indicators.ma_50.length),
                y: data.indicators.ma_50,
                mode: 'lines',
                line: {color: '#9C27B0', width: 1},
                name: 'MA 50'
            });
        }
        
        if (options.showMA200 && data.indicators.ma_200) {
            traces.push({
                type: 'scatter',
                x: dates.slice(-data.indicators.ma_200.length),
                y: data.indicators.ma_200,
                mode: 'lines',
                line: {color: '#795548', width: 2},
                name: 'MA 200'
            });
        }
        
        // Bandas de Bollinger
        if (options.showBB && data.indicators.bb_upper && data.indicators.bb_lower) {
            const bbDates = dates.slice(-data.indicators.bb_upper.length);
            
            traces.push({
                type: 'scatter',
                x: bbDates,
                y: data.indicators.bb_upper,
                mode: 'lines',
                line: {color: 'rgba(255, 152, 0, 0.5)', width: 1},
                name: 'BB Superior',
                showlegend: false
            });
            
            traces.push({
                type: 'scatter',
                x: bbDates,
                y: data.indicators.bb_middle,
                mode: 'lines',
                line: {color: 'rgba(255, 152, 0, 0.7)', width: 1},
                name: 'BB Media',
                showlegend: false
            });
            
            traces.push({
                type: 'scatter',
                x: bbDates,
                y: data.indicators.bb_lower,
                mode: 'lines',
                line: {color: 'rgba(255, 152, 0, 0.5)', width: 1},
                name: 'BB Inferior',
                showlegend: false
            });
            
            // Rellenar entre bandas
            traces.push({
                type: 'scatter',
                x: bbDates.concat(bbDates.slice().reverse()),
                y: data.indicators.bb_upper.concat(data.indicators.bb_lower.slice().reverse()),
                fill: 'toself',
                fillcolor: 'rgba(255, 152, 0, 0.1)',
                line: {color: 'transparent'},
                name: 'Bandas Bollinger',
                showlegend: true
            });
        }
    }
    
    // Calcular rango dinámico para el eje Y
    const visibleHighs = highs.slice(-50);
    const visibleLows = lows.slice(-50);
    const minPrice = Math.min(...visibleLows);
    const maxPrice = Math.max(...visibleHighs);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.05;
    
    const interval = document.getElementById('interval-select').value;
    
    // Configuración del layout
    const layout = {
        title: {
            text: `${data.symbol} - ${interval}`,
            font: {color: '#ffffff', size: 16}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            rangeslider: {visible: false},
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Precio (USDT)',
            gridcolor: '#444',
            zerolinecolor: '#444',
            range: [minPrice - padding, maxPrice + padding]
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 80, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: `candlestick_${data.symbol}`,
            height: 600,
            width: 800,
            scale: 2
        }
    };
    
    if (currentChart) {
        Plotly.purge('candle-chart');
    }
    
    currentChart = Plotly.newPlot('candle-chart', traces, layout, config);
}

function renderAdxChart(data) {
    const chartElement = document.getElementById('adx-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de ADX disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const indicators = data.indicators;
    
    const traces = [
        {
            x: dates.slice(-indicators.adx.length),
            y: indicators.adx,
            type: 'scatter',
            mode: 'lines',
            name: 'ADX',
            line: {color: 'white', width: 2}
        },
        {
            x: dates.slice(-indicators.plus_di.length),
            y: indicators.plus_di,
            type: 'scatter',
            mode: 'lines',
            name: '+DI',
            line: {color: '#00C853', width: 1.5}
        },
        {
            x: dates.slice(-indicators.minus_di.length),
            y: indicators.minus_di,
            type: 'scatter',
            mode: 'lines',
            name: '-DI',
            line: {color: '#FF1744', width: 1.5}
        }
    ];
    
    // Añadir cruces DMI
    if (indicators.di_cross_bullish) {
        const bullishDates = [];
        const bullishValues = [];
        
        dates.slice(-indicators.di_cross_bullish.length).forEach((date, i) => {
            if (indicators.di_cross_bullish[i]) {
                bullishDates.push(date);
                bullishValues.push(indicators.plus_di[i]);
            }
        });
        
        if (bullishDates.length > 0) {
            traces.push({
                x: bullishDates,
                y: bullishValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Cruce Alcista',
                marker: {color: '#00FF00', size: 8, symbol: 'star'}
            });
        }
    }
    
    if (indicators.di_cross_bearish) {
        const bearishDates = [];
        const bearishValues = [];
        
        dates.slice(-indicators.di_cross_bearish.length).forEach((date, i) => {
            if (indicators.di_cross_bearish[i]) {
                bearishDates.push(date);
                bearishValues.push(indicators.minus_di[i]);
            }
        });
        
        if (bearishDates.length > 0) {
            traces.push({
                x: bearishDates,
                y: bearishValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Cruce Bajista',
                marker: {color: '#FF0000', size: 8, symbol: 'star'}
            });
        }
    }
    
    const layout = {
        title: {
            text: 'ADX con Indicadores Direccionales (+DI / -DI)',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Valor del Indicador',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentAdxChart) {
        Plotly.purge('adx-chart');
    }
    
    currentAdxChart = Plotly.newPlot('adx-chart', traces, layout, config);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    
    if (!data.indicators || !data.indicators.trend_strength) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Fuerza de Tendencia</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const indicators = data.indicators;
    
    // Crear barras con colores
    const traces = [{
        x: dates.slice(-indicators.trend_strength.length),
        y: indicators.trend_strength,
        type: 'bar',
        name: 'Fuerza de Tendencia',
        marker: {
            color: indicators.trend_colors || indicators.trend_strength.map(val => 
                val > 0 ? '#00C853' : '#FF1744'),
            line: {
                color: 'rgba(255,255,255,0.3)',
                width: 0.5
            }
        }
    }];
    
    // Añadir zonas de no operar
    if (indicators.no_trade_zones) {
        const noTradeDates = [];
        const noTradeValues = [];
        
        dates.slice(-indicators.no_trade_zones.length).forEach((date, i) => {
            if (indicators.no_trade_zones[i]) {
                noTradeDates.push(date);
                noTradeValues.push(indicators.trend_strength[i] || 0);
            }
        });
        
        if (noTradeDates.length > 0) {
            traces.push({
                x: noTradeDates,
                y: noTradeValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Zona NO OPERAR',
                marker: {
                    color: 'red',
                    size: 12,
                    symbol: 'x',
                    line: {
                        color: 'white',
                        width: 2
                    }
                }
            });
        }
    }
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Fuerza de Tendencia %',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', traces, layout, config);
}

function renderWhaleChart(data) {
    const chartElement = document.getElementById('whale-chart');
    
    if (!data.indicators || !data.indicators.whale_pump) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de ballenas</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const indicators = data.indicators;
    
    const traces = [
        {
            x: dates.slice(-indicators.whale_pump.length),
            y: indicators.whale_pump,
            type: 'bar',
            name: 'Ballenas Compradoras',
            marker: {color: '#00C853'}
        },
        {
            x: dates.slice(-indicators.whale_dump.length),
            y: indicators.whale_dump.map(x => -x), // Negativo para mostrar abajo
            type: 'bar',
            name: 'Ballenas Vendedoras',
            marker: {color: '#FF1744'}
        }
    ];
    
    // Añadir confirmaciones
    if (indicators.confirmed_buy) {
        const buyDates = [];
        const buyValues = [];
        
        dates.slice(-indicators.confirmed_buy.length).forEach((date, i) => {
            if (indicators.confirmed_buy[i]) {
                buyDates.push(date);
                buyValues.push(indicators.whale_pump[i]);
            }
        });
        
        if (buyDates.length > 0) {
            traces.push({
                x: buyDates,
                y: buyValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Compra Confirmada',
                marker: {color: '#00FF00', size: 10, symbol: 'diamond'}
            });
        }
    }
    
    if (indicators.confirmed_sell) {
        const sellDates = [];
        const sellValues = [];
        
        dates.slice(-indicators.confirmed_sell.length).forEach((date, i) => {
            if (indicators.confirmed_sell[i]) {
                sellDates.push(date);
                sellValues.push(-indicators.whale_dump[i]);
            }
        });
        
        if (sellDates.length > 0) {
            traces.push({
                x: sellDates,
                y: sellValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Venta Confirmada',
                marker: {color: '#FF0000', size: 10, symbol: 'diamond'}
            });
        }
    }
    
    const layout = {
        title: {
            text: 'Actividad de Ballenas - Compradoras vs Vendedoras',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Fuerza de Señal',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        barmode: 'overlay',
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentWhaleChart) {
        Plotly.purge('whale-chart');
    }
    
    currentWhaleChart = Plotly.newPlot('whale-chart', traces, layout, config);
}

function renderRsiMaverickChart(data) {
    const chartElement = document.getElementById('rsi-maverick-chart');
    
    if (!data.indicators || !data.indicators.rsi_maverick) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI Maverick</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const indicators = data.indicators;
    
    const traces = [
        {
            x: dates.slice(-indicators.rsi_maverick.length),
            y: indicators.rsi_maverick,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B)',
            line: {color: '#FF6B6B', width: 2}
        }
    ];
    
    // Añadir divergencias
    if (indicators.rsi_maverick_bullish_divergence) {
        const bullishDates = [];
        const bullishValues = [];
        
        dates.slice(-indicators.rsi_maverick_bullish_divergence.length).forEach((date, i) => {
            if (indicators.rsi_maverick_bullish_divergence[i]) {
                bullishDates.push(date);
                bullishValues.push(indicators.rsi_maverick[i]);
            }
        });
        
        if (bullishDates.length > 0) {
            traces.push({
                x: bullishDates,
                y: bullishValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Divergencia Alcista',
                marker: {
                    color: '#00FF00',
                    size: 12,
                    symbol: 'triangle-up'
                }
            });
        }
    }
    
    if (indicators.rsi_maverick_bearish_divergence) {
        const bearishDates = [];
        const bearishValues = [];
        
        dates.slice(-indicators.rsi_maverick_bearish_divergence.length).forEach((date, i) => {
            if (indicators.rsi_maverick_bearish_divergence[i]) {
                bearishDates.push(date);
                bearishValues.push(indicators.rsi_maverick[i]);
            }
        });
        
        if (bearishDates.length > 0) {
            traces.push({
                x: bearishDates,
                y: bearishValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Divergencia Bajista',
                marker: {
                    color: '#FF0000',
                    size: 12,
                    symbol: 'triangle-down'
                }
            });
        }
    }
    
    const layout = {
        title: {
            text: 'RSI Modificado Maverick - Bandas de Bollinger %B',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: '%B Value',
            range: [0, 1],
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        shapes: [
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 0.8,
                y1: 0.8,
                line: {color: 'red', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 0.2,
                y1: 0.2,
                line: {color: 'green', width: 1, dash: 'dash'}
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: -0.2,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 60, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentRsiMaverickChart) {
        Plotly.purge('rsi-maverick-chart');
    }
    
    currentRsiMaverickChart = Plotly.newPlot('rsi-maverick-chart', traces, layout, config);
}

function renderRsiTraditionalChart(data) {
    const chartElement = document.getElementById('rsi-traditional-chart');
    
    if (!data.indicators || !data.indicators.rsi_traditional) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI Tradicional</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const indicators = data.indicators;
    
    const traces = [
        {
            x: dates.slice(-indicators.rsi_traditional.length),
            y: indicators.rsi_traditional,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Tradicional',
            line: {color: '#2196F3', width: 2}
        }
    ];
    
    // Añadir divergencias
    if (indicators.rsi_bullish_divergence) {
        const bullishDates = [];
        const bullishValues = [];
        
        dates.slice(-indicators.rsi_bullish_divergence.length).forEach((date, i) => {
            if (indicators.rsi_bullish_divergence[i]) {
                bullishDates.push(date);
                bullishValues.push(indicators.rsi_traditional[i]);
            }
        });
        
        if (bullishDates.length > 0) {
            traces.push({
                x: bullishDates,
                y: bullishValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Divergencia Alcista',
                marker: {
                    color: '#00FF00',
                    size: 12,
                    symbol: 'triangle-up'
                }
            });
        }
    }
    
    if (indicators.rsi_bearish_divergence) {
        const bearishDates = [];
        const bearishValues = [];
        
        dates.slice(-indicators.rsi_bearish_divergence.length).forEach((date, i) => {
            if (indicators.rsi_bearish_divergence[i]) {
                bearishDates.push(date);
                bearishValues.push(indicators.rsi_traditional[i]);
            }
        });
        
        if (bearishDates.length > 0) {
            traces.push({
                x: bearishDates,
                y: bearishValues,
                type: 'scatter',
                mode: 'markers',
                name: 'Divergencia Bajista',
                marker: {
                    color: '#FF0000',
                    size: 12,
                    symbol: 'triangle-down'
                }
            });
        }
    }
    
    const layout = {
        title: {
            text: 'RSI Tradicional con Divergencias',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'RSI Value',
            range: [0, 100],
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        shapes: [
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 70,
                y1: 70,
                line: {color: 'red', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 30,
                y1: 30,
                line: {color: 'green', width: 1, dash: 'dash'}
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: -0.2,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 60, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentRsiTraditionalChart) {
        Plotly.purge('rsi-traditional-chart');
    }
    
    currentRsiTraditionalChart = Plotly.newPlot('rsi-traditional-chart', traces, layout, config);
}

function renderMacdChart(data) {
    const chartElement = document.getElementById('macd-chart');
    
    if (!data.indicators || !data.indicators.macd) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de MACD</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const indicators = data.indicators;
    
    const traces = [
        {
            x: dates.slice(-indicators.macd.length),
            y: indicators.macd,
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: {color: '#2196F3', width: 2}
        },
        {
            x: dates.slice(-indicators.macd_signal.length),
            y: indicators.macd_signal,
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#FF9800', width: 1.5}
        }
    ];
    
    // Histograma
    if (indicators.macd_histogram) {
        const histogramColors = indicators.macd_histogram.map(value => 
            value >= 0 ? 'rgba(0, 200, 83, 0.8)' : 'rgba(255, 23, 68, 0.8)'
        );
        
        traces.push({
            x: dates.slice(-indicators.macd_histogram.length),
            y: indicators.macd_histogram,
            type: 'bar',
            name: 'Histograma',
            marker: {color: histogramColors}
        });
    }
    
    const layout = {
        title: {
            text: 'MACD con Histograma',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'MACD Value',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentMacdChart) {
        Plotly.purge('macd-chart');
    }
    
    currentMacdChart = Plotly.newPlot('macd-chart', traces, layout, config);
}

function renderVolumeChart(data) {
    const chartElement = document.getElementById('volume-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Volumen</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const indicators = data.indicators;
    const volumes = data.data.slice(-50).map(d => parseFloat(d.volume));
    
    const traces = [];
    
    // Barras de volumen con colores
    if (indicators.volume_colors) {
        const volumeColors = indicators.volume_colors.slice(-volumes.length);
        
        traces.push({
            x: dates,
            y: volumes,
            type: 'bar',
            name: 'Volumen',
            marker: {color: volumeColors}
        });
    } else {
        traces.push({
            x: dates,
            y: volumes,
            type: 'bar',
            name: 'Volumen',
            marker: {color: 'rgba(128, 128, 128, 0.7)'}
        });
    }
    
    // SMA de volumen
    if (indicators.volume_sma) {
        traces.push({
            x: dates.slice(-indicators.volume_sma.length),
            y: indicators.volume_sma,
            type: 'scatter',
            mode: 'lines',
            name: 'SMA Volumen',
            line: {color: '#FFD700', width: 2}
        });
    }
    
    // Anomalías de volumen
    if (indicators.volume_anomaly) {
        const anomalyDates = [];
        const anomalyVolumes = [];
        
        dates.slice(-indicators.volume_anomaly.length).forEach((date, i) => {
            if (indicators.volume_anomaly[i]) {
                anomalyDates.push(date);
                anomalyVolumes.push(volumes[i]);
            }
        });
        
        if (anomalyDates.length > 0) {
            traces.push({
                x: anomalyDates,
                y: anomalyVolumes,
                type: 'scatter',
                mode: 'markers',
                name: 'Anomalías',
                marker: {
                    color: '#FF0000',
                    size: 10,
                    symbol: 'circle',
                    line: {color: 'white', width: 2}
                }
            });
        }
    }
    
    const layout = {
        title: {
            text: 'Indicador de Volumen con Anomalías',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Volumen',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentVolumeChart) {
        Plotly.purge('volume-chart');
    }
    
    currentVolumeChart = Plotly.newPlot('volume-chart', traces, layout, config);
}

function renderCmcVolumeChart() {
    const chartElement = document.getElementById('cmc-volume-chart');
    
    fetch('/api/cmc_volume_signals')
        .then(response => response.json())
        .then(data => {
            if (!data.signals || data.signals.length === 0) {
                chartElement.innerHTML = `
                    <div class="alert alert-secondary text-center">
                        <p class="mb-0">No hay señales de volumen agregado</p>
                    </div>
                `;
                return;
            }
            
            const signals = data.signals;
            const symbols = signals.map(s => s.symbol);
            const prices = signals.map(s => s.current_price);
            const volumeRatios = signals.map(s => s.volume_ratio);
            
            const trace1 = {
                x: symbols,
                y: prices,
                type: 'bar',
                name: 'Precio (USD)',
                yaxis: 'y',
                marker: {
                    color: prices.map((p, i) => 
                        signals[i].signal === 'LONG' ? '#00C853' : 
                        signals[i].signal === 'SHORT' ? '#FF1744' : '#9E9E9E'
                    )
                }
            };
            
            const trace2 = {
                x: symbols,
                y: volumeRatios,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Volumen Ratio',
                yaxis: 'y2',
                line: {color: '#FFD700', width: 2},
                marker: {size: 8}
            };
            
            const layout = {
                title: {
                    text: 'Señales Volumen Agregado - CoinMarketCap',
                    font: {color: '#ffffff', size: 14}
                },
                xaxis: {
                    title: 'Crypto',
                    type: 'category',
                    gridcolor: '#444',
                    zerolinecolor: '#444'
                },
                yaxis: {
                    title: 'Precio (USD)',
                    gridcolor: '#444',
                    zerolinecolor: '#444'
                },
                yaxis2: {
                    title: 'Volumen Ratio',
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: '#444',
                    zerolinecolor: '#444'
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#ffffff'},
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1.1,
                    orientation: 'h',
                    font: {color: '#ffffff'},
                    bgcolor: 'rgba(0,0,0,0)'
                },
                margin: {t: 60, r: 50, b: 50, l: 50}
            };
            
            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };
            
            if (currentCmcVolumeChart) {
                Plotly.purge('cmc-volume-chart');
            }
            
            currentCmcVolumeChart = Plotly.newPlot('cmc-volume-chart', [trace1, trace2], layout, config);
        })
        .catch(error => {
            console.error('Error cargando señales CMC:', error);
            chartElement.innerHTML = `
                <div class="alert alert-warning text-center">
                    <p class="mb-0">Error cargando datos de volumen agregado</p>
                </div>
            `;
        });
}

function updateMarketSummary(data) {
    if (!data) return;
    
    const signalColor = data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary';
    const signalIcon = data.signal === 'LONG' ? '📈' : data.signal === 'SHORT' ? '📉' : '⚖️';
    
    const marketSummary = document.getElementById('market-summary');
    marketSummary.innerHTML = `
        <div class="row g-2">
            <div class="col-12">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Señal:</span>
                    <span class="badge bg-${signalColor}">${signalIcon} ${data.signal}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Score:</span>
                    <span class="fw-bold ${data.signal_score >= 70 ? 'text-success' : data.signal_score >= 65 ? 'text-warning' : 'text-danger'}">
                        ${data.signal_score.toFixed(1)}%
                    </span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Precio:</span>
                    <span class="fw-bold">$${data.current_price.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Entrada:</span>
                    <span class="fw-bold">$${data.entry.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Stop Loss:</span>
                    <span class="fw-bold">$${data.stop_loss.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="text-muted small">Multi-TF:</span>
                    <span class="${data.multi_timeframe_ok ? 'text-success' : 'text-danger'}">
                        ${data.multi_timeframe_ok ? '✅' : '❌'}
                    </span>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    if (!data) return;
    
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (data.signal === 'NEUTRAL' || data.signal_score < 65) {
        signalAnalysis.innerHTML = `
            <div class="alert alert-secondary text-center py-2">
                <i class="fas fa-pause-circle me-2"></i>
                <strong>SEÑAL NEUTRAL</strong>
                <div class="small mt-1">Score: ${data.signal_score.toFixed(1)}%</div>
                <div class="small text-muted">Esperando mejores condiciones</div>
            </div>
        `;
    } else {
        const signalClass = data.signal === 'LONG' ? 'success' : 'danger';
        const signalIcon = data.signal === 'LONG' ? 'arrow-up' : 'arrow-down';
        
        let conditionsHTML = '';
        if (data.fulfilled_conditions && data.fulfilled_conditions.length > 0) {
            conditionsHTML = `
                <div class="mt-2">
                    <small class="text-muted d-block mb-1">Condiciones cumplidas:</small>
                    <div style="max-height: 100px; overflow-y: auto;">
                        ${data.fulfilled_conditions.map(cond => `
                            <div class="small text-success mb-1">✓ ${cond}</div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        signalAnalysis.innerHTML = `
            <div class="alert alert-${signalClass} text-center py-2">
                <i class="fas fa-${signalIcon} me-2"></i>
                <strong>SEÑAL ${data.signal} CONFIRMADA</strong>
                <div class="small mt-1">Score: ${data.signal_score.toFixed(1)}%</div>
            </div>
            
            <div class="mt-2">
                <div class="d-flex justify-content-between small mb-1">
                    <span>Entrada:</span>
                    <span class="fw-bold">$${data.entry.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between small mb-1">
                    <span>Stop Loss:</span>
                    <span class="fw-bold text-danger">$${data.stop_loss.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between small">
                    <span>Take Profit:</span>
                    <span class="fw-bold text-success">$${data.take_profit[0].toFixed(6)}</span>
                </div>
            </div>
            
            ${conditionsHTML}
        `;
    }
}

function updateScatterChart(interval) {
    fetch(`/api/scatter_data?interval=${interval}`)
        .then(response => response.json())
        .then(scatterData => {
            renderScatterChart(scatterData);
        })
        .catch(error => {
            console.error('Error cargando datos de scatter:', error);
        });
}

function renderScatterChart(scatterData) {
    const scatterElement = document.getElementById('scatter-chart');
    
    if (!scatterData || scatterData.length === 0) {
        scatterElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos disponibles para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }
    
    const traces = [{
        x: scatterData.map(d => d.x),
        y: scatterData.map(d => d.y),
        text: scatterData.map(d => 
            `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>Señal: ${d.signal}<br>Precio: $${formatPriceForDisplay(d.current_price)}`
        ),
        mode: 'markers',
        marker: {
            size: scatterData.map(d => 8 + (d.signal_score / 15)),
            color: scatterData.map(d => {
                if (d.signal === 'LONG') {
                    return d.risk_category === 'bajo' ? '#00C853' : 
                           d.risk_category === 'medio' ? '#FFC107' : 
                           d.risk_category === 'alto' ? '#FF9800' : '#9C27B0';
                }
                if (d.signal === 'SHORT') {
                    return d.risk_category === 'bajo' ? '#FF1744' : 
                           d.risk_category === 'medio' ? '#FF5252' : 
                           d.risk_category === 'alto' ? '#F44336' : '#E91E63';
                }
                return '#9E9E9E';
            }),
            opacity: scatterData.map(d => 0.6 + (d.signal_score / 250)),
            line: {
                color: 'white',
                width: 1
            }
        },
        type: 'scatter',
        hovertemplate: '%{text}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Análisis Multi-Indicador',
            font: {color: '#ffffff', size: 16}
        },
        xaxis: {
            title: 'Presión Compradora (%)',
            range: [0, 100],
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Presión Vendedora (%)',
            range: [0, 100],
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        shapes: [
            {type: 'line', x0: 33.3, x1: 33.3, y0: 0, y1: 100, line: {color: 'gray', width: 1, dash: 'dash'}},
            {type: 'line', x0: 66.6, x1: 66.6, y0: 0, y1: 100, line: {color: 'gray', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 100, y0: 33.3, y1: 33.3, line: {color: 'gray', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 100, y0: 66.6, y1: 66.6, line: {color: 'gray', width: 1, dash: 'dash'}},
            
            {
                type: 'rect', x0: 0, x1: 33.3, y0: 66.6, y1: 100,
                fillcolor: 'rgba(255, 0, 0, 0.15)',
                line: {width: 0}
            },
            {
                type: 'rect', x0: 66.6, x1: 100, y0: 0, y1: 33.3,
                fillcolor: 'rgba(0, 255, 0, 0.15)',
                line: {width: 0}
            }
        ],
        annotations: [
            {
                x: 16.65, y: 83.3,
                text: 'Zona VENTA',
                showarrow: false,
                font: {color: 'red', size: 12, weight: 'bold'},
                bgcolor: 'rgba(255, 0, 0, 0.3)'
            },
            {
                x: 83.3, y: 16.65,
                text: 'Zona COMPRA',
                showarrow: false,
                font: {color: 'green', size: 12, weight: 'bold'},
                bgcolor: 'rgba(0, 255, 0, 0.3)'
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 80, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        toImageButtonOptions: {
            format: 'png',
            filename: 'scatter_opportunities',
            height: 600,
            width: 800,
            scale: 2
        }
    };
    
    if (currentScatterChart) {
        Plotly.purge('scatter-chart');
    }
    
    currentScatterChart = Plotly.newPlot('scatter-chart', traces, layout, config);
}

function formatPriceForDisplay(price) {
    if (price >= 1000) {
        return price.toFixed(2);
    } else if (price >= 1) {
        return price.toFixed(4);
    } else if (price >= 0.01) {
        return price.toFixed(6);
    } else {
        return price.toFixed(8);
    }
}

function updateMultipleSignals(interval) {
    fetch(`/api/multiple_signals?interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            updateSignalsTables(data);
        })
        .catch(error => {
            console.error('Error cargando múltiples señales:', error);
        });
}

function updateSignalsTables(data) {
    // Actualizar tabla LONG
    const longTable = document.getElementById('long-table');
    if (data.long_signals && data.long_signals.length > 0) {
        longTable.innerHTML = data.long_signals.slice(0, 5).map((signal, index) => `
            <tr onclick="showSignalDetails('${signal.symbol}', '${signal.signal}')" style="cursor: pointer;" class="hover-row">
                <td class="text-center">${index + 1}</td>
                <td>
                    <strong>${signal.symbol}</strong>
                    <br><small class="text-success">Score: ${signal.signal_score.toFixed(1)}%</small>
                </td>
                <td class="text-center">
                    <span class="badge bg-success">${signal.signal_score.toFixed(0)}%</span>
                </td>
                <td class="text-end">$${formatPriceForDisplay(signal.entry)}</td>
            </tr>
        `).join('');
    } else {
        longTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    <i class="fas fa-search me-2"></i>No hay señales LONG confirmadas
                </td>
            </tr>
        `;
    }
    
    // Actualizar tabla SHORT
    const shortTable = document.getElementById('short-table');
    if (data.short_signals && data.short_signals.length > 0) {
        shortTable.innerHTML = data.short_signals.slice(0, 5).map((signal, index) => `
            <tr onclick="showSignalDetails('${signal.symbol}', '${signal.signal}')" style="cursor: pointer;" class="hover-row">
                <td class="text-center">${index + 1}</td>
                <td>
                    <strong>${signal.symbol}</strong>
                    <br><small class="text-danger">Score: ${signal.signal_score.toFixed(1)}%</small>
                </td>
                <td class="text-center">
                    <span class="badge bg-danger">${signal.signal_score.toFixed(0)}%</span>
                </td>
                <td class="text-end">$${formatPriceForDisplay(signal.entry)}</td>
            </tr>
        `).join('');
    } else {
        shortTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    <i class="fas fa-search me-2"></i>No hay señales SHORT confirmadas
                </td>
            </tr>
        `;
    }
}

function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const alertsElement = document.getElementById('scalping-alerts');
            if (data.alerts && data.alerts.length > 0) {
                alertsElement.innerHTML = data.alerts.slice(0, 3).map(alert => `
                    <div class="alert alert-warning scalping-alert mb-2 p-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <strong>${alert.symbol}</strong>
                                <div class="small">${alert.interval} - ${alert.signal}</div>
                                <div class="small">Score: ${alert.score.toFixed(1)}%</div>
                            </div>
                            <span class="badge bg-${alert.signal === 'LONG' ? 'success' : 'danger'}">
                                $${formatPriceForDisplay(alert.entry)}
                            </span>
                        </div>
                    </div>
                `).join('');
            } else {
                alertsElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="fas fa-bell-slash fa-2x mb-2"></i>
                        <div class="small">No hay alertas activas</div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando alertas:', error);
        });
}

function updateVolumeSignals() {
    fetch('/api/cmc_volume_signals')
        .then(response => response.json())
        .then(data => {
            const volumeElement = document.getElementById('volume-signals');
            if (data.signals && data.signals.length > 0) {
                volumeElement.innerHTML = data.signals.slice(0, 3).map(signal => `
                    <div class="alert alert-success mb-2 p-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <strong>${signal.symbol}</strong>
                                <div class="small">${signal.signal} - Volumen ${signal.volume_ratio.toFixed(1)}x</div>
                                <div class="small">${signal.reason}</div>
                            </div>
                            <span class="badge bg-${signal.signal === 'LONG' ? 'success' : 'danger'}">
                                $${signal.current_price.toFixed(2)}
                            </span>
                        </div>
                    </div>
                `).join('');
                
                // Actualizar gráfico CMC
                renderCmcVolumeChart();
            } else {
                volumeElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="fas fa-chart-area fa-2x mb-2"></i>
                        <div class="small">No hay señales de volumen</div>
                    </div>
                `;
                renderCmcVolumeChart();
            }
        })
        .catch(error => {
            console.error('Error cargando señales de volumen:', error);
        });
}

function showSignalDetails(symbol, signalType) {
    const modal = new bootstrap.Modal(document.getElementById('signalModal'));
    const detailsElement = document.getElementById('signal-details');
    
    detailsElement.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Cargando...</span>
            </div>
            <p class="mt-2 mb-0">Cargando detalles de ${symbol}...</p>
        </div>
    `;
    
    modal.show();
    
    const interval = document.getElementById('interval-select').value;
    const url = `/api/signals?symbol=${symbol}&interval=${interval}`;
    
    fetch(url)
        .then(response => response.json())
        .then(signalData => {
            const signalClass = signalType === 'LONG' ? 'success' : 'danger';
            const signalIcon = signalType === 'LONG' ? 'arrow-up' : 'arrow-down';
            
            let conditionsHTML = '';
            if (signalData.fulfilled_conditions && signalData.fulfilled_conditions.length > 0) {
                conditionsHTML = `
                    <hr>
                    <h6>Condiciones Cumplidas</h6>
                    <div style="max-height: 150px; overflow-y: auto;">
                        ${signalData.fulfilled_conditions.map(cond => `
                            <div class="small text-success mb-1">✓ ${cond}</div>
                        `).join('')}
                    </div>
                `;
            }
            
            let supportResistanceHTML = '';
            if (signalData.support_levels && signalData.resistance_levels) {
                supportResistanceHTML = `
                    <hr>
                    <h6>Soportes y Resistencias</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <h6 class="small">Soportes:</h6>
                            ${signalData.support_levels.map((level, index) => `
                                <div class="d-flex justify-content-between small mb-1">
                                    <span>S${index + 1}:</span>
                                    <span class="fw-bold text-info">$${level.toFixed(6)}</span>
                                </div>
                            `).join('')}
                        </div>
                        <div class="col-md-6">
                            <h6 class="small">Resistencias:</h6>
                            ${signalData.resistance_levels.map((level, index) => `
                                <div class="d-flex justify-content-between small mb-1">
                                    <span>R${index + 1}:</span>
                                    <span class="fw-bold text-info">$${level.toFixed(6)}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }
            
            detailsElement.innerHTML = `
                <h6>Detalles de Señal - ${symbol}</h6>
                <div class="alert alert-${signalClass} text-center py-2 mb-3">
                    <i class="fas fa-${signalIcon} me-2"></i>
                    <strong>SEÑAL ${signalType} CONFIRMADA</strong>
                    <div class="small mt-1">Score: ${signalData.signal_score.toFixed(1)}%</div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <h6>Niveles de Trading</h6>
                        <div class="d-flex justify-content-between small mb-1">
                            <span>Precio Actual:</span>
                            <span class="fw-bold">$${signalData.current_price.toFixed(6)}</span>
                        </div>
                        <div class="d-flex justify-content-between small mb-1">
                            <span>Entrada:</span>
                            <span class="fw-bold text-warning">$${signalData.entry.toFixed(6)}</span>
                        </div>
                        <div class="d-flex justify-content-between small mb-1">
                            <span>Stop Loss:</span>
                            <span class="fw-bold text-danger">$${signalData.stop_loss.toFixed(6)}</span>
                        </div>
                        <div class="d-flex justify-content-between small mb-1">
                            <span>Take Profit 1:</span>
                            <span class="fw-bold text-success">$${signalData.take_profit[0].toFixed(6)}</span>
                        </div>
                        ${signalData.take_profit.length > 1 ? `
                            <div class="d-flex justify-content-between small mb-1">
                                <span>Take Profit 2:</span>
                                <span class="fw-bold text-success">$${signalData.take_profit[1].toFixed(6)}</span>
                            </div>
                        ` : ''}
                        ${signalData.take_profit.length > 2 ? `
                            <div class="d-flex justify-content-between small mb-1">
                                <span>Take Profit 3:</span>
                                <span class="fw-bold text-success">$${signalData.take_profit[2].toFixed(6)}</span>
                            </div>
                        ` : ''}
                    </div>
                    <div class="col-md-6">
                        <h6>Configuración</h6>
                        <div class="d-flex justify-content-between small mb-1">
                            <span>Temporalidad:</span>
                            <span>${interval}</span>
                        </div>
                        <div class="d-flex justify-content-between small mb-1">
                            <span>Multi-TF:</span>
                            <span class="${signalData.multi_timeframe_ok ? 'text-success' : 'text-danger'}">
                                ${signalData.multi_timeframe_ok ? '✅ Confirmado' : '❌ No confirmado'}
                            </span>
                        </div>
                        <div class="d-flex justify-content-between small mb-1">
                            <span>MA200:</span>
                            <span class="${signalData.ma200_condition === 'above' ? 'text-success' : 'text-warning'}">
                                ${signalData.ma200_condition === 'above' ? 'ENCIMA' : 'DEBAJO'}
                            </span>
                        </div>
                        <div class="d-flex justify-content-between small">
                            <span>Leverage Sugerido:</span>
                            <span class="fw-bold">x${document.getElementById('leverage').value}</span>
                        </div>
                    </div>
                </div>
                
                ${conditionsHTML}
                ${supportResistanceHTML}
                
                <div class="mt-3 text-center">
                    <button class="btn btn-primary btn-sm" onclick="selectCrypto('${symbol}')">
                        <i class="fas fa-chart-line me-1"></i>Ver Gráficos
                    </button>
                    <button class="btn btn-success btn-sm ms-2" onclick="downloadReport()">
                        <i class="fas fa-download me-1"></i>Descargar Reporte
                    </button>
                </div>
            `;
        })
        .catch(error => {
            console.error('Error cargando detalles:', error);
            detailsElement.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Error cargando detalles: ${error.message}
                </div>
            `;
        });
}

function downloadReport() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}`;
    window.open(url, '_blank');
}

function showError(message) {
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'error-' + Date.now();
    
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-bg-danger border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-exclamation-triangle me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
    toast.show();
    
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}
