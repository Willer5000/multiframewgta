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
let currentSymbol = 'BTC-USDT';
let currentData = null;
let allCryptos = [];
let updateInterval = null;

// Inicialización cuando el DOM está listo
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
    // Configurar event listeners para los controles
    document.getElementById('interval-select').addEventListener('change', updateCharts);
    document.getElementById('di-period').addEventListener('change', updateCharts);
    document.getElementById('adx-threshold').addEventListener('change', updateCharts);
    document.getElementById('sr-period').addEventListener('change', updateCharts);
    document.getElementById('rsi-length').addEventListener('change', updateCharts);
    document.getElementById('bb-multiplier').addEventListener('change', updateCharts);
    document.getElementById('volume-filter').addEventListener('change', updateCharts);
    document.getElementById('leverage').addEventListener('change', updateCharts);
    
    // Configurar buscador de cryptos
    setupCryptoSearch();
    
    // Configurar controles de indicadores
    setupIndicatorControls();
}

function updateCalendarInfo() {
    // Actualizar información del calendario y horario de trading
    fetch('/api/bolivia_time')
        .then(response => response.json())
        .then(data => {
            const calendarInfo = document.getElementById('calendar-info');
            if (calendarInfo) {
                const now = new Date();
                const dayOfWeek = now.toLocaleDateString('es-BO', { weekday: 'long' });
                const hour = now.getHours();
                const isTradingTime = hour >= 4 && hour < 16 && now.getDay() >= 1 && now.getDay() <= 5;
                
                const tradingStatus = isTradingTime ? 
                    '<span class="text-success">🟢 ACTIVO</span>' : 
                    '<span class="text-danger">🔴 INACTIVO</span>';
                
                calendarInfo.innerHTML = `
                    <small class="text-muted">
                        📅 ${dayOfWeek.charAt(0).toUpperCase() + dayOfWeek.slice(1)} | Trading Multi-TF: ${tradingStatus} | Horario: 4am-4pm L-V
                    </small>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando información del calendario:', error);
        });
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
    const cryptoList = document.getElementById('crypto-list');
    
    searchInput.addEventListener('input', function() {
        const filter = this.value.toUpperCase();
        filterCryptoList(filter);
    });
    
    // Prevenir que el dropdown se cierre al hacer clic en el buscador
    searchInput.addEventListener('click', function(e) {
        e.stopPropagation();
    });
}

function setupIndicatorControls() {
    // Configurar controles de indicadores informativos
    const indicatorControls = document.querySelectorAll('.indicator-control');
    indicatorControls.forEach(control => {
        control.addEventListener('change', function() {
            updateChartIndicators();
        });
    });
}

function updateChartIndicators() {
    // Actualizar indicadores en el gráfico principal
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
    
    // Cerrar el dropdown
    const dropdown = document.getElementById('crypto-dropdown-menu');
    const bootstrapDropdown = bootstrap.Dropdown.getInstance(document.getElementById('cryptoDropdown'));
    if (bootstrapDropdown) {
        bootstrapDropdown.hide();
    }
    
    updateCharts();
}

function loadCryptoRiskClassification() {
    fetch('/api/crypto_risk_classification')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(riskData => {
            if (typeof riskData !== 'object' || riskData === null) {
                throw new Error('Datos de riesgo no válidos');
            }
            
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
            console.error('Error cargando clasificación de riesgo:', error);
            loadBasicCryptoSymbols();
        });
}

function loadBasicCryptoSymbols() {
    const basicSymbols = [
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT', 'XRP-USDT',
        'ADA-USDT', 'AVAX-USDT', 'DOT-USDT', 'LINK-USDT', 'DOGE-USDT'
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
            <p class="text-muted mb-0 small">Evaluando condiciones multi-temporalidad...</p>
        </div>
    `;
}

function startAutoUpdate() {
    // Detener intervalo anterior si existe
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Configurar actualización automática cada 90 segundos
    updateInterval = setInterval(() => {
        if (document.visibilityState === 'visible') {
            console.log('Actualización automática (cada 90 segundos)');
            updateCharts();
            updateCalendarInfo();
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
    const volumeFilter = document.getElementById('volume-filter').value;
    const leverage = document.getElementById('leverage').value;
    
    // Actualizar gráfico principal
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar gráfico de dispersión
    updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar señales múltiples
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar alertas de volumen atípico
    updateVolumeAnomalyAlerts();
}

function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/signals?symbol=${symbol}&interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            currentData = data;
            renderCandleChart(data);
            renderAdxChartImproved(data);
            renderTrendStrengthChart(data);
            renderWhaleChartImproved(data);
            renderRsiMaverickChart(data);
            renderRsiTraditionalChart(data);
            renderMacdChart(data);
            renderVolumeChartImproved(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Error al cargar datos del gráfico: ' + error.message);
            showSampleData(symbol);
        });
}

function showSampleData(symbol) {
    // Datos de ejemplo para cuando falle la API
    const sampleData = {
        symbol: symbol,
        current_price: 50000,
        signal: 'NEUTRAL',
        signal_score: 0,
        entry: 50000,
        stop_loss: 48000,
        take_profit: [52000],
        supports: [48000, 47500, 47000, 46500],
        resistances: [52000, 52500, 53000, 53500],
        volume: 1000000,
        adx: 25,
        plus_di: 30,
        minus_di: 20,
        whale_pump: 15,
        whale_dump: 10,
        rsi_maverick: 0.5,
        rsi_traditional: 50,
        ma200_condition: 'above',
        fulfilled_conditions: []
    };
    
    updateMarketSummary(sampleData);
    updateSignalAnalysis(sampleData);
}

function renderCandleChart(data, indicatorOptions = {}) {
    const chartElement = document.getElementById('candle-chart');
    
    if (!data.indicators || data.indicators.close.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <h5>No hay datos disponibles</h5>
                <p>No se pudieron cargar los datos para el gráfico.</p>
                <button class="btn btn-sm btn-primary mt-2" onclick="updateCharts()">Reintentar</button>
            </div>
        `;
        return;
    }

    const closes = data.indicators.close;
    const dates = Array.from({length: closes.length}, (_, i) => new Date(Date.now() - (closes.length - i - 1) * 3600000));
    
    // Traza de precio
    const priceTrace = {
        type: 'scatter',
        x: dates,
        y: closes,
        mode: 'lines',
        name: 'Precio',
        line: {color: 'white', width: 2}
    };
    
    const traces = [priceTrace];
    
    // Añadir Bandas de Bollinger si está activado
    if (indicatorOptions.showBB && data.indicators.bb_upper && data.indicators.bb_lower) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.bb_upper,
            mode: 'lines',
            name: 'BB Superior',
            line: {color: 'rgba(255, 152, 0, 0.5)', width: 1},
            showlegend: false
        });
        
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.bb_middle,
            mode: 'lines',
            name: 'BB Media',
            line: {color: 'rgba(255, 152, 0, 0.7)', width: 1},
            showlegend: false
        });
        
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.bb_lower,
            mode: 'lines',
            name: 'BB Inferior',
            line: {color: 'rgba(255, 152, 0, 0.5)', width: 1},
            showlegend: false
        });
        
        // Rellenar entre bandas
        traces.push({
            type: 'scatter',
            x: dates.concat(dates.slice().reverse()),
            y: data.indicators.bb_upper.concat(data.indicators.bb_lower.slice().reverse()),
            fill: 'toself',
            fillcolor: 'rgba(255, 152, 0, 0.1)',
            line: {color: 'transparent'},
            name: 'Bandas Bollinger',
            showlegend: true
        });
    }
    
    // Añadir medias móviles si están activadas
    if (indicatorOptions.showMA9 && data.indicators.ma_9) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_9,
            mode: 'lines',
            name: 'MA 9',
            line: {color: '#FF9800', width: 1}
        });
    }
    
    if (indicatorOptions.showMA21 && data.indicators.ma_21) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_21,
            mode: 'lines',
            name: 'MA 21',
            line: {color: '#2196F3', width: 1}
        });
    }
    
    if (indicatorOptions.showMA50 && data.indicators.ma_50) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_50,
            mode: 'lines',
            name: 'MA 50',
            line: {color: '#9C27B0', width: 1}
        });
    }
    
    if (indicatorOptions.showMA200 && data.indicators.ma_200) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_200,
            mode: 'lines',
            name: 'MA 200',
            line: {color: '#795548', width: 3}
        });
    }
    
    // Añadir soportes y resistencias (4 niveles cada uno)
    if (data.supports && data.supports.length >= 4) {
        data.supports.slice(0, 4).forEach((support, index) => {
            traces.push({
                type: 'scatter',
                x: [dates[0], dates[dates.length - 1]],
                y: [support, support],
                mode: 'lines',
                line: {color: '#00FF00', dash: 'dash', width: 1},
                name: `S${index + 1}`,
                showlegend: index === 0
            });
        });
    }
    
    if (data.resistances && data.resistances.length >= 4) {
        data.resistances.slice(0, 4).forEach((resistance, index) => {
            traces.push({
                type: 'scatter',
                x: [dates[0], dates[dates.length - 1]],
                y: [resistance, resistance],
                mode: 'lines',
                line: {color: '#FF0000', dash: 'dash', width: 1},
                name: `R${index + 1}`,
                showlegend: index === 0
            });
        });
    }
    
    // Añadir niveles de trading si hay señal
    if (data.signal !== 'NEUTRAL' && data.signal_score >= 65) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.entry, data.entry],
            mode: 'lines',
            line: {color: '#FFD700', width: 2, dash: 'solid'},
            name: 'Entrada'
        });
        
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.stop_loss, data.stop_loss],
            mode: 'lines',
            line: {color: '#FF4444', width: 1.5, dash: 'dash'},
            name: 'Stop Loss'
        });
        
        data.take_profit.slice(0, 2).forEach((tp, index) => {
            traces.push({
                type: 'scatter',
                x: [dates[0], dates[dates.length - 1]],
                y: [tp, tp],
                mode: 'lines',
                line: {color: '#00FF00', width: 1.5, dash: 'dash'},
                name: `TP${index + 1}`
            });
        });
    }
    
    // Calcular rango dinámico
    const visiblePrices = closes.slice(-50);
    const minPrice = Math.min(...visiblePrices);
    const maxPrice = Math.max(...visiblePrices);
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
            range: [minPrice - padding, maxPrice + padding],
            fixedrange: false
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
    
    // Destruir gráfico existente
    if (currentChart) {
        Plotly.purge('candle-chart');
    }
    
    currentChart = Plotly.newPlot('candle-chart', traces, layout, config);
}

function renderAdxChartImproved(data) {
    const chartElement = document.getElementById('adx-chart');
    
    if (!data.indicators || !data.indicators.adx) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de ADX disponibles</p>
            </div>
        `;
        return;
    }

    const adx = data.indicators.adx;
    const plus_di = data.indicators.plus_di;
    const minus_di = data.indicators.minus_di;
    const dates = Array.from({length: adx.length}, (_, i) => new Date(Date.now() - (adx.length - i - 1) * 3600000));
    
    const traces = [
        {
            x: dates,
            y: adx,
            type: 'scatter',
            mode: 'lines',
            name: 'ADX',
            line: {color: 'white', width: 2}
        },
        {
            x: dates,
            y: plus_di,
            type: 'scatter',
            mode: 'lines',
            name: '+DI',
            line: {color: '#00C853', width: 1.5}
        },
        {
            x: dates,
            y: minus_di,
            type: 'scatter',
            mode: 'lines',
            name: '-DI',
            line: {color: '#FF1744', width: 1.5}
        }
    ];
    
    const layout = {
        title: {
            text: 'ADX con Indicadores Direccionales (+DI / -DI)',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
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
                <p class="mb-0">No hay datos de Fuerza de Tendencia disponibles</p>
            </div>
        `;
        return;
    }

    const trend_strength = data.indicators.trend_strength;
    const no_trade_zones = data.indicators.no_trade_zones || [];
    const colors = data.indicators.colors || [];
    const dates = Array.from({length: trend_strength.length}, (_, i) => new Date(Date.now() - (trend_strength.length - i - 1) * 3600000));
    
    // Crear barras con colores individuales
    const traces = [{
        x: dates,
        y: trend_strength,
        type: 'bar',
        name: 'Fuerza de Tendencia',
        marker: {
            color: colors,
            line: {
                color: 'rgba(255,255,255,0.3)',
                width: 0.5
            }
        }
    }];
    
    // Añadir marcadores para zonas de no operar
    const noTradeDates = [];
    const noTradeValues = [];
    
    dates.forEach((date, i) => {
        if (no_trade_zones[i]) {
            noTradeDates.push(date);
            noTradeValues.push(trend_strength[i] || 0);
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
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick - Ancho Bandas Bollinger %',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
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

function renderWhaleChartImproved(data) {
    const chartElement = document.getElementById('whale-chart');
    
    if (!data.indicators || !data.indicators.whale_pump) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de ballenas disponibles</p>
            </div>
        `;
        return;
    }

    const whale_pump = data.indicators.whale_pump;
    const whale_dump = data.indicators.whale_dump;
    const confirmed_buy = data.indicators.confirmed_buy || [];
    const confirmed_sell = data.indicators.confirmed_sell || [];
    const dates = Array.from({length: whale_pump.length}, (_, i) => new Date(Date.now() - (whale_pump.length - i - 1) * 3600000));
    
    const traces = [
        {
            x: dates,
            y: whale_pump,
            type: 'bar',
            name: 'Ballenas Compradoras',
            marker: {color: '#00C853'}
        },
        {
            x: dates,
            y: whale_dump,
            type: 'bar',
            name: 'Ballenas Vendedoras',
            marker: {color: '#FF1744'}
        }
    ];
    
    // Añadir marcadores para confirmaciones
    const buyDates = [];
    const buyValues = [];
    const sellDates = [];
    const sellValues = [];
    
    dates.forEach((date, i) => {
        if (confirmed_buy[i]) {
            buyDates.push(date);
            buyValues.push(whale_pump[i]);
        }
        if (confirmed_sell[i]) {
            sellDates.push(date);
            sellValues.push(whale_dump[i]);
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
    
    const layout = {
        title: {
            text: 'Actividad de Ballenas - Compradoras vs Vendedoras',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
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
        bargap: 0,
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
                <p class="mb-0">No hay datos de RSI Maverick disponibles</p>
            </div>
        `;
        return;
    }

    const rsi_maverick = data.indicators.rsi_maverick;
    const dates = Array.from({length: rsi_maverick.length}, (_, i) => new Date(Date.now() - (rsi_maverick.length - i - 1) * 3600000));
    
    const traces = [
        {
            x: dates,
            y: rsi_maverick,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B)',
            line: {color: '#FF6B6B', width: 2}
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Modificado Maverick - Bandas de Bollinger %B',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
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
                line: {
                    color: 'red',
                    width: 1,
                    dash: 'dash'
                }
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 0.2,
                y1: 0.2,
                line: {
                    color: 'green',
                    width: 1,
                    dash: 'dash'
                }
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 0.5,
                y1: 0.5,
                line: {
                    color: 'white',
                    width: 1,
                    dash: 'solid'
                }
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
                <p class="mb-0">No hay datos de RSI Tradicional disponibles</p>
            </div>
        `;
        return;
    }

    const rsi_traditional = data.indicators.rsi_traditional;
    const dates = Array.from({length: rsi_traditional.length}, (_, i) => new Date(Date.now() - (rsi_traditional.length - i - 1) * 3600000));
    
    const traces = [
        {
            x: dates,
            y: rsi_traditional,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Tradicional',
            line: {color: '#2196F3', width: 2}
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Tradicional con Divergencias (14 Periodos)',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
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
                line: {
                    color: 'red',
                    width: 1,
                    dash: 'dash'
                }
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 30,
                y1: 30,
                line: {
                    color: 'green',
                    width: 1,
                    dash: 'dash'
                }
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 50,
                y1: 50,
                line: {
                    color: 'white',
                    width: 1,
                    dash: 'solid'
                }
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
                <p class="mb-0">No hay datos de MACD disponibles</p>
            </div>
        `;
        return;
    }

    const macd = data.indicators.macd;
    const macd_signal = data.indicators.macd_signal;
    const macd_histogram = data.indicators.macd_histogram;
    const dates = Array.from({length: macd.length}, (_, i) => new Date(Date.now() - (macd.length - i - 1) * 3600000));
    
    // Colores para el histograma
    const histogramColors = macd_histogram.map(value => 
        value >= 0 ? 'rgba(0, 200, 83, 0.8)' : 'rgba(255, 23, 68, 0.8)'
    );
    
    const traces = [
        {
            x: dates,
            y: macd,
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: {color: '#2196F3', width: 2}
        },
        {
            x: dates,
            y: macd_signal,
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#FF9800', width: 1.5}
        },
        {
            x: dates,
            y: macd_histogram,
            type: 'bar',
            name: 'Histograma',
            marker: {color: histogramColors}
        }
    ];
    
    const layout = {
        title: {
            text: 'MACD con Histograma',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
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

function renderVolumeChartImproved(data) {
    const chartElement = document.getElementById('volume-chart');
    
    if (!data.indicators || !data.indicators.volume) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Volumen disponibles</p>
            </div>
        `;
        return;
    }

    const volume = data.indicators.volume;
    const volume_anomaly = data.indicators.volume_anomaly || [];
    const volume_clusters = data.indicators.volume_clusters || [];
    const volume_direction = data.indicators.volume_direction || [];
    const volume_colors = data.indicators.volume_colors || [];
    const dates = Array.from({length: volume.length}, (_, i) => new Date(Date.now() - (volume.length - i - 1) * 3600000));
    
    // Crear barras de volumen con colores según dirección
    const traces = [{
        x: dates,
        y: volume,
        type: 'bar',
        name: 'Volumen',
        marker: {
            color: volume_colors.length === volume.length ? volume_colors : 'gray',
            opacity: 0.7
        }
    }];
    
    // Añadir anomalías de volumen
    const anomalyDates = [];
    const anomalyVolumes = [];
    const anomalyColors = [];
    
    dates.forEach((date, i) => {
        if (volume_anomaly[i]) {
            anomalyDates.push(date);
            anomalyVolumes.push(volume[i]);
            anomalyColors.push(volume_direction[i] === 1 ? '#00FF00' : volume_direction[i] === -1 ? '#FF0000' : '#FFA500');
        }
    });
    
    if (anomalyDates.length > 0) {
        traces.push({
            x: anomalyDates,
            y: anomalyVolumes,
            type: 'scatter',
            mode: 'markers',
            name: 'Anomalías Volumen',
            marker: {
                color: anomalyColors,
                size: 12,
                symbol: 'circle',
                line: {color: 'white', width: 2}
            }
        });
    }
    
    // Añadir clusters de volumen
    const clusterDates = [];
    const clusterVolumes = [];
    
    dates.forEach((date, i) => {
        if (volume_clusters[i]) {
            clusterDates.push(date);
            clusterVolumes.push(volume[i]);
        }
    });
    
    if (clusterDates.length > 0) {
        traces.push({
            x: clusterDates,
            y: clusterVolumes,
            type: 'scatter',
            mode: 'markers',
            name: 'Clusters Volumen',
            marker: {
                color: '#8A2BE2',
                size: 14,
                symbol: 'diamond',
                line: {color: 'white', width: 2}
            }
        });
    }
    
    const layout = {
        title: {
            text: 'Indicador de Volumen con Anomalías y Clusters',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
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
                    <span class="fw-bold text-warning">$${data.entry.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">ADX:</span>
                    <span class="${data.adx >= 25 ? 'text-success' : 'text-warning'}">${data.adx.toFixed(1)}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="text-muted small">MA200:</span>
                    <span class="${data.ma200_condition === 'above' ? 'text-success' : 'text-warning'}">
                        ${data.ma200_condition === 'above' ? 'ENCIMA' : 'DEBAJO'}
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
        
        signalAnalysis.innerHTML = `
            <div class="alert alert-${signalClass} text-center py-2">
                <i class="fas fa-${signalIcon} me-2"></i>
                <strong>SEÑAL ${data.signal} CONFIRMADA</strong>
                <div class="small mt-1">Score: ${data.signal_score.toFixed(1)}%</div>
                <div class="small">MA200: ${data.ma200_condition.toUpperCase()}</div>
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
                    <span>Take Profit 1:</span>
                    <span class="fw-bold text-success">$${data.take_profit[0].toFixed(6)}</span>
                </div>
            </div>
            
            ${data.fulfilled_conditions && data.fulfilled_conditions.length > 0 ? `
                <div class="mt-2">
                    <small class="text-muted d-block mb-1">Condiciones cumplidas:</small>
                    <div style="max-height: 100px; overflow-y: auto;">
                        ${data.fulfilled_conditions.map(cond => `
                            <div class="small text-success mb-1">✓ ${cond}</div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }
}

function updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/scatter_data_improved?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(scatterData => {
            renderScatterChartImproved(scatterData);
        })
        .catch(error => {
            console.error('Error cargando datos de scatter:', error);
        });
}

function renderScatterChartImproved(scatterData) {
    const scatterElement = document.getElementById('scatter-chart');
    
    if (!scatterData || scatterData.length === 0) {
        scatterElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos disponibles para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }
    
    // Calcular valores para colores basados en señal real
    const traces = [{
        x: scatterData.map(d => d.x),
        y: scatterData.map(d => d.y),
        text: scatterData.map(d => 
            `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>Señal: ${d.signal}<br>Precio: $${formatPriceForDisplay(d.current_price)}<br>Riesgo: ${d.risk_category}`
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
            },
            symbol: scatterData.map(d => {
                if (d.risk_category === 'bajo') return 'circle';
                if (d.risk_category === 'medio') return 'square';
                if (d.risk_category === 'alto') return 'diamond';
                return 'star';
            })
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
            zerolinecolor: '#444',
            showgrid: true
        },
        yaxis: {
            title: 'Presión Vendedora (%)',
            range: [0, 100],
            gridcolor: '#444',
            zerolinecolor: '#444',
            showgrid: true
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
                bgcolor: 'rgba(255, 0, 0, 0.3)',
                bordercolor: 'red'
            },
            {
                x: 83.3, y: 16.65,
                text: 'Zona COMPRA',
                showarrow: false,
                font: {color: 'green', size: 12, weight: 'bold'},
                bgcolor: 'rgba(0, 255, 0, 0.3)',
                bordercolor: 'green'
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

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/multiple_signals?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
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

function updateVolumeAnomalyAlerts() {
    fetch('/api/volume_anomaly_signals')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const alertsElement = document.getElementById('volume-anomaly-alerts');
            if (data.alerts && data.alerts.length > 0) {
                alertsElement.innerHTML = data.alerts.slice(0, 3).map(alert => `
                    <div class="alert ${alert.direction === 'COMPRA' ? 'alert-success' : 'alert-danger'} mb-2 p-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <strong>${alert.symbol}</strong>
                                <div class="small">${alert.direction} ATÍPICA</div>
                                <div class="small">${alert.amount_millions}M USDT</div>
                            </div>
                            <span class="badge ${alert.direction === 'COMPRA' ? 'bg-success' : 'bg-danger'}">
                                ${alert.direction === 'COMPRA' ? 'LONG' : 'SHORT'}
                            </span>
                        </div>
                    </div>
                `).join('');
            } else {
                alertsElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="fas fa-chart-bar fa-2x mb-2"></i>
                        <div class="small">No hay alertas de volumen atípico</div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando alertas de volumen:', error);
            document.getElementById('volume-anomaly-alerts').innerHTML = `
                <div class="text-center text-muted py-3">
                    <div class="small">Error cargando alertas</div>
                </div>
            `;
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
            if (signalData.error) {
                throw new Error(signalData.error);
            }
            
            const signalClass = signalType === 'LONG' ? 'success' : 'danger';
            const signalIcon = signalType === 'LONG' ? 'arrow-up' : 'arrow-down';
            
            detailsElement.innerHTML = `
                <h6>Detalles de Señal - ${symbol}</h6>
                <div class="alert alert-${signalClass} text-center py-2 mb-3">
                    <i class="fas fa-${signalIcon} me-2"></i>
                    <strong>SEÑAL ${signalType} CONFIRMADA</strong>
                    <div class="small mt-1">Score: ${signalData.signal_score.toFixed(1)}% | MA200: ${signalData.ma200_condition.toUpperCase()}</div>
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
                    </div>
                    <div class="col-md-6">
                        <h6>Soportes y Resistencias</h6>
                        ${signalData.supports.slice(0, 4).map((support, index) => `
                            <div class="d-flex justify-content-between small mb-1">
                                <span>Soporte ${index + 1}:</span>
                                <span class="fw-bold text-info">$${support.toFixed(6)}</span>
                            </div>
                        `).join('')}
                        ${signalData.resistances.slice(0, 4).map((resistance, index) => `
                            <div class="d-flex justify-content-between small mb-1">
                                <span>Resistencia ${index + 1}:</span>
                                <span class="fw-bold text-info">$${resistance.toFixed(6)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                ${signalData.fulfilled_conditions && signalData.fulfilled_conditions.length > 0 ? `
                    <hr>
                    <h6>Condiciones Cumplidas</h6>
                    <div style="max-height: 150px; overflow-y: auto;">
                        ${signalData.fulfilled_conditions.map(cond => `
                            <div class="small text-success mb-1">✓ ${cond}</div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <div class="mt-3 text-center">
                    <button class="btn btn-primary btn-sm" onclick="selectCrypto('${symbol}')">
                        <i class="fas fa-chart-line me-1"></i>Ver Gráficos
                    </button>
                    <button class="btn btn-success btn-sm ms-2" onclick="downloadReportForSymbol('${symbol}')">
                        <i class="fas fa-download me-1"></i>Descargar Reporte
                    </button>
                </div>
            `;
        })
        .catch(error => {
            console.error('Error cargando detalles de señal:', error);
            detailsElement.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Error cargando detalles: ${error.message}
                </div>
                <div class="text-center mt-3">
                    <button class="btn btn-primary btn-sm" onclick="selectCrypto('${symbol}')">
                        <i class="fas fa-chart-line me-1"></i>Ver Gráficos
                    </button>
                </div>
            `;
        });
}

function downloadReportForSymbol(symbol) {
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

function downloadReport() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
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
