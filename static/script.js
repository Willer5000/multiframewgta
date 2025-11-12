// Configuración global
let currentChart = null;
let currentScatterChart = null;
let currentRsiTraditionalChart = null;
let currentRsiMaverickChart = null;
let currentMacdChart = null;
let currentAdxChart = null;
let currentTrendStrengthChart = null;
let currentAuxChart = null;
let currentSymbol = 'BTC-USDT';
let currentData = null;
let allCryptos = [];
let updateInterval = null;
let drawingToolsActive = false;

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
    loadMarketIndicators();
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
    document.getElementById('aux-indicator').addEventListener('change', updateAuxChart);
    
    // Configurar buscador de cryptos
    setupCryptoSearch();
    
    // Configurar herramientas de dibujo
    setupDrawingTools();
    
    // Configurar controles de indicadores
    setupIndicatorControls();
}

function updateCalendarInfo() {
    // Actualizar información del calendario y horario de scalping
    fetch('/api/bolivia_time')
        .then(response => response.json())
        .then(data => {
            const calendarInfo = document.getElementById('calendar-info');
            if (calendarInfo) {
                const scalpingStatus = data.is_scalping_time ? 
                    '<span class="text-success">🟢 ACTIVO</span>' : 
                    '<span class="text-danger">🔴 INACTIVO</span>';
                
                calendarInfo.innerHTML = `
                    <small class="text-muted">
                        📅 ${data.day_of_week} | Scalping 15m/30m: ${scalpingStatus} | Horario: 4am-4pm L-V
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

function setupDrawingTools() {
    // Inicializar herramientas de dibujo para cada gráfico
    const drawingButtons = document.querySelectorAll('.drawing-tool');
    drawingButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tool = this.dataset.tool;
            activateDrawingTool(tool);
        });
    });
    
    // Configurar selector de color
    const colorPicker = document.getElementById('drawing-color');
    if (colorPicker) {
        colorPicker.addEventListener('change', function() {
            setDrawingColor(this.value);
        });
    }
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

function activateDrawingTool(tool) {
    drawingToolsActive = true;
    
    // Remover clase activa de todos los botones
    document.querySelectorAll('.drawing-tool').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Activar botón seleccionado
    event.target.classList.add('active');
    
    // Configurar modo de dibujo según la herramienta
    const charts = ['candle-chart', 'rsi-traditional-chart', 'rsi-maverick-chart', 'macd-chart', 'adx-chart', 'trend-strength-chart', 'aux-chart'];
    
    charts.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart && chart.layout) {
            switch(tool) {
                case 'line':
                    chart.layout.dragmode = 'drawline';
                    break;
                case 'rectangle':
                    chart.layout.dragmode = 'drawrect';
                    break;
                case 'circle':
                    chart.layout.dragmode = 'drawcircle';
                    break;
                case 'text':
                    chart.layout.dragmode = 'drawtext';
                    break;
                case 'freehand':
                    chart.layout.dragmode = 'drawfreehand';
                    break;
                case 'marker':
                    chart.layout.dragmode = 'marker';
                    break;
                default:
                    chart.layout.dragmode = false;
            }
            
            if (currentChart) {
                Plotly.relayout(chartId, {dragmode: chart.layout.dragmode});
            }
        }
    });
}

function setDrawingColor(color) {
    // Configurar color para herramientas de dibujo
    const charts = ['candle-chart', 'rsi-traditional-chart', 'rsi-maverick-chart', 'macd-chart', 'adx-chart', 'trend-strength-chart', 'aux-chart'];
    
    charts.forEach(chartId => {
        if (currentChart) {
            Plotly.relayout(chartId, {
                'newshape.line.color': color,
                'newshape.fillcolor': color + '33'
            });
        }
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

function loadMarketIndicators() {
    // Cargar índice de miedo y codicia
    updateFearGreedIndex();
    
    // Cargar recomendaciones de mercado
    updateMarketRecommendations();
    
    // Cargar alertas de trading
    updateTradingAlerts();
    
    // Cargar señales de salida
    updateExitSignals();
    
    // Actualizar información del calendario
    updateCalendarInfo();
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
            updateMarketIndicators();
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
    
    // Actualizar gráfico de dispersión MEJORADO
    updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar señales múltiples
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar gráfico auxiliar
    updateAuxChart();
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateTradingAlerts();
    updateExitSignals();
    updateCalendarInfo();
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
            renderRsiTraditionalChart(data);
            renderRsiMaverickChart(data);
            renderMacdChart(data);
            renderAdxChart(data);
            renderTrendStrengthChart(data);
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
        support: 48000,
        resistance: 52000,
        volume: 1000000,
        volume_ma: 800000,
        adx: 30,
        plus_di: 35,
        minus_di: 25,
        whale_pump: 15,
        whale_dump: 10,
        rsi_traditional: 50,
        rsi_maverick: 0.5,
        winrate: 70.0,
        obligatory_long: false,
        obligatory_short: false,
        trend_strength_signal: 'NEUTRAL',
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
    
    // Añadir líneas de soporte y resistencia
    if (data.support && data.resistance) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.support, data.support],
            mode: 'lines',
            line: {color: 'blue', dash: 'dash', width: 2},
            name: 'Soporte'
        });
        
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.resistance, data.resistance],
            mode: 'lines',
            line: {color: 'red', dash: 'dash', width: 2},
            name: 'Resistencia'
        });
    }

    // Añadir niveles de entrada y take profits
    if (data.entry && data.take_profit) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.entry, data.entry],
            mode: 'lines',
            line: {color: '#FFD700', dash: 'solid', width: 2},
            name: 'Entrada'
        });
        
        // Añadir take profits
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
    
    // Añadir stop loss
    if (data.stop_loss) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.stop_loss, data.stop_loss],
            mode: 'lines',
            line: {color: '#FF0000', dash: 'dash', width: 2},
            name: 'Stop Loss'
        });
    }
    
    // Añadir indicadores informativos si están activados
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
                x: dates,
                y: data.indicators.ma_9,
                mode: 'lines',
                line: {color: '#FF9800', width: 1},
                name: 'MA 9'
            });
        }
        
        if (options.showMA21 && data.indicators.ma_21) {
            traces.push({
                type: 'scatter',
                x: dates,
                y: data.indicators.ma_21,
                mode: 'lines',
                line: {color: '#2196F3', width: 1},
                name: 'MA 21'
            });
        }
        
        if (options.showMA50 && data.indicators.ma_50) {
            traces.push({
                type: 'scatter',
                x: dates,
                y: data.indicators.ma_50,
                mode: 'lines',
                line: {color: '#9C27B0', width: 1},
                name: 'MA 50'
            });
        }
        
        if (options.showMA200 && data.indicators.ma_200) {
            traces.push({
                type: 'scatter',
                x: dates,
                y: data.indicators.ma_200,
                mode: 'lines',
                line: {color: '#795548', width: 1},
                name: 'MA 200'
            });
        }
        
        // Bandas de Bollinger
        if (options.showBB && data.indicators.bb_upper && data.indicators.bb_lower) {
            traces.push({
                type: 'scatter',
                x: dates,
                y: data.indicators.bb_upper,
                mode: 'lines',
                line: {color: 'rgba(255, 152, 0, 0.5)', width: 1},
                name: 'BB Superior',
                showlegend: false
            });
            
            traces.push({
                type: 'scatter',
                x: dates,
                y: data.indicators.bb_middle,
                mode: 'lines',
                line: {color: 'rgba(255, 152, 0, 0.7)', width: 1},
                name: 'BB Media',
                showlegend: false
            });
            
            traces.push({
                type: 'scatter',
                x: dates,
                y: data.indicators.bb_lower,
                mode: 'lines',
                line: {color: 'rgba(255, 152, 0, 0.5)', width: 1},
                name: 'BB Inferior',
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
    }
    
    // Calcular rango dinámico para el eje Y
    const visibleHighs = highs.slice(-50);
    const visibleLows = lows.slice(-50);
    const minPrice = Math.min(...visibleLows);
    const maxPrice = Math.max(...visibleHighs);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.05;
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas con Medias Móviles`,
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
        margin: {t: 80, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false,
        newshape: {
            line: {
                color: document.getElementById('drawing-color') ? document.getElementById('drawing-color').value : '#FFD700',
                width: 2
            }
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle', 'drawtext', 'drawfreehand'],
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

function renderRsiTraditionalChart(data) {
    const chartElement = document.getElementById('rsi-traditional-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI Tradicional disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const rsiTraditional = data.indicators.rsi_traditional || [];
    const bullishDivergence = data.indicators.rsi_traditional_bullish_div || [];
    const bearishDivergence = data.indicators.rsi_traditional_bearish_div || [];
    
    // Preparar datos para divergencias
    const bullishDates = [];
    const bullishValues = [];
    const bearishDates = [];
    const bearishValues = [];
    
    for (let i = 7; i < bullishDivergence.length; i++) {
        if (bullishDivergence[i] && !bullishDivergence[i-1] && !bullishDivergence[i-2]) {
            bullishDates.push(dates[i]);
            bullishValues.push(rsiTraditional[i]);
        }
        if (bearishDivergence[i] && !bearishDivergence[i-1] && !bearishDivergence[i-2]) {
            bearishDates.push(dates[i]);
            bearishValues.push(rsiTraditional[i]);
        }
    }
    
    const traces = [
        {
            x: dates,
            y: rsiTraditional,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Tradicional',
            line: {color: '#2196F3', width: 2}
        },
        {
            x: bullishDates,
            y: bullishValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Divergencia Alcista',
            marker: {
                color: '#00FF00',
                size: 12,
                symbol: 'triangle-up',
                line: {color: 'white', width: 1}
            }
        },
        {
            x: bearishDates,
            y: bearishValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Divergencia Bajista',
            marker: {
                color: '#FF0000',
                size: 12,
                symbol: 'triangle-down',
                line: {color: 'white', width: 1}
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Tradicional (14 periodos) con Detección de Divergencias',
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
                y0: 80,
                y1: 80,
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
                y0: 20,
                y1: 20,
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
        annotations: [
            {
                x: dates[dates.length - 1],
                y: 80,
                xanchor: 'left',
                text: 'Sobrecompra',
                showarrow: false,
                font: {color: 'red', size: 10}
            },
            {
                x: dates[dates.length - 1],
                y: 20,
                xanchor: 'left',
                text: 'Sobreventa',
                showarrow: false,
                font: {color: 'green', size: 10}
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
        margin: {t: 60, r: 50, b: 80, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentRsiTraditionalChart) {
        Plotly.purge('rsi-traditional-chart');
    }
    
    currentRsiTraditionalChart = Plotly.newPlot('rsi-traditional-chart', traces, layout, config);
}

function renderRsiMaverickChart(data) {
    const chartElement = document.getElementById('rsi-maverick-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI Maverick disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const rsiMaverick = data.indicators.rsi_maverick || [];
    const bullishDivergence = data.indicators.rsi_maverick_bullish_div || [];
    const bearishDivergence = data.indicators.rsi_maverick_bearish_div || [];
    
    // Preparar datos para divergencias
    const bullishDates = [];
    const bullishValues = [];
    const bearishDates = [];
    const bearishValues = [];
    
    for (let i = 7; i < bullishDivergence.length; i++) {
        if (bullishDivergence[i] && !bullishDivergence[i-1] && !bullishDivergence[i-2]) {
            bullishDates.push(dates[i]);
            bullishValues.push(rsiMaverick[i]);
        }
        if (bearishDivergence[i] && !bearishDivergence[i-1] && !bearishDivergence[i-2]) {
            bearishDates.push(dates[i]);
            bearishValues.push(rsiMaverick[i]);
        }
    }
    
    const traces = [
        {
            x: dates,
            y: rsiMaverick,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B)',
            line: {color: '#FF9800', width: 2}
        },
        {
            x: bullishDates,
            y: bullishValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Divergencia Alcista',
            marker: {
                color: '#00FF00',
                size: 12,
                symbol: 'triangle-up',
                line: {color: 'white', width: 1}
            }
        },
        {
            x: bearishDates,
            y: bearishValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Divergencia Bajista',
            marker: {
                color: '#FF0000',
                size: 12,
                symbol: 'triangle-down',
                line: {color: 'white', width: 1}
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Maverick - Bandas de Bollinger %B con Detección de Divergencias',
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
        annotations: [
            {
                x: dates[dates.length - 1],
                y: 0.8,
                xanchor: 'left',
                text: 'Sobrecompra',
                showarrow: false,
                font: {color: 'red', size: 10}
            },
            {
                x: dates[dates.length - 1],
                y: 0.2,
                xanchor: 'left',
                text: 'Sobreventa',
                showarrow: false,
                font: {color: 'green', size: 10}
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
        margin: {t: 60, r: 50, b: 80, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentRsiMaverickChart) {
        Plotly.purge('rsi-maverick-chart');
    }
    
    currentRsiMaverickChart = Plotly.newPlot('rsi-maverick-chart', traces, layout, config);
}

function renderMacdChart(data) {
    const chartElement = document.getElementById('macd-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de MACD disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const macdLine = data.indicators.macd_line || [];
    const macdSignal = data.indicators.macd_signal || [];
    const macdHistogram = data.indicators.macd_histogram || [];
    const macdBullish = data.indicators.macd_bullish || [];
    const macdBearish = data.indicators.macd_bearish || [];
    
    // Preparar histograma con colores
    const histogramColors = macdHistogram.map((value, index) => 
        value >= 0 ? 'rgba(0, 200, 83, 0.8)' : 'rgba(255, 23, 68, 0.8)'
    );
    
    const traces = [
        {
            x: dates,
            y: macdLine,
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: {color: '#2196F3', width: 2}
        },
        {
            x: dates,
            y: macdSignal,
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#FF9800', width: 1}
        },
        {
            x: dates,
            y: macdHistogram,
            type: 'bar',
            name: 'Histograma',
            marker: {color: histogramColors}
        },
        {
            x: dates.filter((_, i) => macdBullish[i]),
            y: macdLine.filter((_, i) => macdBullish[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Cruce Alcista',
            marker: {color: '#00FF00', size: 8, symbol: 'triangle-up'}
        },
        {
            x: dates.filter((_, i) => macdBearish[i]),
            y: macdLine.filter((_, i) => macdBearish[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Cruce Bajista',
            marker: {color: '#FF0000', size: 8, symbol: 'triangle-down'}
        }
    ];
    
    const layout = {
        title: {
            text: 'MACD (12,26,9) con Histograma y Cruces',
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
        shapes: [
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 0,
                y1: 0,
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
        margin: {t: 60, r: 50, b: 80, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false,
        barmode: 'group'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentMacdChart) {
        Plotly.purge('macd-chart');
    }
    
    currentMacdChart = Plotly.newPlot('macd-chart', traces, layout, config);
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
    const adx = data.indicators.adx || [];
    const plusDi = data.indicators.plus_di || [];
    const minusDi = data.indicators.minus_di || [];
    const diCrossBullish = data.indicators.di_cross_bullish || [];
    const diCrossBearish = data.indicators.di_cross_bearish || [];
    
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
            y: plusDi,
            type: 'scatter',
            mode: 'lines',
            name: '+DI',
            line: {color: '#00C853', width: 1.5}
        },
        {
            x: dates,
            y: minusDi,
            type: 'scatter',
            mode: 'lines',
            name: '-DI',
            line: {color: '#FF1744', width: 1.5}
        },
        {
            x: dates.filter((_, i) => diCrossBullish[i]),
            y: plusDi.filter((_, i) => diCrossBullish[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Cruce Alcista',
            marker: {color: '#00FF00', size: 8, symbol: 'star'}
        },
        {
            x: dates.filter((_, i) => diCrossBearish[i]),
            y: minusDi.filter((_, i) => diCrossBearish[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Cruce Bajista',
            marker: {color: '#FF0000', size: 8, symbol: 'star'}
        }
    ];
    
    const layout = {
        title: {
            text: 'ADX con Indicadores Direccionales (+DI / -DI) y Cruces',
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
        shapes: [
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 25,
                y1: 25,
                line: {
                    color: 'yellow',
                    width: 1,
                    dash: 'dash'
                }
            }
        ],
        annotations: [
            {
                x: dates[dates.length - 1],
                y: 25,
                xanchor: 'left',
                text: 'Umbral ADX',
                showarrow: false,
                font: {color: 'yellow', size: 10}
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
        margin: {t: 60, r: 50, b: 80, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentAdxChart) {
        Plotly.purge('adx-chart');
    }
    
    currentAdxChart = Plotly.newPlot('adx-chart', traces, layout, config);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Fuerza de Tendencia disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const trendStrength = data.indicators.trend_strength || [];
    const bbWidth = data.indicators.bb_width || [];
    const noTradeZones = data.indicators.no_trade_zones || [];
    const colors = data.indicators.colors || [];
    
    // Preparar datos para zonas de no operar
    const noTradeDates = [];
    const noTradeValues = [];
    
    for (let i = 0; i < noTradeZones.length; i++) {
        if (noTradeZones[i]) {
            noTradeDates.push(dates[i]);
            noTradeValues.push(trendStrength[i]);
        }
    }
    
    const traces = [
        {
            x: dates,
            y: trendStrength,
            type: 'bar',
            name: 'Fuerza de Tendencia',
            marker: {color: colors}
        },
        {
            x: noTradeDates,
            y: noTradeValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Zona NO OPERAR',
            marker: {
                color: '#FF0000',
                size: 12,
                symbol: 'x'
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick - Ancho de Bandas Bollinger %',
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
        shapes: [
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 0,
                y1: 0,
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
        margin: {t: 60, r: 50, b: 80, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', traces, layout, config);
}

function updateAuxChart() {
    const auxIndicator = document.getElementById('aux-indicator').value;
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    const url = `/api/signals?symbol=${symbol}&interval=${interval}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            switch(auxIndicator) {
                case 'rsi':
                    renderRsiTraditionalChart(data);
                    break;
                case 'macd':
                    renderMacdChart(data);
                    break;
                case 'squeeze':
                    renderSqueezeMomentumChart(data);
                    break;
                case 'stoch_rsi':
                    renderStochRsiChart(data);
                    break;
                default:
                    renderMacdChart(data);
            }
        })
        .catch(error => {
            console.error('Error actualizando gráfico auxiliar:', error);
        });
}

function renderSqueezeMomentumChart(data) {
    const chartElement = document.getElementById('aux-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Squeeze Momentum disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    
    // Simular datos de squeeze momentum (en producción real, calcular basado en datos)
    const squeezeMomentum = Array(50).fill(0).map((_, i) => Math.sin(i * 0.3) * 2);
    const squeezeSignal = Array(50).fill(0).map((_, i) => Math.cos(i * 0.3) * 1.5);
    
    const traces = [
        {
            x: dates,
            y: squeezeMomentum,
            type: 'scatter',
            mode: 'lines',
            name: 'Squeeze Momentum',
            line: {color: '#FF9800', width: 2}
        },
        {
            x: dates,
            y: squeezeSignal,
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#2196F3', width: 1}
        }
    ];
    
    const layout = {
        title: {
            text: 'Squeeze Momentum',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Momentum Value',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
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
        margin: {t: 60, r: 50, b: 80, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentAuxChart) {
        Plotly.purge('aux-chart');
    }
    
    currentAuxChart = Plotly.newPlot('aux-chart', traces, layout, config);
}

function renderStochRsiChart(data) {
    const chartElement = document.getElementById('aux-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI Estocástico disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    
    // Simular datos de RSI estocástico (en producción real, calcular basado en RSI)
    const stochRsiK = Array(50).fill(0).map((_, i) => 50 + Math.sin(i * 0.2) * 30);
    const stochRsiD = Array(50).fill(0).map((_, i) => 50 + Math.cos(i * 0.2) * 25);
    
    const traces = [
        {
            x: dates,
            y: stochRsiK,
            type: 'scatter',
            mode: 'lines',
            name: 'Stoch RSI %K',
            line: {color: '#2196F3', width: 2}
        },
        {
            x: dates,
            y: stochRsiD,
            type: 'scatter',
            mode: 'lines',
            name: 'Stoch RSI %D',
            line: {color: '#FF9800', width: 1}
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Estocástico',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Stoch RSI Value',
            range: [0, 100],
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        shapes: [
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 80,
                y1: 80,
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
                y0: 20,
                y1: 20,
                line: {
                    color: 'green',
                    width: 1,
                    dash: 'dash'
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
        margin: {t: 60, r: 50, b: 80, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentAuxChart) {
        Plotly.purge('aux-chart');
    }
    
    currentAuxChart = Plotly.newPlot('aux-chart', traces, layout, config);
}

function updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/scatter_data_improved?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}`;
    
    fetch(url)
        .then(response => response.json())
        .then(scatterData => {
            if (!Array.isArray(scatterData)) {
                throw new Error('Datos de scatter no válidos');
            }
            renderScatterChartImproved(scatterData);
        })
        .catch(error => {
            console.error('Error actualizando scatter chart:', error);
            renderScatterChartImproved([]);
        });
}

function renderScatterChartImproved(scatterData) {
    const chartElement = document.getElementById('scatter-chart');
    
    if (!Array.isArray(scatterData) || scatterData.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos disponibles para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }
    
    // Preparar datos por categoría de riesgo
    const riskCategories = {
        'bajo': {x: [], y: [], text: [], symbol: [], score: [], color: 'green'},
        'medio': {x: [], y: [], text: [], symbol: [], score: [], color: 'orange'},
        'alto': {x: [], y: [], text: [], symbol: [], score: [], color: 'red'},
        'memecoins': {x: [], y: [], text: [], symbol: [], score: [], color: 'purple'}
    };
    
    scatterData.forEach(item => {
        const category = item.risk_category || 'medio';
        if (riskCategories[category]) {
            riskCategories[category].x.push(item.x);
            riskCategories[category].y.push(item.y);
            riskCategories[category].text.push(`${item.symbol}<br>Score: ${item.signal_score || 0}%`);
            riskCategories[category].symbol.push(item.symbol);
            riskCategories[category].score.push(item.signal_score || 0);
        }
    });
    
    const traces = [];
    
    Object.keys(riskCategories).forEach(category => {
        const data = riskCategories[category];
        if (data.x.length > 0) {
            traces.push({
                x: data.x,
                y: data.y,
                text: data.text,
                mode: 'markers',
                type: 'scatter',
                name: category.toUpperCase() + ' RIESGO',
                marker: {
                    size: data.score.map(score => Math.max(8, score / 5)),
                    color: data.color,
                    opacity: 0.7,
                    line: {
                        color: 'white',
                        width: 1
                    }
                },
                hovertemplate: '<b>%{text}</b><br>Compra: %{x:.1f}%<br>Venta: %{y:.1f}%<extra></extra>'
            });
        }
    });
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Presión Compradora vs Vendedora',
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
            {
                type: 'line',
                x0: 70, x1: 70,
                y0: 0, y1: 100,
                line: {color: 'green', width: 2, dash: 'dash'}
            },
            {
                type: 'line',
                x0: 0, x1: 100,
                y0: 70, y1: 70,
                line: {color: 'red', width: 2, dash: 'dash'}
            }
        ],
        annotations: [
            {
                x: 85, y: 50,
                text: 'ZONA LONG<br>(Alta presión compradora)',
                showarrow: false,
                font: {color: 'green', size: 12},
                bgcolor: 'rgba(0,0,0,0.7)',
                borderpad: 4,
                bordercolor: 'green'
            },
            {
                x: 15, y: 85,
                text: 'ZONA SHORT<br>(Alta presión vendedora)',
                showarrow: false,
                font: {color: 'red', size: 12},
                bgcolor: 'rgba(0,0,0,0.7)',
                borderpad: 4,
                bordercolor: 'red'
            },
            {
                x: 50, y: 50,
                text: 'ZONA NEUTRAL',
                showarrow: false,
                font: {color: 'gray', size: 10},
                bgcolor: 'rgba(0,0,0,0.7)',
                borderpad: 4
            }
        ],
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
        margin: {t: 100, r: 50, b: 50, l: 50},
        hovermode: 'closest'
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

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/multiple_signals?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            updateSignalTables(data);
            updateTradingAlerts();
        })
        .catch(error => {
            console.error('Error actualizando señales múltiples:', error);
        });
}

function updateSignalTables(data) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    // Actualizar tabla LONG
    if (data.long_signals && data.long_signals.length > 0) {
        longTable.innerHTML = '';
        data.long_signals.slice(0, 5).forEach((signal, index) => {
            const row = document.createElement('tr');
            row.className = 'hover-row';
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-success">${signal.signal_score.toFixed(1)}%</span></td>
                <td>$${signal.entry.toFixed(4)}</td>
            `;
            row.addEventListener('click', function() {
                showSignalDetails(signal);
            });
            longTable.appendChild(row);
        });
    } else {
        longTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    No hay señales LONG confirmadas
                </td>
            </tr>
        `;
    }
    
    // Actualizar tabla SHORT
    if (data.short_signals && data.short_signals.length > 0) {
        shortTable.innerHTML = '';
        data.short_signals.slice(0, 5).forEach((signal, index) => {
            const row = document.createElement('tr');
            row.className = 'hover-row';
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-danger">${signal.signal_score.toFixed(1)}%</span></td>
                <td>$${signal.entry.toFixed(4)}</td>
            `;
            row.addEventListener('click', function() {
                showSignalDetails(signal);
            });
            shortTable.appendChild(row);
        });
    } else {
        shortTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    No hay señales SHORT confirmadas
                </td>
            </tr>
        `;
    }
}

function showSignalDetails(signal) {
    const modal = new bootstrap.Modal(document.getElementById('signalModal'));
    const modalBody = document.getElementById('signal-details');
    
    const signalTypeClass = signal.signal === 'LONG' ? 'text-success' : 'text-danger';
    const signalIcon = signal.signal === 'LONG' ? '🟢' : '🔴';
    
    modalBody.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h5 class="${signalTypeClass}">${signalIcon} ${signal.signal} - ${signal.symbol}</h5>
                <p><strong>Score:</strong> <span class="badge bg-primary">${signal.signal_score.toFixed(1)}%</span></p>
                <p><strong>Precio Actual:</strong> $${signal.current_price.toFixed(6)}</p>
                <p><strong>Entrada:</strong> $${signal.entry.toFixed(6)}</p>
                <p><strong>Stop Loss:</strong> $${signal.stop_loss.toFixed(6)}</p>
                <p><strong>Take Profit:</strong> $${signal.take_profit[0].toFixed(6)}</p>
            </div>
            <div class="col-md-6">
                <h6>Indicadores Técnicos</h6>
                <p><strong>RSI Tradicional:</strong> ${signal.rsi_traditional.toFixed(1)}</p>
                <p><strong>RSI Maverick:</strong> ${signal.rsi_maverick.toFixed(3)}</p>
                <p><strong>ADX:</strong> ${signal.adx.toFixed(1)}</p>
                <p><strong>+DI:</strong> ${signal.plus_di.toFixed(1)} | <strong>-DI:</strong> ${signal.minus_di.toFixed(1)}</p>
                <p><strong>Volumen:</strong> ${(signal.volume / 1000).toFixed(0)}K</p>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-12">
                <h6>Condiciones Cumplidas</h6>
                <ul class="list-group">
                    ${signal.fulfilled_conditions.map(condition => `
                        <li class="list-group-item bg-dark text-white border-secondary">
                            <i class="fas fa-check text-success me-2"></i>${condition}
                        </li>
                    `).join('')}
                </ul>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-12 text-center">
                <button class="btn btn-primary" onclick="selectCrypto('${signal.symbol}')">
                    <i class="fas fa-chart-line me-2"></i>Ver Gráfico
                </button>
                <button class="btn btn-success ms-2" onclick="downloadReport('${signal.symbol}')">
                    <i class="fas fa-download me-2"></i>Descargar Reporte
                </button>
            </div>
        </div>
    `;
    
    modal.show();
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    
    if (!data) {
        marketSummary.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos disponibles para el resumen</p>
            </div>
        `;
        return;
    }
    
    const signalClass = data.signal === 'LONG' ? 'text-success' : 
                       data.signal === 'SHORT' ? 'text-danger' : 'text-warning';
    const signalIcon = data.signal === 'LONG' ? '🟢' : 
                      data.signal === 'SHORT' ? '🔴' : '🟡';
    
    marketSummary.innerHTML = `
        <div class="row text-center">
            <div class="col-md-4 mb-3">
                <div class="card bg-dark border-secondary h-100">
                    <div class="card-body">
                        <h6 class="card-title text-muted">Señal Actual</h6>
                        <h4 class="${signalClass}">${signalIcon} ${data.signal}</h4>
                        <div class="progress mt-2" style="height: 10px;">
                            <div class="progress-bar ${data.signal === 'LONG' ? 'bg-success' : data.signal === 'SHORT' ? 'bg-danger' : 'bg-warning'}" 
                                 style="width: ${data.signal_score}%"></div>
                        </div>
                        <small class="text-muted">Score: ${data.signal_score.toFixed(1)}%</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-3">
                <div class="card bg-dark border-secondary h-100">
                    <div class="card-body">
                        <h6 class="card-title text-muted">Precio</h6>
                        <h4 class="text-primary">$${data.current_price.toFixed(4)}</h4>
                        <div class="mt-2">
                            <small class="text-muted d-block">Soporte: $${data.support.toFixed(4)}</small>
                            <small class="text-muted d-block">Resistencia: $${data.resistance.toFixed(4)}</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-3">
                <div class="card bg-dark border-secondary h-100">
                    <div class="card-body">
                        <h6 class="card-title text-muted">Volumen</h6>
                        <h6 class="${data.volume > data.volume_ma ? 'text-success' : 'text-danger'}">
                            ${(data.volume / 1000).toFixed(0)}K
                        </h6>
                        <small class="text-muted">Promedio: ${(data.volume_ma / 1000).toFixed(0)}K</small>
                        <div class="mt-2">
                            <small class="text-muted d-block">ATR: ${data.atr.toFixed(4)}</small>
                            <small class="text-muted d-block">(${(data.atr_percentage * 100).toFixed(2)}%)</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (!data) {
        signalAnalysis.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos para análisis</p>
            </div>
        `;
        return;
    }
    
    const signalClass = data.signal === 'LONG' ? 'alert-success' : 
                       data.signal === 'SHORT' ? 'alert-danger' : 'alert-warning';
    
    signalAnalysis.innerHTML = `
        <div class="alert ${signalClass}">
            <h6 class="alert-heading">
                ${data.signal === 'LONG' ? '🟢' : data.signal === 'SHORT' ? '🔴' : '🟡'} 
                Análisis de Señal - ${data.symbol}
            </h6>
            <hr>
            <div class="row">
                <div class="col-6">
                    <small><strong>Entrada:</strong><br>$${data.entry.toFixed(6)}</small>
                </div>
                <div class="col-6">
                    <small><strong>Stop Loss:</strong><br>$${data.stop_loss.toFixed(6)}</small>
                </div>
            </div>
            <div class="row mt-2">
                <div class="col-6">
                    <small><strong>Take Profit:</strong><br>$${data.take_profit[0].toFixed(6)}</small>
                </div>
                <div class="col-6">
                    <small><strong>Winrate:</strong><br>${data.winrate.toFixed(1)}%</small>
                </div>
            </div>
            ${data.fulfilled_conditions.length > 0 ? `
                <hr>
                <small><strong>Condiciones Cumplidas:</strong></small>
                <ul class="small mb-0 mt-1">
                    ${data.fulfilled_conditions.slice(0, 3).map(condition => `
                        <li>${condition}</li>
                    `).join('')}
                </ul>
            ` : ''}
        </div>
    `;
}

function updateFearGreedIndex() {
    const fearGreedElement = document.getElementById('fear-greed-index');
    
    // Simular datos del índice (en producción real, obtener de API)
    const fearGreedValue = Math.floor(Math.random() * 100);
    let status = '';
    let color = '';
    let icon = '';
    
    if (fearGreedValue >= 75) {
        status = 'EXTREMA CODICIA';
        color = 'danger';
        icon = '😱';
    } else if (fearGreedValue >= 55) {
        status = 'CODICIA';
        color = 'warning';
        icon = '😊';
    } else if (fearGreedValue >= 45) {
        status = 'NEUTRAL';
        color = 'info';
        icon = '😐';
    } else if (fearGreedValue >= 25) {
        status = 'MIEDO';
        color = 'primary';
        icon = '😟';
    } else {
        status = 'MIEDO EXTREMO';
        color = 'success';
        icon = '😨';
    }
    
    fearGreedElement.innerHTML = `
        <div class="text-center">
            <h3 class="text-${color}">${fearGreedValue}</h3>
            <p class="mb-1">${icon} ${status}</p>
            <div class="progress mt-2">
                <div class="progress-bar bg-${color}" style="width: ${fearGreedValue}%"></div>
            </div>
            <small class="text-muted mt-2 d-block">Índice de Miedo y Codicia del Mercado</small>
        </div>
    `;
}

function updateMarketRecommendations() {
    const recommendationsElement = document.getElementById('market-recommendations');
    
    // Simular recomendaciones (en producción real, basado en análisis)
    const recommendations = [
        { type: 'success', text: 'Mercado alcista en temporalidades mayores' },
        { type: 'warning', text: 'Volatilidad moderada - Usar stops ajustados' },
        { type: 'info', text: 'Ballenas activas en BTC y ETH' }
    ];
    
    recommendationsElement.innerHTML = `
        <div class="card bg-dark border-secondary">
            <div class="card-header">
                <h6 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Recomendaciones</h6>
            </div>
            <div class="card-body">
                ${recommendations.map(rec => `
                    <div class="alert alert-${rec.type} alert-sm mb-2 py-2">
                        <small class="mb-0">${rec.text}</small>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function updateTradingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const alertsElement = document.getElementById('scalping-alerts');
            
            if (data.alerts && data.alerts.length > 0) {
                alertsElement.innerHTML = '';
                data.alerts.slice(0, 5).forEach(alert => {
                    const alertElement = document.createElement('div');
                    alertElement.className = `alert ${alert.signal === 'LONG' ? 'alert-success' : 'alert-danger'} mb-2 py-2 scalping-alert`;
                    alertElement.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${alert.symbol}</strong>
                                <small class="d-block">${alert.interval} - Score: ${alert.score.toFixed(1)}%</small>
                            </div>
                            <div class="text-end">
                                <small>Entrada: $${alert.entry.toFixed(4)}</small>
                                <small class="d-block">Leverage: x${alert.leverage}</small>
                            </div>
                        </div>
                    `;
                    alertElement.addEventListener('click', function() {
                        selectCrypto(alert.symbol);
                        document.getElementById('interval-select').value = alert.interval;
                        updateCharts();
                    });
                    alertsElement.appendChild(alertElement);
                });
            } else {
                alertsElement.innerHTML = `
                    <div class="text-center py-3">
                        <i class="fas fa-bell-slash text-muted fa-2x mb-2"></i>
                        <p class="text-muted mb-0 small">No hay alertas activas</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando alertas:', error);
        });
}

function updateExitSignals() {
    fetch('/api/exit_signals')
        .then(response => response.json())
        .then(data => {
            const exitSignalsElement = document.getElementById('exit-signals');
            
            if (data.exit_signals && data.exit_signals.length > 0) {
                exitSignalsElement.innerHTML = '';
                data.exit_signals.slice(0, 5).forEach(signal => {
                    const signalElement = document.createElement('div');
                    const pnlClass = signal.pnl_percent >= 0 ? 'text-success' : 'text-danger';
                    const pnlIcon = signal.pnl_percent >= 0 ? '📈' : '📉';
                    
                    signalElement.className = 'alert alert-warning mb-2 py-2';
                    signalElement.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${signal.symbol}</strong>
                                <small class="d-block">${signal.interval} - CERRAR ${signal.signal}</small>
                                <small class="d-block text-muted">${signal.reason}</small>
                            </div>
                            <div class="text-end">
                                <small class="${pnlClass}">${pnlIcon} ${signal.pnl_percent.toFixed(2)}%</small>
                                <small class="d-block text-muted">${signal.timestamp.split(' ')[1]}</small>
                            </div>
                        </div>
                    `;
                    exitSignalsElement.appendChild(signalElement);
                });
            } else {
                exitSignalsElement.innerHTML = `
                    <div class="text-center py-3">
                        <i class="fas fa-check-circle text-muted fa-2x mb-2"></i>
                        <p class="text-muted mb-0 small">No hay señales de salida</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando señales de salida:', error);
        });
}

function downloadReport(symbol = null) {
    const targetSymbol = symbol || currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${targetSymbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

function showError(message) {
    // Crear toast de error
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'error-toast-' + Date.now();
    
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
    
    toastContainer.innerHTML += toastHTML;
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remover el toast después de que se oculte
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

function showSuccess(message) {
    // Crear toast de éxito
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'success-toast-' + Date.now();
    
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-bg-success border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-check-circle me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.innerHTML += toastHTML;
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remover el toast después de que se oculte
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

// Función para manejar la visibilidad de la página
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') {
        // Reanudar actualizaciones automáticas cuando la página vuelve a ser visible
        startAutoUpdate();
    } else {
        // Pausar actualizaciones automáticas cuando la página no es visible
        if (updateInterval) {
            clearInterval(updateInterval);
            updateInterval = null;
        }
    }
});

// Exportar funciones para uso global
window.selectCrypto = selectCrypto;
window.updateCharts = updateCharts;
window.downloadReport = downloadReport;
window.downloadStrategicReport = downloadStrategicReport;
