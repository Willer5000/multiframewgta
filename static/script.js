// Configuración global
let currentChart = null;
let currentScatterChart = null;
let currentWhaleChart = null;
let currentAdxChart = null;
let currentRsiChart = null;
let currentAuxChart = null;
let currentTrendStrengthChart = null;
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
    updateWinRateDisplay();
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
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'aux-chart', 'trend-strength-chart'];
    
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
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'aux-chart', 'trend-strength-chart'];
    
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

// Función para cargar clasificación de riesgo
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
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'XRP-USDT', 'ADA-USDT',
        'SOL-USDT', 'DOT-USDT', 'DOGE-USDT', 'AVAX-USDT', 'LINK-USDT'
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
    
    // Cargar alertas de scalping
    updateScalpingAlerts();
    
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
            <p class="text-muted mb-0 small">Evaluando condiciones de señal...</p>
        </div>
    `;
    
    document.getElementById('multi-tf-analysis').innerHTML = `
        <div class="text-center py-3">
            <div class="spinner-border spinner-border-sm text-teal" role="status">
                <span class="visually-hidden">Analizando...</span>
            </div>
            <p class="text-muted mb-0 small">Verificando temporalidades...</p>
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
            updateWinRateDisplay();
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
    
    // Actualizar gráfico auxiliar
    updateAuxChart();
    
    // Actualizar winrate
    updateWinRateDisplay();
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateScalpingAlerts();
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
            renderWhaleChartCorrected(data, interval);
            renderAdxChartImproved(data);
            renderRsiMaverickChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
            updateMultiTFAnalysis(data);
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
        win_rate: 50.0,
        entry: 50000,
        stop_loss: 48000,
        take_profit: [52000],
        support: 48000,
        resistance: 52000,
        volume: 1000000,
        volume_ma: 800000,
        adx: 25,
        plus_di: 30,
        minus_di: 20,
        whale_pump: 15,
        whale_dump: 10,
        rsi_maverick: 0.5,
        multi_tf_analysis: {
            ok: true,
            details: ['Análisis multi-TF completado']
        },
        fulfilled_conditions: []
    };
    
    updateMarketSummary(sampleData);
    updateSignalAnalysis(sampleData);
    updateMultiTFAnalysis(sampleData);
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
            text: `${data.symbol} - Gráfico de Velas Japonesas | Multi-Temporalidad`,
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

function renderWhaleChartCorrected(data, interval) {
    const chartElement = document.getElementById('whale-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de ballenas disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const whalePump = data.indicators.whale_pump || [];
    const whaleDump = data.indicators.whale_dump || [];
    
    // Actualizar estado del indicador
    const isObligatory = interval === '12h' || interval === '1D';
    const statusElement = document.getElementById('whale-indicator-status');
    if (statusElement) {
        statusElement.textContent = isObligatory ? 'Obligatorio en 12H/1D' : 'Visible en todas las TF';
        statusElement.className = isObligatory ? 'badge bg-warning ms-2' : 'badge bg-info ms-2';
    }
    
    const traces = [
        {
            x: dates,
            y: whalePump,
            type: 'bar',
            name: 'Ballenas Compradoras',
            marker: {color: '#00C853'}
        },
        {
            x: dates,
            y: whaleDump,
            type: 'bar',
            name: 'Ballenas Vendedoras',
            marker: {color: '#FF1744'}
        }
    ];
    
    const layout = {
        title: {
            text: `Actividad de Ballenas - ${isObligatory ? 'SEÑAL OBLIGATORIA' : 'Indicador visible'}`,
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
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentWhaleChart) {
        Plotly.purge('whale-chart');
    }
    
    currentWhaleChart = Plotly.newPlot('whale-chart', traces, layout, config);
}

function renderAdxChartImproved(data) {
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
        margin: {t: 60, r: 50, b: 50, l: 50},
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
    
    const traces = [
        {
            x: dates,
            y: rsiMaverick,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B)',
            line: {color: '#2196F3', width: 2}
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
    
    if (currentRsiChart) {
        Plotly.purge('rsi-maverick-chart');
    }
    
    currentRsiChart = Plotly.newPlot('rsi-maverick-chart', traces, layout, config);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    
    if (!data.indicators || !data.data || !data.indicators.trend_strength) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Fuerza de Tendencia disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const trendStrength = data.indicators.trend_strength || [];
    const noTradeZones = data.indicators.no_trade_zones || [];
    
    // Crear colores basados en la fuerza de tendencia
    const colors = trendStrength.map((strength, index) => {
        if (noTradeZones[index]) {
            return '#FF0000'; // Rojo para zonas de no operar
        } else if (strength > 2) {
            return '#00C853'; // Verde para tendencia alcista fuerte
        } else if (strength > 0) {
            return '#FFC107'; // Amarillo para tendencia alcista débil
        } else if (strength < -2) {
            return '#FF1744'; // Rojo para tendencia bajista fuerte
        } else if (strength < 0) {
            return '#FF9800'; // Naranja para tendencia bajista débil
        } else {
            return '#9E9E9E'; // Gris para neutral
        }
    });
    
    // Crear barras con colores individuales
    const traces = [{
        x: dates,
        y: trendStrength,
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
        if (noTradeZones[i]) {
            noTradeDates.push(date);
            noTradeValues.push(trendStrength[i] || 0);
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
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false,
        annotations: [
            {
                x: 0.02,
                y: 0.98,
                xref: 'paper',
                yref: 'paper',
                text: '🟢 Verde: Fuerza creciente | 🔴 Rojo: Fuerza decreciente',
                showarrow: false,
                font: {color: 'white', size: 10},
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: 'rgba(255,255,255,0.5)',
                borderwidth: 1,
                borderpad: 4
            }
        ]
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
    const indicatorType = document.getElementById('aux-indicator').value;
    
    if (indicatorType === 'multi_tf') {
        renderMultiTFAnalysisChart(currentData);
        return;
    }
    
    if (!currentData || !currentData.indicators) {
        return;
    }
    
    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    const chartElement = document.getElementById('aux-chart');
    
    let traces = [];
    let layout = {};
    
    switch(indicatorType) {
        case 'rsi':
            const rsiTraditional = currentData.indicators.rsi_traditional || [];
            traces = [{
                x: dates,
                y: rsiTraditional,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI Tradicional',
                line: {color: '#FF6B6B', width: 2}
            }];
            
            layout = {
                title: {text: 'RSI Tradicional (14 periodos)', font: {color: '#ffffff', size: 14}},
                xaxis: {title: 'Fecha/Hora', type: 'date', gridcolor: '#444'},
                yaxis: {title: 'RSI', range: [0, 100], gridcolor: '#444'},
                shapes: [
                    {type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 70, y1: 70, line: {color: 'red', dash: 'dash'}},
                    {type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 30, y1: 30, line: {color: 'green', dash: 'dash'}}
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
            break;
            
        case 'macd':
            const macd = currentData.indicators.macd || [];
            const macdSignal = currentData.indicators.macd_signal || [];
            const macdHistogram = currentData.indicators.macd_histogram || [];
            
            traces = [
                {
                    x: dates,
                    y: macd,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MACD',
                    line: {color: '#FF6B6B', width: 2}
                },
                {
                    x: dates,
                    y: macdSignal,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Señal',
                    line: {color: '#4ECDC4', width: 1}
                },
                {
                    x: dates,
                    y: macdHistogram,
                    type: 'bar',
                    name: 'Histograma',
                    marker: {color: macdHistogram.map(val => val >= 0 ? '#00C853' : '#FF1744')}
                }
            ];
            
            layout = {
                title: {text: 'MACD (12,26,9)', font: {color: '#ffffff', size: 14}},
                xaxis: {title: 'Fecha/Hora', type: 'date', gridcolor: '#444'},
                yaxis: {title: 'MACD', gridcolor: '#444'},
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
            break;
            
        case 'squeeze':
            const squeezeOn = currentData.indicators.squeeze_on || [];
            const squeezeOff = currentData.indicators.squeeze_off || [];
            const momentum = currentData.indicators.squeeze_momentum || [];
            
            // Crear arrays para los diferentes estados
            const squeezeOnX = [];
            const squeezeOnY = [];
            const squeezeOffX = [];
            const squeezeOffY = [];
            
            dates.forEach((date, i) => {
                if (squeezeOn[i]) {
                    squeezeOnX.push(date);
                    squeezeOnY.push(momentum[i]);
                } else if (squeezeOff[i]) {
                    squeezeOffX.push(date);
                    squeezeOffY.push(momentum[i]);
                }
            });
            
            traces = [
                {
                    x: dates,
                    y: momentum,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Momentum',
                    line: {color: '#2196F3', width: 2}
                }
            ];
            
            // Añadir marcadores para squeeze on/off
            if (squeezeOnX.length > 0) {
                traces.push({
                    x: squeezeOnX,
                    y: squeezeOnY,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Squeeze ON',
                    marker: {
                        color: '#FFC107',
                        size: 8,
                        symbol: 'circle'
                    }
                });
            }
            
            if (squeezeOffX.length > 0) {
                traces.push({
                    x: squeezeOffX,
                    y: squeezeOffY,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Squeeze OFF',
                    marker: {
                        color: '#9C27B0',
                        size: 8,
                        symbol: 'diamond'
                    }
                });
            }
            
            layout = {
                title: {text: 'Squeeze Momentum', font: {color: '#ffffff', size: 14}},
                xaxis: {title: 'Fecha/Hora', type: 'date', gridcolor: '#444'},
                yaxis: {title: 'Momentum', gridcolor: '#444'},
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
            break;
    }
    
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

function renderMultiTFAnalysisChart(data) {
    const chartElement = document.getElementById('aux-chart');
    
    if (!data || !data.multi_tf_analysis) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de análisis multi-temporalidad disponibles</p>
            </div>
        `;
        return;
    }

    const analysis = data.multi_tf_analysis;
    const results = analysis.results || {};
    
    // Crear gráfico de barras para las tendencias
    const timeframes = Object.keys(results);
    const trends = Object.values(results);
    
    const colors = trends.map(trend => {
        switch(trend) {
            case 'BULLISH': return '#00C853';
            case 'BEARISH': return '#FF1744';
            default: return '#9E9E9E';
        }
    });
    
    const traces = [{
        x: timeframes,
        y: trends.map(() => 1), // Valores constantes para altura
        type: 'bar',
        marker: {
            color: colors
        },
        text: trends,
        textposition: 'auto'
    }];
    
    const layout = {
        title: {
            text: 'Análisis Multi-Temporalidad - Tendencias',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Temporalidad',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Estado',
            showticklabels: false,
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 60, r: 50, b: 80, l: 50},
        annotations: [
            {
                x: 0.5,
                y: 1.1,
                xref: 'paper',
                yref: 'paper',
                text: analysis.ok ? '✅ CONDICIONES MULTI-TF CONFIRMADAS' : '❌ CONDICIONES MULTI-TF NO CUMPLIDAS',
                showarrow: false,
                font: {color: analysis.ok ? '#00C853' : '#FF1744', size: 12},
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: analysis.ok ? '#00C853' : '#FF1744',
                borderwidth: 1,
                borderpad: 4
            }
        ]
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentAuxChart) {
        Plotly.purge('aux-chart');
    }
    
    currentAuxChart = Plotly.newPlot('aux-chart', traces, layout, config);
}

function updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    fetch(`/api/scatter_data_improved?interval=${interval}`)
        .then(response => response.json())
        .then(scatterData => {
            renderScatterChartImproved(scatterData);
        })
        .catch(error => {
            console.error('Error cargando datos de scatter:', error);
        });
}

function renderScatterChartImproved(scatterData) {
    const chartElement = document.getElementById('scatter-chart');
    
    if (!scatterData || scatterData.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos disponibles para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }

    // Preparar datos para el scatter plot
    const traces = [];
    
    // Agrupar por señal
    const longSignals = scatterData.filter(d => d.signal === 'LONG');
    const shortSignals = scatterData.filter(d => d.signal === 'SHORT');
    const neutralSignals = scatterData.filter(d => d.signal === 'NEUTRAL');
    
    // Traza para señales LONG
    if (longSignals.length > 0) {
        traces.push({
            x: longSignals.map(d => d.x),
            y: longSignals.map(d => d.y),
            mode: 'markers',
            type: 'scatter',
            name: 'LONG',
            text: longSignals.map(d => `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>WinRate: ${d.win_rate.toFixed(1)}%`),
            hoverinfo: 'text',
            marker: {
                color: '#00C853',
                size: longSignals.map(d => Math.max(8, d.signal_score / 5)),
                symbol: 'circle',
                line: {
                    color: 'white',
                    width: 1
                }
            }
        });
    }
    
    // Traza para señales SHORT
    if (shortSignals.length > 0) {
        traces.push({
            x: shortSignals.map(d => d.x),
            y: shortSignals.map(d => d.y),
            mode: 'markers',
            type: 'scatter',
            name: 'SHORT',
            text: shortSignals.map(d => `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>WinRate: ${d.win_rate.toFixed(1)}%`),
            hoverinfo: 'text',
            marker: {
                color: '#FF1744',
                size: shortSignals.map(d => Math.max(8, d.signal_score / 5)),
                symbol: 'diamond',
                line: {
                    color: 'white',
                    width: 1
                }
            }
        });
    }
    
    // Traza para señales NEUTRAL
    if (neutralSignals.length > 0) {
        traces.push({
            x: neutralSignals.map(d => d.x),
            y: neutralSignals.map(d => d.y),
            mode: 'markers',
            type: 'scatter',
            name: 'NEUTRAL',
            text: neutralSignals.map(d => `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%`),
            hoverinfo: 'text',
            marker: {
                color: '#9E9E9E',
                size: 6,
                symbol: 'square',
                opacity: 0.5
            }
        });
    }
    
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
        shapes: [
            // Líneas de quadrantes
            {
                type: 'line',
                x0: 50, x1: 50,
                y0: 0, y1: 100,
                line: {color: 'white', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: 0, x1: 100,
                y0: 50, y1: 50,
                line: {color: 'white', width: 1, dash: 'dash'}
            }
        ],
        annotations: [
            {
                x: 25, y: 75,
                text: 'Zona Vendedora',
                showarrow: false,
                font: {color: '#FF1744', size: 12},
                bgcolor: 'rgba(255,23,68,0.1)',
                borderpad: 4
            },
            {
                x: 75, y: 25,
                text: 'Zona Compradora',
                showarrow: false,
                font: {color: '#00C853', size: 12},
                bgcolor: 'rgba(0,200,83,0.1)',
                borderpad: 4
            }
        ]
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentScatterChart) {
        Plotly.purge('scatter-chart');
    }
    
    currentScatterChart = Plotly.newPlot('scatter-chart', traces, layout, config);
}

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    fetch(`/api/multiple_signals?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}`)
        .then(response => response.json())
        .then(data => {
            updateSignalsTable(data.long_signals, 'long-table');
            updateSignalsTable(data.short_signals, 'short-table');
        })
        .catch(error => {
            console.error('Error cargando señales múltiples:', error);
        });
}

function updateSignalsTable(signals, tableId) {
    const tableBody = document.getElementById(tableId);
    
    if (!signals || signals.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    No hay señales confirmadas
                </td>
            </tr>
        `;
        return;
    }
    
    let html = '';
    signals.forEach((signal, index) => {
        const signalClass = signal.signal === 'LONG' ? 'text-success' : 'text-danger';
        const badgeClass = signal.signal === 'LONG' ? 'bg-success' : 'bg-danger';
        
        html += `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;">
                <td>${index + 1}</td>
                <td>
                    <span class="${signalClass} fw-bold">${signal.symbol}</span>
                    <span class="badge ${badgeClass} ms-1">${signal.signal}</span>
                </td>
                <td>
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar ${signalClass.replace('text-', 'bg-')}" 
                             role="progressbar" 
                             style="width: ${signal.signal_score}%"
                             aria-valuenow="${signal.signal_score}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                    <small class="text-muted">${signal.signal_score.toFixed(1)}%</small>
                </td>
                <td>
                    <span class="badge ${signal.win_rate >= 70 ? 'bg-success' : signal.win_rate >= 50 ? 'bg-warning' : 'bg-danger'}">
                        ${signal.win_rate.toFixed(1)}%
                    </span>
                </td>
            </tr>
        `;
    });
    
    tableBody.innerHTML = html;
}

function showSignalDetails(symbol) {
    // Implementar modal con detalles de la señal
    console.log('Mostrando detalles para:', symbol);
    // Aquí puedes implementar un modal con información detallada
}

function updateMarketSummary(data) {
    const summaryElement = document.getElementById('market-summary');
    
    if (!data) {
        summaryElement.innerHTML = `
            <div class="alert alert-warning">
                <p class="mb-0">No hay datos disponibles para el resumen</p>
            </div>
        `;
        return;
    }
    
    const signalClass = data.signal === 'LONG' ? 'text-success' : 
                       data.signal === 'SHORT' ? 'text-danger' : 'text-warning';
    
    const signalBadgeClass = data.signal === 'LONG' ? 'bg-success' : 
                           data.signal === 'SHORT' ? 'bg-danger' : 'bg-warning';
    
    summaryElement.innerHTML = `
        <div class="row text-center">
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-2">
                        <small class="text-muted d-block">Señal Actual</small>
                        <span class="badge ${signalBadgeClass} fs-6">${data.signal}</span>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-2">
                        <small class="text-muted d-block">Score</small>
                        <strong class="${signalClass}">${data.signal_score.toFixed(1)}%</strong>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-2">
                        <small class="text-muted d-block">Precio</small>
                        <strong>$${data.current_price.toFixed(6)}</strong>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-2">
                        <small class="text-muted d-block">Volumen</small>
                        <strong>${formatVolume(data.volume)}</strong>
                    </div>
                </div>
            </div>
        </div>
        <div class="mt-3">
            <small class="text-muted d-block">Niveles Clave</small>
            <div class="d-flex justify-content-between small">
                <span class="text-success">Soporte: $${data.support.toFixed(6)}</span>
                <span class="text-danger">Resistencia: $${data.resistance.toFixed(6)}</span>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const analysisElement = document.getElementById('signal-analysis');
    
    if (!data) {
        analysisElement.innerHTML = `
            <div class="alert alert-warning">
                <p class="mb-0">No hay datos para análisis</p>
            </div>
        `;
        return;
    }
    
    const signalClass = data.signal === 'LONG' ? 'alert-success' : 
                       data.signal === 'SHORT' ? 'alert-danger' : 'alert-warning';
    
    let conditionsHtml = '';
    if (data.fulfilled_conditions && data.fulfilled_conditions.length > 0) {
        conditionsHtml = `
            <div class="mt-2">
                <small class="text-muted d-block">Condiciones cumplidas:</small>
                <ul class="small mb-0 ps-3">
                    ${data.fulfilled_conditions.map(cond => `<li>${cond}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    analysisElement.innerHTML = `
        <div class="alert ${signalClass}">
            <div class="d-flex justify-content-between align-items-center">
                <strong>${data.signal} CONFIRMADO</strong>
                <span class="badge ${signalClass.replace('alert-', 'bg-')}">
                    ${data.signal_score.toFixed(1)}%
                </span>
            </div>
            <div class="mt-2">
                <small>WinRate histórico: <strong>${data.win_rate.toFixed(1)}%</strong></small>
            </div>
            ${conditionsHtml}
        </div>
        <div class="row text-center small">
            <div class="col-6">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-1">
                        <small class="text-muted d-block">ADX</small>
                        <strong class="${data.adx > 25 ? 'text-success' : 'text-warning'}">${data.adx.toFixed(1)}</strong>
                    </div>
                </div>
            </div>
            <div class="col-6">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-1">
                        <small class="text-muted d-block">RSI Maverick</small>
                        <strong class="${data.rsi_maverick < 0.2 ? 'text-success' : data.rsi_maverick > 0.8 ? 'text-danger' : 'text-warning'}">
                            ${(data.rsi_maverick * 100).toFixed(1)}%
                        </strong>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function updateMultiTFAnalysis(data) {
    const analysisElement = document.getElementById('multi-tf-analysis');
    
    if (!data || !data.multi_tf_analysis) {
        analysisElement.innerHTML = `
            <div class="alert alert-warning">
                <p class="mb-0">No hay datos de análisis multi-TF</p>
            </div>
        `;
        return;
    }
    
    const analysis = data.multi_tf_analysis;
    const alertClass = analysis.ok ? 'alert-success' : 'alert-danger';
    const icon = analysis.ok ? '✅' : '❌';
    
    let detailsHtml = '';
    if (analysis.details && analysis.details.length > 0) {
        detailsHtml = `
            <div class="mt-2">
                <small class="text-muted">Detalles:</small>
                <ul class="small mb-0 ps-3">
                    ${analysis.details.map(detail => `<li>${detail}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    analysisElement.innerHTML = `
        <div class="alert ${alertClass}">
            <div class="d-flex align-items-center">
                <span class="me-2">${icon}</span>
                <strong>${analysis.ok ? 'CONDICIONES MULTI-TF CONFIRMADAS' : 'CONDICIONES MULTI-TF NO CUMPLIDAS'}</strong>
            </div>
            ${detailsHtml}
        </div>
    `;
}

function updateWinRateDisplay() {
    const winRateElement = document.getElementById('win-rate-display');
    
    // Simular cálculo de winrate (en una implementación real, esto vendría de una API)
    const simulatedWinRate = Math.random() * 30 + 60; // Entre 60% y 90%
    
    winRateElement.innerHTML = `
        <div class="mb-2">
            <div class="h4 mb-1 text-success">${simulatedWinRate.toFixed(1)}%</div>
            <small class="text-muted">WinRate Promedio</small>
        </div>
        <div class="progress mb-2" style="height: 10px;">
            <div class="progress-bar bg-success" 
                 role="progressbar" 
                 style="width: ${simulatedWinRate}%"
                 aria-valuenow="${simulatedWinRate}" 
                 aria-valuemin="0" 
                 aria-valuemax="100">
            </div>
        </div>
        <small class="text-muted">Basado en análisis histórico</small>
    `;
}

function updateFearGreedIndex() {
    const fearGreedElement = document.getElementById('fear-greed-index');
    
    // Simular índice de miedo y codicia
    const simulatedIndex = Math.floor(Math.random() * 100);
    let sentiment = 'NEUTRAL';
    let color = 'warning';
    
    if (simulatedIndex >= 75) {
        sentiment = 'EXTREME GREED';
        color = 'danger';
    } else if (simulatedIndex >= 55) {
        sentiment = 'GREED';
        color = 'warning';
    } else if (simulatedIndex >= 45) {
        sentiment = 'NEUTRAL';
        color = 'info';
    } else if (simulatedIndex >= 25) {
        sentiment = 'FEAR';
        color = 'primary';
    } else {
        sentiment = 'EXTREME FEAR';
        color = 'success';
    }
    
    fearGreedElement.innerHTML = `
        <div class="text-center">
            <div class="h3 text-${color} mb-1">${simulatedIndex}</div>
            <div class="progress mb-2" style="height: 15px;">
                <div class="progress-bar bg-${color}" 
                     role="progressbar" 
                     style="width: ${simulatedIndex}%"
                     aria-valuenow="${simulatedIndex}" 
                     aria-valuemin="0" 
                     aria-valuemax="100">
                </div>
            </div>
            <small class="text-${color}">${sentiment}</small>
        </div>
    `;
}

function updateMarketRecommendations() {
    const recommendationsElement = document.getElementById('market-recommendations');
    
    const recommendations = [
        'Mercado en fase de acumulación. Buscar entradas LONG en soportes.',
        'Alta volatilidad esperada. Usar stops ajustados.',
        'Tendencia alcista confirmada en múltiples temporalidades.',
        'Posible corrección técnica. Esperar mejores precios para entrar.',
        'Fuerte volumen comprador. Señales LONG predominantes.'
    ];
    
    const randomRecommendation = recommendations[Math.floor(Math.random() * recommendations.length)];
    
    recommendationsElement.innerHTML = `
        <div class="card bg-dark border-info">
            <div class="card-header bg-info bg-opacity-25">
                <h6 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Recomendación</h6>
            </div>
            <div class="card-body">
                <p class="small mb-0">${randomRecommendation}</p>
            </div>
        </div>
    `;
}

function updateScalpingAlerts() {
    const alertsElement = document.getElementById('scalping-alerts');
    
    // Simular alertas de scalping
    const simulatedAlerts = [
        { symbol: 'BTC-USDT', signal: 'LONG', timeframe: '15m', strength: 'ALTA' },
        { symbol: 'ETH-USDT', signal: 'SHORT', timeframe: '30m', strength: 'MEDIA' },
        { symbol: 'SOL-USDT', signal: 'LONG', timeframe: '15m', strength: 'ALTA' }
    ];
    
    let alertsHtml = '';
    simulatedAlerts.forEach(alert => {
        const signalClass = alert.signal === 'LONG' ? 'text-success' : 'text-danger';
        const badgeClass = alert.signal === 'LONG' ? 'bg-success' : 'bg-danger';
        
        alertsHtml += `
            <div class="alert alert-warning scalping-alert mb-2 p-2">
                <div class="d-flex justify-content-between align-items-center">
                    <strong class="${signalClass}">${alert.symbol}</strong>
                    <span class="badge ${badgeClass}">${alert.signal}</span>
                </div>
                <small class="text-muted d-block">${alert.timeframe} | Fuerza: ${alert.strength}</small>
            </div>
        `;
    });
    
    alertsElement.innerHTML = alertsHtml || `
        <div class="text-center text-muted">
            <small>No hay alertas de scalping activas</small>
        </div>
    `;
}

function updateExitSignals() {
    const exitElement = document.getElementById('exit-signals');
    
    // Simular señales de salida
    exitElement.innerHTML = `
        <div class="text-center text-muted">
            <small>No hay señales de salida activas</small>
        </div>
    `;
}

function updateCalendarInfo() {
    // La información del calendario se actualiza automáticamente con el reloj de Bolivia
}

function formatVolume(volume) {
    if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    }
    return volume.toFixed(2);
}

function showError(message) {
    // Mostrar notificación de error
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'error-' + Date.now();
    
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-bg-danger border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-exclamation-triangle me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.innerHTML += toastHtml;
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remover el toast después de que se oculta
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

function downloadReport() {
    // Simular descarga de reporte
    showError('Función de descarga en desarrollo');
}

function downloadStrategicReport() {
    // Simular descarga de reporte estratégico
    showError('Función de reporte estratégico en desarrollo');
}

// Función para manejar errores no capturados
window.addEventListener('error', function(e) {
    console.error('Error no capturado:', e.error);
    showError('Error en la aplicación: ' + e.error.message);
});

// Función para manejar promesas rechazadas no capturadas
window.addEventListener('unhandledrejection', function(e) {
    console.error('Promesa rechazada no capturada:', e.reason);
    showError('Error en promesa: ' + e.reason);
});
