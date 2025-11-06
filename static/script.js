// Configuración global
let currentChart = null;
let currentScatterChart = null;
let currentWhaleChart = null;
let currentAdxChart = null;
let currentRsiChart = null;
let currentAuxChart = null;
let currentBtcDominanceChart = null;
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
    updateCalendarInfo();
    updateWinrateDisplay(); // Añadir esta línea
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


// Función para cargar clasificación de riesgo MEJORADA
function loadCryptoRiskClassification() {
    fetch('/api/crypto_risk_classification')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(riskData => {
            // Verificar que riskData es un objeto válido
            if (typeof riskData !== 'object' || riskData === null) {
                throw new Error('Datos de riesgo no válidos');
            }
            
            allCryptos = [];
            Object.keys(riskData).forEach(category => {
                // Verificar que riskData[category] es un array
                if (Array.isArray(riskData[category])) {
                    riskData[category].forEach(symbol => {
                        allCryptos.push({
                            symbol: symbol,
                            category: category
                        });
                    });
                } else {
                    console.warn(`Categoría ${category} no contiene un array válido:`, riskData[category]);
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

//Nueva liena cargada
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
    }, 90000); // 90 segundos
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
    
    // Actualizar gráfico principal con nuevo sistema
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    
    // Actualizar gráfico de dispersión MEJORADO
    updateScatterChartImproved(interval);
    
    // Actualizar señales múltiples
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    
    // Actualizar análisis multi-temporalidad
    updateMultiTimeframeAnalysis(symbol, interval);
    
    // Actualizar winrate
    updateWinrateDisplay();
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateScalpingAlerts();
    updateCalendarInfo();
}




function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage) {
    const url = `/api/signals?symbol=${symbol}&interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            currentData = data;
            renderCandleChart(data);
            renderTrendStrengthChart(data);
            renderRsiComparisonChart(data);
            renderAdxChartImproved(data);
            renderMacdChart(data);
            renderSqueezeChartImproved(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
            updateMultiTimeframeAnalysis(symbol, interval);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Error al cargar datos: ' + error.message);
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
        take_profit: [52000, 54000, 56000],
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
        trend_strength_signal: 'NEUTRAL',
        no_trade_zone: false,
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
    
    // Añadir líneas de soporte y resistencia si están disponibles
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

    // Añadir líneas de liquidación LONG y SHORT si están disponibles
    if (data.liquidation_long && data.liquidation_short) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.liquidation_long, data.liquidation_long],
            mode: 'lines',
            line: {color: '#FF6B6B', dash: 'dot', width: 3},
            name: 'Liquidación LONG'
        });
        
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.liquidation_short, data.liquidation_short],
            mode: 'lines',
            line: {color: '#4ECDC4', dash: 'dot', width: 3},
            name: 'Liquidación SHORT'
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
    
    // MEJORA 1: Calcular rango dinámico para el eje Y basado en los datos visibles
    const visibleHighs = highs.slice(-50); // Últimos 50 puntos
    const visibleLows = lows.slice(-50);
    const minPrice = Math.min(...visibleLows);
    const maxPrice = Math.max(...visibleHighs);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.05; // 5% de padding
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas Japonesas`,
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
            // MEJORA 1: Escala automática basada en datos visibles
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
        // Configuración para herramientas de dibujo
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

function renderWhaleChartImproved(data) {
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
    
    // MEJORA: Añadir marcadores para cruces DI
    const diCrossBullish = data.indicators.di_cross_bullish || [];
    const diCrossBearish = data.indicators.di_cross_bearish || [];
    
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
        },
        {
            x: dates.filter((_, i) => diCrossBullish[i] && whalePump[i] > 0),
            y: whalePump.filter((_, i) => diCrossBullish[i] && whalePump[i] > 0),
            type: 'scatter',
            mode: 'markers',
            name: 'Cruce DI Compra',
            marker: {color: '#00FF00', size: 10, symbol: 'diamond'}
        },
        {
            x: dates.filter((_, i) => diCrossBearish[i] && whaleDump[i] > 0),
            y: whaleDump.filter((_, i) => diCrossBearish[i] && whaleDump[i] > 0),
            type: 'scatter',
            mode: 'markers',
            name: 'Cruce DI Venta',
            marker: {color: '#FF0000', size: 10, symbol: 'diamond'}
        }
    ];
    
    const layout = {
        title: {
            text: 'Actividad de Ballenas - Compradoras vs Vendedoras (Con Cruces DI)',
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
    
    // Destruir gráfico existente
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
    
    // Destruir gráfico existente
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
    const bullishDivergence = data.indicators.bullish_divergence || [];
    const bearishDivergence = data.indicators.bearish_divergence || [];
    
    // MEJORA 2: Preparar datos para divergencias extendidas (7 velas)
    const bullishDates = [];
    const bullishValues = [];
    const bearishDates = [];
    const bearishValues = [];
    
    // Detectar puntos significativos para divergencias (evitar sobrecargar)
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
            line: {color: '#2196F3', width: 2}
        },
        // MEJORA 2: Añadir marcadores para divergencias alcistas extendidas
        {
            x: bullishDates,
            y: bullishValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Divergencia Alcista (7v)',
            marker: {
                color: '#00FF00',
                size: 12,
                symbol: 'triangle-up',
                line: {color: 'white', width: 1}
            }
        },
        // MEJORA 2: Añadir marcadores para divergencias bajistas extendidas
        {
            x: bearishDates,
            y: bearishValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Divergencia Bajista (7v)',
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
            text: 'RSI Modificado Maverick - Bandas de Bollinger %B (Divergencias 7 Velas)',
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
            // Línea de sobrecompra (0.8)
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
            // Línea de sobreventa (0.2)
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
            // Línea media (0.5)
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
        // MEJORA 3: Leyenda en posición horizontal inferior
        legend: {
            x: 0,
            y: -0.2,
            orientation: 'h',
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 60, r: 50, b: 80, l: 50}, // Aumentar margen inferior para leyenda
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    // Destruir gráfico existente
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
    const bbWidth = data.indicators.bb_width || [];
    const noTradeZones = data.indicators.no_trade_zones || [];
    const colors = data.indicators.colors || [];
    const highZoneThreshold = data.indicators.high_zone_threshold || 5;
    
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
    
    // Añadir líneas de referencia
    traces.push({
        x: [dates[0], dates[dates.length - 1]],
        y: [highZoneThreshold, highZoneThreshold],
        type: 'scatter',
        mode: 'lines',
        name: 'Umbral Alto',
        line: {
            color: 'orange',
            width: 2,
            dash: 'dash'
        }
    });
    
    traces.push({
        x: [dates[0], dates[dates.length - 1]],
        y: [-highZoneThreshold, -highZoneThreshold],
        type: 'scatter',
        mode: 'lines',
        name: 'Umbral Bajo',
        line: {
            color: 'orange',
            width: 2,
            dash: 'dash'
        },
        showlegend: false
    });
    
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
    
    // Destruir gráfico existente
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', traces, layout, config);
}

function renderBtcDominanceChart(data) {
    const chartElement = document.getElementById('aux-chart');
    
    if (!data || !data.mass_selling || !data.mass_buying) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de dominancia BTC disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.timestamp.map(d => new Date(d));
    const massSelling = data.mass_selling || [];
    const massBuying = data.mass_buying || [];
    const dominance = data.dominance || [];
    
    const traces = [
        {
            x: dates,
            y: massSelling,
            type: 'bar',
            name: 'Venta Masiva Criptos',
            marker: {color: '#2196F3'}  // Azul para venta masiva
        },
        {
            x: dates,
            y: massBuying,
            type: 'bar',
            name: 'Compra Masiva Criptos',
            marker: {color: '#FFEB3B'}  // Amarillo para compra masiva
        },
        {
            x: dates,
            y: dominance,
            type: 'scatter',
            mode: 'lines',
            name: 'Dominancia BTC',
            line: {color: 'white', width: 2},
            yaxis: 'y2'
        }
    ];
    
    const layout = {
        title: {
            text: 'Dominancia Bitcoin - Actividad Institucional',
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
            zerolinecolor: '#444',
            side: 'left'
        },
        yaxis2: {
            title: 'Dominancia %',
            gridcolor: '#444',
            zerolinecolor: '#444',
            side: 'right',
            overlaying: 'y',
            range: [40, 60]  // Rango típico de dominancia BTC
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
    
    // Destruir gráfico existente
    if (currentBtcDominanceChart) {
        Plotly.purge('aux-chart');
    }
    
    currentBtcDominanceChart = Plotly.newPlot('aux-chart', traces, layout, config);
}

function updateAuxChart() {
    if (!currentData || !currentData.indicators) return;
    
    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    const squeezeMomentum = currentData.indicators.squeeze_momentum || [];
    const squeezeOn = currentData.indicators.squeeze_on || [];

    const traces = [
        {
            x: dates,
            y: squeezeMomentum,
            type: 'bar',
            name: 'Squeeze Momentum',
            marker: {
                color: squeezeMomentum.map((val, i) => 
                    val >= 0 ? '#00C853' : '#FF1744'
                )
            }
        }
    ];

    // Añadir marcadores para períodos de squeeze
    const squeezeDates = [];
    const squeezeValues = [];
    
    dates.forEach((date, i) => {
        if (squeezeOn[i]) {
            squeezeDates.push(date);
            squeezeValues.push(squeezeMomentum[i] || 0);
        }
    });

    if (squeezeDates.length > 0) {
        traces.push({
            x: squeezeDates,
            y: squeezeValues,
            type: 'scatter',
            mode: 'markers',
            name: 'Squeeze ON',
            marker: {
                color: 'yellow',
                size: 8,
                symbol: 'diamond'
            }
        });
    }

    const layout = {
        title: {text: 'Squeeze Momentum - Compresión y Expansión', font: {color: '#ffffff', size: 14}},
        xaxis: {title: 'Fecha/Hora', type: 'date', gridcolor: '#444'},
        yaxis: {title: 'Momentum', gridcolor: '#444'},
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {x: 0, y: -0.2, orientation: 'h'},
        margin: {t: 60, r: 50, b: 80, l: 50},
        shapes: [{
            type: 'line',
            x0: dates[0],
            x1: dates[dates.length-1],
            y0: 0,
            y1: 0,
            line: {color: 'white', width: 1}
        }]
    };

    Plotly.newPlot('squeeze-chart', traces, layout, {responsive: true});
}

function updateMarketSummary(data) {
    if (!data) return;
    
    const multiTF = data.multi_timeframe_analysis || {};
    const trendStrength = data.trend_strength_signal || 'NEUTRAL';
    const noTradeZone = data.no_trade_zone || false;

    const summaryHTML = `
        <div class="fade-in">
            <div class="row text-center mb-3">
                <div class="col-6">
                    <div class="card bg-dark border-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Señal</small>
                            <h4 class="mb-0 text-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'muted'}">
                                ${data.signal}
                            </h4>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card bg-dark border-${data.signal_score >= 70 ? 'success' : 'warning'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Score</small>
                            <h4 class="mb-0 text-${data.signal_score >= 70 ? 'success' : 'warning'}">
                                ${data.signal_score.toFixed(0)}%
                            </h4>
                        </div>
                    </div>
                </div>
            </div>

            ${noTradeZone ? `
            <div class="alert alert-danger text-center py-1 mb-2">
                <small><i class="fas fa-ban me-1"></i>ZONA DE NO OPERAR</small>
            </div>
            ` : ''}

            <div class="mb-3">
                <h6><i class="fas fa-dollar-sign me-2"></i>Precio Actual</h6>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="fs-5 fw-bold">$${formatPriceForDisplay(data.current_price)}</span>
                    <small class="text-muted">USDT</small>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-layer-group me-2"></i>Multi-Temporalidad</h6>
                <div class="d-flex justify-content-between">
                    <span>Mayor:</span>
                    <span class="text-${multiTF.mayor === 'BULLISH' ? 'success' : multiTF.mayor === 'BEARISH' ? 'danger' : 'muted'}">
                        ${multiTF.mayor || 'NEUTRAL'}
                    </span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Media:</span>
                    <span class="text-${multiTF.media === 'BULLISH' ? 'success' : multiTF.media === 'BEARISH' ? 'danger' : 'muted'}">
                        ${multiTF.media || 'NEUTRAL'}
                    </span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Menor:</span>
                    <span class="text-${multiTF.menor === 'BULLISH' ? 'success' : multiTF.menor === 'BEARISH' ? 'danger' : 'muted'}">
                        ${multiTF.menor || 'NEUTRAL'}
                    </span>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-bolt me-2"></i>Fuerza Tendencia</h6>
                <div class="d-flex justify-content-between">
                    <span>Estado:</span>
                    <span class="text-${getTrendStrengthColor(trendStrength)}">${trendStrength}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Apalancamiento:</span>
                    <span class="text-info">x${data.optimal_leverage || 15}</span>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('market-summary').innerHTML = summaryHTML;
}


function getTrendStrengthColor(signal) {
    switch(signal) {
        case 'STRONG_UP': return 'success';
        case 'WEAK_UP': return 'info';
        case 'STRONG_DOWN': return 'danger';
        case 'WEAK_DOWN': return 'warning';
        default: return 'muted';
    }
}

function formatPriceForDisplay(price) {
    if (price < 0.01) {
        return price.toFixed(6);
    } else if (price < 1) {
        return price.toFixed(4);
    } else {
        return price.toFixed(2);
    }
}

function updateSignalAnalysis(data) {
    if (!data) return;
    
    let analysisHTML = '';
    
    // MEJORA: Mostrar umbrales dinámicos aplicados
    const minScoreRequired = data.signal === 'LONG' ? data.long_min_score : data.short_min_score;
    const ma200Position = data.above_ma_200 ? 'ENCIMA' : 'DEBAJO';
    
    // NUEVO: Información de fuerza de tendencia
    const trendStrengthInfo = data.trend_strength_signal ? 
        `<p class="mb-2 small"><strong>Fuerza de Tendencia:</strong> <span class="text-${getTrendStrengthColor(data.trend_strength_signal)}">${data.trend_strength_signal}</span></p>` : '';
    
    const trendStrengthWarning = !data.trend_strength_filter ? 
        `<div class="alert alert-danger mt-2">
            <h6><i class="fas fa-ban me-2"></i>SEÑAL FILTRADA</h6>
            <p class="mb-0 small">La señal ha sido eliminada por el filtro de fuerza de tendencia. Evitar entrada.</p>
        </div>` : '';
    
    if (data.signal === 'NEUTRAL' || data.signal_score < minScoreRequired) {
        analysisHTML = `
            <div class="text-center">
                <div class="alert alert-secondary">
                    <h6><i class="fas fa-info-circle me-2"></i>Señal No Confirmada</h6>
                    <p class="mb-2 small">Score: <strong>${data.signal_score.toFixed(1)}%</strong> (mínimo requerido: ${minScoreRequired}%)</p>
                    <p class="mb-2 small">Posición vs MA200: <strong>${ma200Position}</strong></p>
                    ${trendStrengthInfo}
                    <p class="mb-0 small text-muted">Esperando confirmación de indicadores...</p>
                </div>
                ${trendStrengthWarning}
            </div>
        `;
    } else {
        const signalColor = data.signal === 'LONG' ? 'success' : 'danger';
        const signalIcon = data.signal === 'LONG' ? 'arrow-up' : 'arrow-down';
        
        analysisHTML = `
            <div class="alert alert-${signalColor}">
                <h6><i class="fas fa-${signalIcon} me-2"></i>Señal ${data.signal} CONFIRMADA</h6>
                <p class="mb-2 small"><strong>Score:</strong> ${data.signal_score.toFixed(1)}% (Mín: ${minScoreRequired}%)</p>
                <p class="mb-2 small"><strong>Posición vs MA200:</strong> ${ma200Position}</p>
                ${trendStrengthInfo}
                
                <h6 class="mt-3 mb-2">Condiciones Cumplidas:</h6>
                <ul class="list-unstyled small mb-3">
                    ${data.fulfilled_conditions.map(condition => `
                        <li><i class="fas fa-check text-${signalColor} me-2"></i>${condition}</li>
                    `).join('')}
                </ul>
                
                <div class="row text-center mt-3">
                    <div class="col-4">
                        <small class="text-muted d-block">Entrada</small>
                        <strong class="text-${signalColor}">$${formatPriceForDisplay(data.entry)}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted d-block">Stop Loss</small>
                        <strong class="text-danger">$${formatPriceForDisplay(data.stop_loss)}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted d-block">ATR</small>
                        <strong class="text-warning">${(data.atr_percentage * 100).toFixed(2)}%</strong>
                    </div>
                </div>
                
                <div class="row text-center mt-2">
                    <div class="col-4">
                        <small class="text-muted d-block">TP1</small>
                        <strong class="text-success">$${formatPriceForDisplay(data.take_profit[0])}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted d-block">TP2</small>
                        <strong class="text-success">$${formatPriceForDisplay(data.take_profit[1])}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted d-block">TP3</small>
                        <strong class="text-success">$${formatPriceForDisplay(data.take_profit[2])}</strong>
                    </div>
                </div>
            </div>
            ${trendStrengthWarning}
        `;
    }
    
    document.getElementById('signal-analysis').innerHTML = analysisHTML;
}




// Función para actualizar datos de dispersión MEJORADA
function updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/scatter_data_improved?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (!data || data.length === 0) {
                throw new Error('No hay datos para el gráfico de dispersión');
            }
            renderScatterChartImproved(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showScatterError('Error al cargar gráfico de dispersión: ' + error.message);
        });
}




function renderScatterChartImproved(scatterData) {
    const scatterElement = document.getElementById('scatter-chart');
    
    // MEJORA: Calcular valores para colores basados en señal real
    const traces = [{
        x: scatterData.map(d => d.x),
        y: scatterData.map(d => d.y),
        text: scatterData.map(d => 
            `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>Señal: ${d.signal}<br>Precio: $${formatPriceForDisplay(d.current_price)}<br>Riesgo: ${d.risk_category}`
        ),
        mode: 'markers',
        marker: {
            size: scatterData.map(d => 8 + (d.signal_score / 15)), // Tamaño basado en score
            color: scatterData.map(d => {
                // MEJORA: Color basado en señal real y categoría de riesgo
                if (d.signal === 'LONG') {
                    return d.risk_category === 'bajo' ? '#00C853' : 
                           d.risk_category === 'medio' ? '#FFC107' : 
                           d.risk_category === 'alto' ? '#FF9800' : '#9C27B0'; // Memecoins: púrpura
                }
                if (d.signal === 'SHORT') {
                    return d.risk_category === 'bajo' ? '#FF1744' : 
                           d.risk_category === 'medio' ? '#FF5252' : 
                           d.risk_category === 'alto' ? '#F44336' : '#E91E63'; // Memecoins: rosa
                }
                return '#9E9E9E'; // Neutro - gris
            }),
            opacity: scatterData.map(d => 0.6 + (d.signal_score / 250)), // Opacidad basada en score
            line: {
                color: 'white',
                width: 1
            },
            symbol: scatterData.map(d => {
                // MEJORA: Símbolos diferentes por categoría de riesgo
                if (d.risk_category === 'bajo') return 'circle';
                if (d.risk_category === 'medio') return 'square';
                if (d.risk_category === 'alto') return 'diamond';
                return 'star'; // Memecoins: estrella
            })
        },
        type: 'scatter',
        hovertemplate: '%{text}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Análisis Multi-Indicador (75 Criptomonedas)',
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
            // Líneas divisorias para 3x3 grid
            {type: 'line', x0: 33.3, x1: 33.3, y0: 0, y1: 100, line: {color: 'gray', width: 1, dash: 'dash'}},
            {type: 'line', x0: 66.6, x1: 66.6, y0: 0, y1: 100, line: {color: 'gray', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 100, y0: 33.3, y1: 33.3, line: {color: 'gray', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 100, y0: 66.6, y1: 66.6, line: {color: 'gray', width: 1, dash: 'dash'}},
            
            // Zona de VENTA (Fila1Columna1) - fondo rojo transparente
            {
                type: 'rect', x0: 0, x1: 33.3, y0: 66.6, y1: 100,
                fillcolor: 'rgba(255, 0, 0, 0.15)',
                line: {width: 0}
            },
            // Zona de COMPRA (Fila3Columna3) - fondo verde transparente
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
            },
            {
                x: 50, y: 95,
                text: '● LONG (Bajo) ● LONG (Medio) ● LONG (Alto) ● LONG (Memecoin) ● SHORT (Bajo) ● SHORT (Medio) ● SHORT (Alto) ● SHORT (Memecoin)',
                showarrow: false,
                font: {color: 'white', size: 9},
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: 'white'
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 80, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    // Destruir gráfico existente
    if (currentScatterChart) {
        Plotly.purge('scatter-chart');
    }
    
    currentScatterChart = Plotly.newPlot('scatter-chart', traces, layout, config);
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
            console.error('Error:', error);
        });
}

function updateSignalsTables(data) {
    // Actualizar tabla LONG
    const longTable = document.getElementById('long-table');
    if (data.long_signals && data.long_signals.length > 0) {
        longTable.innerHTML = data.long_signals.map((signal, index) => `
            <tr onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;" class="hover-row">
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
        shortTable.innerHTML = data.short_signals.map((signal, index) => `
            <tr onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;" class="hover-row">
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

function updateFearGreedIndex() {
    fetch('/api/fear_greed_index')
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
            
            const fgiElement = document.getElementById('fear-greed-index');
            if (fgiElement) {
                fgiElement.innerHTML = `
                    <div class="card bg-dark border-${data.color}">
                        <div class="card-body text-center">
                            <h6><i class="fas fa-brain me-2"></i>Índice Miedo y Codicia</h6>
                            <div class="progress mb-2" style="height: 20px;">
                                <div class="progress-bar bg-${data.color}" role="progressbar" 
                                     style="width: ${data.value}%" aria-valuenow="${data.value}" 
                                     aria-valuemin="0" aria-valuemax="100">
                                    ${data.value}
                                </div>
                            </div>
                            <small class="text-${data.color}">${data.sentiment}</small>
                            <br>
                            <small class="text-muted">Actualizado: ${data.timestamp}</small>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando índice de miedo y codicia:', error);
            // Mostrar estado de error
            const fgiElement = document.getElementById('fear-greed-index');
            if (fgiElement) {
                fgiElement.innerHTML = `
                    <div class="card bg-dark border-warning">
                        <div class="card-body text-center">
                            <h6><i class="fas fa-brain me-2"></i>Índice Miedo y Codicia</h6>
                            <p class="text-warning mb-1">No disponible</p>
                            <small class="text-muted">Actualizando...</small>
                        </div>
                    </div>
                `;
            }
        });
}


function updateMarketRecommendations() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/market_recommendations?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            const recommendationsElement = document.getElementById('market-recommendations');
            if (recommendationsElement) {
                recommendationsElement.innerHTML = `
                    <div class="card bg-dark border-info">
                        <div class="card-header bg-info bg-opacity-25">
                            <h6 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Recomendaciones de Mercado</h6>
                        </div>
                        <div class="card-body">
                            <p class="small mb-2">${data.recommendation}</p>
                            <small class="text-muted">Actualizado: ${data.timestamp}</small>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando recomendaciones:', error);
        });
}




// Función para actualizar alertas de scalping MEJORADA
function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const alertsElement = document.getElementById('scalping-alerts');
            if (!alertsElement) return;
            
            if (data.alerts && data.alerts.length > 0) {
                let alertsHTML = '';
                
                data.alerts.forEach((alert, index) => {
                    const alertType = alert.signal === 'LONG' ? 'success' : 'danger';
                    const alertIcon = alert.signal === 'LONG' ? 'arrow-up' : 'arrow-down';
                    const riskBadge = getRiskBadge(alert.risk_category);
                    
                    // MEJORA: Mostrar información de temporalidad específica
                    const intervalBadge = alert.interval === '15m' || alert.interval === '30m' ? 
                        '<span class="badge bg-warning ms-1">SCALPING</span>' : 
                        `<span class="badge bg-secondary ms-1">${alert.interval}</span>`;
                    
                    // NUEVO: Información de fuerza de tendencia
                    const trendStrengthBadge = alert.trend_strength ? 
                        `<span class="badge bg-${getTrendStrengthColor(alert.trend_strength)} ms-1">${alert.trend_strength}</span>` : '';
                    
                    alertsHTML += `
                        <div class="alert alert-${alertType} mb-2">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <h6 class="mb-1">
                                        <i class="fas fa-${alertIcon} me-1"></i>
                                        ${alert.symbol} ${riskBadge} ${intervalBadge} ${trendStrengthBadge}
                                    </h6>
                                    <p class="mb-1 small">
                                        <strong>Score: ${alert.score.toFixed(1)}%</strong><br>
                                        Entrada: $${formatPriceForDisplay(alert.entry)} | Leverage: x${alert.leverage}
                                    </p>
                                </div>
                                <button class="btn btn-sm btn-outline-${alertType}" onclick="tradeAlert('${alert.symbol}', '${alert.interval}', ${alert.leverage})">
                                    Operar
                                </button>
                            </div>
                            <small class="text-muted">${alert.timestamp}</small>
                        </div>
                    `;
                });
                
                alertsElement.innerHTML = alertsHTML;
            } else {
                alertsElement.innerHTML = `
                    <div class="text-center py-3 text-muted">
                        <i class="fas fa-bell-slash fa-2x mb-2"></i>
                        <p class="mb-0">No hay alertas activas</p>
                        <small>Las alertas aparecerán cuando se detecten oportunidades</small>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando alertas de scalping:', error);
        });
}


// NUEVAS FUNCIONES PARA SEÑALES DE SALIDA
function updateExitSignals() {
    fetch('/api/exit_signals')
        .then(response => response.json())
        .then(data => {
            const exitElement = document.getElementById('exit-signals');
            if (!exitElement) return;
            
            if (data.exit_signals && data.exit_signals.length > 0) {
                let exitHTML = '';
                
                data.exit_signals.forEach((alert, index) => {
                    const alertType = alert.pnl_percent >= 0 ? 'success' : 'danger';
                    const alertIcon = alert.pnl_percent >= 0 ? 'trophy' : 'exclamation-triangle';
                    
                    exitHTML += `
                        <div class="alert alert-${alertType} mb-2">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <h6 class="mb-1">
                                        <i class="fas fa-${alertIcon} me-1"></i>
                                        ${alert.symbol} - SALIDA ${alert.signal}
                                    </h6>
                                    <p class="mb-1 small">
                                        <strong>Razón: ${alert.reason}</strong><br>
                                        Entrada: $${formatPriceForDisplay(alert.entry_price)} | 
                                        Salida: $${formatPriceForDisplay(alert.exit_price)}<br>
                                        <strong class="text-${alertType}">P&L: ${alert.pnl_percent.toFixed(2)}%</strong>
                                    </p>
                                </div>
                            </div>
                            <small class="text-muted">${alert.timestamp}</small>
                        </div>
                    `;
                });
                
                exitElement.innerHTML = exitHTML;
            } else {
                exitElement.innerHTML = `
                    <div class="text-center py-3 text-muted">
                        <i class="fas fa-check-circle fa-2x mb-2"></i>
                        <p class="mb-0">No hay señales de salida activas</p>
                        <small>Todas las operaciones están en condiciones favorables</small>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando señales de salida:', error);
        });
}


function getRiskBadge(riskCategory) {
    const badges = {
        'bajo': '<span class="badge bg-success ms-1">Bajo</span>',
        'medio': '<span class="badge bg-warning ms-1">Medio</span>',
        'alto': '<span class="badge bg-danger ms-1">Alto</span>',
        'memecoins': '<span class="badge bg-info ms-1">Memecoin</span>'
    };
    return badges[riskCategory] || '';
}

function tradeAlert(symbol, interval, leverage) {
    // Cambiar a la crypto y temporalidad de la alerta
    currentSymbol = symbol;
    document.getElementById('selected-crypto').textContent = symbol;
    document.getElementById('interval-select').value = interval;
    document.getElementById('leverage').value = leverage;
    
    // Actualizar gráficos
    updateCharts();
    
    // Mostrar mensaje de confirmación
    showNotification(`Configurado para operar ${symbol} en ${interval} con leverage x${leverage}`, 'success');
}

function showNotification(message, type = 'info') {
    // Crear notificación toast
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(container);
    }
    
    const toastId = 'toast-' + Date.now();
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    document.getElementById('toast-container').innerHTML += toastHTML;
    
    // Mostrar toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remover toast después de ocultarse
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

function showSignalDetails(symbol) {
    const modal = new bootstrap.Modal(document.getElementById('signalModal'));
    
    const signalData = currentData && currentData.symbol === symbol ? currentData : null;
    
    const detailsHTML = signalData ? `
        <div class="signal-details">
            <h4 class="text-${signalData.signal === 'LONG' ? 'success' : 'danger'}">
                <i class="fas fa-${signalData.signal === 'LONG' ? 'arrow-up' : 'arrow-down'} me-2"></i>
                ${symbol} - Señal ${signalData.signal} Confirmada
            </h4>
            <p class="text-muted">Score de señal: <strong>${signalData.signal_score.toFixed(1)}%</strong></p>
            <p class="text-muted">Posición vs MA200: <strong>${signalData.above_ma_200 ? 'ENCIMA' : 'DEBAJO'}</strong></p>
            <p class="text-muted">Fuerza de Tendencia: <strong class="text-${getTrendStrengthColor(signalData.trend_strength_signal)}">${signalData.trend_strength_signal}</strong></p>
            <p class="text-muted">Umbral aplicado: <strong>${signalData.signal === 'LONG' ? signalData.long_min_score : signalData.short_min_score}%</strong></p>
            
            <div class="row mt-3">
                <div class="col-md-6">
                    <h6>Información de Trading</h6>
                    <table class="table table-sm table-dark">
                        <tr><td>Precio Actual:</td><td class="text-end">$${formatPriceForDisplay(signalData.current_price)}</td></tr>
                        <tr><td>Entrada Recomendada:</td><td class="text-end text-${signalData.signal === 'LONG' ? 'success' : 'danger'}">$${formatPriceForDisplay(signalData.entry)}</td></tr>
                        <tr><td>Stop Loss:</td><td class="text-end text-danger">$${formatPriceForDisplay(signalData.stop_loss)}</td></tr>
                        <tr><td>Soporte:</td><td class="text-end text-info">$${formatPriceForDisplay(signalData.support)}</td></tr>
                        <tr><td>Resistencia:</td><td class="text-end text-warning">$${formatPriceForDisplay(signalData.resistance)}</td></tr>
                        <tr><td>ATR:</td><td class="text-end text-muted">${(signalData.atr_percentage * 100).toFixed(2)}%</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Take Profits Escalonados</h6>
                    <table class="table table-sm table-dark">
                        ${signalData.take_profit.map((tp, index) => `
                            <tr>
                                <td>TP${index + 1}:</td>
                                <td class="text-end text-success">$${formatPriceForDisplay(tp)}</td>
                                <td class="text-end">
                                    <small class="text-muted">
                                        ${(((tp - signalData.entry) / signalData.entry) * 100).toFixed(2)}%
                                    </small>
                                </td>
                            </tr>
                        `).join('')}
                    </table>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-12">
                    <h6>Indicadores Técnicos</h6>
                    <div class="d-flex justify-content-between flex-wrap">
                        <span class="badge bg-primary me-2 mb-1">ADX: ${signalData.adx.toFixed(1)}</span>
                        <span class="badge bg-success me-2 mb-1">D+: ${signalData.plus_di.toFixed(1)}</span>
                        <span class="badge bg-danger me-2 mb-1">D-: ${signalData.minus_di.toFixed(1)}</span>
                        <span class="badge bg-info me-2 mb-1">Ballenas Comp: ${signalData.whale_pump.toFixed(1)}</span>
                        <span class="badge bg-warning me-2 mb-1">Ballenas Vend: ${signalData.whale_dump.toFixed(1)}</span>
                        <span class="badge bg-secondary me-2 mb-1">RSI Maverick: ${(signalData.rsi_maverick * 100).toFixed(1)}%</span>
                        <span class="badge bg-${getTrendStrengthColor(signalData.trend_strength_signal)} me-2 mb-1">Fuerza: ${signalData.trend_strength_signal}</span>
                    </div>
                </div>
            </div>
            
            ${signalData.fulfilled_conditions.length > 0 ? `
            <div class="row mt-3">
                <div class="col-12">
                    <h6>Condiciones Cumplidas</h6>
                    <div class="alert alert-${signalData.signal === 'LONG' ? 'success' : 'danger'}">
                        <ul class="mb-0">
                            ${signalData.fulfilled_conditions.map(condition => `
                                <li>${condition}</li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            </div>
            ` : ''}
            
            <div class="mt-3 text-center">
                <button class="btn btn-primary me-2" onclick="downloadSignalReport('${symbol}')">
                    <i class="fas fa-download me-1"></i>Descargar Reporte Completo
                </button>
                <button class="btn btn-outline-secondary" data-bs-dismiss="modal">
                    Cerrar
                </button>
            </div>
        </div>
    ` : `
        <div class="text-center py-4">
            <h4>${symbol}</h4>
            <p class="text-muted">No hay información detallada disponible para esta señal.</p>
            <button class="btn btn-primary" onclick="downloadSignalReport('${symbol}')">
                <i class="fas fa-download me-1"></i>Generar Reporte
            </button>
        </div>
    `;
    
    document.getElementById('signal-details').innerHTML = detailsHTML;
    modal.show();
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

function downloadSignalReport(symbol) {
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

// Funciones auxiliares
function getVolumeLevel(currentVolume, averageVolume) {
    if (!averageVolume || averageVolume === 0) return {text: 'N/A', color: 'muted'};
    
    const ratio = currentVolume / averageVolume;
    if (ratio > 3) return {text: 'Muy Alto', color: 'success'};
    if (ratio > 2) return {text: 'Alto', color: 'info'};
    if (ratio > 1.5) return {text: 'Medio', color: 'warning'};
    if (ratio > 1) return {text: 'Bajo', color: 'muted'};
    return {text: 'Muy Bajo', color: 'danger'};
}

function getADXStrength(adx) {
    if (adx > 50) return {text: 'Muy Fuerte', color: 'success'};
    if (adx > 25) return {text: 'Fuerte', color: 'info'};
    if (adx > 20) return {text: 'Moderado', color: 'warning'};
    return {text: 'Débil', color: 'danger'};
}

function getSignalStrength(data) {
    const strength = data.signal_score;
    if (strength >= 80) return {text: 'Muy Fuerte', color: 'success'};
    if (strength >= 70) return {text: 'Fuerte', color: 'info'};
    if (strength >= 50) return {text: 'Moderada', color: 'warning'};
    return {text: 'Débil', color: 'danger'};
}

function showError(message) {
    const chartElement = document.getElementById('candle-chart');
    chartElement.innerHTML = `
        <div class="alert alert-danger mt-3" role="alert">
            <h5><i class="fas fa-exclamation-triangle me-2"></i>Error</h5>
            <p>${message}</p>
            <button class="btn btn-sm btn-primary mt-2" onclick="updateCharts()">
                <i class="fas fa-sync-alt me-1"></i>Reintentar
            </button>
        </div>
    `;
}

function showScatterError(message) {
    const scatterElement = document.getElementById('scatter-chart');
    scatterElement.innerHTML = `
        <div class="alert alert-warning mt-3" role="alert">
            <h5><i class="fas fa-exclamation-triangle me-2"></i>Aviso</h5>
            <p>${message}</p>
            <button class="btn btn-sm btn-primary mt-2" onclick="updateScatterChartImproved('4h')">
                <i class="fas fa-sync-alt me-1"></i>Reintentar
            </button>
        </div>
    `;
}
function updateWinrateDisplay() {
    fetch('/api/winrate_data')
        .then(response => response.json())
        .then(data => {
            const winrateDisplay = document.getElementById('winrate-display');
            if (winrateDisplay && data.winrate_data) {
                const winrate = data.winrate_data.global_winrate;
                const totalOps = data.winrate_data.total_operations;
                
                winrateDisplay.innerHTML = `
                    <h3 class="text-success mb-1">${winrate.toFixed(1)}%</h3>
                    <p class="small text-muted mb-0">${totalOps} operaciones</p>
                    <div class="progress mt-2" style="height: 8px;">
                        <div class="progress-bar bg-success" style="width: ${winrate}%"></div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
        });
}

// Función para descargar reporte de estrategias
function downloadStrategyReport() {
    fetch('/api/winrate_data')
        .then(response => response.json())
        .then(data => {
            // Crear contenido del reporte
            let reportContent = "REPORTE DE ESTRATEGIAS - MULTI-TIMEFRAME CRYPTO WGTA PRO\n\n";
            reportContent += `Winrate Global: ${data.winrate_data.global_winrate.toFixed(1)}%\n`;
            reportContent += `Total Operaciones: ${data.winrate_data.total_operations}\n\n`;
            
            if (data.strategy_recommendations && data.strategy_recommendations.top_strategies) {
                reportContent += "TOP ESTRATEGIAS:\n";
                data.strategy_recommendations.top_strategies.forEach((strategy, index) => {
                    reportContent += `${index + 1}. Winrate: ${strategy.winrate.toFixed(1)}% | Count: ${strategy.count} | Score Prom: ${strategy.avg_score.toFixed(1)}\n`;
                });
            }
            
            // Crear y descargar archivo
            const blob = new Blob([reportContent], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `estrategias_${new Date().toISOString().split('T')[0]}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error('Error generando reporte de estrategias:', error);
            showNotification('Error generando reporte de estrategias', 'danger');
        });
}

