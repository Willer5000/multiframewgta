// Configuración global
let currentChart = null;
let currentScatterChart = null;
let currentMaChart = null;
let currentRsiComparisonChart = null;
let currentMacdChart = null;
let currentSqueezeChart = null;
let currentAdxChart = null;
let currentBollingerChart = null;
let currentWhaleChart = null;
let currentRsiChart = null;
let currentAuxChart = null;
let currentTrendStrengthChart = null;
let currentSymbol = 'BTC-USDT';
let currentData = null;
let allCryptos = [];
let updateInterval = null;
let drawingToolsActive = false;
let currentWinRate = 50.0;

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    updateCharts();
    startAutoUpdate();
    initializeWinRateTracking();
});

function initializeApp() {
    console.log('MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializado');
    loadCryptoRiskClassification();
    loadMarketIndicators();
    updateCalendarInfo();
    initializeWinRateTracking();
}

function initializeWinRateTracking() {
    // Inicializar tracking de winrate
    updateWinRateDisplay();
    
    // Actualizar winrate cada 30 segundos
    setInterval(updateWinRateDisplay, 30000);
}

function updateWinRateDisplay() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/win_rate?symbol=${symbol}&interval=${interval}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            currentWinRate = data.win_rate || 50.0;
            updateWinRateUI();
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
            currentWinRate = 50.0;
            updateWinRateUI();
        });
}

function updateWinRateUI() {
    const winRateElement = document.getElementById('current-win-rate');
    const winRateInfo = document.getElementById('win-rate-info');
    
    if (winRateElement && winRateInfo) {
        winRateElement.textContent = `${currentWinRate.toFixed(1)}%`;
        
        // Actualizar colores según el winrate
        if (currentWinRate >= 70) {
            winRateElement.className = 'text-success mb-1';
            winRateInfo.innerHTML = '<span class="text-success">✅ Estrategia óptima</span>';
        } else if (currentWinRate >= 60) {
            winRateElement.className = 'text-warning mb-1';
            winRateInfo.innerHTML = '<span class="text-warning">⚠️ Estrategia buena</span>';
        } else if (currentWinRate >= 50) {
            winRateElement.className = 'text-warning mb-1';
            winRateInfo.innerHTML = '<span class="text-warning">📊 Estrategia regular</span>';
        } else {
            winRateElement.className = 'text-danger mb-1';
            winRateInfo.innerHTML = '<span class="text-danger">❌ Revisar estrategia</span>';
        }
    }
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
                        📅 ${data.date} | Scalping 15m/30m: ${scalpingStatus} | Horario: 4am-4pm L-V
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
    const charts = [
        'candle-chart', 'ma-chart', 'rsi-comparison-chart', 
        'macd-chart', 'squeeze-chart', 'trend-strength-chart',
        'aux-chart'
    ];
    
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
    const charts = [
        'candle-chart', 'ma-chart', 'rsi-comparison-chart', 
        'macd-chart', 'squeeze-chart', 'trend-strength-chart',
        'aux-chart'
    ];
    
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
    updateWinRateDisplay();
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

    document.getElementById('obligatory-conditions').innerHTML = `
        <div class="text-center py-2">
            <div class="spinner-border spinner-border-sm text-warning" role="status">
                <span class="visually-hidden">Verificando...</span>
            </div>
            <p class="mt-2 mb-0 small">Verificando condiciones...</p>
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
    
    // Actualizar winrate
    updateWinRateDisplay();
}

function updateMarketIndicators() {
    updateMarketRecommendations();
    updateScalpingAlerts();
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
            renderMaChart(data);
            renderRsiComparisonChart(data);
            renderMacdChart(data);
            renderSqueezeChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
            updateObligatoryConditions(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Error al cargar datos del gráfico: ' + error.message);
            
            // Mostrar datos de ejemplo en caso de error
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
        whale_pump: 0,
        whale_dump: 0,
        rsi_maverick: 0.5,
        trend_strength_signal: 'NEUTRAL',
        no_trade_zone: false,
        obligatory_met_long: false,
        obligatory_met_short: false,
        fulfilled_conditions: []
    };
    
    updateMarketSummary(sampleData);
    updateSignalAnalysis(sampleData);
    updateObligatoryConditions(sampleData);
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
    
    // Calcular rango dinámico para el eje Y
    const visibleHighs = highs.slice(-50);
    const visibleLows = lows.slice(-50);
    const minPrice = Math.min(...visibleLows);
    const maxPrice = Math.max(...visibleHighs);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.05;
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas (Multi-Temporalidad)`,
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

function renderMaChart(data) {
    const chartElement = document.getElementById('ma-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de medias móviles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const closes = data.data.slice(-50).map(d => parseFloat(d.close));
    
    const traces = [
        {
            x: dates,
            y: closes,
            type: 'scatter',
            mode: 'lines',
            name: 'Precio',
            line: {color: 'white', width: 2}
        }
    ];
    
    // Añadir medias móviles si están disponibles
    if (data.indicators.ma_9 && data.indicators.ma_21 && data.indicators.ma_50) {
        traces.push({
            x: dates,
            y: data.indicators.ma_9.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'MA 9',
            line: {color: '#FF9800', width: 1}
        });
        
        traces.push({
            x: dates,
            y: data.indicators.ma_21.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'MA 21',
            line: {color: '#2196F3', width: 1}
        });
        
        traces.push({
            x: dates,
            y: data.indicators.ma_50.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'MA 50',
            line: {color: '#9C27B0', width: 1}
        });
    }
    
    const layout = {
        title: {
            text: 'Medias Móviles',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Precio',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2,
            font: {color: '#ffffff'}
        },
        margin: {t: 40, r: 30, b: 60, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentMaChart) {
        Plotly.purge('ma-chart');
    }
    
    currentMaChart = Plotly.newPlot('ma-chart', traces, layout, config);
}

function renderRsiComparisonChart(data) {
    const chartElement = document.getElementById('rsi-comparison-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    
    const traces = [];
    
    // RSI Tradicional
    if (data.indicators.rsi) {
        traces.push({
            x: dates,
            y: data.indicators.rsi.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Tradicional',
            line: {color: '#FF6B6B', width: 2}
        });
    }
    
    // RSI Maverick (convertido a escala 0-100)
    if (data.indicators.rsi_maverick) {
        const rsiMaverickScaled = data.indicators.rsi_maverick.slice(-50).map(x => x * 100);
        traces.push({
            x: dates,
            y: rsiMaverickScaled,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick',
            line: {color: '#4ECDC4', width: 2}
        });
    }
    
    const layout = {
        title: {
            text: 'RSI Tradicional vs Maverick',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'RSI',
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
            orientation: 'h',
            y: -0.2,
            font: {color: '#ffffff'}
        },
        margin: {t: 40, r: 30, b: 60, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentRsiComparisonChart) {
        Plotly.purge('rsi-comparison-chart');
    }
    
    currentRsiComparisonChart = Plotly.newPlot('rsi-comparison-chart', traces, layout, config);
}

function renderMacdChart(data) {
    const chartElement = document.getElementById('macd-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de MACD</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    
    const traces = [];
    
    // MACD Line
    if (data.indicators.macd) {
        traces.push({
            x: dates,
            y: data.indicators.macd.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: {color: '#FF6B6B', width: 2}
        });
    }
    
    // Signal Line
    if (data.indicators.macd_signal) {
        traces.push({
            x: dates,
            y: data.indicators.macd_signal.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#4ECDC4', width: 1}
        });
    }
    
    // Histogram
    if (data.indicators.macd_histogram) {
        const colors = data.indicators.macd_histogram.slice(-50).map(val => 
            val >= 0 ? '#00C853' : '#FF1744'
        );
        
        traces.push({
            x: dates,
            y: data.indicators.macd_histogram.slice(-50),
            type: 'bar',
            name: 'Histograma',
            marker: {color: colors}
        });
    }
    
    const layout = {
        title: {
            text: 'MACD',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'MACD',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2,
            font: {color: '#ffffff'}
        },
        margin: {t: 40, r: 30, b: 60, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
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

function renderSqueezeChart(data) {
    const chartElement = document.getElementById('squeeze-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Squeeze Momentum</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const squeezeMomentum = data.indicators.squeeze_momentum.slice(-50);
    
    const colors = squeezeMomentum.map(val => val >= 0 ? '#00C853' : '#FF1744');
    
    const trace = {
        x: dates,
        y: squeezeMomentum,
        type: 'bar',
        name: 'Squeeze Momentum',
        marker: {color: colors}
    };
    
    const layout = {
        title: {
            text: 'Squeeze Momentum',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Momentum',
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
                line: {color: 'white', width: 1}
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 40, r: 30, b: 60, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentSqueezeChart) {
        Plotly.purge('squeeze-chart');
    }
    
    currentSqueezeChart = Plotly.newPlot('squeeze-chart', [trace], layout, config);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Fuerza de Tendencia</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const trendStrength = data.indicators.trend_strength.slice(-50);
    
    const colors = trendStrength.map(val => val >= 0 ? '#00C853' : '#FF1744');
    
    const trace = {
        x: dates,
        y: trendStrength,
        type: 'bar',
        name: 'Fuerza de Tendencia',
        marker: {color: colors}
    };
    
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
            title: 'Fuerza %',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 40, r: 30, b: 60, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', [trace], layout, config);
}

function updateAuxChart() {
    const auxIndicator = document.getElementById('aux-indicator').value;
    const chartElement = document.getElementById('aux-chart');
    
    if (!currentData || !currentData.indicators) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos para el indicador auxiliar</p>
            </div>
        `;
        return;
    }

    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    let traces = [];
    let title = '';
    
    switch(auxIndicator) {
        case 'rsi':
            title = 'RSI Tradicional';
            traces = [{
                x: dates,
                y: currentData.indicators.rsi.slice(-50),
                type: 'scatter',
                mode: 'lines',
                name: 'RSI',
                line: {color: '#FF6B6B', width: 2}
            }];
            break;
            
        case 'macd':
            title = 'MACD';
            traces = [
                {
                    x: dates,
                    y: currentData.indicators.macd.slice(-50),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MACD',
                    line: {color: '#FF6B6B', width: 2}
                },
                {
                    x: dates,
                    y: currentData.indicators.macd_signal.slice(-50),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Señal',
                    line: {color: '#4ECDC4', width: 1}
                }
            ];
            break;
            
        case 'squeeze':
            title = 'Squeeze Momentum';
            const squeezeMomentum = currentData.indicators.squeeze_momentum.slice(-50);
            const colors = squeezeMomentum.map(val => val >= 0 ? '#00C853' : '#FF1744');
            
            traces = [{
                x: dates,
                y: squeezeMomentum,
                type: 'bar',
                name: 'Squeeze',
                marker: {color: colors}
            }];
            break;
            
        case 'moving_averages':
            title = 'Medias Móviles';
            traces = [
                {
                    x: dates,
                    y: currentData.data.slice(-50).map(d => parseFloat(d.close)),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Precio',
                    line: {color: 'white', width: 2}
                },
                {
                    x: dates,
                    y: currentData.indicators.ma_9.slice(-50),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MA 9',
                    line: {color: '#FF9800', width: 1}
                },
                {
                    x: dates,
                    y: currentData.indicators.ma_21.slice(-50),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MA 21',
                    line: {color: '#2196F3', width: 1}
                }
            ];
            break;
    }
    
    const layout = {
        title: {
            text: title,
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2,
            font: {color: '#ffffff'}
        },
        margin: {t: 40, r: 30, b: 60, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
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
            console.error('Error actualizando scatter chart:', error);
        });
}

function renderScatterChartImproved(data) {
    const chartElement = document.getElementById('scatter-chart');
    
    if (!data || data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }
    
    const traces = [];
    const riskCategories = {
        'bajo': { color: '#00C853', name: 'Bajo Riesgo' },
        'medio': { color: '#FF9800', name: 'Medio Riesgo' },
        'alto': { color: '#FF1744', name: 'Alto Riesgo' },
        'memecoins': { color: '#8B5CF6', name: 'Memecoins' }
    };
    
    Object.keys(riskCategories).forEach(category => {
        const categoryData = data.filter(d => d.risk_category === category);
        
        if (categoryData.length > 0) {
            traces.push({
                x: categoryData.map(d => d.x),
                y: categoryData.map(d => d.y),
                text: categoryData.map(d => `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>Precio: $${d.current_price.toFixed(2)}`),
                mode: 'markers',
                type: 'scatter',
                name: riskCategories[category].name,
                marker: {
                    color: riskCategories[category].color,
                    size: categoryData.map(d => Math.max(10, d.signal_score / 3)),
                    line: { width: 1, color: 'white' }
                },
                hovertemplate: '%{text}<extra></extra>'
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
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: {color: '#ffffff'}
        },
        margin: {t: 80, r: 50, b: 50, l: 50},
        shapes: [
            {
                type: 'line',
                x0: 70, x1: 70,
                y0: 0, y1: 100,
                line: {color: 'green', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: 0, x1: 100,
                y0: 70, y1: 70,
                line: {color: 'red', width: 1, dash: 'dash'}
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
            updateSignalTables(data);
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
        longTable.innerHTML = data.long_signals.slice(0, 5).map((signal, index) => `
            <tr class="crypto-item" onclick="showSignalDetails('${signal.symbol}', '${signal.interval}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-success">${signal.signal_score.toFixed(1)}%</span></td>
                <td>$${signal.entry.toFixed(2)}</td>
            </tr>
        `).join('');
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
        shortTable.innerHTML = data.short_signals.slice(0, 5).map((signal, index) => `
            <tr class="crypto-item" onclick="showSignalDetails('${signal.symbol}', '${signal.interval}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-danger">${signal.signal_score.toFixed(1)}%</span></td>
                <td>$${signal.entry.toFixed(2)}</td>
            </tr>
        `).join('');
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

function showSignalDetails(symbol, interval) {
    fetch(`/api/signals?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
        .then(signalData => {
            const modalBody = document.getElementById('signal-details');
            modalBody.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Detalles de la Señal</h6>
                        <p><strong>Símbolo:</strong> ${signalData.symbol}</p>
                        <p><strong>Señal:</strong> <span class="badge ${signalData.signal === 'LONG' ? 'bg-success' : 'bg-danger'}">${signalData.signal}</span></p>
                        <p><strong>Score:</strong> ${signalData.signal_score.toFixed(1)}%</p>
                        <p><strong>Win Rate:</strong> ${signalData.win_rate.toFixed(1)}%</p>
                    </div>
                    <div class="col-md-6">
                        <h6>Niveles de Trading</h6>
                        <p><strong>Entrada:</strong> $${signalData.entry.toFixed(6)}</p>
                        <p><strong>Stop Loss:</strong> $${signalData.stop_loss.toFixed(6)}</p>
                        <p><strong>Take Profit:</strong> $${signalData.take_profit[0].toFixed(6)}</p>
                        <p><strong>Ratio R:R:</strong> ${calculateRiskReward(signalData).toFixed(2)}:1</p>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Condiciones Cumplidas</h6>
                        <ul>
                            ${signalData.fulfilled_conditions.map(cond => `<li>${cond}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
            
            const modal = new bootstrap.Modal(document.getElementById('signalModal'));
            modal.show();
        })
        .catch(error => {
            console.error('Error mostrando detalles de señal:', error);
        });
}

function calculateRiskReward(signalData) {
    const risk = Math.abs(signalData.entry - signalData.stop_loss);
    const reward = Math.abs(signalData.take_profit[0] - signalData.entry);
    return reward / risk;
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    
    if (!data) {
        marketSummary.innerHTML = `
            <div class="alert alert-warning">
                <p class="mb-0">No hay datos disponibles para el resumen del mercado</p>
            </div>
        `;
        return;
    }
    
    const signalBadge = data.signal === 'LONG' ? 
        '<span class="badge bg-success">LONG</span>' : 
        data.signal === 'SHORT' ? 
        '<span class="badge bg-danger">SHORT</span>' : 
        '<span class="badge bg-secondary">NEUTRAL</span>';
    
    marketSummary.innerHTML = `
        <div class="row text-center">
            <div class="col-md-4 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body">
                        <h6 class="card-title">Señal Actual</h6>
                        <div class="display-6 fw-bold ${data.signal === 'LONG' ? 'text-success' : data.signal === 'SHORT' ? 'text-danger' : 'text-secondary'}">
                            ${signalBadge}
                        </div>
                        <small class="text-muted">Score: ${data.signal_score.toFixed(1)}%</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body">
                        <h6 class="card-title">Precio Actual</h6>
                        <div class="h4 fw-bold text-warning">$${data.current_price.toFixed(6)}</div>
                        <small class="text-muted">USDT</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body">
                        <h6 class="card-title">Volumen</h6>
                        <div class="h4 fw-bold text-info">${formatVolume(data.volume)}</div>
                        <small class="text-muted">Ratio: ${(data.volume / data.volume_ma).toFixed(2)}x</small>
                    </div>
                </div>
            </div>
        </div>
        <div class="row mt-2">
            <div class="col-12">
                <div class="alert ${data.no_trade_zone ? 'alert-danger' : 'alert-success'}">
                    <small>
                        <strong>${data.no_trade_zone ? '⛔ ZONA DE NO OPERAR' : '✅ ZONA OPERABLE'}</strong> | 
                        Fuerza Tendencia: <strong>${data.trend_strength_signal}</strong> |
                        ADX: <strong>${data.adx.toFixed(1)}</strong>
                    </small>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (!data) return;
    
    const riskReward = calculateRiskReward(data);
    const positionSize = calculatePositionSize(data);
    
    signalAnalysis.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Análisis de Riesgo</h6>
                <p><strong>Ratio R:R:</strong> <span class="${riskReward >= 2 ? 'text-success' : 'text-warning'}">${riskReward.toFixed(2)}:1</span></p>
                <p><strong>Stop Loss:</strong> $${data.stop_loss.toFixed(6)}</p>
                <p><strong>Take Profit:</strong> $${data.take_profit[0].toFixed(6)}</p>
            </div>
            <div class="col-md-6">
                <h6>Indicadores Clave</h6>
                <p><strong>ADX:</strong> ${data.adx.toFixed(1)} ${data.adx > 25 ? '📈' : '📉'}</p>
                <p><strong>RSI Maverick:</strong> ${(data.rsi_maverick * 100).toFixed(1)}</p>
                <p><strong>Ballenas:</strong> ${data.whale_pump > 0 ? '🐋' : ''}${data.whale_dump > 0 ? '📉' : '➖'}</p>
            </div>
        </div>
        ${data.fulfilled_conditions.length > 0 ? `
        <div class="row mt-2">
            <div class="col-12">
                <h6>Condiciones Cumplidas (${data.fulfilled_conditions.length})</h6>
                <div class="small">
                    ${data.fulfilled_conditions.slice(0, 3).map(cond => `<span class="badge bg-success me-1 mb-1">${cond}</span>`).join('')}
                    ${data.fulfilled_conditions.length > 3 ? `<span class="badge bg-info">+${data.fulfilled_conditions.length - 3} más</span>` : ''}
                </div>
            </div>
        </div>
        ` : ''}
    `;
}

function updateObligatoryConditions(data) {
    const obligatoryConditions = document.getElementById('obligatory-conditions');
    
    if (!data) return;
    
    const longMet = data.obligatory_met_long ? '✅' : '❌';
    const shortMet = data.obligatory_met_short ? '✅' : '❌';
    
    obligatoryConditions.innerHTML = `
        <div class="row text-center">
            <div class="col-6">
                <h6>LONG</h6>
                <div class="h4 ${data.obligatory_met_long ? 'text-success' : 'text-danger'}">${longMet}</div>
                <small class="text-muted">Condiciones obligatorias</small>
            </div>
            <div class="col-6">
                <h6>SHORT</h6>
                <div class="h4 ${data.obligatory_met_short ? 'text-success' : 'text-danger'}">${shortMet}</div>
                <small class="text-muted">Condiciones obligatorias</small>
            </div>
        </div>
    `;
}

function updateMarketRecommendations() {
    fetch('/api/multiple_signals?interval=4h')
        .then(response => response.json())
        .then(data => {
            const recommendations = document.getElementById('market-recommendations');
            const totalSignals = (data.long_signals?.length || 0) + (data.short_signals?.length || 0);
            
            let marketSentiment = 'NEUTRAL';
            let sentimentColor = 'secondary';
            
            if (totalSignals > 8) {
                marketSentiment = 'ALCISTA FUERTE';
                sentimentColor = 'success';
            } else if (totalSignals > 4) {
                marketSentiment = 'ALCISTA';
                sentimentColor = 'success';
            } else if (totalSignals > 2) {
                marketSentiment = 'NEUTRO';
                sentimentColor = 'warning';
            } else {
                marketSentiment = 'BAJISTA';
                sentimentColor = 'danger';
            }
            
            recommendations.innerHTML = `
                <div class="card bg-dark border-${sentimentColor}">
                    <div class="card-body text-center">
                        <h6 class="card-title">Sentimiento del Mercado</h6>
                        <div class="h4 text-${sentimentColor}">${marketSentiment}</div>
                        <small class="text-muted">${totalSignals} señales activas</small>
                    </div>
                </div>
            `;
        })
        .catch(error => {
            console.error('Error actualizando recomendaciones:', error);
        });
}

function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const scalpingAlerts = document.getElementById('scalping-alerts');
            
            if (data.alerts && data.alerts.length > 0) {
                scalpingAlerts.innerHTML = data.alerts.slice(0, 3).map(alert => `
                    <div class="alert ${alert.signal === 'LONG' ? 'alert-success' : 'alert-danger'} mb-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <strong>${alert.symbol}</strong>
                            <span class="badge ${alert.signal === 'LONG' ? 'bg-success' : 'bg-danger'}">${alert.signal}</span>
                        </div>
                        <small>Score: ${alert.score.toFixed(1)}% | Entrada: $${alert.entry.toFixed(2)}</small>
                    </div>
                `).join('');
            } else {
                scalpingAlerts.innerHTML = `
                    <div class="text-center text-muted">
                        <small>No hay alertas de scalping activas</small>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando alertas de scalping:', error);
        });
}

function updateExitSignals() {
    fetch('/api/exit_signals')
        .then(response => response.json())
        .then(data => {
            const exitSignals = document.getElementById('exit-signals');
            
            if (data.exit_signals && data.exit_signals.length > 0) {
                exitSignals.innerHTML = data.exit_signals.slice(0, 3).map(signal => `
                    <div class="alert alert-warning mb-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <strong>${signal.symbol}</strong>
                            <span class="badge bg-warning">SALIR</span>
                        </div>
                        <small>${signal.reason} | P&L: ${signal.pnl_percent.toFixed(2)}%</small>
                    </div>
                `).join('');
            } else {
                exitSignals.innerHTML = `
                    <div class="text-center text-muted">
                        <small>No hay señales de salida activas</small>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando señales de salida:', error);
        });
}

function formatVolume(volume) {
    if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    }
    return volume.toFixed(2);
}

function calculatePositionSize(data) {
    // Cálculo simplificado del tamaño de posición
    const riskPercent = 2; // 2% de riesgo
    const accountSize = 1000; // Tamaño de cuenta ejemplo
    const riskAmount = accountSize * (riskPercent / 100);
    const stopDistance = Math.abs(data.current_price - data.stop_loss);
    const positionSize = riskAmount / stopDistance;
    
    return Math.min(positionSize, accountSize / data.current_price);
}

function showError(message) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(container);
    }
    
    const toastHTML = `
        <div class="toast align-items-center text-bg-danger border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    document.getElementById('toast-container').innerHTML += toastHTML;
    
    const toastElement = document.querySelector('#toast-container .toast:last-child');
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}`;
    window.open(url, '_blank');
    
    showNotification('📊 Generando reporte completo...', 'info');
}

function downloadStrategicReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/strategic_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
    
    showNotification('🎯 Generando reporte estratégico...', 'info');
}

function showNotification(message, type = 'info') {
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
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

// Función para actualizar el reloj de Bolivia
function updateBoliviaClock() {
    fetch('/api/bolivia_time')
        .then(response => response.json())
        .then(data => {
            document.getElementById('bolivia-clock').textContent = data.time;
            document.getElementById('bolivia-date').textContent = data.date;
        })
        .catch(error => {
            console.error('Error actualizando reloj:', error);
            // Fallback a JavaScript local
            const now = new Date();
            document.getElementById('bolivia-clock').textContent = now.toLocaleTimeString('es-BO');
            document.getElementById('bolivia-date').textContent = now.toLocaleDateString('es-BO');
        });
}

// Actualizar el reloj cada segundo
setInterval(updateBoliviaClock, 1000);

// Inicializar el reloj al cargar la página
document.addEventListener('DOMContentLoaded', function() {
    updateBoliviaClock();
});
