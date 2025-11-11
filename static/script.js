// Configuración global
let currentChart = null;
let currentScatterChart = null;
let currentWhaleChart = null;
let currentAdxChart = null;
let currentRsiComparisonChart = null;
let currentMacdChart = null;
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
    updateSystemWinrate();
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

function updateSystemWinrate() {
    fetch('/api/winrate?symbol=BTC-USDT&interval=4h')
        .then(response => response.json())
        .then(data => {
            const winrateElement = document.querySelector('.winrate-value');
            const winrateLabel = document.querySelector('.winrate-label');
            
            if (data.winrate && data.winrate > 0) {
                winrateElement.textContent = `${data.winrate.toFixed(1)}%`;
                winrateElement.className = 'winrate-value display-4 fw-bold ' + 
                    (data.winrate >= 70 ? 'text-success' : 
                     data.winrate >= 60 ? 'text-warning' : 'text-danger');
                winrateLabel.textContent = `Performance Histórica (${data.interval})`;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
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
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-comparison-chart', 'macd-chart', 'trend-strength-chart', 'aux-chart'];
    
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
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-comparison-chart', 'macd-chart', 'trend-strength-chart', 'aux-chart'];
    
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
    
    // Cargar alertas de scalping
    updateScalpingAlerts();

    // Nueva línea cargada
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
            <p class="mt-2 mb-0 small">Verificando condiciones obligatorias...</p>
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
            updateSystemWinrate();
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
            renderWhaleChartImproved(data);
            renderAdxChartImproved(data);
            renderRsiComparisonChart(data);
            renderMacdChartImproved(data);
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
        rsi_traditional: 50,
        trend_strength_signal: 'NEUTRAL',
        no_trade_zone: false,
        obligatory_conditions_met: false,
        winrate: 0.0,
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
    
    // Calcular rango dinámico para el eje Y basado en los datos visibles
    const visibleHighs = highs.slice(-50);
    const visibleLows = lows.slice(-50);
    const minPrice = Math.min(...visibleLows);
    const maxPrice = Math.max(...visibleHighs);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.05;
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas Japonesas (${document.getElementById('interval-select').value})`,
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
    
    // Añadir marcadores para señales confirmadas
    const confirmedBuy = data.indicators.confirmed_buy || [];
    const confirmedSell = data.indicators.confirmed_sell || [];
    
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
            x: dates.filter((_, i) => confirmedBuy[i] && whalePump[i] > 0),
            y: whalePump.filter((_, i) => confirmedBuy[i] && whalePump[i] > 0),
            type: 'scatter',
            mode: 'markers',
            name: 'Señal Compra Confirmada',
            marker: {color: '#00FF00', size: 10, symbol: 'diamond'}
        },
        {
            x: dates.filter((_, i) => confirmedSell[i] && whaleDump[i] > 0),
            y: whaleDump.filter((_, i) => confirmedSell[i] && whaleDump[i] > 0),
            type: 'scatter',
            mode: 'markers',
            name: 'Señal Venta Confirmada',
            marker: {color: '#FF0000', size: 10, symbol: 'diamond'}
        }
    ];
    
    const layout = {
        title: {
            text: 'Indicador Ballenas - Compradoras vs Vendedoras',
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
    
    // Destruir gráfico existente
    if (currentAdxChart) {
        Plotly.purge('adx-chart');
    }
    
    currentAdxChart = Plotly.newPlot('adx-chart', traces, layout, config);
}

function renderRsiComparisonChart(data) {
    const chartElement = document.getElementById('rsi-comparison-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const rsiTraditional = data.indicators.rsi_traditional || [];
    const rsiMaverick = data.indicators.rsi_maverick || [];
    const bullishDivTraditional = data.indicators.bullish_divergence_rsi || [];
    const bearishDivTraditional = data.indicators.bearish_divergence_rsi || [];
    
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
            x: dates,
            y: rsiMaverick,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B)',
            line: {color: '#FF9800', width: 2}
        },
        {
            x: dates.filter((_, i) => bullishDivTraditional[i]),
            y: rsiTraditional.filter((_, i) => bullishDivTraditional[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Divergencia Alcista',
            marker: {color: '#00FF00', size: 8, symbol: 'triangle-up'}
        },
        {
            x: dates.filter((_, i) => bearishDivTraditional[i]),
            y: rsiTraditional.filter((_, i) => bearishDivTraditional[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Divergencia Bajista',
            marker: {color: '#FF0000', size: 8, symbol: 'triangle-down'}
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Tradicional vs RSI Maverick (Bollinger %B)',
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
            zerolinecolor: '#444',
            range: [0, 1]
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
    if (currentRsiComparisonChart) {
        Plotly.purge('rsi-comparison-chart');
    }
    
    currentRsiComparisonChart = Plotly.newPlot('rsi-comparison-chart', traces, layout, config);
}

function renderMacdChartImproved(data) {
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
            line: {color: '#FF5722', width: 1.5}
        },
        {
            x: dates,
            y: macdHistogram,
            type: 'bar',
            name: 'Histograma',
            marker: {
                color: macdHistogram.map(val => val >= 0 ? '#00C853' : '#FF1744')
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'MACD con Histograma de Colores',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Valor MACD',
            gridcolor: '#444',
            zerolinecolor: '#444',
            zeroline: true,
            zerolinewidth: 1
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
    if (currentMacdChart) {
        Plotly.purge('macd-chart');
    }
    
    currentMacdChart = Plotly.newPlot('macd-chart', traces, layout, config);
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
    const colors = data.indicators.colors || [];
    const noTradeZones = data.indicators.no_trade_zones || [];
    
    const traces = [
        {
            x: dates,
            y: trendStrength,
            type: 'bar',
            name: 'Fuerza de Tendencia',
            marker: {
                color: colors
            }
        },
        {
            x: dates.filter((_, i) => noTradeZones[i]),
            y: trendStrength.filter((_, i) => noTradeZones[i]),
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
            title: 'Fuerza de Tendencia (%)',
            gridcolor: '#444',
            zerolinecolor: '#444',
            zeroline: true,
            zerolinewidth: 1
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
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', traces, layout, config);
}

function updateAuxChart() {
    const indicatorType = document.getElementById('aux-indicator').value;
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
    
    switch(indicatorType) {
        case 'rsi':
            const rsiTraditional = currentData.indicators.rsi_traditional || [];
            traces = [{
                x: dates,
                y: rsiTraditional,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI Tradicional',
                line: {color: '#2196F3', width: 2}
            }];
            title = 'RSI Tradicional (14 periodos)';
            break;
            
        case 'macd':
            const macdLine = currentData.indicators.macd_line || [];
            const macdSignal = currentData.indicators.macd_signal || [];
            traces = [
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
                    line: {color: '#FF5722', width: 1.5}
                }
            ];
            title = 'MACD (12,26,9)';
            break;
            
        case 'squeeze':
            const squeezeMomentum = currentData.indicators.squeeze_momentum || [];
            const squeezeOn = currentData.indicators.squeeze_on || [];
            traces = [
                {
                    x: dates,
                    y: squeezeMomentum,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Squeeze Momentum',
                    line: {color: '#9C27B0', width: 2}
                },
                {
                    x: dates.filter((_, i) => squeezeOn[i]),
                    y: squeezeMomentum.filter((_, i) => squeezeOn[i]),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Squeeze ON',
                    marker: {color: '#FF0000', size: 6}
                }
            ];
            title = 'Squeeze Momentum';
            break;
            
        case 'moving_averages':
            const ma9 = currentData.indicators.ma_9 || [];
            const ma21 = currentData.indicators.ma_21 || [];
            const ma50 = currentData.indicators.ma_50 || [];
            traces = [
                {
                    x: dates,
                    y: ma9,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MA 9',
                    line: {color: '#FF9800', width: 1}
                },
                {
                    x: dates,
                    y: ma21,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MA 21',
                    line: {color: '#2196F3', width: 1}
                },
                {
                    x: dates,
                    y: ma50,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MA 50',
                    line: {color: '#9C27B0', width: 1}
                }
            ];
            title = 'Medias Móviles (9, 21, 50)';
            break;
    }
    
    const layout = {
        title: {
            text: title,
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
    if (currentAuxChart) {
        Plotly.purge('aux-chart');
    }
    
    currentAuxChart = Plotly.newPlot('aux-chart', traces, layout, config);
}

function updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/scatter_data_improved?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(scatterData => {
            renderScatterChartImproved(scatterData);
        })
        .catch(error => {
            console.error('Error cargando datos del scatter plot:', error);
        });
}

function renderScatterChartImproved(data) {
    const chartElement = document.getElementById('scatter-chart');
    
    if (!data || data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <h5>No hay datos disponibles</h5>
                <p>No se pudieron cargar los datos para el mapa de oportunidades.</p>
            </div>
        `;
        return;
    }

    const traces = [];
    const riskCategories = {
        'bajo': {color: '#00C853', name: 'Bajo Riesgo'},
        'medio': {color: '#FF9800', name: 'Medio Riesgo'},
        'alto': {color: '#FF1744', name: 'Alto Riesgo'},
        'memecoins': {color: '#9C27B0', name: 'Memecoins'}
    };
    
    // Crear trazas por categoría de riesgo
    Object.keys(riskCategories).forEach(category => {
        const categoryData = data.filter(item => item.risk_category === category);
        
        if (categoryData.length > 0) {
            traces.push({
                x: categoryData.map(item => item.x),
                y: categoryData.map(item => item.y),
                text: categoryData.map(item => 
                    `${item.symbol}<br>Score: ${item.signal_score}%<br>Winrate: ${item.winrate.toFixed(1)}%<br>Señal: ${item.signal}`
                ),
                mode: 'markers',
                type: 'scatter',
                name: riskCategories[category].name,
                marker: {
                    color: riskCategories[category].color,
                    size: categoryData.map(item => Math.max(8, item.signal_score / 3)),
                    sizemode: 'diameter',
                    sizeref: 8,
                    opacity: 0.7
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
            gridcolor: '#444',
            zerolinecolor: '#444',
            range: [0, 100]
        },
        yaxis: {
            title: 'Presión Vendedora (%)',
            gridcolor: '#444',
            zerolinecolor: '#444',
            range: [0, 100]
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
            // Zona de oportunidades LONG (esquina inferior derecha)
            {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: 70,
                y0: 0,
                x1: 100,
                y1: 30,
                fillcolor: 'rgba(0, 200, 83, 0.1)',
                line: {color: 'rgba(0, 200, 83, 0.5)'},
                linewidth: 1
            },
            // Zona de oportunidades SHORT (esquina superior izquierda)
            {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: 0,
                y0: 70,
                x1: 30,
                y1: 100,
                fillcolor: 'rgba(255, 23, 68, 0.1)',
                line: {color: 'rgba(255, 23, 68, 0.5)'},
                linewidth: 1
            }
        ],
        annotations: [
            {
                x: 85,
                y: 15,
                text: 'ZONA<br>LONG',
                showarrow: false,
                font: {color: '#00C853', size: 12},
                bgcolor: 'rgba(0, 0, 0, 0.7)',
                bordercolor: '#00C853'
            },
            {
                x: 15,
                y: 85,
                text: 'ZONA<br>SHORT',
                showarrow: false,
                font: {color: '#FF1744', size: 12},
                bgcolor: 'rgba(0, 0, 0, 0.7)',
                bordercolor: '#FF1744'
            }
        ]
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        toImageButtonOptions: {
            format: 'png',
            filename: 'scatter_opportunities_map',
            height: 600,
            width: 800,
            scale: 2
        }
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
                    No hay señales disponibles
                </td>
            </tr>
        `;
        return;
    }
    
    let html = '';
    signals.slice(0, 5).forEach((signal, index) => {
        const scoreClass = signal.signal_score >= 80 ? 'text-success' : 
                          signal.signal_score >= 70 ? 'text-warning' : 'text-danger';
        
        html += `
            <tr style="cursor: pointer;" onclick="showSignalDetails('${signal.symbol}', '${signal.signal}')">
                <td class="text-center">${index + 1}</td>
                <td>
                    <small>${signal.symbol}</small>
                    ${signal.winrate > 70 ? '<span class="badge bg-success ms-1">★</span>' : ''}
                </td>
                <td class="${scoreClass} fw-bold">${signal.signal_score.toFixed(1)}%</td>
                <td class="text-end">
                    <small>${signal.entry ? signal.entry.toFixed(6) : 'N/A'}</small>
                </td>
            </tr>
        `;
    });
    
    tableBody.innerHTML = html;
}

function showSignalDetails(symbol, signalType) {
    // Aquí puedes implementar la lógica para mostrar detalles de la señal
    // Por ahora, simplemente seleccionamos la crypto y actualizamos los gráficos
    selectCrypto(symbol);
    showNotification(`Analizando ${symbol} - Señal ${signalType}`, 'info');
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    
    if (!data) {
        marketSummary.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos disponibles</p>
            </div>
        `;
        return;
    }
    
    const signalClass = data.signal === 'LONG' ? 'text-success' : 
                       data.signal === 'SHORT' ? 'text-danger' : 'text-muted';
    
    const signalIcon = data.signal === 'LONG' ? '🟢' : 
                      data.signal === 'SHORT' ? '🔴' : '⚪';
    
    const scoreClass = data.signal_score >= 80 ? 'text-success' : 
                      data.signal_score >= 70 ? 'text-warning' : 'text-danger';
    
    const riskLevel = getRiskLevel(data.atr_percentage);
    
    marketSummary.innerHTML = `
        <div class="row text-center mb-3">
            <div class="col-12">
                <h4 class="${signalClass}">
                    ${signalIcon} ${data.signal}
                </h4>
                <div class="display-6 fw-bold ${scoreClass}">
                    ${data.signal_score.toFixed(1)}%
                </div>
                <small class="text-muted">Score de Confianza</small>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="d-flex justify-content-between mb-2">
                    <small>Precio Actual:</small>
                    <small class="fw-bold">$${data.current_price.toFixed(6)}</small>
                </div>
                <div class="d-flex justify-content-between mb-2">
                    <small>Volumen:</small>
                    <small class="fw-bold">${formatVolume(data.volume)}</small>
                </div>
                <div class="d-flex justify-content-between mb-2">
                    <small>ATR:</small>
                    <small class="fw-bold ${riskLevel.class}">${(data.atr_percentage * 100).toFixed(2)}%</small>
                </div>
                <div class="d-flex justify-content-between mb-2">
                    <small>Winrate:</small>
                    <small class="fw-bold ${data.winrate >= 70 ? 'text-success' : data.winrate >= 60 ? 'text-warning' : 'text-danger'}">
                        ${data.winrate.toFixed(1)}%
                    </small>
                </div>
                <div class="d-flex justify-content-between">
                    <small>Condiciones:</small>
                    <small class="fw-bold ${data.obligatory_conditions_met ? 'text-success' : 'text-danger'}">
                        ${data.obligatory_conditions_met ? '✅' : '❌'}
                    </small>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <div class="progress mb-2" style="height: 8px;">
                    <div class="progress-bar ${scoreClass}" 
                         style="width: ${data.signal_score}%">
                    </div>
                </div>
                <small class="text-muted">Fuerza de la Señal</small>
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
    
    let analysisHTML = '';
    
    if (data.signal === 'NEUTRAL') {
        analysisHTML = `
            <div class="text-center">
                <div class="text-muted mb-2">
                    <i class="fas fa-pause-circle fa-2x"></i>
                </div>
                <h6 class="text-muted">SIN SEÑAL ACTIVA</h6>
                <p class="small text-muted">
                    El análisis no detecta oportunidades de trading con suficiente confianza en este momento.
                </p>
            </div>
        `;
    } else {
        const signalColor = data.signal === 'LONG' ? 'success' : 'danger';
        const signalDirection = data.signal === 'LONG' ? 'ALCISTA' : 'BAJISTA';
        
        analysisHTML = `
            <div class="alert alert-${signalColor} border-${signalColor}">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h6 class="mb-0">SEÑAL ${signalDirection} CONFIRMADA</h6>
                    <span class="badge bg-${signalColor}">${data.signal_score.toFixed(1)}%</span>
                </div>
                
                <div class="row small">
                    <div class="col-6">
                        <strong>Entrada:</strong><br>
                        $${data.entry.toFixed(6)}
                    </div>
                    <div class="col-6">
                        <strong>Stop Loss:</strong><br>
                        $${data.stop_loss.toFixed(6)}
                    </div>
                </div>
                
                <div class="row small mt-2">
                    <div class="col-6">
                        <strong>Take Profit:</strong><br>
                        $${data.take_profit[0].toFixed(6)}
                    </div>
                    <div class="col-6">
                        <strong>Riesgo:</strong><br>
                        ${calculateRiskReward(data).toFixed(2)}:1
                    </div>
                </div>
                
                ${data.fulfilled_conditions.length > 0 ? `
                    <div class="mt-2">
                        <small><strong>Condiciones Cumplidas:</strong></small>
                        <div class="mt-1">
                            ${data.fulfilled_conditions.slice(0, 3).map(cond => 
                                `<span class="badge bg-success me-1 mb-1 small">${cond}</span>`
                            ).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    signalAnalysis.innerHTML = analysisHTML;
}

function updateObligatoryConditions(data) {
    const obligatoryConditions = document.getElementById('obligatory-conditions');
    
    if (!data) {
        obligatoryConditions.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de condiciones</p>
            </div>
        `;
        return;
    }
    
    const conditionsMet = data.obligatory_conditions_met;
    const noTradeZone = data.no_trade_zone;
    const trendStrength = data.trend_strength_signal;
    
    let conditionsHTML = '';
    
    if (conditionsMet && !noTradeZone) {
        conditionsHTML = `
            <div class="alert alert-success border-success">
                <div class="d-flex align-items-center">
                    <i class="fas fa-check-circle fa-2x me-3"></i>
                    <div>
                        <h6 class="mb-1">CONDICIONES OBLIGATORIAS: CUMPLIDAS</h6>
                        <p class="mb-0 small">
                            ✅ Multi-temporalidad confirmada<br>
                            ✅ ${trendStrength.replace('_', ' ')}<br>
                            ✅ Sin zona de NO OPERAR
                        </p>
                    </div>
                </div>
            </div>
        `;
    } else {
        const reasons = [];
        if (!conditionsMet) reasons.push('Condiciones multi-temporalidad no cumplidas');
        if (noTradeZone) reasons.push('Zona de NO OPERAR activa');
        if (trendStrength.includes('WEAK')) reasons.push('Fuerza de tendencia débil');
        
        conditionsHTML = `
            <div class="alert alert-danger border-danger">
                <div class="d-flex align-items-center">
                    <i class="fas fa-times-circle fa-2x me-3"></i>
                    <div>
                        <h6 class="mb-1">CONDICIONES OBLIGATORIAS: NO CUMPLIDAS</h6>
                        <p class="mb-0 small">
                            ${reasons.map(reason => `❌ ${reason}`).join('<br>')}
                        </p>
                    </div>
                </div>
            </div>
        `;
    }
    
    obligatoryConditions.innerHTML = conditionsHTML;
}

function updateFearGreedIndex() {
    // Simulación del índice de miedo y codicia
    // En una implementación real, esto vendría de una API
    const fearGreedIndex = document.getElementById('fear-greed-index');
    
    const indexValue = Math.floor(Math.random() * 100);
    let level = '';
    let color = '';
    let description = '';
    
    if (indexValue >= 75) {
        level = 'Codicia Extrema';
        color = 'danger';
        description = 'Mercado sobrecomprado - Posible corrección';
    } else if (indexValue >= 55) {
        level = 'Codicia';
        color = 'warning';
        description = 'Mercado optimista - Cautela recomendada';
    } else if (indexValue >= 45) {
        level = 'Neutral';
        color = 'info';
        description = 'Mercado equilibrado - Buenas condiciones';
    } else if (indexValue >= 25) {
        level = 'Miedo';
        color = 'primary';
        description = 'Mercado pesimista - Oportunidades emergentes';
    } else {
        level = 'Miedo Extremo';
        color = 'success';
        description = 'Mercado sobrevendido - Oportunidad de compra';
    }
    
    fearGreedIndex.innerHTML = `
        <div class="text-center">
            <div class="display-6 fw-bold text-${color}">${indexValue}</div>
            <div class="small text-${color} mb-2">${level}</div>
            <div class="progress mb-2" style="height: 8px;">
                <div class="progress-bar bg-${color}" style="width: ${indexValue}%"></div>
            </div>
            <small class="text-muted">${description}</small>
        </div>
    `;
}

function updateMarketRecommendations() {
    const marketRecommendations = document.getElementById('market-recommendations');
    
    // Simulación de recomendaciones de mercado
    const recommendations = [
        {type: 'success', icon: '🟢', text: 'BTC: Fuerte tendencia alcista en 4H'},
        {type: 'warning', icon: '🟡', text: 'ETH: Consolidación - Esperar breakout'},
        {type: 'danger', icon: '🔴', text: 'Altcoins: Alta volatilidad - Riesgo elevado'},
        {type: 'info', icon: '🔵', text: 'DXY: En niveles clave - Monitorear'}
    ];
    
    let html = '';
    recommendations.forEach(rec => {
        html += `
            <div class="alert alert-${rec.type} border-${rec.type} py-2 mb-2">
                <div class="d-flex align-items-center">
                    <span class="me-2">${rec.icon}</span>
                    <small class="mb-0">${rec.text}</small>
                </div>
            </div>
        `;
    });
    
    marketRecommendations.innerHTML = html;
}

function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const scalpingAlerts = document.getElementById('scalping-alerts');
            
            if (!data.alerts || data.alerts.length === 0) {
                scalpingAlerts.innerHTML = `
                    <div class="text-center py-3">
                        <i class="fas fa-bed fa-2x text-muted mb-2"></i>
                        <p class="small text-muted mb-0">No hay alertas activas</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            data.alerts.slice(0, 5).forEach(alert => {
                const signalClass = alert.signal === 'LONG' ? 'success' : 'danger';
                const riskClass = alert.risk_category === 'bajo' ? 'success' : 
                                 alert.risk_category === 'medio' ? 'warning' : 
                                 alert.risk_category === 'alto' ? 'danger' : 'info';
                
                html += `
                    <div class="alert alert-${signalClass} border-${signalClass} py-2 mb-2">
                        <div class="d-flex justify-content-between align-items-start mb-1">
                            <strong class="small">${alert.symbol}</strong>
                            <span class="badge bg-${riskClass} small">${alert.risk_category}</span>
                        </div>
                        <div class="small">
                            <div class="d-flex justify-content-between">
                                <span>${alert.signal}</span>
                                <span class="fw-bold">${alert.score.toFixed(1)}%</span>
                            </div>
                            <div class="d-flex justify-content-between text-muted">
                                <span>${alert.interval}</span>
                                <span>x${alert.leverage}</span>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            scalpingAlerts.innerHTML = html;
        })
        .catch(error => {
            console.error('Error cargando alertas de scalping:', error);
        });
}

function updateExitSignals() {
    fetch('/api/exit_signals')
        .then(response => response.json())
        .then(data => {
            const exitSignals = document.getElementById('exit-signals');
            
            if (!data.exit_signals || data.exit_signals.length === 0) {
                exitSignals.innerHTML = `
                    <div class="text-center py-3">
                        <i class="fas fa-check-circle fa-2x text-muted mb-2"></i>
                        <p class="small text-muted mb-0">Sin señales de salida</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            data.exit_signals.slice(0, 3).forEach(signal => {
                const pnlClass = signal.pnl_percent >= 0 ? 'success' : 'danger';
                const pnlIcon = signal.pnl_percent >= 0 ? '📈' : '📉';
                
                html += `
                    <div class="alert alert-${pnlClass} border-${pnlClass} py-2 mb-2">
                        <div class="d-flex justify-content-between align-items-start mb-1">
                            <strong class="small">${signal.symbol}</strong>
                            <span class="badge bg-${pnlClass} small">
                                ${pnlIcon} ${signal.pnl_percent.toFixed(2)}%
                            </span>
                        </div>
                        <div class="small text-muted">
                            ${signal.reason}<br>
                            <small>${signal.timestamp}</small>
                        </div>
                    </div>
                `;
            });
            
            exitSignals.innerHTML = html;
        })
        .catch(error => {
            console.error('Error cargando señales de salida:', error);
        });
}

// Funciones utilitarias
function formatVolume(volume) {
    if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    }
    return volume.toFixed(2);
}

function getRiskLevel(atrPercentage) {
    const risk = atrPercentage * 100;
    if (risk >= 5) return {level: 'Alto', class: 'text-danger'};
    if (risk >= 2) return {level: 'Medio', class: 'text-warning'};
    return {level: 'Bajo', class: 'text-success'};
}

function calculateRiskReward(data) {
    if (!data.entry || !data.stop_loss || !data.take_profit || data.take_profit.length === 0) {
        return 0;
    }
    
    const risk = Math.abs(data.entry - data.stop_loss);
    const reward = Math.abs(data.take_profit[0] - data.entry);
    
    return risk > 0 ? reward / risk : 0;
}

function showError(message) {
    showNotification(message, 'danger');
}

function showNotification(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
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
    
    toastContainer.innerHTML += toastHTML;
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remover el toast del DOM después de que se oculte
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

function downloadReport() {
    const symbol = document.getElementById('selected-crypto').textContent || 'BTC-USDT';
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
    
    showNotification('📊 Generando reporte técnico completo...', 'info');
}

// Inicializar el reloj de Bolivia
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
