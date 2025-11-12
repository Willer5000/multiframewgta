// Configuración global MULTI-TIMEFRAME CRYPTO WGTA PRO
let currentChart = null;
let currentScatterChart = null;
let currentWhaleChart = null;
let currentAdxChart = null;
let currentRsiTraditionalChart = null;
let currentRsiMaverickChart = null;
let currentMacdChart = null;
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
    updateCalendarInfo();
    updateWinrateDisplay();
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

function updateWinrateDisplay() {
    // Actualizar display de winrate
    fetch('/api/winrate')
        .then(response => response.json())
        .then(data => {
            const winrateDisplay = document.getElementById('winrate-display');
            if (winrateDisplay) {
                winrateDisplay.innerHTML = `
                    <div class="winrate-value display-4 fw-bold text-success">${data.winrate.toFixed(1)}%</div>
                    <p class="small text-muted mb-0">Tasa de acierto histórico</p>
                    <div class="progress mt-2" style="height: 8px;">
                        <div class="progress-bar bg-success" role="progressbar" 
                             style="width: ${data.winrate}%" 
                             aria-valuenow="${data.winrate}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
            document.getElementById('winrate-display').innerHTML = `
                <div class="text-warning">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    <span>Winrate no disponible</span>
                </div>
            `;
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
    // Configurar controles de indicadores operativos
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
        'candle-chart', 'whale-chart', 'adx-chart', 
        'rsi-traditional-chart', 'rsi-maverick-chart', 
        'macd-chart', 'trend-strength-chart'
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
        'candle-chart', 'whale-chart', 'adx-chart', 
        'rsi-traditional-chart', 'rsi-maverick-chart', 
        'macd-chart', 'trend-strength-chart'
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
    
    // Cargar alertas de trading
    updateTradingAlerts();
    
    // Cargar señales de salida
    updateExitSignals();
    
    // Actualizar información del calendario
    updateCalendarInfo();
    
    // Actualizar winrate
    updateWinrateDisplay();
}

function showLoadingState() {
    document.getElementById('market-summary').innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Cargando...</span>
            </div>
            <p class="mt-2 mb-0">Analizando mercado multi-timeframe...</p>
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
            console.log('Actualización automática MULTI-TIMEFRAME (cada 90 segundos)');
            updateCharts();
            updateMarketIndicators();
            updateWinrateDisplay();
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
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateTradingAlerts();
    updateExitSignals();
    updateCalendarInfo();
    updateWinrateDisplay();
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
            renderRsiTraditionalChart(data);
            renderRsiMaverickChart(data);
            renderMacdChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
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
        multi_timeframe_ok: false,
        trend_strength_filter: false,
        whale_signal_ok: true,
        above_ma200: true,
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
    
    // Añadir indicadores operativos si están activados
    if (data.indicators) {
        const options = indicatorOptions || {
            showMA9: true,
            showMA21: true,
            showMA50: true,
            showMA200: true,
            showBB: true
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
            text: `${data.symbol} - Gráfico de Velas con Indicadores Operativos`,
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
            marker: {color: '#00FF00', size: 12, symbol: 'diamond'}
        },
        {
            x: dates.filter((_, i) => confirmedSell[i] && whaleDump[i] > 0),
            y: whaleDump.filter((_, i) => confirmedSell[i] && whaleDump[i] > 0),
            type: 'scatter',
            mode: 'markers',
            name: 'Señal Venta Confirmada',
            marker: {color: '#FF0000', size: 12, symbol: 'diamond'}
        }
    ];
    
    const layout = {
        title: {
            text: 'Indicador Ballenas Compradoras/Vendedoras',
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
            x: [dates[0], dates[dates.length - 1]],
            y: [25, 25],
            type: 'scatter',
            mode: 'lines',
            name: 'Umbral 25',
            line: {color: 'yellow', dash: 'dash', width: 1}
        }
    ];
    
    const layout = {
        title: {
            text: 'ADX con DMI (+D y -D) - Umbral 25',
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
    const bullishDivergence = data.indicators.bullish_divergence_rsi || [];
    const bearishDivergence = data.indicators.bearish_divergence_rsi || [];
    
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
                symbol: 'triangle-up'
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
                symbol: 'triangle-down'
            }
        },
        {
            x: [dates[0], dates[dates.length - 1]],
            y: [70, 70],
            type: 'scatter',
            mode: 'lines',
            name: 'Sobrecompra',
            line: {color: 'red', dash: 'dash', width: 1}
        },
        {
            x: [dates[0], dates[dates.length - 1]],
            y: [30, 30],
            type: 'scatter',
            mode: 'lines',
            name: 'Sobreventa',
            line: {color: 'green', dash: 'dash', width: 1}
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Tradicional (Periodo 14)',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'RSI',
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
    const bullishDivergence = data.indicators.bullish_divergence_maverick || [];
    const bearishDivergence = data.indicators.bearish_divergence_maverick || [];
    
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
                symbol: 'triangle-up'
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
                symbol: 'triangle-down'
            }
        },
        {
            x: [dates[0], dates[dates.length - 1]],
            y: [0.8, 0.8],
            type: 'scatter',
            mode: 'lines',
            name: 'Sobrecompra',
            line: {color: 'red', dash: 'dash', width: 1}
        },
        {
            x: [dates[0], dates[dates.length - 1]],
            y: [0.2, 0.2],
            type: 'scatter',
            mode: 'lines',
            name: 'Sobreventa',
            line: {color: 'green', dash: 'dash', width: 1}
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Modificado Maverick (%B)',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'RSI Maverick',
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
    const macd = data.indicators.macd || [];
    const macdSignal = data.indicators.macd_signal || [];
    const macdHistogram = data.indicators.macd_histogram || [];
    
    // Colores para el histograma
    const histogramColors = macdHistogram.map(value => 
        value >= 0 ? '#00C853' : '#FF1744'
    );
    
    const traces = [
        {
            x: dates,
            y: macd,
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: {color: '#FF9800', width: 2}
        },
        {
            x: dates,
            y: macdSignal,
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#2196F3', width: 1.5}
        },
        {
            x: dates,
            y: macdHistogram,
            type: 'bar',
            name: 'Histograma',
            marker: {color: histogramColors}
        }
    ];
    
    const layout = {
        title: {
            text: 'MACD (12,26,9) con Histograma',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'MACD',
            gridcolor: '#444',
            zerolinecolor: '#444',
            zeroline: true,
            zerolinewidth: 1,
            zerolinecolor: '#666'
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
        barmode: 'overlay'
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
    
    // Crear barras con colores según la fuerza
    const traces = [{
        x: dates,
        y: trendStrength,
        type: 'bar',
        marker: {
            color: colors,
            line: {
                color: 'rgba(255,255,255,0.3)',
                width: 0.5
            }
        },
        name: 'Fuerza Tendencia'
    }];
    
    // Añadir líneas para zonas de no operar
    noTradeZones.forEach((isNoTrade, index) => {
        if (isNoTrade && index < dates.length) {
            traces.push({
                x: [dates[index], dates[index]],
                y: [Math.min(...trendStrength), Math.max(...trendStrength)],
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: 'red',
                    width: 3,
                    dash: 'dot'
                },
                name: 'NO OPERAR',
                showlegend: index === 0
            });
        }
    });
    
    // Añadir umbral alto si está disponible
    if (data.indicators.high_zone_threshold) {
        traces.push({
            x: [dates[0], dates[dates.length - 1]],
            y: [data.indicators.high_zone_threshold, data.indicators.high_zone_threshold],
            type: 'scatter',
            mode: 'lines',
            line: {
                color: 'orange',
                width: 1,
                dash: 'dash'
            },
            name: 'Umbral Alto'
        });
        
        traces.push({
            x: [dates[0], dates[dates.length - 1]],
            y: [-data.indicators.high_zone_threshold, -data.indicators.high_zone_threshold],
            type: 'scatter',
            mode: 'lines',
            line: {
                color: 'orange',
                width: 1,
                dash: 'dash'
            },
            name: 'Umbral Bajo',
            showlegend: false
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
            title: 'Fuerza Tendencia %',
            gridcolor: '#444',
            zerolinecolor: '#444',
            zeroline: true,
            zerolinewidth: 1,
            zerolinecolor: '#666'
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

function updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    fetch(`/api/scatter_data_improved?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`)
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
                <p class="mb-0">No hay datos disponibles para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }
    
    // Agrupar por categoría de riesgo
    const traces = {};
    const riskCategories = {
        'bajo': { name: 'Bajo Riesgo', color: '#00C853' },
        'medio': { name: 'Medio Riesgo', color: '#FFC107' },
        'alto': { name: 'Alto Riesgo', color: '#FF1744' },
        'memecoins': { name: 'Memecoins', color: '#9C27B0' }
    };
    
    Object.keys(riskCategories).forEach(category => {
        traces[category] = {
            x: [],
            y: [],
            text: [],
            mode: 'markers',
            type: 'scatter',
            name: riskCategories[category].name,
            marker: {
                size: 12,
                color: riskCategories[category].color,
                line: {
                    color: 'white',
                    width: 1
                }
            },
            hovertemplate: '<b>%{text}</b><br>Compra: %{x:.1f}%<br>Venta: %{y:.1f}%<extra></extra>'
        };
    });
    
    // Agregar datos a las trazas
    data.forEach(item => {
        const category = item.risk_category;
        if (traces[category]) {
            traces[category].x.push(item.buy_pressure);
            traces[category].y.push(item.sell_pressure);
            traces[category].text.push(`${item.symbol} (Score: ${item.signal_score.toFixed(1)}%)`);
        }
    });
    
    // Convertir objeto de trazas a array
    const tracesArray = Object.values(traces);
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Análisis Multi-Indicador (40 Criptomonedas)',
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
            // Línea diagonal
            {
                type: 'line',
                x0: 0,
                y0: 0,
                x1: 100,
                y1: 100,
                line: {
                    color: 'white',
                    width: 1,
                    dash: 'dot'
                }
            },
            // Zona de compra (esquina inferior derecha)
            {
                type: 'rect',
                x0: 70,
                y0: 0,
                x1: 100,
                y1: 30,
                line: {
                    color: 'rgba(0, 200, 83, 0.3)',
                    width: 1,
                },
                fillcolor: 'rgba(0, 200, 83, 0.1)',
            },
            // Zona de venta (esquina superior izquierda)
            {
                type: 'rect',
                x0: 0,
                y0: 70,
                x1: 30,
                y1: 100,
                line: {
                    color: 'rgba(255, 23, 68, 0.3)',
                    width: 1,
                },
                fillcolor: 'rgba(255, 23, 68, 0.1)',
            }
        ],
        annotations: [
            {
                x: 85,
                y: 15,
                text: 'ZONA COMPRA',
                showarrow: false,
                font: {
                    color: '#00C853',
                    size: 12
                }
            },
            {
                x: 15,
                y: 85,
                text: 'ZONA VENTA',
                showarrow: false,
                font: {
                    color: '#FF1744',
                    size: 12
                }
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
    
    currentScatterChart = Plotly.newPlot('scatter-chart', tracesArray, layout, config);
}

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    fetch(`/api/multiple_signals?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`)
        .then(response => response.json())
        .then(data => {
            updateSignalTables(data.long_signals, data.short_signals);
        })
        .catch(error => {
            console.error('Error actualizando señales múltiples:', error);
        });
}

function updateSignalTables(longSignals, shortSignals) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    // Actualizar tabla LONG
    longTable.innerHTML = '';
    if (longSignals && longSignals.length > 0) {
        longSignals.slice(0, 5).forEach((signal, index) => {
            const row = document.createElement('tr');
            row.className = 'hover-row';
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-success">${signal.signal_score.toFixed(1)}%</span></td>
                <td>${signal.entry.toFixed(6)}</td>
            `;
            row.addEventListener('click', () => showSignalDetails(signal));
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
    shortTable.innerHTML = '';
    if (shortSignals && shortSignals.length > 0) {
        shortSignals.slice(0, 5).forEach((signal, index) => {
            const row = document.createElement('tr');
            row.className = 'hover-row';
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-danger">${signal.signal_score.toFixed(1)}%</span></td>
                <td>${signal.entry.toFixed(6)}</td>
            `;
            row.addEventListener('click', () => showSignalDetails(signal));
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
    const detailsElement = document.getElementById('signal-details');
    
    const conditionsList = signal.fulfilled_conditions.map(condition => 
        `<li class="text-success">${condition}</li>`
    ).join('');
    
    detailsElement.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Información Básica</h6>
                <table class="table table-sm table-dark">
                    <tr>
                        <td><strong>Símbolo:</strong></td>
                        <td>${signal.symbol}</td>
                    </tr>
                    <tr>
                        <td><strong>Señal:</strong></td>
                        <td><span class="badge ${signal.signal === 'LONG' ? 'bg-success' : 'bg-danger'}">${signal.signal}</span></td>
                    </tr>
                    <tr>
                        <td><strong>Score:</strong></td>
                        <td>${signal.signal_score.toFixed(1)}%</td>
                    </tr>
                    <tr>
                        <td><strong>Precio Actual:</strong></td>
                        <td>${signal.current_price.toFixed(6)}</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-6">
                <h6>Niveles de Trading</h6>
                <table class="table table-sm table-dark">
                    <tr>
                        <td><strong>Entrada:</strong></td>
                        <td>${signal.entry.toFixed(6)}</td>
                    </tr>
                    <tr>
                        <td><strong>Stop Loss:</strong></td>
                        <td>${signal.stop_loss.toFixed(6)}</td>
                    </tr>
                    <tr>
                        <td><strong>Take Profit:</strong></td>
                        <td>${signal.take_profit[0].toFixed(6)}</td>
                    </tr>
                    <tr>
                        <td><strong>ATR:</strong></td>
                        <td>${signal.atr.toFixed(6)} (${(signal.atr_percentage * 100).toFixed(1)}%)</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <h6>Condiciones Cumplidas</h6>
                <ul>
                    ${conditionsList}
                </ul>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <div class="alert ${signal.multi_timeframe_ok ? 'alert-success' : 'alert-warning'}">
                    <strong>Multi-Timeframe:</strong> ${signal.multi_timeframe_ok ? '✅ CONFIRMADO' : '❌ NO CONFIRMADO'}
                </div>
                <div class="alert ${signal.trend_strength_filter ? 'alert-success' : 'alert-warning'}">
                    <strong>Fuerza Tendencia:</strong> ${signal.trend_strength_filter ? '✅ FAVORABLE' : '❌ DESFAVORABLE'}
                </div>
                <div class="alert ${signal.whale_signal_ok ? 'alert-success' : 'alert-warning'}">
                    <strong>Señal Ballenas:</strong> ${signal.whale_signal_ok ? '✅ ACTIVA' : '❌ INACTIVA'}
                </div>
            </div>
        </div>
    `;
    
    modal.show();
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    
    if (!data || !data.symbol) {
        marketSummary.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos disponibles para el resumen</p>
            </div>
        `;
        return;
    }
    
    const signalBadge = data.signal === 'LONG' ? 
        '<span class="badge bg-success">LONG</span>' : 
        data.signal === 'SHORT' ? 
        '<span class="badge bg-danger">SHORT</span>' : 
        '<span class="badge bg-secondary">NEUTRAL</span>';
    
    const multiTfStatus = data.multi_timeframe_ok ? 
        '<span class="badge bg-success">✅</span>' : 
        '<span class="badge bg-danger">❌</span>';
    
    const trendStatus = data.trend_strength_filter ? 
        '<span class="badge bg-success">✅</span>' : 
        '<span class="badge bg-danger">❌</span>';
    
    const whaleStatus = data.whale_signal_ok ? 
        '<span class="badge bg-success">✅</span>' : 
        '<span class="badge bg-warning">⚠️</span>';
    
    const noTradeStatus = data.no_trade_zone ? 
        '<span class="badge bg-danger">🔴 NO OPERAR</span>' : 
        '<span class="badge bg-success">🟢 OPERAR</span>';
    
    marketSummary.innerHTML = `
        <div class="row text-center mb-3">
            <div class="col-12">
                <h4>${data.symbol}</h4>
                <h2 class="${data.signal === 'LONG' ? 'text-success' : data.signal === 'SHORT' ? 'text-danger' : 'text-secondary'}">
                    ${data.current_price.toFixed(6)}
                </h2>
                <div class="mb-2">
                    ${signalBadge}
                    <span class="badge ${data.signal_score >= 70 ? 'bg-success' : 'bg-warning'}">
                        Score: ${data.signal_score.toFixed(1)}%
                    </span>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <table class="table table-sm table-borderless">
                    <tr>
                        <td><strong>Señal:</strong></td>
                        <td>${signalBadge}</td>
                    </tr>
                    <tr>
                        <td><strong>Multi-Timeframe:</strong></td>
                        <td>${multiTfStatus}</td>
                    </tr>
                    <tr>
                        <td><strong>Fuerza Tendencia:</strong></td>
                        <td>${trendStatus}</td>
                    </tr>
                    <tr>
                        <td><strong>Ballenas:</strong></td>
                        <td>${whaleStatus}</td>
                    </tr>
                    <tr>
                        <td><strong>Zona Operar:</strong></td>
                        <td>${noTradeStatus}</td>
                    </tr>
                    <tr>
                        <td><strong>Volumen:</strong></td>
                        <td>${(data.volume / 1000).toFixed(0)}K</td>
                    </tr>
                    <tr>
                        <td><strong>ADX:</strong></td>
                        <td>${data.adx.toFixed(1)}</td>
                    </tr>
                    <tr>
                        <td><strong>RSI Trad:</strong></td>
                        <td>${data.rsi_traditional.toFixed(1)}</td>
                    </tr>
                    <tr>
                        <td><strong>RSI Mav:</strong></td>
                        <td>${data.rsi_maverick.toFixed(3)}</td>
                    </tr>
                </table>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (!data || data.signal === 'NEUTRAL') {
        signalAnalysis.innerHTML = `
            <div class="text-center">
                <div class="signal-neutral-enhanced p-3 rounded">
                    <h6 class="text-warning">🔍 ANÁLISIS DE SEÑAL</h6>
                    <p class="mb-2">No hay señales activas confirmadas</p>
                    <small class="text-muted">Esperando condiciones multi-timeframe favorables</small>
                </div>
            </div>
        `;
        return;
    }
    
    const isLong = data.signal === 'LONG';
    const signalClass = isLong ? 'signal-long-enhanced' : 'signal-short-enhanced';
    const signalIcon = isLong ? '📈' : '📉';
    const signalColor = isLong ? 'success' : 'danger';
    
    const conditionsCount = data.fulfilled_conditions ? data.fulfilled_conditions.length : 0;
    const totalConditions = 8; // Número total de condiciones posibles
    
    signalAnalysis.innerHTML = `
        <div class="${signalClass} p-3 rounded">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <h6 class="text-${signalColor} mb-0">${signalIcon} SEÑAL ${data.signal} CONFIRMADA</h6>
                <span class="badge bg-${signalColor}">${data.signal_score.toFixed(1)}%</span>
            </div>
            
            <div class="progress mb-2" style="height: 6px;">
                <div class="progress-bar bg-${signalColor}" 
                     style="width: ${data.signal_score}%"
                     role="progressbar">
                </div>
            </div>
            
            <div class="row text-center">
                <div class="col-6">
                    <small class="text-muted">Condiciones</small>
                    <div class="fw-bold">${conditionsCount}/${totalConditions}</div>
                </div>
                <div class="col-6">
                    <small class="text-muted">Fuerza</small>
                    <div class="fw-bold text-${signalColor}">${data.trend_strength_signal}</div>
                </div>
            </div>
            
            ${data.fulfilled_conditions && data.fulfilled_conditions.length > 0 ? `
                <div class="mt-2">
                    <small class="text-muted d-block">Condiciones clave:</small>
                    <div class="mt-1">
                        ${data.fulfilled_conditions.slice(0, 3).map(cond => 
                            `<span class="badge bg-${signalColor} bg-opacity-25 text-${signalColor} me-1 mb-1" style="font-size: 0.7rem;">${cond}</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function updateFearGreedIndex() {
    const fearGreedElement = document.getElementById('fear-greed-index');
    
    // Simular índice de miedo y codicia (en un sistema real esto vendría de una API)
    const fearGreedValue = Math.floor(Math.random() * 100);
    let status = '';
    let color = '';
    let description = '';
    
    if (fearGreedValue >= 80) {
        status = 'Extrema Codicia';
        color = 'danger';
        description = 'Mercado sobrecomprado - Posible corrección';
    } else if (fearGreedValue >= 60) {
        status = 'Codicia';
        color = 'warning';
        description = 'Mercado optimista';
    } else if (fearGreedValue >= 40) {
        status = 'Neutral';
        color = 'info';
        description = 'Mercado equilibrado';
    } else if (fearGreedValue >= 20) {
        status = 'Miedo';
        color = 'primary';
        description = 'Mercado pesimista';
    } else {
        status = 'Miedo Extremo';
        color = 'success';
        description = 'Oportunidad de compra';
    }
    
    fearGreedElement.innerHTML = `
        <div class="text-center">
            <div class="fear-greed-value display-4 fw-bold text-${color}">${fearGreedValue}</div>
            <div class="fear-greed-status h5 text-${color}">${status}</div>
            <div class="fear-greed-description small text-muted">${description}</div>
            <div class="fear-greed-progress mt-2">
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${fearGreedValue}%; background: linear-gradient(90deg, #dc3545, #ffc107, #198754);"
                         aria-valuenow="${fearGreedValue}" aria-valuemin="0" aria-valuemax="100">
                    </div>
                </div>
            </div>
        </div>
    `;
}

function updateMarketRecommendations() {
    const recommendationsElement = document.getElementById('market-recommendations');
    
    // Simular recomendaciones de mercado
    const recommendations = [
        { symbol: 'BTC-USDT', action: 'LONG', confidence: 85 },
        { symbol: 'ETH-USDT', action: 'LONG', confidence: 78 },
        { symbol: 'SOL-USDT', action: 'SHORT', confidence: 72 }
    ];
    
    let recommendationsHTML = `
        <div class="card bg-dark border-secondary mb-3">
            <div class="card-header">
                <h6 class="mb-0"><i class="fas fa-star me-2"></i>Recomendaciones Top</h6>
            </div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    <table class="table table-dark table-hover mb-0">
                        <thead>
                            <tr>
                                <th>Crypto</th>
                                <th>Acción</th>
                                <th>Confianza</th>
                            </tr>
                        </thead>
                        <tbody>
    `;
    
    recommendations.forEach(rec => {
        const actionClass = rec.action === 'LONG' ? 'success' : 'danger';
        recommendationsHTML += `
            <tr class="hover-row">
                <td>${rec.symbol}</td>
                <td><span class="badge bg-${actionClass}">${rec.action}</span></td>
                <td>
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar bg-${actionClass}" 
                             style="width: ${rec.confidence}%"
                             role="progressbar">
                        </div>
                    </div>
                    <small>${rec.confidence}%</small>
                </td>
            </tr>
        `;
    });
    
    recommendationsHTML += `
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    recommendationsElement.innerHTML = recommendationsHTML;
}

function updateTradingAlerts() {
    const alertsElement = document.getElementById('scalping-alerts');
    
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const alerts = data.alerts || [];
            
            if (alerts.length === 0) {
                alertsElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="fas fa-bell-slash fa-2x mb-2"></i>
                        <p class="mb-0 small">No hay alertas activas</p>
                    </div>
                `;
                return;
            }
            
            let alertsHTML = '';
            alerts.slice(0, 5).forEach(alert => {
                const signalClass = alert.signal === 'LONG' ? 'success' : 'danger';
                alertsHTML += `
                    <div class="scalping-alert mb-2 p-2 rounded">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <strong class="text-${signalClass}">${alert.symbol}</strong>
                                <br>
                                <small>${alert.interval} • ${alert.signal}</small>
                            </div>
                            <div class="text-end">
                                <span class="badge bg-${signalClass}">${alert.score.toFixed(1)}%</span>
                                <br>
                                <small class="text-muted">x${alert.leverage}</small>
                            </div>
                        </div>
                        <div class="mt-1">
                            <small class="text-muted">
                                Entrada: ${alert.entry.toFixed(6)}
                            </small>
                        </div>
                    </div>
                `;
            });
            
            alertsElement.innerHTML = alertsHTML;
        })
        .catch(error => {
            console.error('Error cargando alertas:', error);
            alertsElement.innerHTML = `
                <div class="text-center text-warning py-3">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    <span>Error cargando alertas</span>
                </div>
            `;
        });
}

function updateExitSignals() {
    const exitSignalsElement = document.getElementById('exit-signals');
    
    fetch('/api/exit_signals')
        .then(response => response.json())
        .then(data => {
            const exitSignals = data.exit_signals || [];
            
            if (exitSignals.length === 0) {
                exitSignalsElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="fas fa-check-circle fa-2x mb-2"></i>
                        <p class="mb-0 small">No hay señales de salida activas</p>
                    </div>
                `;
                return;
            }
            
            let exitHTML = '';
            exitSignals.slice(0, 5).forEach(signal => {
                const pnlClass = signal.pnl_percent >= 0 ? 'success' : 'danger';
                const pnlIcon = signal.pnl_percent >= 0 ? '📈' : '📉';
                
                exitHTML += `
                    <div class="mb-2 p-2 rounded border border-${pnlClass} border-opacity-25">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <strong>${signal.symbol}</strong>
                                <br>
                                <small>${signal.interval} • ${signal.signal}</small>
                            </div>
                            <div class="text-end">
                                <span class="badge bg-${pnlClass}">${pnlIcon} ${signal.pnl_percent.toFixed(2)}%</span>
                            </div>
                        </div>
                        <div class="mt-1">
                            <small class="text-muted">${signal.reason}</small>
                        </div>
                        <div class="mt-1">
                            <small class="text-muted">
                                ${signal.timestamp}
                            </small>
                        </div>
                    </div>
                `;
            });
            
            exitSignalsElement.innerHTML = exitHTML;
        })
        .catch(error => {
            console.error('Error cargando señales de salida:', error);
            exitSignalsElement.innerHTML = `
                <div class="text-center text-warning py-3">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    <span>Error cargando señales de salida</span>
                </div>
            `;
        });
}

function showError(message) {
    // Crear toast de error
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'toast-' + Date.now();
    
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-bg-danger border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-exclamation-triangle me-2"></i>
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
    
    // Remover el toast después de que se oculte
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function showSuccess(message) {
    // Crear toast de éxito
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'toast-' + Date.now();
    
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-bg-success border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-check-circle me-2"></i>
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
    
    // Remover el toast después de que se oculte
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

function downloadStrategicReport() {
    const symbol = currentSymbol || 'BTC-USDT';
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}&strategic=true`;
    window.open(url, '_blank');
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
            const now = new Date();
            document.getElementById('bolivia-clock').textContent = now.toLocaleTimeString('es-BO');
            document.getElementById('bolivia-date').textContent = now.toLocaleDateString('es-BO');
        });
}

// Función para actualizar el winrate
function updateWinrate() {
    fetch('/api/winrate')
        .then(response => response.json())
        .then(data => {
            const winrateDisplay = document.getElementById('winrate-display');
            winrateDisplay.innerHTML = `
                <div class="winrate-value display-4 fw-bold text-success">${data.winrate.toFixed(1)}%</div>
                <p class="small text-muted mb-0">Tasa de acierto histórico</p>
            `;
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
            document.getElementById('winrate-display').innerHTML = `
                <div class="text-warning">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    <span>Winrate no disponible</span>
                </div>
            `;
        });
}

// Actualizar el reloj cada segundo
setInterval(updateBoliviaClock, 1000);

// Actualizar winrate cada 2 minutos
setInterval(updateWinrate, 120000);

// Inicializar el reloj y winrate al cargar la página
document.addEventListener('DOMContentLoaded', function() {
    updateBoliviaClock();
    updateWinrate();
});
