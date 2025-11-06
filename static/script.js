// MULTI-TIMEFRAME CRYPTO WGTA PRO - Script Principal
// Sistema Profesional de Trading con Análisis Multi-Temporalidad

// Configuración global
let currentChart = null;
let currentScatterChart = null;
let currentTrendStrengthChart = null;
let currentRsiChart = null;
let currentAdxChart = null;
let currentAuxChart = null;
let currentSymbol = 'BTC-USDT';
let currentData = null;
let allCryptos = [];
let updateInterval = null;
let drawingToolsActive = false;

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializando Sistema Profesional');
    initializeApp();
    setupEventListeners();
    updateCharts();
    startAutoUpdate();
});

function initializeApp() {
    console.log('✅ Sistema inicializado correctamente');
    loadCryptoRiskClassification();
    updateCalendarInfo();
    updateWinrateDisplay();
    loadMarketIndicators();
    
    // Mostrar notificación de bienvenida
    setTimeout(() => {
        showNotification('Sistema MULTI-TIMEFRAME CRYPTO WGTA PRO cargado correctamente', 'success');
    }, 1000);
}

function setupEventListeners() {
    // Configurar event listeners para los controles principales
    const controls = [
        'interval-select', 'di-period', 'adx-threshold', 'sr-period',
        'rsi-length', 'bb-multiplier', 'volume-filter', 'leverage'
    ];
    
    controls.forEach(controlId => {
        const element = document.getElementById(controlId);
        if (element) {
            element.addEventListener('change', updateCharts);
        }
    });
    
    // Configurar selector de indicador auxiliar
    const auxIndicator = document.getElementById('aux-indicator');
    if (auxIndicator) {
        auxIndicator.addEventListener('change', updateAuxChart);
    }
    
    // Configurar buscador de cryptos
    setupCryptoSearch();
    
    // Configurar herramientas de dibujo
    setupDrawingTools();
    
    // Configurar controles de indicadores informativos
    setupIndicatorControls();
    
    // Configurar eventos de teclado
    setupKeyboardShortcuts();
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
    const cryptoList = document.getElementById('crypto-list');
    
    if (searchInput && cryptoList) {
        searchInput.addEventListener('input', function() {
            const filter = this.value.toUpperCase();
            filterCryptoList(filter);
        });
        
        // Prevenir que el dropdown se cierre al hacer clic en el buscador
        searchInput.addEventListener('click', function(e) {
            e.stopPropagation();
        });
    }
}

function setupDrawingTools() {
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
    const indicatorControls = document.querySelectorAll('.indicator-control');
    indicatorControls.forEach(control => {
        control.addEventListener('change', function() {
            updateChartIndicators();
        });
    });
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl+R o F5 para actualizar
        if ((e.ctrlKey && e.key === 'r') || e.key === 'F5') {
            e.preventDefault();
            updateCharts();
            showNotification('Gráficos actualizados', 'info');
        }
        
        // Escape para desactivar herramientas de dibujo
        if (e.key === 'Escape' && drawingToolsActive) {
            deactivateDrawingTools();
            showNotification('Herramientas de dibujo desactivadas', 'warning');
        }
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
    const dragModes = {
        'line': 'drawline',
        'rectangle': 'drawrect',
        'circle': 'drawcircle',
        'text': 'drawtext',
        'freehand': 'drawfreehand',
        'marker': 'marker'
    };
    
    const dragmode = dragModes[tool] || false;
    
    // Aplicar a todos los gráficos
    const chartIds = ['candle-chart', 'trend-strength-chart', 'rsi-maverick-chart', 'adx-chart', 'aux-chart'];
    
    chartIds.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart) {
            Plotly.relayout(chartId, {dragmode: dragmode});
        }
    });
    
    showNotification(`Herramienta de dibujo activada: ${tool}`, 'info');
}

function deactivateDrawingTools() {
    drawingToolsActive = false;
    document.querySelectorAll('.drawing-tool').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const chartIds = ['candle-chart', 'trend-strength-chart', 'rsi-maverick-chart', 'adx-chart', 'aux-chart'];
    chartIds.forEach(chartId => {
        Plotly.relayout(chartId, {dragmode: false});
    });
}

function setDrawingColor(color) {
    const chartIds = ['candle-chart', 'trend-strength-chart', 'rsi-maverick-chart', 'adx-chart', 'aux-chart'];
    
    chartIds.forEach(chartId => {
        Plotly.relayout(chartId, {
            'newshape.line.color': color,
            'newshape.fillcolor': color + '33'
        });
    });
}

function updateChartIndicators() {
    const showMA9 = document.getElementById('show-ma9')?.checked || false;
    const showMA21 = document.getElementById('show-ma21')?.checked || false;
    const showMA50 = document.getElementById('show-ma50')?.checked || false;
    const showMA200 = document.getElementById('show-ma200')?.checked || false;
    const showBB = document.getElementById('show-bollinger')?.checked || false;
    
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
    if (!cryptoList) return;
    
    cryptoList.innerHTML = '';
    
    const filteredCryptos = allCryptos.filter(crypto => 
        crypto.symbol.toUpperCase().includes(filter)
    );
    
    if (filteredCryptos.length === 0) {
        cryptoList.innerHTML = `
            <div class="dropdown-item text-muted text-center">
                <i class="fas fa-search me-1"></i>No se encontraron resultados
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
        
        const categoryConfig = {
            'bajo': { icon: '🟢', className: 'text-success', text: 'BAJO RIESGO' },
            'medio': { icon: '🟡', className: 'text-warning', text: 'MEDIO RIESGO' },
            'alto': { icon: '🔴', className: 'text-danger', text: 'ALTO RIESGO' },
            'memecoins': { icon: '🟣', className: 'text-info', text: 'MEMECOINS' }
        };
        
        const config = categoryConfig[category] || categoryConfig['medio'];
        
        categoryDiv.innerHTML = `${config.icon} ${config.text}`;
        categoryDiv.classList.add(config.className, 'small');
        cryptoList.appendChild(categoryDiv);
        
        categories[category].forEach(crypto => {
            const item = document.createElement('a');
            item.className = 'dropdown-item crypto-item';
            item.href = '#';
            item.innerHTML = `
                ${crypto.symbol}
                <small class="text-muted float-end">${getRiskLevelText(crypto.category)}</small>
            `;
            item.addEventListener('click', function(e) {
                e.preventDefault();
                selectCrypto(crypto.symbol);
            });
            cryptoList.appendChild(item);
        });
        
        cryptoList.appendChild(document.createElement('div')).className = 'dropdown-divider';
    });
}

function getRiskLevelText(category) {
    const texts = {
        'bajo': 'Bajo',
        'medio': 'Medio', 
        'alto': 'Alto',
        'memecoins': 'Memecoin'
    };
    return texts[category] || 'Medio';
}

function selectCrypto(symbol) {
    currentSymbol = symbol;
    const selectedCryptoElement = document.getElementById('selected-crypto');
    if (selectedCryptoElement) {
        selectedCryptoElement.textContent = symbol;
    }
    
    // Cerrar el dropdown
    const dropdownElement = document.getElementById('cryptoDropdown');
    if (dropdownElement) {
        const bootstrapDropdown = bootstrap.Dropdown.getInstance(dropdownElement);
        if (bootstrapDropdown) {
            bootstrapDropdown.hide();
        }
    }
    
    updateCharts();
    showNotification(`Crypto seleccionada: ${symbol}`, 'info');
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
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'XRP-USDT', 'ADA-USDT',
        'SOL-USDT', 'DOT-USDT', 'DOGE-USDT', 'AVAX-USDT', 'LINK-USDT'
    ];
    
    allCryptos = basicSymbols.map(symbol => ({
        symbol: symbol,
        category: 'bajo'
    }));
    
    filterCryptoList('');
}

function updateCalendarInfo() {
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

function loadMarketIndicators() {
    updateScalpingAlerts();
    updateExitSignals();
    updateCalendarInfo();
}

function showLoadingState() {
    const marketSummary = document.getElementById('market-summary');
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (marketSummary) {
        marketSummary.innerHTML = `
            <div class="text-center py-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Cargando...</span>
                </div>
                <p class="mt-2 mb-0">Analizando mercado...</p>
            </div>
        `;
    }
    
    if (signalAnalysis) {
        signalAnalysis.innerHTML = `
            <div class="text-center py-3">
                <div class="spinner-border spinner-border-sm text-info" role="status">
                    <span class="visually-hidden">Analizando...</span>
                </div>
                <p class="text-muted mb-0 small">Evaluando condiciones de señal...</p>
            </div>
        `;
    }
}

function startAutoUpdate() {
    // Detener intervalo anterior si existe
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Configurar actualización automática cada 90 segundos
    updateInterval = setInterval(() => {
        if (document.visibilityState === 'visible') {
            console.log('🔄 Actualización automática (cada 90 segundos)');
            updateCharts();
            loadMarketIndicators();
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
    
    // Actualizar gráfico principal
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    
    // Actualizar gráfico de dispersión
    updateScatterChartImproved(interval);
    
    // Actualizar señales múltiples
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    
    // Actualizar análisis multi-temporalidad
    updateMultiTimeframeAnalysis(symbol, interval);
}

function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage) {
    const params = new URLSearchParams({
        symbol: symbol,
        interval: interval,
        di_period: diPeriod,
        adx_threshold: adxThreshold,
        sr_period: srPeriod,
        rsi_length: rsiLength,
        bb_multiplier: bbMultiplier,
        leverage: leverage
    });
    
    fetch(`/api/signals?${params}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            currentData = data;
            renderCandleChart(data);
            renderTrendStrengthChart(data);
            renderRsiMaverickChart(data);
            renderAdxChartImproved(data);
            updateAuxChart();
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

function renderCandleChart(data, indicatorOptions = {}) {
    const chartElement = document.getElementById('candle-chart');
    
    if (!data.data || data.data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <h5><i class="fas fa-exclamation-triangle me-2"></i>No hay datos disponibles</h5>
                <p>No se pudieron cargar los datos para el gráfico.</p>
                <button class="btn btn-sm btn-primary mt-2" onclick="updateCharts()">
                    <i class="fas fa-sync-alt me-1"></i>Reintentar
                </button>
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

    // Añadir niveles de trading
    if (data.entry && data.take_profit) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.entry, data.entry],
            mode: 'lines',
            line: {color: '#FFD700', dash: 'solid', width: 2},
            name: 'Entrada'
        });
        
        // Añadir stop loss
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.stop_loss, data.stop_loss],
            mode: 'lines',
            line: {color: '#FF4444', dash: 'solid', width: 2},
            name: 'Stop Loss'
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
            text: `${data.symbol} - Análisis de Precio Multi-Temporalidad`,
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
                color: document.getElementById('drawing-color')?.value || '#FFD700',
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
    const colors = data.indicators.colors || [];
    const noTradeZones = data.indicators.no_trade_zones || [];
    
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
            text: 'Fuerza de Tendencia Maverick - Indicador Obligatorio',
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
            text: 'RSI Modificado Maverick (%B) con Divergencias',
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

function updateAuxChart() {
    if (!currentData || !currentData.indicators) return;
    
    const auxIndicator = document.getElementById('aux-indicator')?.value || 'macd';
    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    
    let traces = [];
    let title = '';
    
    switch(auxIndicator) {
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
                    line: {color: '#2196F3', width: 1.5}
                },
                {
                    x: dates,
                    y: macdSignal,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Señal',
                    line: {color: '#FF9800', width: 1.5}
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
            title = 'MACD - Convergencia/Divergencia de Medias Móviles';
            break;
            
        case 'rsi':
            const rsiTraditional = currentData.indicators.rsi_traditional || [];
            
            traces = [{
                x: dates,
                y: rsiTraditional,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI Tradicional',
                line: {color: '#9C27B0', width: 2}
            }];
            title = 'RSI Tradicional (14 periodos)';
            break;
            
        case 'squeeze':
            const squeezeMomentum = currentData.indicators.squeeze_momentum || [];
            const squeezeOn = currentData.indicators.squeeze_on || [];
            
            traces = [{
                x: dates,
                y: squeezeMomentum,
                type: 'bar',
                name: 'Squeeze Momentum',
                marker: {
                    color: squeezeMomentum.map((val, i) => 
                        val >= 0 ? '#00C853' : '#FF1744'
                    )
                }
            }];
            
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
            title = 'Squeeze Momentum - Compresión y Expansión';
            break;
            
        case 'whale':
            const whalePump = currentData.indicators.whale_pump || [];
            const whaleDump = currentData.indicators.whale_dump || [];
            
            traces = [
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
            title = 'Actividad de Ballenas - Compradoras vs Vendedoras';
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
            title: 'Valor',
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
    
    if (currentAuxChart) {
        Plotly.purge('aux-chart');
    }
    
    currentAuxChart = Plotly.newPlot('aux-chart', traces, layout, config);
}

function updateMarketSummary(data) {
    if (!data) return;
    
    const multiTF = data.multi_timeframe_analysis || {};
    const signalType = data.signal || 'NEUTRAL';
    const signalScore = data.signal_score || 0;
    
    const summaryHTML = `
        <div class="fade-in">
            <div class="row text-center mb-3">
                <div class="col-6">
                    <div class="card bg-dark border-${signalType === 'LONG' ? 'success' : signalType === 'SHORT' ? 'danger' : 'secondary'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Señal</small>
                            <h4 class="mb-0 text-${signalType === 'LONG' ? 'success' : signalType === 'SHORT' ? 'danger' : 'muted'}">
                                ${signalType}
                            </h4>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card bg-dark border-${signalScore >= 70 ? 'success' : 'warning'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Score</small>
                            <h4 class="mb-0 text-${signalScore >= 70 ? 'success' : 'warning'}">
                                ${signalScore.toFixed(0)}%
                            </h4>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-dollar-sign me-2"></i>Precio Actual</h6>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="fs-5 fw-bold">$${formatPriceForDisplay(data.current_price)}</span>
                    <small class="text-muted">USDT</small>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-layer-group me-2"></i>Confirmación Multi-TF</h6>
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
                <h6><i class="fas fa-bolt me-2"></i>Indicadores Clave</h6>
                <div class="d-flex justify-content-between">
                    <span>ADX:</span>
                    <span class="text-${data.adx > 25 ? 'success' : 'warning'}">${data.adx?.toFixed(1) || '0.0'}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>RSI Maverick:</span>
                    <span class="text-${data.rsi_maverick > 0.7 ? 'danger' : data.rsi_maverick < 0.3 ? 'success' : 'info'}">
                        ${(data.rsi_maverick * 100)?.toFixed(1) || '0'}%
                    </span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Volumen:</span>
                    <span class="text-${data.volume > data.volume_ma ? 'success' : 'muted'}">
                        ${(data.volume / data.volume_ma)?.toFixed(1) || '1.0'}x
                    </span>
                </div>
            </div>

            ${data.mandatory_conditions_met ? `
            <div class="alert alert-success text-center py-1">
                <small><i class="fas fa-check-circle me-1"></i>Condiciones Obligatorias CUMPLIDAS</small>
            </div>
            ` : `
            <div class="alert alert-warning text-center py-1">
                <small><i class="fas fa-exclamation-triangle me-1"></i>Condiciones Obligatorias PENDIENTES</small>
            </div>
            `}
        </div>
    `;
    
    document.getElementById('market-summary').innerHTML = summaryHTML;
}

function updateSignalAnalysis(data) {
    if (!data) return;
    
    let analysisHTML = '';
    const signalType = data.signal || 'NEUTRAL';
    const signalScore = data.signal_score || 0;
    
    if (signalType === 'NEUTRAL' || signalScore < 70) {
        analysisHTML = `
            <div class="text-center">
                <div class="alert alert-secondary">
                    <h6><i class="fas fa-info-circle me-2"></i>Señal No Confirmada</h6>
                    <p class="mb-2 small">Score: <strong>${signalScore.toFixed(1)}%</strong></p>
                    <p class="mb-2 small">Condiciones Obligatorias: <strong>${data.mandatory_conditions_met ? 'CUMPLIDAS' : 'PENDIENTES'}</strong></p>
                    <p class="mb-0 small text-muted">Esperando confirmación de indicadores...</p>
                </div>
            </div>
        `;
    } else {
        const signalColor = signalType === 'LONG' ? 'success' : 'danger';
        const signalIcon = signalType === 'LONG' ? 'arrow-up' : 'arrow-down';
        
        analysisHTML = `
            <div class="alert alert-${signalColor}">
                <h6><i class="fas fa-${signalIcon} me-2"></i>Señal ${signalType} CONFIRMADA</h6>
                <p class="mb-2 small"><strong>Score:</strong> ${signalScore.toFixed(1)}%</p>
                <p class="mb-2 small"><strong>Condiciones Obligatorias:</strong> <span class="text-success">CUMPLIDAS</span></p>
                
                ${data.fulfilled_conditions && data.fulfilled_conditions.length > 0 ? `
                <h6 class="mt-3 mb-2">Condiciones Cumplidas:</h6>
                <ul class="list-unstyled small mb-3">
                    ${data.fulfilled_conditions.slice(0, 3).map(condition => `
                        <li><i class="fas fa-check text-${signalColor} me-2"></i>${condition}</li>
                    `).join('')}
                </ul>
                ` : ''}
                
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
                        <strong class="text-warning">${(data.atr_percentage * 100)?.toFixed(2) || '0.00'}%</strong>
                    </div>
                </div>
                
                <div class="text-center mt-2">
                    <button class="btn btn-sm btn-outline-${signalColor}" onclick="downloadReport()">
                        <i class="fas fa-download me-1"></i>Reporte Completo
                    </button>
                </div>
            </div>
        `;
    }
    
    document.getElementById('signal-analysis').innerHTML = analysisHTML;
}

function updateMultiTimeframeAnalysis(symbol, interval) {
    const analysisElement = document.getElementById('multi-timeframe-analysis');
    if (!analysisElement || !currentData) return;
    
    const multiTF = currentData.multi_timeframe_analysis || {};
    
    const analysisHTML = `
        <div class="text-center">
            <div class="row">
                <div class="col-12 mb-2">
                    <h6 class="text-info">Análisis Multi-Temporalidad</h6>
                </div>
                <div class="col-4">
                    <div class="card bg-dark border-${multiTF.mayor === 'BULLISH' ? 'success' : multiTF.mayor === 'BEARISH' ? 'danger' : 'secondary'}">
                        <div class="card-body py-1">
                            <small class="text-muted">Mayor</small>
                            <div class="text-${multiTF.mayor === 'BULLISH' ? 'success' : multiTF.mayor === 'BEARISH' ? 'danger' : 'muted'}">
                                <i class="fas fa-${multiTF.mayor === 'BULLISH' ? 'arrow-up' : multiTF.mayor === 'BEARISH' ? 'arrow-down' : 'minus'}"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="card bg-dark border-${multiTF.media === 'BULLISH' ? 'success' : multiTF.media === 'BEARISH' ? 'danger' : 'secondary'}">
                        <div class="card-body py-1">
                            <small class="text-muted">Media</small>
                            <div class="text-${multiTF.media === 'BULLISH' ? 'success' : multiTF.media === 'BEARISH' ? 'danger' : 'muted'}">
                                <i class="fas fa-${multiTF.media === 'BULLISH' ? 'arrow-up' : multiTF.media === 'BEARISH' ? 'arrow-down' : 'minus'}"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="card bg-dark border-${multiTF.menor === 'BULLISH' ? 'success' : multiTF.menor === 'BEARISH' ? 'danger' : 'secondary'}">
                        <div class="card-body py-1">
                            <small class="text-muted">Menor</small>
                            <div class="text-${multiTF.menor === 'BULLISH' ? 'success' : multiTF.menor === 'BEARISH' ? 'danger' : 'muted'}">
                                <i class="fas fa-${multiTF.menor === 'BULLISH' ? 'arrow-up' : multiTF.menor === 'BEARISH' ? 'arrow-down' : 'minus'}"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            ${currentData.mandatory_conditions_met ? `
            <div class="alert alert-success mt-2 py-1">
                <small><i class="fas fa-check me-1"></i>Alineación Confirmada</small>
            </div>
            ` : `
            <div class="alert alert-warning mt-2 py-1">
                <small><i class="fas fa-exclamation-triangle me-1"></i>Alineación Pendiente</small>
            </div>
            `}
        </div>
    `;
    
    analysisElement.innerHTML = analysisHTML;
}

function updateScatterChartImproved(interval) {
    const params = new URLSearchParams({ interval: interval });
    
    fetch(`/api/scatter_data_improved?${params}`)
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
    
    const traces = [{
        x: scatterData.map(d => d.x),
        y: scatterData.map(d => d.y),
        text: scatterData.map(d => 
            `${d.symbol}<br>Score: ${d.signal_score?.toFixed(1) || '0'}%<br>Señal: ${d.signal || 'NEUTRAL'}<br>Precio: $${formatPriceForDisplay(d.current_price)}<br>Riesgo: ${d.risk_category || 'medio'}`
        ),
        mode: 'markers',
        marker: {
            size: scatterData.map(d => 8 + ((d.signal_score || 0) / 15)),
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
            opacity: scatterData.map(d => 0.6 + ((d.signal_score || 0) / 250)),
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
        margin: {t: 80, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle']
    };
    
    if (currentScatterChart) {
        Plotly.purge('scatter-chart');
    }
    
    currentScatterChart = Plotly.newPlot('scatter-chart', traces, layout, config);
}

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage) {
    const params = new URLSearchParams({
        interval: interval,
        di_period: diPeriod,
        adx_threshold: adxThreshold,
        sr_period: srPeriod,
        rsi_length: rsiLength,
        bb_multiplier: bbMultiplier,
        leverage: leverage
    });
    
    fetch(`/api/multiple_signals?${params}`)
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
    if (longTable) {
        if (data.long_signals && data.long_signals.length > 0) {
            longTable.innerHTML = data.long_signals.slice(0, 5).map((signal, index) => `
                <tr onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;" class="hover-row">
                    <td class="text-center">${index + 1}</td>
                    <td>
                        <strong>${signal.symbol}</strong>
                        <br><small class="text-success">Score: ${signal.signal_score?.toFixed(1) || '0'}%</small>
                    </td>
                    <td class="text-center">
                        <span class="badge bg-success">${signal.signal_score?.toFixed(0) || '0'}%</span>
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
    }
    
    // Actualizar tabla SHORT
    const shortTable = document.getElementById('short-table');
    if (shortTable) {
        if (data.short_signals && data.short_signals.length > 0) {
            shortTable.innerHTML = data.short_signals.slice(0, 5).map((signal, index) => `
                <tr onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;" class="hover-row">
                    <td class="text-center">${index + 1}</td>
                    <td>
                        <strong>${signal.symbol}</strong>
                        <br><small class="text-danger">Score: ${signal.signal_score?.toFixed(1) || '0'}%</small>
                    </td>
                    <td class="text-center">
                        <span class="badge bg-danger">${signal.signal_score?.toFixed(0) || '0'}%</span>
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
}

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
                    
                    alertsHTML += `
                        <div class="alert alert-${alertType} mb-2">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <h6 class="mb-1">
                                        <i class="fas fa-${alertIcon} me-1"></i>
                                        ${alert.symbol} ${riskBadge}
                                        <span class="badge bg-${alert.interval === '15m' || alert.interval === '30m' ? 'warning' : 'secondary'} ms-1">
                                            ${alert.interval}
                                        </span>
                                    </h6>
                                    <p class="mb-1 small">
                                        <strong>Score: ${alert.score?.toFixed(1) || '0'}%</strong><br>
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
                                        <strong class="text-${alertType}">P&L: ${alert.pnl_percent?.toFixed(2) || '0.00'}%</strong>
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

function updateWinrateDisplay() {
    fetch('/api/winrate_data')
        .then(response => response.json())
        .then(data => {
            const winrateDisplay = document.getElementById('winrate-display');
            if (winrateDisplay) {
                const winrate = data.winrate || 0;
                const totalOps = data.total_operations || 0;
                const successfulOps = data.successful_operations || 0;
                
                winrateDisplay.innerHTML = `
                    <h4 class="text-success mb-1">${winrate.toFixed(1)}%</h4>
                    <p class="small text-muted mb-0">${successfulOps}/${totalOps} operaciones</p>
                    <div class="progress mt-2" style="height: 6px;">
                        <div class="progress-bar bg-success" style="width: ${winrate}%"></div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
        });
}

function tradeAlert(symbol, interval, leverage) {
    currentSymbol = symbol;
    const selectedCryptoElement = document.getElementById('selected-crypto');
    if (selectedCryptoElement) {
        selectedCryptoElement.textContent = symbol;
    }
    
    const intervalSelect = document.getElementById('interval-select');
    const leverageSelect = document.getElementById('leverage');
    
    if (intervalSelect) intervalSelect.value = interval;
    if (leverageSelect) leverageSelect.value = leverage;
    
    updateCharts();
    showNotification(`Configurado para operar ${symbol} en ${interval} con leverage x${leverage}`, 'success');
}

function showSignalDetails(symbol) {
    const modalElement = document.getElementById('signalModal');
    if (!modalElement) return;
    
    const modal = new bootstrap.Modal(modalElement);
    const signalData = currentData && currentData.symbol === symbol ? currentData : null;
    
    const detailsHTML = signalData ? `
        <div class="signal-details">
            <h4 class="text-${signalData.signal === 'LONG' ? 'success' : 'danger'}">
                <i class="fas fa-${signalData.signal === 'LONG' ? 'arrow-up' : 'arrow-down'} me-2"></i>
                ${symbol} - Señal ${signalData.signal} Confirmada
            </h4>
            <p class="text-muted">Score de señal: <strong>${signalData.signal_score.toFixed(1)}%</strong></p>
            
            <div class="row mt-3">
                <div class="col-md-6">
                    <h6>Información de Trading</h6>
                    <table class="table table-sm table-dark">
                        <tr><td>Precio Actual:</td><td class="text-end">$${formatPriceForDisplay(signalData.current_price)}</td></tr>
                        <tr><td>Entrada Recomendada:</td><td class="text-end text-${signalData.signal === 'LONG' ? 'success' : 'danger'}">$${formatPriceForDisplay(signalData.entry)}</td></tr>
                        <tr><td>Stop Loss:</td><td class="text-end text-danger">$${formatPriceForDisplay(signalData.stop_loss)}</td></tr>
                        <tr><td>Soporte:</td><td class="text-end text-info">$${formatPriceForDisplay(signalData.support)}</td></tr>
                        <tr><td>Resistencia:</td><td class="text-end text-warning">$${formatPriceForDisplay(signalData.resistance)}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Take Profits</h6>
                    <table class="table table-sm table-dark">
                        ${signalData.take_profit.map((tp, index) => `
                            <tr>
                                <td>TP${index + 1}:</td>
                                <td class="text-end text-success">$${formatPriceForDisplay(tp)}</td>
                            </tr>
                        `).join('')}
                    </table>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-12">
                    <h6>Indicadores Técnicos</h6>
                    <div class="d-flex justify-content-between flex-wrap">
                        <span class="badge bg-primary me-2 mb-1">ADX: ${signalData.adx?.toFixed(1) || '0.0'}</span>
                        <span class="badge bg-success me-2 mb-1">D+: ${signalData.plus_di?.toFixed(1) || '0.0'}</span>
                        <span class="badge bg-danger me-2 mb-1">D-: ${signalData.minus_di?.toFixed(1) || '0.0'}</span>
                        <span class="badge bg-info me-2 mb-1">RSI Maverick: ${(signalData.rsi_maverick * 100)?.toFixed(1) || '0'}%</span>
                    </div>
                </div>
            </div>
            
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

function downloadStrategyReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}`;
    window.open(url, '_blank');
}

// Funciones auxiliares
function formatPriceForDisplay(price) {
    if (!price || price === 0) return '0.00';
    if (price < 0.01) {
        return price.toFixed(6);
    } else if (price < 1) {
        return price.toFixed(4);
    } else {
        return price.toFixed(2);
    }
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

function showNotification(message, type = 'info') {
    // Crear notificación toast
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
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
    
    toastContainer.innerHTML += toastHTML;
    
    // Mostrar toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remover toast después de ocultarse
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

function showError(message) {
    const chartElement = document.getElementById('candle-chart');
    if (chartElement) {
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
}

function showScatterError(message) {
    const scatterElement = document.getElementById('scatter-chart');
    if (scatterElement) {
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
        support: 48000,
        resistance: 52000,
        volume: 1000000,
        volume_ma: 800000,
        adx: 25,
        plus_di: 30,
        minus_di: 20,
        rsi_maverick: 0.5,
        multi_timeframe_analysis: {
            mayor: 'NEUTRAL',
            media: 'NEUTRAL', 
            menor: 'NEUTRAL'
        },
        mandatory_conditions_met: false,
        fulfilled_conditions: []
    };
    
    updateMarketSummary(sampleData);
    updateSignalAnalysis(sampleData);
}

// Exportar funciones globales para uso en HTML
window.updateCharts = updateCharts;
window.downloadReport = downloadReport;
window.downloadSignalReport = downloadSignalReport;
window.downloadStrategyReport = downloadStrategyReport;
window.tradeAlert = tradeAlert;
window.showSignalDetails = showSignalDetails;
window.selectCrypto = selectCrypto;
