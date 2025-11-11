// MULTI-TIMEFRAME CRYPTO WGTA PRO - Script Principal
// Optimizado para servidores con recursos limitados (Render - 512MB RAM)

// Configuración global optimizada
let currentChart = null;
let currentScatterChart = null;
let currentWhaleChart = null;
let currentAdxChart = null;
let currentRsiChart = null;
let currentMacdChart = null;
let currentSqueezeChart = null;
let currentTrendStrengthChart = null;
let currentAuxChart = null;
let currentSymbol = 'BTC-USDT';
let currentData = null;
let allCryptos = [];
let updateInterval = null;
let drawingToolsActive = false;
let isUpdating = false;

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializando Sistema');
    initializeApp();
    setupEventListeners();
    
    // Inicializar con un pequeño delay para evitar sobrecarga
    setTimeout(() => {
        updateCharts();
        startAutoUpdate();
    }, 1000);
});

function initializeApp() {
    console.log('✅ Inicializando aplicación...');
    loadCryptoRiskClassification();
    loadMarketIndicators();
    updateCalendarInfo();
    updateWinrateDisplay();
    
    // Mostrar estado del sistema
    showToast('Sistema inicializado correctamente', 'success');
}

function setupEventListeners() {
    console.log('🔧 Configurando event listeners...');
    
    // Configurar event listeners para los controles principales
    const controls = [
        'interval-select', 'di-period', 'adx-threshold', 'sr-period',
        'rsi-length', 'bb-multiplier', 'volume-filter', 'leverage'
    ];
    
    controls.forEach(controlId => {
        const element = document.getElementById(controlId);
        if (element) {
            element.addEventListener('change', function() {
                if (!isUpdating) {
                    updateCharts();
                }
            });
        }
    });
    
    // Configurar buscador de cryptos
    setupCryptoSearch();
    
    // Configurar herramientas de dibujo (solo si existen)
    setupDrawingTools();
    
    // Configurar controles de indicadores
    setupIndicatorControls();
    
    // Configurar selector de indicador auxiliar
    const auxIndicator = document.getElementById('aux-indicator');
    if (auxIndicator) {
        auxIndicator.addEventListener('change', updateAuxChart);
    }
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
    const cryptoList = document.getElementById('crypto-list');
    
    if (!searchInput || !cryptoList) return;
    
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
    const drawingButtons = document.querySelectorAll('.drawing-tool');
    if (drawingButtons.length === 0) return;
    
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

function activateDrawingTool(tool) {
    drawingToolsActive = true;
    
    // Remover clase activa de todos los botones
    document.querySelectorAll('.drawing-tool').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Activar botón seleccionado
    event.target.classList.add('active');
    
    console.log(`🛠️ Herramienta de dibujo activada: ${tool}`);
}

function setDrawingColor(color) {
    console.log(`🎨 Color de dibujo cambiado a: ${color}`);
}

function updateChartIndicators() {
    if (currentData && currentChart) {
        renderCandleChart(currentData);
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
                No se encontraron resultados
            </div>
        `;
        return;
    }
    
    // Mostrar máximo 20 resultados para evitar sobrecarga
    const displayCryptos = filteredCryptos.slice(0, 20);
    
    displayCryptos.forEach(crypto => {
        const item = document.createElement('a');
        item.className = 'dropdown-item crypto-item';
        item.href = '#';
        item.innerHTML = `
            ${crypto.symbol} 
            <span class="badge ${getRiskBadgeClass(crypto.category)} float-end">
                ${crypto.category}
            </span>
        `;
        item.addEventListener('click', function(e) {
            e.preventDefault();
            selectCrypto(crypto.symbol);
        });
        cryptoList.appendChild(item);
    });
}

function getRiskBadgeClass(category) {
    switch(category) {
        case 'bajo': return 'bg-success';
        case 'medio': return 'bg-warning';
        case 'alto': return 'bg-danger';
        case 'memecoins': return 'bg-purple';
        default: return 'bg-secondary';
    }
}

function selectCrypto(symbol) {
    currentSymbol = symbol;
    document.getElementById('selected-crypto').textContent = symbol;
    
    // Cerrar el dropdown
    const bootstrapDropdown = bootstrap.Dropdown.getInstance(document.getElementById('cryptoDropdown'));
    if (bootstrapDropdown) {
        bootstrapDropdown.hide();
    }
    
    updateCharts();
    updateWinrateDisplay();
    
    showToast(`Criptomoneda cambiada a: ${symbol}`, 'info');
}

function loadCryptoRiskClassification() {
    console.log('📊 Cargando clasificación de riesgo...');
    
    // Usar lista estática para evitar llamadas a API al inicio
    const staticCryptos = [
        {symbol: 'BTC-USDT', category: 'bajo'},
        {symbol: 'ETH-USDT', category: 'bajo'},
        {symbol: 'BNB-USDT', category: 'bajo'},
        {symbol: 'SOL-USDT', category: 'bajo'},
        {symbol: 'XRP-USDT', category: 'bajo'},
        {symbol: 'ADA-USDT', category: 'medio'},
        {symbol: 'AVAX-USDT', category: 'medio'},
        {symbol: 'DOT-USDT', category: 'medio'},
        {symbol: 'DOGE-USDT', category: 'medio'},
        {symbol: 'SHIB-USDT', category: 'memecoins'},
        {symbol: 'MATIC-USDT', category: 'medio'},
        {symbol: 'LTC-USDT', category: 'bajo'},
        {symbol: 'LINK-USDT', category: 'medio'},
        {symbol: 'ATOM-USDT', category: 'medio'},
        {symbol: 'XLM-USDT', category: 'medio'}
    ];
    
    allCryptos = staticCryptos;
    filterCryptoList('');
    
    // Intentar cargar datos actualizados en segundo plano
    setTimeout(() => {
        fetch('/api/crypto_risk_classification')
            .then(response => {
                if (!response.ok) throw new Error('Error en respuesta');
                return response.json();
            })
            .then(riskData => {
                if (riskData && typeof riskData === 'object') {
                    allCryptos = [];
                    Object.keys(riskData).forEach(category => {
                        if (Array.isArray(riskData[category])) {
                            riskData[category].forEach(symbol => {
                                allCryptos.push({symbol, category});
                            });
                        }
                    });
                    filterCryptoList('');
                    console.log('✅ Clasificación de riesgo actualizada');
                }
            })
            .catch(error => {
                console.log('ℹ️ Usando clasificación de riesgo estática');
            });
    }, 2000);
}

function loadMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateTradingAlerts();
    updateExitSignals();
}

function showLoadingState() {
    const elements = {
        'market-summary': 'Analizando mercado...',
        'signal-analysis': 'Evaluando condiciones de señal...',
        'obligatory-conditions': 'Verificando condiciones...',
        'scalping-alerts': 'Buscando oportunidades...',
        'exit-signals': 'Monitoreando operaciones...'
    };
    
    Object.keys(elements).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = `
                <div class="text-center py-3">
                    <div class="spinner-border spinner-border-sm text-primary" role="status">
                        <span class="visually-hidden">Cargando...</span>
                    </div>
                    <p class="mt-2 mb-0 small">${elements[id]}</p>
                </div>
            `;
        }
    });
}

function startAutoUpdate() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Actualización más espaciada para reducir carga (3 minutos)
    updateInterval = setInterval(() => {
        if (document.visibilityState === 'visible' && !isUpdating) {
            console.log('🔄 Actualización automática iniciada');
            updateMarketIndicators();
            updateWinrateDisplay();
        }
    }, 180000); // 3 minutos
}

function updateCharts() {
    if (isUpdating) {
        console.log('⚠️ Actualización en curso, omitiendo...');
        return;
    }
    
    isUpdating = true;
    showLoadingState();
    
    console.log('📈 Actualizando gráficos...');
    
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const diPeriod = document.getElementById('di-period').value;
    const adxThreshold = document.getElementById('adx-threshold').value;
    const srPeriod = document.getElementById('sr-period').value;
    const rsiLength = document.getElementById('rsi-length').value;
    const bbMultiplier = document.getElementById('bb-multiplier').value;
    const volumeFilter = document.getElementById('volume-filter').value;
    const leverage = document.getElementById('leverage').value;
    
    // Actualizar gráfico principal primero
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar otros componentes de forma secuencial para reducir carga
    setTimeout(() => {
        updateScatterChart(interval);
    }, 1000);
    
    setTimeout(() => {
        updateMultipleSignals(interval);
    }, 2000);
    
    setTimeout(() => {
        updateAuxChart();
        isUpdating = false;
    }, 3000);
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateTradingAlerts();
    updateCalendarInfo();
}

function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const params = new URLSearchParams({
        symbol, interval, di_period: diPeriod, adx_threshold: adxThreshold,
        sr_period: srPeriod, rsi_length: rsiLength, bb_multiplier: bbMultiplier,
        volume_filter: volumeFilter, leverage
    });
    
    fetch(`/api/signals?${params}`)
        .then(response => {
            if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.error) throw new Error(data.error);
            
            currentData = data;
            
            // Renderizar gráficos secuencialmente para mejor performance
            renderCandleChart(data);
            setTimeout(() => renderWhaleChart(data), 100);
            setTimeout(() => renderAdxChart(data), 200);
            setTimeout(() => renderRsiComparativeChart(data), 300);
            setTimeout(() => renderMacdChart(data), 400);
            setTimeout(() => renderSqueezeChart(data), 500);
            setTimeout(() => renderTrendStrengthChart(data), 600);
            
            updateMarketSummary(data);
            updateSignalAnalysis(data);
            updateObligatoryConditions(data);
            
            console.log('✅ Gráfico principal actualizado');
        })
        .catch(error => {
            console.error('❌ Error en updateMainChart:', error);
            showError('Error al cargar datos: ' + error.message);
            isUpdating = false;
        });
}

function renderCandleChart(data) {
    const chartElement = document.getElementById('candle-chart');
    if (!chartElement) return;
    
    if (!data.data || data.data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center m-3">
                <h6>No hay datos disponibles</h6>
                <p class="mb-2">No se pudieron cargar los datos para el gráfico.</p>
                <button class="btn btn-sm btn-primary" onclick="updateCharts()">Reintentar</button>
            </div>
        `;
        return;
    }

    // Limitar datos para mejor performance
    const displayData = data.data.slice(-50);
    const dates = displayData.map(d => new Date(d.timestamp));
    const opens = displayData.map(d => parseFloat(d.open));
    const highs = displayData.map(d => parseFloat(d.high));
    const lows = displayData.map(d => parseFloat(d.low));
    const closes = displayData.map(d => parseFloat(d.close));
    
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
    
    // Añadir niveles clave si están disponibles
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
            line: {color: '#FF6B6B', dash: 'solid', width: 2},
            name: 'Stop Loss'
        });
    }
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas | Score: ${data.signal_score || 0}%`,
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Precio (USDT)',
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
            orientation: 'h'
        },
        margin: {t: 60, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    };
    
    Plotly.newPlot('candle-chart', traces, layout, config).then(chart => {
        currentChart = chart;
    });
}

function renderWhaleChart(data) {
    const chartElement = document.getElementById('whale-chart');
    if (!chartElement || !data.indicators) return;
    
    const dates = data.data ? data.data.slice(-50).map(d => new Date(d.timestamp)) : [];
    const whalePump = data.indicators.whale_pump || [];
    const whaleDump = data.indicators.whale_dump || [];
    
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
            text: 'Indicador Ballenas - Compradoras vs Vendedoras',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        barmode: 'overlay',
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('whale-chart', traces, layout, config).then(chart => {
        currentWhaleChart = chart;
    });
}

function renderAdxChart(data) {
    const chartElement = document.getElementById('adx-chart');
    if (!chartElement || !data.indicators) return;
    
    const dates = data.data ? data.data.slice(-50).map(d => new Date(d.timestamp)) : [];
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
            text: 'ADX con DMI (+D y -D)',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('adx-chart', traces, layout, config).then(chart => {
        currentAdxChart = chart;
    });
}

function renderRsiComparativeChart(data) {
    const chartElement = document.getElementById('rsi-comparative-chart');
    if (!chartElement || !data.indicators) return;
    
    const dates = data.data ? data.data.slice(-50).map(d => new Date(d.timestamp)) : [];
    const rsiTraditional = data.indicators.rsi_traditional || [];
    const rsiMaverick = data.indicators.rsi_maverick || [];
    
    // Escalar RSI Maverick a 0-100 para comparación
    const scaledMaverick = rsiMaverick.map(val => val * 100);
    
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
            y: scaledMaverick,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B × 100)',
            line: {color: '#FF9800', width: 2}
        },
        {
            x: [dates[0], dates[dates.length - 1]],
            y: [70, 70],
            mode: 'lines',
            line: {color: 'red', dash: 'dash', width: 1},
            name: 'Sobrecompra',
            showlegend: false
        },
        {
            x: [dates[0], dates[dates.length - 1]],
            y: [30, 30],
            mode: 'lines',
            line: {color: 'green', dash: 'dash', width: 1},
            name: 'Sobreventa',
            showlegend: false
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Comparativo - Tradicional vs Maverick',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            range: [0, 100],
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('rsi-comparative-chart', traces, layout, config).then(chart => {
        currentRsiChart = chart;
    });
}

function renderMacdChart(data) {
    const chartElement = document.getElementById('macd-chart');
    if (!chartElement || !data.indicators) return;
    
    const dates = data.data ? data.data.slice(-50).map(d => new Date(d.timestamp)) : [];
    const macd = data.indicators.macd || [];
    const macdSignal = data.indicators.macd_signal || [];
    const macdHistogram = data.indicators.macd_histogram || [];
    
    // Crear histograma con colores
    const histogramColors = macdHistogram.map(val => val >= 0 ? '#00C853' : '#FF1744');
    
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
            y: macdSignal,
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#FF9800', width: 2}
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
            text: 'MACD - Convergencia/Divergencia',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('macd-chart', traces, layout, config).then(chart => {
        currentMacdChart = chart;
    });
}

function renderSqueezeChart(data) {
    const chartElement = document.getElementById('squeeze-chart');
    if (!chartElement || !data.indicators) return;
    
    const dates = data.data ? data.data.slice(-50).map(d => new Date(d.timestamp)) : [];
    const squeezeMomentum = data.indicators.squeeze_momentum || [];
    
    // Crear colores para el momentum
    const colors = squeezeMomentum.map(val => val >= 0 ? '#00C853' : '#FF1744');
    
    const traces = [
        {
            x: dates,
            y: squeezeMomentum,
            type: 'bar',
            name: 'Squeeze Momentum',
            marker: {color: colors}
        }
    ];
    
    const layout = {
        title: {
            text: 'Squeeze Momentum - Compresión/Expansión',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('squeeze-chart', traces, layout, config).then(chart => {
        currentSqueezeChart = chart;
    });
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    if (!chartElement || !data.indicators) return;
    
    const dates = data.data ? data.data.slice(-50).map(d => new Date(d.timestamp)) : [];
    const trendStrength = data.indicators.trend_strength || [];
    const colors = data.indicators.colors || Array(trendStrength.length).fill('gray');
    
    const traces = [
        {
            x: dates,
            y: trendStrength,
            type: 'bar',
            name: 'Fuerza Tendencia',
            marker: {color: colors}
        }
    ];
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('trend-strength-chart', traces, layout, config).then(chart => {
        currentTrendStrengthChart = chart;
    });
}

function updateAuxChart() {
    const auxIndicator = document.getElementById('aux-indicator');
    if (!auxIndicator || !currentData) return;
    
    const selectedIndicator = auxIndicator.value;
    renderAuxChart(currentData, selectedIndicator);
}

function renderAuxChart(data, indicatorType) {
    const chartElement = document.getElementById('aux-chart');
    if (!chartElement || !data.indicators) return;
    
    const dates = data.data ? data.data.slice(-50).map(d => new Date(d.timestamp)) : [];
    let trace;
    
    switch(indicatorType) {
        case 'rsi':
            trace = {
                x: dates,
                y: data.indicators.rsi_traditional || [],
                type: 'scatter',
                mode: 'lines',
                name: 'RSI Tradicional',
                line: {color: '#2196F3', width: 2}
            };
            break;
        case 'macd':
            trace = {
                x: dates,
                y: data.indicators.macd || [],
                type: 'scatter',
                mode: 'lines',
                name: 'MACD',
                line: {color: '#FF9800', width: 2}
            };
            break;
        case 'squeeze':
            trace = {
                x: dates,
                y: data.indicators.squeeze_momentum || [],
                type: 'bar',
                name: 'Squeeze Momentum',
                marker: {color: '#00C853'}
            };
            break;
        default:
            return;
    }
    
    const layout = {
        title: {
            text: `Indicador Auxiliar - ${indicatorType.toUpperCase()}`,
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('aux-chart', [trace], layout, config).then(chart => {
        currentAuxChart = chart;
    });
}

function updateScatterChart(interval) {
    fetch(`/api/scatter_data_improved?interval=${interval}`)
        .then(response => {
            if (!response.ok) throw new Error('Error en scatter data');
            return response.json();
        })
        .then(scatterData => {
            renderScatterChart(scatterData);
        })
        .catch(error => {
            console.error('Error actualizando scatter chart:', error);
        });
}

function renderScatterChart(data) {
    const chartElement = document.getElementById('scatter-chart');
    if (!chartElement) return;
    
    if (!data || data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-info text-center m-3">
                <p class="mb-0">No hay datos para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }
    
    const traces = {};
    
    data.forEach(item => {
        const category = item.risk_category || 'medio';
        if (!traces[category]) {
            traces[category] = {
                x: [],
                y: [],
                text: [],
                mode: 'markers',
                type: 'scatter',
                name: getCategoryName(category),
                marker: {
                    size: 12,
                    color: getCategoryColor(category),
                    symbol: getCategorySymbol(category)
                }
            };
        }
        
        traces[category].x.push(item.x);
        traces[category].y.push(item.y);
        traces[category].text.push(
            `${item.symbol}<br>Score: ${item.signal_score}%<br>Precio: $${item.current_price}`
        );
    });
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Análisis Multi-Indicador',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Presión Compradora',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Presión Vendedora',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 60, r: 30, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('scatter-chart', Object.values(traces), layout, config);
}

function getCategoryName(category) {
    const names = {
        'bajo': 'Bajo Riesgo',
        'medio': 'Medio Riesgo',
        'alto': 'Alto Riesgo',
        'memecoins': 'Memecoins'
    };
    return names[category] || category;
}

function getCategoryColor(category) {
    const colors = {
        'bajo': '#00C853',
        'medio': '#FF9800',
        'alto': '#FF1744',
        'memecoins': '#9C27B0'
    };
    return colors[category] || '#666';
}

function getCategorySymbol(category) {
    const symbols = {
        'bajo': 'circle',
        'medio': 'square',
        'alto': 'diamond',
        'memecoins': 'star'
    };
    return symbols[category] || 'circle';
}

function updateMultipleSignals(interval) {
    fetch(`/api/multiple_signals?interval=${interval}`)
        .then(response => {
            if (!response.ok) throw new Error('Error en múltiples señales');
            return response.json();
        })
        .then(data => {
            updateSignalsTables(data.long_signals, data.short_signals);
        })
        .catch(error => {
            console.error('Error actualizando múltiples señales:', error);
        });
}

function updateSignalsTables(longSignals, shortSignals) {
    updateSignalTable('long-table', longSignals || []);
    updateSignalTable('short-table', shortSignals || []);
}

function updateSignalTable(tableId, signals) {
    const tableBody = document.getElementById(tableId);
    if (!tableBody) return;
    
    if (signals.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    No hay señales disponibles
                </td>
            </tr>
        `;
        return;
    }
    
    tableBody.innerHTML = signals.slice(0, 10).map((signal, index) => `
        <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
            <td>${index + 1}</td>
            <td>${signal.symbol}</td>
            <td>
                <span class="badge ${getScoreBadgeClass(signal.signal_score)}">
                    ${signal.signal_score}%
                </span>
            </td>
            <td>${signal.winrate ? signal.winrate.toFixed(1) + '%' : 'N/A'}</td>
        </tr>
    `).join('');
}

function getScoreBadgeClass(score) {
    if (score >= 80) return 'bg-success';
    if (score >= 70) return 'bg-warning';
    return 'bg-danger';
}

function showSignalDetails(symbol) {
    // Implementación básica - puedes expandir esto según necesites
    showToast(`Detalles de ${symbol} - Función en desarrollo`, 'info');
}

function updateMarketSummary(data) {
    const summaryElement = document.getElementById('market-summary');
    if (!summaryElement) return;
    
    const signalClass = data.signal === 'LONG' ? 'text-success' : 
                       data.signal === 'SHORT' ? 'text-danger' : 'text-muted';
    
    summaryElement.innerHTML = `
        <div class="market-summary-content">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h6 class="mb-0">${data.symbol}</h6>
                <span class="badge ${signalClass}">${data.signal}</span>
            </div>
            
            <div class="row g-2 mb-3">
                <div class="col-6">
                    <small class="text-muted">Precio</small>
                    <div class="fw-bold">$${data.current_price?.toFixed(6) || '0'}</div>
                </div>
                <div class="col-6">
                    <small class="text-muted">Score</small>
                    <div class="fw-bold ${signalClass}">${data.signal_score}%</div>
                </div>
            </div>
            
            <div class="row g-2">
                <div class="col-6">
                    <small class="text-muted">Entrada</small>
                    <div class="small">$${data.entry?.toFixed(6) || '0'}</div>
                </div>
                <div class="col-6">
                    <small class="text-muted">Stop Loss</small>
                    <div class="small">$${data.stop_loss?.toFixed(6) || '0'}</div>
                </div>
            </div>
            
            <div class="mt-3">
                <small class="text-muted d-block">Volumen (24h)</small>
                <div class="small">${formatVolume(data.volume || 0)}</div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const analysisElement = document.getElementById('signal-analysis');
    if (!analysisElement) return;
    
    const conditions = data.fulfilled_conditions || [];
    
    analysisElement.innerHTML = `
        <div class="signal-analysis-content">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <strong>Análisis de Señal</strong>
                <span class="badge ${data.signal_score >= 70 ? 'bg-success' : 'bg-warning'}">
                    ${data.signal_score}%
                </span>
            </div>
            
            <div class="mb-2">
                <small class="text-muted">Tendencia:</small>
                <div class="small fw-bold ${getTrendStrengthClass(data.trend_strength_signal)}">
                    ${data.trend_strength_signal || 'NEUTRAL'}
                </div>
            </div>
            
            <div class="mb-2">
                <small class="text-muted">Zona NO OPERAR:</small>
                <div class="small ${data.no_trade_zone ? 'text-danger' : 'text-success'}">
                    ${data.no_trade_zone ? '🔴 ACTIVA' : '🟢 INACTIVA'}
                </div>
            </div>
            
            ${conditions.length > 0 ? `
                <div class="mt-2">
                    <small class="text-muted">Condiciones cumplidas:</small>
                    <div class="small">
                        ${conditions.slice(0, 3).map(cond => `• ${cond}`).join('<br>')}
                        ${conditions.length > 3 ? `<br><small class="text-muted">+${conditions.length - 3} más</small>` : ''}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function updateObligatoryConditions(data) {
    const conditionsElement = document.getElementById('obligatory-conditions');
    if (!conditionsElement) return;
    
    const isMet = data.obligatory_conditions_met;
    
    conditionsElement.innerHTML = `
        <div class="obligatory-conditions-content">
            <div class="d-flex align-items-center mb-2">
                <i class="fas ${isMet ? 'fa-check-circle text-success' : 'fa-times-circle text-danger'} me-2"></i>
                <strong>Multi-TF Obligatorios</strong>
            </div>
            <div class="small ${isMet ? 'text-success' : 'text-danger'}">
                ${isMet ? '✅ CUMPLIDOS' : '❌ NO CUMPLIDOS'}
            </div>
            ${data.trend_strength_signal ? `
                <div class="small text-muted mt-1">
                    Fuerza: ${data.trend_strength_signal}
                </div>
            ` : ''}
        </div>
    `;
}

function getTrendStrengthClass(strength) {
    if (strength?.includes('STRONG')) return 'text-success';
    if (strength?.includes('WEAK')) return 'text-warning';
    return 'text-muted';
}

function updateFearGreedIndex() {
    const element = document.getElementById('fear-greed-index');
    if (!element) return;
    
    // Simulación - en una implementación real, esto vendría de una API
    const simulatedIndex = Math.floor(Math.random() * 100);
    let status, color, emoji;
    
    if (simulatedIndex >= 75) { status = 'Extrema Codicia'; color = 'danger'; emoji = '😱'; }
    else if (simulatedIndex >= 55) { status = 'Codicia'; color = 'warning'; emoji = '😊'; }
    else if (simulatedIndex >= 45) { status = 'Neutral'; color = 'info'; emoji = '😐'; }
    else if (simulatedIndex >= 25) { status = 'Miedo'; color = 'primary'; emoji = '😟'; }
    else { status = 'Miedo Extremo'; color = 'success'; emoji = '😨'; }
    
    element.innerHTML = `
        <div class="text-center">
            <div class="fear-greed-value display-6 fw-bold text-${color}">
                ${simulatedIndex}
            </div>
            <div class="fear-greed-status small">
                ${emoji} ${status}
            </div>
            <div class="progress mt-2" style="height: 8px;">
                <div class="progress-bar bg-${color}" 
                     style="width: ${simulatedIndex}%"
                     role="progressbar">
                </div>
            </div>
        </div>
    `;
}

function updateMarketRecommendations() {
    const element = document.getElementById('market-recommendations');
    if (!element) return;
    
    const recommendations = [
        {type: 'success', text: 'Mercado alcista en curso', icon: '📈'},
        {type: 'warning', text: 'Volatilidad moderada', icon: '⚡'},
        {type: 'info', text: 'Buena liquidez', icon: '💧'}
    ];
    
    element.innerHTML = recommendations.map(rec => `
        <div class="alert alert-${rec.type} alert-dismissible fade show py-2" role="alert">
            <small>${rec.icon} ${rec.text}</small>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="alert"></button>
        </div>
    `).join('');
}

function updateTradingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => {
            if (!response.ok) throw new Error('Error en alertas');
            return response.json();
        })
        .then(data => {
            updateAlertsDisplay('scalping-alerts', data.alerts || []);
        })
        .catch(error => {
            console.error('Error actualizando alertas:', error);
            updateAlertsDisplay('scalping-alerts', []);
        });
}

function updateExitSignals() {
    fetch('/api/exit_signals')
        .then(response => {
            if (!response.ok) throw new Error('Error en señales de salida');
            return response.json();
        })
        .then(data => {
            updateAlertsDisplay('exit-signals', data.exit_signals || []);
        })
        .catch(error => {
            console.error('Error actualizando señales de salida:', error);
            updateAlertsDisplay('exit-signals', []);
        });
}

function updateAlertsDisplay(elementId, alerts) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    if (alerts.length === 0) {
        element.innerHTML = `
            <div class="text-center py-2">
                <small class="text-muted">No hay alertas activas</small>
            </div>
        `;
        return;
    }
    
    element.innerHTML = alerts.slice(0, 5).map(alert => `
        <div class="alert-item mb-2 p-2 rounded ${getAlertClass(alert.signal)}">
            <div class="d-flex justify-content-between align-items-start">
                <strong class="small">${alert.symbol}</strong>
                <span class="badge ${getSignalBadgeClass(alert.signal)}">${alert.signal}</span>
            </div>
            <div class="small">
                <div>Score: ${alert.score}%</div>
                <div>Entrada: $${alert.entry?.toFixed(6)}</div>
                ${alert.pnl_percent ? `<div>P&L: ${alert.pnl_percent > 0 ? '+' : ''}${alert.pnl_percent?.toFixed(2)}%</div>` : ''}
            </div>
        </div>
    `).join('');
}

function getAlertClass(signal) {
    return signal === 'LONG' ? 'alert-success' : 'alert-danger';
}

function getSignalBadgeClass(signal) {
    return signal === 'LONG' ? 'bg-success' : 'bg-danger';
}

function updateWinrateDisplay() {
    fetch('/api/winrate')
        .then(response => {
            if (!response.ok) throw new Error('Error en winrate');
            return response.json();
        })
        .then(data => {
            const winrateElement = document.getElementById('overall-winrate');
            if (winrateElement && data.overall_winrate !== undefined) {
                const winrate = data.overall_winrate;
                winrateElement.textContent = `${winrate.toFixed(1)}%`;
                
                if (winrate >= 70) {
                    winrateElement.className = 'winrate-value display-4 fw-bold text-success';
                } else if (winrate >= 60) {
                    winrateElement.className = 'winrate-value display-4 fw-bold text-warning';
                } else {
                    winrateElement.className = 'winrate-value display-4 fw-bold text-danger';
                }
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
        });
}

function updateCalendarInfo() {
    fetch('/api/bolivia_time')
        .then(response => {
            if (!response.ok) throw new Error('Error en tiempo');
            return response.json();
        })
        .then(data => {
            const calendarInfo = document.getElementById('calendar-info');
            if (calendarInfo) {
                const tradingStatus = data.is_scalping_time ? 
                    '<span class="text-success">🟢 ACTIVO</span>' : 
                    '<span class="text-danger">🔴 INACTIVO</span>';
                
                calendarInfo.innerHTML = `
                    <small class="text-muted">
                        📅 ${data.day_of_week} | Scalping: ${tradingStatus} | ${data.time}
                    </small>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando información del calendario:', error);
        });
}

// Utilidades
function formatVolume(volume) {
    if (volume >= 1e9) return (volume / 1e9).toFixed(2) + 'B';
    if (volume >= 1e6) return (volume / 1e6).toFixed(2) + 'M';
    if (volume >= 1e3) return (volume / 1e3).toFixed(2) + 'K';
    return volume.toFixed(2);
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;
    
    const toastId = 'toast-' + Date.now();
    const bgClass = {
        'success': 'bg-success',
        'error': 'bg-danger',
        'warning': 'bg-warning',
        'info': 'bg-info'
    }[type] || 'bg-info';
    
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-white ${bgClass} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {delay: 3000});
    toast.show();
    
    // Remover del DOM después de ocultar
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function showError(message) {
    showToast(message, 'error');
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

function downloadStrategicReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

// Manejo de errores global
window.addEventListener('error', function(e) {
    console.error('Error global:', e.error);
    showToast('Error en la aplicación. Recargando...', 'error');
    setTimeout(() => {
        window.location.reload();
    }, 3000);
});

// Optimización de memoria: limpiar gráficos cuando no son visibles
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Liberar recursos cuando la pestaña no es visible
        console.log('🔄 Liberando recursos (pestaña oculta)');
    } else {
        // Recargar datos cuando la pestaña vuelve a ser visible
        console.log('🔄 Recargando datos (pestaña visible)');
        updateMarketIndicators();
    }
});

// Exportar funciones globales
window.updateCharts = updateCharts;
window.downloadReport = downloadReport;
window.downloadStrategicReport = downloadStrategicReport;
window.showSignalDetails = showSignalDetails;
