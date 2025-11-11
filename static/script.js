// MULTI-TIMEFRAME CRYPTO WGTA PRO - Script Principal Optimizado
// Configuración global optimizada para servidores con 400MB RAM
let currentChart = null;
let currentScatterChart = null;
let currentWhaleChart = null;
let currentAdxChart = null;
let currentRsiChart = null;
let currentMacdChart = null;
let currentSqueezeChart = null;
let currentTrendStrengthChart = null;
let currentSymbol = 'BTC-USDT';
let currentData = null;
let allCryptos = [];
let updateInterval = null;
let drawingToolsActive = false;
let isUpdating = false; // Prevenir actualizaciones simultáneas

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializando Sistema');
    initializeApp();
    setupEventListeners();
    
    // Cargar datos iniciales con retardo para evitar sobrecarga
    setTimeout(() => {
        updateCharts();
    }, 1000);
    
    startAutoUpdate();
});

function initializeApp() {
    console.log('✅ Sistema inicializado');
    loadCryptoRiskClassification();
    loadMarketIndicators();
    updateCalendarInfo();
    updateWinRateDisplay();
}

function setupEventListeners() {
    // Configurar event listeners con debounce para evitar múltiples llamadas
    const debouncedUpdate = debounce(updateCharts, 1000);
    
    document.getElementById('interval-select').addEventListener('change', debouncedUpdate);
    document.getElementById('di-period').addEventListener('change', debouncedUpdate);
    document.getElementById('adx-threshold').addEventListener('change', debouncedUpdate);
    document.getElementById('sr-period').addEventListener('change', debouncedUpdate);
    document.getElementById('rsi-length').addEventListener('change', debouncedUpdate);
    document.getElementById('bb-multiplier').addEventListener('change', debouncedUpdate);
    document.getElementById('leverage').addEventListener('change', debouncedUpdate);
    
    // Configurar buscador de cryptos
    setupCryptoSearch();
    
    // Configurar herramientas de dibujo
    setupDrawingTools();
    
    // Configurar controles de indicadores
    setupIndicatorControls();
}

// Función debounce para evitar múltiples llamadas
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
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
    drawingButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tool = this.dataset.tool;
            activateDrawingTool(tool);
        });
    });
    
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
    
    document.querySelectorAll('.drawing-tool').forEach(btn => {
        btn.classList.remove('active');
    });
    
    event.target.classList.add('active');
    
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'macd-chart', 'squeeze-chart', 'trend-strength-chart'];
    
    charts.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart) {
            let dragmode = false;
            switch(tool) {
                case 'line': dragmode = 'drawline'; break;
                case 'rectangle': dragmode = 'drawrect'; break;
                case 'circle': dragmode = 'drawcircle'; break;
                case 'freehand': dragmode = 'drawfreehand'; break;
            }
            
            Plotly.relayout(chartId, {dragmode: dragmode});
        }
    });
}

function setDrawingColor(color) {
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'macd-chart', 'squeeze-chart', 'trend-strength-chart'];
    
    charts.forEach(chartId => {
        Plotly.relayout(chartId, {
            'newshape.line.color': color,
            'newshape.fillcolor': color + '33'
        });
    });
}

function updateChartIndicators() {
    const showMA9 = document.getElementById('show-ma9').checked;
    const showMA21 = document.getElementById('show-ma21').checked;
    const showMA50 = document.getElementById('show-ma50').checked;
    const showMA200 = document.getElementById('show-ma200').checked;
    
    if (currentData && currentChart) {
        renderCandleChart(currentData, {
            showMA9: showMA9,
            showMA21: showMA21,
            showMA50: showMA50,
            showMA200: showMA200
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
                No se encontraron resultados
            </div>
        `;
        return;
    }
    
    const categories = {};
    filteredCryptos.forEach(crypto => {
        if (!categories[crypto.category]) {
            categories[crypto.category] = [];
        }
        categories[crypto.category].push(crypto);
    });
    
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
    
    const bootstrapDropdown = bootstrap.Dropdown.getInstance(document.getElementById('cryptoDropdown'));
    if (bootstrapDropdown) {
        bootstrapDropdown.hide();
    }
    
    updateCharts();
    updateWinRateDisplay();
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
            console.log(`✅ Cargadas ${allCryptos.length} criptomonedas`);
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
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateTradingAlerts();
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
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    updateInterval = setInterval(() => {
        if (document.visibilityState === 'visible' && !isUpdating) {
            console.log('🔄 Actualización automática del sistema');
            updateCharts();
            updateMarketIndicators();
            updateWinRateDisplay();
        }
    }, 120000); // 120 segundos para reducir carga
}

function updateCharts() {
    if (isUpdating) {
        console.log('⚠️ Actualización en curso, omitiendo...');
        return;
    }
    
    isUpdating = true;
    showLoadingState();
    
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const diPeriod = document.getElementById('di-period').value;
    const adxThreshold = document.getElementById('adx-threshold').value;
    const srPeriod = document.getElementById('sr-period').value;
    const rsiLength = document.getElementById('rsi-length').value;
    const bbMultiplier = document.getElementById('bb-multiplier').value;
    const leverage = document.getElementById('leverage').value;
    
    console.log(`📊 Actualizando gráficos para ${symbol} en ${interval}`);
    
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    updateScatterChart(interval);
    updateMultipleSignals(interval);
    updateWinRateDisplay();
    
    // Liberar flag después de un tiempo mínimo
    setTimeout(() => {
        isUpdating = false;
    }, 2000);
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateTradingAlerts();
    updateCalendarInfo();
}

function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage) {
    const url = `/api/signals?symbol=${symbol}&interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&leverage=${leverage}`;
    
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
            renderWhaleChart(data);
            renderAdxChart(data);
            renderRsiMaverickChart(data);
            renderMacdChart(data);
            renderSqueezeChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
        })
        .catch(error => {
            console.error('Error en updateMainChart:', error);
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
        win_rate: 65,
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
                <p>Intentando cargar datos de respaldo...</p>
            </div>
        `;
        return;
    }

    // Limitar datos para mejor rendimiento
    const displayData = data.data.slice(-50);
    const dates = displayData.map(d => {
        if (typeof d.timestamp === 'string') {
            return new Date(d.timestamp);
        } else if (d.timestamp instanceof Date) {
            return d.timestamp;
        } else {
            return new Date();
        }
    });
    
    const opens = displayData.map(d => parseFloat(d.open));
    const highs = displayData.map(d => parseFloat(d.high));
    const lows = displayData.map(d => parseFloat(d.low));
    const closes = displayData.map(d => parseFloat(d.close));
    
    const traces = [{
        type: 'candlestick',
        x: dates,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        increasing: {line: {color: '#00C853'}, fillcolor: '#00C853'},
        decreasing: {line: {color: '#FF1744'}, fillcolor: '#FF1744'},
        name: 'Precio'
    }];
    
    // Añadir niveles de trading
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
    
    if (data.entry && data.take_profit) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.entry, data.entry],
            mode: 'lines',
            line: {color: '#FFD700', dash: 'solid', width: 2},
            name: 'Entrada'
        });
        
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
    
    // Añadir medias móviles si están disponibles y activadas
    if (data.indicators) {
        const options = indicatorOptions || {
            showMA9: document.getElementById('show-ma9')?.checked || false,
            showMA21: document.getElementById('show-ma21')?.checked || false,
            showMA50: document.getElementById('show-ma50')?.checked || false,
            showMA200: document.getElementById('show-ma200')?.checked || false
        };
        
        const indicatorData = data.indicators;
        
        if (options.showMA9 && indicatorData.ma_9) {
            traces.push({
                type: 'scatter',
                x: dates,
                y: indicatorData.ma_9.slice(-50),
                mode: 'lines',
                line: {color: '#FF9800', width: 1},
                name: 'MA 9'
            });
        }
        
        if (options.showMA21 && indicatorData.ma_21) {
            traces.push({
                type: 'scatter',
                x: dates,
                y: indicatorData.ma_21.slice(-50),
                mode: 'lines',
                line: {color: '#2196F3', width: 1},
                name: 'MA 21'
            });
        }
        
        if (options.showMA50 && indicatorData.ma_50) {
            traces.push({
                type: 'scatter',
                x: dates,
                y: indicatorData.ma_50.slice(-50),
                mode: 'lines',
                line: {color: '#9C27B0', width: 1},
                name: 'MA 50'
            });
        }
    }
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas | Señal: ${data.signal} | Score: ${data.signal_score}%`,
            font: {color: '#ffffff', size: 14}
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
        dragmode: drawingToolsActive ? 'drawline' : 'pan'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawrect', 'drawcircle'],
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
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

function renderWhaleChart(data) {
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
    const whalePump = data.indicators.whale_pump?.slice(-50) || [];
    const whaleDump = data.indicators.whale_dump?.slice(-50) || [];
    
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
        barmode: 'overlay',
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : 'pan'
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
    const adx = data.indicators.adx?.slice(-50) || [];
    const plusDi = data.indicators.plus_di?.slice(-50) || [];
    const minusDi = data.indicators.minus_di?.slice(-50) || [];
    
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
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : 'pan'
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

function renderRsiMaverickChart(data) {
    const chartElement = document.getElementById('rsi-maverick-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const rsi = data.indicators.rsi?.slice(-50) || [];
    const rsiMaverick = data.indicators.rsi_maverick?.slice(-50) || [];
    
    // Convertir RSI Maverick a escala 0-100 para comparación
    const rsiMaverickScaled = rsiMaverick.map(val => val * 100);
    
    const traces = [
        {
            x: dates,
            y: rsi,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Tradicional',
            line: {color: '#2196F3', width: 2}
        },
        {
            x: dates,
            y: rsiMaverickScaled,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B)',
            line: {color: '#FF9800', width: 2}
        },
        {
            x: [dates[0], dates[dates.length - 1]],
            y: [70, 70],
            type: 'scatter',
            mode: 'lines',
            line: {color: 'red', dash: 'dash', width: 1},
            name: 'Sobrecompra',
            showlegend: false
        },
        {
            x: [dates[0], dates[dates.length - 1]],
            y: [30, 30],
            type: 'scatter',
            mode: 'lines',
            line: {color: 'green', dash: 'dash', width: 1},
            name: 'Sobreventa',
            showlegend: false
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Tradicional vs RSI Maverick (Bandas Bollinger %B)',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Valor RSI',
            gridcolor: '#444',
            zerolinecolor: '#444',
            range: [0, 100]
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : 'pan'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentRsiChart) {
        Plotly.purge('rsi-maverick-chart');
    }
    
    currentRsiChart = Plotly.newPlot('rsi-maverick-chart', traces, layout, config);
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
    const macdLine = data.indicators.macd?.slice(-50) || [];
    const macdSignal = data.indicators.macd_signal?.slice(-50) || [];
    const macdHistogram = data.indicators.macd_histogram?.slice(-50) || [];
    
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
            line: {color: '#FF9800', width: 2}
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
            title: 'Valor MACD',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : 'pan'
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
                <p class="mb-0">No hay datos de Squeeze disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const squeezeMomentum = data.indicators.squeeze_momentum?.slice(-50) || [];
    const squeezeOn = data.indicators.squeeze_on?.slice(-50) || [];
    
    const traces = [
        {
            x: dates,
            y: squeezeMomentum,
            type: 'bar',
            name: 'Squeeze Momentum',
            marker: {
                color: squeezeMomentum.map((val, idx) => 
                    squeezeOn[idx] ? '#FF9800' : (val >= 0 ? '#00C853' : '#FF1744')
                )
            }
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
            title: 'Momentum',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : 'pan'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentSqueezeChart) {
        Plotly.purge('squeeze-chart');
    }
    
    currentSqueezeChart = Plotly.newPlot('squeeze-chart', traces, layout, config);
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
    const trendStrength = data.indicators.trend_strength?.slice(-50) || [];
    const noTradeZones = data.indicators.no_trade_zones?.slice(-50) || [];
    const colors = data.indicators.colors?.slice(-50) || [];
    
    const traces = [{
        x: dates,
        y: trendStrength,
        type: 'bar',
        name: 'Fuerza de Tendencia Maverick',
        marker: {
            color: colors
        }
    }];
    
    // Añadir marcadores para zonas de no operar
    noTradeZones.forEach((isNoTrade, index) => {
        if (isNoTrade) {
            traces.push({
                x: [dates[index]],
                y: [trendStrength[index]],
                type: 'scatter',
                mode: 'markers',
                marker: {
                    color: 'red',
                    size: 8,
                    symbol: 'x'
                },
                name: 'Zona NO OPERAR',
                showlegend: index === 0
            });
        }
    });
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Fuerza de Tendencia',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : 'pan'
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

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    if (!marketSummary) return;
    
    const signalClass = data.signal === 'LONG' ? 'text-success' : 
                      data.signal === 'SHORT' ? 'text-danger' : 'text-warning';
    
    marketSummary.innerHTML = `
        <div class="market-summary-content">
            <div class="row text-center mb-3">
                <div class="col-6">
                    <div class="card bg-dark border-secondary">
                        <div class="card-body py-2">
                            <small class="text-muted">Precio Actual</small>
                            <div class="h6 mb-0">$${data.current_price?.toFixed(6) || '0.000000'}</div>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card bg-dark border-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Señal</small>
                            <div class="h6 mb-0 ${signalClass}">${data.signal}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row text-center mb-3">
                <div class="col-4">
                    <small class="text-muted">Score</small>
                    <div class="h6 ${data.signal_score >= 70 ? 'text-success' : data.signal_score >= 50 ? 'text-warning' : 'text-danger'}">
                        ${data.signal_score}%
                    </div>
                </div>
                <div class="col-4">
                    <small class="text-muted">ADX</small>
                    <div class="h6 ${data.adx > 25 ? 'text-success' : 'text-warning'}">
                        ${data.adx?.toFixed(1) || '0'}
                    </div>
                </div>
                <div class="col-4">
                    <small class="text-muted">Volumen</small>
                    <div class="h6 text-info">
                        ${(data.volume > data.volume_ma ? '📈' : '📉')}
                    </div>
                </div>
            </div>
            
            <div class="signal-details">
                <div class="d-flex justify-content-between small mb-1">
                    <span>Entrada:</span>
                    <span class="text-warning">$${data.entry?.toFixed(6) || '0.000000'}</span>
                </div>
                <div class="d-flex justify-content-between small mb-1">
                    <span>Stop Loss:</span>
                    <span class="text-danger">$${data.stop_loss?.toFixed(6) || '0.000000'}</span>
                </div>
                <div class="d-flex justify-content-between small">
                    <span>Take Profit:</span>
                    <span class="text-success">$${data.take_profit?.[0]?.toFixed(6) || '0.000000'}</span>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const signalAnalysis = document.getElementById('signal-analysis');
    if (!signalAnalysis) return;
    
    const conditionsList = data.fulfilled_conditions?.map(condition => 
        `<li class="text-success">✓ ${condition}</li>`
    ).join('') || '<li class="text-muted">No se cumplen condiciones suficientes</li>';
    
    signalAnalysis.innerHTML = `
        <div class="signal-analysis-content">
            <div class="alert alert-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'warning'}">
                <h6 class="alert-heading">Señal: ${data.signal}</h6>
                <p class="mb-2">Score: <strong>${data.signal_score}%</strong></p>
                <p class="mb-2">WinRate: <strong>${data.win_rate}%</strong></p>
            </div>
            
            <div class="conditions-list">
                <h6>Condiciones Cumplidas:</h6>
                <ul class="small mb-0">
                    ${conditionsList}
                </ul>
            </div>
            
            ${data.obligatory_conditions_met ? 
                '<div class="alert alert-success mt-2 py-1 small">✅ Condiciones Multi-TF Obligatorias Cumplidas</div>' : 
                '<div class="alert alert-warning mt-2 py-1 small">⚠️ Esperando confirmación Multi-TF</div>'
            }
        </div>
    `;
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
    
    if (!data || data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }

    const traces = [];
    
    // Agrupar por categoría de riesgo
    const riskCategories = {
        'bajo': {x: [], y: [], text: [], score: [], symbol: [], color: '#00C853'},
        'medio': {x: [], y: [], text: [], score: [], symbol: [], color: '#FFC107'},
        'alto': {x: [], y: [], text: [], score: [], symbol: [], color: '#FF1744'},
        'memecoins': {x: [], y: [], text: [], score: [], symbol: [], color: '#9C27B0'}
    };
    
    data.forEach(item => {
        const category = item.risk_category || 'medio';
        if (riskCategories[category]) {
            riskCategories[category].x.push(item.x);
            riskCategories[category].y.push(item.y);
            riskCategories[category].text.push(
                `${item.symbol}<br>Score: ${item.signal_score}%<br>Precio: $${item.current_price?.toFixed(6)}`
            );
            riskCategories[category].score.push(item.signal_score);
            riskCategories[category].symbol.push(item.symbol);
        }
    });
    
    Object.keys(riskCategories).forEach(category => {
        const categoryData = riskCategories[category];
        if (categoryData.x.length > 0) {
            traces.push({
                x: categoryData.x,
                y: categoryData.y,
                text: categoryData.text,
                mode: 'markers',
                type: 'scatter',
                name: category.toUpperCase(),
                marker: {
                    color: categoryData.color,
                    size: categoryData.score.map(score => Math.max(8, score / 2)),
                    sizemode: 'diameter',
                    sizeref: 10,
                    opacity: 0.7
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
            title: 'Presión Compradora (Ballenas +DI)',
            gridcolor: '#444',
            zerolinecolor: '#444',
            range: [0, 100]
        },
        yaxis: {
            title: 'Presión Vendedora (Ballenas -DI)',
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
            font: {color: '#ffffff'}
        },
        margin: {t: 80, r: 50, b: 50, l: 50},
        hovermode: 'closest'
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

function updateMultipleSignals(interval) {
    fetch(`/api/multiple_signals?interval=${interval}`)
        .then(response => {
            if (!response.ok) throw new Error('Error en múltiples señales');
            return response.json();
        })
        .then(signalsData => {
            updateSignalsTables(signalsData);
        })
        .catch(error => {
            console.error('Error actualizando múltiples señales:', error);
        });
}

function updateSignalsTables(signalsData) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    if (longTable) {
        if (signalsData.long_signals && signalsData.long_signals.length > 0) {
            longTable.innerHTML = signalsData.long_signals.slice(0, 5).map((signal, index) => `
                <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                    <td>${index + 1}</td>
                    <td>${signal.symbol}</td>
                    <td><span class="badge bg-success">${signal.signal_score}%</span></td>
                    <td>${signal.win_rate}%</td>
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
    }
    
    if (shortTable) {
        if (signalsData.short_signals && signalsData.short_signals.length > 0) {
            shortTable.innerHTML = signalsData.short_signals.slice(0, 5).map((signal, index) => `
                <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                    <td>${index + 1}</td>
                    <td>${signal.symbol}</td>
                    <td><span class="badge bg-danger">${signal.signal_score}%</span></td>
                    <td>${signal.win_rate}%</td>
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
}

function showSignalDetails(symbol) {
    // Implementar modal de detalles de señal
    console.log('Mostrar detalles para:', symbol);
    // Aquí puedes implementar un modal con información detallada
}

function updateWinRateDisplay() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/win_rate?symbol=${symbol}&interval=${interval}`)
        .then(response => {
            if (!response.ok) throw new Error('Error en winrate');
            return response.json();
        })
        .then(data => {
            const mainWinRate = document.getElementById('main-win-rate');
            const winRateDisplay = document.getElementById('win-rate-display');
            
            if (mainWinRate) {
                mainWinRate.innerHTML = `
                    <h2 class="text-success mb-1">${data.win_rate?.toFixed(1) || '0'}%</h2>
                    <p class="small text-muted mb-0">Efectividad del Sistema</p>
                `;
            }
            
            if (winRateDisplay) {
                winRateDisplay.innerHTML = `
                    <h4 class="text-success mb-1">${data.win_rate?.toFixed(1) || '0'}%</h4>
                    <p class="small text-muted mb-0">WinRate Actual</p>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
        });
}

function updateFearGreedIndex() {
    // Implementar índice de miedo y codicia
}

function updateMarketRecommendations() {
    // Implementar recomendaciones de mercado
}

function updateTradingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => {
            if (!response.ok) throw new Error('Error en alertas');
            return response.json();
        })
        .then(data => {
            updateAlertsDisplay(data.alerts || []);
        })
        .catch(error => {
            console.error('Error actualizando alertas:', error);
        });
}

function updateAlertsDisplay(alerts) {
    const alertsContainer = document.getElementById('scalping-alerts');
    if (!alertsContainer) return;
    
    if (alerts.length === 0) {
        alertsContainer.innerHTML = `
            <div class="text-center py-2">
                <p class="text-muted mb-0 small">No hay alertas activas</p>
            </div>
        `;
        return;
    }
    
    alertsContainer.innerHTML = alerts.slice(0, 5).map(alert => `
        <div class="alert alert-${alert.signal === 'LONG' ? 'success' : 'danger'} py-2 mb-2">
            <div class="d-flex justify-content-between align-items-center">
                <strong>${alert.symbol}</strong>
                <span class="badge bg-${alert.signal === 'LONG' ? 'success' : 'danger'}">
                    ${alert.signal}
                </span>
            </div>
            <small class="d-block">Score: ${alert.score}%</small>
            <small class="d-block">${alert.interval}</small>
        </div>
    `).join('');
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
                        📅 ${data.day_of_week} | Scalping: ${tradingStatus} | Horario: 4am-4pm L-V
                    </small>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando información del calendario:', error);
        });
}

function showError(message) {
    console.error('❌ Error:', message);
    
    // Mostrar notificación toast
    const toastContainer = document.getElementById('toast-container');
    if (toastContainer) {
        const toastId = 'error-' + Date.now();
        toastContainer.innerHTML += `
            <div class="toast align-items-center text-bg-danger border-0" id="${toastId}">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
    }
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

// Función para limpiar memoria y destruir gráficos
function cleanup() {
    if (currentChart) {
        Plotly.purge('candle-chart');
        currentChart = null;
    }
    if (currentScatterChart) {
        Plotly.purge('scatter-chart');
        currentScatterChart = null;
    }
    if (currentWhaleChart) {
        Plotly.purge('whale-chart');
        currentWhaleChart = null;
    }
    if (currentAdxChart) {
        Plotly.purge('adx-chart');
        currentAdxChart = null;
    }
    if (currentRsiChart) {
        Plotly.purge('rsi-maverick-chart');
        currentRsiChart = null;
    }
    if (currentMacdChart) {
        Plotly.purge('macd-chart');
        currentMacdChart = null;
    }
    if (currentSqueezeChart) {
        Plotly.purge('squeeze-chart');
        currentSqueezeChart = null;
    }
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
        currentTrendStrengthChart = null;
    }
    
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
    
    currentData = null;
    isUpdating = false;
    
    console.log('🧹 Memoria limpiada');
}

// Limpiar al cerrar la página
window.addEventListener('beforeunload', cleanup);
window.addEventListener('pagehide', cleanup);

// Exportar funciones para uso global
window.updateCharts = updateCharts;
window.downloadReport = downloadReport;
window.selectCrypto = selectCrypto;
window.cleanup = cleanup;
