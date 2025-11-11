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
let isUpdating = false; // Prevenir recursión

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializando Sistema');
    initializeApp();
    setupEventListeners();
    loadInitialData();
});

function initializeApp() {
    console.log('📊 Inicializando aplicación...');
    loadCryptoRiskClassification();
    updateCalendarInfo();
    setupCryptoDropdown();
}

function setupEventListeners() {
    console.log('🔧 Configurando event listeners...');
    
    // Configurar event listeners para los controles
    const controls = [
        'interval-select', 'di-period', 'adx-threshold', 'sr-period',
        'rsi-length', 'bb-multiplier', 'volume-filter', 'leverage'
    ];
    
    controls.forEach(controlId => {
        const element = document.getElementById(controlId);
        if (element) {
            element.addEventListener('change', debounce(updateCharts, 500));
        }
    });
    
    document.getElementById('aux-indicator').addEventListener('change', updateAuxChart);
    
    // Configurar buscador de cryptos
    setupCryptoSearch();
    
    // Configurar herramientas de dibujo
    setupDrawingTools();
    
    // Configurar controles de indicadores
    setupIndicatorControls();
}

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

function loadInitialData() {
    console.log('📥 Cargando datos iniciales...');
    updateCharts();
    startAutoUpdate();
}

function setupCryptoDropdown() {
    const cryptoList = document.getElementById('crypto-list');
    if (!cryptoList) return;
    
    // Cargar lista básica inicial
    const basicSymbols = [
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT', 'XRP-USDT',
        'ADA-USDT', 'AVAX-USDT', 'DOT-USDT', 'DOGE-USDT', 'MATIC-USDT'
    ];
    
    basicSymbols.forEach(symbol => {
        const item = document.createElement('a');
        item.className = 'dropdown-item crypto-item';
        item.href = '#';
        item.textContent = symbol;
        item.addEventListener('click', function(e) {
            e.preventDefault();
            selectCrypto(symbol);
        });
        cryptoList.appendChild(item);
    });
}

function updateCalendarInfo() {
    fetch('/api/bolivia_time')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            const calendarInfo = document.getElementById('calendar-info');
            if (calendarInfo) {
                const tradingStatus = data.is_scalping_time ? 
                    '<span class="text-success">🟢 ACTIVO</span>' : 
                    '<span class="text-danger">🔴 INACTIVO</span>';
                
                calendarInfo.innerHTML = `
                    <small class="text-muted text-center d-block">
                        📅 ${data.date} | Hora Bolivia: ${data.time} | Trading: ${tradingStatus}
                    </small>
                `;
            }
        })
        .catch(error => {
            console.log('ℹ️ Usando hora local (fallback)');
            const now = new Date();
            const calendarInfo = document.getElementById('calendar-info');
            if (calendarInfo) {
                calendarInfo.innerHTML = `
                    <small class="text-muted text-center d-block">
                        📅 ${now.toLocaleDateString()} | Hora Local: ${now.toLocaleTimeString()}
                    </small>
                `;
            }
        });
}

function updateWinrateDisplay() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/win_rate?symbol=${symbol}&interval=${interval}`)
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            const winrateDisplay = document.getElementById('winrate-display');
            if (winrateDisplay) {
                const winrateClass = data.win_rate >= 60 ? 'text-success' : 
                                   data.win_rate >= 50 ? 'text-warning' : 'text-danger';
                
                winrateDisplay.innerHTML = `
                    <div class="winrate-main text-center">
                        <div class="winrate-value ${winrateClass}" style="font-size: 2rem; font-weight: bold;">
                            ${data.win_rate}%
                        </div>
                        <div class="winrate-stats small text-muted mt-2">
                            ${data.successful_signals}/${data.total_signals} señales exitosas
                        </div>
                        <div class="progress mt-2" style="height: 8px;">
                            <div class="progress-bar ${winrateClass.replace('text-', 'bg-')}" 
                                 role="progressbar" 
                                 style="width: ${data.win_rate}%"
                                 aria-valuenow="${data.win_rate}" 
                                 aria-valuemin="0" 
                                 aria-valuemax="100">
                            </div>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.log('ℹ️ Winrate no disponible temporalmente');
            const winrateDisplay = document.getElementById('winrate-display');
            if (winrateDisplay) {
                winrateDisplay.innerHTML = `
                    <div class="text-center text-muted">
                        <div class="winrate-value text-secondary">--%</div>
                        <small>Datos no disponibles</small>
                    </div>
                `;
            }
        });
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', function() {
        const filter = this.value.toUpperCase();
        filterCryptoList(filter);
    });
    
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
    
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'aux-chart', 'trend-strength-chart'];
    
    charts.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart) {
            let dragmode = false;
            switch(tool) {
                case 'line': dragmode = 'drawline'; break;
                case 'rectangle': dragmode = 'drawrect'; break;
                case 'circle': dragmode = 'drawcircle'; break;
                case 'text': dragmode = 'drawtext'; break;
                case 'freehand': dragmode = 'drawfreehand'; break;
                case 'marker': dragmode = 'marker'; break;
            }
            
            Plotly.relayout(chartId, {dragmode: dragmode});
        }
    });
}

function setDrawingColor(color) {
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'aux-chart', 'trend-strength-chart'];
    
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
    const showBB = document.getElementById('show-bollinger').checked;
    
    if (currentData) {
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
    console.log(`🎯 Seleccionando crypto: ${symbol}`);
    currentSymbol = symbol;
    document.getElementById('selected-crypto').textContent = symbol;
    
    const bootstrapDropdown = bootstrap.Dropdown.getInstance(document.getElementById('cryptoDropdown'));
    if (bootstrapDropdown) {
        bootstrapDropdown.hide();
    }
    
    updateCharts();
    updateWinrateDisplay();
}

function loadCryptoRiskClassification() {
    fetch('/api/crypto_risk_classification')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
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
            
            console.log(`✅ Cargadas ${allCryptos.length} criptomonedas`);
            filterCryptoList('');
        })
        .catch(error => {
            console.log('ℹ️ Cargando lista básica de cryptos (fallback)');
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
            console.log('🔄 Actualización automática');
            updateCharts();
            updateMarketIndicators();
        }
    }, 120000); // 120 segundos para reducir carga
}

function updateCharts() {
    if (isUpdating) {
        console.log('⚠️ Actualización en curso, omitiendo...');
        return;
    }
    
    isUpdating = true;
    console.log('📈 Actualizando gráficos...');
    
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
    
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage)
        .finally(() => {
            isUpdating = false;
        });
    
    updateScatterChart();
    updateMultipleSignals();
    updateWinrateDisplay();
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateTradingAlerts();
    updateCalendarInfo();
}

async function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/signals?symbol=${symbol}&interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        currentData = data;
        renderCandleChart(data);
        renderWhaleChart(data);
        renderAdxChart(data);
        renderRsiMaverickChart(data);
        renderTrendStrengthChart(data);
        updateMarketSummary(data);
        updateSignalAnalysis(data);
        
    } catch (error) {
        console.error('❌ Error en updateMainChart:', error);
        showError('Error al cargar datos. Intentando nuevamente...');
        showSampleData(symbol);
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
        whale_pump: 15,
        whale_dump: 10,
        rsi_maverick: 0.5,
        trend_strength_signal: 'NEUTRAL',
        no_trade_zone: false,
        win_rate: 65,
        total_signals: 100,
        successful_signals: 65,
        fulfilled_conditions: [],
        data: generateSampleData(50)
    };
    
    updateMarketSummary(sampleData);
    updateSignalAnalysis(sampleData);
}

function generateSampleData(count) {
    const data = [];
    let price = 50000;
    const now = new Date();
    
    for (let i = 0; i < count; i++) {
        const date = new Date(now - (count - i) * 3600000);
        const change = (Math.random() - 0.5) * 1000;
        price += change;
        
        data.push({
            timestamp: date.toISOString(),
            open: price - Math.random() * 100,
            high: price + Math.random() * 200,
            low: price - Math.random() * 200,
            close: price,
            volume: 1000000 + Math.random() * 500000
        });
    }
    
    return data;
}

function renderCandleChart(data, indicatorOptions = {}) {
    const chartElement = document.getElementById('candle-chart');
    if (!chartElement) return;
    
    if (!data.data || data.data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center m-3">
                <h5>📊 No hay datos disponibles</h5>
                <p>Intentando recuperar datos...</p>
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
    
    const visibleHighs = highs.slice(-50);
    const visibleLows = lows.slice(-50);
    const minPrice = Math.min(...visibleLows);
    const maxPrice = Math.max(...visibleHighs);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.05;
    
    const layout = {
        title: {
            text: `${data.symbol} - Velas Japonesas | Score: ${data.signal_score || 0}%`,
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
            zerolinecolor: '#444',
            range: [minPrice - padding, maxPrice + padding]
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
        margin: {t: 60, r: 50, b: 50, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentChart) {
        Plotly.purge('candle-chart');
    }
    
    currentChart = Plotly.newPlot('candle-chart', traces, layout, config);
}

function renderWhaleChart(data) {
    const chartElement = document.getElementById('whale-chart');
    if (!chartElement) return;
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center m-2">
                <p class="mb-0">No hay datos de ballenas disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const whalePump = data.indicators.whale_pump || Array(50).fill(0);
    const whaleDump = data.indicators.whale_dump || Array(50).fill(0);
    
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
            text: 'Actividad de Ballenas',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            title: 'Fuerza de Señal',
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        barmode: 'overlay',
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    if (currentWhaleChart) {
        Plotly.purge('whale-chart');
    }
    
    currentWhaleChart = Plotly.newPlot('whale-chart', traces, layout);
}

function renderAdxChart(data) {
    const chartElement = document.getElementById('adx-chart');
    if (!chartElement) return;
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center m-2">
                <p class="mb-0">No hay datos de ADX disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const adx = data.indicators.adx || Array(50).fill(25);
    const plusDi = data.indicators.plus_di || Array(50).fill(20);
    const minusDi = data.indicators.minus_di || Array(50).fill(20);
    
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
            text: 'ADX con Indicadores Direccionales',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            title: 'Valor del Indicador',
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    if (currentAdxChart) {
        Plotly.purge('adx-chart');
    }
    
    currentAdxChart = Plotly.newPlot('adx-chart', traces, layout);
}

function renderRsiMaverickChart(data) {
    const chartElement = document.getElementById('rsi-maverick-chart');
    if (!chartElement) return;
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center m-2">
                <p class="mb-0">No hay datos de RSI Maverick disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const rsiMaverick = data.indicators.rsi_maverick || Array(50).fill(0.5);
    
    const trace = {
        x: dates,
        y: rsiMaverick,
        type: 'scatter',
        mode: 'lines',
        name: 'RSI Maverick (%B)',
        line: {color: '#9C27B0', width: 2}
    };
    
    const layout = {
        title: {
            text: 'RSI Modificado Maverick (Bollinger %B)',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444'
        },
        yaxis: {
            title: 'Valor %B',
            gridcolor: '#444',
            range: [0, 1]
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    if (currentRsiChart) {
        Plotly.purge('rsi-maverick-chart');
    }
    
    currentRsiChart = Plotly.newPlot('rsi-maverick-chart', [trace], layout);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    if (!chartElement) return;
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center m-2">
                <p class="mb-0">No hay datos de Fuerza de Tendencia</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const trendStrength = data.indicators.trend_strength || Array(50).fill(0);
    const colors = data.indicators.colors || Array(50).fill('gray');
    
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
            gridcolor: '#444'
        },
        yaxis: {
            title: 'Fuerza',
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', [trace], layout);
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    if (!marketSummary) return;
    
    const signalClass = data.signal === 'LONG' ? 'text-success' : 
                      data.signal === 'SHORT' ? 'text-danger' : 'text-warning';
    
    marketSummary.innerHTML = `
        <div class="row text-center">
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-2">
                        <small class="text-muted d-block">Señal Actual</small>
                        <span class="${signalClass} fw-bold">${data.signal}</span>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-2">
                        <small class="text-muted d-block">Score</small>
                        <span class="text-warning fw-bold">${data.signal_score}%</span>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-2">
                        <small class="text-muted d-block">Precio</small>
                        <span class="text-white fw-bold">$${data.current_price?.toFixed(4) || '0'}</span>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary">
                    <div class="card-body py-2">
                        <small class="text-muted d-block">Volumen</small>
                        <span class="text-info fw-bold">${formatVolume(data.volume || 0)}</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="mt-2">
            <small class="text-muted">
                <i class="fas fa-info-circle me-1"></i>
                Winrate histórico: <strong>${data.win_rate || 0}%</strong>
            </small>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const signalAnalysis = document.getElementById('signal-analysis');
    if (!signalAnalysis) return;
    
    const conditions = data.fulfilled_conditions || ['Analizando condiciones...'];
    
    signalAnalysis.innerHTML = `
        <div class="signal-analysis-enhanced ${data.signal === 'LONG' ? 'signal-long-enhanced' : data.signal === 'SHORT' ? 'signal-short-enhanced' : 'signal-neutral-enhanced'}">
            <h6 class="mb-3">Análisis de Señal</h6>
            <div class="mb-2">
                <strong>Señal:</strong> 
                <span class="badge ${data.signal === 'LONG' ? 'bg-success' : data.signal === 'SHORT' ? 'bg-danger' : 'bg-secondary'}">
                    ${data.signal} (${data.signal_score}%)
                </span>
            </div>
            <div class="mb-2">
                <strong>Fuerza Tendencia:</strong>
                <span class="trend-strength-indicator badge-${data.trend_strength_signal?.toLowerCase() || 'neutral'}">
                    ${data.trend_strength_signal || 'NEUTRAL'}
                </span>
            </div>
            ${data.no_trade_zone ? `
            <div class="no-trade-alert mt-2">
                <small class="text-danger">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    ZONA DE NO OPERAR ACTIVA
                </small>
            </div>
            ` : ''}
            <div class="mt-3">
                <small class="text-muted d-block mb-2"><strong>Condiciones Cumplidas:</strong></small>
                <div style="max-height: 120px; overflow-y: auto;">
                    ${conditions.map(cond => `
                        <div class="signal-condition condition-${data.signal?.toLowerCase() || 'neutral'} mb-1 p-2 small">
                            <i class="fas fa-check-circle me-2 text-success"></i>${cond}
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
}

function updateScatterChart() {
    const scatterChart = document.getElementById('scatter-chart');
    if (!scatterChart) return;
    
    fetch('/api/scatter_data_improved?interval=4h')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(scatterData => {
            renderScatterChart(scatterData);
        })
        .catch(error => {
            console.log('ℹ️ Scatter chart no disponible');
            scatterChart.innerHTML = `
                <div class="alert alert-warning text-center m-3">
                    <p class="mb-0">Mapa de oportunidades no disponible temporalmente</p>
                </div>
            `;
        });
}

function renderScatterChart(scatterData) {
    const chartElement = document.getElementById('scatter-chart');
    if (!chartElement) return;
    
    if (!scatterData || scatterData.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-info text-center m-2">
                <p class="mb-0">No hay datos para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }

    const traces = [];
    const riskCategories = ['bajo', 'medio', 'alto', 'memecoins'];
    const colors = ['#00C853', '#FFC107', '#FF1744', '#9C27B0'];
    const symbols = ['circle', 'square', 'diamond', 'cross'];
    
    riskCategories.forEach((category, index) => {
        const categoryData = scatterData.filter(d => d.risk_category === category);
        if (categoryData.length > 0) {
            traces.push({
                x: categoryData.map(d => d.x),
                y: categoryData.map(d => d.y),
                text: categoryData.map(d => `${d.symbol}<br>Score: ${d.signal_score}%`),
                mode: 'markers',
                type: 'scatter',
                name: category.toUpperCase(),
                marker: {
                    color: colors[index],
                    size: categoryData.map(d => Math.max(8, d.signal_score / 3)),
                    symbol: symbols[index],
                    line: {width: 1, color: 'white'}
                },
                hovertemplate: '<b>%{text}</b><br>Compra: %{x:.1f}%<br>Venta: %{y:.1f}%<extra></extra>'
            });
        }
    });
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Análisis Multi-Indicador',
            font: {color: '#ffffff', size: 14}
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
            font: {color: '#ffffff'}
        },
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    if (currentScatterChart) {
        Plotly.purge('scatter-chart');
    }
    
    currentScatterChart = Plotly.newPlot('scatter-chart', traces, layout);
}

function updateMultipleSignals() {
    fetch('/api/multiple_signals?interval=4h')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            updateSignalsTable('long-table', data.long_signals || []);
            updateSignalsTable('short-table', data.short_signals || []);
        })
        .catch(error => {
            console.log('ℹ️ Señales múltiples no disponibles');
            updateSignalsTable('long-table', []);
            updateSignalsTable('short-table', []);
        });
}

function updateSignalsTable(tableId, signals) {
    const tableBody = document.getElementById(tableId);
    if (!tableBody) return;
    
    if (signals.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    No hay señales ${tableId.includes('long') ? 'LONG' : 'SHORT'} confirmadas
                </td>
            </tr>
        `;
        return;
    }
    
    tableBody.innerHTML = signals.slice(0, 5).map((signal, index) => `
        <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;">
            <td>${index + 1}</td>
            <td>
                <small>${signal.symbol}</small>
                ${signal.trend_strength_signal === 'STRONG_UP' || signal.trend_strength_signal === 'STRONG_DOWN' ? 
                  '<i class="fas fa-bolt text-warning ms-1"></i>' : ''}
            </td>
            <td>
                <span class="badge ${signal.signal_score >= 80 ? 'bg-success' : signal.signal_score >= 70 ? 'bg-warning' : 'bg-secondary'}">
                    ${signal.signal_score}%
                </span>
            </td>
            <td>
                <small>$${signal.entry?.toFixed(4) || '0'}</small>
            </td>
        </tr>
    `).join('');
}

function updateAuxChart() {
    const auxIndicator = document.getElementById('aux-indicator').value;
    const auxChart = document.getElementById('aux-chart');
    
    if (!auxChart || !currentData) return;
    
    let trace;
    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    
    switch(auxIndicator) {
        case 'rsi':
            const rsi = currentData.indicators?.rsi || Array(50).fill(50);
            trace = {
                x: dates,
                y: rsi,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI Tradicional',
                line: {color: '#2196F3', width: 2}
            };
            break;
        case 'macd':
            const macd = currentData.indicators?.macd || Array(50).fill(0);
            const macdSignal = currentData.indicators?.macd_signal || Array(50).fill(0);
            const macdHistogram = currentData.indicators?.macd_histogram || Array(50).fill(0);
            
            trace = [
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
                    line: {color: '#4ECDC4', width: 1.5}
                },
                {
                    x: dates,
                    y: macdHistogram,
                    type: 'bar',
                    name: 'Histograma',
                    marker: {color: '#FFE66D'}
                }
            ];
            break;
        case 'squeeze':
            const squeezeMomentum = currentData.indicators?.squeeze_momentum || Array(50).fill(0);
            trace = {
                x: dates,
                y: squeezeMomentum,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Squeeze Momentum',
                line: {color: '#9C27B0', width: 2},
                marker: {
                    size: 4,
                    color: squeezeMomentum.map(val => val > 0 ? '#00C853' : '#FF1744')
                }
            };
            break;
        default:
            trace = {
                x: dates,
                y: Array(50).fill(0),
                type: 'scatter',
                mode: 'lines',
                name: 'Datos no disponibles',
                line: {color: '#666', width: 1}
            };
    }
    
    const layout = {
        title: {
            text: `Indicador: ${auxIndicator.toUpperCase()}`,
            font: {color: '#ffffff', size: 14}
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
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    if (currentAuxChart) {
        Plotly.purge('aux-chart');
    }
    
    currentAuxChart = Plotly.newPlot('aux-chart', Array.isArray(trace) ? trace : [trace], layout);
}

function updateFearGreedIndex() {
    const fearGreedElement = document.getElementById('fear-greed-index');
    if (!fearGreedElement) return;
    
    // Simular datos del índice (en un sistema real esto vendría de una API)
    const fearGreedValue = Math.floor(Math.random() * 100);
    let level, color, description;
    
    if (fearGreedValue >= 75) {
        level = 'Extrema Codicia';
        color = 'danger';
        description = 'Mercado sobrecomprado - Precaución';
    } else if (fearGreedValue >= 55) {
        level = 'Codicia';
        color = 'warning';
        description = 'Mercado alcista';
    } else if (fearGreedValue >= 45) {
        level = 'Neutral';
        color = 'info';
        description = 'Mercado equilibrado';
    } else if (fearGreedValue >= 25) {
        level = 'Miedo';
        color = 'primary';
        description = 'Mercado bajista';
    } else {
        level = 'Miedo Extremo';
        color = 'success';
        description = 'Posible oportunidad de compra';
    }
    
    fearGreedElement.innerHTML = `
        <div class="text-center">
            <div class="fear-greed-value display-6 text-${color} fw-bold">${fearGreedValue}</div>
            <div class="fear-greed-level text-${color} fw-bold mb-2">${level}</div>
            <div class="progress fear-greed-progress mb-2">
                <div class="progress-bar bg-${color}" style="width: ${fearGreedValue}%"></div>
            </div>
            <small class="text-muted">${description}</small>
        </div>
    `;
}

function updateMarketRecommendations() {
    const recommendationsElement = document.getElementById('market-recommendations');
    if (!recommendationsElement) return;
    
    const recommendations = [
        {type: 'success', message: '✅ BTC en tendencia alcista confirmada'},
        {type: 'warning', message: '⚠️ Volatilidad moderada en altcoins'},
        {type: 'info', message: '💡 Buen momento para scalping en majors'}
    ];
    
    recommendationsElement.innerHTML = recommendations.map(rec => `
        <div class="alert alert-${rec.type} alert-dismissible fade show mb-2" role="alert">
            <small class="mb-0">${rec.message}</small>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="alert"></button>
        </div>
    `).join('');
}

function updateTradingAlerts() {
    const alertsElement = document.getElementById('scalping-alerts');
    if (!alertsElement) return;
    
    fetch('/api/scalping_alerts')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
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
            
            alertsElement.innerHTML = alerts.slice(0, 3).map(alert => `
                <div class="scalping-alert alert alert-warning mb-2">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <strong>${alert.symbol}</strong> (${alert.interval})<br>
                            <small>${alert.signal} - Score: ${alert.score}%</small>
                        </div>
                        <span class="badge bg-${alert.risk_category === 'bajo' ? 'success' : alert.risk_category === 'medio' ? 'warning' : 'danger'}">
                            ${alert.risk_category}
                        </span>
                    </div>
                    <small class="text-muted d-block mt-1">
                        Entrada: $${alert.entry?.toFixed(4) || '0'}
                    </small>
                </div>
            `).join('');
        })
        .catch(error => {
            console.log('ℹ️ Alertas no disponibles');
            alertsElement.innerHTML = `
                <div class="text-center text-muted py-3">
                    <i class="fas fa-wifi-slash fa-2x mb-2"></i>
                    <p class="mb-0 small">Alertas no disponibles</p>
                </div>
            `;
        });
}

// Funciones utilitarias
function formatVolume(volume) {
    if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    }
    return volume.toFixed(0);
}

function showError(message) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;
    
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-bg-danger border-0';
    toast.id = toastId;
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-exclamation-triangle me-2"></i>${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    setTimeout(() => {
        if (document.getElementById(toastId)) {
            document.getElementById(toastId).remove();
        }
    }, 5000);
}

function showSignalDetails(symbol) {
    // Implementación básica - en un sistema real esto abriría un modal con detalles
    console.log(`Mostrando detalles para: ${symbol}`);
    selectCrypto(symbol);
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}`;
    window.open(url, '_blank');
}

function downloadStrategicReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&strategic=true`;
    window.open(url, '_blank');
}

// Manejo de errores globales
window.addEventListener('error', function(e) {
    console.error('Error global capturado:', e.error);
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Promise rechazada:', e.reason);
});

// Exportar funciones globales
window.selectCrypto = selectCrypto;
window.updateCharts = updateCharts;
window.downloadReport = downloadReport;
window.downloadStrategicReport = downloadStrategicReport;
window.showSignalDetails = showSignalDetails;
