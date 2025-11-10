// MULTI-TIMEFRAME CRYPTO WGTA PRO - JavaScript Completo
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
    updateWinRateDisplay();
    setInterval(updateWinRateDisplay, 30000);
}

function updateWinRateDisplay() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/win_rate?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
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
    document.getElementById('interval-select').addEventListener('change', updateCharts);
    document.getElementById('di-period').addEventListener('change', updateCharts);
    document.getElementById('adx-threshold').addEventListener('change', updateCharts);
    document.getElementById('sr-period').addEventListener('change', updateCharts);
    document.getElementById('rsi-length').addEventListener('change', updateCharts);
    document.getElementById('bb-multiplier').addEventListener('change', updateCharts);
    document.getElementById('volume-filter').addEventListener('change', updateCharts);
    document.getElementById('leverage').addEventListener('change', updateCharts);
    document.getElementById('aux-indicator').addEventListener('change', updateAuxChart);
    
    setupCryptoSearch();
    setupDrawingTools();
    setupIndicatorControls();
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
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
    
    const charts = [
        'candle-chart', 'ma-chart', 'rsi-comparison-chart', 
        'macd-chart', 'squeeze-chart', 'trend-strength-chart',
        'aux-chart'
    ];
    
    charts.forEach(chartId => {
        if (currentChart) {
            Plotly.relayout(chartId, {dragmode: 'drawline'});
        }
    });
}

function setDrawingColor(color) {
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
    
    const dropdown = document.getElementById('crypto-dropdown-menu');
    const bootstrapDropdown = bootstrap.Dropdown.getInstance(document.getElementById('cryptoDropdown'));
    if (bootstrapDropdown) {
        bootstrapDropdown.hide();
    }
    
    updateCharts();
    updateWinRateDisplay();
}

function loadCryptoRiskClassification() {
    fetch('/api/crypto_risk_classification')
        .then(response => response.json())
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
    updateMarketRecommendations();
    updateScalpingAlerts();
    updateExitSignals();
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
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
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
    
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    updateAuxChart();
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
        .then(response => response.json())
        .then(data => {
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
            showSampleData(symbol);
        });
}

function showSampleData(symbol) {
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
    
    if (data.indicators) {
        const options = indicatorOptions || {
            showMA9: false,
            showMA21: false,
            showMA50: false,
            showMA200: false,
            showBB: false
        };
        
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
                <p class="mb-0">No hay datos de Squeeze</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const squeezeMomentum = data.indicators.squeeze_momentum || [];
    
    const colors = squeezeMomentum.map(val => val >= 0 ? '#00C853' : '#FF1744');
    
    const traces = [{
        x: dates,
        y: squeezeMomentum,
        type: 'bar',
        name: 'Squeeze Momentum',
        marker: {color: colors}
    }];
    
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
    
    if (currentSqueezeChart) {
        Plotly.purge('squeeze-chart');
    }
    
    currentSqueezeChart = Plotly.newPlot('squeeze-chart', traces, layout, config);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    
    if (!data.indicators || !data.data || !data.indicators.trend_strength) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Fuerza de Tendencia</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const trendStrength = data.indicators.trend_strength.slice(-50);
    
    const colors = trendStrength.map(val => {
        if (val > 5) return '#00C853';
        if (val < -5) return '#FF1744';
        return '#FF9800';
    });
    
    const traces = [{
        x: dates,
        y: trendStrength,
        type: 'bar',
        name: 'Fuerza de Tendencia',
        marker: {color: colors}
    }];
    
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
        shapes: [
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 5,
                y1: 5,
                line: {color: '#00C853', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: -5,
                y1: -5,
                line: {color: '#FF1744', width: 1, dash: 'dash'}
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
    
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', traces, layout, config);
}

function updateAuxChart() {
    const indicatorType = document.getElementById('aux-indicator').value;
    
    if (!currentData || !currentData.indicators) {
        return;
    }

    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    const chartElement = document.getElementById('aux-chart');
    
    let traces = [];
    let layout = {};
    
    switch(indicatorType) {
        case 'rsi':
            if (currentData.indicators.rsi) {
                traces = [{
                    x: dates,
                    y: currentData.indicators.rsi.slice(-50),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'RSI Tradicional',
                    line: {color: '#FF6B6B', width: 2}
                }];
            }
            break;
            
        case 'macd':
            if (currentData.indicators.macd) {
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
            }
            break;
            
        case 'squeeze':
            if (currentData.indicators.squeeze_momentum) {
                const squeezeMomentum = currentData.indicators.squeeze_momentum.slice(-50);
                const colors = squeezeMomentum.map(val => val >= 0 ? '#00C853' : '#FF1744');
                
                traces = [{
                    x: dates,
                    y: squeezeMomentum,
                    type: 'bar',
                    name: 'Squeeze Momentum',
                    marker: {color: colors}
                }];
            }
            break;
            
        case 'moving_averages':
            traces = [{
                x: dates,
                y: currentData.data.slice(-50).map(d => parseFloat(d.close)),
                type: 'scatter',
                mode: 'lines',
                name: 'Precio',
                line: {color: 'white', width: 2}
            }];
            
            if (currentData.indicators.ma_9) {
                traces.push({
                    x: dates,
                    y: currentData.indicators.ma_9.slice(-50),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MA 9',
                    line: {color: '#FF9800', width: 1}
                });
            }
            
            if (currentData.indicators.ma_21) {
                traces.push({
                    x: dates,
                    y: currentData.indicators.ma_21.slice(-50),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MA 21',
                    line: {color: '#2196F3', width: 1}
                });
            }
            break;
    }
    
    if (traces.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos disponibles para este indicador</p>
            </div>
        `;
        return;
    }
    
    layout = {
        title: {
            text: `Indicador: ${indicatorType.toUpperCase()}`,
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
    fetch(`/api/scatter_signals?interval=${interval}&volume_filter=${volumeFilter}`)
        .then(response => response.json())
        .then(data => {
            renderScatterChart(data);
        })
        .catch(error => {
            console.error('Error cargando gráfico de dispersión:', error);
        });
}

function renderScatterChart(data) {
    const chartElement = document.getElementById('scatter-chart');
    
    if (!data || !data.signals) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de señales disponibles</p>
            </div>
        `;
        return;
    }

    const longSignals = data.signals.filter(s => s.signal === 'LONG');
    const shortSignals = data.signals.filter(s => s.signal === 'SHORT');
    const neutralSignals = data.signals.filter(s => s.signal === 'NEUTRAL');
    
    const traces = [];
    
    if (longSignals.length > 0) {
        traces.push({
            x: longSignals.map(s => s.score),
            y: longSignals.map(s => s.volume_change || 0),
            text: longSignals.map(s => s.symbol),
            mode: 'markers',
            type: 'scatter',
            name: 'LONG',
            marker: {
                color: '#00C853',
                size: 12,
                symbol: 'circle'
            }
        });
    }
    
    if (shortSignals.length > 0) {
        traces.push({
            x: shortSignals.map(s => s.score),
            y: shortSignals.map(s => s.volume_change || 0),
            text: shortSignals.map(s => s.symbol),
            mode: 'markers',
            type: 'scatter',
            name: 'SHORT',
            marker: {
                color: '#FF1744',
                size: 12,
                symbol: 'circle'
            }
        });
    }
    
    if (neutralSignals.length > 0) {
        traces.push({
            x: neutralSignals.map(s => s.score),
            y: neutralSignals.map(s => s.volume_change || 0),
            text: neutralSignals.map(s => s.symbol),
            mode: 'markers',
            type: 'scatter',
            name: 'NEUTRAL',
            marker: {
                color: '#FF9800',
                size: 8,
                symbol: 'circle'
            }
        });
    }
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Score vs Volumen',
            font: {color: '#ffffff', size: 16}
        },
        xaxis: {
            title: 'Score de Señal',
            gridcolor: '#444',
            zerolinecolor: '#444',
            range: [0, 100]
        },
        yaxis: {
            title: 'Cambio de Volumen %',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 60, r: 50, b: 50, l: 50}
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
    fetch(`/api/multiple_signals?interval=${interval}&volume_filter=${volumeFilter}`)
        .then(response => response.json())
        .then(data => {
            updateSignalTables(data);
        })
        .catch(error => {
            console.error('Error cargando señales múltiples:', error);
        });
}

function updateSignalTables(data) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    if (!data || !data.signals) {
        longTable.innerHTML = '<tr><td colspan="4" class="text-center py-3">No hay señales LONG</td></tr>';
        shortTable.innerHTML = '<tr><td colspan="4" class="text-center py-3">No hay señales SHORT</td></tr>';
        return;
    }
    
    const longSignals = data.signals.filter(s => s.signal === 'LONG').slice(0, 5);
    const shortSignals = data.signals.filter(s => s.signal === 'SHORT').slice(0, 5);
    
    // Actualizar tabla LONG
    if (longSignals.length > 0) {
        longTable.innerHTML = longSignals.map((signal, index) => `
            <tr>
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-success">${signal.score}%</span></td>
                <td>${signal.entry_price ? signal.entry_price.toFixed(2) : 'N/A'}</td>
            </tr>
        `).join('');
    } else {
        longTable.innerHTML = '<tr><td colspan="4" class="text-center py-3">No hay señales LONG confirmadas</td></tr>';
    }
    
    // Actualizar tabla SHORT
    if (shortSignals.length > 0) {
        shortTable.innerHTML = shortSignals.map((signal, index) => `
            <tr>
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-danger">${signal.score}%</span></td>
                <td>${signal.entry_price ? signal.entry_price.toFixed(2) : 'N/A'}</td>
            </tr>
        `).join('');
    } else {
        shortTable.innerHTML = '<tr><td colspan="4" class="text-center py-3">No hay señales SHORT confirmadas</td></tr>';
    }
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    
    if (!marketSummary) return;
    
    const signalColor = data.signal === 'LONG' ? 'success' : 
                       data.signal === 'SHORT' ? 'danger' : 'warning';
    
    const signalIcon = data.signal === 'LONG' ? '🟢' : 
                      data.signal === 'SHORT' ? '🔴' : '🟡';
    
    marketSummary.innerHTML = `
        <div class="row text-center">
            <div class="col-6 mb-3">
                <div class="card trading-card ${data.signal.toLowerCase()}">
                    <div class="card-body p-2">
                        <h6 class="mb-1">Señal</h6>
                        <div class="h4 text-${signalColor}">${signalIcon} ${data.signal}</div>
                        <small class="text-muted">Score: ${data.signal_score}%</small>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark">
                    <div class="card-body p-2">
                        <h6 class="mb-1">Precio Actual</h6>
                        <div class="h4 text-warning">$${data.current_price ? data.current_price.toFixed(2) : 'N/A'}</div>
                        <small class="text-muted">${data.symbol}</small>
                    </div>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-12">
                <div class="d-flex justify-content-between small mb-2">
                    <span>Entrada:</span>
                    <span class="text-warning">$${data.entry ? data.entry.toFixed(2) : 'N/A'}</span>
                </div>
                <div class="d-flex justify-content-between small mb-2">
                    <span>Stop Loss:</span>
                    <span class="text-danger">$${data.stop_loss ? data.stop_loss.toFixed(2) : 'N/A'}</span>
                </div>
                <div class="d-flex justify-content-between small">
                    <span>Take Profit:</span>
                    <span class="text-success">$${data.take_profit && data.take_profit[0] ? data.take_profit[0].toFixed(2) : 'N/A'}</span>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (!signalAnalysis) return;
    
    let analysisHTML = '';
    
    if (data.signal === 'LONG') {
        analysisHTML = `
            <div class="alert alert-success">
                <h6><i class="fas fa-arrow-up me-2"></i>SEÑAL LONG DETECTADA</h6>
                <p class="mb-1">Score: <strong>${data.signal_score}%</strong></p>
                <p class="mb-1">Confianza: <strong>${data.signal_score >= 70 ? 'ALTA' : data.signal_score >= 50 ? 'MEDIA' : 'BAJA'}</strong></p>
            </div>
        `;
    } else if (data.signal === 'SHORT') {
        analysisHTML = `
            <div class="alert alert-danger">
                <h6><i class="fas fa-arrow-down me-2"></i>SEÑAL SHORT DETECTADA</h6>
                <p class="mb-1">Score: <strong>${data.signal_score}%</strong></p>
                <p class="mb-1">Confianza: <strong>${data.signal_score >= 70 ? 'ALTA' : data.signal_score >= 50 ? 'MEDIA' : 'BAJA'}</strong></p>
            </div>
        `;
    } else {
        analysisHTML = `
            <div class="alert alert-warning">
                <h6><i class="fas fa-pause me-2"></i>SIN SEÑAL CLARA</h6>
                <p class="mb-0">Esperando mejores condiciones de entrada</p>
            </div>
        `;
    }
    
    signalAnalysis.innerHTML = analysisHTML;
}

function updateObligatoryConditions(data) {
    const conditionsElement = document.getElementById('obligatory-conditions');
    
    if (!conditionsElement) return;
    
    let conditionsHTML = '';
    
    if (data.obligatory_met_long) {
        conditionsHTML = `
            <div class="alert alert-success">
                <h6><i class="fas fa-check-circle me-2"></i>CONDICIONES LONG CUMPLIDAS</h6>
                <p class="mb-0">Todas las condiciones obligatorias para LONG están verificadas</p>
            </div>
        `;
    } else if (data.obligatory_met_short) {
        conditionsHTML = `
            <div class="alert alert-danger">
                <h6><i class="fas fa-check-circle me-2"></i>CONDICIONES SHORT CUMPLIDAS</h6>
                <p class="mb-0">Todas las condiciones obligatorias para SHORT están verificadas</p>
            </div>
        `;
    } else {
        conditionsHTML = `
            <div class="alert alert-secondary">
                <h6><i class="fas fa-times-circle me-2"></i>CONDICIONES NO CUMPLIDAS</h6>
                <p class="mb-0">Faltan condiciones obligatorias para generar señal</p>
            </div>
        `;
    }
    
    conditionsElement.innerHTML = conditionsHTML;
}

function updateMarketRecommendations() {
    const recommendationsElement = document.getElementById('market-recommendations');
    
    if (!recommendationsElement) return;
    
    fetch('/api/market_recommendations')
        .then(response => response.json())
        .then(data => {
            let html = '<div class="card bg-dark border-info mb-3">';
            html += '<div class="card-header bg-info bg-opacity-25"><h6 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Recomendaciones</h6></div>';
            html += '<div class="card-body">';
            
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(rec => {
                    html += `<div class="alert alert-${rec.type} small mb-2">${rec.message}</div>`;
                });
            } else {
                html += '<p class="text-muted mb-0 small">No hay recomendaciones específicas en este momento</p>';
            }
            
            html += '</div></div>';
            recommendationsElement.innerHTML = html;
        })
        .catch(error => {
            recommendationsElement.innerHTML = `
                <div class="alert alert-warning">
                    <p class="mb-0 small">No se pudieron cargar las recomendaciones</p>
                </div>
            `;
        });
}

function updateScalpingAlerts() {
    const alertsElement = document.getElementById('scalping-alerts');
    
    if (!alertsElement) return;
    
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            let html = '';
            
            if (data.alerts && data.alerts.length > 0) {
                data.alerts.forEach(alert => {
                    html += `<div class="alert alert-${alert.type} small mb-2">${alert.message}</div>`;
                });
            } else {
                html = '<p class="text-muted text-center small">No hay alertas de scalping activas</p>';
            }
            
            alertsElement.innerHTML = html;
        })
        .catch(error => {
            alertsElement.innerHTML = '<p class="text-muted text-center small">Error cargando alertas</p>';
        });
}

function updateExitSignals() {
    const exitSignalsElement = document.getElementById('exit-signals');
    
    if (!exitSignalsElement) return;
    
    fetch('/api/exit_signals')
        .then(response => response.json())
        .then(data => {
            let html = '';
            
            if (data.signals && data.signals.length > 0) {
                data.signals.forEach(signal => {
                    html += `<div class="alert alert-${signal.type} small mb-2">${signal.message}</div>`;
                });
            } else {
                html = '<p class="text-muted text-center small">No hay señales de salida activas</p>';
            }
            
            exitSignalsElement.innerHTML = html;
        })
        .catch(error => {
            exitSignalsElement.innerHTML = '<p class="text-muted text-center small">Error cargando señales de salida</p>';
        });
}

function showError(message) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(container);
    }
    
    const toastId = 'toast-' + Date.now();
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
    
    document.getElementById('toast-container').innerHTML += toastHTML;
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

function downloadReport() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}`;
    window.open(url, '_blank');
    
    showNotification('📊 Generando reporte completo...', 'info');
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
            const now = new Date();
            document.getElementById('bolivia-clock').textContent = now.toLocaleTimeString('es-BO');
            document.getElementById('bolivia-date').textContent = now.toLocaleDateString('es-BO');
        });
}

// Inicializar el reloj
setInterval(updateBoliviaClock, 1000);
updateBoliviaClock();
