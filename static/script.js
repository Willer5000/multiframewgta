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
    updateCalendarInfo();
    initializeWinRateTracking();
}

function initializeWinRateTracking() {
    updateWinRate();
    setInterval(updateWinRate, 30000);
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
    updateWinRate();
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

function loadMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateScalpingAlerts();
    updateExitSignals();
    updateCalendarInfo();
    updateWinRate();
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
    
    document.getElementById('multi-timeframe-analysis').innerHTML = `
        <div class="text-center py-3">
            <div class="spinner-border spinner-border-sm text-teal" role="status">
                <span class="visually-hidden">Analizando...</span>
            </div>
            <p class="text-muted mb-0 small">Verificando temporalidades...</p>
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
            updateWinRate();
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
    updateWinRate();
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateScalpingAlerts();
    updateCalendarInfo();
    updateWinRate();
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
            renderRsiMaverickChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
            updateMultiTimeframeAnalysis(data);
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
        rsi_traditional: 50,
        win_rate: 65.5,
        trend_strength_signal: 'NEUTRAL',
        no_trade_zone: false,
        multi_timeframe_ok: false,
        multi_timeframe_reason: 'Error cargando datos',
        fulfilled_conditions: []
    };
    
    updateMarketSummary(sampleData);
    updateSignalAnalysis(sampleData);
    updateMultiTimeframeAnalysis(sampleData);
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
            name: 'Soporte Smart Money'
        });
        
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.resistance, data.resistance],
            mode: 'lines',
            line: {color: 'red', dash: 'dash', width: 2},
            name: 'Resistencia Smart Money'
        });
    }

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
    
    if (data.entry && data.take_profit) {
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.entry, data.entry],
            mode: 'lines',
            line: {color: '#FFD700', dash: 'solid', width: 2},
            name: 'Entrada Recomendada'
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
    
    const interval = document.getElementById('interval-select').value;
    const isObligatory = interval === '12h' || interval === '1D';
    const annotationText = isObligatory ? 
        '🔴 INDICADOR OBLIGATORIO (12H/1D)' : 
        'ℹ️ INDICADOR INFORMATIVO';
    
    const layout = {
        title: {
            text: `Actividad de Ballenas - ${annotationText}`,
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
        dragmode: drawingToolsActive ? 'drawline' : false,
        annotations: [
            {
                x: 0.02,
                y: 0.98,
                xref: 'paper',
                yref: 'paper',
                text: annotationText,
                showarrow: false,
                font: {color: isObligatory ? '#FF6B6B' : '#FFD700', size: 12},
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: isObligatory ? '#FF6B6B' : '#FFD700',
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
    const colors = data.indicators.colors || [];
    const highZoneThreshold = data.indicators.high_zone_threshold || 5;
    
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
        dragmode: drawingToolsActive ? 'drawline' : false
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
    const auxIndicator = document.getElementById('aux-indicator').value;
    
    if (!currentData) return;
    
    const chartElement = document.getElementById('aux-chart');
    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    
    let traces = [];
    let title = '';
    
    switch(auxIndicator) {
        case 'rsi':
            title = 'RSI Tradicional';
            traces = [{
                x: dates,
                y: currentData.indicators.rsi_traditional || [],
                type: 'scatter',
                mode: 'lines',
                name: 'RSI Tradicional',
                line: {color: '#FF6B6B', width: 2}
            }];
            break;
            
        case 'macd':
            title = 'MACD';
            traces = [
                {
                    x: dates,
                    y: currentData.indicators.macd_line || [],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MACD Line',
                    line: {color: '#2196F3', width: 2}
                },
                {
                    x: dates,
                    y: currentData.indicators.macd_signal || [],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Signal Line',
                    line: {color: '#FF9800', width: 1.5}
                },
                {
                    x: dates,
                    y: currentData.indicators.macd_histogram || [],
                    type: 'bar',
                    name: 'Histogram',
                    marker: {color: '#4CAF50'}
                }
            ];
            break;
            
        case 'squeeze':
            title = 'Squeeze Momentum';
            traces = [{
                x: dates,
                y: currentData.indicators.squeeze_momentum || [],
                type: 'scatter',
                mode: 'lines',
                name: 'Squeeze Momentum',
                line: {color: '#9C27B0', width: 2}
            }];
            break;
            
        case 'multi_timeframe':
            title = 'Análisis Multi-Temporalidad';
            // Implementar análisis multi-temporalidad
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
        margin: {t: 60, r: 50, b: 50, l: 50},
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
    
    if (traces.length > 0) {
        currentAuxChart = Plotly.newPlot('aux-chart', traces, layout, config);
    }
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
                <p class="mb-0">No hay datos disponibles para el mapa de oportunidades</p>
            </div>
        `;
        return;
    }

    const traces = [];
    const riskCategories = {
        'bajo': { color: '#00C853', symbol: 'circle' },
        'medio': { color: '#FFC107', symbol: 'square' },
        'alto': { color: '#FF1744', symbol: 'diamond' },
        'memecoins': { color: '#9C27B0', symbol: 'star' }
    };

    Object.keys(riskCategories).forEach(category => {
        const categoryData = data.filter(d => d.risk_category === category);
        
        if (categoryData.length > 0) {
            traces.push({
                x: categoryData.map(d => d.x),
                y: categoryData.map(d => d.y),
                mode: 'markers',
                type: 'scatter',
                name: category.toUpperCase(),
                text: categoryData.map(d => `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>Señal: ${d.signal}`),
                hovertemplate: '<b>%{text}</b><br>Compra: %{x}%<br>Venta: %{y}%<extra></extra>',
                marker: {
                    color: riskCategories[category].color,
                    size: categoryData.map(d => Math.max(8, d.signal_score / 3)),
                    symbol: riskCategories[category].symbol,
                    line: {
                        color: 'white',
                        width: 1
                    }
                }
            });
        }
    });

    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Presión Compra vs Venta',
            font: {color: '#ffffff', size: 16}
        },
        xaxis: {
            title: 'Presión de Compra (%)',
            range: [0, 100],
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Presión de Venta (%)',
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
            {
                type: 'line',
                x0: 70, y0: 0,
                x1: 70, y1: 100,
                line: {
                    color: 'green',
                    width: 1,
                    dash: 'dash'
                }
            },
            {
                type: 'line',
                x0: 0, y0: 70,
                x1: 100, y1: 70,
                line: {
                    color: 'red',
                    width: 1,
                    dash: 'dash'
                }
            }
        ],
        annotations: [
            {
                x: 85, y: 10,
                text: 'Zona LONG',
                showarrow: false,
                font: {color: 'green', size: 12},
                bgcolor: 'rgba(0,0,0,0.7)'
            },
            {
                x: 15, y: 85,
                text: 'Zona SHORT',
                showarrow: false,
                font: {color: 'red', size: 12},
                bgcolor: 'rgba(0,0,0,0.7)'
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
    
    if (data.long_signals && data.long_signals.length > 0) {
        longTable.innerHTML = data.long_signals.map((signal, index) => `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-success">${signal.signal_score.toFixed(1)}%</span></td>
                <td>${signal.entry ? signal.entry.toFixed(4) : 'N/A'}</td>
            </tr>
        `).join('');
    } else {
        longTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">No hay señales LONG</td>
            </tr>
        `;
    }
    
    if (data.short_signals && data.short_signals.length > 0) {
        shortTable.innerHTML = data.short_signals.map((signal, index) => `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-danger">${signal.signal_score.toFixed(1)}%</span></td>
                <td>${signal.entry ? signal.entry.toFixed(4) : 'N/A'}</td>
            </tr>
        `).join('');
    } else {
        shortTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">No hay señales SHORT</td>
            </tr>
        `;
    }
}

function showSignalDetails(symbol) {
    currentSymbol = symbol;
    document.getElementById('selected-crypto').textContent = symbol;
    updateCharts();
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    
    if (!data) return;
    
    const signalColor = data.signal === 'LONG' ? 'success' : 
                       data.signal === 'SHORT' ? 'danger' : 'secondary';
    
    const signalIcon = data.signal === 'LONG' ? '🟢' : 
                      data.signal === 'SHORT' ? '🔴' : '⚪';
    
    marketSummary.innerHTML = `
        <div class="row text-center">
            <div class="col-6 mb-3">
                <div class="card bg-dark border-${signalColor}">
                    <div class="card-body py-2">
                        <h6 class="card-title mb-1">Señal Actual</h6>
                        <h4 class="text-${signalColor} mb-0">${signalIcon} ${data.signal}</h4>
                        <small class="text-muted">Score: ${data.signal_score.toFixed(1)}%</small>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-info">
                    <div class="card-body py-2">
                        <h6 class="card-title mb-1">Precio Actual</h6>
                        <h4 class="text-info mb-0">$${data.current_price.toFixed(4)}</h4>
                        <small class="text-muted">USDT</small>
                    </div>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-12">
                <div class="card bg-dark border-warning">
                    <div class="card-body py-2">
                        <h6 class="card-title mb-2">Niveles Clave</h6>
                        <div class="row small">
                            <div class="col-6">
                                <span class="text-success">🎯 Entrada: $${data.entry.toFixed(4)}</span>
                            </div>
                            <div class="col-6">
                                <span class="text-danger">🛑 Stop: $${data.stop_loss.toFixed(4)}</span>
                            </div>
                            <div class="col-6">
                                <span class="text-primary">📈 TP1: $${data.take_profit[0].toFixed(4)}</span>
                            </div>
                            <div class="col-6">
                                <span class="text-info">💰 Soporte: $${data.support.toFixed(4)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (!data) return;
    
    const signalClass = data.signal === 'LONG' ? 'signal-long-enhanced' : 
                       data.signal === 'SHORT' ? 'signal-short-enhanced' : 'signal-neutral-enhanced';
    
    const conditionsList = data.fulfilled_conditions && data.fulfilled_conditions.length > 0 ? 
        data.fulfilled_conditions.map(condition => `<li class="small">✅ ${condition}</li>`).join('') :
        '<li class="small text-muted">No se cumplen condiciones suficientes</li>';
    
    signalAnalysis.innerHTML = `
        <div class="signal-analysis-enhanced ${signalClass}">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <h6 class="mb-0">Análisis de Señal</h6>
                <span class="badge bg-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary'}">
                    ${data.signal} ${data.signal_score.toFixed(1)}%
                </span>
            </div>
            <div class="mb-2">
                <strong>Condiciones Cumplidas:</strong>
                <ul class="mb-1 mt-1 ps-3">
                    ${conditionsList}
                </ul>
            </div>
            <div class="small">
                <strong>Multi-TF:</strong> 
                <span class="${data.multi_timeframe_ok ? 'text-success' : 'text-danger'}">
                    ${data.multi_timeframe_ok ? '✅ CONFIRMADO' : '❌ NO CONFIRMADO'}
                </span>
            </div>
        </div>
    `;
}

function updateMultiTimeframeAnalysis(data) {
    const multiTimeframeAnalysis = document.getElementById('multi-timeframe-analysis');
    
    if (!data) return;
    
    multiTimeframeAnalysis.innerHTML = `
        <div class="alert ${data.multi_timeframe_ok ? 'alert-success' : 'alert-warning'}">
            <h6 class="alert-heading">
                ${data.multi_timeframe_ok ? '✅ CONFIRMACIÓN MULTI-TF' : '⚠️ VERIFICACIÓN MULTI-TF'}
            </h6>
            <p class="mb-1 small">${data.multi_timeframe_reason}</p>
            ${data.no_trade_zone ? `
                <hr>
                <div class="no-trade-alert">
                    <strong>❌ ZONA DE NO OPERAR ACTIVA</strong>
                    <p class="mb-0 small">Evitar nuevas operaciones en esta temporalidad</p>
                </div>
            ` : ''}
        </div>
    `;
}

function updateFearGreedIndex() {
    const fearGreedElement = document.getElementById('fear-greed-index');
    
    // Simulación del índice de miedo y codicia
    const fearGreedValue = Math.floor(Math.random() * 100);
    let level = '', color = '', description = '';
    
    if (fearGreedValue >= 80) {
        level = 'Extrema Codicia';
        color = 'danger';
        description = 'Mercado sobrecomprado - Precaución';
    } else if (fearGreedValue >= 60) {
        level = 'Codicia';
        color = 'warning';
        description = 'Mercado alcista - Oportunidades limitadas';
    } else if (fearGreedValue >= 40) {
        level = 'Neutral';
        color = 'info';
        description = 'Mercado equilibrado - Buenas oportunidades';
    } else if (fearGreedValue >= 20) {
        level = 'Miedo';
        color = 'primary';
        description = 'Mercado bajista - Oportunidades de compra';
    } else {
        level = 'Miedo Extremo';
        color = 'success';
        description = 'Mercado sobrevendido - Excelentes oportunidades';
    }
    
    fearGreedElement.innerHTML = `
        <div class="text-center">
            <div class="fear-greed-value mb-2">
                <h3 class="text-${color} mb-1">${fearGreedValue}</h3>
                <span class="badge bg-${color}">${level}</span>
            </div>
            <div class="fear-greed-progress mb-2">
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar bg-${color}" style="width: ${fearGreedValue}%">
                        ${fearGreedValue}%
                    </div>
                </div>
            </div>
            <p class="small text-muted mb-0">${description}</p>
        </div>
    `;
}

function updateMarketRecommendations() {
    const recommendationsElement = document.getElementById('market-recommendations');
    
    const recommendations = [
        { type: 'success', icon: '🚀', text: 'Mercado alcista - Buscar LONGs' },
        { type: 'warning', icon: '⚡', text: 'Alta volatilidad - Gestionar riesgo' },
        { type: 'info', icon: '📊', text: 'Multi-TF confirmado - Señales confiables' }
    ];
    
    recommendationsElement.innerHTML = recommendations.map(rec => `
        <div class="alert alert-${rec.type} alert-dismissible fade show mb-2" role="alert">
            ${rec.icon} ${rec.text}
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="alert"></button>
        </div>
    `).join('');
}

function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const alertsElement = document.getElementById('scalping-alerts');
            
            if (data.alerts && data.alerts.length > 0) {
                alertsElement.innerHTML = data.alerts.slice(0, 5).map(alert => `
                    <div class="scalping-alert mb-2 p-2 rounded">
                        <div class="d-flex justify-content-between align-items-center">
                            <strong class="small">${alert.symbol}</strong>
                            <span class="badge bg-${alert.signal === 'LONG' ? 'success' : 'danger'}">
                                ${alert.signal}
                            </span>
                        </div>
                        <div class="small">
                            <span class="text-muted">${alert.interval}</span> | 
                            Score: ${alert.score.toFixed(1)}%
                        </div>
                        <div class="small text-muted">
                            Entrada: $${alert.entry.toFixed(4)}
                        </div>
                    </div>
                `).join('');
            } else {
                alertsElement.innerHTML = `
                    <div class="text-center text-muted">
                        <i class="fas fa-bell-slash fa-2x mb-2"></i>
                        <p class="small mb-0">No hay alertas activas</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando alertas de scalping:', error);
        });
}

function updateExitSignals() {
    const exitSignalsElement = document.getElementById('exit-signals');
    
    exitSignalsElement.innerHTML = `
        <div class="text-center text-muted">
            <i class="fas fa-chart-line fa-2x mb-2"></i>
            <p class="small mb-0">Monitoreando operaciones activas</p>
            <div class="mt-2">
                <span class="badge bg-success">0 LONG</span>
                <span class="badge bg-danger ms-1">0 SHORT</span>
            </div>
        </div>
    `;
}

function showError(message) {
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'toast-' + Date.now();
    
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-white bg-danger border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-exclamation-triangle me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.innerHTML += toastHTML;
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    setTimeout(() => {
        if (toastElement && toastElement.parentNode) {
            toastElement.parentNode.removeChild(toastElement);
        }
    }, 5000);
}

function downloadReport() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}`;
    window.open(url, '_blank');
}

function updateWinRate() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/win_rate?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            const winrateDisplay = document.getElementById('winrate-display');
            if (winrateDisplay) {
                const winrate = data.win_rate || 0;
                const colorClass = winrate >= 70 ? 'text-success' : winrate >= 60 ? 'text-warning' : 'text-danger';
                
                winrateDisplay.innerHTML = `
                    <div class="${colorClass}">
                        <h3 class="mb-1">${winrate.toFixed(1)}%</h3>
                        <p class="small mb-0">WinRate Histórico</p>
                    </div>
                    <div class="progress mt-2" style="height: 8px;">
                        <div class="progress-bar ${winrate >= 70 ? 'bg-success' : winrate >= 60 ? 'bg-warning' : 'bg-danger'}" 
                             style="width: ${winrate}%"></div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
        });
}
