// Configuración global
let currentChart = null;
let currentScatterChart = null;
let currentWhaleChart = null;
let currentAdxChart = null;
let currentRsiComparisonChart = null;
let currentMacdChart = null;
let currentSqueezeChart = null;
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
    updateBoliviaClock();
    updateWinrate();
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
    
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-comparison-chart', 'macd-chart', 'squeeze-chart', 'aux-chart', 'trend-strength-chart'];
    
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
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-comparison-chart', 'macd-chart', 'squeeze-chart', 'aux-chart', 'trend-strength-chart'];
    
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
    updateFearGreedIndex();
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
            renderSqueezeChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
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
        trend_strength_signal: 'NEUTRAL',
        no_trade_zone: false,
        obligatory_conditions_met: false,
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
    const bullishDivergence = data.indicators.rsi_maverick_bullish_div || [];
    const bearishDivergence = data.indicators.rsi_maverick_bearish_div || [];
    
    const bullishDates = [];
    const bullishValues = [];
    const bearishDates = [];
    const bearishValues = [];
    
    for (let i = 7; i < bullishDivergence.length; i++) {
        if (bullishDivergence[i] && !bullishDivergence[i-1] && !bullishDivergence[i-2]) {
            bullishDates.push(dates[i]);
            bullishValues.push((rsiMaverick[i] * 100) || 50);
        }
        if (bearishDivergence[i] && !bearishDivergence[i-1] && !bearishDivergence[i-2]) {
            bearishDates.push(dates[i]);
            bearishValues.push((rsiMaverick[i] * 100) || 50);
        }
    }
    
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
            y: rsiMaverick.map(x => x * 100),
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B × 100)',
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
                symbol: 'triangle-up',
                line: {color: 'white', width: 1}
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
                symbol: 'triangle-down',
                line: {color: 'white', width: 1}
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'Comparación RSI Tradicional vs RSI Maverick',
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
        annotations: [
            {
                x: dates[dates.length - 1],
                y: 70,
                xanchor: 'left',
                text: 'Sobrecompra',
                showarrow: false,
                font: {color: 'red', size: 10}
            },
            {
                x: dates[dates.length - 1],
                y: 30,
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
    const macdBullish = data.indicators.macd_bullish || [];
    const macdBearish = data.indicators.macd_bearish || [];
    
    const histogramColors = macdHistogram.map(value => value >= 0 ? '#00C853' : '#FF1744');
    
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
            line: {color: '#FF9800', width: 1}
        },
        {
            x: dates,
            y: macdHistogram,
            type: 'bar',
            name: 'Histograma',
            marker: {color: histogramColors}
        },
        {
            x: dates.filter((_, i) => macdBullish[i]),
            y: macdLine.filter((_, i) => macdBullish[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Cruce Alcista',
            marker: {color: '#00FF00', size: 8, symbol: 'triangle-up'}
        },
        {
            x: dates.filter((_, i) => macdBearish[i]),
            y: macdLine.filter((_, i) => macdBearish[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Cruce Bajista',
            marker: {color: '#FF0000', size: 8, symbol: 'triangle-down'}
        }
    ];
    
    const layout = {
        title: {
            text: 'MACD con Histograma Coloreado',
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
                text: '🟢 Verde: Histograma positivo | 🔴 Rojo: Histograma negativo',
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
                <p class="mb-0">No hay datos de Squeeze Momentum disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const squeezeOn = data.indicators.squeeze_on || [];
    const squeezeOff = data.indicators.squeeze_off || [];
    const squeezeMomentum = data.indicators.squeeze_momentum || [];
    const squeezeColors = data.indicators.squeeze_colors || [];
    
    const traces = [
        {
            x: dates,
            y: squeezeMomentum,
            type: 'bar',
            name: 'Squeeze Momentum',
            marker: {color: squeezeColors}
        }
    ];
    
    const layout = {
        title: {
            text: 'Squeeze Momentum - Compresión/Expansión del Mercado',
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
    const trendStrength = data.indicators.trend_strength || [];
    const bbWidth = data.indicators.bb_width || [];
    const noTradeZones = data.indicators.no_trade_zones || [];
    const colors = data.indicators.colors || [];
    const highZoneThreshold = data.indicators.high_zone_threshold || 5;
    
    const traces = [
        {
            x: dates,
            y: trendStrength,
            type: 'bar',
            name: 'Fuerza de Tendencia',
            marker: {color: colors}
        }
    ];
    
    traces.push({
        x: [dates[0], dates[dates.length - 1]],
        y: [highZoneThreshold, highZoneThreshold],
        type: 'scatter',
        mode: 'lines',
        name: 'Umbral Alto',
        line: {color: 'orange', dash: 'dash', width: 1}
    });
    
    traces.push({
        x: [dates[0], dates[dates.length - 1]],
        y: [-highZoneThreshold, -highZoneThreshold],
        type: 'scatter',
        mode: 'lines',
        name: 'Umbral Bajo',
        line: {color: 'orange', dash: 'dash', width: 1},
        showlegend: false
    });
    
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

function renderScatterChartImproved(scatterData) {
    const chartElement = document.getElementById('scatter-chart');
    
    if (!scatterData || scatterData.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
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
                text: categoryData.map(d => d.symbol),
                mode: 'markers',
                type: 'scatter',
                name: category.toUpperCase(),
                marker: {
                    size: categoryData.map(d => Math.max(10, d.signal_score / 3)),
                    color: colors[index],
                    symbol: symbols[index],
                    line: {width: 1, color: 'white'}
                },
                hovertemplate: '<b>%{text}</b><br>Presión Compra: %{x}<br>Presión Venta: %{y}<br>Score: %{marker.size}<extra></extra>'
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
            font: {color: '#ffffff'},
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: {t: 80, r: 50, b: 50, l: 50},
        shapes: [
            {
                type: 'line',
                x0: 50,
                y0: 0,
                x1: 50,
                y1: 100,
                line: {color: 'white', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: 0,
                y0: 50,
                x1: 100,
                y1: 50,
                line: {color: 'white', width: 1, dash: 'dash'}
            }
        ],
        annotations: [
            {
                x: 75,
                y: 25,
                xref: 'x',
                yref: 'y',
                text: 'Zona LONG',
                showarrow: false,
                font: {color: 'green', size: 14}
            },
            {
                x: 25,
                y: 75,
                xref: 'x',
                yref: 'y',
                text: 'Zona SHORT',
                showarrow: false,
                font: {color: 'red', size: 14}
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
    fetch(`/api/multiple_signals?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`)
        .then(response => response.json())
        .then(signalsData => {
            updateSignalsTable(signalsData);
        })
        .catch(error => {
            console.error('Error actualizando múltiples señales:', error);
        });
}

function updateSignalsTable(signalsData) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');

    if (signalsData.long_signals && signalsData.long_signals.length > 0) {
        longTable.innerHTML = signalsData.long_signals.map((signal, index) => `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}', '${signal.interval}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-success">${signal.signal_score.toFixed(1)}%</span></td>
                <td>${signal.entry.toFixed(6)}</td>
            </tr>
        `).join('');
    } else {
        longTable.innerHTML = '<tr><td colspan="4" class="text-center py-3">No hay señales LONG</td></tr>';
    }

    if (signalsData.short_signals && signalsData.short_signals.length > 0) {
        shortTable.innerHTML = signalsData.short_signals.map((signal, index) => `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}', '${signal.interval}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-danger">${signal.signal_score.toFixed(1)}%</span></td>
                <td>${signal.entry.toFixed(6)}</td>
            </tr>
        `).join('');
    } else {
        shortTable.innerHTML = '<tr><td colspan="4" class="text-center py-3">No hay señales SHORT</td></tr>';
    }
}

function showSignalDetails(symbol, interval) {
    currentSymbol = symbol;
    document.getElementById('selected-crypto').textContent = symbol;
    updateCharts();
}

function updateAuxChart() {
    const auxIndicator = document.getElementById('aux-indicator').value;
    console.log('Indicador auxiliar seleccionado:', auxIndicator);
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    
    if (!data) {
        marketSummary.innerHTML = `
            <div class="alert alert-danger text-center">
                <p class="mb-0">Error cargando resumen del mercado</p>
            </div>
        `;
        return;
    }

    const signalClass = data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary';
    const signalIcon = data.signal === 'LONG' ? 'fa-arrow-up' : data.signal === 'SHORT' ? 'fa-arrow-down' : 'fa-pause';
    
    marketSummary.innerHTML = `
        <div class="row">
            <div class="col-12">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="mb-0">${data.symbol}</h5>
                    <span class="badge bg-${signalClass}"><i class="fas ${signalIcon} me-1"></i>${data.signal}</span>
                </div>
                <div class="mb-2">
                    <strong>Precio Actual:</strong> $${data.current_price.toFixed(6)}
                </div>
                <div class="mb-2">
                    <strong>Score Señal:</strong> 
                    <span class="badge bg-${data.signal_score >= 70 ? 'success' : data.signal_score >= 50 ? 'warning' : 'danger'}">
                        ${data.signal_score.toFixed(1)}%
                    </span>
                </div>
                <div class="mb-2">
                    <strong>Volumen (24h):</strong> ${(data.volume / 1000000).toFixed(2)}M
                </div>
                <div class="mb-2">
                    <strong>ADX:</strong> ${data.adx.toFixed(2)}
                </div>
                <div class="mb-2">
                    <strong>+DI:</strong> ${data.plus_di.toFixed(2)} | <strong>-DI:</strong> ${data.minus_di.toFixed(2)}
                </div>
                <div class="mb-2">
                    <strong>RSI Maverick:</strong> ${(data.rsi_maverick * 100).toFixed(2)}%
                </div>
                <div class="mb-2">
                    <strong>Ballenas Compradoras:</strong> ${data.whale_pump.toFixed(2)}
                </div>
                <div class="mb-2">
                    <strong>Ballenas Vendedoras:</strong> ${data.whale_dump.toFixed(2)}
                </div>
                <div class="mb-2">
                    <strong>Fuerza Tendencia:</strong> 
                    <span class="trend-strength-indicator badge-${data.trend_strength_signal.toLowerCase().replace('_', '-')}">
                        ${data.trend_strength_signal}
                    </span>
                </div>
                <div class="mb-2">
                    <strong>Zona NO OPERAR:</strong> 
                    <span class="badge ${data.no_trade_zone ? 'bg-danger' : 'bg-success'}">
                        ${data.no_trade_zone ? 'ACTIVA' : 'INACTIVA'}
                    </span>
                </div>
                <div class="mb-2">
                    <strong>Condiciones Obligatorias:</strong> 
                    <span class="badge ${data.obligatory_conditions_met ? 'bg-success' : 'bg-danger'}">
                        ${data.obligatory_conditions_met ? 'CUMPLIDAS' : 'NO CUMPLIDAS'}
                    </span>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (!data) {
        signalAnalysis.innerHTML = `
            <div class="alert alert-danger text-center">
                <p class="mb-0">Error en el análisis de señal</p>
            </div>
        `;
        return;
    }

    let conditionsHtml = '';
    if (data.fulfilled_conditions && data.fulfilled_conditions.length > 0) {
        conditionsHtml = `
            <div class="mt-3">
                <strong>Condiciones Cumplidas:</strong>
                <ul class="mt-2 small">
                    ${data.fulfilled_conditions.map(condition => `<li>${condition}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    signalAnalysis.innerHTML = `
        <div class="row">
            <div class="col-12">
                <div class="mb-3">
                    <strong>Entrada:</strong> $${data.entry.toFixed(6)}
                </div>
                <div class="mb-3">
                    <strong>Stop Loss:</strong> $${data.stop_loss.toFixed(6)}
                </div>
                <div class="mb-3">
                    <strong>Take Profit:</strong> 
                    <ul class="mt-1 small">
                        ${data.take_profit.map((tp, index) => `<li>TP${index + 1}: $${tp.toFixed(6)}</li>`).join('')}
                    </ul>
                </div>
                <div class="mb-3">
                    <strong>Soporte:</strong> $${data.support.toFixed(6)}
                </div>
                <div class="mb-3">
                    <strong>Resistencia:</strong> $${data.resistance.toFixed(6)}
                </div>
                <div class="mb-3">
                    <strong>ATR:</strong> ${data.atr.toFixed(6)} (${(data.atr_percentage * 100).toFixed(2)}%)
                </div>
                ${conditionsHtml}
            </div>
        </div>
    `;
}

function updateFearGreedIndex() {
    const fearGreedIndex = document.getElementById('fear-greed-index');
    const randomIndex = Math.floor(Math.random() * 100);
    
    let level, color, description;
    if (randomIndex >= 80) {
        level = 'Extrema Codicia';
        color = 'danger';
        description = 'Mercado sobrecomprado, posible corrección';
    } else if (randomIndex >= 60) {
        level = 'Codicia';
        color = 'warning';
        description = 'Mercado alcista, cuidado con entradas';
    } else if (randomIndex >= 40) {
        level = 'Neutral';
        color = 'secondary';
        description = 'Mercado en equilibrio';
    } else if (randomIndex >= 20) {
        level = 'Miedo';
        color = 'info';
        description = 'Mercado bajista, oportunidades de compra';
    } else {
        level = 'Miedo Extremo';
        color = 'success';
        description = 'Mercado sobrevendido, posible rebote';
    }
    
    fearGreedIndex.innerHTML = `
        <div class="text-center">
            <div class="display-4 fw-bold text-${color}">${randomIndex}</div>
            <div class="mt-2">
                <span class="badge bg-${color}">${level}</span>
            </div>
            <p class="small text-muted mt-2">${description}</p>
        </div>
    `;
}

function updateMarketRecommendations() {
    const marketRecommendations = document.getElementById('market-recommendations');
    
    const recommendations = [
        { type: 'success', text: 'Mercado alcista en BTC y ETH' },
        { type: 'warning', text: 'Posible corrección en memecoins' },
        { type: 'info', text: 'Alta volatilidad en altcoins' }
    ];
    
    marketRecommendations.innerHTML = `
        <div class="card bg-dark border-secondary">
            <div class="card-header">
                <h6 class="mb-0"><i class="fas fa-bullhorn me-2"></i>Recomendaciones</h6>
            </div>
            <div class="card-body">
                ${recommendations.map(rec => `
                    <div class="alert alert-${rec.type} py-2 mb-2 small" role="alert">
                        <i class="fas fa-${rec.type === 'success' ? 'check' : rec.type === 'warning' ? 'exclamation-triangle' : 'info-circle'} me-1"></i>
                        ${rec.text}
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const scalpingAlerts = document.getElementById('scalping-alerts');
            
            if (!data.alerts || data.alerts.length === 0) {
                scalpingAlerts.innerHTML = `
                    <div class="text-center py-2">
                        <p class="text-muted mb-0">No hay alertas activas</p>
                    </div>
                `;
                return;
            }
            
            scalpingAlerts.innerHTML = data.alerts.map(alert => `
                <div class="alert alert-warning py-2 mb-2 small scalping-alert" role="alert">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <strong>${alert.symbol}</strong> (${alert.interval})<br>
                            <span class="badge bg-${alert.signal === 'LONG' ? 'success' : 'danger'}">${alert.signal}</span>
                            Score: ${alert.score.toFixed(1)}%<br>
                            Entrada: $${alert.entry.toFixed(6)}<br>
                            SL: $${alert.stop_loss.toFixed(6)} | TP: $${alert.take_profit.toFixed(6)}
                        </div>
                        <small class="text-muted">${alert.timestamp}</small>
                    </div>
                </div>
            `).join('');
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
            
            if (!data.exit_signals || data.exit_signals.length === 0) {
                exitSignals.innerHTML = `
                    <div class="text-center py-2">
                        <p class="text-muted mb-0">No hay señales de salida</p>
                    </div>
                `;
                return;
            }
            
            exitSignals.innerHTML = data.exit_signals.map(signal => `
                <div class="alert alert-danger py-2 mb-2 small" role="alert">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <strong>${signal.symbol}</strong> (${signal.interval})<br>
                            <span class="badge bg-${signal.signal === 'LONG' ? 'success' : 'danger'}">${signal.signal}</span>
                            P&L: <span class="text-${signal.pnl_percent >= 0 ? 'success' : 'danger'}">${signal.pnl_percent.toFixed(2)}%</span><br>
                            Razón: ${signal.reason}
                        </div>
                        <small class="text-muted">${signal.timestamp}</small>
                    </div>
                </div>
            `).join('');
        })
        .catch(error => {
            console.error('Error actualizando señales de salida:', error);
        });
}

function showError(message) {
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'toast-' + Date.now();
    
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-danger border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-exclamation-circle me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

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

function updateWinrate() {
    fetch('/api/winrate')
        .then(response => response.json())
        .then(data => {
            const winrateDisplay = document.getElementById('winrate-display');
            if (data.winrate) {
                winrateDisplay.innerHTML = `
                    <div class="winrate-value display-4 fw-bold text-success">${data.winrate}%</div>
                    <div class="winrate-label small text-muted">Tasa de Acierto</div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
        });
}

function downloadStrategicReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}&strategic=true`;
    window.open(url, '_blank');
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

// Actualizar el reloj cada segundo
setInterval(updateBoliviaClock, 1000);
// Actualizar winrate cada 2 minutos
setInterval(updateWinrate, 120000);
