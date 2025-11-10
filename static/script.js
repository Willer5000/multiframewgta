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
    updateWinRate();
}

function setupEventListeners() {
    // Configurar event listeners para los controles
    document.getElementById('interval-select').addEventListener('change', updateCharts);
    document.getElementById('di-period').addEventListener('change', updateCharts);
    document.getElementById('adx-threshold').addEventListener('change', updateCharts);
    document.getElementById('sr-period').addEventListener('change', updateCharts);
    document.getElementById('rsi-length').addEventListener('change', updateCharts);
    document.getElementById('bb-multiplier').addEventListener('change', updateCharts);
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

function updateWinRate() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/win_rate?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            const winRateDisplay = document.getElementById('win-rate-display');
            if (winRateDisplay) {
                winRateDisplay.innerHTML = `
                    <h3 class="text-success mb-1">${data.win_rate}%</h3>
                    <p class="small mb-0">
                        ${data.successful_trades}/${data.total_trades} operaciones<br>
                        <small class="text-muted">WinRate estratégico</small>
                    </p>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
            const winRateDisplay = document.getElementById('win-rate-display');
            if (winRateDisplay) {
                winRateDisplay.innerHTML = `
                    <h3 class="text-warning mb-1">N/A</h3>
                    <p class="small mb-0 text-muted">WinRate no disponible</p>
                `;
            }
        });
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
    const cryptoList = document.getElementById('crypto-list');
    
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const filter = this.value.toUpperCase();
            filterCryptoList(filter);
        });
        
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
            let dragmode = false;
            switch(tool) {
                case 'line': dragmode = 'drawline'; break;
                case 'rectangle': dragmode = 'drawrect'; break;
                case 'circle': dragmode = 'drawcircle'; break;
                case 'text': dragmode = 'drawtext'; break;
                case 'freehand': dragmode = 'drawfreehand'; break;
                case 'marker': dragmode = 'marker'; break;
            }
            
            if (currentChart) {
                Plotly.relayout(chartId, {dragmode: dragmode});
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
    updateWinRate();
}

function loadCryptoRiskClassification() {
    fetch('/api/crypto_risk_classification')
        .then(response => response.json())
        .then(riskData => {
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
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT', 'XRP-USDT',
        'ADA-USDT', 'AVAX-USDT', 'DOT-USDT', 'LINK-USDT', 'MATIC-USDT'
    ];
    
    allCryptos = basicSymbols.map(symbol => ({
        symbol: symbol,
        category: 'bajo'
    }));
    
    filterCryptoList('');
}

function loadMarketIndicators() {
    updateCalendarInfo();
    updateMultiTFConditions();
    updateScalpingAlerts();
    updateExitSignals();
}

function showLoadingState() {
    const marketSummary = document.getElementById('market-summary');
    const signalAnalysis = document.getElementById('signal-analysis');
    const multiTfConditions = document.getElementById('multi-tf-conditions');
    
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
    
    if (multiTfConditions) {
        multiTfConditions.innerHTML = `
            <div class="text-center py-2">
                <div class="spinner-border spinner-border-sm text-info" role="status">
                    <span class="visually-hidden">Analizando...</span>
                </div>
                <p class="mt-2 mb-0 small">Verificando temporalidades...</p>
            </div>
        `;
    }
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
    const leverage = document.getElementById('leverage').value;
    
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    updateScatterChart(interval, diPeriod, adxThreshold);
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    updateAuxChart();
    updateMultiTFConditions();
}

function updateMarketIndicators() {
    updateCalendarInfo();
    updateMultiTFConditions();
    updateScalpingAlerts();
    updateExitSignals();
}

function updateMultiTFConditions() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/signals?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            updateMultiTFConditionsDisplay(data);
        })
        .catch(error => {
            console.error('Error actualizando condiciones multi-TF:', error);
        });
}

function updateMultiTFConditionsDisplay(data) {
    const multiTfElement = document.getElementById('multi-tf-conditions');
    if (!multiTfElement) return;

    const isObligatoryOk = data.obligatory_conditions_ok || false;
    const trendStrength = data.trend_strength_signal || 'NEUTRAL';
    const noTradeZone = data.no_trade_zone || false;

    let conditionsHTML = '';

    if (isObligatoryOk) {
        conditionsHTML = `
            <div class="text-center text-success mb-2">
                <i class="fas fa-check-circle fa-2x mb-2"></i>
                <h6>CONDICIONES OBLIGATORIAS</h6>
                <p class="small mb-1">✅ Multi-Timeframe Confirmado</p>
                <p class="small mb-1">✅ Tendencia Alineada</p>
                <p class="small mb-0">✅ Sin Zonas No Operar</p>
            </div>
        `;
    } else {
        conditionsHTML = `
            <div class="text-center text-danger mb-2">
                <i class="fas fa-times-circle fa-2x mb-2"></i>
                <h6>CONDICIONES NO CUMPLIDAS</h6>
                <p class="small mb-1">❌ Filtro Multi-Timeframe</p>
                <p class="small mb-1">❌ Tendencia Desalineada</p>
                <p class="small mb-0">❌ Zona No Operar Activa</p>
            </div>
        `;
    }

    conditionsHTML += `
        <div class="border-top pt-2">
            <div class="d-flex justify-content-between small">
                <span>Fuerza Tendencia:</span>
                <span class="text-${getTrendStrengthColor(trendStrength)}">${trendStrength}</span>
            </div>
            <div class="d-flex justify-content-between small">
                <span>Zona No Operar:</span>
                <span class="text-${noTradeZone ? 'danger' : 'success'}">${noTradeZone ? 'ACTIVA' : 'INACTIVA'}</span>
            </div>
        </div>
    `;

    multiTfElement.innerHTML = conditionsHTML;
}

function getTrendStrengthColor(signal) {
    switch(signal) {
        case 'STRONG_UP': return 'success';
        case 'WEAK_UP': return 'info';
        case 'STRONG_DOWN': return 'danger';
        case 'WEAK_DOWN': return 'warning';
        default: return 'muted';
    }
}

function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage) {
    const url = `/api/signals?symbol=${symbol}&interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            currentData = data;
            renderCandleChart(data);
            renderWhaleChart(data);
            renderAdxChart(data);
            renderRsiMaverickChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
            updateMultiTFConditionsDisplay(data);
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
        trend_strength_signal: 'NEUTRAL',
        no_trade_zone: false,
        obligatory_conditions_ok: false,
        fulfilled_conditions: []
    };
    
    updateMarketSummary(sampleData);
    updateSignalAnalysis(sampleData);
    updateMultiTFConditionsDisplay(sampleData);
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
            text: 'Indicador Cazador de Ballenas',
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
    const colors = data.indicators.colors || [];
    const noTradeZones = data.indicators.no_trade_zones || [];
    
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
            const rsiTraditional = currentData.indicators.rsi_traditional || [];
            traces = [{
                x: dates,
                y: rsiTraditional,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI Tradicional',
                line: {color: '#FF6B6B', width: 2}
            }];
            
            layout = {
                title: {text: 'RSI Tradicional', font: {color: '#ffffff', size: 14}},
                xaxis: {title: 'Fecha/Hora', type: 'date', gridcolor: '#444'},
                yaxis: {title: 'RSI', range: [0, 100], gridcolor: '#444'},
                shapes: [
                    {type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 70, y1: 70, line: {color: 'red', dash: 'dash'}},
                    {type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 30, y1: 30, line: {color: 'green', dash: 'dash'}}
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
            break;
            
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
                    line: {color: '#FF6B6B', width: 2}
                },
                {
                    x: dates,
                    y: macdSignal,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Señal',
                    line: {color: '#4ECDC4', width: 1}
                },
                {
                    x: dates,
                    y: macdHistogram,
                    type: 'bar',
                    name: 'Histograma',
                    marker: {color: macdHistogram.map(val => val >= 0 ? '#00C853' : '#FF1744')}
                }
            ];
            
            layout = {
                title: {text: 'MACD', font: {color: '#ffffff', size: 14}},
                xaxis: {title: 'Fecha/Hora', type: 'date', gridcolor: '#444'},
                yaxis: {title: 'MACD', gridcolor: '#444'},
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
            break;
            
        case 'squeeze':
            const squeezeOn = currentData.indicators.squeeze_on || [];
            const squeezeOff = currentData.indicators.squeeze_off || [];
            const momentum = currentData.indicators.squeeze_momentum || [];
            
            const squeezeOnX = [];
            const squeezeOnY = [];
            const squeezeOffX = [];
            const squeezeOffY = [];
            
            dates.forEach((date, i) => {
                if (squeezeOn[i]) {
                    squeezeOnX.push(date);
                    squeezeOnY.push(momentum[i]);
                } else if (squeezeOff[i]) {
                    squeezeOffX.push(date);
                    squeezeOffY.push(momentum[i]);
                }
            });
            
            traces = [
                {
                    x: dates,
                    y: momentum,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Momentum',
                    line: {color: '#666', width: 1}
                },
                {
                    x: squeezeOnX,
                    y: squeezeOnY,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Squeeze ON',
                    marker: {color: 'red', size: 8, symbol: 'diamond'}
                },
                {
                    x: squeezeOffX,
                    y: squeezeOffY,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Squeeze OFF',
                    marker: {color: 'green', size: 8, symbol: 'diamond'}
                }
            ];
            
            layout = {
                title: {text: 'Squeeze Momentum', font: {color: '#ffffff', size: 14}},
                xaxis: {title: 'Fecha/Hora', type: 'date', gridcolor: '#444'},
                yaxis: {title: 'Momentum', gridcolor: '#444'},
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
            break;
            
        case 'multi_tf':
            traces = [{
                x: ['Mayor', 'Media', 'Menor'],
                y: [1, 1, 1],
                type: 'bar',
                name: 'Temporalidades',
                marker: {color: ['#00C853', '#FFC107', '#2196F3']}
            }];
            
            layout = {
                title: {text: 'Análisis Multi-Timeframe', font: {color: '#ffffff', size: 14}},
                xaxis: {title: 'Temporalidad', gridcolor: '#444'},
                yaxis: {title: 'Estado', range: [0, 1.2], gridcolor: '#444'},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#ffffff'},
                showlegend: false,
                margin: {t: 60, r: 50, b: 50, l: 50},
                dragmode: drawingToolsActive ? 'drawline' : false,
                annotations: [
                    {
                        x: 'Mayor',
                        y: 1.1,
                        text: '✅ Confirmada',
                        showarrow: false,
                        font: {color: 'white', size: 10}
                    },
                    {
                        x: 'Media',
                        y: 1.1,
                        text: '✅ Confirmada',
                        showarrow: false,
                        font: {color: 'white', size: 10}
                    },
                    {
                        x: 'Menor',
                        y: 1.1,
                        text: '✅ Confirmada',
                        showarrow: false,
                        font: {color: 'white', size: 10}
                    }
                ]
            };
            break;
    }
    
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
    
    const volumeLevel = getVolumeLevel(data.volume, data.volume_ma);
    const adxStrength = getADXStrength(data.adx);
    const signalStrength = getSignalStrength(data);
    
    const trendStrengthInfo = data.trend_strength_signal ? 
        `<div class="d-flex justify-content-between">
            <span>Fuerza Tendencia:</span>
            <span class="text-${getTrendStrengthColor(data.trend_strength_signal)}">${data.trend_strength_signal}</span>
        </div>` : '';
    
    const obligatoryConditions = data.obligatory_conditions_ok ?
        `<div class="alert alert-success mt-2 py-1 text-center">
            <small><i class="fas fa-check-circle me-1"></i>CONDICIONES OBLIGATORIAS CUMPLIDAS</small>
        </div>` :
        `<div class="alert alert-danger mt-2 py-1 text-center">
            <small><i class="fas fa-times-circle me-1"></i>CONDICIONES OBLIGATORIAS NO CUMPLIDAS</small>
        </div>`;
    
    const noTradeWarning = data.no_trade_zone ? 
        `<div class="alert alert-danger mt-2 py-1 text-center">
            <small><i class="fas fa-exclamation-triangle me-1"></i>ZONA DE NO OPERAR - Evitar entradas</small>
        </div>` : '';
    
    const summaryHTML = `
        <div class="fade-in">
            <div class="row text-center mb-4">
                <div class="col-6">
                    <div class="card bg-dark border-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Señal Actual</small>
                            <h4 class="mb-0 text-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'muted'}">
                                ${data.signal}
                            </h4>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card bg-dark border-${signalStrength.color}">
                        <div class="card-body py-2">
                            <small class="text-muted">Score Señal</small>
                            <h4 class="mb-0 text-${signalStrength.color}">
                                ${data.signal_score.toFixed(0)}%
                            </h4>
                        </div>
                    </div>
                </div>
            </div>
            
            ${obligatoryConditions}
            ${noTradeWarning}
            
            <div class="mb-3">
                <h6><i class="fas fa-dollar-sign me-2"></i>Precio Actual</h6>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="fs-5 fw-bold">$${formatPriceForDisplay(data.current_price)}</span>
                    <small class="text-muted">USDT</small>
                </div>
            </div>
            
            <div class="mb-3">
                <h6><i class="fas fa-shield-alt me-2"></i>Niveles Clave</h6>
                <div class="d-flex justify-content-between">
                    <span>Soporte:</span>
                    <span class="text-info">$${formatPriceForDisplay(data.support)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Resistencia:</span>
                    <span class="text-warning">$${formatPriceForDisplay(data.resistance)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>ATR:</span>
                    <span class="text-muted">${(data.atr_percentage * 100).toFixed(2)}%</span>
                </div>
            </div>
            
            <div class="mb-3">
                <h6><i class="fas fa-balance-scale me-2"></i>Liquidaciones</h6>
                <div class="d-flex justify-content-between">
                    <span>LONG:</span>
                    <span class="text-danger">$${formatPriceForDisplay(data.liquidation_long)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>SHORT:</span>
                    <span class="text-danger">$${formatPriceForDisplay(data.liquidation_short)}</span>
                </div>
            </div>
            
            <div class="mb-3">
                <h6><i class="fas fa-chart-bar me-2"></i>Volumen</h6>
                <div class="d-flex justify-content-between">
                    <span>Actual:</span>
                    <span class="text-${volumeLevel.color}">${(data.volume / 1e6).toFixed(2)}M</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Promedio:</span>
                    <span class="text-muted">${(data.volume_ma / 1e6).toFixed(2)}M</span>
                </div>
                <small class="text-${volumeLevel.color}">${volumeLevel.text}</small>
            </div>
            
            <div class="mb-3">
                <h6><i class="fas fa-tachometer-alt me-2"></i>Indicadores</h6>
                <div class="d-flex justify-content-between">
                    <span>ADX:</span>
                    <span class="text-${adxStrength.color}">${data.adx.toFixed(1)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>D+:</span>
                    <span class="text-success">${data.plus_di.toFixed(1)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>D-:</span>
                    <span class="text-danger">${data.minus_di.toFixed(1)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>RSI Maverick:</span>
                    <span class="text-info">${(data.rsi_maverick * 100).toFixed(1)}%</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Ballenas Comp:</span>
                    <span class="text-success">${data.whale_pump.toFixed(1)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Ballenas Vend:</span>
                    <span class="text-danger">${data.whale_dump.toFixed(1)}</span>
                </div>
                ${trendStrengthInfo}
            </div>
        </div>
    `;
    
    document.getElementById('market-summary').innerHTML = summaryHTML;
}

function formatPriceForDisplay(price) {
    if (price < 0.01) {
        return price.toFixed(6);
    } else if (price < 1) {
        return price.toFixed(4);
    } else {
        return price.toFixed(2);
    }
}

function getVolumeLevel(volume, volumeMA) {
    const ratio = volume / volumeMA;
    if (ratio > 2) return {color: 'success', text: 'Volumen muy alto'};
    if (ratio > 1.5) return {color: 'warning', text: 'Volumen alto'};
    if (ratio > 0.8) return {color: 'info', text: 'Volumen normal'};
    return {color: 'danger', text: 'Volumen bajo'};
}

function getADXStrength(adx) {
    if (adx > 50) return {color: 'success', text: 'Tendencia muy fuerte'};
    if (adx > 25) return {color: 'warning', text: 'Tendencia fuerte'};
    return {color: 'danger', text: 'Tendencia débil'};
}

function getSignalStrength(data) {
    if (data.signal_score >= 85) return {color: 'success'};
    if (data.signal_score >= 70) return {color: 'warning'};
    return {color: 'danger'};
}

function updateSignalAnalysis(data) {
    if (!data) return;
    
    let analysisHTML = '';
    
    if (data.signal === 'NEUTRAL' || data.signal_score < 70 || !data.obligatory_conditions_ok) {
        analysisHTML = `
            <div class="text-center">
                <div class="alert alert-secondary">
                    <h6><i class="fas fa-info-circle me-2"></i>Señal No Confirmada</h6>
                    <p class="mb-2 small">Score: <strong>${data.signal_score.toFixed(1)}%</strong> (mínimo requerido: 70%)</p>
                    <p class="mb-2 small">Condiciones Obligatorias: <strong>${data.obligatory_conditions_ok ? 'CUMPLIDAS' : 'NO CUMPLIDAS'}</strong></p>
                    <p class="mb-0 small text-muted">Esperando confirmación de indicadores...</p>
                </div>
            </div>
        `;
    } else {
        const signalColor = data.signal === 'LONG' ? 'success' : 'danger';
        const signalIcon = data.signal === 'LONG' ? 'arrow-up' : 'arrow-down';
        
        analysisHTML = `
            <div class="alert alert-${signalColor}">
                <h6><i class="fas fa-${signalIcon} me-2"></i>Señal ${data.signal} CONFIRMADA</h6>
                <p class="mb-2 small"><strong>Score:</strong> ${data.signal_score.toFixed(1)}% (Mín: 70%)</p>
                <p class="mb-2 small"><strong>Condiciones Obligatorias:</strong> ✅ CUMPLIDAS</p>
                
                <h6 class="mt-3 mb-2">Indicadores Cumplidos:</h6>
                <ul class="list-unstyled small mb-3">
                    ${data.fulfilled_conditions.map(condition => `
                        <li><i class="fas fa-check text-${signalColor} me-2"></i>${condition}</li>
                    `).join('')}
                </ul>
                
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
                        <strong class="text-warning">${(data.atr_percentage * 100).toFixed(2)}%</strong>
                    </div>
                </div>
                
                <div class="row text-center mt-2">
                    <div class="col-4">
                        <small class="text-muted d-block">TP1</small>
                        <strong class="text-success">$${formatPriceForDisplay(data.take_profit[0])}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted d-block">TP2</small>
                        <strong class="text-success">$${formatPriceForDisplay(data.take_profit[1])}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted d-block">TP3</small>
                        <strong class="text-success">$${formatPriceForDisplay(data.take_profit[2])}</strong>
                    </div>
                </div>
            </div>
        `;
    }
    
    document.getElementById('signal-analysis').innerHTML = analysisHTML;
}

function updateScatterChart(interval, diPeriod, adxThreshold) {
    const url = `/api/scatter_data_improved?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            renderScatterChart(data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function renderScatterChart(scatterData) {
    const scatterElement = document.getElementById('scatter-chart');
    
    if (!scatterData || scatterData.length === 0) {
        scatterElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos para el gráfico de dispersión</p>
            </div>
        `;
        return;
    }

    const traces = [{
        x: scatterData.map(d => d.x),
        y: scatterData.map(d => d.y),
        text: scatterData.map(d => 
            `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>Señal: ${d.signal}<br>Precio: $${formatPriceForDisplay(d.current_price)}<br>Riesgo: ${d.risk_category}`
        ),
        mode: 'markers',
        marker: {
            size: scatterData.map(d => 8 + (d.signal_score / 15)),
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
            opacity: scatterData.map(d => 0.6 + (d.signal_score / 250)),
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
    const url = `/api/multiple_signals?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            updateSignalsTables(data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function updateSignalsTables(data) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    if (longTable) {
        if (data.long_signals && data.long_signals.length > 0) {
            longTable.innerHTML = data.long_signals.slice(0, 5).map((signal, index) => `
                <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                    <td>${index + 1}</td>
                    <td>${signal.symbol}</td>
                    <td><span class="badge bg-success">${signal.signal_score.toFixed(0)}%</span></td>
                    <td>$${formatPriceForDisplay(signal.entry)}</td>
                </tr>
            `).join('');
        } else {
            longTable.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center py-3 text-muted">
                        <i class="fas fa-info-circle me-2"></i>
                        No hay señales LONG confirmadas
                    </td>
                </tr>
            `;
        }
    }
    
    if (shortTable) {
        if (data.short_signals && data.short_signals.length > 0) {
            shortTable.innerHTML = data.short_signals.slice(0, 5).map((signal, index) => `
                <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                    <td>${index + 1}</td>
                    <td>${signal.symbol}</td>
                    <td><span class="badge bg-danger">${signal.signal_score.toFixed(0)}%</span></td>
                    <td>$${formatPriceForDisplay(signal.entry)}</td>
                </tr>
            `).join('');
        } else {
            shortTable.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center py-3 text-muted">
                        <i class="fas fa-info-circle me-2"></i>
                        No hay señales SHORT confirmadas
                    </td>
                </tr>
            `;
        }
    }
}

function updateScalpingAlerts() {
    const scalpingAlerts = document.getElementById('scalping-alerts');
    if (scalpingAlerts) {
        scalpingAlerts.innerHTML = `
            <div class="text-center py-2">
                <div class="spinner-border spinner-border-sm text-warning" role="status">
                    <span class="visually-hidden">Cargando...</span>
                </div>
                <p class="mt-2 mb-0 small">Buscando oportunidades...</p>
            </div>
        `;
        
        setTimeout(() => {
            scalpingAlerts.innerHTML = `
                <div class="alert alert-info text-center py-2">
                    <small><i class="fas fa-info-circle me-1"></i>Sin alertas activas</small>
                </div>
            `;
        }, 2000);
    }
}

function updateExitSignals() {
    const exitSignals = document.getElementById('exit-signals');
    if (exitSignals) {
        exitSignals.innerHTML = `
            <div class="text-center py-2">
                <div class="spinner-border spinner-border-sm text-danger" role="status">
                    <span class="visually-hidden">Cargando...</span>
                </div>
                <p class="mt-2 mb-0 small">Monitoreando operaciones...</p>
            </div>
        `;
        
        setTimeout(() => {
            exitSignals.innerHTML = `
                <div class="alert alert-secondary text-center py-2">
                    <small><i class="fas fa-info-circle me-1"></i>Sin señales de salida</small>
                </div>
            `;
        }, 2000);
    }
}

function showSignalDetails(symbol) {
    const modal = new bootstrap.Modal(document.getElementById('signalModal'));
    const modalBody = document.getElementById('signal-details');
    
    modalBody.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Cargando...</span>
            </div>
            <p class="mt-2 mb-0">Cargando detalles de ${symbol}...</p>
        </div>
    `;
    
    modal.show();
    
    setTimeout(() => {
        modalBody.innerHTML = `
            <div class="alert alert-info">
                <h6>Detalles de Señal - ${symbol}</h6>
                <p class="mb-2">Función en desarrollo - Próximamente</p>
                <p class="mb-0 small">Esta característica mostrará análisis detallado multi-temporalidad, backtesting y recomendaciones avanzadas.</p>
            </div>
        `;
    }, 1500);
}

function showError(message) {
    const toastContainer = document.getElementById('toast-container');
    if (toastContainer) {
        const toast = document.createElement('div');
        toast.className = 'toast align-items-center text-white bg-danger border-0';
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
            toast.remove();
        }, 5000);
    }
}

function showScatterError(message) {
    const scatterElement = document.getElementById('scatter-chart');
    if (scatterElement) {
        scatterElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <h5>Datos Limitados</h5>
                <p>${message}</p>
                <button class="btn btn-sm btn-primary mt-2" onclick="updateCharts()">Reintentar</button>
            </div>
        `;
    }
}

function downloadReport() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

function downloadStrategicReport() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

// Función global para actualizar desde HTML
function updateCharts() {
    updateCharts();
}
