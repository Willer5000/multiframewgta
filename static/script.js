// Configuración global
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
    
    // Configurar buscador de cryptos
    setupCryptoSearch();
    
    // Configurar controles de indicadores
    setupIndicatorControls();
}

function updateCalendarInfo() {
    // Actualizar información del calendario y horario de trading
    fetch('/api/bolivia_time')
        .then(response => response.json())
        .then(data => {
            const calendarInfo = document.getElementById('calendar-info');
            if (calendarInfo) {
                const tradingStatus = data.is_trading_time ? 
                    '<span class="text-success">🟢 ACTIVO</span>' : 
                    '<span class="text-danger">🔴 INACTIVO</span>';
                
                calendarInfo.innerHTML = `
                    <small class="text-muted">
                        📅 ${data.day_of_week} | Trading Multi-TF: ${tradingStatus} | Horario: 4am-4pm L-V
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
    
    // Prevenir que el dropdown se cierre al hacer clic en el buscador
    searchInput.addEventListener('click', function(e) {
        e.stopPropagation();
    });
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

// Función para cargar clasificación de riesgo
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
    // Actualizar información del mercado
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateScalpingAlerts();
    updateExitSignals();
    updateCalendarInfo();
    updateSystemWinrate();
}

function updateSystemWinrate() {
    // Actualizar winrate del sistema
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/winrate?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            const winrateElement = document.getElementById('system-winrate');
            if (winrateElement) {
                const winrate = data.winrate || 65.0;
                const winrateClass = winrate >= 70 ? 'text-success' : winrate >= 60 ? 'text-warning' : 'text-danger';
                
                winrateElement.innerHTML = `
                    <div class="winrate-display">
                        <div class="winrate-value ${winrateClass} display-4 fw-bold">${winrate.toFixed(1)}%</div>
                        <div class="winrate-label text-muted small">Winrate Histórico</div>
                        <div class="progress mt-2" style="height: 8px;">
                            <div class="progress-bar ${winrate >= 70 ? 'bg-success' : winrate >= 60 ? 'bg-warning' : 'bg-danger'}" 
                                 role="progressbar" style="width: ${winrate}%" 
                                 aria-valuenow="${winrate}" aria-valuemin="0" aria-valuemax="100">
                            </div>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando winrate:', error);
        });
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
    
    // Actualizar gráfico principal
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar gráfico de dispersión
    updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar señales múltiples
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar winrate
    updateSystemWinrate();
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateScalpingAlerts();
    updateExitSignals();
    updateCalendarInfo();
    updateSystemWinrate();
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
        winrate: 65.0,
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
        multi_timeframe_ok: false,
        ma200_condition: 'above',
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
    
    // Calcular rango dinámico para el eje Y
    const visibleHighs = highs.slice(-50);
    const visibleLows = lows.slice(-50);
    const minPrice = Math.min(...visibleLows);
    const maxPrice = Math.max(...visibleHighs);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.05;
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas con Indicadores Multi-Temporalidad`,
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
        margin: {t: 80, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
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
            x: dates.filter((_, i) => confirmedBuy[i]),
            y: whalePump.filter((_, i) => confirmedBuy[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Compra Confirmada',
            marker: {color: '#00FF00', size: 10, symbol: 'diamond'}
        },
        {
            x: dates.filter((_, i) => confirmedSell[i]),
            y: whaleDump.filter((_, i) => confirmedSell[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Venta Confirmada',
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
        margin: {t: 60, r: 50, b: 50, l: 50}
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
        margin: {t: 60, r: 50, b: 50, l: 50}
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
    const bullishDivergence = data.indicators.rsi_bullish_divergence || [];
    const bearishDivergence = data.indicators.rsi_bearish_divergence || [];
    
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
            text: 'RSI Tradicional con Divergencias (14 Periodos)',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'RSI Value',
            range: [0, 100],
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        shapes: [
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 80,
                y1: 80,
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
                y0: 20,
                y1: 20,
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
                y0: 50,
                y1: 50,
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
                y: 80,
                xanchor: 'left',
                text: 'Sobrecompra',
                showarrow: false,
                font: {color: 'red', size: 10}
            },
            {
                x: dates[dates.length - 1],
                y: 20,
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
        margin: {t: 60, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
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
    const bullishDivergence = data.indicators.rsi_maverick_bullish_divergence || [];
    const bearishDivergence = data.indicators.rsi_maverick_bearish_divergence || [];
    
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
            line: {color: '#FF6B6B', width: 2}
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
        margin: {t: 60, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
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
        value >= 0 ? 'rgba(0, 200, 83, 0.8)' : 'rgba(255, 23, 68, 0.8)'
    );
    
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
            line: {color: '#FF9800', width: 1.5}
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
            title: 'MACD Value',
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
        margin: {t: 60, r: 50, b: 50, l: 50}
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
    const highZoneThreshold = data.indicators.high_zone_threshold || 5;
    
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
    
    // Añadir líneas de referencia
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
        displaylogo: false
    };
    
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', traces, layout, config);
}

function updateMarketSummary(data) {
    if (!data) return;
    
    const volumeLevel = getVolumeLevel(data.volume, data.volume_ma);
    const signalColor = data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary';
    const signalIcon = data.signal === 'LONG' ? '📈' : data.signal === 'SHORT' ? '📉' : '⚖️';
    
    const marketSummary = document.getElementById('market-summary');
    marketSummary.innerHTML = `
        <div class="row g-2">
            <div class="col-12">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Señal:</span>
                    <span class="badge bg-${signalColor}">${signalIcon} ${data.signal}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Score:</span>
                    <span class="fw-bold ${data.signal_score >= 70 ? 'text-success' : data.signal_score >= 50 ? 'text-warning' : 'text-danger'}">
                        ${data.signal_score.toFixed(1)}%
                    </span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Precio:</span>
                    <span class="fw-bold">$${data.current_price.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Volumen:</span>
                    <span class="badge ${volumeLevel.class}">${volumeLevel.text}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">ADX:</span>
                    <span class="${data.adx >= 25 ? 'text-success' : 'text-warning'}">${data.adx.toFixed(1)}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Multi-TF:</span>
                    <span class="${data.multi_timeframe_ok ? 'text-success' : 'text-danger'}">
                        ${data.multi_timeframe_ok ? '✅' : '❌'}
                    </span>
                </div>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="text-muted small">MA200:</span>
                    <span class="${data.ma200_condition === 'above' ? 'text-success' : 'text-warning'}">
                        ${data.ma200_condition === 'above' ? 'ENCIMA' : 'DEBAJO'}
                    </span>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    if (!data) return;
    
    const signalAnalysis = document.getElementById('signal-analysis');
    const conditionsCount = data.fulfilled_conditions ? data.fulfilled_conditions.length : 0;
    
    let analysisHTML = '';
    
    if (data.signal === 'NEUTRAL' || data.signal_score < 70) {
        analysisHTML = `
            <div class="alert alert-secondary text-center py-2">
                <i class="fas fa-pause-circle me-2"></i>
                <strong>SEÑAL NEUTRAL</strong>
                <div class="small mt-1">Score: ${data.signal_score.toFixed(1)}%</div>
                <div class="small text-muted">Esperando mejores condiciones</div>
            </div>
        `;
    } else {
        const signalClass = data.signal === 'LONG' ? 'success' : 'danger';
        const signalIcon = data.signal === 'LONG' ? 'arrow-up' : 'arrow-down';
        
        analysisHTML = `
            <div class="alert alert-${signalClass} text-center py-2">
                <i class="fas fa-${signalIcon} me-2"></i>
                <strong>SEÑAL ${data.signal} CONFIRMADA</strong>
                <div class="small mt-1">Score: ${data.signal_score.toFixed(1)}% | Condiciones: ${conditionsCount}/8</div>
                <div class="small">Winrate: ${data.winrate.toFixed(1)}%</div>
            </div>
            
            <div class="mt-2">
                <div class="d-flex justify-content-between small mb-1">
                    <span>Entrada:</span>
                    <span class="fw-bold">$${data.entry.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between small mb-1">
                    <span>Stop Loss:</span>
                    <span class="fw-bold text-danger">$${data.stop_loss.toFixed(6)}</span>
                </div>
                <div class="d-flex justify-content-between small">
                    <span>Take Profit:</span>
                    <span class="fw-bold text-success">$${data.take_profit[0].toFixed(6)}</span>
                </div>
            </div>
            
            ${data.fulfilled_conditions && data.fulfilled_conditions.length > 0 ? `
                <div class="mt-2">
                    <small class="text-muted d-block mb-1">Condiciones cumplidas:</small>
                    <div style="max-height: 100px; overflow-y: auto;">
                        ${data.fulfilled_conditions.map(cond => `
                            <div class="small text-success mb-1">✓ ${cond}</div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }
    
    signalAnalysis.innerHTML = analysisHTML;
}

function getVolumeLevel(currentVolume, averageVolume) {
    const ratio = currentVolume / averageVolume;
    if (ratio > 2.0) return { text: 'MUY ALTO', class: 'bg-success' };
    if (ratio > 1.5) return { text: 'ALTO', class: 'bg-info' };
    if (ratio > 1.0) return { text: 'MEDIO', class: 'bg-warning' };
    if (ratio > 0.5) return { text: 'BAJO', class: 'bg-secondary' };
    return { text: 'MUY BAJO', class: 'bg-dark' };
}

function updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/scatter_data_improved?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(scatterData => {
            renderScatterChartImproved(scatterData);
        })
        .catch(error => {
            console.error('Error cargando datos de scatter:', error);
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
    
    const traces = {
        'bajo': { x: [], y: [], text: [], marker: { size: [], color: [] }, name: 'Bajo Riesgo', mode: 'markers', type: 'scatter' },
        'medio': { x: [], y: [], text: [], marker: { size: [], color: [] }, name: 'Medio Riesgo', mode: 'markers', type: 'scatter' },
        'alto': { x: [], y: [], text: [], marker: { size: [], color: [] }, name: 'Alto Riesgo', mode: 'markers', type: 'scatter' },
        'memecoins': { x: [], y: [], text: [], marker: { size: [], color: [] }, name: 'Memecoins', mode: 'markers', type: 'scatter' }
    };
    
    data.forEach(item => {
        const trace = traces[item.risk_category];
        if (trace) {
            trace.x.push(item.x);
            trace.y.push(item.y);
            trace.text.push(`${item.symbol}<br>Score: ${item.signal_score}%<br>Precio: $${item.current_price.toFixed(6)}`);
            trace.marker.size.push(Math.max(10, item.signal_score / 2));
            trace.marker.color.push(item.signal === 'LONG' ? '#00C853' : item.signal === 'SHORT' ? '#FF1744' : '#9E9E9E');
        }
    });
    
    const plotData = Object.values(traces).filter(trace => trace.x.length > 0);
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Presión Compradora vs Vendedora',
            font: { color: '#ffffff', size: 16 }
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
        font: { color: '#ffffff' },
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: { color: '#ffffff' },
            bgcolor: 'rgba(0,0,0,0)'
        },
        margin: { t: 80, r: 50, b: 50, l: 50 },
        shapes: [
            {
                type: 'line',
                x0: 50,
                y0: 0,
                x1: 50,
                y1: 100,
                line: { color: 'white', width: 1, dash: 'dot' }
            },
            {
                type: 'line',
                x0: 0,
                y0: 50,
                x1: 100,
                y1: 50,
                line: { color: 'white', width: 1, dash: 'dot' }
            }
        ],
        annotations: [
            {
                x: 75,
                y: 75,
                xref: 'x',
                yref: 'y',
                text: 'Zona LONG<br>(Alta Compra, Baja Venta)',
                showarrow: false,
                font: { color: '#00C853', size: 12 },
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: '#00C853',
                borderwidth: 1,
                borderpad: 4
            },
            {
                x: 25,
                y: 25,
                xref: 'x',
                yref: 'y',
                text: 'Zona SHORT<br>(Baja Compra, Alta Venta)',
                showarrow: false,
                font: { color: '#FF1744', size: 12 },
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: '#FF1744',
                borderwidth: 1,
                borderpad: 4
            }
        ]
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        toImageButtonOptions: {
            format: 'png',
            filename: 'scatter_opportunities',
            height: 600,
            width: 800,
            scale: 2
        }
    };
    
    if (currentScatterChart) {
        Plotly.purge('scatter-chart');
    }
    
    currentScatterChart = Plotly.newPlot('scatter-chart', plotData, layout, config);
}

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/multiple_signals?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            updateSignalTables(data.long_signals, data.short_signals);
        })
        .catch(error => {
            console.error('Error cargando múltiples señales:', error);
        });
}

function updateSignalTables(longSignals, shortSignals) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    // Actualizar tabla LONG
    if (longSignals && longSignals.length > 0) {
        longTable.innerHTML = longSignals.slice(0, 5).map((signal, index) => `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}', '${signal.signal}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-success">${signal.signal_score.toFixed(1)}%</span></td>
                <td>$${signal.entry.toFixed(6)}</td>
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
    
    // Actualizar tabla SHORT
    if (shortSignals && shortSignals.length > 0) {
        shortTable.innerHTML = shortSignals.slice(0, 5).map((signal, index) => `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}', '${signal.signal}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-danger">${signal.signal_score.toFixed(1)}%</span></td>
                <td>$${signal.entry.toFixed(6)}</td>
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

function updateFearGreedIndex() {
    // Simulación del índice de miedo y codicia
    const fearGreedElement = document.getElementById('fear-greed-index');
    const randomIndex = Math.floor(Math.random() * 100);
    
    let level, color, description;
    if (randomIndex >= 75) {
        level = 'EXTREME GREED';
        color = 'danger';
        description = 'Mercado muy optimista - Precaución';
    } else if (randomIndex >= 55) {
        level = 'GREED';
        color = 'warning';
        description = 'Mercado optimista';
    } else if (randomIndex >= 45) {
        level = 'NEUTRAL';
        color = 'secondary';
        description = 'Mercado equilibrado';
    } else if (randomIndex >= 25) {
        level = 'FEAR';
        color = 'info';
        description = 'Mercado pesimista';
    } else {
        level = 'EXTREME FEAR';
        color = 'success';
        description = 'Oportunidades de compra';
    }
    
    fearGreedElement.innerHTML = `
        <div class="text-center">
            <div class="fear-greed-value display-6 fw-bold text-${color}">${randomIndex}</div>
            <div class="fear-greed-level small text-${color}">${level}</div>
            <div class="progress fear-greed-progress mt-2">
                <div class="progress-bar bg-${color}" role="progressbar" 
                     style="width: ${randomIndex}%" 
                     aria-valuenow="${randomIndex}" aria-valuemin="0" aria-valuemax="100">
                </div>
            </div>
            <div class="small text-muted mt-1">${description}</div>
        </div>
    `;
}

function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const alertsElement = document.getElementById('scalping-alerts');
            if (data.alerts && data.alerts.length > 0) {
                alertsElement.innerHTML = data.alerts.slice(0, 3).map(alert => `
                    <div class="alert alert-warning scalping-alert mb-2 p-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <strong>${alert.symbol}</strong>
                                <div class="small">${alert.interval} - ${alert.signal}</div>
                                <div class="small">Score: ${alert.score.toFixed(1)}%</div>
                            </div>
                            <span class="badge bg-${alert.signal === 'LONG' ? 'success' : 'danger'}">
                                $${alert.entry.toFixed(6)}
                            </span>
                        </div>
                    </div>
                `).join('');
            } else {
                alertsElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="fas fa-bell-slash fa-2x mb-2"></i>
                        <div class="small">No hay alertas activas</div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando alertas:', error);
        });
}

function updateExitSignals() {
    fetch('/api/exit_signals')
        .then(response => response.json())
        .then(data => {
            const exitElement = document.getElementById('exit-signals');
            if (data.exit_signals && data.exit_signals.length > 0) {
                exitElement.innerHTML = data.exit_signals.slice(0, 3).map(signal => `
                    <div class="alert alert-danger mb-2 p-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <strong>${signal.symbol}</strong>
                                <div class="small">${signal.interval} - CERRAR ${signal.signal}</div>
                                <div class="small">${signal.reason}</div>
                            </div>
                            <span class="badge ${signal.pnl_percent >= 0 ? 'bg-success' : 'bg-danger'}">
                                ${signal.pnl_percent >= 0 ? '+' : ''}${signal.pnl_percent.toFixed(2)}%
                            </span>
                        </div>
                    </div>
                `).join('');
            } else {
                exitElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="fas fa-check-circle fa-2x mb-2"></i>
                        <div class="small">No hay señales de salida</div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando señales de salida:', error);
        });
}

function showSignalDetails(symbol, signalType) {
    // Implementar modal con detalles de la señal
    const modal = new bootstrap.Modal(document.getElementById('signalModal'));
    const detailsElement = document.getElementById('signal-details');
    
    detailsElement.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Cargando...</span>
            </div>
            <p class="mt-2 mb-0">Cargando detalles de ${symbol}...</p>
        </div>
    `;
    
    modal.show();
    
    // Simular carga de detalles
    setTimeout(() => {
        detailsElement.innerHTML = `
            <h6>Detalles de Señal - ${symbol}</h6>
            <p><strong>Tipo:</strong> <span class="badge bg-${signalType === 'LONG' ? 'success' : 'danger'}">${signalType}</span></p>
            <p><strong>Score:</strong> 85.5%</p>
            <p><strong>Winrate Histórico:</strong> 72.3%</p>
            <p><strong>Condiciones Multi-TF:</strong> ✅ Confirmadas</p>
            <p><strong>Fuerza Tendencia:</strong> 🟢 Favorable</p>
            <hr>
            <h6>Niveles de Trading</h6>
            <p><strong>Entrada:</strong> $50,250.00</p>
            <p><strong>Stop Loss:</strong> $48,500.00</p>
            <p><strong>Take Profit:</strong> $52,800.00</p>
            <p><strong>Ratio R/R:</strong> 1:2.1</p>
        `;
    }, 1000);
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

function showError(message) {
    // Mostrar notificación de error
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'error-' + Date.now();
    
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
    
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
    toast.show();
    
    // Remover el toast del DOM después de que se oculte
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
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
            console.error('Error actualizando reloj:', error);
            const now = new Date();
            document.getElementById('bolivia-clock').textContent = now.toLocaleTimeString('es-BO');
            document.getElementById('bolivia-date').textContent = now.toLocaleDateString('es-BO');
        });
}

// Función para descargar reporte estratégico
function downloadStrategicReport() {
    const symbol = document.getElementById('selected-crypto').textContent;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

// Actualizar el reloj cada segundo
setInterval(updateBoliviaClock, 1000);

// Inicializar el reloj al cargar la página
document.addEventListener('DOMContentLoaded', function() {
    updateBoliviaClock();
});
