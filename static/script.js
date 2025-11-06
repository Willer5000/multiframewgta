// MULTI-TIMEFRAME CRYPTO WGTA PRO - Script Optimizado
// Sistema responsivo y ligero para servidores de 300MB

// Configuración global optimizada
let currentChart = null;
let currentScatterChart = null;
let currentData = null;
let currentSymbol = 'BTC-USDT';
let allCryptos = [];
let updateInterval = null;

// Inicialización optimizada
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    updateCharts();
    startAutoUpdate();
});

function initializeApp() {
    console.log('MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializado');
    loadCryptoRiskClassification();
    updateCalendarInfo();
    updateWinrateDisplay();
    updateBoliviaClock();
}

function setupEventListeners() {
    // Event listeners optimizados
    const controls = [
        'interval-select', 'di-period', 'adx-threshold', 
        'sr-period', 'rsi-length', 'bb-multiplier', 'leverage'
    ];
    
    controls.forEach(control => {
        const element = document.getElementById(control);
        if (element) {
            element.addEventListener('change', updateCharts);
        }
    });
    
    // Buscador de cryptos
    setupCryptoSearch();
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterCryptoList(this.value.toUpperCase());
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
        cryptoList.innerHTML = '<div class="dropdown-item text-muted text-center">No se encontraron resultados</div>';
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
        
        let icon = '🟢', className = 'text-success';
        if (category === 'medio') {
            icon = '🟡', className = 'text-warning';
        } else if (category === 'alto') {
            icon = '🔴', className = 'text-danger';
        } else if (category === 'memecoins') {
            icon = '🟣', className = 'text-info';
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
    
    // Cerrar dropdown
    const bootstrapDropdown = bootstrap.Dropdown.getInstance(document.getElementById('cryptoDropdown'));
    if (bootstrapDropdown) bootstrapDropdown.hide();
    
    updateCharts();
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
                        📅 ${data.date} ${data.time} | Scalping 15m/30m: ${scalpingStatus} | Horario: 4am-4pm L-V
                    </small>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando información del calendario:', error);
        });
}

function updateBoliviaClock() {
    fetch('/api/bolivia_time')
        .then(response => response.json())
        .then(data => {
            const clockElement = document.getElementById('bolivia-clock');
            const dateElement = document.getElementById('bolivia-date');
            if (clockElement) clockElement.textContent = data.time;
            if (dateElement) dateElement.textContent = data.date;
        })
        .catch(error => {
            // Fallback local
            const now = new Date();
            const clockElement = document.getElementById('bolivia-clock');
            const dateElement = document.getElementById('bolivia-date');
            if (clockElement) clockElement.textContent = now.toLocaleTimeString('es-BO');
            if (dateElement) dateElement.textContent = now.toLocaleDateString('es-BO');
        });
}

function showLoadingState() {
    const elements = {
        'market-summary': 'Analizando mercado...',
        'signal-analysis': 'Evaluando condiciones de señal...',
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
    if (updateInterval) clearInterval(updateInterval);
    
    updateInterval = setInterval(() => {
        if (document.visibilityState === 'visible') {
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
    const leverage = document.getElementById('leverage').value;
    
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    updateScatterChart(interval);
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage);
    updateMultiTimeframeAnalysis(symbol, interval);
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateScalpingAlerts();
    updateExitSignals();
    updateCalendarInfo();
    updateBoliviaClock();
}

function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage) {
    const params = new URLSearchParams({
        symbol, interval, di_period: diPeriod, adx_threshold: adxThreshold,
        sr_period: srPeriod, rsi_length: rsiLength, bb_multiplier: bbMultiplier, leverage
    });
    
    fetch(`/api/signals?${params}`)
        .then(response => response.json())
        .then(data => {
            currentData = data;
            renderCandleChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Error al cargar datos: ' + error.message);
        });
}

function renderCandleChart(data) {
    const chartElement = document.getElementById('candle-chart');
    if (!chartElement) return;
    
    if (!data.data || data.data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <h5>No hay datos disponibles</h5>
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
    
    // Traza de velas japonesas optimizada
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
    
    // Añadir niveles importantes si están disponibles
    if (data.entry && data.stop_loss && data.take_profit) {
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
            line: {color: '#FF1744', dash: 'dash', width: 2},
            name: 'Stop Loss'
        });
        
        data.take_profit.forEach((tp, index) => {
            traces.push({
                type: 'scatter',
                x: [dates[0], dates[dates.length - 1]],
                y: [tp, tp],
                mode: 'lines',
                line: {color: '#00C853', dash: 'dash', width: 1.5},
                name: `TP${index + 1}`
            });
        });
    }
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas`,
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
            orientation: 'h',
            font: {color: '#ffffff'}
        },
        margin: {t: 60, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d']
    };
    
    if (currentChart) Plotly.purge('candle-chart');
    currentChart = Plotly.newPlot('candle-chart', traces, layout, config);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    if (!chartElement || !data.indicators || !data.indicators.trend_strength) return;

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const trendStrength = data.indicators.trend_strength || [];
    const colors = data.indicators.colors || [];
    
    const trace = {
        x: dates,
        y: trendStrength,
        type: 'bar',
        name: 'Fuerza de Tendencia',
        marker: { color: colors }
    };
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: { type: 'date', gridcolor: '#444' },
        yaxis: { title: 'Fuerza %', gridcolor: '#444' },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 40, r: 30, b: 40, l: 50},
        height: 250
    };
    
    const config = { responsive: true, displayModeBar: false };
    
    Plotly.newPlot('trend-strength-chart', [trace], layout, config);
}

function updateMarketSummary(data) {
    const summaryElement = document.getElementById('market-summary');
    if (!summaryElement) return;
    
    // Validar que data existe y tiene las propiedades necesarias
    if (!data) {
        summaryElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <h6><i class="fas fa-exclamation-triangle me-2"></i>Datos no disponibles</h6>
                <p class="small mb-0">No se pudieron cargar los datos del mercado</p>
            </div>
        `;
        return;
    }
    
    const multiTF = data.multi_timeframe_analysis || {};
    const signal = data.signal || 'NEUTRAL';
    const signalScore = data.signal_score || 0;
    const currentPrice = data.current_price || 0;
    
    summaryElement.innerHTML = `
        <div class="fade-in">
            <div class="row text-center mb-3">
                <div class="col-6">
                    <div class="card bg-dark border-${signal === 'LONG' ? 'success' : signal === 'SHORT' ? 'danger' : 'secondary'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Señal</small>
                            <h5 class="mb-0 text-${signal === 'LONG' ? 'success' : signal === 'SHORT' ? 'danger' : 'muted'}">
                                ${signal}
                            </h5>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card bg-dark border-${signalScore >= 70 ? 'success' : 'warning'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Score</small>
                            <h5 class="mb-0 text-${signalScore >= 70 ? 'success' : 'warning'}">
                                ${signalScore.toFixed(0)}%
                            </h5>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-dollar-sign me-2"></i>Precio Actual</h6>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="fs-6 fw-bold">$${formatPriceForDisplay(currentPrice)}</span>
                    <small class="text-muted">USDT</small>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-layer-group me-2"></i>Multi-Temporalidad</h6>
                <div class="small">
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
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const analysisElement = document.getElementById('signal-analysis');
    if (!analysisElement) return;
    
    let analysisHTML = '';
    
    if (data.signal === 'NEUTRAL' || data.signal_score < 70) {
        analysisHTML = `
            <div class="text-center">
                <div class="alert alert-secondary">
                    <h6><i class="fas fa-info-circle me-2"></i>Señal No Confirmada</h6>
                    <p class="mb-2 small">Score: <strong>${data.signal_score.toFixed(1)}%</strong></p>
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
                <p class="mb-2 small"><strong>Score:</strong> ${data.signal_score.toFixed(1)}%</p>
                
                <div class="row text-center mt-2">
                    <div class="col-4">
                        <small class="text-muted d-block">Entrada</small>
                        <strong class="text-${signalColor}">$${formatPriceForDisplay(data.entry)}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted d-block">Stop Loss</small>
                        <strong class="text-danger">$${formatPriceForDisplay(data.stop_loss)}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted d-block">TP1</small>
                        <strong class="text-success">$${formatPriceForDisplay(data.take_profit[0])}</strong>
                    </div>
                </div>
            </div>
        `;
    }
    
    analysisElement.innerHTML = analysisHTML;
}

function updateScatterChart(interval) {
    fetch(`/api/scatter_data_improved?interval=${interval}`)
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
    if (!scatterElement || !scatterData || scatterData.length === 0) return;
    
    const trace = {
        x: scatterData.map(d => d.x),
        y: scatterData.map(d => d.y),
        text: scatterData.map(d => 
            `${d.symbol}<br>Score: ${d.signal_score.toFixed(1)}%<br>Señal: ${d.signal}`
        ),
        mode: 'markers',
        marker: {
            size: scatterData.map(d => 8 + (d.signal_score / 15)),
            color: scatterData.map(d => {
                if (d.signal === 'LONG') {
                    return d.risk_category === 'bajo' ? '#00C853' : 
                           d.risk_category === 'medio' ? '#FFC107' : '#FF9800';
                }
                if (d.signal === 'SHORT') {
                    return d.risk_category === 'bajo' ? '#FF1744' : 
                           d.risk_category === 'medio' ? '#FF5252' : '#F44336';
                }
                return '#9E9E9E';
            }),
            opacity: 0.7
        },
        type: 'scatter',
        hovertemplate: '%{text}<extra></extra>'
    };
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Análisis Multi-Indicador',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Presión Compradora (%)',
            range: [0, 100],
            gridcolor: '#444'
        },
        yaxis: {
            title: 'Presión Vendedora (%)',
            range: [0, 100],
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 60, r: 30, b: 50, l: 50},
        height: 350
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    if (currentScatterChart) Plotly.purge('scatter-chart');
    currentScatterChart = Plotly.newPlot('scatter-chart', [trace], layout, config);
}

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, leverage) {
    const params = new URLSearchParams({
        interval, di_period: diPeriod, adx_threshold: adxThreshold,
        sr_period: srPeriod, rsi_length: rsiLength, bb_multiplier: bbMultiplier, leverage
    });
    
    fetch(`/api/multiple_signals?${params}`)
        .then(response => response.json())
        .then(data => {
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
            longTable.innerHTML = data.long_signals.slice(0, 3).map((signal, index) => `
                <tr onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;">
                    <td class="text-center">${index + 1}</td>
                    <td><small><strong>${signal.symbol}</strong></small></td>
                    <td class="text-center"><span class="badge bg-success">${signal.signal_score.toFixed(0)}%</span></td>
                    <td class="text-end"><small>$${formatPriceForDisplay(signal.entry)}</small></td>
                </tr>
            `).join('');
        } else {
            longTable.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center py-2 text-muted">
                        <small>No hay señales LONG</small>
                    </td>
                </tr>
            `;
        }
    }
    
    // Actualizar tabla SHORT
    const shortTable = document.getElementById('short-table');
    if (shortTable) {
        if (data.short_signals && data.short_signals.length > 0) {
            shortTable.innerHTML = data.short_signals.slice(0, 3).map((signal, index) => `
                <tr onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;">
                    <td class="text-center">${index + 1}</td>
                    <td><small><strong>${signal.symbol}</strong></small></td>
                    <td class="text-center"><span class="badge bg-danger">${signal.signal_score.toFixed(0)}%</span></td>
                    <td class="text-end"><small>$${formatPriceForDisplay(signal.entry)}</small></td>
                </tr>
            `).join('');
        } else {
            shortTable.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center py-2 text-muted">
                        <small>No hay señales SHORT</small>
                    </td>
                </tr>
            `;
        }
    }
}

function updateFearGreedIndex() {
    fetch('/api/fear_greed_index')
        .then(response => response.json())
        .then(data => {
            const fgiElement = document.getElementById('fear-greed-index');
            if (fgiElement) {
                fgiElement.innerHTML = `
                    <div class="card bg-dark border-${data.color || 'secondary'}">
                        <div class="card-body text-center py-2">
                            <h6 class="mb-2"><i class="fas fa-brain me-2"></i>Índice Miedo y Codicia</h6>
                            <div class="progress mb-2" style="height: 15px;">
                                <div class="progress-bar bg-${data.color || 'secondary'}" role="progressbar" 
                                     style="width: ${data.value || 50}%">
                                    ${data.value || 'N/A'}
                                </div>
                            </div>
                            <small class="text-${data.color || 'muted'}">${data.sentiment || 'No disponible'}</small>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando índice:', error);
        });
}

function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => response.json())
        .then(data => {
            const alertsElement = document.getElementById('scalping-alerts');
            if (!alertsElement) return;
            
            if (data.alerts && data.alerts.length > 0) {
                let alertsHTML = '';
                
                data.alerts.slice(0, 3).forEach((alert, index) => {
                    const alertType = alert.signal === 'LONG' ? 'success' : 'danger';
                    const alertIcon = alert.signal === 'LONG' ? 'arrow-up' : 'arrow-down';
                    
                    alertsHTML += `
                        <div class="alert alert-${alertType} mb-2 py-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="mb-1 small">
                                        <i class="fas fa-${alertIcon} me-1"></i>
                                        ${alert.symbol}
                                    </h6>
                                    <p class="mb-0 small">
                                        Score: ${alert.score.toFixed(1)}% | Leverage: x${alert.leverage}
                                    </p>
                                </div>
                                <button class="btn btn-sm btn-outline-${alertType}" onclick="tradeAlert('${alert.symbol}', '${alert.interval}', ${alert.leverage})">
                                    Operar
                                </button>
                            </div>
                        </div>
                    `;
                });
                
                alertsElement.innerHTML = alertsHTML;
            } else {
                alertsElement.innerHTML = `
                    <div class="text-center py-2 text-muted">
                        <small>No hay alertas activas</small>
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
            if (!exitElement) return;
            
            if (data.exit_signals && data.exit_signals.length > 0) {
                let exitHTML = '';
                
                data.exit_signals.slice(0, 3).forEach((alert, index) => {
                    const alertType = alert.pnl_percent >= 0 ? 'success' : 'danger';
                    const alertIcon = alert.pnl_percent >= 0 ? 'trophy' : 'exclamation-triangle';
                    
                    exitHTML += `
                        <div class="alert alert-${alertType} mb-2 py-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="mb-1 small">
                                        <i class="fas fa-${alertIcon} me-1"></i>
                                        ${alert.symbol}
                                    </h6>
                                    <p class="mb-0 small">
                                        P&L: ${alert.pnl_percent.toFixed(2)}%
                                    </p>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                exitElement.innerHTML = exitHTML;
            } else {
                exitElement.innerHTML = `
                    <div class="text-center py-2 text-muted">
                        <small>No hay señales de salida</small>
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
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const winrateDisplay = document.getElementById('winrate-display');
            if (winrateDisplay) {
                const winrate = data && typeof data.winrate !== 'undefined' ? data.winrate : 0;
                const totalOps = data && data.total_operations ? data.total_operations : 0;
                
                winrateDisplay.innerHTML = `
                    <h4 class="text-success mb-1">${winrate.toFixed(1)}%</h4>
                    <p class="small text-muted mb-0">${totalOps} operaciones</p>
                    <div class="progress mt-2" style="height: 6px;">
                        <div class="progress-bar bg-success" style="width: ${Math.min(winrate, 100)}%"></div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
            // Fallback para evitar errores en UI
            const winrateDisplay = document.getElementById('winrate-display');
            if (winrateDisplay) {
                winrateDisplay.innerHTML = `
                    <h4 class="text-secondary mb-1">0.0%</h4>
                    <p class="small text-muted mb-0">0 operaciones</p>
                    <div class="progress mt-2" style="height: 6px;">
                        <div class="progress-bar bg-secondary" style="width: 0%"></div>
                    </div>
                `;
            }
        });
}


function loadCryptoRiskClassification() {
    fetch('/api/crypto_risk_classification')
        .then(response => response.json())
        .then(riskData => {
            allCryptos = [];
            Object.keys(riskData).forEach(category => {
                if (Array.isArray(riskData[category])) {
                    riskData[category].forEach(symbol => {
                        allCryptos.push({ symbol, category });
                    });
                }
            });
            filterCryptoList('');
        })
        .catch(error => {
            console.error('Error cargando clasificación:', error);
            loadBasicCryptoSymbols();
        });
}

function loadBasicCryptoSymbols() {
    const basicSymbols = [
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'XRP-USDT', 'ADA-USDT',
        'SOL-USDT', 'DOT-USDT', 'DOGE-USDT', 'AVAX-USDT', 'LINK-USDT'
    ];
    
    allCryptos = basicSymbols.map(symbol => ({ symbol, category: 'bajo' }));
    filterCryptoList('');
}

function tradeAlert(symbol, interval, leverage) {
    currentSymbol = symbol;
    document.getElementById('selected-crypto').textContent = symbol;
    document.getElementById('interval-select').value = interval;
    document.getElementById('leverage').value = leverage;
    
    updateCharts();
    showNotification(`Configurado para operar ${symbol} en ${interval}`, 'success');
}

function showNotification(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toastId = 'toast-' + Date.now();
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <small>${message}</small>
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.innerHTML += toastHTML;
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    document.body.appendChild(container);
    return container;
}

function showSignalDetails(symbol) {
    const modal = new bootstrap.Modal(document.getElementById('signalModal'));
    
    const signalData = currentData && currentData.symbol === symbol ? currentData : null;
    
    const detailsHTML = signalData ? `
        <div class="signal-details">
            <h5 class="text-${signalData.signal === 'LONG' ? 'success' : 'danger'}">
                <i class="fas fa-${signalData.signal === 'LONG' ? 'arrow-up' : 'arrow-down'} me-2"></i>
                ${symbol} - Señal ${signalData.signal}
            </h5>
            
            <div class="row mt-3">
                <div class="col-md-6">
                    <h6>Información de Trading</h6>
                    <table class="table table-sm table-dark">
                        <tr><td>Precio Actual:</td><td class="text-end">$${formatPriceForDisplay(signalData.current_price)}</td></tr>
                        <tr><td>Entrada:</td><td class="text-end text-${signalData.signal === 'LONG' ? 'success' : 'danger'}">$${formatPriceForDisplay(signalData.entry)}</td></tr>
                        <tr><td>Stop Loss:</td><td class="text-end text-danger">$${formatPriceForDisplay(signalData.stop_loss)}</td></tr>
                        <tr><td>Score:</td><td class="text-end text-warning">${signalData.signal_score.toFixed(1)}%</td></tr>
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
            
            <div class="mt-3 text-center">
                <button class="btn btn-primary me-2" onclick="downloadSignalReport('${symbol}')">
                    <i class="fas fa-download me-1"></i>Descargar Reporte
                </button>
                <button class="btn btn-outline-secondary" data-bs-dismiss="modal">
                    Cerrar
                </button>
            </div>
        </div>
    ` : `
        <div class="text-center py-4">
            <h5>${symbol}</h5>
            <p class="text-muted">No hay información detallada disponible.</p>
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

function updateMultiTimeframeAnalysis(symbol, interval) {
    // Esta función se puede expandir para mostrar análisis multi-temporalidad detallado
    console.log(`Análisis multi-temporalidad para ${symbol} en ${interval}`);
}

// Funciones auxiliares optimizadas
function formatPriceForDisplay(price) {
    if (!price || price === 0) return '0.00';
    if (price < 0.01) return price.toFixed(6);
    if (price < 1) return price.toFixed(4);
    return price.toFixed(2);
}

function showError(message) {
    const chartElement = document.getElementById('candle-chart');
    if (chartElement) {
        chartElement.innerHTML = `
            <div class="alert alert-danger text-center">
                <h6><i class="fas fa-exclamation-triangle me-2"></i>Error</h6>
                <p class="small">${message}</p>
                <button class="btn btn-sm btn-primary mt-1" onclick="updateCharts()">
                    Reintentar
                </button>
            </div>
        `;
    }
}

// Actualizaciones automáticas
setInterval(updateBoliviaClock, 1000);
setInterval(updateWinrateDisplay, 30000);
