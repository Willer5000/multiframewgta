// Configuración global
let currentChart = null;
let currentAdxChart = null;
let currentStochRsiChart = null;
let currentTrendStrengthChart = null;
let currentVolumeChart = null;
let currentWhaleChart = null;
let currentRsiMaverickChart = null;
let currentRsiTraditionalChart = null;
let currentMacdChart = null;
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
    updateCalendarInfo();
    setupIndicatorControls();
}

function setupEventListeners() {
    // Configurar event listeners para los controles
    document.getElementById('symbol-select').addEventListener('change', updateCharts);
    document.getElementById('interval-select').addEventListener('change', updateCharts);
    document.getElementById('leverage').addEventListener('change', updateCharts);
    
    // Configurar controles de indicadores
    setupIndicatorControls();
    
    // Configurar botones de colapso/expansión
    setupCollapseButtons();
    
    // Configurar botones de movimiento
    setupMoveButtons();
}

function setupIndicatorControls() {
    const indicatorControls = document.querySelectorAll('.indicator-control');
    indicatorControls.forEach(control => {
        control.addEventListener('change', function() {
            updateChartIndicators();
        });
    });
}

function setupCollapseButtons() {
    const collapseButtons = document.querySelectorAll('.btn-collapse');
    collapseButtons.forEach(button => {
        button.addEventListener('click', function() {
            const card = this.closest('.indicator-card');
            const content = card.querySelector('.indicator-content');
            const isCollapsed = this.getAttribute('data-collapsed') === 'true';
            
            if (isCollapsed) {
                content.style.display = 'block';
                content.style.height = '300px';
                this.innerHTML = '<i class="fas fa-minus"></i>';
                this.setAttribute('data-collapsed', 'false');
            } else {
                content.style.display = 'none';
                content.style.height = '0';
                this.innerHTML = '<i class="fas fa-plus"></i>';
                this.setAttribute('data-collapsed', 'true');
            }
        });
    });
}

function setupMoveButtons() {
    const moveButtons = document.querySelectorAll('.btn-move');
    moveButtons.forEach(button => {
        button.addEventListener('click', function() {
            const direction = this.getAttribute('data-direction');
            const card = this.closest('.indicator-card');
            const container = document.getElementById('indicators-container');
            
            if (direction === 'up') {
                const prevCard = card.previousElementSibling;
                if (prevCard && prevCard.classList.contains('indicator-card')) {
                    container.insertBefore(card, prevCard);
                }
            } else if (direction === 'down') {
                const nextCard = card.nextElementSibling;
                if (nextCard && nextCard.classList.contains('indicator-card')) {
                    container.insertBefore(nextCard, card);
                }
            }
            
            // Guardar orden en localStorage
            saveIndicatorOrder();
        });
    });
}

function saveIndicatorOrder() {
    const container = document.getElementById('indicators-container');
    const cards = Array.from(container.querySelectorAll('.indicator-card'));
    const order = cards.map(card => card.getAttribute('data-indicator'));
    localStorage.setItem('indicatorOrder', JSON.stringify(order));
}

function loadIndicatorOrder() {
    const savedOrder = localStorage.getItem('indicatorOrder');
    if (savedOrder) {
        try {
            const order = JSON.parse(savedOrder);
            const container = document.getElementById('indicators-container');
            const cards = Array.from(container.querySelectorAll('.indicator-card'));
            
            // Crear un mapa de tarjetas por indicador
            const cardMap = {};
            cards.forEach(card => {
                const indicator = card.getAttribute('data-indicator');
                cardMap[indicator] = card;
                container.removeChild(card);
            });
            
            // Reordenar según lo guardado
            order.forEach(indicator => {
                if (cardMap[indicator]) {
                    container.appendChild(cardMap[indicator]);
                }
            });
        } catch (error) {
            console.error('Error cargando orden de indicadores:', error);
        }
    }
}

function updateCalendarInfo() {
    fetch('/api/bolivia_time')
        .then(response => response.json())
        .then(data => {
            const calendarInfo = document.getElementById('calendar-info');
            if (calendarInfo) {
                const now = new Date();
                const dayOfWeek = now.toLocaleDateString('es-BO', { weekday: 'long' });
                const hour = now.getHours();
                const isTradingTime = hour >= 4 && hour < 16 && now.getDay() >= 1 && now.getDay() <= 5;
                
                const tradingStatus = isTradingTime ? 
                    '<span class="text-success">🟢 ACTIVO</span>' : 
                    '<span class="text-danger">🔴 INACTIVO</span>';
                
                calendarInfo.innerHTML = `
                    <small class="text-muted">
                        📅 ${dayOfWeek.charAt(0).toUpperCase() + dayOfWeek.slice(1)} | Trading Multi-TF: ${tradingStatus} | Horario: 4am-4pm L-V
                    </small>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando información del calendario:', error);
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
        }
    }, 90000);
}

function updateCharts() {
    showLoadingState();
    
    const symbol = document.getElementById('symbol-select').value;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    // Actualizar gráfico principal
    updateMainChart(symbol, interval, leverage);
    
    // Actualizar señales de estrategias
    updateStrategySignals();
}

function updateMainChart(symbol, interval, leverage) {
    const url = `/api/signals?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    
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
            
            // Renderizar todos los gráficos
            renderCandleChart(data);
            renderAdxChart(data);
            renderStochRsiChart(data);
            renderTrendStrengthChart(data);
            renderVolumeChart(data);
            renderWhaleChart(data);
            renderRsiMaverickChart(data);
            renderRsiTraditionalChart(data);
            renderMacdChart(data);
            
            // Actualizar resúmenes
            updateMarketSummary(data);
            updateSignalAnalysis(data);
        })
        .catch(error => {
            console.error('Error cargando datos:', error);
            showError('Error al cargar datos: ' + error.message);
        });
}

function updateStrategySignals() {
    fetch('/api/strategy_signals')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.signals && data.signals.length > 0) {
                updateStrategiesTable(data.signals);
                updateSignalTables(data.signals);
            } else {
                resetSignalTables();
            }
        })
        .catch(error => {
            console.error('Error cargando señales de estrategias:', error);
            resetSignalTables();
        });
}

function updateStrategiesTable(signals) {
    const table = document.getElementById('strategies-table');
    const groupedSignals = {};
    
    // Agrupar señales por estrategia
    signals.forEach(signal => {
        if (!groupedSignals[signal.strategy]) {
            groupedSignals[signal.strategy] = { long: 0, short: 0 };
        }
        if (signal.signal === 'LONG') {
            groupedSignals[signal.strategy].long++;
        } else {
            groupedSignals[signal.strategy].short++;
        }
    });
    
    const strategies = Object.keys(groupedSignals).slice(0, 5);
    
    if (strategies.length > 0) {
        table.innerHTML = strategies.map((strategy, index) => {
            const counts = groupedSignals[strategy];
            const signalText = counts.long > counts.short ? 'LONG' : 'SHORT';
            const signalClass = signalText === 'LONG' ? 'success' : 'danger';
            
            return `
                <tr onclick="showStrategyDetails('${strategy}')" style="cursor: pointer;">
                    <td>${index + 1}</td>
                    <td><small>${strategy}</small></td>
                    <td class="text-center"><span class="badge bg-${signalClass}">${signalText}</span></td>
                </tr>
            `;
        }).join('');
    } else {
        table.innerHTML = `
            <tr>
                <td colspan="3" class="text-center py-3 text-muted">
                    No hay señales activas
                </td>
            </tr>
        `;
    }
}

function updateSignalTables(signals) {
    // Filtrar señales LONG
    const longSignals = signals.filter(s => s.signal === 'LONG').slice(0, 5);
    const shortSignals = signals.filter(s => s.signal === 'SHORT').slice(0, 5);
    
    // Actualizar tabla LONG
    const longTable = document.getElementById('long-table');
    if (longSignals.length > 0) {
        longTable.innerHTML = longSignals.map((signal, index) => `
            <tr onclick="showSignalDetailsModal('${signal.symbol}', '${signal.interval}', 'LONG')" style="cursor: pointer;">
                <td>${index + 1}</td>
                <td><strong>${signal.symbol.replace('-USDT', '')}</strong></td>
                <td><small>${signal.interval}</small></td>
            </tr>
        `).join('');
    } else {
        longTable.innerHTML = `
            <tr>
                <td colspan="3" class="text-center py-3 text-muted">
                    No hay señales LONG
                </td>
            </tr>
        `;
    }
    
    // Actualizar tabla SHORT
    const shortTable = document.getElementById('short-table');
    if (shortSignals.length > 0) {
        shortTable.innerHTML = shortSignals.map((signal, index) => `
            <tr onclick="showSignalDetailsModal('${signal.symbol}', '${signal.interval}', 'SHORT')" style="cursor: pointer;">
                <td>${index + 1}</td>
                <td><strong>${signal.symbol.replace('-USDT', '')}</strong></td>
                <td><small>${signal.interval}</small></td>
            </tr>
        `).join('');
    } else {
        shortTable.innerHTML = `
            <tr>
                <td colspan="3" class="text-center py-3 text-muted">
                    No hay señales SHORT
                </td>
            </tr>
        `;
    }
}

function resetSignalTables() {
    const tables = ['strategies-table', 'long-table', 'short-table'];
    tables.forEach(tableId => {
        const table = document.getElementById(tableId);
        table.innerHTML = `
            <tr>
                <td colspan="3" class="text-center py-3 text-muted">
                    Error cargando señales
                </td>
            </tr>
        `;
    });
}

function showStrategyDetails(strategy) {
    alert(`Detalles de la estrategia ${strategy}\n\nEsta funcionalidad está en desarrollo.`);
}

function showSignalDetailsModal(symbol, interval, signalType) {
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
    
    // Aquí puedes cargar los detalles específicos de la señal
    // Por ahora mostramos información básica
    setTimeout(() => {
        detailsElement.innerHTML = `
            <h6>Detalles de Señal - ${symbol}</h6>
            <div class="alert alert-${signalType === 'LONG' ? 'success' : 'danger'} text-center py-2 mb-3">
                <i class="fas fa-${signalType === 'LONG' ? 'arrow-up' : 'arrow-down'} me-2"></i>
                <strong>SEÑAL ${signalType} CONFIRMADA</strong>
                <div class="small mt-1">${symbol} - ${interval}</div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <h6>Información Básica</h6>
                    <div class="d-flex justify-content-between small mb-1">
                        <span>Símbolo:</span>
                        <span class="fw-bold">${symbol}</span>
                    </div>
                    <div class="d-flex justify-content-between small mb-1">
                        <span>Temporalidad:</span>
                        <span class="fw-bold">${interval}</span>
                    </div>
                    <div class="d-flex justify-content-between small mb-1">
                        <span>Señal:</span>
                        <span class="fw-bold text-${signalType === 'LONG' ? 'success' : 'danger'}">${signalType}</span>
                    </div>
                </div>
                <div class="col-md-6">
                    <h6>Acciones</h6>
                    <div class="d-grid gap-2">
                        <button class="btn btn-primary" onclick="selectSymbol('${symbol}')">
                            <i class="fas fa-chart-line me-1"></i>Ver Gráficos
                        </button>
                        <button class="btn btn-success" onclick="downloadReport()">
                            <i class="fas fa-download me-1"></i>Descargar Reporte
                        </button>
                    </div>
                </div>
            </div>
        `;
    }, 1000);
}

function selectSymbol(symbol) {
    document.getElementById('symbol-select').value = symbol;
    updateCharts();
    bootstrap.Modal.getInstance(document.getElementById('signalModal')).hide();
}

// Función para renderizar gráfico de velas
function renderCandleChart(data) {
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
    
    // Añadir indicadores si están activados
    const showMA9 = document.getElementById('show-ma9').checked;
    const showMA21 = document.getElementById('show-ma21').checked;
    const showMA50 = document.getElementById('show-ma50').checked;
    const showMA200 = document.getElementById('show-ma200').checked;
    const showBB = document.getElementById('show-bollinger').checked;
    
    if (showMA9 && data.indicators && data.indicators.ma_9) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_9,
            mode: 'lines',
            line: {color: '#FF9800', width: 1},
            name: 'MA 9'
        });
    }
    
    if (showMA21 && data.indicators && data.indicators.ma_21) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_21,
            mode: 'lines',
            line: {color: '#2196F3', width: 1},
            name: 'MA 21'
        });
    }
    
    if (showMA50 && data.indicators && data.indicators.ma_50) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_50,
            mode: 'lines',
            line: {color: '#9C27B0', width: 1},
            name: 'MA 50'
        });
    }
    
    if (showMA200 && data.indicators && data.indicators.ma_200) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_200,
            mode: 'lines',
            line: {color: '#795548', width: 3},
            name: 'MA 200'
        });
    }
    
    if (showBB && data.indicators && data.indicators.bb_upper && data.indicators.bb_lower) {
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
    
    // Añadir niveles de trading si existen
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
            line: {color: '#FF0000', dash: 'dash', width: 2},
            name: 'Stop Loss'
        });
        
        if (data.take_profit && data.take_profit.length > 0) {
            data.take_profit.slice(0, 3).forEach((tp, index) => {
                traces.push({
                    type: 'scatter',
                    x: [dates[0], dates[dates.length - 1]],
                    y: [tp, tp],
                    mode: 'lines',
                    line: {color: '#00FF00', dash: 'dash', width: 1.5},
                    name: `TP${index + 1}`,
                    showlegend: index === 0
                });
            });
        }
    }
    
    const interval = document.getElementById('interval-select').value;
    const symbol = document.getElementById('symbol-select').value;
    
    const layout = {
        title: {
            text: `${symbol} - ${interval}`,
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
        modeBarButtonsToRemove: ['pan2d', 'lasso2d']
    };
    
    // Destruir gráfico existente
    if (currentChart) {
        Plotly.purge('candle-chart');
    }
    
    currentChart = Plotly.newPlot('candle-chart', traces, layout, config);
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
            y: adx.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'ADX',
            line: {color: 'white', width: 2}
        },
        {
            x: dates,
            y: plusDi.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: '+DI',
            line: {color: '#00C853', width: 1.5}
        },
        {
            x: dates,
            y: minusDi.slice(-50),
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

function renderStochRsiChart(data) {
    const chartElement = document.getElementById('stoch-rsi-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de RSI Estocástico disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const stochRsi = data.indicators.stoch_rsi || [];
    const kLine = data.indicators.stoch_k || [];
    const dLine = data.indicators.stoch_d || [];
    
    const traces = [
        {
            x: dates,
            y: stochRsi.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Estocástico',
            line: {color: '#2196F3', width: 2}
        },
        {
            x: dates,
            y: kLine.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: '%K',
            line: {color: '#00C853', width: 1}
        },
        {
            x: dates,
            y: dLine.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: '%D',
            line: {color: '#FF1744', width: 1}
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Estocástico (%K y %D)',
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
                line: {color: 'red', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 20,
                y1: 20,
                line: {color: 'green', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 50,
                y1: 50,
                line: {color: 'white', width: 1, dash: 'solid'}
            }
        ],
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
    
    if (currentStochRsiChart) {
        Plotly.purge('stoch-rsi-chart');
    }
    
    currentStochRsiChart = Plotly.newPlot('stoch-rsi-chart', traces, layout, config);
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
    const colors = data.indicators.colors || [];
    
    const traces = [{
        x: dates,
        y: trendStrength.slice(-50),
        type: 'bar',
        name: 'Fuerza de Tendencia',
        marker: {
            color: colors.slice(-50),
            line: {
                color: 'rgba(255,255,255,0.3)',
                width: 0.5
            }
        }
    }];
    
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
            title: 'Fuerza de Tendencia %',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 60, r: 50, b: 50, l: 50}
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

function renderVolumeChart(data) {
    const chartElement = document.getElementById('volume-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Volumen disponibles</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const volumes = data.data.slice(-50).map(d => parseFloat(d.volume));
    const volumeSignal = data.indicators.volume_signal || [];
    
    // Colorear barras según señal de volumen
    const volumeColors = volumes.map((vol, i) => {
        const signal = volumeSignal[i] || 'NEUTRAL';
        if (signal === 'COMPRA') return '#00C853';
        if (signal === 'VENTA') return '#FF1744';
        return 'rgba(128, 128, 128, 0.7)';
    });
    
    const traces = [{
        x: dates,
        y: volumes,
        type: 'bar',
        name: 'Volumen',
        marker: {color: volumeColors}
    }];
    
    const layout = {
        title: {
            text: 'Indicador de Volumen con Anomalías',
            font: {color: '#ffffff', size: 14}
        },
        xaxis: {
            title: 'Fecha/Hora',
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Volumen',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 60, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentVolumeChart) {
        Plotly.purge('volume-chart');
    }
    
    currentVolumeChart = Plotly.newPlot('volume-chart', traces, layout, config);
}

function renderWhaleChart(data) {
    const chartElement = document.getElementById('whale-chart');
    
    if (!data.indicators || !data.data) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center">
                <p class="mb-0">No hay datos de Ballenas disponibles</p>
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
            y: whalePump.slice(-50),
            type: 'bar',
            name: 'Ballenas Compradoras',
            marker: {color: '#00C853', opacity: 0.7}
        },
        {
            x: dates,
            y: whaleDump.slice(-50),
            type: 'bar',
            name: 'Ballenas Vendedoras',
            marker: {color: '#FF1744', opacity: 0.7}
        }
    ];
    
    const layout = {
        title: {
            text: 'Indicador Ballenas Compradoras/Vendedoras',
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
    
    const traces = [{
        x: dates,
        y: rsiMaverick.slice(-50),
        type: 'scatter',
        mode: 'lines',
        name: 'RSI Maverick (%B)',
        line: {color: '#FF6B6B', width: 2}
    }];
    
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
                line: {color: 'red', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 0.2,
                y1: 0.2,
                line: {color: 'green', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 0.5,
                y1: 0.5,
                line: {color: 'white', width: 1, dash: 'solid'}
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 60, r: 50, b: 50, l: 50}
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
    
    const traces = [{
        x: dates,
        y: rsiTraditional.slice(-50),
        type: 'scatter',
        mode: 'lines',
        name: 'RSI Tradicional',
        line: {color: '#2196F3', width: 2}
    }];
    
    const layout = {
        title: {
            text: 'RSI Tradicional (14 Periodos)',
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
                line: {color: 'red', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 20,
                y1: 20,
                line: {color: 'green', width: 1, dash: 'dash'}
            },
            {
                type: 'line',
                x0: dates[0],
                x1: dates[dates.length - 1],
                y0: 50,
                y1: 50,
                line: {color: 'white', width: 1, dash: 'solid'}
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 60, r: 50, b: 50, l: 50}
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
    const histogramColors = macdHistogram.slice(-50).map(value => 
        value >= 0 ? 'rgba(0, 200, 83, 0.8)' : 'rgba(255, 23, 68, 0.8)'
    );
    
    const traces = [
        {
            x: dates,
            y: macd.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: {color: '#2196F3', width: 2}
        },
        {
            x: dates,
            y: macdSignal.slice(-50),
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#FF9800', width: 1.5}
        },
        {
            x: dates,
            y: macdHistogram.slice(-50),
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

function updateMarketSummary(data) {
    if (!data) return;
    
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
                    <span class="fw-bold ${data.signal_score >= 70 ? 'text-success' : data.signal_score >= 65 ? 'text-warning' : 'text-danger'}">
                        ${data.signal_score ? data.signal_score.toFixed(1) : '0'}%
                    </span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Precio:</span>
                    <span class="fw-bold">$${data.current_price ? data.current_price.toFixed(6) : '0.000000'}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">Volumen:</span>
                    <span class="badge bg-info">${data.volume ? (data.volume / 1000000).toFixed(1) : '0'}M</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">ADX:</span>
                    <span class="${data.adx >= 25 ? 'text-success' : 'text-warning'}">${data.adx ? data.adx.toFixed(1) : '0.0'}</span>
                </div>
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small">RSI:</span>
                    <span class="${data.rsi_traditional >= 70 ? 'text-danger' : data.rsi_traditional <= 30 ? 'text-success' : 'text-warning'}">
                        ${data.rsi_traditional ? data.rsi_traditional.toFixed(1) : '0.0'}
                    </span>
                </div>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="text-muted small">Stoch RSI:</span>
                    <span class="${data.stoch_rsi >= 80 ? 'text-danger' : data.stoch_rsi <= 20 ? 'text-success' : 'text-warning'}">
                        ${data.stoch_rsi ? data.stoch_rsi.toFixed(1) : '0.0'}
                    </span>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    if (!data) return;
    
    const signalAnalysis = document.getElementById('signal-analysis');
    
    if (data.signal === 'NEUTRAL' || !data.signal_score || data.signal_score < 65) {
        signalAnalysis.innerHTML = `
            <div class="alert alert-secondary text-center py-2">
                <i class="fas fa-pause-circle me-2"></i>
                <strong>SEÑAL NEUTRAL</strong>
                <div class="small mt-1">Score: ${data.signal_score ? data.signal_score.toFixed(1) : '0'}%</div>
                <div class="small text-muted">Esperando mejores condiciones</div>
            </div>
        `;
    } else {
        const signalClass = data.signal === 'LONG' ? 'success' : 'danger';
        const signalIcon = data.signal === 'LONG' ? 'arrow-up' : 'arrow-down';
        
        signalAnalysis.innerHTML = `
            <div class="alert alert-${signalClass} text-center py-2">
                <i class="fas fa-${signalIcon} me-2"></i>
                <strong>SEÑAL ${data.signal} CONFIRMADA</strong>
                <div class="small mt-1">Score: ${data.signal_score.toFixed(1)}%</div>
            </div>
            
            <div class="mt-2">
                <div class="d-flex justify-content-between small mb-1">
                    <span>Entrada:</span>
                    <span class="fw-bold">$${data.entry ? data.entry.toFixed(6) : '0.000000'}</span>
                </div>
                <div class="d-flex justify-content-between small mb-1">
                    <span>Stop Loss:</span>
                    <span class="fw-bold text-danger">$${data.stop_loss ? data.stop_loss.toFixed(6) : '0.000000'}</span>
                </div>
                <div class="d-flex justify-content-between small">
                    <span>Take Profit 1:</span>
                    <span class="fw-bold text-success">$${data.take_profit && data.take_profit[0] ? data.take_profit[0].toFixed(6) : '0.000000'}</span>
                </div>
            </div>
            
            ${data.filters && data.filters.length > 0 ? `
                <div class="mt-2">
                    <small class="text-muted d-block mb-1">Filtros activos:</small>
                    <div style="max-height: 100px; overflow-y: auto;">
                        ${data.filters.slice(0, 3).map(filter => `
                            <div class="small text-success mb-1">✓ ${filter}</div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }
}

function showError(message) {
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
    
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function updateChartIndicators() {
    const symbol = document.getElementById('symbol-select').value;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    updateMainChart(symbol, interval, leverage);
}

// Cargar el orden de indicadores al iniciar
window.addEventListener('load', function() {
    loadIndicatorOrder();
});
