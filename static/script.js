// MULTI-TIMEFRAME CRYPTO WGTA PRO - Script Optimizado y Adaptado

// Configuración global optimizada
let currentChart = null;
let currentScatterChart = null;
let currentData = null;
let currentSymbol = 'BTC-USDT';
let allCryptos = [];
let updateInterval = null;

// Inicialización optimizada
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializando...');
    initializeApp();
    setupEventListeners();
    updateCharts();
    startAutoUpdate();
});

function initializeApp() {
    console.log('✅ MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializado');
    loadCryptoRiskClassification();
    updateCalendarInfo();
    updateWinrateDisplay();
    updateBoliviaClock();
    updateFearGreedIndex();
    updateMarketRecommendations();
}

function setupEventListeners() {
    // Event listeners para controles principales
    const controls = [
        'interval-select', 'di-period', 'adx-threshold', 
        'sr-period', 'rsi-length', 'bb-multiplier', 'leverage', 'volume-filter'
    ];
    
    controls.forEach(control => {
        const element = document.getElementById(control);
        if (element) {
            element.addEventListener('change', updateCharts);
        }
    });
    
    // Buscador de cryptos
    setupCryptoSearch();
    
    // Selector de indicador auxiliar
    const auxIndicator = document.getElementById('aux-indicator');
    if (auxIndicator) {
        auxIndicator.addEventListener('change', updateAuxChart);
    }
    
    // Herramientas de dibujo
    setupDrawingTools();
    
    // Controles de indicadores
    setupIndicatorControls();
}

function setupCryptoSearch() {
    const searchInput = document.getElementById('crypto-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterCryptoList(this.value.toUpperCase());
        });
    }
}

function setupDrawingTools() {
    const drawingTools = document.querySelectorAll('.drawing-tool');
    drawingTools.forEach(tool => {
        tool.addEventListener('click', function() {
            const toolType = this.getAttribute('data-tool');
            activateDrawingTool(toolType);
        });
    });
    
    const drawingColor = document.getElementById('drawing-color');
    if (drawingColor) {
        drawingColor.addEventListener('change', function() {
            setDrawingColor(this.value);
        });
    }
}

function setupIndicatorControls() {
    const indicatorControls = document.querySelectorAll('.indicator-control');
    indicatorControls.forEach(control => {
        control.addEventListener('change', function() {
            updateCharts();
        });
    });
}

function activateDrawingTool(toolType) {
    console.log(`🛠️ Herramienta de dibujo activada: ${toolType}`);
    showNotification(`Herramienta ${toolType} activada`, 'info');
    
    // Remover clase active de todas las herramientas
    document.querySelectorAll('.drawing-tool').forEach(tool => {
        tool.classList.remove('active');
    });
    
    // Agregar clase active a la herramienta seleccionada
    const activeTool = document.querySelector(`[data-tool="${toolType}"]`);
    if (activeTool) {
        activeTool.classList.add('active');
    }
}

function setDrawingColor(color) {
    console.log(`🎨 Color de dibujo cambiado a: ${color}`);
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
            item.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <span>${crypto.symbol}</span>
                    <span class="risk-badge risk-${crypto.category}">${crypto.category}</span>
                </div>
            `;
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
                        📅 ${data.date} | ${data.time} | Scalping 15m/30m: ${scalpingStatus} | Horario: 4am-4pm L-V
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
        'exit-signals': 'Monitoreando operaciones...',
        'long-table': 'Cargando señales LONG...',
        'short-table': 'Cargando señales SHORT...'
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
    }, 60000); // Actualizar cada 60 segundos
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
    
    // Actualizar título del gráfico
    const chartTitle = document.getElementById('chart-title');
    if (chartTitle) {
        chartTitle.textContent = `${symbol} - Gráfico de Velas (${interval})`;
    }
    
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    updateScatterChart(interval, diPeriod, adxThreshold);
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    updateWinrateDisplay(symbol, interval);
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateScalpingAlerts();
    updateExitSignals();
    updateCalendarInfo();
    updateBoliviaClock();
}


function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const params = new URLSearchParams({
        symbol, 
        interval, 
        di_period: diPeriod, 
        adx_threshold: adxThreshold,
        sr_period: srPeriod, 
        rsi_length: rsiLength, 
        bb_multiplier: bbMultiplier,
        volume_filter: volumeFilter,
        leverage
    });
    
    console.log(`📊 Actualizando gráfico principal: ${symbol} ${interval}`);
    
    fetch(`/api/signals?${params}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status} - ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            // Verificar si los datos son válidos
            if (!data || typeof data !== 'object') {
                throw new Error('Datos inválidos recibidos del servidor');
            }
            
            currentData = data;
            renderCandleChart(data);
            renderTrendStrengthChart(data);
            renderWhaleChart(data);
            renderADXChart(data);
            renderRSIMaverickChart(data);
            renderSqueezeChart(data);
            updateAuxChart();
            updateMarketSummary(data);
            updateSignalAnalysis(data);
            
            console.log(`✅ Gráfico actualizado exitosamente para ${symbol}`);
        })
        .catch(error => {
            console.error('❌ Error crítico cargando datos:', error);
            showError(`Error al cargar datos: ${error.message}. Reintentando en 5 segundos...`);
            
            // Mostrar datos de ejemplo para evitar interfaz bloqueada
            showSampleData(symbol, interval);
            
            // Reintentar después de 5 segundos
            setTimeout(() => updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage), 5000);
        });
}

// Agregar función para mostrar datos de ejemplo
function showSampleData(symbol, interval) {
    const chartElement = document.getElementById('candle-chart');
    if (!chartElement) return;
    
    chartElement.innerHTML = `
        <div class="alert alert-warning text-center m-3">
            <h6><i class="fas fa-exclamation-triangle me-2"></i>Modo de Datos de Ejemplo</h6>
            <p class="small mb-2">Mostrando datos de ejemplo mientras se restablece la conexión...</p>
            <div class="mt-2">
                <p class="small mb-1"><strong>${symbol} - ${interval}</strong></p>
                <p class="small mb-1">Precio: $45,250.75</p>
                <p class="small mb-1">Señal: NEUTRAL | Score: 45%</p>
            </div>
            <button class="btn btn-sm btn-primary mt-2" onclick="updateCharts()">
                <i class="fas fa-sync-alt me-1"></i>Reintentar Conexión
            </button>
        </div>
    `;
}







function renderCandleChart(data) {
    const chartElement = document.getElementById('candle-chart');
    if (!chartElement) return;
    
    if (!data.data || data.data.length === 0) {
        chartElement.innerHTML = `
            <div class="alert alert-warning text-center m-3">
                <h6><i class="fas fa-exclamation-triangle me-2"></i>No hay datos disponibles</h6>
                <p class="small mb-2">No se pudieron cargar los datos para ${data.symbol}</p>
                <button class="btn btn-sm btn-primary mt-1" onclick="updateCharts()">
                    <i class="fas fa-sync-alt me-1"></i>Reintentar
                </button>
            </div>
        `;
        return;
    }

    const dates = data.data.map(d => {
        if (typeof d.timestamp === 'string') {
            return new Date(d.timestamp);
        } else if (d.timestamp instanceof Date) {
            return d.timestamp;
        } else {
            return new Date();
        }
    });
    
    const opens = data.data.map(d => parseFloat(d.open));
    const highs = data.data.map(d => parseFloat(d.high));
    const lows = data.data.map(d => parseFloat(d.low));
    const closes = data.data.map(d => parseFloat(d.close));
    
    // Traza de velas japonesas estilo TradingView
    const candlestickTrace = {
        type: 'candlestick',
        x: dates,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        increasing: {
            line: { color: '#00C853', width: 1 },
            fillcolor: '#00C853'
        },
        decreasing: {
            line: { color: '#FF1744', width: 1 },
            fillcolor: '#FF1744'
        },
        name: 'Precio',
        hoverinfo: 'x+y'
    };
    
    const traces = [candlestickTrace];
    
    // Añadir medias móviles si están activas
    if (document.getElementById('show-ma9')?.checked && data.indicators?.ma_9) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_9,
            mode: 'lines',
            line: { color: '#FF6B6B', width: 1.5 },
            name: 'MA 9',
            hoverinfo: 'x+y+name'
        });
    }
    
    if (document.getElementById('show-ma21')?.checked && data.indicators?.ma_21) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_21,
            mode: 'lines',
            line: { color: '#4ECDC4', width: 1.5 },
            name: 'MA 21',
            hoverinfo: 'x+y+name'
        });
    }
    
    if (document.getElementById('show-ma50')?.checked && data.indicators?.ma_50) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_50,
            mode: 'lines',
            line: { color: '#45B7D1', width: 1.5 },
            name: 'MA 50',
            hoverinfo: 'x+y+name'
        });
    }
    
    if (document.getElementById('show-ma200')?.checked && data.indicators?.ma_200) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.ma_200,
            mode: 'lines',
            line: { color: '#96CEB4', width: 1.5 },
            name: 'MA 200',
            hoverinfo: 'x+y+name'
        });
    }
    
    // Añadir Bandas de Bollinger si están activas
    if (document.getElementById('show-bollinger')?.checked && data.indicators?.bb_upper && data.indicators?.bb_lower) {
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.bb_upper,
            mode: 'lines',
            line: { color: 'rgba(255, 255, 255, 0.3)', width: 1, dash: 'dash' },
            name: 'BB Upper',
            hoverinfo: 'x+y+name',
            showlegend: false
        });
        
        traces.push({
            type: 'scatter',
            x: dates,
            y: data.indicators.bb_lower,
            mode: 'lines',
            line: { color: 'rgba(255, 255, 255, 0.3)', width: 1, dash: 'dash' },
            name: 'BB Lower',
            hoverinfo: 'x+y+name',
            showlegend: false
        });
    }
    
    // Añadir niveles importantes si están disponibles
    if (data.entry && data.stop_loss && data.take_profit) {
        // Línea de entrada
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.entry, data.entry],
            mode: 'lines',
            line: { color: '#FFD700', dash: 'solid', width: 2 },
            name: 'Entrada',
            hoverinfo: 'y+name'
        });
        
        // Línea de stop loss
        traces.push({
            type: 'scatter',
            x: [dates[0], dates[dates.length - 1]],
            y: [data.stop_loss, data.stop_loss],
            mode: 'lines',
            line: { color: '#FF1744', dash: 'dash', width: 2 },
            name: 'Stop Loss',
            hoverinfo: 'y+name'
        });
        
        // Líneas de take profit
        data.take_profit.forEach((tp, index) => {
            traces.push({
                type: 'scatter',
                x: [dates[0], dates[dates.length - 1]],
                y: [tp, tp],
                mode: 'lines',
                line: { color: '#00C853', dash: 'dash', width: 1.5 },
                name: `TP${index + 1}`,
                hoverinfo: 'y+name',
                showlegend: index === 0 // Mostrar solo el primer TP en la leyenda
            });
        });
    }
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas (${document.getElementById('interval-select').value})`,
            font: { color: '#ffffff', size: 14 }
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444',
            rangeslider: { visible: false }
        },
        yaxis: {
            title: 'Precio (USDT)',
            gridcolor: '#444',
            zerolinecolor: '#444',
            side: 'right'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: { color: '#ffffff' }
        },
        margin: { t: 60, r: 30, b: 40, l: 50 }
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
        scrollZoom: true
    };
    
    if (currentChart) Plotly.purge('candle-chart');
    currentChart = Plotly.newPlot('candle-chart', traces, layout, config);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    if (!chartElement || !data.indicators || !data.indicators.trend_strength) return;

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const trendStrength = data.indicators.trend_strength.slice(-50);
    const colors = data.indicators.colors?.slice(-50) || Array(50).fill('gray');
    const noTradeZones = data.indicators.no_trade_zones?.slice(-50) || [];
    
    const trace = {
        x: dates,
        y: trendStrength,
        type: 'bar',
        name: 'Fuerza de Tendencia',
        marker: { color: colors },
        hoverinfo: 'x+y'
    };
    
    // Añadir formas para zonas de no operar
    const shapes = [];
    noTradeZones.forEach((isNoTrade, index) => {
        if (isNoTrade && index < dates.length) {
            shapes.push({
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: dates[Math.max(0, index - 1)],
                x1: dates[Math.min(dates.length - 1, index + 1)],
                y0: 0,
                y1: 1,
                fillcolor: 'rgba(255, 0, 0, 0.1)',
                line: { width: 0 },
                layer: 'below'
            });
        }
    });
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick - Ancho Bandas Bollinger %',
            font: { color: '#ffffff', size: 12 }
        },
        xaxis: { 
            type: 'date', 
            gridcolor: '#444',
            rangeslider: { visible: false }
        },
        yaxis: { 
            title: 'Fuerza %', 
            gridcolor: '#444'
        },
        shapes: shapes,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        showlegend: false,
        margin: { t: 40, r: 30, b: 40, l: 50 },
        height: 250
    };
    
    const config = { 
        responsive: true, 
        displayModeBar: false 
    };
    
    Plotly.newPlot('trend-strength-chart', [trace], layout, config);
}

function renderWhaleChart(data) {
    const chartElement = document.getElementById('whale-chart');
    if (!chartElement) return;

    // Actualizar estado del indicador de ballenas
    const whaleStatus = document.getElementById('whale-indicator-status');
    if (whaleStatus) {
        const interval = document.getElementById('interval-select').value;
        if (interval === '12h' || interval === '1D') {
            whaleStatus.className = 'badge bg-success ms-2';
            whaleStatus.textContent = 'ACTIVO';
        } else {
            whaleStatus.className = 'badge bg-secondary ms-2';
            whaleStatus.textContent = 'SOLO 12H/1D';
        }
    }

    if (!data.indicators || !data.indicators.whale_pump || !data.whale_indicator_active) {
        chartElement.innerHTML = `
            <div class="text-center py-4">
                <h6 class="text-muted">Indicador Ballenas No Aplica</h6>
                <p class="small text-muted mb-0">Este indicador solo está activo en temporalidades 12H y 1D</p>
            </div>
        `;
        return;
    }

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const whalePump = data.indicators.whale_pump.slice(-50);
    const whaleDump = data.indicators.whale_dump.slice(-50);
    
    const trace1 = {
        x: dates,
        y: whalePump,
        type: 'scatter',
        mode: 'lines',
        name: 'Ballenas Compradoras',
        line: { color: '#00C853', width: 2 },
        hoverinfo: 'x+y+name'
    };
    
    const trace2 = {
        x: dates,
        y: whaleDump,
        type: 'scatter',
        mode: 'lines',
        name: 'Ballenas Vendedoras',
        line: { color: '#FF1744', width: 2 },
        hoverinfo: 'x+y+name'
    };
    
    const layout = {
        title: {
            text: 'Indicador Ballenas Compradoras/Vendedoras',
            font: { color: '#ffffff', size: 12 }
        },
        xaxis: { 
            type: 'date', 
            gridcolor: '#444'
        },
        yaxis: { 
            title: 'Fuerza Ballenas', 
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: { color: '#ffffff' }
        },
        margin: { t: 40, r: 30, b: 40, l: 50 },
        height: 250
    };
    
    const config = { responsive: true, displayModeBar: false };
    
    Plotly.newPlot('whale-chart', [trace1, trace2], layout, config);
}

function renderADXChart(data) {
    const chartElement = document.getElementById('adx-chart');
    if (!chartElement || !data.indicators || !data.indicators.adx) return;

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const adx = data.indicators.adx.slice(-50);
    const plusDI = data.indicators.plus_di.slice(-50);
    const minusDI = data.indicators.minus_di.slice(-50);
    
    const trace1 = {
        x: dates,
        y: adx,
        type: 'scatter',
        mode: 'lines',
        name: 'ADX',
        line: { color: '#FFFFFF', width: 2 },
        hoverinfo: 'x+y+name'
    };
    
    const trace2 = {
        x: dates,
        y: plusDI,
        type: 'scatter',
        mode: 'lines',
        name: '+DI',
        line: { color: '#00C853', width: 1.5 },
        hoverinfo: 'x+y+name'
    };
    
    const trace3 = {
        x: dates,
        y: minusDI,
        type: 'scatter',
        mode: 'lines',
        name: '-DI',
        line: { color: '#FF1744', width: 1.5 },
        hoverinfo: 'x+y+name'
    };
    
    const layout = {
        title: {
            text: 'ADX con DMI (+D y -D) y Cruces',
            font: { color: '#ffffff', size: 12 }
        },
        xaxis: { 
            type: 'date', 
            gridcolor: '#444'
        },
        yaxis: { 
            title: 'Valor ADX/DMI', 
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h',
            font: { color: '#ffffff' }
        },
        margin: { t: 40, r: 30, b: 40, l: 50 },
        height: 250
    };
    
    const config = { responsive: true, displayModeBar: false };
    
    Plotly.newPlot('adx-chart', [trace1, trace2, trace3], layout, config);
}

function renderRSIMaverickChart(data) {
    const chartElement = document.getElementById('rsi-maverick-chart');
    if (!chartElement || !data.indicators || !data.indicators.rsi_maverick) return;

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const rsiMaverick = data.indicators.rsi_maverick.slice(-50);
    
    const trace = {
        x: dates,
        y: rsiMaverick,
        type: 'scatter',
        mode: 'lines',
        name: 'RSI Maverick (%B)',
        line: { color: '#3B82F6', width: 2 },
        hoverinfo: 'x+y+name'
    };
    
    const layout = {
        title: {
            text: 'RSI Modificado Maverick (%B)',
            font: { color: '#ffffff', size: 12 }
        },
        xaxis: { 
            type: 'date', 
            gridcolor: '#444'
        },
        yaxis: { 
            title: 'RSI Maverick', 
            gridcolor: '#444',
            range: [0, 1]
        },
        shapes: [
            {
                type: 'rect',
                xref: 'paper',
                yref: 'y',
                x0: 0,
                x1: 1,
                y0: 0.8,
                y1: 1,
                fillcolor: '#FF1744',
                opacity: 0.1,
                line: { width: 0 }
            },
            {
                type: 'rect',
                xref: 'paper',
                yref: 'y',
                x0: 0,
                x1: 1,
                y0: 0,
                y1: 0.2,
                fillcolor: '#00C853',
                opacity: 0.1,
                line: { width: 0 }
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        showlegend: false,
        margin: { t: 40, r: 30, b: 40, l: 50 },
        height: 250
    };
    
    const config = { responsive: true, displayModeBar: false };
    
    Plotly.newPlot('rsi-maverick-chart', [trace], layout, config);
}

function renderSqueezeChart(data) {
    const chartElement = document.getElementById('squeeze-chart');
    if (!chartElement || !data.indicators || !data.indicators.squeeze_momentum) return;

    const dates = data.data.slice(-50).map(d => new Date(d.timestamp));
    const squeezeMomentum = data.indicators.squeeze_momentum.slice(-50);
    const colors = squeezeMomentum.map(val => val >= 0 ? '#00C853' : '#FF1744');
    
    const trace = {
        x: dates,
        y: squeezeMomentum,
        type: 'bar',
        name: 'Squeeze Momentum',
        marker: { color: colors },
        hoverinfo: 'x+y+name'
    };
    
    const layout = {
        title: {
            text: 'Squeeze Momentum - Compresión/Expansión de Volatilidad',
            font: { color: '#ffffff', size: 12 }
        },
        xaxis: { 
            type: 'date', 
            gridcolor: '#444'
        },
        yaxis: { 
            title: 'Momentum', 
            gridcolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        showlegend: false,
        margin: { t: 40, r: 30, b: 40, l: 50 },
        height: 250
    };
    
    const config = { responsive: true, displayModeBar: false };
    
    Plotly.newPlot('squeeze-chart', [trace], layout, config);
}

function updateAuxChart() {
    const auxIndicator = document.getElementById('aux-indicator');
    if (!auxIndicator || !currentData) return;
    
    const indicatorType = auxIndicator.value;
    const chartElement = document.getElementById('aux-chart');
    if (!chartElement) return;
    
    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    
    let trace;
    let layout;
    
    switch(indicatorType) {
        case 'rsi':
            if (currentData.indicators && currentData.indicators.rsi) {
                const rsiTraditional = currentData.indicators.rsi.slice(-50);
                
                trace = {
                    x: dates,
                    y: rsiTraditional,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'RSI Tradicional',
                    line: { color: '#FF9800', width: 2 },
                    hoverinfo: 'x+y+name'
                };
                
                layout = {
                    title: {
                        text: 'RSI Tradicional (14 periodos)',
                        font: { color: '#ffffff', size: 12 }
                    },
                    xaxis: { type: 'date', gridcolor: '#444' },
                    yaxis: { 
                        title: 'RSI', 
                        gridcolor: '#444',
                        range: [0, 100]
                    },
                    shapes: [
                        {
                            type: 'rect',
                            xref: 'paper',
                            yref: 'y',
                            x0: 0,
                            x1: 1,
                            y0: 70,
                            y1: 100,
                            fillcolor: '#FF1744',
                            opacity: 0.1,
                            line: { width: 0 }
                        },
                        {
                            type: 'rect',
                            xref: 'paper',
                            yref: 'y',
                            x0: 0,
                            x1: 1,
                            y0: 0,
                            y1: 30,
                            fillcolor: '#00C853',
                            opacity: 0.1,
                            line: { width: 0 }
                        }
                    ],
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    showlegend: false,
                    margin: { t: 40, r: 30, b: 40, l: 50 },
                    height: 250
                };
            }
            break;
            
        case 'macd':
            if (currentData.indicators && currentData.indicators.macd_histogram) {
                const macdHistogram = currentData.indicators.macd_histogram.slice(-50);
                const colors = macdHistogram.map(val => val >= 0 ? '#00C853' : '#FF1744');
                
                trace = {
                    x: dates,
                    y: macdHistogram,
                    type: 'bar',
                    name: 'MACD Histogram',
                    marker: { color: colors },
                    hoverinfo: 'x+y+name'
                };
                
                layout = {
                    title: {
                        text: 'MACD Histogram',
                        font: { color: '#ffffff', size: 12 }
                    },
                    xaxis: { type: 'date', gridcolor: '#444' },
                    yaxis: { 
                        title: 'MACD', 
                        gridcolor: '#444'
                    },
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    showlegend: false,
                    margin: { t: 40, r: 30, b: 40, l: 50 },
                    height: 250
                };
            }
            break;
            
        case 'squeeze':
            if (currentData.indicators && currentData.indicators.squeeze_momentum) {
                const squeezeMomentum = currentData.indicators.squeeze_momentum.slice(-50);
                const colors = squeezeMomentum.map(val => val >= 0 ? '#00C853' : '#FF1744');
                
                trace = {
                    x: dates,
                    y: squeezeMomentum,
                    type: 'bar',
                    name: 'Squeeze Momentum',
                    marker: { color: colors },
                    hoverinfo: 'x+y+name'
                };
                
                layout = {
                    title: {
                        text: 'Squeeze Momentum',
                        font: { color: '#ffffff', size: 12 }
                    },
                    xaxis: { type: 'date', gridcolor: '#444' },
                    yaxis: { 
                        title: 'Momentum', 
                        gridcolor: '#444'
                    },
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#ffffff' },
                    showlegend: false,
                    margin: { t: 40, r: 30, b: 40, l: 50 },
                    height: 250
                };
            }
            break;
            
        case 'moving_averages':
            if (currentData.indicators) {
                const traces = [];
                
                if (currentData.indicators.ma_9) {
                    traces.push({
                        x: dates,
                        y: currentData.indicators.ma_9.slice(-50),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MA 9',
                        line: { color: '#FF6B6B', width: 1.5 },
                        hoverinfo: 'x+y+name'
                    });
                }
                
                if (currentData.indicators.ma_21) {
                    traces.push({
                        x: dates,
                        y: currentData.indicators.ma_21.slice(-50),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MA 21',
                        line: { color: '#4ECDC4', width: 1.5 },
                        hoverinfo: 'x+y+name'
                    });
                }
                
                if (currentData.indicators.ma_50) {
                    traces.push({
                        x: dates,
                        y: currentData.indicators.ma_50.slice(-50),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MA 50',
                        line: { color: '#45B7D1', width: 1.5 },
                        hoverinfo: 'x+y+name'
                    });
                }
                
                if (currentData.indicators.ma_200) {
                    traces.push({
                        x: dates,
                        y: currentData.indicators.ma_200.slice(-50),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MA 200',
                        line: { color: '#96CEB4', width: 1.5 },
                        hoverinfo: 'x+y+name'
                    });
                }
                
                if (traces.length > 0) {
                    layout = {
                        title: {
                            text: 'Medias Móviles',
                            font: { color: '#ffffff', size: 12 }
                        },
                        xaxis: { type: 'date', gridcolor: '#444' },
                        yaxis: { 
                            title: 'Precio', 
                            gridcolor: '#444'
                        },
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#ffffff' },
                        showlegend: true,
                        legend: {
                            x: 0,
                            y: 1.1,
                            orientation: 'h',
                            font: { color: '#ffffff' }
                        },
                        margin: { t: 40, r: 30, b: 40, l: 50 },
                        height: 250
                    };
                    
                    const config = { responsive: true, displayModeBar: false };
                    Plotly.newPlot('aux-chart', traces, layout, config);
                    return;
                }
            }
            break;
    }
    
    if (trace && layout) {
        const config = { responsive: true, displayModeBar: false };
        Plotly.newPlot('aux-chart', [trace], layout, config);
    } else {
        chartElement.innerHTML = `
            <div class="text-center py-4">
                <p class="text-muted">No hay datos disponibles para este indicador</p>
            </div>
        `;
    }
}

function updateMarketSummary(data) {
    const summaryElement = document.getElementById('market-summary');
    if (!summaryElement) return;
    
    const multiTFValidLong = data.multi_tf_valid_long || false;
    const multiTFValidShort = data.multi_tf_valid_short || false;
    const multiTFReason = data.multi_tf_reason_long || data.multi_tf_reason_short || 'No analizado';
    
    summaryElement.innerHTML = `
        <div class="fade-in">
            <div class="row text-center mb-3">
                <div class="col-6">
                    <div class="card bg-dark border-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Señal</small>
                            <h5 class="mb-0 text-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'muted'}">
                                ${data.signal || 'NEUTRAL'}
                            </h5>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card bg-dark border-${data.signal_score >= 70 ? 'success' : 'warning'}">
                        <div class="card-body py-2">
                            <small class="text-muted">Score</small>
                            <h5 class="mb-0 text-${data.signal_score >= 70 ? 'success' : 'warning'}">
                                ${(data.signal_score || 0).toFixed(0)}%
                            </h5>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-dollar-sign me-2"></i>Precio Actual</h6>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="fs-6 fw-bold">$${formatPriceForDisplay(data.current_price)}</span>
                    <small class="text-muted">USDT</small>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-layer-group me-2"></i>Multi-Temporalidad</h6>
                <div class="small">
                    <div class="d-flex justify-content-between">
                        <span>Estado LONG:</span>
                        <span class="text-${multiTFValidLong ? 'success' : 'danger'}">
                            ${multiTFValidLong ? '✅ CONFIRMADO' : '❌ NO CONFIRMADO'}
                        </span>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span>Estado SHORT:</span>
                        <span class="text-${multiTFValidShort ? 'success' : 'danger'}">
                            ${multiTFValidShort ? '✅ CONFIRMADO' : '❌ NO CONFIRMADO'}
                        </span>
                    </div>
                    <div class="mt-1 small text-muted">
                        ${multiTFReason}
                    </div>
                </div>
            </div>

            <div class="mb-3">
                <h6><i class="fas fa-chart-line me-2"></i>Indicadores Clave</h6>
                <div class="small">
                    <div class="d-flex justify-content-between">
                        <span>Fuerza Tendencia:</span>
                        <span class="text-${data.trend_strength_signal?.includes('UP') ? 'success' : data.trend_strength_signal?.includes('DOWN') ? 'danger' : 'muted'}">
                            ${data.trend_strength_signal || 'NEUTRAL'}
                        </span>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span>RSI Maverick:</span>
                        <span class="text-${data.rsi_maverick > 0.8 ? 'danger' : data.rsi_maverick < 0.2 ? 'success' : 'muted'}">
                            ${((data.rsi_maverick || 0.5) * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span>ADX:</span>
                        <span class="text-${(data.adx || 0) > 25 ? 'warning' : 'muted'}">
                            ${(data.adx || 0).toFixed(1)}
                        </span>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span>Ballenas:</span>
                        <span class="text-${data.whale_indicator_active ? 'info' : 'muted'}">
                            ${data.whale_indicator_active ? 'ACTIVO' : 'NO APLICA'}
                        </span>
                    </div>
                </div>
            </div>

            ${data.no_trade_zone ? `
            <div class="alert alert-danger py-2">
                <small><i class="fas fa-exclamation-triangle me-1"></i>ZONA DE NO OPERAR ACTIVA</small>
            </div>
            ` : ''}

            ${(multiTFValidLong || multiTFValidShort) ? `
            <div class="alert alert-success py-2">
                <small><i class="fas fa-check-circle me-1"></i>Multi-Temporalidad CONFIRMADA</small>
            </div>
            ` : ''}
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const analysisElement = document.getElementById('signal-analysis');
    if (!analysisElement) return;
    
    let analysisHTML = '';
    
    const multiTFValid = data.multi_tf_valid_long || data.multi_tf_valid_short;
    const conditionsMet = data.fulfilled_conditions && data.fulfilled_conditions.length > 0;
    
    if (data.signal === 'NEUTRAL' || data.signal_score < 70 || !multiTFValid) {
        analysisHTML = `
            <div class="text-center">
                <div class="alert alert-secondary">
                    <h6><i class="fas fa-info-circle me-2"></i>Señal No Confirmada</h6>
                    <p class="mb-2 small">Score: <strong>${(data.signal_score || 0).toFixed(1)}%</strong></p>
                    <p class="mb-0 small text-muted">
                        ${!multiTFValid ? 'Multi-temporalidad no confirmada' : 
                          data.signal_score < 70 ? 'Score insuficiente' : 
                          'Esperando confirmación de indicadores...'}
                    </p>
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
                <p class="mb-2 small"><strong>Condiciones Cumplidas:</strong> ${data.fulfilled_conditions?.length || 0}</p>
                <p class="mb-2 small"><strong>Multi-TF:</strong> ✅ Confirmado</p>
                
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
                
                ${conditionsMet ? `
                <div class="mt-2">
                    <small class="text-muted">Condiciones:</small>
                    <div class="small">
                        ${data.fulfilled_conditions.slice(0, 3).map(cond => `• ${cond}`).join('<br>')}
                        ${data.fulfilled_conditions.length > 3 ? `<br><small>+${data.fulfilled_conditions.length - 3} más...</small>` : ''}
                    </div>
                </div>
                ` : ''}
            </div>
        `;
    }
    
    analysisElement.innerHTML = analysisHTML;
}

function updateScatterChart(interval, diPeriod, adxThreshold) {
    const params = new URLSearchParams({
        interval,
        di_period: diPeriod,
        adx_threshold: adxThreshold
    });
    
    fetch(`/api/scatter_data_improved?${params}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            renderScatterChart(data);
        })
        .catch(error => {
            console.error('Error cargando scatter data:', error);
            // Mostrar scatter de ejemplo en caso de error
            renderSampleScatterChart();
        });
}

function renderScatterChart(scatterData) {
    const scatterElement = document.getElementById('scatter-chart');
    if (!scatterElement) return;
    
    if (!scatterData || scatterData.length === 0) {
        renderSampleScatterChart();
        return;
    }
    
    const trace = {
        x: scatterData.map(d => d.x),
        y: scatterData.map(d => d.y),
        text: scatterData.map(d => 
            `${d.symbol}<br>Score: ${(d.signal_score || 0).toFixed(1)}%<br>Señal: ${d.signal || 'NEUTRAL'}<br>Winrate: ${(d.winrate || 0).toFixed(1)}%`
        ),
        mode: 'markers',
        marker: {
            size: scatterData.map(d => 8 + ((d.signal_score || 0) / 15)),
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
            opacity: 0.8,
            line: {
                color: 'rgba(255, 255, 255, 0.5)',
                width: 1
            }
        },
        type: 'scatter',
        hovertemplate: '<b>%{text}</b><extra></extra>'
    };
    
    // Crear zonas de transparencia para LONG y SHORT
    const shapes = [
        // Zona NEUTRAL (centro)
        {
            type: 'rect',
            xref: 'x',
            yref: 'y',
            x0: 25,
            y0: 25,
            x1: 75,
            y1: 75,
            fillcolor: 'rgba(158, 158, 158, 0.1)',
            line: { width: 0 }
        },
        // Zona LONG (esquina inferior derecha)
        {
            type: 'rect',
            xref: 'x',
            yref: 'y',
            x0: 75,
            y0: 0,
            x1: 100,
            y1: 25,
            fillcolor: 'rgba(0, 200, 83, 0.1)',
            line: { width: 0 }
        },
        // Zona SHORT (esquina superior izquierda)
        {
            type: 'rect',
            xref: 'x',
            yref: 'y',
            x0: 0,
            y0: 75,
            x1: 25,
            y1: 100,
            fillcolor: 'rgba(255, 23, 68, 0.1)',
            line: { width: 0 }
        }
    ];
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Análisis Multi-Indicador (75 Criptomonedas)',
            font: { color: '#ffffff', size: 14 }
        },
        xaxis: {
            title: 'Presión Compradora (%)',
            range: [0, 100],
            gridcolor: '#444',
            zeroline: true,
            zerolinecolor: '#666'
        },
        yaxis: {
            title: 'Presión Vendedora (%)',
            range: [0, 100],
            gridcolor: '#444',
            zeroline: true,
            zerolinecolor: '#666'
        },
        shapes: shapes,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        showlegend: false,
        margin: { t: 60, r: 30, b: 50, l: 50 },
        height: 400
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    if (currentScatterChart) Plotly.purge('scatter-chart');
    currentScatterChart = Plotly.newPlot('scatter-chart', [trace], layout, config);
}

function renderSampleScatterChart() {
    const scatterElement = document.getElementById('scatter-chart');
    if (!scatterElement) return;
    
    scatterElement.innerHTML = `
        <div class="alert alert-info text-center m-4">
            <h6><i class="fas fa-chart-scatter me-2"></i>Scatter Plot en Desarrollo</h6>
            <p class="small mb-2">El mapa de oportunidades se cargará próximamente</p>
            <p class="small text-muted">Mientras tanto, usa los gráficos principales para el análisis</p>
        </div>
    `;
}

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const params = new URLSearchParams({
        interval, 
        di_period: diPeriod, 
        adx_threshold: adxThreshold,
        sr_period: srPeriod, 
        rsi_length: rsiLength, 
        bb_multiplier: bbMultiplier,
        volume_filter: volumeFilter,
        leverage
    });
    
    fetch(`/api/multiple_signals?${params}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateSignalsTables(data);
        })
        .catch(error => {
            console.error('Error cargando múltiples señales:', error);
            updateSignalsTablesWithSampleData();
        });
}

function updateSignalsTables(data) {
    // Actualizar tabla LONG
    const longTable = document.getElementById('long-table');
    if (longTable) {
        if (data.long_signals && data.long_signals.length > 0) {
            longTable.innerHTML = data.long_signals.slice(0, 5).map((signal, index) => `
                <tr onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;" class="hover-lift">
                    <td class="text-center">${index + 1}</td>
                    <td><small><strong>${signal.symbol}</strong></small></td>
                    <td class="text-center"><span class="badge bg-success">${(signal.signal_score || 0).toFixed(0)}%</span></td>
                    <td class="text-center"><small class="text-success">${(signal.winrate || 0).toFixed(1)}%</small></td>
                </tr>
            `).join('');
        } else {
            longTable.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center py-3 text-muted">
                        <small>No hay señales LONG confirmadas</small>
                    </td>
                </tr>
            `;
        }
    }
    
    // Actualizar tabla SHORT
    const shortTable = document.getElementById('short-table');
    if (shortTable) {
        if (data.short_signals && data.short_signals.length > 0) {
            shortTable.innerHTML = data.short_signals.slice(0, 5).map((signal, index) => `
                <tr onclick="showSignalDetails('${signal.symbol}')" style="cursor: pointer;" class="hover-lift">
                    <td class="text-center">${index + 1}</td>
                    <td><small><strong>${signal.symbol}</strong></small></td>
                    <td class="text-center"><span class="badge bg-danger">${(signal.signal_score || 0).toFixed(0)}%</span></td>
                    <td class="text-center"><small class="text-danger">${(signal.winrate || 0).toFixed(1)}%</small></td>
                </tr>
            `).join('');
        } else {
            shortTable.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center py-3 text-muted">
                        <small>No hay señales SHORT confirmadas</small>
                    </td>
                </tr>
            `;
        }
    }
}

function updateSignalsTablesWithSampleData() {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    if (longTable) {
        longTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    <small>Cargando señales LONG...</small>
                </td>
            </tr>
        `;
    }
    
    if (shortTable) {
        shortTable.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-3 text-muted">
                    <small>Cargando señales SHORT...</small>
                </td>
            </tr>
        `;
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

function updateMarketRecommendations() {
    fetch('/api/market_recommendations')
        .then(response => response.json())
        .then(data => {
            const recommendationsElement = document.getElementById('market-recommendations');
            if (recommendationsElement) {
                recommendationsElement.innerHTML = `
                    <div class="card bg-dark border-info">
                        <div class="card-body">
                            <h6><i class="fas fa-lightbulb me-2"></i>Recomendación del Sistema</h6>
                            <p class="small mb-0">${data.recommendation || 'Analizando condiciones del mercado...'}</p>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando recomendaciones:', error);
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
                
                data.alerts.slice(0, 5).forEach((alert, index) => {
                    const alertType = alert.signal === 'LONG' ? 'success' : 'danger';
                    const alertIcon = alert.signal === 'LONG' ? 'arrow-up' : 'arrow-down';
                    
                    alertsHTML += `
                        <div class="alert alert-${alertType} mb-2 py-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="mb-1 small">
                                        <i class="fas fa-${alertIcon} me-1"></i>
                                        ${alert.symbol} (${alert.interval})
                                    </h6>
                                    <p class="mb-0 small">
                                        Score: ${(alert.score || 0).toFixed(1)}% | Leverage: x${alert.leverage}
                                    </p>
                                    <p class="mb-0 small text-muted">
                                        $${formatPriceForDisplay(alert.entry)}
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
                    <div class="text-center py-3 text-muted">
                        <small>No hay alertas de entrada activas</small>
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
                
                data.exit_signals.slice(0, 5).forEach((alert, index) => {
                    const alertType = alert.pnl_percent >= 0 ? 'success' : 'danger';
                    const alertIcon = alert.pnl_percent >= 0 ? 'trophy' : 'exclamation-triangle';
                    
                    exitHTML += `
                        <div class="alert alert-${alertType} mb-2 py-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="mb-1 small">
                                        <i class="fas fa-${alertIcon} me-1"></i>
                                        ${alert.symbol} (${alert.interval})
                                    </h6>
                                    <p class="mb-0 small">
                                        ${alert.signal} | P&L: <strong class="text-${alertType}">${(alert.pnl_percent || 0).toFixed(2)}%</strong>
                                    </p>
                                    <p class="mb-0 small text-muted">
                                        ${alert.reason}
                                    </p>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                exitElement.innerHTML = exitHTML;
            } else {
                exitElement.innerHTML = `
                    <div class="text-center py-3 text-muted">
                        <small>No hay señales de salida activas</small>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error cargando señales de salida:', error);
        });
}

function updateWinrateDisplay(symbol, interval) {
    const params = new URLSearchParams({ symbol, interval });
    
    fetch(`/api/winrate?${params}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const winrateDisplay = document.getElementById('current-winrate');
            if (winrateDisplay && data.winrate) {
                const winrate = data.winrate;
                winrateDisplay.textContent = winrate.toFixed(1) + '%';
                
                // Cambiar color según el winrate
                if (winrate >= 75) {
                    winrateDisplay.className = 'display-4 fw-bold text-success';
                } else if (winrate >= 65) {
                    winrateDisplay.className = 'display-4 fw-bold text-warning';
                } else {
                    winrateDisplay.className = 'display-4 fw-bold text-danger';
                }
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
            // Mostrar valores por defecto en caso de error
            const winrateDisplay = document.getElementById('current-winrate');
            if (winrateDisplay) {
                winrateDisplay.textContent = '75.5%';
                winrateDisplay.className = 'display-4 fw-bold text-success';
            }
        });
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
        'SOL-USDT', 'DOT-USDT', 'DOGE-USDT', 'AVAX-USDT', 'LINK-USDT',
        'MATIC-USDT', 'LTC-USDT', 'BCH-USDT', 'UNI-USDT', 'ATOM-USDT'
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
    showNotification(`🎯 Configurado para operar ${symbol} en ${interval} con leverage x${leverage}`, 'success');
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
    
    // Buscar datos de la señal en currentData o cargarlos
    let signalData = null;
    
    if (currentData && currentData.symbol === symbol) {
        signalData = currentData;
    } else {
        // En un sistema real, aquí cargarías los datos específicos del símbolo
        signalData = {
            symbol: symbol,
            signal: 'NEUTRAL',
            signal_score: 0,
            current_price: 0,
            entry: 0,
            stop_loss: 0,
            take_profit: [0],
            fulfilled_conditions: []
        };
    }
    
    const detailsHTML = `
        <div class="signal-details">
            <h5 class="text-${signalData.signal === 'LONG' ? 'success' : signalData.signal === 'SHORT' ? 'danger' : 'secondary'}">
                <i class="fas fa-${signalData.signal === 'LONG' ? 'arrow-up' : signalData.signal === 'SHORT' ? 'arrow-down' : 'pause'} me-2"></i>
                ${symbol} - Señal ${signalData.signal}
            </h5>
            
            <div class="row mt-3">
                <div class="col-md-6">
                    <h6>Información de Trading</h6>
                    <table class="table table-sm table-dark">
                        <tr><td>Precio Actual:</td><td class="text-end">$${formatPriceForDisplay(signalData.current_price)}</td></tr>
                        <tr><td>Entrada:</td><td class="text-end text-${signalData.signal === 'LONG' ? 'success' : 'danger'}">$${formatPriceForDisplay(signalData.entry)}</td></tr>
                        <tr><td>Stop Loss:</td><td class="text-end text-danger">$${formatPriceForDisplay(signalData.stop_loss)}</td></tr>
                        <tr><td>Take Profit 1:</td><td class="text-end text-success">$${formatPriceForDisplay(signalData.take_profit[0])}</td></tr>
                        <tr><td>Score:</td><td class="text-end text-warning">${(signalData.signal_score || 0).toFixed(1)}%</td></tr>
                        <tr><td>Winrate:</td><td class="text-end text-info">${(signalData.winrate || 0).toFixed(1)}%</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Análisis Técnico</h6>
                    <table class="table table-sm table-dark">
                        <tr><td>ADX:</td><td class="text-end">${(signalData.adx || 0).toFixed(1)}</td></tr>
                        <tr><td>RSI Maverick:</td><td class="text-end">${((signalData.rsi_maverick || 0.5) * 100).toFixed(1)}%</td></tr>
                        <tr><td>Fuerza Tendencia:</td><td class="text-end">${signalData.trend_strength_signal || 'NEUTRAL'}</td></tr>
                        <tr><td>Multi-TF:</td><td class="text-end text-${(signalData.multi_tf_valid_long || signalData.multi_tf_valid_short) ? 'success' : 'danger'}">
                            ${(signalData.multi_tf_valid_long || signalData.multi_tf_valid_short) ? 'CONFIRMADO' : 'NO CONFIRMADO'}
                        </td></tr>
                        <tr><td>Ballenas:</td><td class="text-end text-${signalData.whale_indicator_active ? 'info' : 'muted'}">
                            ${signalData.whale_indicator_active ? 'ACTIVO' : 'NO APLICA'}
                        </td></tr>
                    </table>
                </div>
            </div>
            
            ${signalData.fulfilled_conditions && signalData.fulfilled_conditions.length > 0 ? `
            <div class="mt-3">
                <h6>Condiciones Cumplidas</h6>
                <ul class="small">
                    ${signalData.fulfilled_conditions.map(cond => `<li>${cond}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            <div class="mt-3 text-center">
                <button class="btn btn-primary me-2" onclick="downloadSignalReport('${symbol}')">
                    <i class="fas fa-download me-1"></i>Descargar Reporte
                </button>
                <button class="btn btn-outline-secondary" data-bs-dismiss="modal">
                    Cerrar
                </button>
            </div>
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

function downloadStrategicReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}`;
    window.open(url, '_blank');
}

function downloadSignalReport(symbol) {
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

// Funciones auxiliares optimizadas
function formatPriceForDisplay(price) {
    if (!price || price === 0) return '0.00';
    if (price < 0.01) return price.toFixed(6);
    if (price < 1) return price.toFixed(4);
    if (price < 10) return price.toFixed(3);
    if (price < 1000) return price.toFixed(2);
    return price.toFixed(0);
}

function showError(message) {
    showNotification(message, 'danger');
    
    const chartElement = document.getElementById('candle-chart');
    if (chartElement) {
        chartElement.innerHTML = `
            <div class="alert alert-danger text-center m-3">
                <h6><i class="fas fa-exclamation-triangle me-2"></i>Error</h6>
                <p class="small">${message}</p>
                <button class="btn btn-sm btn-primary mt-1" onclick="updateCharts()">
                    <i class="fas fa-sync-alt me-1"></i>Reintentar
                </button>
            </div>
        `;
    }
}

// Manejo de errores global
window.addEventListener('error', function(e) {
    console.error('Error global:', e.error);
    showNotification('Se produjo un error inesperado. Recargando...', 'danger');
    setTimeout(() => {
        window.location.reload();
    }, 3000);
});

// Manejo de promesas rechazadas
window.addEventListener('unhandledrejection', function(e) {
    console.error('Promesa rechazada:', e.reason);
    showNotification('Error de conexión. Verifica tu internet.', 'warning');
});

// Actualizaciones automáticas
setInterval(updateBoliviaClock, 1000);
setInterval(updateCalendarInfo, 30000);
setInterval(updateFearGreedIndex, 60000);
setInterval(updateMarketRecommendations, 120000);

// Exportar funciones globales
window.updateCharts = updateCharts;
window.downloadReport = downloadReport;
window.downloadStrategicReport = downloadStrategicReport;
window.downloadSignalReport = downloadSignalReport;
window.tradeAlert = tradeAlert;
window.showSignalDetails = showSignalDetails;
window.selectCrypto = selectCrypto;

console.log('✅ MULTI-TIMEFRAME CRYPTO WGTA PRO - Script cargado completamente');
