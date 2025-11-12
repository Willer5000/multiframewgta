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
    updateWinrateDisplay();
    populateCryptoList();
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
    
    // Configurar herramientas de dibujo
    setupDrawingTools();
    
    // Configurar botón de actualización manual
    document.querySelector('.btn-primary').addEventListener('click', updateCharts);
}

function populateCryptoList() {
    const cryptoList = document.getElementById('crypto-list');
    const cryptoOptions = [
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT', 'XRP-USDT',
        'ADA-USDT', 'AVAX-USDT', 'DOT-USDT', 'LINK-USDT', 'DOGE-USDT',
        'LTC-USDT', 'BCH-USDT', 'ATOM-USDT', 'XLM-USDT', 'ETC-USDT',
        'XMR-USDT', 'ALGO-USDT', 'FIL-USDT', 'VET-USDT', 'THETA-USDT',
        'NEAR-USDT', 'FTM-USDT', 'EGLD-USDT', 'HBAR-USDT', 'GRT-USDT',
        'ENJ-USDT', 'CHZ-USDT', 'BAT-USDT', 'ZIL-USDT', 'ONE-USDT',
        'APE-USDT', 'GMT-USDT', 'GAL-USDT', 'OP-USDT', 'ARB-USDT',
        'MAGIC-USDT', 'RNDR-USDT', 'SHIB-USDT', 'PEPE-USDT', 'FLOKI-USDT'
    ];
    
    cryptoOptions.forEach(symbol => {
        const item = document.createElement('div');
        item.className = 'dropdown-item crypto-item';
        item.textContent = symbol;
        item.addEventListener('click', function() {
            selectCrypto(symbol);
        });
        cryptoList.appendChild(item);
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

function filterCryptoList(filter) {
    const items = document.querySelectorAll('.crypto-item');
    items.forEach(item => {
        const text = item.textContent.toUpperCase();
        if (text.indexOf(filter) > -1) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });
}

function selectCrypto(symbol) {
    currentSymbol = symbol;
    document.getElementById('selected-crypto').textContent = symbol;
    document.getElementById('chart-title').textContent = `Gráfico de ${symbol} - Análisis Multi-Temporalidad`;
    updateCharts();
}

function setupDrawingTools() {
    // Inicializar herramientas de dibujo para cada gráfico
    const drawingButtons = document.querySelectorAll('.drawing-tool');
    drawingButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tool = this.dataset.tool;
            activateDrawingTool(tool);
        });
    });
    
    // Configurar selector de color
    const colorPicker = document.getElementById('drawing-color');
    if (colorPicker) {
        colorPicker.addEventListener('change', function() {
            setDrawingColor(this.value);
        });
    }
}

function activateDrawingTool(tool) {
    drawingToolsActive = true;
    
    // Remover clase activa de todos los botones
    document.querySelectorAll('.drawing-tool').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Activar botón seleccionado
    event.target.classList.add('active');
    
    // Configurar modo de dibujo según la herramienta
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-traditional-chart', 
                   'rsi-maverick-chart', 'macd-chart', 'trend-strength-chart'];
    
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
    // Configurar color para herramientas de dibujo
    const charts = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-traditional-chart',
                   'rsi-maverick-chart', 'macd-chart', 'trend-strength-chart'];
    
    charts.forEach(chartId => {
        if (currentChart) {
            Plotly.relayout(chartId, {
                'newshape.line.color': color,
                'newshape.fillcolor': color + '33'
            });
        }
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

    document.getElementById('multi-timeframe-status').innerHTML = `
        <div class="text-center py-2">
            <div class="spinner-border spinner-border-sm text-warning" role="status">
                <span class="visually-hidden">Cargando...</span>
            </div>
            <p class="mt-2 mb-0 small">Verificando temporalidades...</p>
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
    }, 90000); // 90 segundos
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
    
    // Actualizar gráfico de dispersión MEJORADO
    updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar señales múltiples
    updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar winrate
    updateWinrateDisplay();
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateScalpingAlerts();
    updateExitSignals();
    updateCalendarInfo();
    updateWinrateDisplay();
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
            renderWhaleChart(data);
            renderAdxChart(data);
            renderRsiTraditionalChart(data);
            renderRsiMaverickChart(data);
            renderMacdChart(data);
            renderTrendStrengthChart(data);
            updateMarketSummary(data);
            updateSignalAnalysis(data);
            updateMultiTimeframeStatus(data);
            
        })
        .catch(error => {
            console.error('Error actualizando gráfico principal:', error);
            showErrorState('market-summary', 'Error cargando datos del mercado');
        });
}

function renderCandleChart(data) {
    const chartElement = document.getElementById('candle-chart');
    
    if (!data.data || data.data.length === 0) {
        chartElement.innerHTML = '<div class="text-center py-5 text-muted">No hay datos disponibles para el gráfico</div>';
        return;
    }
    
    const dates = data.data.map(d => new Date(d.timestamp));
    const closes = data.data.map(d => parseFloat(d.close));
    const opens = data.data.map(d => parseFloat(d.open));
    const highs = data.data.map(d => parseFloat(d.high));
    const lows = data.data.map(d => parseFloat(d.low));
    
    // Trazas de velas
    const candlestickTrace = {
        x: dates,
        close: closes,
        high: highs,
        low: lows,
        open: opens,
        type: 'candlestick',
        name: 'Precio',
        increasing: {line: {color: '#00C853'}},
        decreasing: {line: {color: '#FF1744'}}
    };
    
    // Medias móviles si están disponibles
    const traces = [candlestickTrace];
    
    if (data.indicators && data.indicators.ma_9) {
        traces.push({
            x: dates.slice(-data.indicators.ma_9.length),
            y: data.indicators.ma_9,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 9',
            line: {color: '#FF9800', width: 1}
        });
    }
    
    if (data.indicators && data.indicators.ma_21) {
        traces.push({
            x: dates.slice(-data.indicators.ma_21.length),
            y: data.indicators.ma_21,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 21',
            line: {color: '#2196F3', width: 1}
        });
    }
    
    if (data.indicators && data.indicators.ma_50) {
        traces.push({
            x: dates.slice(-data.indicators.ma_50.length),
            y: data.indicators.ma_50,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 50',
            line: {color: '#9C27B0', width: 1}
        });
    }
    
    if (data.indicators && data.indicators.ma_200) {
        traces.push({
            x: dates.slice(-data.indicators.ma_200.length),
            y: data.indicators.ma_200,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 200',
            line: {color: '#F44336', width: 2}
        });
    }
    
    const layout = {
        title: {
            text: `${currentSymbol} - Gráfico de Velas con Medias Móviles`,
            font: {size: 16, color: '#ffffff'}
        },
        xaxis: {
            title: 'Fecha/Hora',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        yaxis: {
            title: 'Precio (USDT)',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2,
            x: 0.5,
            xanchor: 'center'
        },
        margin: {t: 50, r: 50, b: 50, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    };
    
    Plotly.newPlot('candle-chart', traces, layout, config);
}

function renderWhaleChart(data) {
    const chartElement = document.getElementById('whale-chart');
    
    if (!data.indicators || !data.indicators.whale_pump) {
        chartElement.innerHTML = '<div class="text-center py-4 text-muted">Datos de ballenas no disponibles</div>';
        return;
    }
    
    const dates = data.data.map(d => new Date(d.timestamp));
    const whalePump = data.indicators.whale_pump;
    const whaleDump = data.indicators.whale_dump;
    
    const traces = [
        {
            x: dates.slice(-whalePump.length),
            y: whalePump,
            type: 'bar',
            name: 'Ballenas Compradoras',
            marker: {color: '#00C853'}
        },
        {
            x: dates.slice(-whaleDump.length),
            y: whaleDump,
            type: 'bar',
            name: 'Ballenas Vendedoras',
            marker: {color: '#FF1744'}
        }
    ];
    
    const layout = {
        title: {
            text: 'Indicador Ballenas Compradoras/Vendedoras',
            font: {size: 14, color: '#ffffff'}
        },
        xaxis: {
            title: 'Fecha/Hora',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        yaxis: {
            title: 'Fuerza Ballenas',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        barmode: 'group',
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.3,
            x: 0.5,
            xanchor: 'center'
        },
        margin: {t: 50, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('whale-chart', traces, layout, config);
}

function renderAdxChart(data) {
    const chartElement = document.getElementById('adx-chart');
    
    if (!data.indicators || !data.indicators.adx) {
        chartElement.innerHTML = '<div class="text-center py-4 text-muted">Datos ADX no disponibles</div>';
        return;
    }
    
    const dates = data.data.map(d => new Date(d.timestamp));
    const adx = data.indicators.adx;
    const plusDi = data.indicators.plus_di;
    const minusDi = data.indicators.minus_di;
    
    const traces = [
        {
            x: dates.slice(-adx.length),
            y: adx,
            type: 'scatter',
            mode: 'lines',
            name: 'ADX',
            line: {color: '#FFFFFF', width: 2}
        },
        {
            x: dates.slice(-plusDi.length),
            y: plusDi,
            type: 'scatter',
            mode: 'lines',
            name: '+DI',
            line: {color: '#00C853', width: 1.5}
        },
        {
            x: dates.slice(-minusDi.length),
            y: minusDi,
            type: 'scatter',
            mode: 'lines',
            name: '-DI',
            line: {color: '#FF1744', width: 1.5}
        }
    ];
    
    const layout = {
        title: {
            text: 'ADX con DMI (+DI y -DI)',
            font: {size: 14, color: '#ffffff'}
        },
        xaxis: {
            title: 'Fecha/Hora',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        yaxis: {
            title: 'ADX/DMI',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.3,
            x: 0.5,
            xanchor: 'center'
        },
        margin: {t: 50, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('adx-chart', traces, layout, config);
}

function renderRsiTraditionalChart(data) {
    const chartElement = document.getElementById('rsi-traditional-chart');
    
    if (!data.indicators || !data.indicators.rsi) {
        chartElement.innerHTML = '<div class="text-center py-4 text-muted">Datos RSI no disponibles</div>';
        return;
    }
    
    const dates = data.data.map(d => new Date(d.timestamp));
    const rsi = data.indicators.rsi;
    
    const traces = [
        {
            x: dates.slice(-rsi.length),
            y: rsi,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Tradicional',
            line: {color: '#00BFFF', width: 2}
        }
    ];
    
    // Líneas de sobrecompra/sobreventa
    const shapes = [
        {
            type: 'line',
            x0: dates[0],
            x1: dates[dates.length - 1],
            y0: 70,
            y1: 70,
            line: {
                color: '#FF1744',
                width: 1,
                dash: 'dash'
            }
        },
        {
            type: 'line',
            x0: dates[0],
            x1: dates[dates.length - 1],
            y0: 30,
            y1: 30,
            line: {
                color: '#00C853',
                width: 1,
                dash: 'dash'
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Tradicional con Divergencias',
            font: {size: 14, color: '#ffffff'}
        },
        xaxis: {
            title: 'Fecha/Hora',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        yaxis: {
            title: 'RSI',
            gridcolor: '#333333',
            color: '#ffffff',
            range: [0, 100]
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        shapes: shapes,
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.3,
            x: 0.5,
            xanchor: 'center'
        },
        margin: {t: 50, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('rsi-traditional-chart', traces, layout, config);
}

function renderRsiMaverickChart(data) {
    const chartElement = document.getElementById('rsi-maverick-chart');
    
    if (!data.indicators || !data.indicators.rsi_maverick) {
        chartElement.innerHTML = '<div class="text-center py-4 text-muted">Datos RSI Maverick no disponibles</div>';
        return;
    }
    
    const dates = data.data.map(d => new Date(d.timestamp));
    const rsiMaverick = data.indicators.rsi_maverick;
    
    const traces = [
        {
            x: dates.slice(-rsiMaverick.length),
            y: rsiMaverick,
            type: 'scatter',
            mode: 'lines',
            name: 'RSI Maverick (%B)',
            line: {color: '#FF9800', width: 2}
        }
    ];
    
    // Líneas de sobrecompra/sobreventa
    const shapes = [
        {
            type: 'line',
            x0: dates[0],
            x1: dates[dates.length - 1],
            y0: 0.8,
            y1: 0.8,
            line: {
                color: '#FF1744',
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
                color: '#00C853',
                width: 1,
                dash: 'dash'
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'RSI Modificado Maverick (%B Bollinger)',
            font: {size: 14, color: '#ffffff'}
        },
        xaxis: {
            title: 'Fecha/Hora',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        yaxis: {
            title: 'RSI Maverick',
            gridcolor: '#333333',
            color: '#ffffff',
            range: [0, 1]
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        shapes: shapes,
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.3,
            x: 0.5,
            xanchor: 'center'
        },
        margin: {t: 50, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('rsi-maverick-chart', traces, layout, config);
}

function renderMacdChart(data) {
    const chartElement = document.getElementById('macd-chart');
    
    if (!data.indicators || !data.indicators.macd) {
        chartElement.innerHTML = '<div class="text-center py-4 text-muted">Datos MACD no disponibles</div>';
        return;
    }
    
    const dates = data.data.map(d => new Date(d.timestamp));
    const macd = data.indicators.macd;
    const macdSignal = data.indicators.macd_signal;
    const macdHistogram = data.indicators.macd_histogram;
    
    const traces = [
        {
            x: dates.slice(-macd.length),
            y: macd,
            type: 'scatter',
            mode: 'lines',
            name: 'MACD',
            line: {color: '#FFEB3B', width: 1.5}
        },
        {
            x: dates.slice(-macdSignal.length),
            y: macdSignal,
            type: 'scatter',
            mode: 'lines',
            name: 'Señal',
            line: {color: '#FF1744', width: 1}
        },
        {
            x: dates.slice(-macdHistogram.length),
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
            text: 'MACD con Histograma y Cruces',
            font: {size: 14, color: '#ffffff'}
        },
        xaxis: {
            title: 'Fecha/Hora',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        yaxis: {
            title: 'MACD',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.3,
            x: 0.5,
            xanchor: 'center'
        },
        margin: {t: 50, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('macd-chart', traces, layout, config);
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    
    if (!data.indicators || !data.indicators.trend_strength) {
        chartElement.innerHTML = '<div class="text-center py-4 text-muted">Datos Fuerza de Tendencia no disponibles</div>';
        return;
    }
    
    const dates = data.data.map(d => new Date(d.timestamp));
    const trendStrength = data.indicators.trend_strength;
    const colors = data.indicators.colors || Array(trendStrength.length).fill('#666666');
    
    const traces = [
        {
            x: dates.slice(-trendStrength.length),
            y: trendStrength,
            type: 'bar',
            name: 'Fuerza de Tendencia',
            marker: {color: colors}
        }
    ];
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick - Ancho Bandas Bollinger %',
            font: {size: 14, color: '#ffffff'}
        },
        xaxis: {
            title: 'Fecha/Hora',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        yaxis: {
            title: 'Fuerza Tendencia %',
            gridcolor: '#333333',
            color: '#ffffff'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: false,
        margin: {t: 50, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('trend-strength-chart', traces, layout, config);
}

function updateScatterChartImproved(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/scatter_data_improved?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(scatterData => {
            renderScatterChart(scatterData);
        })
        .catch(error => {
            console.error('Error actualizando scatter chart:', error);
            document.getElementById('scatter-chart').innerHTML = '<div class="text-center py-5 text-muted">Error cargando mapa de oportunidades</div>';
        });
}

function renderScatterChart(scatterData) {
    const chartElement = document.getElementById('scatter-chart');
    
    if (!scatterData || scatterData.length === 0) {
        chartElement.innerHTML = '<div class="text-center py-5 text-muted">No hay datos disponibles para el mapa de oportunidades</div>';
        return;
    }
    
    const traces = [];
    const riskCategories = {
        'bajo': {color: '#00C853', name: 'Bajo Riesgo'},
        'medio': {color: '#FFC107', name: 'Medio Riesgo'},
        'alto': {color: '#FF1744', name: 'Alto Riesgo'},
        'memecoins': {color: '#E91E63', name: 'Memecoin'}
    };
    
    // Agrupar por categoría de riesgo
    Object.keys(riskCategories).forEach(category => {
        const categoryData = scatterData.filter(item => item.risk_category === category);
        
        if (categoryData.length > 0) {
            traces.push({
                x: categoryData.map(item => item.x),
                y: categoryData.map(item => item.y),
                text: categoryData.map(item => 
                    `${item.symbol}<br>Score: ${item.signal_score.toFixed(1)}%<br>Winrate: ${item.winrate.toFixed(1)}%`
                ),
                mode: 'markers',
                type: 'scatter',
                name: riskCategories[category].name,
                marker: {
                    color: riskCategories[category].color,
                    size: categoryData.map(item => Math.min(20, Math.max(8, item.signal_score / 5))),
                    line: {width: 1, color: 'white'}
                },
                hovertemplate: '<b>%{text}</b><br>Compra: %{x:.1f}%<br>Venta: %{y:.1f}%<extra></extra>'
            });
        }
    });
    
    const layout = {
        title: {
            text: 'Mapa de Oportunidades - Análisis Multi-Indicador (40 Criptomonedas)',
            font: {size: 16, color: '#ffffff'}
        },
        xaxis: {
            title: 'Presión Compradora (%)',
            gridcolor: '#333333',
            color: '#ffffff',
            range: [0, 100]
        },
        yaxis: {
            title: 'Presión Vendedora (%)',
            gridcolor: '#333333',
            color: '#ffffff',
            range: [0, 100]
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2,
            x: 0.5,
            xanchor: 'center'
        },
        margin: {t: 50, r: 50, b: 80, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('scatter-chart', traces, layout, config);
}

function updateMultipleSignals(interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const url = `/api/multiple_signals?interval=${interval}&di_period=${diPeriod}&adx_threshold=${adxThreshold}&sr_period=${srPeriod}&rsi_length=${rsiLength}&bb_multiplier=${bbMultiplier}&volume_filter=${volumeFilter}&leverage=${leverage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(signalsData => {
            updateLongShortTables(signalsData);
        })
        .catch(error => {
            console.error('Error actualizando señales múltiples:', error);
        });
}

function updateLongShortTables(signalsData) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    // Actualizar tabla LONG
    if (signalsData.long_signals && signalsData.long_signals.length > 0) {
        longTable.innerHTML = signalsData.long_signals.slice(0, 5).map((signal, index) => `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-success">${signal.signal_score.toFixed(1)}%</span></td>
                <td>${signal.entry.toFixed(6)}</td>
            </tr>
        `).join('');
    } else {
        longTable.innerHTML = '<tr><td colspan="4" class="text-center py-3 text-muted">No hay señales LONG confirmadas</td></tr>';
    }
    
    // Actualizar tabla SHORT
    if (signalsData.short_signals && signalsData.short_signals.length > 0) {
        shortTable.innerHTML = signalsData.short_signals.slice(0, 5).map((signal, index) => `
            <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                <td>${index + 1}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-danger">${signal.signal_score.toFixed(1)}%</span></td>
                <td>${signal.entry.toFixed(6)}</td>
            </tr>
        `).join('');
    } else {
        shortTable.innerHTML = '<tr><td colspan="4" class="text-center py-3 text-muted">No hay señales SHORT confirmadas</td></tr>';
    }
}

function showSignalDetails(symbol) {
    // Implementar modal con detalles de la señal
    const modal = new bootstrap.Modal(document.getElementById('signalModal'));
    document.getElementById('signal-details').innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Cargando...</span>
            </div>
            <p class="mt-2 mb-0">Cargando detalles para ${symbol}...</p>
        </div>
    `;
    modal.show();
    
    // Aquí se cargarían los detalles específicos de la señal
    setTimeout(() => {
        document.getElementById('signal-details').innerHTML = `
            <h6>Detalles de Señal - ${symbol}</h6>
            <p>Análisis multi-temporalidad en proceso...</p>
            <div class="alert alert-info">
                <small>Esta funcionalidad está en desarrollo. Próximamente mostrará análisis detallado de la señal.</small>
            </div>
        `;
    }, 1000);
}

function updateMarketSummary(data) {
    const summaryElement = document.getElementById('market-summary');
    
    if (!data) {
        summaryElement.innerHTML = '<div class="text-center py-3 text-muted">Datos no disponibles</div>';
        return;
    }
    
    const signalClass = data.signal === 'LONG' ? 'success' : 
                       data.signal === 'SHORT' ? 'danger' : 'secondary';
    const signalIcon = data.signal === 'LONG' ? 'fa-arrow-up' : 
                      data.signal === 'SHORT' ? 'fa-arrow-down' : 'fa-minus';
    
    summaryElement.innerHTML = `
        <div class="row text-center">
            <div class="col-6 mb-3">
                <div class="card bg-dark border-${signalClass}">
                    <div class="card-body py-2">
                        <i class="fas ${signalIcon} text-${signalClass} fa-2x mb-2"></i>
                        <h6 class="mb-1">Señal</h6>
                        <div class="badge bg-${signalClass}">${data.signal}</div>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-info">
                    <div class="card-body py-2">
                        <i class="fas fa-chart-line text-info fa-2x mb-2"></i>
                        <h6 class="mb-1">Score</h6>
                        <div class="h5 text-info">${data.signal_score.toFixed(1)}%</div>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-warning">
                    <div class="card-body py-2">
                        <i class="fas fa-dollar-sign text-warning fa-2x mb-2"></i>
                        <h6 class="mb-1">Precio</h6>
                        <div class="h6 text-warning">${data.current_price.toFixed(6)}</div>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-success">
                    <div class="card-body py-2">
                        <i class="fas fa-trophy text-success fa-2x mb-2"></i>
                        <h6 class="mb-1">Winrate</h6>
                        <div class="h6 text-success">${data.winrate.toFixed(1)}%</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function updateSignalAnalysis(data) {
    const analysisElement = document.getElementById('signal-analysis');
    
    if (!data) {
        analysisElement.innerHTML = '<div class="text-center py-3 text-muted">Análisis no disponible</div>';
        return;
    }
    
    const conditionsList = data.fulfilled_conditions && data.fulfilled_conditions.length > 0 ? 
        data.fulfilled_conditions.map(condition => `<li class="small">${condition}</li>`).join('') :
        '<li class="small text-muted">No se cumplen condiciones suficientes</li>';
    
    analysisElement.innerHTML = `
        <div class="alert alert-${data.signal === 'LONG' ? 'success' : data.signal === 'SHORT' ? 'danger' : 'secondary'}">
            <h6 class="alert-heading">
                <i class="fas fa-${data.signal === 'LONG' ? 'arrow-up' : data.signal === 'SHORT' ? 'arrow-down' : 'minus'} me-2"></i>
                Señal ${data.signal} - ${data.signal_score.toFixed(1)}%
            </h6>
            <hr>
            <p class="mb-2 small"><strong>Condiciones cumplidas:</strong></p>
            <ul class="small mb-0">
                ${conditionsList}
            </ul>
        </div>
    `;
}

function updateMultiTimeframeStatus(data) {
    const statusElement = document.getElementById('multi-timeframe-status');
    
    if (!data) {
        statusElement.innerHTML = '<div class="text-center py-3 text-muted">Status no disponible</div>';
        return;
    }
    
    const multiTfStatus = data.multi_timeframe_ok ? 
        '<span class="badge bg-success">CONFIRMADO</span>' : 
        '<span class="badge bg-danger">NO CONFIRMADO</span>';
    
    const obligatoryStatus = data.obligatory_conditions_met ? 
        '<span class="badge bg-success">CUMPLIDAS</span>' : 
        '<span class="badge bg-danger">FALTANTES</span>';
    
    const noTradeStatus = data.no_trade_zone ? 
        '<span class="badge bg-danger">NO OPERAR</span>' : 
        '<span class="badge bg-success">OPERABLE</span>';
    
    statusElement.innerHTML = `
        <div class="row text-center">
            <div class="col-12 mb-2">
                <small class="text-muted">Multi-TF:</small>
                <div>${multiTfStatus}</div>
            </div>
            <div class="col-12 mb-2">
                <small class="text-muted">Obligatorias:</small>
                <div>${obligatoryStatus}</div>
            </div>
            <div class="col-12">
                <small class="text-muted">Zona:</small>
                <div>${noTradeStatus}</div>
            </div>
        </div>
    `;
}

function updateFearGreedIndex() {
    const fearGreedElement = document.getElementById('fear-greed-index');
    
    // Simulación del índice (en un sistema real esto vendría de una API)
    const fearGreedValue = Math.floor(Math.random() * 100);
    let status, color, description;
    
    if (fearGreedValue >= 75) {
        status = 'Extrema Codicia';
        color = 'danger';
        description = 'Mercado sobrecomprado - Precaución';
    } else if (fearGreedValue >= 55) {
        status = 'Codicia';
        color = 'warning';
        description = 'Mercado alcista';
    } else if (fearGreedValue >= 45) {
        status = 'Neutral';
        color = 'info';
        description = 'Mercado equilibrado';
    } else if (fearGreedValue >= 25) {
        status = 'Miedo';
        color = 'primary';
        description = 'Mercado bajista';
    } else {
        status = 'Miedo Extremo';
        color = 'success';
        description = 'Oportunidad de compra';
    }
    
    fearGreedElement.innerHTML = `
        <div class="text-center">
            <div class="h1 text-${color}">${fearGreedValue}</div>
            <div class="badge bg-${color} mb-2">${status}</div>
            <p class="small text-muted mb-2">${description}</p>
            <div class="progress fear-greed-progress">
                <div class="progress-bar bg-${color}" role="progressbar" 
                     style="width: ${fearGreedValue}%" 
                     aria-valuenow="${fearGreedValue}" 
                     aria-valuemin="0" 
                     aria-valuemax="100">
                </div>
            </div>
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
                    <div class="alert alert-${alert.signal === 'LONG' ? 'success' : 'danger'} mb-2 py-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <strong class="small">${alert.symbol}</strong>
                            <span class="badge bg-${alert.signal === 'LONG' ? 'success' : 'danger'}">${alert.signal}</span>
                        </div>
                        <small class="d-block">${alert.interval} - Score: ${alert.score.toFixed(1)}%</small>
                    </div>
                `).join('');
            } else {
                alertsElement.innerHTML = '<div class="text-center py-3 text-muted small">No hay alertas activas</div>';
            }
        })
        .catch(error => {
            console.error('Error actualizando alertas:', error);
            document.getElementById('scalping-alerts').innerHTML = '<div class="text-center py-3 text-muted small">Error cargando alertas</div>';
        });
}

function updateExitSignals() {
    fetch('/api/exit_signals')
        .then(response => response.json())
        .then(data => {
            const exitElement = document.getElementById('exit-signals');
            
            if (data.exit_signals && data.exit_signals.length > 0) {
                exitElement.innerHTML = data.exit_signals.slice(0, 3).map(signal => `
                    <div class="alert alert-warning mb-2 py-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <strong class="small">${signal.symbol}</strong>
                            <span class="badge bg-warning">SALIR</span>
                        </div>
                        <small class="d-block">${signal.reason}</small>
                        <small class="d-block">P&L: ${signal.pnl_percent}%</small>
                    </div>
                `).join('');
            } else {
                exitElement.innerHTML = '<div class="text-center py-3 text-muted small">No hay señales de salida</div>';
            }
        })
        .catch(error => {
            console.error('Error actualizando señales de salida:', error);
            document.getElementById('exit-signals').innerHTML = '<div class="text-center py-3 text-muted small">Error cargando salidas</div>';
        });
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

function updateWinrateDisplay() {
    fetch('/api/winrate')
        .then(response => response.json())
        .then(data => {
            const winrateElement = document.getElementById('winrate-value');
            if (winrateElement && data.winrate > 0) {
                winrateElement.textContent = `${data.winrate}%`;
                winrateElement.className = 'winrate-value ' + 
                    (data.winrate >= 70 ? 'text-success' : 
                     data.winrate >= 60 ? 'text-warning' : 'text-danger');
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
        });
}

function loadCryptoRiskClassification() {
    fetch('/api/crypto_risk_classification')
        .then(response => response.json())
        .then(data => {
            // Los datos de clasificación de riesgo están disponibles
            console.log('Clasificación de riesgo cargada');
        })
        .catch(error => {
            console.error('Error cargando clasificación de riesgo:', error);
        });
}

function loadMarketIndicators() {
    updateFearGreedIndex();
    updateScalpingAlerts();
    updateExitSignals();
}

function showErrorState(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="text-center py-4 text-danger">
                <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                <p class="mb-0">${message}</p>
            </div>
        `;
    }
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    const leverage = document.getElementById('leverage').value;
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}&leverage=${leverage}`;
    window.open(url, '_blank');
}

// Función para mostrar notificaciones toast
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'toast-' + Date.now();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-bg-${type} border-0`;
    toast.id = toastId;
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remover el toast del DOM después de que se oculte
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

// Back to top functionality
function setupBackToTop() {
    const backToTopButton = document.createElement('button');
    backToTopButton.className = 'back-to-top';
    backToTopButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
    backToTopButton.setAttribute('aria-label', 'Volver arriba');
    document.body.appendChild(backToTopButton);

    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTopButton.classList.add('show');
        } else {
            backToTopButton.classList.remove('show');
        }
    });

    backToTopButton.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// Inicializar back to top cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', setupBackToTop);
