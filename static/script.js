// Configuración global - OPTIMIZADO PARA BAJO CONSUMO DE MEMORIA
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
let isUpdating = false; // Prevenir actualizaciones simultáneas

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔧 MULTI-TIMEFRAME CRYPTO WGTA PRO - Inicializando...');
    initializeApp();
    setupEventListeners();
    
    // Iniciar con un pequeño delay para evitar carga simultánea
    setTimeout(() => {
        updateCharts();
        startAutoUpdate();
    }, 1000);
});

function initializeApp() {
    console.log('✅ Sistema inicializado correctamente');
    loadCryptoRiskClassification();
    loadMarketIndicators();
    updateCalendarInfo();
    
    // Configurar manejo de errores global
    window.addEventListener('error', function(e) {
        console.error('❌ Error global capturado:', e.error);
        showError('Error en la aplicación: ' + e.message);
    });
}

function setupEventListeners() {
    try {
        // Configurar event listeners para los controles con debounce
        const debouncedUpdate = debounce(updateCharts, 1000);
        
        document.getElementById('interval-select').addEventListener('change', debouncedUpdate);
        document.getElementById('di-period').addEventListener('change', debouncedUpdate);
        document.getElementById('adx-threshold').addEventListener('change', debouncedUpdate);
        document.getElementById('sr-period').addEventListener('change', debouncedUpdate);
        document.getElementById('rsi-length').addEventListener('change', debouncedUpdate);
        document.getElementById('bb-multiplier').addEventListener('change', debouncedUpdate);
        document.getElementById('volume-filter').addEventListener('change', debouncedUpdate);
        document.getElementById('leverage').addEventListener('change', debouncedUpdate);
        document.getElementById('aux-indicator').addEventListener('change', updateAuxChart);
        
        // Configurar buscador de cryptos
        setupCryptoSearch();
        
        // Configurar herramientas de dibujo
        setupDrawingTools();
        
        // Configurar controles de indicadores
        setupIndicatorControls();
        
        console.log('✅ Event listeners configurados');
    } catch (error) {
        console.error('❌ Error configurando event listeners:', error);
    }
}

// Función debounce para evitar múltiples llamadas
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

function updateCalendarInfo() {
    fetch('/api/bolivia_time')
        .then(response => {
            if (!response.ok) throw new Error('Error en respuesta del servidor');
            return response.json();
        })
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
    
    if (!searchInput || !cryptoList) return;
    
    searchInput.addEventListener('input', debounce(function() {
        const filter = this.value.toUpperCase();
        filterCryptoList(filter);
    }, 300));
    
    // Prevenir que el dropdown se cierre al hacer clic en el buscador
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
    
    // Remover clase activa de todos los botones
    document.querySelectorAll('.drawing-tool').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Activar botón seleccionado
    event.target.classList.add('active');
    
    // Configurar modo de dibujo según la herramienta
    const chartIds = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'aux-chart', 'trend-strength-chart'];
    
    chartIds.forEach(chartId => {
        const chartElement = document.getElementById(chartId);
        if (chartElement && window.Plotly) {
            let dragmode;
            switch(tool) {
                case 'line': dragmode = 'drawline'; break;
                case 'rectangle': dragmode = 'drawrect'; break;
                case 'circle': dragmode = 'drawcircle'; break;
                case 'text': dragmode = 'drawtext'; break;
                case 'freehand': dragmode = 'drawfreehand'; break;
                case 'marker': dragmode = 'marker'; break;
                default: dragmode = false;
            }
            
            Plotly.relayout(chartId, {dragmode: dragmode});
        }
    });
}

function setDrawingColor(color) {
    const chartIds = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'aux-chart', 'trend-strength-chart'];
    
    chartIds.forEach(chartId => {
        if (window.Plotly) {
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
    const dropdownElement = document.getElementById('cryptoDropdown');
    const bootstrapDropdown = bootstrap.Dropdown.getInstance(dropdownElement);
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
            console.log(`✅ Cargadas ${allCryptos.length} criptomonedas`);
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
    updateCalendarInfo();
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
                <p class="text-muted mb-0 small">Evaluando condiciones multi-temporalidad...</p>
            </div>
        `;
    }
}

function startAutoUpdate() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Actualización más espaciada para reducir carga
    updateInterval = setInterval(() => {
        if (document.visibilityState === 'visible' && !isUpdating) {
            console.log('🔄 Actualización automática iniciada');
            updateCharts();
            updateMarketIndicators();
            updateWinRate();
        }
    }, 120000); // 120 segundos (2 minutos)
}

function updateCharts() {
    if (isUpdating) {
        console.log('⚠️ Actualización en curso, omitiendo...');
        return;
    }
    
    isUpdating = true;
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
    
    console.log(`📊 Actualizando gráficos para ${symbol} en ${interval}`);
    
    // Actualizar gráfico principal
    updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage);
    
    // Actualizar gráfico de dispersión (con delay para reducir carga)
    setTimeout(() => {
        updateScatterChartImproved(interval);
    }, 2000);
    
    // Actualizar señales múltiples (con más delay)
    setTimeout(() => {
        updateMultipleSignals(interval);
    }, 4000);
    
    // Actualizar winrate
    updateWinRate();
}

function updateMarketIndicators() {
    updateFearGreedIndex();
    updateMarketRecommendations();
    updateScalpingAlerts();
    updateCalendarInfo();
}

function updateMainChart(symbol, interval, diPeriod, adxThreshold, srPeriod, rsiLength, bbMultiplier, volumeFilter, leverage) {
    const params = new URLSearchParams({
        symbol: symbol,
        interval: interval,
        di_period: diPeriod,
        adx_threshold: adxThreshold,
        sr_period: srPeriod,
        rsi_length: rsiLength,
        bb_multiplier: bbMultiplier,
        volume_filter: volumeFilter,
        leverage: leverage
    });
    
    const url = `/api/signals?${params}`;
    
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
            
            // Renderizar gráficos secuencialmente para reducir carga
            renderCandleChart(data);
            
            setTimeout(() => {
                renderWhaleChartImproved(data);
            }, 500);
            
            setTimeout(() => {
                renderAdxChartImproved(data);
            }, 1000);
            
            setTimeout(() => {
                renderRsiMaverickChart(data);
            }, 1500);
            
            setTimeout(() => {
                renderTrendStrengthChart(data);
                updateMarketSummary(data);
                updateSignalAnalysis(data);
                isUpdating = false; // Marcar como completado
            }, 2000);
            
        })
        .catch(error => {
            console.error('Error en updateMainChart:', error);
            showError('Error al cargar datos: ' + error.message);
            showSampleData(symbol);
            isUpdating = false;
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
        obligatory_conditions_met: false,
        fulfilled_conditions: [],
        data: []
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
                <p>Intentando cargar datos de respaldo...</p>
                <button class="btn btn-sm btn-primary mt-2" onclick="updateCharts()">Reintentar</button>
            </div>
        `;
        return;
    }

    // Limitar datos para mejor rendimiento
    const displayData = data.data.slice(-50);
    const dates = displayData.map(d => {
        const date = new Date(d.timestamp);
        return isNaN(date.getTime()) ? new Date() : date;
    });
    
    const opens = displayData.map(d => parseFloat(d.open) || 0);
    const highs = displayData.map(d => parseFloat(d.high) || 0);
    const lows = displayData.map(d => parseFloat(d.low) || 0);
    const closes = displayData.map(d => parseFloat(d.close) || 0);
    
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
    
    // Añadir líneas de soporte y resistencia si están disponibles
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
            if (index < 2) { // Limitar a 2 take profits para mejor rendimiento
                traces.push({
                    type: 'scatter',
                    x: [dates[0], dates[dates.length - 1]],
                    y: [tp, tp],
                    mode: 'lines',
                    line: {color: '#00FF00', dash: 'dash', width: 1.5},
                    name: `TP${index + 1}`
                });
            }
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
        
        // Medias móviles (solo mostrar si hay datos)
        if (options.showMA9 && data.indicators.ma_9) {
            const ma9Data = data.indicators.ma_9.slice(-50);
            traces.push({
                type: 'scatter',
                x: dates,
                y: ma9Data,
                mode: 'lines',
                line: {color: '#FF9800', width: 1},
                name: 'MA 9'
            });
        }
        
        if (options.showMA21 && data.indicators.ma_21) {
            const ma21Data = data.indicators.ma_21.slice(-50);
            traces.push({
                type: 'scatter',
                x: dates,
                y: ma21Data,
                mode: 'lines',
                line: {color: '#2196F3', width: 1},
                name: 'MA 21'
            });
        }
    }
    
    const layout = {
        title: {
            text: `${data.symbol} - Gráfico de Velas - ${document.getElementById('interval-select').value}`,
            font: {color: '#ffffff', size: 14}
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
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: `chart_${data.symbol}`,
            height: 400,
            width: 800,
            scale: 1
        }
    };
    
    // Destruir gráfico existente antes de crear uno nuevo
    if (currentChart) {
        Plotly.purge('candle-chart');
    }
    
    try {
        currentChart = Plotly.newPlot('candle-chart', traces, layout, config);
    } catch (error) {
        console.error('Error renderizando gráfico de velas:', error);
        chartElement.innerHTML = `
            <div class="alert alert-danger text-center">
                <h5>Error en el gráfico</h5>
                <p>${error.message}</p>
            </div>
        `;
    }
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

    const displayData = data.data.slice(-50);
    const dates = displayData.map(d => {
        const date = new Date(d.timestamp);
        return isNaN(date.getTime()) ? new Date() : date;
    });
    
    const whalePump = (data.indicators.whale_pump || []).slice(-50);
    const whaleDump = (data.indicators.whale_dump || []).slice(-50);
    
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
            text: `Indicador Ballenas`,
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: 'Fuerza',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        barmode: 'overlay',
        margin: {t: 40, r: 30, b: 40, l: 50},
        dragmode: drawingToolsActive ? 'drawline' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentWhaleChart) {
        Plotly.purge('whale-chart');
    }
    
    try {
        currentWhaleChart = Plotly.newPlot('whale-chart', traces, layout, config);
    } catch (error) {
        console.error('Error renderizando gráfico de ballenas:', error);
    }
}

function renderAdxChartImproved(data) {
    const chartElement = document.getElementById('adx-chart');
    
    if (!data.indicators || !data.data) {
        return;
    }

    const displayData = data.data.slice(-50);
    const dates = displayData.map(d => {
        const date = new Date(d.timestamp);
        return isNaN(date.getTime()) ? new Date() : date;
    });
    
    const adx = (data.indicators.adx || []).slice(-50);
    const plusDi = (data.indicators.plus_di || []).slice(-50);
    const minusDi = (data.indicators.minus_di || []).slice(-50);
    
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
            text: 'ADX + DMI',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
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
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentAdxChart) {
        Plotly.purge('adx-chart');
    }
    
    try {
        currentAdxChart = Plotly.newPlot('adx-chart', traces, layout, config);
    } catch (error) {
        console.error('Error renderizando gráfico ADX:', error);
    }
}

function renderRsiMaverickChart(data) {
    const chartElement = document.getElementById('rsi-maverick-chart');
    
    if (!data.indicators || !data.data) {
        return;
    }

    const displayData = data.data.slice(-50);
    const dates = displayData.map(d => {
        const date = new Date(d.timestamp);
        return isNaN(date.getTime()) ? new Date() : date;
    });
    
    const rsiMaverick = (data.indicators.rsi_maverick || []).slice(-50);
    
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
            text: 'RSI Maverick (%B)',
            font: {color: '#ffffff', size: 12}
        },
        xaxis: {
            type: 'date',
            gridcolor: '#444',
            zerolinecolor: '#444'
        },
        yaxis: {
            title: '%B',
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
            }
        ],
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentRsiChart) {
        Plotly.purge('rsi-maverick-chart');
    }
    
    try {
        currentRsiChart = Plotly.newPlot('rsi-maverick-chart', traces, layout, config);
    } catch (error) {
        console.error('Error renderizando gráfico RSI:', error);
    }
}

function renderTrendStrengthChart(data) {
    const chartElement = document.getElementById('trend-strength-chart');
    
    if (!data.indicators || !data.data || !data.indicators.trend_strength) {
        return;
    }

    const displayData = data.data.slice(-50);
    const dates = displayData.map(d => {
        const date = new Date(d.timestamp);
        return isNaN(date.getTime()) ? new Date() : date;
    });
    
    const trendStrength = (data.indicators.trend_strength || []).slice(-50);
    const colors = (data.indicators.colors || []).slice(-50);
    
    const traces = [{
        x: dates,
        y: trendStrength,
        type: 'bar',
        name: 'Fuerza Tendencia',
        marker: {color: colors}
    }];
    
    const layout = {
        title: {
            text: 'Fuerza de Tendencia Maverick',
            font: {color: '#ffffff', size: 12}
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
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#ffffff'},
        showlegend: true,
        margin: {t: 40, r: 30, b: 40, l: 50}
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentTrendStrengthChart) {
        Plotly.purge('trend-strength-chart');
    }
    
    try {
        currentTrendStrengthChart = Plotly.newPlot('trend-strength-chart', traces, layout, config);
    } catch (error) {
        console.error('Error renderizando gráfico de fuerza:', error);
    }
}

function updateAuxChart() {
    if (!currentData || !currentData.indicators) return;
    
    const indicatorType = document.getElementById('aux-indicator').value;
    const dates = currentData.data.slice(-50).map(d => new Date(d.timestamp));
    
    let traces = [];
    let layout = {};
    
    switch(indicatorType) {
        case 'rsi':
            const rsiData = (currentData.indicators.rsi_traditional || []).slice(-50);
            traces = [{
                x: dates,
                y: rsiData,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI Tradicional',
                line: {color: '#FF6B6B', width: 2}
            }];
            
            layout = {
                title: {text: 'RSI Tradicional', font: {color: '#ffffff', size: 12}},
                xaxis: {type: 'date', gridcolor: '#444'},
                yaxis: {title: 'RSI', range: [0, 100], gridcolor: '#444'},
                shapes: [
                    {type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 70, y1: 70, line: {color: 'red', dash: 'dash'}},
                    {type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 30, y1: 30, line: {color: 'green', dash: 'dash'}}
                ],
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#ffffff'},
                showlegend: true,
                margin: {t: 40, r: 30, b: 40, l: 50}
            };
            break;
            
        case 'macd':
            const macdData = (currentData.indicators.macd || []).slice(-50);
            const signalData = (currentData.indicators.macd_signal || []).slice(-50);
            const histData = (currentData.indicators.macd_histogram || []).slice(-50);
            
            traces = [
                {
                    x: dates,
                    y: macdData,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MACD',
                    line: {color: '#FF6B6B', width: 2}
                },
                {
                    x: dates,
                    y: signalData,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Señal',
                    line: {color: '#4ECDC4', width: 1}
                }
            ];
            
            layout = {
                title: {text: 'MACD', font: {color: '#ffffff', size: 12}},
                xaxis: {type: 'date', gridcolor: '#444'},
                yaxis: {title: 'MACD', gridcolor: '#444'},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#ffffff'},
                showlegend: true,
                margin: {t: 40, r: 30, b: 40, l: 50}
            };
            break;
    }
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    if (currentAuxChart) {
        Plotly.purge('aux-chart');
    }
    
    try {
        currentAuxChart = Plotly.newPlot('aux-chart', traces, layout, config);
    } catch (error) {
        console.error('Error renderizando gráfico auxiliar:', error);
    }
}

function updateMarketSummary(data) {
    const marketSummary = document.getElementById('market-summary');
    if (!marketSummary) return;
    
    const signalClass = data.signal === 'LONG' ? 'success' : 
                      data.signal === 'SHORT' ? 'danger' : 'secondary';
    
    const signalIcon = data.signal === 'LONG' ? '📈' : 
                      data.signal === 'SHORT' ? '📉' : '➡️';
    
    marketSummary.innerHTML = `
        <div class="row text-center">
            <div class="col-6 mb-3">
                <div class="card bg-dark border-${signalClass}">
                    <div class="card-body py-2">
                        <h6 class="card-title mb-1">Señal</h6>
                        <div class="h4 text-${signalClass} mb-0">
                            ${signalIcon} ${data.signal}
                        </div>
                        <small class="text-muted">Score: ${data.signal_score.toFixed(1)}%</small>
                    </div>
                </div>
            </div>
            <div class="col-6 mb-3">
                <div class="card bg-dark border-info">
                    <div class="card-body py-2">
                        <h6 class="card-title mb-1">Precio</h6>
                        <div class="h5 text-info mb-0">
                            $${data.current_price.toFixed(2)}
                        </div>
                        <small class="text-muted">Actual</small>
                    </div>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-12">
                <div class="card bg-dark border-warning">
                    <div class="card-body py-2">
                        <div class="row text-center">
                            <div class="col-4">
                                <small class="text-muted d-block">Entrada</small>
                                <strong>$${data.entry.toFixed(2)}</strong>
                            </div>
                            <div class="col-4">
                                <small class="text-muted d-block">Stop Loss</small>
                                <strong class="text-danger">$${data.stop_loss.toFixed(2)}</strong>
                            </div>
                            <div class="col-4">
                                <small class="text-muted d-block">Take Profit</small>
                                <strong class="text-success">$${data.take_profit[0].toFixed(2)}</strong>
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
    if (!signalAnalysis) return;
    
    const conditionsList = data.fulfilled_conditions.map(condition => 
        `<li class="list-group-item bg-dark border-success">✅ ${condition}</li>`
    ).join('');
    
    const obligatoryStatus = data.obligatory_conditions_met ? 
        '<span class="badge bg-success">CUMPLIDAS</span>' : 
        '<span class="badge bg-danger">NO CUMPLIDAS</span>';
    
    const noTradeWarning = data.no_trade_zone ? 
        '<div class="alert alert-danger py-2 mt-2"><small>⚠️ ZONA DE NO OPERAR DETECTADA</small></div>' : '';
    
    signalAnalysis.innerHTML = `
        <div class="signal-analysis-enhanced signal-${data.signal.toLowerCase()}-enhanced">
            <h6 class="mb-2">Análisis Multi-Temporalidad</h6>
            <div class="d-flex justify-content-between align-items-center mb-2">
                <small>Condiciones Obligatorias:</small>
                ${obligatoryStatus}
            </div>
            ${noTradeWarning}
            <div class="mt-2">
                <small class="text-muted d-block">Condiciones Cumplidas:</small>
                <ul class="list-group list-group-flush mt-1">
                    ${conditionsList.length > 0 ? conditionsList : '<li class="list-group-item bg-dark text-muted">No se cumplen condiciones suficientes</li>'}
                </ul>
            </div>
            <div class="mt-2">
                <small class="text-muted">Fuerza de Tendencia:</small>
                <div class="badge trend-strength-indicator bg-${data.trend_strength_signal.includes('UP') ? 'success' : data.trend_strength_signal.includes('DOWN') ? 'danger' : 'secondary'}">
                    ${data.trend_strength_signal}
                </div>
            </div>
        </div>
    `;
}

function updateScatterChartImproved(interval) {
    fetch(`/api/scatter_data_improved?interval=${interval}`)
        .then(response => {
            if (!response.ok) throw new Error('Error en scatter data');
            return response.json();
        })
        .then(scatterData => {
            if (!Array.isArray(scatterData)) return;
            
            const buyPressure = scatterData.map(d => d.x);
            const sellPressure = scatterData.map(d => d.y);
            const symbols = scatterData.map(d => d.symbol);
            const scores = scatterData.map(d => d.signal_score);
            const signals = scatterData.map(d => d.signal);
            const risks = scatterData.map(d => d.risk_category);
            
            const colors = risks.map(risk => {
                switch(risk) {
                    case 'bajo': return '#28a745';
                    case 'medio': return '#ffc107'; 
                    case 'alto': return '#dc3545';
                    case 'memecoins': return '#e83e8c';
                    default: return '#6c757d';
                }
            });
            
            const traces = [{
                x: buyPressure,
                y: sellPressure,
                mode: 'markers',
                type: 'scatter',
                text: symbols,
                hoverinfo: 'text',
                marker: {
                    size: 8,
                    color: colors,
                    opacity: 0.7
                }
            }];
            
            const layout = {
                title: {
                    text: 'Mapa de Oportunidades - Presión Compra vs Venta',
                    font: {color: '#ffffff', size: 14}
                },
                xaxis: {
                    title: 'Presión Compra (%)',
                    gridcolor: '#444',
                    zerolinecolor: '#444',
                    range: [0, 100]
                },
                yaxis: {
                    title: 'Presión Venta (%)',
                    gridcolor: '#444',
                    zerolinecolor: '#444',
                    range: [0, 100]
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
            
            if (currentScatterChart) {
                Plotly.purge('scatter-chart');
            }
            
            currentScatterChart = Plotly.newPlot('scatter-chart', traces, layout, config);
        })
        .catch(error => {
            console.error('Error actualizando scatter chart:', error);
        });
}

function updateMultipleSignals(interval) {
    fetch(`/api/multiple_signals?interval=${interval}`)
        .then(response => {
            if (!response.ok) throw new Error('Error en múltiples señales');
            return response.json();
        })
        .then(data => {
            updateSignalTables(data.long_signals, data.short_signals);
        })
        .catch(error => {
            console.error('Error actualizando múltiples señales:', error);
        });
}

function updateSignalTables(longSignals, shortSignals) {
    const longTable = document.getElementById('long-table');
    const shortTable = document.getElementById('short-table');
    
    if (longTable) {
        longTable.innerHTML = longSignals && longSignals.length > 0 ? 
            longSignals.map((signal, index) => `
                <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                    <td>${index + 1}</td>
                    <td>${signal.symbol}</td>
                    <td><span class="badge bg-success">${signal.signal_score.toFixed(1)}%</span></td>
                    <td>$${signal.entry.toFixed(2)}</td>
                </tr>
            `).join('') : 
            '<tr><td colspan="4" class="text-center py-3 text-muted">No hay señales LONG</td></tr>';
    }
    
    if (shortTable) {
        shortTable.innerHTML = shortSignals && shortSignals.length > 0 ? 
            shortSignals.map((signal, index) => `
                <tr class="hover-row" onclick="showSignalDetails('${signal.symbol}')">
                    <td>${index + 1}</td>
                    <td>${signal.symbol}</td>
                    <td><span class="badge bg-danger">${signal.signal_score.toFixed(1)}%</span></td>
                    <td>$${signal.entry.toFixed(2)}</td>
                </tr>
            `).join('') : 
            '<tr><td colspan="4" class="text-center py-3 text-muted">No hay señales SHORT</td></tr>';
    }
}

function showSignalDetails(symbol) {
    // Implementación básica - puedes expandir esto
    console.log('Mostrar detalles para:', symbol);
    showToast(`Detalles de ${symbol}`, 'info');
}

function updateFearGreedIndex() {
    const fearGreedElement = document.getElementById('fear-greed-index');
    if (!fearGreedElement) return;
    
    // Simular datos por ahora
    const randomIndex = Math.floor(Math.random() * 100) + 1;
    let level, color, emoji;
    
    if (randomIndex >= 75) { level = 'Extrema Codicia'; color = 'danger'; emoji = '😱'; }
    else if (randomIndex >= 55) { level = 'Codicia'; color = 'warning'; emoji = '😊'; }
    else if (randomIndex >= 45) { level = 'Neutral'; color = 'secondary'; emoji = '😐'; }
    else if (randomIndex >= 25) { level = 'Miedo'; color = 'info'; emoji = '😟'; }
    else { level = 'Miedo Extremo'; color = 'success'; emoji = '😨'; }
    
    fearGreedElement.innerHTML = `
        <div class="text-center">
            <div class="h4 text-${color} mb-1">${randomIndex}</div>
            <div class="small">${emoji} ${level}</div>
            <div class="progress fear-greed-progress mt-2">
                <div class="progress-bar bg-${color}" style="width: ${randomIndex}%"></div>
            </div>
        </div>
    `;
}

function updateMarketRecommendations() {
    const recommendationsElement = document.getElementById('market-recommendations');
    if (!recommendationsElement) return;
    
    // Recomendaciones simuladas por ahora
    const recommendations = [
        { type: 'success', text: '📈 Mercado alcista en curso' },
        { type: 'warning', text: '⚡ Alta volatilidad esperada' },
        { type: 'info', text: '🔍 Oportunidades en altcoins' }
    ];
    
    recommendationsElement.innerHTML = recommendations.map(rec => `
        <div class="alert alert-${rec.type} py-2 mb-2">
            <small>${rec.text}</small>
        </div>
    `).join('');
}

function updateScalpingAlerts() {
    fetch('/api/scalping_alerts')
        .then(response => {
            if (!response.ok) throw new Error('Error en alertas de scalping');
            return response.json();
        })
        .then(data => {
            const alertsElement = document.getElementById('scalping-alerts');
            if (!alertsElement) return;
            
            if (data.alerts && data.alerts.length > 0) {
                alertsElement.innerHTML = data.alerts.map(alert => `
                    <div class="scalping-alert alert alert-warning py-2 mb-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <strong class="small">${alert.symbol}</strong>
                            <span class="badge bg-${alert.signal === 'LONG' ? 'success' : 'danger'}">
                                ${alert.signal}
                            </span>
                        </div>
                        <div class="small">
                            Score: ${alert.score.toFixed(1)}% | Entrada: $${alert.entry.toFixed(2)}
                        </div>
                    </div>
                `).join('');
            } else {
                alertsElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <small>No hay alertas de scalping activas</small>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando alertas de scalping:', error);
        });
}

function updateExitSignals() {
    fetch('/api/exit_signals')
        .then(response => {
            if (!response.ok) throw new Error('Error en señales de salida');
            return response.json();
        })
        .then(data => {
            const exitSignalsElement = document.getElementById('exit-signals');
            if (!exitSignalsElement) return;
            
            if (data.exit_signals && data.exit_signals.length > 0) {
                exitSignalsElement.innerHTML = data.exit_signals.map(signal => `
                    <div class="alert alert-danger py-2 mb-2">
                        <div class="small">
                            <strong>${signal.symbol}</strong> - ${signal.signal}<br>
                            ${signal.reason}<br>
                            <small>P&L: ${signal.pnl_percent > 0 ? '+' : ''}${signal.pnl_percent.toFixed(2)}%</small>
                        </div>
                    </div>
                `).join('');
            } else {
                exitSignalsElement.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <small>No hay señales de salida activas</small>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando señales de salida:', error);
        });
}

function updateWinRate() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    fetch(`/api/win_rate?symbol=${symbol}&interval=${interval}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const winRateElement = document.getElementById('win-rate-display');
            if (!winRateElement) return;
            
            if (data.win_rate > 0) {
                winRateElement.innerHTML = `
                    <h4 class="text-${data.win_rate >= 60 ? 'success' : data.win_rate >= 50 ? 'warning' : 'danger'}">
                        ${data.win_rate}%
                    </h4>
                    <p class="small text-muted mb-1">Win Rate</p>
                    <p class="small mb-0">${data.successful_signals}/${data.total_signals} señales</p>
                `;
            } else {
                winRateElement.innerHTML = `
                    <p class="text-muted mb-1">No hay datos</p>
                    <p class="small mb-0">históricos</p>
                `;
            }
        })
        .catch(error => {
            console.error('Error actualizando winrate:', error);
            const winRateElement = document.getElementById('win-rate-display');
            if (winRateElement) {
                winRateElement.innerHTML = `
                    <p class="text-muted mb-0">Error calculando</p>
                `;
            }
        });
}

function downloadReport() {
    const symbol = currentSymbol;
    const interval = document.getElementById('interval-select').value;
    
    showToast('Generando reporte...', 'info');
    
    const url = `/api/generate_report?symbol=${symbol}&interval=${interval}`;
    window.open(url, '_blank');
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;
    
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remover del DOM después de que se oculte
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function showError(message) {
    showToast(`❌ ${message}`, 'danger');
}

// Limpiar recursos cuando la página se descargue
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    // Limpiar gráficos Plotly para liberar memoria
    const chartIds = ['candle-chart', 'whale-chart', 'adx-chart', 'rsi-maverick-chart', 'aux-chart', 'trend-strength-chart', 'scatter-chart'];
    chartIds.forEach(chartId => {
        const element = document.getElementById(chartId);
        if (element && window.Plotly) {
            Plotly.purge(chartId);
        }
    });
});

console.log('🚀 MULTI-TIMEFRAME CRYPTO WGTA PRO - Script cargado correctamente');
