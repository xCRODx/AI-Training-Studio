// ============================================================
// Configuração Global e Constantes
// ============================================================
const MAX_DATA_POINTS = 100;
let ws = null;
let reconnectInterval = null;
let isTraining = false;

// ============================================================
// Elementos DOM
// ============================================================
const wsStatus = document.getElementById('ws-status');
const btnStart = document.getElementById('btn-start');
const btnResume = document.getElementById('btn-resume');
const btnStop = document.getElementById('btn-stop');
const btnCheckpoint = document.getElementById('btn-checkpoint');
const controlStatus = document.getElementById('control-status');

const statVram = document.getElementById('stat-vram');
const statTps = document.getElementById('stat-tps');
const statBps = document.getElementById('stat-bps');
const statTokens = document.getElementById('stat-tokens');
const statSteps = document.getElementById('stat-steps');
const statLr = document.getElementById('stat-lr');
const milestonesContainer = document.getElementById('milestones-container');
const milestonesSummary = document.getElementById('milestones-summary');

const contextTextEl = document.getElementById('context-text');
const targetTokenEl = document.getElementById('target-token');
const predictionsListEl = document.getElementById('predictions-list');

// Alertas de Saúde removidos


// ============================================================
// Milestones de Tokens (Preservado)
// ============================================================
const TOKEN_MILESTONES = [
    { value: 100_000_000, label: '100M' },
    { value: 500_000_000, label: '500M' },
    { value: 1_000_000_000, label: '1B' },
    { value: 2_000_000_000, label: '2B' },
    { value: 5_000_000_000, label: '5B' },
    { value: 10_000_000_000, label: '10B' },
    { value: 20_000_000_000, label: '20B' }
];

function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return 'agora';
    const total = Math.floor(seconds);
    const days = Math.floor(total / 86400);
    const hours = Math.floor((total % 86400) / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m`;
    return `${total}s`;
}

function renderMilestones(tokensConsumed, tokensPerSec) {
    if (!milestonesContainer) return;
    const currentTokens = Number(tokensConsumed) || 0;
    const speed = Number(tokensPerSec) || 0;
    const now = Date.now();
    const nextMilestone = TOKEN_MILESTONES.find((milestone) => currentTokens < milestone.value);

    if (milestonesSummary) {
        if (!nextMilestone) {
            milestonesSummary.textContent = 'Todos os marcos concluídos.';
        } else if (speed > 0) {
            const remaining = nextMilestone.value - currentTokens;
            const etaSeconds = remaining / speed;
            const etaDate = new Date(now + etaSeconds * 1000).toLocaleString('pt-BR');
            milestonesSummary.textContent = `Próximo marco ${nextMilestone.label}: ${formatDuration(etaSeconds)} (${etaDate})`;
        } else {
            milestonesSummary.textContent = `Próximo marco ${nextMilestone.label}: sem previsão até receber tokens/s`;
        }
    }

    milestonesContainer.innerHTML = TOKEN_MILESTONES.map((milestone) => {
        const reached = currentTokens >= milestone.value;
        const remainingTokens = Math.max(0, milestone.value - currentTokens);
        const etaSeconds = !reached && speed > 0 ? remainingTokens / speed : null;
        const expectedDate = reached
            ? 'Concluído'
            : etaSeconds === null
                ? 'Sem previsão'
                : new Date(now + etaSeconds * 1000).toLocaleString('pt-BR');
        const etaText = reached
            ? 'Concluído'
            : etaSeconds === null
                ? 'Sem previsão'
                : formatDuration(etaSeconds);
        const dotClass = reached
            ? 'bg-green-500 border-green-400 text-gray-950'
            : 'bg-gray-900 border-cyan-400 text-cyan-200';
        const labelClass = reached ? 'text-green-400' : 'text-gray-400';
        const etaMini = reached ? '✓' : (etaSeconds === null ? '--' : formatDuration(etaSeconds));
        const statusText = reached ? '✓' : '●';

        return `
            <div class="group relative flex flex-1 flex-col items-center min-w-0">
                <div class="w-6 h-6 rounded-full border-2 ${dotClass} flex items-center justify-center text-xs font-bold leading-none shadow-[0_0_10px_rgba(34,211,238,0.2)]">${statusText}</div>
                <span class="mt-2 text-xs font-semibold ${labelClass} truncate">${milestone.label}</span>
                <span class="mt-1 text-[10px] text-gray-500 truncate">${etaMini}</span>
                <div class="pointer-events-none absolute bottom-full mb-3 hidden group-hover:block bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 shadow-xl min-w-[180px] z-20">
                    <div class="text-[11px] text-cyan-300 font-semibold">${milestone.label}</div>
                    <div class="mt-1 text-[11px] text-gray-300">Tempo até marco: <span class="text-white">${etaText}</span></div>
                    <div class="text-[11px] text-gray-300">Data prevista: <span class="text-white">${expectedDate}</span></div>
                </div>
            </div>
        `;
    }).join('');
}

// ============================================================
// Plugin: Crosshair Vertical (linha que segue o cursor)
// Usa API pública do Chart.js v3/v4: getActiveElements()
// ============================================================
const trainingStudioCrosshair = {
    id: 'trainingStudioCrosshair',
    afterDraw(chart) {
        try {
            if (chart.config.type === 'doughnut' || chart.config.type === 'pie') return;
            const active = chart.getActiveElements();
            if (!active || !active.length) return;
            const el = active[0]?.element;
            if (!el) return;
            const x = el.x;
            if (x == null || isNaN(x)) return;
            const { ctx, chartArea } = chart;
            if (!chartArea) return;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(x, chartArea.top);
            ctx.lineTo(x, chartArea.bottom);
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(160, 195, 240, 0.45)';
            ctx.setLineDash([5, 4]);
            ctx.stroke();
            ctx.restore();
        } catch (_) { /* nunca quebrar o ciclo de draw */ }
    }
};
Chart.register(trainingStudioCrosshair);

// ============================================================
// Configuração Chart.js com Tema Escuro
// ============================================================
Chart.defaults.color = '#9ca3af';
Chart.defaults.font.family = 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif';
Chart.defaults.scale.grid.color = '#374151';

// ============================================================
// Interação global: tooltip sem precisar "caçar" o ponto exato
// mode 'index' + intersect false → tooltip aparece ao longo de todo o eixo X
// ============================================================
Chart.defaults.interaction = { mode: 'index', intersect: false };
Chart.defaults.plugins.tooltip.enabled = true;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(9, 12, 18, 0.97)';
Chart.defaults.plugins.tooltip.titleColor = '#f5f7fb';
Chart.defaults.plugins.tooltip.bodyColor = '#cbd2dd';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255, 255, 255, 0.14)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 10;
Chart.defaults.plugins.tooltip.displayColors = true;
Chart.defaults.plugins.tooltip.boxPadding = 4;
Chart.defaults.plugins.tooltip.usePointStyle = true;
Chart.defaults.plugins.tooltip.caretSize = 6;
Chart.defaults.plugins.tooltip.titleFont = { family: 'ui-monospace, monospace', size: 11, weight: '700' };
Chart.defaults.plugins.tooltip.bodyFont = { family: 'ui-monospace, monospace', size: 11 };
Chart.defaults.plugins.tooltip.callbacks.label = (ctx) => {
    const val = ctx.parsed?.y;
    if (val == null) return null;
    const formatted = Number.isInteger(val) ? val : val.toFixed(4).replace(/\.?0+$/, '');
    return ` ${ctx.dataset.label}: ${formatted}`;
};
Chart.defaults.elements.point.radius = 2;
Chart.defaults.elements.point.hoverRadius = 6;
Chart.defaults.elements.point.hitRadius = 20;

// ============================================================
// Gráfico 1: Loss (Preservado)
// ============================================================
const ctxLoss = document.getElementById('chart-loss').getContext('2d');
const chartLoss = new Chart(ctxLoss, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Total Loss',
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                data: []
            },
            {
                label: 'CE Loss',
                borderColor: '#f59e0b',
                borderWidth: 1,
                borderDash: [5, 5],
                tension: 0.3,
                data: []
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false } }
    }
});

// ============================================================
// Gráfico 2: Acurácia (Preservado)
// ============================================================
const ctxAcc = document.getElementById('chart-acc').getContext('2d');
const chartAcc = new Chart(ctxAcc, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'Top-1', borderColor: '#3b82f6', borderWidth: 2, tension: 0.3, data: [] },
            { label: 'Top-5', borderColor: '#8b5cf6', borderWidth: 2, tension: 0.3, data: [] },
            { label: 'Top-50', borderColor: '#10b981', borderWidth: 2, tension: 0.3, data: [] }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false }, y: { min: 0, max: 1 } }
    }
});

// ============================================================
// Gráfico 3: Gradient Norm (Preservado)
// ============================================================
const ctxGrad = document.getElementById('gradChart').getContext('2d');
const gradChart = new Chart(ctxGrad, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Gradient Norm',
                data: [],
                borderColor: 'rgba(255, 206, 86, 1)',
                backgroundColor: 'rgba(255, 206, 86, 0.2)',
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 2
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        scales: {
            x: { display: false },
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255, 255, 255, 0.05)' }
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: { enabled: true }
        }
    }
});

// ============================================================
// NOVO — Gráfico 4: Training vs. Validation Loss
// ============================================================
const ctxValLoss = document.getElementById('chart-val-loss').getContext('2d');
const chartValLoss = new Chart(ctxValLoss, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Training Loss',
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.08)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                data: []
            },
            {
                label: 'Validation Loss',
                borderColor: '#22d3ee',
                backgroundColor: 'rgba(34, 211, 238, 0.08)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 3,
                pointBackgroundColor: '#22d3ee',
                data: [],
                spanGaps: true
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false } }
    }
});

// ============================================================
// NOVO — Gráfico 5: Perplexidade (Treino + Validação)
// ============================================================
const ctxPerplexity = document.getElementById('chart-perplexity').getContext('2d');
const chartPerplexity = new Chart(ctxPerplexity, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Perplexidade Treino',
                borderColor: '#f97316',
                backgroundColor: 'rgba(249, 115, 22, 0.08)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                data: []
            },
            {
                label: 'Perplexidade Validação',
                borderColor: '#a78bfa',
                backgroundColor: 'rgba(167, 139, 250, 0.08)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 3,
                pointBackgroundColor: '#a78bfa',
                data: [],
                spanGaps: true
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false } }
    }
});

// ============================================================
// NOVO — Gráfico 6: Learning Rate (com overlay opcional de Loss)
// ============================================================
let lrOverlayActive = false;
const ctxLR = document.getElementById('chart-lr').getContext('2d');
const chartLR = new Chart(ctxLR, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Learning Rate',
                borderColor: '#facc15',
                backgroundColor: 'rgba(250, 204, 21, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                data: [],
                yAxisID: 'y'
            },
            {
                label: 'Loss (overlay)',
                borderColor: 'rgba(239, 68, 68, 0.5)',
                borderWidth: 1,
                borderDash: [4, 4],
                tension: 0.3,
                data: [],
                yAxisID: 'y1',
                hidden: true
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: {
            x: { display: false },
            y: {
                type: 'linear',
                position: 'left',
                title: { display: true, text: 'LR', color: '#facc15' },
                ticks: { color: '#facc15' }
            },
            y1: {
                type: 'linear',
                position: 'right',
                display: false,
                title: { display: true, text: 'Loss', color: '#ef4444' },
                ticks: { color: '#ef4444' },
                grid: { drawOnChartArea: false }
            }
        }
    }
});

// Botão de Overlay LR + Loss
const btnLrOverlay = document.getElementById('btn-lr-overlay');
if (btnLrOverlay) {
    btnLrOverlay.addEventListener('click', () => {
        lrOverlayActive = !lrOverlayActive;
        chartLR.data.datasets[1].hidden = !lrOverlayActive;
        chartLR.options.scales.y1.display = lrOverlayActive;
        chartLR.update('none');
        btnLrOverlay.textContent = lrOverlayActive ? 'Remover Loss' : 'Sobrepor Loss';
        btnLrOverlay.classList.toggle('border-blue-500', lrOverlayActive);
        btnLrOverlay.classList.toggle('text-white', lrOverlayActive);
    });
}

// ============================================================
// NOVO — Gráfico 7: Weight Norms (com threshold de explosão)
// ============================================================
const WEIGHT_NORM_CRITICAL_THRESHOLD = 500; // Limiar visual de referência
const ctxWeightNorm = document.getElementById('chart-weight-norm').getContext('2d');
const chartWeightNorm = new Chart(ctxWeightNorm, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Weight Norm (média L2)',
                borderColor: '#34d399',
                backgroundColor: 'rgba(52, 211, 153, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                data: []
            },
            {
                label: 'Limite Crítico',
                borderColor: '#ef4444',
                borderWidth: 1,
                borderDash: [8, 4],
                pointRadius: 0,
                data: [],
                fill: false
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: {
            x: { display: false },
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255, 255, 255, 0.05)' }
            }
        }
    }
});

// ============================================================
// NOVO — Gráfico 8: Distribuição Ternária (Pizza / Doughnut)
// ============================================================
const ctxTernary = document.getElementById('chart-ternary').getContext('2d');
const chartTernary = new Chart(ctxTernary, {
    type: 'doughnut',
    data: {
        labels: ['Negativos (-1)', 'Zeros (0)', 'Positivos (+1)'],
        datasets: [
            {
                data: [0, 100, 0],
                backgroundColor: ['#ef4444', '#ffffff', '#3b82f6'],
                borderColor: '#111622',
                borderWidth: 2,
                hoverOffset: 6
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '60%',
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    color: '#9ca3af',
                    boxWidth: 10,
                    boxHeight: 10,
                    usePointStyle: true,
                    font: { size: 11, family: 'ui-monospace, monospace' }
                }
            },
            tooltip: {
                enabled: true,
                callbacks: {
                    label: (ctx) => ` ${ctx.label}: ${Number(ctx.parsed).toFixed(1)}%`
                }
            }
        }
    }
});

// ============================================================
// NOVO — Gráfico 9: RMSE & Firing Rate
// ============================================================
const ctxRmseFiring = document.getElementById('chart-rmse-firing').getContext('2d');
const chartRmseFiring = new Chart(ctxRmseFiring, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'RMSE', borderColor: '#f59e0b', borderWidth: 2, tension: 0.3, data: [], yAxisID: 'y' },
            { label: 'Firing Rate (%)', borderColor: '#8b5cf6', borderWidth: 2, tension: 0.3, data: [], yAxisID: 'y1' }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: {
            x: { display: false },
            y: { type: 'linear', position: 'left' },
            y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false } }
        }
    }
});

// ============================================================
// NOVO — Gráfico 10: Evolução da Resolução
// ============================================================
const ctxResolution = document.getElementById('chart-resolution').getContext('2d');
const chartResolution = new Chart(ctxResolution, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'Group Size', borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderWidth: 2, tension: 0, stepLine: true, fill: true, data: [] }
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false }, y: { beginAtZero: true } }
    }
});


// ============================================================
// NOVO — Gráfico: Entropia dos logits (confiança das distribuições)
// ============================================================
const ctxEntropy = document.getElementById('chart-entropy').getContext('2d');
const chartEntropy = new Chart(ctxEntropy, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Entropia média',
                borderColor: '#22d3ee',
                backgroundColor: 'rgba(34, 211, 238, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                data: []
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false } }
    }
});

// ============================================================
// NOVO — Gráfico: Loss com média móvel EMA (lê do CSV)
// EMA de 500 e 5000 steps para enxergar a tendência real escondida pelo ruído.
// ============================================================
const ctxEma = document.getElementById('chart-ema').getContext('2d');
const chartEma = new Chart(ctxEma, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Loss (raw)',
                borderColor: 'rgba(239, 68, 68, 0.35)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1,
                data: []
            },
            {
                label: 'EMA 500 steps',
                borderColor: '#f59e0b',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                data: []
            },
            {
                label: 'EMA 5000 steps',
                borderColor: '#22d3ee',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                data: []
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false } }
    }
});

// ============================================================
// NOVO — Gráfico: Perplexidade com média móvel EMA (lê do CSV)
// EMA de 500 e 5000 + validação (val loss) como linha de referência.
// ============================================================
const ctxPpl = document.getElementById('chart-ppl-ema').getContext('2d');
const chartPpl = new Chart(ctxPpl, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Perplexidade (raw)',
                borderColor: 'rgba(168, 85, 247, 0.35)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1,
                data: []
            },
            {
                label: 'EMA 500 steps',
                borderColor: '#f59e0b',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                data: []
            },
            {
                label: 'EMA 5000 steps',
                borderColor: '#22d3ee',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                data: []
            },
            {
                label: 'Validação (val loss)',
                borderColor: 'rgba(52, 211, 153, 0.9)',
                borderWidth: 2,
                borderDash: [6, 4],
                pointRadius: 0,
                tension: 0.2,
                data: []
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false } }
    }
});

// ============================================================
// NOVO — Gráfico: Top-1 com média móvel EMA (lê do CSV)
// EMA de 500 e 5000 steps.
// ============================================================
const ctxTop1 = document.getElementById('chart-top1-ema').getContext('2d');
const chartTop1 = new Chart(ctxTop1, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Top-1 Acc (raw)',
                borderColor: 'rgba(34, 211, 238, 0.35)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1,
                data: []
            },
            {
                label: 'EMA 500 steps',
                borderColor: '#f59e0b',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                data: []
            },
            {
                label: 'EMA 5000 steps',
                borderColor: '#22d3ee',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                data: []
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false }, y: { min: 0, max: 1 } }
    }
});

// ============================================================
// NOVO — Gráfico: Entropia dos logits com média móvel EMA (lê do CSV)
// EMA de 500 e 5000 steps.
// ============================================================
const ctxEntEma = document.getElementById('chart-ent-ema').getContext('2d');
const chartEntEma = new Chart(ctxEntEma, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Entropia (raw)',
                borderColor: 'rgba(251, 191, 36, 0.35)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1,
                data: []
            },
            {
                label: 'EMA 500 steps',
                borderColor: '#f59e0b',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                data: []
            },
            {
                label: 'EMA 5000 steps',
                borderColor: '#22d3ee',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                data: []
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
        scales: { x: { display: false } }
    }
});
function computeEma(values, span) {
    if (!values || values.length === 0) return [];
    const alpha = 2 / (span + 1);
    const out = [];
    let ema = values[0];
    for (let i = 0; i < values.length; i++) {
        if (i === 0) ema = values[0];
        else ema = alpha * values[i] + (1 - alpha) * ema;
        out.push(ema);
    }
    return out;
}

// Versão tolerante a lacunas (valores nulos/vazios) — usada em séries que
// podem ficar sem amostra (ex.: validação) preservando o alinhamento por step.
function computeEmaWithNulls(values, span) {
    if (!values || values.length === 0) return [];
    const alpha = 2 / (span + 1);
    const out = [];
    let ema = null;
    for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v === null || v === undefined || isNaN(v)) {
            out.push(null);
            continue;
        }
        ema = ema === null ? v : alpha * v + (1 - alpha) * ema;
        out.push(ema);
    }
    return out;
}

async function loadEmaHistory() {
    try {
        const res = await fetch('/api/train/metrics/history', { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        const steps = data.steps || [];
        const losses = data.losses || [];
        const perplexity = data.perplexity || [];
        const valLoss = data.val_loss || [];
        const top1 = data.top1 || [];
        const entropy = data.entropy || [];

        chartEma.data.labels = steps;
        chartEma.data.datasets[0].data = losses;
        chartEma.data.datasets[1].data = computeEma(losses, 500);
        chartEma.data.datasets[2].data = computeEma(losses, 5000);
        chartEma.update('none');

        chartPpl.data.labels = steps;
        chartPpl.data.datasets[0].data = perplexity;
        chartPpl.data.datasets[1].data = computeEmaWithNulls(perplexity, 500);
        chartPpl.data.datasets[2].data = computeEmaWithNulls(perplexity, 5000);
        chartPpl.data.datasets[3].data = valLoss;
        chartPpl.update('none');

        chartTop1.data.labels = steps;
        chartTop1.data.datasets[0].data = top1;
        chartTop1.data.datasets[1].data = computeEmaWithNulls(top1, 500);
        chartTop1.data.datasets[2].data = computeEmaWithNulls(top1, 5000);
        chartTop1.update('none');

        chartEntEma.data.labels = steps;
        chartEntEma.data.datasets[0].data = entropy;
        chartEntEma.data.datasets[1].data = computeEmaWithNulls(entropy, 500);
        chartEntEma.data.datasets[2].data = computeEmaWithNulls(entropy, 5000);
        chartEntEma.update('none');
    } catch (_) {
        // Mantém o gráfico anterior; será tentado novamente no próximo ciclo.
    }
}

// (Código de alertas removido)

// ============================================================
// Atualiza Gráficos (mantendo limite de pontos)
// ============================================================
function updateCharts(metrics) {
    const step = metrics.step;

    // --- Loss Chart (Preservado) ---
    chartLoss.data.labels.push(step);
    chartLoss.data.datasets[0].data.push(metrics.loss);
    chartLoss.data.datasets[1].data.push(metrics.ce_loss);

    if (chartLoss.data.labels.length > MAX_DATA_POINTS) {
        chartLoss.data.labels.shift();
        chartLoss.data.datasets[0].data.shift();
        chartLoss.data.datasets[1].data.shift();
    }
    chartLoss.update('none');

    // --- Acc Chart (Preservado) ---
    chartAcc.data.labels.push(step);
    chartAcc.data.datasets[0].data.push(metrics.top1_acc);
    chartAcc.data.datasets[1].data.push(metrics.top5_acc);
    chartAcc.data.datasets[2].data.push(metrics.top50_acc);

    if (chartAcc.data.labels.length > MAX_DATA_POINTS) {
        chartAcc.data.labels.shift();
        chartAcc.data.datasets[0].data.shift();
        chartAcc.data.datasets[1].data.shift();
        chartAcc.data.datasets[2].data.shift();
    }
    chartAcc.update('none');

    // --- Grad Norm Chart (Preservado) ---
    gradChart.data.labels.push(step);
    gradChart.data.datasets[0].data.push(Number(metrics.grad_norm ?? 0));
    if (gradChart.data.labels.length > MAX_DATA_POINTS) {
        gradChart.data.labels.shift();
        gradChart.data.datasets[0].data.shift();
    }
    gradChart.update('none');

    // --- NOVO: Training vs Validation Loss ---
    chartValLoss.data.labels.push(step);
    chartValLoss.data.datasets[0].data.push(metrics.loss);
    chartValLoss.data.datasets[1].data.push(metrics.val_loss != null ? metrics.val_loss : null);

    if (chartValLoss.data.labels.length > MAX_DATA_POINTS) {
        chartValLoss.data.labels.shift();
        chartValLoss.data.datasets[0].data.shift();
        chartValLoss.data.datasets[1].data.shift();
    }
    chartValLoss.update('none');

    // --- NOVO: Perplexity ---
    chartPerplexity.data.labels.push(step);
    chartPerplexity.data.datasets[0].data.push(
        metrics.train_perplexity != null ? Math.min(metrics.train_perplexity, 100000) : null
    );
    chartPerplexity.data.datasets[1].data.push(
        metrics.val_perplexity != null ? Math.min(metrics.val_perplexity, 100000) : null
    );

    if (chartPerplexity.data.labels.length > MAX_DATA_POINTS) {
        chartPerplexity.data.labels.shift();
        chartPerplexity.data.datasets[0].data.shift();
        chartPerplexity.data.datasets[1].data.shift();
    }
    chartPerplexity.update('none');

    // --- NOVO: Learning Rate ---
    const lrValue = metrics.learning_rate != null ? metrics.learning_rate : 0;
    chartLR.data.labels.push(step);
    chartLR.data.datasets[0].data.push(lrValue);
    chartLR.data.datasets[1].data.push(metrics.loss); // Overlay data (hidden by default)

    if (chartLR.data.labels.length > MAX_DATA_POINTS) {
        chartLR.data.labels.shift();
        chartLR.data.datasets[0].data.shift();
        chartLR.data.datasets[1].data.shift();
    }
    chartLR.update('none');

    // --- NOVO: Weight Norms ---
    const wnValue = Number(metrics.weight_norm ?? 0);
    chartWeightNorm.data.labels.push(step);
    chartWeightNorm.data.datasets[0].data.push(wnValue);
    chartWeightNorm.data.datasets[1].data.push(WEIGHT_NORM_CRITICAL_THRESHOLD);

    if (chartWeightNorm.data.labels.length > MAX_DATA_POINTS) {
        chartWeightNorm.data.labels.shift();
        chartWeightNorm.data.datasets[0].data.shift();
        chartWeightNorm.data.datasets[1].data.shift();
    }
    chartWeightNorm.update('none');

    // --- NOVO: Distribuição Ternária (Pizza / Doughnut) ---
    const negPct = metrics.ternary_neg1_pct ?? 0;
    const zeroPct = metrics.ternary_zero_pct ?? 0;
    const posPct = metrics.ternary_pos1_pct ?? 0;

    chartTernary.data.datasets[0].data = [negPct, zeroPct, posPct];
    chartTernary.update('none');

    // % sempre visível (readout fixo no painel 100% do tempo)
    const ternaryReadout = document.getElementById('ternary-readout');
    if (ternaryReadout) {
        const tNeg = ternaryReadout.querySelector('.t-neg b');
        const tZero = ternaryReadout.querySelector('.t-zero b');
        const tPos = ternaryReadout.querySelector('.t-pos b');
        if (tNeg) tNeg.textContent = `${negPct.toFixed(1)}%`;
        if (tZero) tZero.textContent = `${zeroPct.toFixed(1)}%`;
        if (tPos) tPos.textContent = `${posPct.toFixed(1)}%`;
    }

    // --- NOVO: RMSE & Firing Rate ---
    chartRmseFiring.data.labels.push(step);
    chartRmseFiring.data.datasets[0].data.push(Math.sqrt(metrics.ce_loss ?? 0));
    chartRmseFiring.data.datasets[1].data.push(metrics.firing_rate ?? 0);

    if (chartRmseFiring.data.labels.length > MAX_DATA_POINTS) {
        chartRmseFiring.data.labels.shift();
        chartRmseFiring.data.datasets[0].data.shift();
        chartRmseFiring.data.datasets[1].data.shift();
    }
    chartRmseFiring.update('none');

    // --- NOVO: Evolução da Resolução ---
    chartResolution.data.labels.push(step);
    chartResolution.data.datasets[0].data.push(metrics.group_size ?? 0);

    if (chartResolution.data.labels.length > MAX_DATA_POINTS) {
        chartResolution.data.labels.shift();
        chartResolution.data.datasets[0].data.shift();
    }
    chartResolution.update('none');

    // --- NOVO: Entropia dos logits ---
    chartEntropy.data.labels.push(step);
    chartEntropy.data.datasets[0].data.push(metrics.logits_entropy ?? null);

    if (chartEntropy.data.labels.length > MAX_DATA_POINTS) {
        chartEntropy.data.labels.shift();
        chartEntropy.data.datasets[0].data.shift();
    }
    chartEntropy.update('none');
}

// ============================================================
// Atualiza "O que o modelo está pensando" (Preservado)
// ============================================================
function updatePredictions(predictions) {
    if (!predictions || !predictions.target) return;

    if (predictions.context) {
        contextTextEl.textContent = predictions.context;
    }

    targetTokenEl.textContent = predictions.target;

    predictionsListEl.innerHTML = '';

    const rawLogits = predictions.top5_logits || [];

    predictions.top5_predictions.forEach((pred, index) => {
        // Probabilidade real extraída do texto "word (xx%)"
        const probMatch = pred.match(/\(([\d.]+)%\)/);
        const realProb = probMatch ? parseFloat(probMatch[1]) : 0;
        // Barra de progresso com a probabilidade REAL (não mais mock)
        const barProb = Math.max(0, Math.min(100, realProb));
        const rawLogit = rawLogits[index] != null ? Number(rawLogits[index]).toFixed(2) : null;
        const isMatch = pred === predictions.target;

        let bgColor = 'bg-gray-800';
        let barColor = 'bg-blue-600';
        let textColor = 'text-gray-300';

        if (isMatch) {
            bgColor = 'bg-green-900/30';
            barColor = 'bg-green-500';
            textColor = 'text-green-400 font-bold';
        } else if (index === 0 && !isMatch) {
            bgColor = 'bg-red-900/20';
            barColor = 'bg-red-500/50';
            textColor = 'text-red-300';
        }

        const html = `
            <div class="relative w-full ${bgColor} border border-gray-700 rounded overflow-hidden p-2 flex justify-between items-center z-10">
                <!-- Barra de progresso ao fundo (probabilidade real) -->
                <div class="absolute left-0 top-0 bottom-0 ${barColor} opacity-20 z-[-1]" style="width: ${barProb}%"></div>

                <span class="text-sm ${textColor} z-10 break-all font-mono">${index + 1}. ${pred}</span>
                <span class="text-[11px] text-gray-400 z-10 font-mono whitespace-nowrap ml-2">${rawLogit != null ? `logit ${rawLogit}` : ''}</span>
            </div>
        `;
        predictionsListEl.insertAdjacentHTML('beforeend', html);
    });
}

// ============================================================
// WebSocket Logic (Preservado + novas métricas)
// ============================================================
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/metrics`);

    ws.onopen = () => {
        wsStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e]"></span> Conectado';
        wsStatus.className = 'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-green-900/30 text-green-400 border border-green-800';
        if (reconnectInterval) clearInterval(reconnectInterval);
    };

    ws.onmessage = (event) => {
        const metrics = JSON.parse(event.data);

        // Stats Cards (Preservado)
        statVram.textContent = metrics.vram_allocated_mb;
        statTps.textContent = metrics.tokens_per_sec;
        statBps.textContent = metrics.batches_per_sec;
        statTokens.textContent = Number(metrics.tokens_consumed).toLocaleString('pt-BR');
        statSteps.textContent = Number(metrics.step).toLocaleString('pt-BR');
        renderMilestones(metrics.tokens_consumed, metrics.tokens_per_sec);

        // NOVO: Stat card de Learning Rate
        if (statLr && metrics.learning_rate != null) {
            statLr.textContent = metrics.learning_rate.toExponential(2);
        }

        updateCharts(metrics);

        if (metrics.predictions && metrics.predictions.target) {
            updatePredictions(metrics.predictions);
        }

    };

    ws.onclose = () => {
        wsStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-500"></span> Desconectado';
        wsStatus.className = 'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-red-900/30 text-red-400 border border-red-800';
        loadTrainingStatus();
        // Tenta reconectar a cada 3 segundos
        reconnectInterval = setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket Error:', err);
        ws.close();
    };
}

// ============================================================
// UI State Management (Preservado)
// ============================================================
function renderTrainingState(mode) {
    isTraining = mode === 'running';
    btnStart.disabled = mode !== 'idle';
    btnResume.disabled = mode !== 'paused';
    btnStop.disabled = mode !== 'running';

    btnStart.classList.toggle('primary', mode === 'idle');
    btnResume.classList.toggle('primary', mode === 'paused');
    btnStop.classList.toggle('active-danger', mode === 'running');
}

async function loadTrainingStatus() {
    try {
        const res = await fetch('/api/train/status', { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        renderTrainingState(data.state || 'idle');
        renderArchitectureChip(data);
    } catch (_) {
        // A reconexão do WebSocket e a próxima consulta restauram o estado visual.
    }
}

function renderArchitectureChip(data) {
    const arch = data.architecture || 'transformer';
    // O gráfico de distribuição ternária só faz sentido em arquiteturas com
    // A métrica ternária só se aplica às arquiteturas que quantizam os pesos.
    const ternaryPanel = document.getElementById('ternary-panel');
    if (ternaryPanel) {
        ternaryPanel.style.display = (arch === 'bitnet' || arch === 'retbit') ? '' : 'none';
    }
    const chip = document.getElementById('arch-chip');
    if (!chip) return;
    const source = data.source === 'checkpoint' ? 'checkpoint' : '.env';
    const flags = [];
    if (data.use_bitnet) flags.push('BITNET');
    if (data.use_retnet) flags.push('RETNET');
    const flagsText = flags.length ? flags.join(' + ') : 'TRANSFORMER';
    chip.textContent = `${flagsText} · ${source}`;
    chip.title = data.architecture_label || arch;
    chip.setAttribute('data-architecture', arch);
    chip.className = 'runtime-label';
    const kvChip = document.getElementById('kv-cache-chip');
    if (kvChip) {
        const enabled = Boolean(data.kv_cache?.enabled);
        kvChip.textContent = enabled ? 'KV CACHE · INFERÊNCIA ATIVO' : 'ESTADO RECORRENTE · ATIVO';
        kvChip.title = enabled
            ? 'Treino paralelo sem cache; inferências usam prefill + cache K/V incremental.'
            : 'A arquitetura usa estado recorrente fixo em vez de KV cache.';
        kvChip.setAttribute('data-enabled', String(enabled));
    }
}

function showStatus(msg, isError = false) {
    controlStatus.textContent = msg;
    controlStatus.classList.remove('hidden');
    controlStatus.className = `mt-3 text-xs text-center ${isError ? 'text-red-400' : 'text-green-400'}`;
    setTimeout(() => controlStatus.classList.add('hidden'), 5000);
}

// ============================================================
// API Calls (Preservado)
// ============================================================
btnStart.addEventListener('click', async () => {
    try {
        const res = await fetch('/api/train/start', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            showStatus(data.message);
            renderTrainingState(data.state || 'running');
        } else {
            showStatus(data.detail, true);
        }
    } catch (e) {
        showStatus('Erro de conexão', true);
    }
});

btnResume.addEventListener('click', async () => {
    try {
        const res = await fetch('/api/train/resume', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            showStatus(data.message);
            renderTrainingState(data.state || 'running');
        } else {
            showStatus(data.detail, true);
            loadTrainingStatus();
        }
    } catch (e) {
        showStatus('Erro de conexão', true);
    }
});

btnStop.addEventListener('click', async () => {
    try {
        showStatus('Aguardando fronteira segura e confirmação do checkpoint...');
        renderTrainingState('pausing');
        const res = await fetch('/api/train/stop', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            showStatus(data.message);
            renderTrainingState(data.state || 'paused');
        } else {
            showStatus(data.detail, true);
        }
    } catch (e) {
        showStatus('Erro de conexão', true);
    }
});

btnCheckpoint.addEventListener('click', async () => {
    btnCheckpoint.disabled = true;
    btnCheckpoint.innerHTML = `<svg class="animate-spin h-4 w-4 mr-2" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Salvando...`;

    try {
        const res = await fetch('/api/checkpoint/create', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            showStatus(`Salvo em: ${data.path.split('/').pop()}`);
        } else {
            showStatus(data.detail, true);
        }
    } catch (e) {
        showStatus('Erro ao criar checkpoint', true);
    } finally {
        btnCheckpoint.disabled = false;
        btnCheckpoint.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"></path></svg> Criar Checkpoint`;
    }
});

// ============================================================
// Inferência Logic (Preservado)
// ============================================================
const btnInfer = document.getElementById('btn-infer');
const inferPrompt = document.getElementById('infer-prompt');
const inferResult = document.getElementById('infer-result');
const inferStatus = document.getElementById('infer-status');

btnInfer.addEventListener('click', async () => {
    const prompt = inferPrompt.value.trim();
    if (!prompt) return;

    const payload = {
        prompt: prompt,
        max_new_tokens: parseInt(document.getElementById('infer-max-tokens').value),
        sampling_mode: document.getElementById('infer-sampling-mode').value,
        temperature: parseFloat(document.getElementById('infer-temp').value),
        top_p: parseFloat(document.getElementById('infer-top-p').value),
        device: document.getElementById('infer-device').value
    };

    btnInfer.disabled = true;
    inferStatus.classList.remove('hidden');
    inferResult.textContent = prompt; // Inicia com o prompt

    try {
        const response = await fetch('/api/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Erro na inferência');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            inferResult.textContent += chunk;
            // Scroll para o fim
            inferResult.scrollTop = inferResult.scrollHeight;
        }
    } catch (e) {
        inferResult.textContent = `[Erro]: ${e.message}`;
    } finally {
        btnInfer.disabled = false;
        inferStatus.classList.add('hidden');
    }
});

// ============================================================
// Inicialização
// ============================================================
renderMilestones(0, 0);
renderTrainingState('idle');
loadTrainingStatus();
setInterval(loadTrainingStatus, 5000);
loadEmaHistory();
setInterval(loadEmaHistory, 10000);
connectWebSocket();
