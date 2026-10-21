(() => {
    const $ = (id) => document.getElementById(id);
    const MAX_POINTS = 160;
    const state = { lastStep: null, previousLoss: null, socket: null, chat: [], architectureLabel: null, methodTouched: false };
    const palette = {
        lime: '#c8ff62', cyan: '#54d9ff', violet: '#a98cff', orange: '#ffae57',
        pink: '#ff6ba6', red: '#ff657a', grid: 'rgba(255,255,255,.055)', muted: '#798394'
    };

    const formatNumber = (value, digits = 2) => {
        const number = Number(value);
        return Number.isFinite(number) ? number.toLocaleString('pt-BR', { maximumFractionDigits: digits }) : '—';
    };
    const formatDuration = (seconds) => {
        const total = Number(seconds);
        if (!Number.isFinite(total) || total <= 0) return '—';
        const days = Math.floor(total / 86400);
        const hours = Math.floor((total % 86400) / 3600);
        const minutes = Math.floor((total % 3600) / 60);
        if (days) return `${days}d ${hours}h`;
        if (hours) return `${hours}h ${minutes}m`;
        return `${minutes}m`;
    };
    const escapeText = (value) => String(value ?? '');

    function toast(message, error = false) {
        const element = $('toast');
        element.textContent = message;
        element.className = `toast show${error ? ' error' : ''}`;
        clearTimeout(toast.timer);
        toast.timer = setTimeout(() => element.className = 'toast', 4200);
    }

    // Crosshair plugin — linha vertical que segue o cursor (API pública)
    if (window.Chart && !Chart.registry.plugins.get('trainingStudioCrosshair')) {
        Chart.register({
            id: 'trainingStudioCrosshair',
            afterDraw(chart) {
                try {
                    const active = chart.getActiveElements();
                    if (!active || !active.length) return;
                    const el = active[0]?.element;
                    if (!el) return;
                    const x = el.x;
                    if (x == null || isNaN(x)) return;
                    const { ctx, chartArea: { top, bottom } } = chart;
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(x, top);
                    ctx.lineTo(x, bottom);
                    ctx.lineWidth = 1;
                    ctx.strokeStyle = 'rgba(160, 195, 240, 0.45)';
                    ctx.setLineDash([5, 4]);
                    ctx.stroke();
                    ctx.restore();
                } catch (_) { /* nunca quebrar o draw */ }
            }
        });
    }

    function chart(canvasId, datasets, scales = true) {
        if (!window.Chart) return null;
        return new Chart($(canvasId), {
            type: 'line',
            data: { labels: [], datasets },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { position: 'top', align: 'end', labels: { color: palette.muted, boxWidth: 8, boxHeight: 8, usePointStyle: true, pointStyle: 'circle', font: { size: 9 } } },
                    tooltip: {
                        enabled: true,
                        backgroundColor: '#10151e',
                        borderColor: 'rgba(255,255,255,.12)',
                        borderWidth: 1,
                        titleColor: '#fff',
                        bodyColor: '#aeb6c4',
                        padding: 10,
                        cornerRadius: 8
                    }
                },
                scales: scales ? {
                    x: { grid: { display: false }, ticks: { color: palette.muted, maxTicksLimit: 8, font: { size: 9 } } },
                    y: { grid: { color: palette.grid }, ticks: { color: palette.muted, font: { size: 9 } }, beginAtZero: false }
                } : undefined
            }
        });
    }
    const line = (label, color, axis = 'y') => ({
        label, data: [], borderColor: color, backgroundColor: `${color}18`, borderWidth: 1.8,
        pointRadius: 0, pointHoverRadius: 3, tension: .28, fill: false, yAxisID: axis
    });

    const charts = {
        loss: chart('loss-chart', [line('Train loss', palette.lime), line('Validation loss', palette.cyan)]),
        accuracy: chart('accuracy-chart', [line('Top-1', palette.orange), line('Top-5', palette.violet)]),
        optimization: chart('optimization-chart', [line('Gradient norm', palette.pink), line('Learning rate', palette.cyan, 'y1')]),
        hardware: chart('hardware-chart', [line('VRAM MB', palette.blue), line('GPU %', palette.lime, 'y1')])
    };
    if (charts.optimization) {
        charts.optimization.options.scales.y1 = { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: palette.cyan, font: { size: 9 } } };
    }
    if (charts.hardware) {
        charts.hardware.options.scales.y1 = { position: 'right', min: 0, max: 100, grid: { drawOnChartArea: false }, ticks: { color: palette.lime, font: { size: 9 } } };
    }
    const ternaryChart = window.Chart ? new Chart($('ternary-chart'), {
        type: 'doughnut',
        data: { labels: ['-1', '0', '+1'], datasets: [{ data: [0, 100, 0], backgroundColor: ['#ef4444', '#ffffff', '#3b82f6'], borderWidth: 0, hoverOffset: 3 }] },
        options: { responsive: true, maintainAspectRatio: false, cutout: '72%', plugins: { legend: { display: false }, tooltip: { enabled: true } } }
    }) : null;

    function trimChart(instance) {
        if (!instance) return;
        while (instance.data.labels.length > MAX_POINTS) {
            instance.data.labels.shift();
            instance.data.datasets.forEach(dataset => dataset.data.shift());
        }
    }
    function pushChart(instance, label, values) {
        if (!instance) return;
        instance.data.labels.push(label);
        values.forEach((value, index) => instance.data.datasets[index].data.push(value === '' || value == null ? null : Number(value)));
        trimChart(instance);
        instance.update('none');
    }

    function updateMetrics(metrics, force = false) {
        if (!metrics || !Object.keys(metrics).length) return;
        const step = Number(metrics.optimizer_step ?? 0);
        if (!force && state.lastStep === step) return;
        state.lastStep = step;

        $('kpi-loss').textContent = formatNumber(metrics.loss, 4);
        $('kpi-val-loss').textContent = formatNumber(metrics.val_loss, 4);
        $('kpi-perplexity').textContent = formatNumber(metrics.perplexity, 2);
        $('kpi-accuracy').textContent = metrics.top1_accuracy === '' ? '—' : `${formatNumber(Number(metrics.top1_accuracy) * 100, 1)}%`;
        $('kpi-top5').textContent = metrics.top5_accuracy === '' ? 'top-5 —' : `top-5 ${formatNumber(Number(metrics.top5_accuracy) * 100, 1)}%`;
        $('kpi-throughput').textContent = formatNumber(metrics.tokens_per_second, 0);
        $('kpi-gpu').textContent = `${formatNumber(metrics.gpu_utilization_pct, 0)}%`;
        $('kpi-vram').textContent = `${formatNumber(metrics.vram_allocated_mb, 0)} MB alocados`;
        $('kpi-gap').textContent = metrics.generalization_gap === '' ? 'gap —' : `gap ${formatNumber(metrics.generalization_gap, 3)}`;

        const loss = Number(metrics.loss);
        if (Number.isFinite(loss) && state.previousLoss != null) {
            const delta = loss - state.previousLoss;
            $('kpi-loss-trend').textContent = `${delta <= 0 ? '↓' : '↑'} ${formatNumber(Math.abs(delta), 4)} desde o último ponto`;
            $('kpi-loss-trend').style.color = delta <= 0 ? palette.lime : palette.red;
        }
        if (Number.isFinite(loss)) state.previousLoss = loss;

        const progress = Math.max(0, Math.min(Number(metrics.progress_pct || 0), 100));
        $('progress-value').textContent = `${formatNumber(progress, 1)}%`;
        $('progress-bar').style.width = `${progress}%`;
        $('epoch-value').textContent = `Epoch ${formatNumber(metrics.epoch, 0)}`;
        $('eta-value').textContent = `ETA ${formatDuration(metrics.eta_seconds)}`;

        pushChart(charts.loss, step, [metrics.loss, metrics.val_loss]);
        pushChart(charts.accuracy, step, [Number(metrics.top1_accuracy || 0) * 100, Number(metrics.top5_accuracy || 0) * 100]);
        pushChart(charts.optimization, step, [metrics.gradient_norm, metrics.learning_rate]);
        pushChart(charts.hardware, step, [metrics.vram_allocated_mb, metrics.gpu_utilization_pct]);

        const ternary = [Number(metrics.ternary_neg1_pct || 0), Number(metrics.ternary_zero_pct || 0), Number(metrics.ternary_pos1_pct || 0)];
        $('ternary-neg').textContent = `${formatNumber(ternary[0], 1)}%`;
        $('ternary-zero').textContent = `${formatNumber(ternary[1], 1)}%`;
        $('ternary-pos').textContent = `${formatNumber(ternary[2], 1)}%`;
        if (ternaryChart && ternary.some(Boolean)) { ternaryChart.data.datasets[0].data = ternary; ternaryChart.update('none'); }

        updateHealth(metrics);
    }

    function updateHealth(metrics) {
        let score = 100;
        const loss = Number(metrics.loss);
        const gap = Number(metrics.generalization_gap);
        const grad = Number(metrics.gradient_norm);
        const gpu = Number(metrics.gpu_utilization_pct);
        if (!Number.isFinite(loss)) score -= 40;
        if (Number.isFinite(gap) && gap > .8) score -= 25;
        if (Number.isFinite(grad) && (grad > 5 || grad === 0)) score -= 20;
        if (Number.isFinite(gpu) && gpu < 20) score -= 10;
        score = Math.max(0, score);
        $('health-score').textContent = score;
        $('health-ring').style.setProperty('--score', score);
        $('signal-loss').textContent = Number.isFinite(loss) ? (state.previousLoss != null ? 'monitorando' : 'estável') : 'aguardando';
        $('signal-generalization').textContent = Number.isFinite(gap) ? (gap < .5 ? 'saudável' : gap < 1 ? 'atenção' : 'crítico') : 'aguardando validação';
        $('signal-gradients').textContent = Number.isFinite(grad) ? (grad > 0 && grad <= 5 ? 'normal' : 'atenção') : 'aguardando';
        $('signal-hardware').textContent = Number.isFinite(gpu) ? `${formatNumber(gpu, 0)}% GPU` : 'aguardando';
    }

    function renderDataset(report, configuration) {
        $('dataset-name').textContent = report?.dataset_name || configuration?.dataset_name || '—';
        if (!report?.train) return;
        $('dataset-badge').textContent = Number(report.prompt_preservation_rate || 0) === 1 ? 'pares íntegros' : 'preparado';
        const ratio = Number(report.supervised_token_ratio || 0) * 100;
        $('supervised-ratio').textContent = `${formatNumber(ratio, 1)}%`;
        $('supervised-bar').style.width = `${Math.min(ratio, 100)}%`;
        $('data-train').textContent = formatNumber(report.train.accepted, 0);
        $('data-val').textContent = formatNumber(report.validation?.accepted, 0);
        $('data-rejected').textContent = formatNumber((report.train.invalid || 0) + (report.train.too_short || 0), 0);
        $('data-duplicates').textContent = formatNumber(report.train.duplicates, 0);
        $('data-pairs').textContent = formatNumber(report.train.pairs_preserved, 0);
        $('data-preserved').textContent = `${formatNumber(Number(report.train.prompt_preservation_rate || 0) * 100, 1)}%`;
        $('data-split').textContent = formatNumber(report.train.split_conversations, 0);
        $('data-truncated').textContent = formatNumber(report.train.truncated, 0);
    }

    function renderCheckpoints(items = []) {
        $('checkpoint-count').textContent = `${items.length} salvos`;
        const body = $('checkpoint-table');
        body.textContent = '';
        if (!items.length) {
            const row = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = 5; cell.className = 'empty-row'; cell.textContent = 'Nenhum checkpoint publicado.';
            row.appendChild(cell); body.appendChild(row); return;
        }
        items.forEach(item => {
            const row = document.createElement('tr');
            [item.name, formatNumber(item.optimizer_step, 0), formatNumber(item.epoch, 0), `${formatNumber(item.size_mb, 1)} MB`, 'íntegro'].forEach((value, index) => {
                const cell = document.createElement('td'); cell.textContent = value;
                if (index === 4) cell.style.color = palette.lime;
                row.appendChild(cell);
            });
            body.appendChild(row);
        });
    }

    function renderConfig(configuration = {}) {
        const labels = {
            engine: 'Engine', architecture: 'Arquitetura', method: 'Método', dataset_name: 'Dataset', context_window: 'Contexto', micro_batch_size: 'Micro batch',
            global_batch_size: 'Global batch', gradient_accumulation_steps: 'Grad accumulation', learning_rate: 'Learning rate',
            epochs: 'Epochs', device: 'Device', torch_compile: 'torch.compile', world_size: 'World size'
        };
        const values = { engine: $('engine-name').textContent, architecture: state.architectureLabel || '—', ...configuration };
        const list = $('config-list'); list.textContent = '';
        Object.entries(labels).forEach(([key, label]) => {
            const wrapper = document.createElement('div');
            const dt = document.createElement('dt'); const dd = document.createElement('dd');
            dt.textContent = label; dd.textContent = values[key] ?? '—';
            wrapper.append(dt, dd); list.appendChild(wrapper);
        });
    }

    function renderStatus(payload) {
        const current = payload.state || 'idle';
        $('state-badge').textContent = current.toUpperCase();
        $('state-badge').className = `state-badge state-${current}`;
        $('status-message').textContent = payload.error || payload.message || '—';
        $('model-path').textContent = payload.model?.ready ? `Modelo SFT: ${payload.model.path}` : 'Na primeira inicialização, o modelo base será copiado para base_finetuning.';
        $('engine-name').textContent = payload.engine || 'local';
        const archInfo = payload.architecture || {};
        state.architectureLabel = archInfo.label || archInfo.architecture || '—';
        const archName = $('architecture-name');
        if (archName) archName.textContent = state.architectureLabel;
        const kvCache = $('finetuning-kv-cache');
        if (kvCache) {
            kvCache.textContent = archInfo.kv_cache?.enabled ? 'KV cache · inferência' : 'Estado recorrente';
            kvCache.title = archInfo.kv_cache?.enabled
                ? 'SFT treina em paralelo; o chat usa cache K/V incremental.'
                : 'Esta arquitetura usa estado RetNet de tamanho fixo.';
        }
        // O estado ternário só faz sentido em arquiteturas com pesos ternários
        // Exibe a distribuição apenas quando há quantização ternária.
        const archValue = archInfo.architecture || '';
        const ternaryPanel = $('ternary-panel');
        if (ternaryPanel) {
            ternaryPanel.style.display = (archValue === 'bitnet' || archValue === 'retbit') ? '' : 'none';
        }

        const active = ['starting', 'running', 'pausing', 'stopping'].includes(current);
        const methodSelect = $('fine-tuning-method');
        if (!state.methodTouched || active || current === 'paused') {
            methodSelect.value = payload.configuration?.method || 'full';
            $('lora-rank').value = payload.configuration?.lora_rank ?? 16;
            $('lora-alpha').value = payload.configuration?.lora_alpha ?? 32;
            $('lora-dropout').value = payload.configuration?.lora_dropout ?? 0.05;
            $('lora-target-modules').value = payload.configuration?.lora_target_modules || $('lora-target-modules').value;
        }
        methodSelect.disabled = active || current === 'paused';
        ['lora-rank', 'lora-alpha', 'lora-dropout', 'lora-target-modules'].forEach(id => {
            $(id).disabled = active || current === 'paused';
        });
        $('lora-options').hidden = methodSelect.value !== 'lora';
        $('btn-start').disabled = active || current === 'paused';
        $('btn-resume').disabled = current !== 'paused';
        $('btn-pause').disabled = !['starting', 'running'].includes(current);
        $('btn-stop').disabled = !active && current !== 'paused';
        $('btn-checkpoint').disabled = current !== 'running';
        renderDataset(payload.dataset, payload.configuration);
        renderCheckpoints(payload.checkpoints);
        renderConfig(payload.configuration);
        updateMetrics(payload.latest_metrics);
    }

    async function action(name) {
        const labels = { start: 'Iniciando...', resume: 'Retomando...', pause: 'Pausando...', stop: 'Parando...', checkpoint: 'Agendando checkpoint...' };
        toast(labels[name]);
        try {
            const options = { method: 'POST' };
            if (name === 'start') {
                options.headers = { 'Content-Type': 'application/json' };
                options.body = JSON.stringify({
                    method: $('fine-tuning-method').value,
                    lora_rank: Number($('lora-rank').value),
                    lora_alpha: Number($('lora-alpha').value),
                    lora_dropout: Number($('lora-dropout').value),
                    lora_target_modules: $('lora-target-modules').value
                });
            }
            const response = await fetch(`/api/finetuning/${name}`, options);
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.detail || 'Falha na operação.');
            renderStatus(payload); toast(payload.message || 'Operação concluída.');
        } catch (error) { toast(error.message, true); }
    }

    async function loadHistory() {
        try {
            const response = await fetch('/api/finetuning/metrics/history?limit=300');
            const payload = await response.json();
            for (const metric of payload.items || []) {
                state.lastStep = null;
                updateMetrics(metric, true);
            }
        } catch (_) { /* The live socket remains the source of truth. */ }
    }

    function connectSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socket = new WebSocket(`${protocol}//${location.host}/ws/finetuning/metrics`);
        state.socket = socket;
        socket.onopen = () => {
            $('connection-badge').className = 'connection online';
            $('connection-badge').innerHTML = '<i></i>Telemetria online';
        };
        socket.onmessage = event => renderStatus(JSON.parse(event.data));
        socket.onclose = () => {
            $('connection-badge').className = 'connection offline';
            $('connection-badge').innerHTML = '<i></i>Telemetria offline';
            setTimeout(connectSocket, 2500);
        };
        socket.onerror = () => socket.close();
    }

    function saveChat() { localStorage.setItem('adamw-sft-chat', JSON.stringify(state.chat.slice(-40))); }
    function renderChat() {
        const shell = $('chat-messages'); shell.textContent = '';
        if (!state.chat.length) {
            const empty = document.createElement('div'); empty.className = 'chat-empty';
            const strong = document.createElement('strong'); strong.textContent = 'Pronto para conversar';
            const span = document.createElement('span'); span.textContent = 'Inicie o fine-tuning e envie uma mensagem.';
            empty.append(strong, span); shell.appendChild(empty); return;
        }
        state.chat.forEach(message => {
            const element = document.createElement('div');
            element.className = `message ${message.role}`; element.textContent = escapeText(message.content);
            shell.appendChild(element);
        });
        shell.scrollTop = shell.scrollHeight;
    }

    async function sendChat(event) {
        event.preventDefault();
        const input = $('chat-input'); const content = input.value.trim();
        if (!content) return;
        state.chat.push({ role: 'user', content }); input.value = ''; renderChat(); saveChat();
        const pending = document.createElement('div'); pending.className = 'message assistant pending';
        $('chat-messages').appendChild(pending); $('chat-messages').scrollTop = $('chat-messages').scrollHeight;
        $('btn-send-chat').disabled = true;
        try {
            const response = await fetch('/api/finetuning/chat', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: state.chat,
                    max_new_tokens: Number($('chat-max-tokens').value),
                    sampling_mode: $('chat-sampling-mode').value,
                    temperature: Number($('chat-temperature').value),
                    top_p: Number($('chat-top-p').value), top_k: 50,
                    repetition_penalty: Number($('chat-repetition').value)
                })
            });
            if (!response.ok) { const body = await response.json(); throw new Error(body.detail || 'Falha na inferência.'); }
            const reader = response.body.getReader(); const decoder = new TextDecoder(); let answer = '';
            while (true) {
                const { done, value } = await reader.read(); if (done) break;
                answer += decoder.decode(value, { stream: true }); pending.textContent = answer;
                $('chat-messages').scrollTop = $('chat-messages').scrollHeight;
            }
            pending.classList.remove('pending');
            state.chat.push({ role: 'assistant', content: answer || '[Resposta vazia]' }); saveChat();
        } catch (error) {
            pending.classList.remove('pending'); pending.textContent = `Erro: ${error.message}`; toast(error.message, true);
        } finally { $('btn-send-chat').disabled = false; }
    }

    $('btn-start').addEventListener('click', () => action('start'));
    $('btn-resume').addEventListener('click', () => action('resume'));
    $('btn-pause').addEventListener('click', () => action('pause'));
    $('btn-stop').addEventListener('click', () => action('stop'));
    $('btn-checkpoint').addEventListener('click', () => action('checkpoint'));
    $('fine-tuning-method').addEventListener('change', event => {
        state.methodTouched = true;
        $('lora-options').hidden = event.target.value !== 'lora';
    });
    $('chat-form').addEventListener('submit', sendChat);
    $('btn-clear-chat').addEventListener('click', () => { state.chat = []; saveChat(); renderChat(); });
    function updateSamplingControls() {
        const mode = $('chat-sampling-mode').value;
        const automatic = mode === 'auto';
        const greedy = mode === 'greedy';
        $('chat-temperature').disabled = automatic || greedy;
        $('chat-top-p').disabled = automatic || greedy;
        $('chat-repetition').disabled = automatic || greedy;
        $('temperature-value').textContent = automatic ? 'AUTO' : greedy ? 'GREEDY' : Number($('chat-temperature').value).toFixed(2);
    }
    $('chat-sampling-mode').addEventListener('change', () => {
        if ($('chat-sampling-mode').value === 'sample' && Number($('chat-temperature').value) <= 0) {
            $('chat-temperature').value = '0.7';
        }
        updateSamplingControls();
    });
    $('chat-temperature').addEventListener('input', event => $('temperature-value').textContent = Number(event.target.value).toFixed(2));
    $('chat-top-p').addEventListener('input', event => $('top-p-value').textContent = Number(event.target.value).toFixed(2));
    $('chat-input').addEventListener('input', event => { event.target.style.height = 'auto'; event.target.style.height = `${Math.min(event.target.scrollHeight, 130)}px`; });
    $('chat-input').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('chat-form').requestSubmit(); } });

    try { state.chat = JSON.parse(localStorage.getItem('adamw-sft-chat') || '[]'); } catch (_) { state.chat = []; }
    updateSamplingControls();
    renderChat();
    setInterval(() => $('clock').textContent = new Date().toLocaleTimeString('pt-BR'), 1000);
    $('clock').textContent = new Date().toLocaleTimeString('pt-BR');
    loadHistory().then(connectSocket);
})();
