(() => {
    const $ = id => document.getElementById(id);
    const storageKey = 'adamw-compiled-chat';
    let artifacts = [], messages = [], capabilities = { cpu: true, cuda: false };
    function toast(message, error = false) { const el = $('toast'); el.textContent = message; el.className = `toast show${error ? ' error' : ''}`; setTimeout(() => el.className = 'toast', 4200); }
    function saveMessages() { localStorage.setItem(storageKey, JSON.stringify(messages.map(({ streaming, ...message }) => message))); }
    function loadMessages() { try { messages = JSON.parse(localStorage.getItem(storageKey) || '[]'); } catch (_) { messages = []; } if (!Array.isArray(messages)) messages = []; }
    function renderMessages() {
        const container = $('pg-messages'); container.textContent = '';
        if (!messages.length) { const empty = document.createElement('div'); empty.className = 'chat-empty'; const title = document.createElement('strong'); title.textContent = 'Modelo residente'; const subtitle = document.createElement('span'); subtitle.textContent = 'Envie uma mensagem para começar.'; empty.append(title, subtitle); container.appendChild(empty); return; }
        messages.forEach(message => { const bubble = document.createElement('div'); bubble.className = `message ${message.role}${message.streaming ? ' pending' : ''}`; bubble.textContent = message.content || (message.streaming ? 'Gerando' : '[EOS]'); if (message.meta) { const meta = document.createElement('small'); meta.className = 'message-meta'; meta.textContent = message.meta; bubble.appendChild(meta); } container.appendChild(bubble); });
        container.scrollTop = container.scrollHeight;
    }
    const selectedArtifact = () => artifacts[Number($('artifact-select').value)];
    const selectedDevice = () => $('pg-device').value;
    const targetLabels = { 'retbit-cpp': 'Runtime C++ nativo', 'bitnet.cpp': 'bitnet.cpp', 'llama.cpp': 'llama.cpp' };
    const targetLabel = target => targetLabels[target] || target || 'Runtime C++ nativo';
    function updateModelCard() { const artifact = selectedArtifact(), device = selectedDevice(); if (!artifact) return; $('active-model').textContent = artifact.name; $('active-model-size').textContent = device === 'cuda' ? `${artifact.size_mb} MB · PyTorch (CUDA Graph)` : `${artifact.size_mb} MB · ${targetLabel(artifact.target)}`; $('load-state').textContent = 'Aguardando carga'; $('load-state').className = 'load-state'; }
    async function loadSelected() {
        const artifact = selectedArtifact(), device = selectedDevice(); if (!artifact) return false;
        const button = $('pg-load'); button.disabled = true; button.textContent = 'Carregando…'; $('load-state').textContent = `Carregando em ${device.toUpperCase()}…`;
        try { const response = await fetch('/api/playground/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ artifact: artifact.name, device, threads: Number($('pg-threads').value) }) }); const body = await response.json(); if (!response.ok) throw new Error(body.detail || 'Falha ao carregar o modelo.'); const detail = body.load_seconds !== undefined ? ` · ${body.load_seconds}s` : ''; $('load-state').textContent = `${device.toUpperCase()} carregada${detail}`; $('load-state').className = 'load-state ready'; $('runtime-status').innerHTML = `<i></i>${device.toUpperCase()} residente`; return true; }
        catch (error) { $('load-state').textContent = 'Falha na carga'; $('load-state').className = 'load-state error'; toast(error.message, true); return false; }
        finally { button.disabled = false; button.textContent = '↻ Carregar modelo'; }
    }
    function render(data) {
        artifacts = data.artifacts || []; capabilities = data.devices || capabilities; const ready = Boolean(data.ready && artifacts.length);
        $('runtime-status').className = `connection ${ready ? 'online' : 'offline'}`; $('runtime-status').innerHTML = `<i></i>${ready ? 'Preparando runtime' : 'Runtime indisponível'}`;
        if ($('pg-target-chip')) {
            const target = targetLabel(artifacts.find(item => (item.target || 'retbit-cpp') === 'retbit-cpp')?.target || artifacts[0]?.target);
            $('pg-target-chip').textContent = target;
        }
        const gpuOption = $('pg-device').querySelector('option[value="cuda"]'); gpuOption.disabled = !capabilities.cuda; gpuOption.textContent = capabilities.cuda ? `GPU · PyTorch (${data.gpu_name || 'CUDA'})` : 'GPU · CUDA indisponível';
        const cpuOption = $('pg-device').querySelector('option[value="cpu"]'); cpuOption.disabled = !capabilities.cpu; cpuOption.textContent = capabilities.cpu ? 'CPU · runtime C++ nativo' : 'CPU · runtime indisponível';
        if ($('pg-kv-cache')) $('pg-kv-cache').textContent = data.kv_cache?.enabled ? 'KV cache · ativo' : 'Estado recorrente';
        const select = $('artifact-select'); select.textContent = ''; if (!ready) return;
        artifacts.forEach((item, index) => { const option = document.createElement('option'); option.value = index; option.textContent = `${item.name} · ${targetLabel(item.target)}`; select.appendChild(option); });
        select.disabled = false; $('pg-device').disabled = false; $('pg-load').disabled = false; $('pg-input').disabled = false; $('pg-send').disabled = false; loadMessages(); renderMessages(); updateModelCard();
    }
    async function status() { try { const response = await fetch('/api/playground/status'); render(await response.json()); if (artifacts.length) await loadSelected(); } catch (error) { toast(error.message, true); } }
    $('artifact-select').addEventListener('change', async () => { updateModelCard(); await loadSelected(); });
    $('pg-device').addEventListener('change', async () => { updateModelCard(); await loadSelected(); });
    $('pg-load').addEventListener('click', loadSelected);
    $('pg-threads').addEventListener('change', async () => { if (selectedDevice() === 'cpu') await loadSelected(); });
    function updateSamplingControls() {
        const mode = $('pg-sampling-mode').value;
        const locked = mode !== 'sample';
        $('pg-temperature').disabled = locked;
        $('pg-top-p').disabled = locked;
        $('pg-top-k').disabled = locked;
        $('pg-temp-output').textContent = mode === 'auto' ? 'AUTO' : mode === 'greedy' ? 'GREEDY' : Number($('pg-temperature').value).toFixed(2);
    }
    $('pg-sampling-mode').addEventListener('change', () => {
        if ($('pg-sampling-mode').value === 'sample' && Number($('pg-temperature').value) <= 0) $('pg-temperature').value = '0.7';
        updateSamplingControls();
    });
    $('pg-temperature').addEventListener('input', event => $('pg-temp-output').textContent = Number(event.target.value).toFixed(2));
    $('pg-top-p').addEventListener('input', event => $('pg-top-p-output').textContent = Number(event.target.value).toFixed(2));
    $('pg-clear').addEventListener('click', () => { messages = []; localStorage.removeItem(storageKey); renderMessages(); });
    $('pg-form').addEventListener('submit', async event => {
        event.preventDefault(); const input = $('pg-input'), content = input.value.trim(), artifact = selectedArtifact(), device = selectedDevice(); if (!content || !artifact) return;
        const payloadMessages = [...messages.map(({ role, content }) => ({ role, content })), { role: 'user', content }]; messages.push({ role: 'user', content }); const assistant = { role: 'assistant', content: '', streaming: true, meta: `Streaming · ${device.toUpperCase()}` }; messages.push(assistant); input.value = ''; renderMessages(); $('pg-send').disabled = true;
        try {
            const response = await fetch('/api/playground/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ artifact: artifact.name, device, threads: Number($('pg-threads').value), messages: payloadMessages, sampling_mode: $('pg-sampling-mode').value, temperature: Number($('pg-temperature').value), top_p: Number($('pg-top-p').value), top_k: Number($('pg-top-k').value), max_new_tokens: Number($('pg-max-tokens').value), repetition_penalty: 1.05, seed: Number($('pg-seed').value) }) });
            if (!response.ok) { const body = await response.json(); throw new Error(body.detail || 'Falha no runtime.'); }
            const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
            while (true) { const { value, done } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done }); const lines = buffer.split('\n'); buffer = lines.pop() || ''; for (const line of lines) { if (!line.trim()) continue; const item = JSON.parse(line); if (item.type === 'token') { assistant.content += item.text; renderMessages(); } else if (item.type === 'done') { assistant.streaming = false; assistant.meta = `${item.token_count} tokens · ${item.tokens_per_second} tok/s · ${item.elapsed_seconds}s · ${item.device.toUpperCase()}`; } else if (item.type === 'error') throw new Error(item.message); } if (done) break; }
            assistant.streaming = false; saveMessages(); renderMessages();
        } catch (error) { messages.pop(); saveMessages(); renderMessages(); toast(error.message, true); } finally { $('pg-send').disabled = false; input.focus(); }
    });
    updateSamplingControls();
    status();
})();
