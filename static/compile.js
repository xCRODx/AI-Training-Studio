(() => {
    const $ = id => document.getElementById(id);
    const fmt = (value, digits = 0) => Number(value || 0).toLocaleString('pt-BR', { maximumFractionDigits: digits });
    const labels = { fine_tuning_model: 'Modelo base_finetuning', cmake: 'CMake', clang: 'Clang / Clang++', retnet_core: 'Núcleo de inferência', architecture_supported: 'Runtime de inferência' };
    const targetMeta = {
        'retbit-cpp': { format: 'GGUF v3', quant: 'Pesos ternários empacotados', target: 'Runtime C++ nativo', foot: 'Conversão GGUF e execução em runtime C++ nativo.' },
        'bitnet.cpp': { format: 'GGUF Llama-compatible', quant: 'Pesos ternários efetivos em BF16', target: 'bitnet.cpp nativo', foot: 'Grafo treinado preservado e validado no bitnet.cpp.' },
        'llama.cpp': { format: 'GGUF llama', quant: 'BF16 (quantizável no llama.cpp)', target: 'llama.cpp nativo', foot: 'Artefato validado e executado pelo llama.cpp.' }
    };
    function toast(message, error = false) { const el = $('toast'); el.textContent = message; el.className = `toast show${error ? ' error' : ''}`; setTimeout(() => el.className = 'toast', 4200); }
    function renderArtifacts(items) {
        $('artifact-count').textContent = `${items.length} modelos`; const list = $('artifact-list'); list.textContent = '';
        if (!items.length) { const empty = document.createElement('div'); empty.className = 'empty-row'; empty.textContent = 'Nenhum modelo compilado.'; list.appendChild(empty); return; }
        items.forEach(item => { const row = document.createElement('div'); row.className = 'artifact'; const info = document.createElement('div'); const name = document.createElement('strong'); const path = document.createElement('span'); name.textContent = item.name; path.textContent = `${item.path} · ${targetMeta[item.target]?.target || item.target || 'Runtime C++ nativo'}`; info.append(name, path); const size = document.createElement('b'); size.textContent = `${fmt(item.size_mb, 1)} MB`; row.append(info, size); list.appendChild(row); });
    }
    function renderTarget(data) {
        const target = data.target || 'retbit-cpp';
        const meta = targetMeta[target] || targetMeta['retbit-cpp'];
        if ($('pipeline-arch')) $('pipeline-arch').textContent = data.architecture_label || (data.architecture || '?').toUpperCase();
        if ($('target-chip')) $('target-chip').textContent = meta.target;
        if ($('target-foot')) $('target-foot').textContent = meta.foot;
        if ($('diagram-format')) $('diagram-format').textContent = meta.format;
        if ($('diagram-quant')) $('diagram-quant').textContent = meta.quant;
        if ($('diagram-target')) $('diagram-target').textContent = meta.target;
        if ($('compile-kv-cache')) $('compile-kv-cache').textContent = data.kv_cache?.enabled ? 'KV cache · runtime' : 'Estado recorrente';
    }
    async function refresh() {
        try {
            const response = await fetch('/api/compile/preflight'); const data = await response.json();
            $('compile-status').className = `connection ${data.ready ? 'online' : 'offline'}`; $('compile-status').innerHTML = `<i></i>${data.ready ? 'Pronto' : 'Bloqueado'}`;
            $('route-status').textContent = data.ready ? 'Pronto para compilar' : 'Ação necessária'; $('route-status').className = `route-status${data.ready ? ' ready' : ''}`;
            $('source-badge').textContent = data.model.valid ? 'íntegro' : 'ausente'; $('source-path').textContent = data.model.path || '—';
            $('source-params').textContent = data.model.parameter_count ? fmt(data.model.parameter_count) : '—'; $('source-tensors').textContent = data.model.tensor_count ? fmt(data.model.tensor_count) : '—'; $('source-size').textContent = data.model.size_bytes ? `${fmt(data.model.size_bytes / 1024 ** 2, 1)} MB` : '—';
            const list = $('check-list'); list.textContent = ''; Object.entries(data.checks).forEach(([key, pass]) => { const row = document.createElement('div'); row.className = `check-item ${pass ? 'pass' : 'fail'}`; const dot = document.createElement('i'); const title = document.createElement('strong'); title.textContent = labels[key] || key; const status = document.createElement('small'); const detail = data.check_details?.[key]?.version; const compactDetail = detail?.replace(/^cmake version /i, 'v').replace(/^clang version /i, 'v'); status.textContent = pass ? (compactDetail || 'OK') : (key === 'architecture_supported' ? 'A IMPLEMENTAR' : 'PENDENTE'); if (data.check_details?.[key]?.path) status.title = data.check_details[key].path; row.append(dot, title, status); list.appendChild(row); });
            $('compile-blocker').textContent = data.ready ? 'Todos os requisitos foram atendidos.' : data.blocker; $('btn-compile').disabled = !data.ready; renderArtifacts(data.artifacts || []); renderTarget(data);
        } catch (error) { toast(`Falha no preflight: ${error.message}`, true); }
    }
    $('btn-refresh').addEventListener('click', refresh);
    $('btn-compile').addEventListener('click', async () => { const button = $('btn-compile'); button.disabled = true; const old = button.textContent; button.textContent = 'Compilando…'; try { const response = await fetch('/api/compile/start', { method: 'POST' }); const body = await response.json(); if (!response.ok) throw new Error(body.detail); const target = targetMeta[body.artifact.target]?.target || body.artifact.target || 'Runtime nativo'; toast(`Modelo compilado: ${body.artifact.size_mb} MB · ${target}`); await refresh(); } catch (error) { toast(error.message, true); } finally { button.textContent = old; } });
    refresh();
})();
