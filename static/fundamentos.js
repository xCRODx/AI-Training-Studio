// ============================================================
// Fundamentos · Renderização de markdown, LaTeX, Mermaid e busca
// ============================================================
const DOCS_BASE = '/docs';

const state = {
    sections: [],
    flat: [],
    contentCache: new Map(),
};

let mermaidSeq = 0;

// ------------------------------------------------------------
// Configuração das bibliotecas
// ------------------------------------------------------------
if (window.marked) {
    marked.setOptions({ gfm: true, breaks: false, smartLists: true });
}
if (window.mermaid) {
    mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        theme: 'base',
        fontFamily: 'ui-monospace, monospace',
        flowchart: { useMaxWidth: true, htmlLabels: true, curve: 'basis', padding: 12 },
        themeVariables: {
            background: '#0c1017',
            primaryColor: '#171c26',
            primaryTextColor: '#f5f7fb',
            primaryBorderColor: '#54d9ff',
            secondaryColor: '#1b2230',
            secondaryTextColor: '#cbd2dd',
            secondaryBorderColor: '#a98cff',
            tertiaryColor: '#0c1017',
            tertiaryTextColor: '#8d96a7',
            lineColor: '#8d96a7',
            textColor: '#e6eaf2',
            mainBkg: '#0c1017',
            nodeBorder: '#54d9ff',
            nodeTextColor: '#f5f7fb',
            clusterBkg: '#11151d',
            clusterBorder: 'rgba(84,217,255,.22)',
            titleColor: '#f5f7fb',
            edgeLabelBackground: '#171c26',
            fontSize: '14px',
        },
    });
}

// ------------------------------------------------------------
// Utilidades
// ------------------------------------------------------------
// Cache diário dos arquivos .md: evita rebaixar tudo a cada carregamento.
const CACHE_NAME = 'ai-training-studio-docs';
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const LS_PREFIX = 'ai-training-studio-docs:';

function cacheApiAvailable() {
    return typeof caches !== 'undefined' && typeof caches.open === 'function';
}

async function readFromCache(url) {
    if (cacheApiAvailable()) {
        try {
            const cache = await caches.open(CACHE_NAME);
            const hit = await cache.match(url);
            if (hit) {
                const data = await hit.json();
                if (Date.now() - data.savedAt < CACHE_TTL_MS) return data.text;
                await cache.delete(url);
            }
        } catch (_) {}
        return null;
    }
    try {
        const raw = localStorage.getItem(LS_PREFIX + url);
        if (!raw) return null;
        const data = JSON.parse(raw);
        if (Date.now() - data.savedAt < CACHE_TTL_MS) return data.text;
        localStorage.removeItem(LS_PREFIX + url);
    } catch (_) {}
    return null;
}

async function writeToCache(url, text) {
    const payload = JSON.stringify({ text, savedAt: Date.now() });
    if (cacheApiAvailable()) {
        try {
            const cache = await caches.open(CACHE_NAME);
            await cache.put(url, new Response(payload));
            return;
        } catch (_) {}
    }
    try {
        localStorage.setItem(LS_PREFIX + url, payload);
    } catch (_) {}
}

async function fetchText(url) {
    const cached = await readFromCache(url);
    if (cached !== null) return cached;
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    await writeToCache(url, text);
    return text;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ------------------------------------------------------------
// Índice (index.md)
// ------------------------------------------------------------
function parseIndex(md) {
    const lines = md.split(/\r?\n/);
    const sections = [];
    let current = { title: '', topics: [] };

    const flush = () => {
        if (current.title || current.topics.length) {
            sections.push(current);
        }
    };

    lines.forEach((line) => {
        const t = line.trim();
        if (t.startsWith('## ')) {
            flush();
            current = { title: t.slice(3).trim(), topics: [] };
        } else {
            const m = t.match(/^[-*]\s+\[([^\]]+)\]\(([^)\s]+)\)/);
            if (m) {
                current.topics.push({ title: m[1].trim(), file: m[2].trim().replace(/^\.\//, '') });
            }
        }
    });
    flush();

    // Se houver tópicos soltos antes da primeira seção, agrupa sob um título padrão.
    if (sections.length === 1 && !sections[0].title && sections[0].topics.length) {
        sections[0].title = 'Tópicos';
    }
    return sections;
}

function renderIndex() {
    const root = document.getElementById('docs-root');
    root.innerHTML = '';
    state.flat = [];
    let globalIdx = 0;

    state.sections.forEach((section, si) => {
        const secEl = document.createElement('section');
        secEl.className = 'docs-section';
        secEl.dataset.section = String(si);

        const head = document.createElement('div');
        head.className = 'docs-section-head';
        head.innerHTML = '<span class="section-dot"></span><h3></h3><small></small>';
        head.querySelector('h3').textContent = section.title;
        head.querySelector('small').textContent = `${section.topics.length} tópico(s)`;
        secEl.appendChild(head);

        section.topics.forEach((topic) => {
            const details = document.createElement('details');
            details.className = 'topic';
            details.dataset.file = topic.file;
            details.innerHTML =
                '<summary>' +
                '<span class="topic-index"></span>' +
                '<span class="topic-title"></span>' +
                '<span class="topic-chevron">▾</span>' +
                '</summary>' +
                '<div class="doc-content"></div>';

            details.querySelector('.topic-index').textContent = String(++globalIdx).padStart(2, '0');
            details.querySelector('.topic-title').textContent = topic.title;
            details.addEventListener('toggle', () => {
                if (details.open && !details.dataset.loaded) {
                    loadTopic(details, topic.file);
                }
            });
            secEl.appendChild(details);
            state.flat.push({ section: section.title, title: topic.title, file: topic.file, el: details });
        });

        root.appendChild(secEl);
    });
}

// ------------------------------------------------------------
// Renderização de markdown + LaTeX
// ------------------------------------------------------------
function protectRegions(md) {
    const stash = [];
    const save = (s) => {
        stash.push(s);
        return `\u0000S${stash.length - 1}\u0000`;
    };

    // Protege blocos de código (```) e spans inline (`) antes de mexer na matemática.
    md = md.replace(/```[\s\S]*?```/g, save);
    md = md.replace(/`[^`\n]+`/g, save);

    const tokens = [];
    md = md.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
        tokens.push({ display: true, tex: tex.trim() });
        return `@@KB${tokens.length - 1}@@`;
    });
    md = md.replace(/(^|[^$])\$([^$\n]+?)\$/g, (_, p, tex) => {
        tokens.push({ display: false, tex: tex.trim() });
        return `${p}@@KI${tokens.length - 1}@@`;
    });

    md = md.replace(/\u0000S(\d+)\u0000/g, (_, i) => stash[+i]);
    return { md, tokens };
}

function renderMarkdown(md) {
    const { md: cleaned, tokens } = protectRegions(md);
    let html = window.marked ? marked.parse(cleaned) : cleaned;

    // Matemática em bloco ($$...$$) em linha própria sai do <p>.
    html = html.replace(/<p>\s*@@KB(\d+)@@\s*<\/p>/g, (_, i) => {
        const t = tokens[+i];
        return window.katex
            ? katex.renderToString(t.tex, { displayMode: true, throwOnError: false, strict: false })
            : `$$${t.tex}$$`;
    });

    // Restante (inline e quaisquer blocos fora do padrão acima).
    html = html.replace(/@@K([BI])(\d+)@@/g, (_, kind, i) => {
        const t = tokens[+i];
        if (!window.katex) return t.tex;
        return katex.renderToString(t.tex, { displayMode: kind === 'B', throwOnError: false, strict: false });
    });

    return html;
}

// ------------------------------------------------------------
// Pós-processamento: Mermaid e syntax highlighting
// ------------------------------------------------------------
async function renderMermaidBlocks(container) {
    const blocks = container.querySelectorAll('pre code.language-mermaid');
    for (const code of blocks) {
        const text = code.textContent;
        const wrapper = document.createElement('div');
        wrapper.className = 'mermaid-wrapper';
        code.parentElement.replaceWith(wrapper);
        const id = 'mmd' + (mermaidSeq++) + '_' + Math.floor(Math.random() * 1e6);
        try {
            if (!window.mermaid) throw new Error('Mermaid indisponível');
            const { svg } = await mermaid.render(id, text);
            wrapper.innerHTML = svg;
        } catch (e) {
            wrapper.classList.add('mermaid-error');
            wrapper.textContent = 'Erro ao renderizar o diagrama: ' + e.message;
        }
    }
}

function highlightCode(container) {
    if (!window.hljs) return;
    container.querySelectorAll('pre code').forEach((block) => {
        if (block.classList.contains('language-mermaid')) return;
        try {
            hljs.highlightElement(block);
        } catch (_) {
            /* ignora blocos não suportados */
        }
    });
}

// ------------------------------------------------------------
// Carregamento sob demanda de um tópico
// ------------------------------------------------------------
async function loadTopic(details, file) {
    const content = details.querySelector('.doc-content');
    content.innerHTML = '<div class="doc-content-empty">Carregando...</div>';
    try {
        const md = await fetchText(`${DOCS_BASE}/${file}`);
        const html = renderMarkdown(md);
        content.innerHTML = html;
        await renderMermaidBlocks(content);
        highlightCode(content);
        details.dataset.loaded = '1';
    } catch (e) {
        content.innerHTML = `<div class="doc-content-empty">Não foi possível carregar este tópico (${escapeHtml(e.message)}).</div>`;
    }
}

// ------------------------------------------------------------
// Busca
// ------------------------------------------------------------
async function preloadContents() {
    const files = [...new Set(state.flat.map((t) => t.file))];
    await Promise.all(
        files.map(async (f) => {
            try {
                state.contentCache.set(f, await fetchText(`${DOCS_BASE}/${f}`));
            } catch (_) {
                state.contentCache.set(f, '');
            }
        }),
    );
}

function highlightTitle(text, q) {
    if (!q) return escapeHtml(text);
    const idx = text.toLowerCase().indexOf(q);
    if (idx < 0) return escapeHtml(text);
    return (
        escapeHtml(text.slice(0, idx)) +
        '<mark>' + escapeHtml(text.slice(idx, idx + q.length)) + '</mark>' +
        escapeHtml(text.slice(idx + q.length))
    );
}

function applySearch(query) {
    const q = query.toLowerCase().trim();
    let visible = 0;

    state.flat.forEach((t) => {
        const inTitle = t.title.toLowerCase().includes(q);
        const raw = state.contentCache.get(t.file) || '';
        const inContent = raw.toLowerCase().includes(q);
        const match = !q || inTitle || inContent;
        t.el.classList.toggle('match-hidden', !match);
        if (match) visible++;
        t.el.querySelector('.topic-title').innerHTML = highlightTitle(t.title, q);
    });

    document.querySelectorAll('.docs-section').forEach((sec) => {
        const any = [...sec.querySelectorAll('.topic')].some((el) => !el.classList.contains('match-hidden'));
        sec.style.display = any ? '' : 'none';
    });

    const count = document.getElementById('search-count');
    if (q) {
        count.classList.remove('hidden');
        count.textContent = `${visible} resultado(s)`;
    } else {
        count.classList.add('hidden');
    }

    const root = document.getElementById('docs-root');
    let nr = root.querySelector('.no-results');
    if (q && visible === 0) {
        if (!nr) {
            nr = document.createElement('div');
            nr.className = 'no-results';
            nr.innerHTML = '<strong>Nenhum tópico encontrado.</strong>Tente outro termo.';
            root.appendChild(nr);
        }
    } else if (nr) {
        nr.remove();
    }
}

// ------------------------------------------------------------
// Inicialização
// ------------------------------------------------------------
async function init() {
    try {
        const md = await fetchText(`${DOCS_BASE}/index.md`);
        state.sections = parseIndex(md);
        renderIndex();
        preloadContents();
    } catch (e) {
        document.getElementById('docs-root').innerHTML =
            `<div class="no-results"><strong>Índice indisponível.</strong>${escapeHtml(e.message)}</div>`;
    }

    const input = document.getElementById('docs-search');
    let debounce;
    input.addEventListener('input', () => {
        clearTimeout(debounce);
        debounce = setTimeout(() => applySearch(input.value), 150);
    });
}

init();
