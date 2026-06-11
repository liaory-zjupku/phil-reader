import os
import uuid
import json
import re
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(os.environ.get('DATA_DIR',    'data'))
UPLOAD_DIR  = Path(os.environ.get('UPLOAD_DIR',  'uploads'))
WIKIS_DIR   = BASE_DIR / 'wikis'
CHUNKS_DIR  = BASE_DIR / 'chunks'
DOCS_FILE   = BASE_DIR / 'documents.json'

for _d in [BASE_DIR, UPLOAD_DIR, WIKIS_DIR, CHUNKS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder='.')
CORS(app)

# Claude 从环境变量 CLAUDE_KEY 读取默认 Key，用户不填时自动使用。
# DeepSeek / 通义千问无默认 Key，用户必须在界面自行填入才能使用。
# 注意：不在模块级缓存，每次调用时实时读取，确保部署平台的环境变量始终生效。
def _default_key(provider: str) -> str:
    if provider == 'claude':
        return os.environ.get('CLAUDE_KEY', '')
    return ''

MODEL_CONFIG = {
    "claude":   {"model": "claude-sonnet-4-6", "api_type": "anthropic"},
    "deepseek": {"model": "deepseek-chat",     "api_type": "openai",
                 "base_url": "https://api.deepseek.com"},
    "qwen":     {"model": "qwen-plus",         "api_type": "openai",
                 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
}

# ── Runtime state (in-memory, not persisted) ───────────────────────────────────
_clients       = {}   # "provider_key8" -> client object
_conversations = {}   # session_id -> [{"role","content"}]

# ── Persistence helpers ────────────────────────────────────────────────────────
def _load_documents() -> dict:
    if DOCS_FILE.exists():
        try:
            return json.loads(DOCS_FILE.read_text('utf-8'))
        except Exception:
            pass
    return {}

def _save_documents():
    DOCS_FILE.write_text(json.dumps(_documents, ensure_ascii=False, indent=2), 'utf-8')

def _load_chunks(doc_id: str) -> list:
    f = CHUNKS_DIR / f'{doc_id}.json'
    if f.exists():
        try:
            return json.loads(f.read_text('utf-8'))
        except Exception:
            pass
    return []

def _save_chunks(doc_id: str, chunks: list):
    (CHUNKS_DIR / f'{doc_id}.json').write_text(
        json.dumps(chunks, ensure_ascii=False), 'utf-8')

def _load_wiki(doc_id: str) -> dict | None:
    f = WIKIS_DIR / f'{doc_id}.json'
    if f.exists():
        try:
            return json.loads(f.read_text('utf-8'))
        except Exception:
            pass
    return None

def _save_wiki(doc_id: str, wiki: dict):
    (WIKIS_DIR / f'{doc_id}.json').write_text(
        json.dumps(wiki, ensure_ascii=False, indent=2), 'utf-8')

_documents: dict = _load_documents()

# ── Parsers ────────────────────────────────────────────────────────────────────
def _make_chunks(text: str, page: int, size=450, step=380) -> list:
    chunks = []
    for i in range(0, len(text), step):
        piece = text[i:i + size].strip()
        if len(piece) > 40:
            chunks.append({'page': page, 'text': piece})
    return chunks

def parse_pdf(filepath) -> tuple[list, int]:
    import pypdf
    reader = pypdf.PdfReader(str(filepath))
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        if len(text.strip()) >= 20:
            chunks.extend(_make_chunks(text, i + 1))
    return chunks, len(reader.pages)

def parse_epub(filepath) -> tuple[list, int]:
    import zipfile
    from bs4 import BeautifulSoup

    chunks, page_num = [], 0
    with zipfile.ZipFile(str(filepath), 'r') as zf:
        # 找出所有 html/xhtml 文件，按路径排序保证顺序稳定
        html_names = sorted(
            name for name in zf.namelist()
            if name.lower().endswith(('.html', '.xhtml', '.htm'))
        )
        for name in html_names:
            try:
                raw = zf.read(name)
                soup = BeautifulSoup(raw, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
            except Exception:
                continue
            if len(text.strip()) < 20:
                continue
            page_num += 1
            chunks.extend(_make_chunks(text, page_num))

    return chunks, page_num

def parse_docx(filepath) -> tuple[list, int]:
    from docx import Document
    doc = Document(str(filepath))
    full = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    chunks = _make_chunks(full, 1)
    for i, c in enumerate(chunks):
        c['page'] = i // 4 + 1
    return chunks, max((c['page'] for c in chunks), default=1)

def parse_txt(filepath) -> tuple[list, int]:
    text = Path(filepath).read_text(encoding='utf-8', errors='ignore')
    chunks = []
    for i in range(0, len(text), 380):
        piece = text[i:i + 450].strip()
        if len(piece) > 40:
            chunks.append({'page': i // 380 // 4 + 1, 'text': piece})
    return chunks, max((c['page'] for c in chunks), default=1)

# ── Search ─────────────────────────────────────────────────────────────────────
def _search(query: str, doc_ids: list | None = None, top_k: int = 6) -> list:
    q = query.lower()
    grams = {q[i:i+2] for i in range(len(q) - 1)} | set(q.split())
    grams = {w for w in grams if len(w) >= 2}

    results = []
    target = doc_ids if doc_ids else list(_documents.keys())
    for did in target:
        if did not in _documents:
            continue
        for chunk in _load_chunks(did):
            t = chunk['text'].lower()
            score = sum(1 for w in grams if w in t)
            if score > 0:
                results.append({
                    'score': score,
                    'doc_id': did,
                    'name': _documents[did]['name'],
                    'page': chunk['page'],
                    'text': chunk['text'],
                })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# ── LLM client & caller ────────────────────────────────────────────────────────
def _get_client(provider: str, api_key: str | None = None):
    key = api_key or _default_key(provider)
    if not key:
        return None, f'未配置 {provider} 的 API Key'
    ck = f'{provider}_{key[:8]}'
    if ck in _clients:
        return _clients[ck], None
    try:
        cfg = MODEL_CONFIG[provider]
        if cfg['api_type'] == 'anthropic':
            from anthropic import Anthropic
            c = Anthropic(api_key=key)
        else:
            from openai import OpenAI
            c = OpenAI(api_key=key, base_url=cfg['base_url'])
        _clients[ck] = c
        return c, None
    except Exception as e:
        return None, str(e)

def _call_llm(provider: str, client, messages: list,
              system: str | None = None, max_tokens: int = 2000) -> str:
    cfg = MODEL_CONFIG[provider]
    if cfg['api_type'] == 'anthropic':
        kw = dict(model=cfg['model'], max_tokens=max_tokens, messages=messages)
        if system:
            kw['system'] = system
        return client.messages.create(**kw).content[0].text
    else:
        oai = ([{'role': 'system', 'content': system}] if system else []) + messages
        return client.chat.completions.create(
            model=cfg['model'], max_tokens=max_tokens, messages=oai
        ).choices[0].message.content

# ── System prompts ─────────────────────────────────────────────────────────────
READER_SYSTEM = """你是"诠释者"——专为哲学与人文社科文本深度阅读设计的智能体。

## 核心原则
1. 分层递进：先给最重要的一层（问题意识 + 核心论点），等用户追问再深入
2. 每次回复聚焦一个层面，不要一次性全部输出
3. 主动引导：每次回复末尾给出 2-3 个追问方向

## 第一轮只做：
- 还原问题意识（这段文本在回应什么问题）
- 核心论点（一句话概括）
- 文本类型标注
- 提出 2-3 个追问方向

## 后续根据追问展开：
- 前置知识 → 补充概念和背景
- 论证结构 → P1/P2→C 格式拆解，附原文引用
- 思想来源 → 追溯哲学史上游和下游
- 术语 → 关键术语在该语境的精确含义

## 引用规范
- 引用原文时注明来源文献和页码
- 区分「原文明确说的」和「你的推断」
- 不确定时直接说明

语气：像耐心的、知识渊博的学长在带你读书。"""

WIKI_SYSTEM = """你是哲学文本分析专家。根据提供的文本摘样，生成结构化 Wiki 分析报告。
严格返回合法 JSON（不加 ``` 标记，不加注释），格式：
{
  "thesis": "整体核心论点（2-4句，概括全文核心主张）",
  "chapter_structure": [
    {"title": "章节标题", "summary": "核心内容一句话", "key_claim": "该章节的核心论断"}
  ],
  "core_concepts": [
    {"term": "术语", "definition": "在本文语境的精确含义", "related": ["相关术语1"]}
  ],
  "key_figures": [
    {"name": "人物/思想家", "role": "在文中角色或被引用方式", "relation": "与作者立场的关系"}
  ],
  "intellectual_sources": [
    {"source": "思想来源/传统", "influence": "对本文的具体影响"}
  ],
  "prerequisites": ["前置知识点1", "前置知识点2"],
  "philosophical_problems": [
    {"problem": "哲学史经典问题", "treatment": "本文如何处理或回应"}
  ]
}
chapter_structure ≤ 8 项，core_concepts ≤ 10 项，key_figures ≤ 8 项。所有内容从文本提炼，不确定处标注（据文本推断）。"""

def _sample_for_wiki(chunks: list, max_chars: int = 7000) -> str:
    if not chunks:
        return ''
    n = len(chunks)
    if n <= 25:
        selected = chunks
    else:
        mid = n // 2
        idxs = (list(range(min(10, n))) +
                list(range(mid - 3, min(mid + 4, n))) +
                list(range(max(0, n - 6), n)))
        idxs = sorted(set(i for i in idxs if 0 <= i < n))
        selected = [chunks[i] for i in idxs]
    return '\n\n'.join(c['text'] for c in selected)[:max_chars]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/documents', methods=['GET'])
def get_documents():
    return jsonify({'documents': list(_documents.values())})

@app.route('/api/check-defaults', methods=['GET'])
def check_defaults():
    return jsonify({'available': {p: bool(_default_key(p)) for p in ('claude', 'deepseek', 'qwen')}})

@app.route('/api/set-key', methods=['POST'])
def set_key():
    data = request.json or {}
    provider = data.get('provider', 'claude')
    api_key  = data.get('api_key', '').strip()
    client, err = _get_client(provider, api_key)
    if err:
        return jsonify({'error': err}), 400
    return jsonify({'success': True})

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    file = request.files['file']
    fname = file.filename or 'untitled'
    ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''
    if ext not in ('pdf', 'epub', 'docx', 'txt', 'md'):
        return jsonify({'error': '支持格式：PDF、EPUB、DOCX、TXT'}), 400

    doc_id = uuid.uuid4().hex[:8]
    saved  = UPLOAD_DIR / f'{doc_id}_{fname}'
    file.save(str(saved))

    try:
        if ext == 'pdf':
            chunks, pages = parse_pdf(saved)
        elif ext == 'epub':
            chunks, pages = parse_epub(saved)
        elif ext == 'docx':
            chunks, pages = parse_docx(saved)
        else:
            chunks, pages = parse_txt(saved)
    except Exception as e:
        saved.unlink(missing_ok=True)
        return jsonify({'error': f'解析失败：{e}'}), 500

    meta = {
        'id': doc_id, 'name': fname,
        'pages': pages, 'chunks_count': len(chunks),
        'has_wiki': False, 'filepath': str(saved),
    }
    _save_chunks(doc_id, chunks)
    _documents[doc_id] = meta
    _save_documents()
    return jsonify({'success': True, **meta})

@app.route('/api/delete-doc', methods=['POST'])
def delete_doc():
    doc_id = (request.json or {}).get('doc_id')
    if doc_id not in _documents:
        return jsonify({'error': '文献不存在'}), 404
    meta = _documents.pop(doc_id)
    _save_documents()
    for f in [CHUNKS_DIR / f'{doc_id}.json',
              WIKIS_DIR  / f'{doc_id}.json',
              Path(meta.get('filepath', ''))]:
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:
            pass
    return jsonify({'success': True})

@app.route('/api/generate-wiki', methods=['POST'])
def generate_wiki():
    data     = request.json or {}
    doc_id   = data.get('doc_id')
    provider = data.get('provider', 'claude')
    api_key  = data.get('api_key', '').strip() or None

    if doc_id not in _documents:
        return jsonify({'error': '文献不存在'}), 404

    client, err = _get_client(provider, api_key)
    if not client:
        return jsonify({'error': f'模型未配置：{err}'}), 400

    chunks = _load_chunks(doc_id)
    sample = _sample_for_wiki(chunks)
    if not sample:
        return jsonify({'error': '文献内容为空'}), 400

    user_msg = f'文献名称：{_documents[doc_id]["name"]}\n\n文本摘样：\n\n{sample}'
    try:
        raw = _call_llm(provider, client,
                        [{'role': 'user', 'content': user_msg}],
                        system=WIKI_SYSTEM, max_tokens=3000)
        raw = raw.strip()
        if raw.startswith('```'):
            raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        wiki = json.loads(raw)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Wiki JSON 解析失败：{e}'}), 500
    except Exception as e:
        return jsonify({'error': f'生成失败：{e}'}), 500

    _save_wiki(doc_id, wiki)
    _documents[doc_id]['has_wiki'] = True
    _save_documents()
    return jsonify({'success': True, 'wiki': wiki})

@app.route('/api/wiki/<doc_id>', methods=['GET'])
def get_wiki(doc_id):
    wiki = _load_wiki(doc_id)
    if wiki is None:
        return jsonify({'error': 'Wiki 尚未生成'}), 404
    return jsonify({'wiki': wiki})

@app.route('/api/new-session', methods=['POST'])
def new_session():
    sid = uuid.uuid4().hex[:8]
    _conversations[sid] = []
    return jsonify({'session_id': sid})

@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json or {}
    message    = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    provider   = data.get('provider', 'claude')
    api_key    = data.get('api_key', '').strip() or None
    user_bg    = data.get('user_bg', '')
    doc_ids    = data.get('doc_ids') or list(_documents.keys())

    if not message:
        return jsonify({'error': '请输入内容'}), 400

    client, err = _get_client(provider, api_key)
    if not client:
        return jsonify({'error': f'模型未配置：{err}'}), 400

    if session_id not in _conversations:
        _conversations[session_id] = []

    # Build system prompt
    system = READER_SYSTEM
    if user_bg:
        system += f'\n\n读者背景：{user_bg}'

    wiki_parts = []
    for did in doc_ids:
        w = _load_wiki(did)
        if w:
            name  = _documents.get(did, {}).get('name', did)
            terms = '、'.join(c['term'] for c in w.get('core_concepts', [])[:5])
            wiki_parts.append(f'【{name}】\n论点：{w.get("thesis","")}\n核心概念：{terms}')
    if wiki_parts:
        system += '\n\n## 已加载文献 Wiki 摘要\n\n' + '\n\n'.join(wiki_parts)

    # Retrieve chunks
    relevant = _search(message, doc_ids=doc_ids, top_k=6)
    sources  = []
    if relevant:
        ctx, seen = [], set()
        for r in relevant:
            ctx.append(f'[{r["name"]} 第{r["page"]}页]\n{r["text"]}')
            k = f'{r["doc_id"]}_p{r["page"]}'
            if k not in seen:
                seen.add(k)
                sources.append({'doc_id': r['doc_id'], 'name': r['name'],
                                 'page': r['page'], 'text': r['text']})
        aug = message + '\n\n## 相关原文片段\n\n' + '\n\n---\n\n'.join(ctx)
    else:
        aug = message + '\n\n（未检索到直接相关原文，请基于你的知识回答并注明这是推断）'

    conv = _conversations[session_id]
    conv.append({'role': 'user', 'content': aug})
    if len(conv) > 20:
        _conversations[session_id] = conv[-20:]
        conv = _conversations[session_id]

    try:
        reply = _call_llm(provider, client, conv, system=system)
        conv.append({'role': 'assistant', 'content': reply})
        return jsonify({'success': True, 'reply': reply, 'sources': sources})
    except Exception as e:
        return jsonify({'error': f'调用失败：{e}'}), 500

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('诠释者 · 哲学 Wiki 阅读智能体')
    for p, k in DEFAULT_KEYS.items():
        status = 'OK' if k else '-- (请在界面填入 Key)'
        print(f'  {p}: {status}')
    print('http://localhost:5000')
    app.run(debug=False, port=5000)
