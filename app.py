import os
import sys
import uuid
import json
import re
from pathlib import Path

# Windows 控制台默认 GBK 编码，打印中文/特殊字符会崩溃；强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── 本地 .env 加载（无依赖）。Railway/Render 用平台环境变量，不需要此文件 ──────────
def _load_dotenv():
    f = Path('.env')
    if not f.exists():
        return
    for line in f.read_text('utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_dotenv()

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

# Claude 从环境变量 ANTHROPIC_API_KEY 读取（本地走 .env，Railway/Render 走平台变量）
# DeepSeek / 通义千问无默认 Key，用户在界面自行填入
def _default_key(provider: str) -> str:
    if provider == "claude":
        return os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return ""

MODEL_CONFIG = {
    "claude":   {"model": "claude-sonnet-4-6", "api_type": "anthropic"},
    "deepseek": {"model": "deepseek-chat",     "api_type": "openai",
                 "base_url": "https://api.deepseek.com"},
    "qwen":     {"model": "qwen-plus",         "api_type": "openai",
                 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
}

# ── Runtime state (in-memory, not persisted) ───────────────────────────────────
_clients       = {}   # "provider_hash" -> openai.OpenAI client (仅 openai-compatible)
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

def _representative_chunks(doc_ids: list | None = None, top_k: int = 6) -> list:
    """检索兜底：关键词无命中时，从每篇文献里均匀取若干代表性片段
    （开头/四分位/中段/末尾），保证对话始终有原文可依。"""
    target = doc_ids if doc_ids else list(_documents.keys())
    out = []
    for did in target:
        if did not in _documents:
            continue
        chunks = _load_chunks(did)
        n = len(chunks)
        if n == 0:
            continue
        idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        for i in idxs:
            c = chunks[i]
            out.append({'doc_id': did, 'name': _documents[did]['name'],
                        'page': c['page'], 'text': c['text']})
    return out[:top_k]

# ── LLM client & caller ────────────────────────────────────────────────────────
def _get_client(provider: str, api_key: str | None = None):
    """返回 (client, error_str)。Claude 用 Anthropic SDK，其余用 OpenAI 兼容 SDK。
    两个 SDK 底层都用 httpx/openai，会自动读取 HTTPS_PROXY 环境变量走系统代理。"""
    key = (api_key or _default_key(provider) or '').strip()
    if not key:
        return None, f'未配置 {provider} 的 API Key'

    import hashlib
    ck = f'{provider}_{hashlib.md5(key.encode()).hexdigest()[:12]}'
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

def _is_auth_error(e: Exception) -> bool:
    s = str(e).lower()
    return (getattr(e, 'status_code', None) == 401
            or '401' in s or 'authentication' in s or 'invalid x-api-key' in s)

def _llm_call(provider: str, api_key: str | None, messages: list,
              system: str | None = None, max_tokens: int = 2000) -> str:
    """构建 client 并调用；若用户自带 key 认证失败，自动回退到默认 key 重试。"""
    client, err = _get_client(provider, api_key)
    if not client:
        raise RuntimeError(err)
    try:
        return _call_llm(provider, client, messages, system=system, max_tokens=max_tokens)
    except Exception as e:
        default = _default_key(provider)
        if api_key and default and api_key.strip() != default and _is_auth_error(e):
            print('[LLM] 用户自带 key 认证失败，回退默认 key 重试', flush=True)
            c2, err2 = _get_client(provider, default)
            if c2:
                return _call_llm(provider, c2, messages, system=system, max_tokens=max_tokens)
        raise

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

## 回答时必须主动做到（这是深度的来源，不要等用户追问）
1. 追溯哲学史脉络：遇到关键论点，主动指出它的上游来源和下游影响，把它放回哲学史的对话链条里，而不是孤立解释。
2. 引入论敌视角：主动呈现与本文观点对立的立场（文本内部的论敌，或哲学史上的对手），说明双方分歧何在、本文如何回应，让读者看到论争而非单方面陈述。
3. 概念给足解释：对核心概念给出详细、分层的解释（字面义、在该语境的精确义、与相邻概念的区别），不要一笔带过或只给一句话定义。

## 关于斯坦福哲学百科（SEP）
- 如果你的知识中包含与当前问题相关的 SEP（Stanford Encyclopedia of Philosophy）条目内容，请主动引用，并明确标注「（据斯坦福哲学百科 SEP）」。
- 这是为了让读者知道哪些是学界公认的权威梳理；不确定是否真出自 SEP 时，不要假托。

## 引用规范
- 引用原文时注明来源文献和页码（优先使用对话中提供的「相关原文片段」）
- 区分「原文明确说的」「学界共识（如 SEP）」和「你的推断」
- 不确定时直接说明

语气：像耐心的、知识渊博的学长在带你读书。"""

WIKI_SYSTEM = """你是哲学文本分析专家。根据提供的文本摘样，生成结构化、有深度、点与点之间有逻辑关联的 Wiki 分析报告。

【输出规则——必须严格遵守】
- 只输出一个 JSON 对象，不得有任何其他内容
- 不加 ```json 代码块，不加任何说明文字，不加注释
- 关键：字符串值内部需要引用或强调时，一律使用中文引号「」或书名号《》，绝对禁止使用英文双引号 "。英文双引号只能作为 JSON 的结构符号（包裹键和值），不得出现在值的文字内容里
- 不得有尾随逗号（最后一项后面不加逗号）
- 必须是可被 json.loads() 直接解析的合法 JSON

【内容质量要求——这是重点，不要写得简略或孤立】
1. chapter_structure：每个 summary 不少于 100 字，要说清这一章在论证什么、用什么方式论证、与前后章节的承接关系，不要只写一句话标题式概括。
2. core_concepts：每个概念除定义外，必须在 relation 字段说明它与其他核心概念的逻辑关系（谁是谁的前提、谁与谁对立、谁由谁推出），让概念之间连成网络，而不是孤立罗列。
3. intellectual_sources：必须写清「谁影响了谁、通过哪个观点、如何影响」的具体路径（例如 A 的某概念被 B 改造为某主张），不能只列人名或学派名。
4. debates：标注文本内部的争论对手与论战。说明本文在反对谁、争论焦点是什么、双方各自立场（例如格朗丹 vs 瓦蒂莫在诠释学激进化问题上的分歧）。文本中若无明显论敌可给空数组 []。
5. 不确定处加（据文本推断），但要尽量给出实质内容而非回避。

【JSON 结构】
{"thesis":"整体核心论点2-4句","chapter_structure":[{"title":"章节标题","summary":"不少于100字的章节分析","key_claim":"核心论断"}],"core_concepts":[{"term":"术语","definition":"本文语境含义","relation":"与其他核心概念的逻辑关系","related":["相关术语"]}],"key_figures":[{"name":"人物","role":"在文中角色","relation":"与作者关系"}],"intellectual_sources":[{"source":"思想来源","influence":"谁通过什么观点、如何影响了谁"}],"debates":[{"opponent":"论敌","issue":"争论焦点","author_position":"本文立场","opponent_position":"对手立场"}],"prerequisites":["前置知识"],"philosophical_problems":[{"problem":"哲学史问题","treatment":"本文如何回应"}]}

【数量限制】chapter_structure ≤ 8，core_concepts ≤ 10，key_figures ≤ 8，debates ≤ 5。内容从文本提炼。"""

def _extract_json_block(raw: str) -> str:
    """剥掉 ```代码块``` 包裹，并截取第一个 { 到最后一个 } 之间的内容。"""
    s = raw.strip()
    s = re.sub(r'^```[a-zA-Z]*\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    s = s.strip()
    start, end = s.find('{'), s.rfind('}')
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    return s

def _repair_json(s: str) -> str:
    """对已截取的 JSON 文本做"安全"修复——只处理确定不会破坏合法 JSON 的问题。
    注意：不做单引号→双引号替换（会破坏含撇号的合法 JSON）。"""
    # 去掉 JSON 注释（// 行注释 和 /* */ 块注释）
    s = re.sub(r'//[^\n]*', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    # 去掉尾随逗号：, 后面紧跟 } 或 ]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    # 字符串值内的裸换行/制表符需转义
    def fix_ctrl(m):
        return m.group(0).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    s = re.sub(r'"(?:[^"\\]|\\.)*"', fix_ctrl, s)
    return s

def _escape_inner_quotes(s: str) -> str:
    """状态机：转义字符串值内部未转义的英文双引号。
    判定规则：字符串内遇到 "，若其后第一个非空白字符是 , : } ] 或到结尾，
    视为结构性结束引号；否则视为内容引号，转义为 \\"。
    这能修复 LLM 在中文内容里误用英文引号导致的 JSON 破坏。"""
    out, i, n, in_str = [], 0, len(s), False
    while i < n:
        c = s[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            i += 1
            continue
        # 在字符串内部
        if c == '\\':                       # 转义序列，整体保留
            out.append(c)
            if i + 1 < n:
                out.append(s[i + 1])
                i += 2
            else:
                i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and s[j] in ' \t\r\n':
                j += 1
            nxt = s[j] if j < n else ''
            if nxt in ',:}]' or nxt == '':  # 结构性结束引号
                out.append(c)
                in_str = False
            else:                            # 内容引号 → 转义
                out.append('\\"')
            i += 1
            continue
        out.append(c)
        i += 1
    return ''.join(out)

def _parse_wiki_json(raw: str) -> dict:
    """多策略解析 LLM 返回为 dict，依次尝试，返回第一个成功的。全部失败抛异常。"""
    block = _extract_json_block(raw)
    attempts = [
        block,                                          # 1. 原始（多数情况合法）
        _repair_json(block),                            # 2. 安全修复（注释/尾逗号/控制符）
        _escape_inner_quotes(block),                    # 3. 转义内部引号
        _escape_inner_quotes(_repair_json(block)),      # 4. 修复 + 转义引号
        _repair_json(_escape_inner_quotes(block)),      # 5. 转义引号 + 修复
    ]
    last_err = None
    for cand in attempts:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
    raise last_err

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

@app.route('/api/debug-env', methods=['GET'])
def debug_env():
    def mask(s):
        s = s.strip()
        return f'{s[:4]}...({len(s)} chars)' if s else '(empty)'

    env_val = os.environ.get('ANTHROPIC_API_KEY', '')
    final   = _default_key('claude')
    return jsonify({
        'env_ANTHROPIC_API_KEY_len':    len(env_val),
        'env_ANTHROPIC_API_KEY_prefix': mask(env_val),
        'env_present':                  'ANTHROPIC_API_KEY' in os.environ,
        'final_key_len':                len(final),
        'final_key_prefix':             mask(final),
        'final_key_ok':                 bool(final),
        'related_env_names': sorted(k for k in os.environ if 'ANTHROPIC' in k.upper() or 'KEY' in k.upper()),
    })

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

    chunks = _load_chunks(doc_id)
    sample = _sample_for_wiki(chunks)
    if not sample:
        return jsonify({'error': '文献内容为空'}), 400

    user_msg = f'文献名称：{_documents[doc_id]["name"]}\n\n文本摘样：\n\n{sample}'
    try:
        raw = _llm_call(provider, api_key,
                        [{'role': 'user', 'content': user_msg}],
                        system=WIKI_SYSTEM, max_tokens=3000)

        try:
            wiki = _parse_wiki_json(raw)
        except json.JSONDecodeError as e:
            # JSON 解析失败：降级为纯文本 Wiki 保存，不报错
            print(f'[Wiki] JSON 解析失败（{e}），降级为纯文本模式', flush=True)
            wiki = {'_plaintext': True, 'thesis': raw}

    except Exception as e:
        print(f'[Wiki EXCEPTION] {e}', flush=True)
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

    if not (api_key or _default_key(provider)):
        return jsonify({'error': f'模型未配置：未配置 {provider} 的 API Key'}), 400

    if session_id not in _conversations:
        _conversations[session_id] = []

    # Build system prompt
    system = READER_SYSTEM
    if user_bg:
        system += f'\n\n读者背景：{user_bg}'

    wiki_parts, concept_terms = [], []
    for did in doc_ids:
        w = _load_wiki(did)
        if w:
            name     = _documents.get(did, {}).get('name', did)
            concepts = [c['term'] for c in w.get('core_concepts', []) if c.get('term')]
            concept_terms.extend(concepts)
            terms = '、'.join(concepts[:5])
            wiki_parts.append(f'【{name}】\n论点：{w.get("thesis","")}\n核心概念：{terms}')
    if wiki_parts:
        system += '\n\n## 已加载文献 Wiki 摘要\n\n' + '\n\n'.join(wiki_parts)

    # Retrieve chunks：把「提问 + Wiki 核心概念」一起当检索词，扩大召回——
    # 哲学提问常较抽象、与原文用词对不上，补上核心概念能勾到更多相关段落。
    search_query = message + ' ' + ' '.join(concept_terms[:20])
    relevant = _search(search_query, doc_ids=doc_ids, top_k=6)
    # 兜底：关键词一个都没命中时，也从文献里均匀取代表性片段，
    # 确保每次对话都带着原文，回答有据可依、能引用原文。
    if not relevant:
        relevant = _representative_chunks(doc_ids, top_k=6)

    ctx, seen, sources = [], set(), []
    for r in relevant:
        ctx.append(f'[{r["name"]} 第{r["page"]}页]\n{r["text"]}')
        k = f'{r["doc_id"]}_p{r["page"]}'
        if k not in seen:
            seen.add(k)
            sources.append({'doc_id': r['doc_id'], 'name': r['name'],
                             'page': r['page'], 'text': r['text']})
    if ctx:
        aug = message + '\n\n## 相关原文片段（请优先依据这些原文作答并标注页码）\n\n' + '\n\n---\n\n'.join(ctx)
    else:
        aug = message + '\n\n（文献库为空，请基于你的知识回答并注明这是推断）'

    conv = _conversations[session_id]
    conv.append({'role': 'user', 'content': aug})
    if len(conv) > 20:
        _conversations[session_id] = conv[-20:]
        conv = _conversations[session_id]

    try:
        reply = _llm_call(provider, api_key, conv, system=system)
        conv.append({'role': 'assistant', 'content': reply})
        return jsonify({'success': True, 'reply': reply, 'sources': sources})
    except Exception as e:
        return jsonify({'error': f'调用失败：{e}'}), 500

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('诠释者 · 哲学 Wiki 阅读智能体')
    for p in ('claude', 'deepseek', 'qwen'):
        status = 'OK' if _default_key(p) else '-- (请在界面填入 Key)'
        print(f'  {p}: {status}')
    print('http://localhost:5000')
    app.run(debug=False, port=5000)
