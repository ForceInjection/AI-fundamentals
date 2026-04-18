import argparse
import datetime as dt
import gzip
import json
import os
import random
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import List, Dict, Any, Optional, Tuple

"""
AI Coding Plan 价格数据抓取与解析脚本

此脚本用于自动从国内外各大 AI 编程助手（Coding Plan）官方定价页面抓取价格与额度信息。
主要功能：
1. 支持原生 urllib 以及 Playwright（无头浏览器）两种抓取模式。
2. 内置反反爬机制（随机 User-Agent、模拟真实浏览器请求头）。
3. 针对不同厂商的网页结构，提供独立的文本抽取与解析函数。
4. 支持使用 manual_overrides.json 进行人工数据兜底与修正。
5. 最终生成规范化的 JSON 数据文件与 Markdown 比价表格。
"""


class _TextExtractor(HTMLParser):
    """HTML 纯文本提取器。忽略所有标签，仅保留并拼接内部文本数据。"""
    def __init__(self):
        super().__init__()
        self._chunks = []

    def handle_data(self, data: str):
        if data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def _html_to_text(html_str: str) -> str:
    """将 HTML 字符串转换为紧凑的纯文本（合并多余空格与换行）。"""
    parser = _TextExtractor()
    parser.feed(html_str)
    text = parser.get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_mkdir(path: str):
    """安全创建多级目录，如果已存在则忽略。"""
    os.makedirs(path, exist_ok=True)


def _today_ymd() -> str:
    """获取当天的日期字符串，格式为 YYYY-MM-DD。"""
    return dt.date.today().strftime("%Y-%m-%d")


# =============================================================================
# 反爬与网络请求模块
# =============================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0"
]


def _get_random_user_agent() -> str:
    """从常用浏览器标识池中随机抽取一个 User-Agent。"""
    return random.choice(USER_AGENTS)


def _read_bytes_from_url(url: str, timeout_seconds: int = 20) -> bytes:
    """
    使用 urllib 发起 HTTP 请求，并自动解压 gzip 数据。
    内置了多项反反爬 headers 混淆。
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        },
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout_seconds, context=context) as resp:
        raw = resp.read()
        enc = (resp.headers.get("Content-Encoding") or "").lower()
        if "gzip" in enc:
            raw = gzip.decompress(raw)
        return raw


# =============================================================================
# 文件读写与缓存模块
# =============================================================================

def _write_text(path: str, text: str):
    """将文本字符串安全写入指定文件，目录不存在则自动创建。"""
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_bytes(path: str, data: bytes):
    """将字节流安全写入指定文件，目录不存在则自动创建。"""
    _safe_mkdir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)


def _slugify(s: str) -> str:
    """将字符串转换为安全的 URL/文件名格式 (连字符分隔的小写字母数字)。"""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "snapshot"


def _snapshot_paths(base_dir: str, ymd: str, vendor: str, name: str, ext: str) -> Tuple[str, str]:
    """
    计算特定网页的本地快照保存目录和文件路径。
    结构形如: data/pricing_raw/{YYYY-MM-DD}/{vendor}/{name}.{ext}
    """
    vendor_dir = os.path.join(base_dir, "data", "pricing_raw", ymd, _slugify(vendor))
    file_name = f"{_slugify(name)}.{ext}"
    return vendor_dir, os.path.join(vendor_dir, file_name)


def _fetch_snapshot_text(base_dir: str, ymd: str, vendor: str, name: str, url: str, refresh: bool = False, page=None) -> Tuple[Optional[str], Optional[str]]:
    """
    获取指定 URL 的网页内容。
    - 如果启用了 Playwright (page != None)，则使用无头浏览器渲染。
    - 如果本地已有当日缓存且未强制 refresh，则直接读取缓存。
    - 若抓取失败，自动将错误堆栈写入 {name}.error.txt 并返回缓存旧数据（如果有）。
    """
    vendor_dir, snapshot_path = _snapshot_paths(base_dir, ymd, vendor, name, "html")
    if (not refresh) and os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            return snapshot_path, f.read()

    try:
        if page:
            print(f"Fetching (headless browser): {url}")
            page.goto(url, wait_until="networkidle", timeout=30000)
            html = page.content()
            raw = html.encode("utf-8")
        else:
            print(f"Fetching (urllib): {url}")
            raw = _read_bytes_from_url(url)
            html = raw.decode("utf-8", errors="replace")
            
        _safe_mkdir(vendor_dir)
        _write_bytes(snapshot_path, raw)
        return snapshot_path, html
    except Exception as e:
        err_path = os.path.join(vendor_dir, f"{_slugify(name)}.error.txt")
        _write_text(err_path, f"url: {url}\nerror: {repr(e)}\n")
        if os.path.exists(snapshot_path):
            with open(snapshot_path, "r", encoding="utf-8") as f:
                return snapshot_path, f.read()
        return None, None


# =============================================================================
# 价格提取与基础解析模块
# =============================================================================

def _drop_noisy_failures(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    清理解析记录。
    如果同一个产品已经有解析成功的套餐信息（status != fail），
    则丢弃其同属产品下的 "unknown" 套餐错误记录，减少最终数据的噪音。
    """
    by_key = {}
    for r in records:
        key = (r.get("vendor"), r.get("product"))
        by_key.setdefault(key, []).append(r)

    kept = []
    bad_statuses = {"download_failed", "parse_error", "parse_failed"}
    for _, items in by_key.items():
        has_good = any((i.get("status") not in bad_statuses) and i.get("plan_name") != "unknown" for i in items)
        for i in items:
            if has_good and i.get("plan_name") == "unknown" and (i.get("status") in bad_statuses):
                continue
            kept.append(i)
    return kept


def _extract_usd_prices(text: str) -> List[float]:
    """提取文本中所有的美元价格数值 (如 $10, $19.99)。"""
    items = []
    for m in re.finditer(r"\$\s*([0-9]+(?:\.[0-9]+)?)", text):
        items.append(float(m.group(1)))
    return items


def _extract_cny_prices(text: str) -> List[float]:
    """提取文本中所有的人民币价格数值 (如 ¥40, ￥299)。"""
    items = []
    for m in re.finditer(r"(?:¥|￥)\s*([0-9]+(?:\.[0-9]+)?)", text):
        items.append(float(m.group(1)))
    return items


def _extract_price_near_keyword(text: str, keyword: str, currency: str = "USD", window: int = 240) -> Optional[float]:
    """
    定位关键词 keyword 出现的位置，并在其后指定的 window 字符窗口内，
    使用正则提取出现的第一个价格。适用于提取诸如 "Pro Tier" 后的月费。
    """
    if not text or not keyword:
        return None
    lower = text.lower()
    idx = lower.find(keyword.lower())
    if idx < 0:
        return None
    segment = text[idx : idx + window]
    if currency.upper() == "USD":
        m = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", segment)
        return float(m.group(1)) if m else None
    if currency.upper() == "CNY":
        m = re.search(r"(?:¥|￥)\s*([0-9]+(?:\.[0-9]+)?)", segment)
        return float(m.group(1)) if m else None
    return None


def _extract_price_in_section(text: str, start_keyword: str, end_keywords: List[str], currency: str = "USD") -> Optional[float]:
    """
    基于区间截断提取价格。
    定位 start_keyword，并将其至第一个出现的 end_keywords 之间的文本截取出来，
    在此区间内寻找符合货币类型的价格。
    同时能识别出表示免费的文本并返回 0.0。
    """
    if not text or not start_keyword:
        return None
    lower = text.lower()
    start = lower.find(start_keyword.lower())
    if start < 0:
        return None
    end = len(text)
    for kw in end_keywords or []:
        if not kw:
            continue
        idx = lower.find(kw.lower(), start + len(start_keyword))
        if idx >= 0 and idx < end:
            end = idx
    section = text[start:end]
    price = _extract_price_near_keyword(section, start_keyword, currency=currency, window=len(section))
    if price is None and "free" in section.lower() and currency.upper() == "USD":
        return 0.0
    if price is None and "免费" in section and currency.upper() == "CNY":
        return 0.0
    return price


def _guess_plans_by_keywords(text: str, plan_keywords: List[str]) -> List[str]:
    """通过模糊匹配快速猜测文本中存在哪些套餐名称。"""
    hits = []
    lower = text.lower()
    for kw in plan_keywords:
        idx = lower.find(kw.lower())
        if idx >= 0:
            hits.append(kw)
    return hits


def _make_record(
    vendor: str,
    product: str,
    plan_name: str,
    currency: str,
    price_monthly: Optional[float] = None,
    price_yearly_effective: Optional[float] = None,
    quota: Optional[str] = None,
    rate_limit_policy: Optional[str] = None,
    restrictions: Optional[str] = None,
    promo_price: Optional[float] = None,
    promo_conditions: Optional[str] = None,
    source_url: Optional[str] = None,
    accessed_at: Optional[str] = None,
    raw_snapshot_path: Optional[str] = None,
    extraction_method: Optional[str] = None,
    status: str = "ok",
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """构建标准化的定价数据字典记录。所有厂商的解析结果均通过此函数统一输出格式。"""
    return {
        "vendor": vendor,
        "product": product,
        "plan_name": plan_name,
        "currency": currency,
        "price_monthly": price_monthly,
        "price_yearly_effective": price_yearly_effective,
        "promo_price": promo_price,
        "promo_conditions": promo_conditions,
        "quota": quota,
        "rate_limit_policy": rate_limit_policy,
        "restrictions": restrictions,
        "source_url": source_url,
        "accessed_at": accessed_at,
        "raw_snapshot_path": raw_snapshot_path,
        "extraction_method": extraction_method,
        "status": status,
        "notes": notes,
    }

# =============================================================================
# 各厂商专用解析函数模块
# =============================================================================

def _parse_github_copilot(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "GitHub"
    product = "GitHub Copilot"
    text = _html_to_text(html_text)
    records = []
    plan_candidates = [
        ("Free", "usd"),
        ("Pro", "usd"),
        ("Pro+", "usd"),
        ("Business", "usd"),
        ("Enterprise", "usd"),
    ]
    hits = _guess_plans_by_keywords(text, [p[0] for p in plan_candidates])
    if not hits:
        return [
            _make_record(
                vendor=vendor,
                product=product,
                plan_name="unknown",
                currency="USD",
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_regex",
                status="parse_failed",
                notes="无法从页面文本中识别套餐名称，需人工复核或补充手动覆盖。",
            )
        ]

    for name, _ in plan_candidates:
        if name not in hits:
            continue
        end_kws = [p[0] for p in plan_candidates if p[0] != name]
        price = _extract_price_in_section(text, name, end_kws, currency="USD")
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=name,
                currency="USD",
                price_monthly=price,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if price is not None else "needs_review",
                notes=None if price is not None else "识别到套餐但未能可靠抽取价格，需人工复核。",
            )
        )
    return records


def _parse_cursor(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "Cursor"
    product = "Cursor"
    text = _html_to_text(html_text)
    records = []
    ordered = ["Hobby", "Pro", "Business", "Ultra"]
    for i, plan in enumerate(ordered):
        if plan.lower() not in text.lower():
            continue
        end_kws = ordered[i + 1 :]
        price = _extract_price_in_section(text, plan, end_kws, currency="USD")
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=plan,
                currency="USD",
                price_monthly=price,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if price is not None else "needs_review",
                notes=None if price is not None else "识别到套餐但未能可靠抽取价格，需人工复核。",
            )
        )
    if not records:
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name="unknown",
                currency="USD",
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_regex",
                status="parse_failed",
                notes="无法从页面文本中识别套餐，需人工复核。",
            )
        )
    return records


def _parse_windsurf(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "Windsurf"
    product = "Windsurf (Codeium)"
    text = _html_to_text(html_text)
    records = []
    ordered = ["Free", "Pro", "Teams", "Ultimate"]
    for i, plan in enumerate(ordered):
        if plan.lower() not in text.lower():
            continue
        end_kws = ordered[i + 1 :]
        price = _extract_price_in_section(text, plan, end_kws, currency="USD")
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=plan,
                currency="USD",
                price_monthly=price,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if price is not None else "needs_review",
                notes=None if price is not None else "识别到套餐但未能可靠抽取价格，需人工复核。",
            )
        )
    if not records:
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name="unknown",
                currency="USD",
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_regex",
                status="parse_failed",
                notes="无法从页面文本中识别套餐，需人工复核。",
            )
        )
    return records


def _parse_amazon_q(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "AWS"
    product = "Amazon Q Developer"
    text = _html_to_text(html_text)
    pro_price = _extract_price_near_keyword(text, "Pro Tier", currency="USD", window=800)
    if pro_price is None:
        pro_price = _extract_price_near_keyword(text, "Pro", currency="USD", window=800)
    return [
        _make_record(
            vendor=vendor,
            product=product,
            plan_name="Pro",
            currency="USD",
            price_monthly=pro_price,
            source_url=source_url,
            accessed_at=accessed_at,
            raw_snapshot_path=snapshot_path,
            extraction_method="html_text_regex",
            status="ok" if pro_price is not None else "needs_review",
            notes=None if pro_price is not None else "未能可靠抽取 Pro 月费，需人工复核。",
        )
    ]


def _parse_claude_pricing(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "Anthropic"
    product = "Claude (含 Claude Code)"
    text = _html_to_text(html_text)
    pro = _extract_price_near_keyword(text, "Pro For", currency="USD", window=600) or _extract_price_near_keyword(
        text, "Pro", currency="USD", window=600
    )
    max_plan = _extract_price_near_keyword(text, "Max For", currency="USD", window=600) or _extract_price_near_keyword(
        text, "Max", currency="USD", window=600
    )
    records = [
        _make_record(
            vendor=vendor,
            product=product,
            plan_name="Pro",
            currency="USD",
            price_monthly=pro,
            source_url=source_url,
            accessed_at=accessed_at,
            raw_snapshot_path=snapshot_path,
            extraction_method="html_text_heuristic",
            status="ok" if pro is not None else "needs_review",
            notes=None if pro is not None else "未能可靠抽取 Claude Pro 月费，需人工复核。",
        ),
        _make_record(
            vendor=vendor,
            product=product,
            plan_name="Max",
            currency="USD",
            price_monthly=max_plan,
            source_url=source_url,
            accessed_at=accessed_at,
            raw_snapshot_path=snapshot_path,
            extraction_method="html_text_heuristic",
            status="ok" if max_plan is not None else "needs_review",
            notes=None if max_plan is not None else "未能可靠抽取 Claude Max 月费，需人工复核。",
        ),
    ]
    return records


def _parse_volcengine_codingplan_activity(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "火山方舟"
    product = "火山方舟 Coding Plan"
    text = _html_to_text(html_text)
    lite = _extract_price_near_keyword(text, "Lite plan", currency="CNY", window=600)
    pro = _extract_price_near_keyword(text, "Pro plan", currency="CNY", window=600)
    records = [
        _make_record(
            vendor=vendor,
            product=product,
            plan_name="Lite",
            currency="CNY",
            price_monthly=lite,
            source_url=source_url,
            accessed_at=accessed_at,
            raw_snapshot_path=snapshot_path,
            extraction_method="html_text_window_price",
            status="ok" if lite is not None else "needs_review",
            notes=None if lite is not None else "未能可靠抽取 Lite 月费，需人工复核。",
        ),
        _make_record(
            vendor=vendor,
            product=product,
            plan_name="Pro",
            currency="CNY",
            price_monthly=pro,
            source_url=source_url,
            accessed_at=accessed_at,
            raw_snapshot_path=snapshot_path,
            extraction_method="html_text_window_price",
            status="ok" if pro is not None else "needs_review",
            notes=None if pro is not None else "未能可靠抽取 Pro 月费，需人工复核。",
        ),
    ]
    return records


def _parse_minimax_token_plan(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "MiniMax"
    product = "MiniMax Token Plan"
    text = _html_to_text(html_text)
    monthly_prices = _extract_cny_prices(text)
    monthly = []
    yearly = []
    for p in monthly_prices:
        if p >= 1000:
            continue
        monthly.append(p)
    for m in re.finditer(r"(?:¥|￥)\s*([0-9]+)\s*/\s*年", text):
        yearly.append(float(m.group(1)))

    quotas = []
    for m in re.finditer(r"([0-9,]+)\s*次请求/5\s*小时", text):
        quotas.append(m.group(1).replace(",", ""))

    def safe_get(arr, idx):
        return arr[idx] if idx < len(arr) else None

    plan_names = ["Starter（标准版）", "Plus（标准版）", "Max（标准版）"]
    records = []
    for i, name in enumerate(plan_names):
        pm = safe_get(monthly, i)
        py = safe_get(yearly, i)
        pye = round(py / 12.0, 2) if py else None
        quota = f"M2.7：{safe_get(quotas, i)} 次请求/5 小时（以官方定价文档为准）" if safe_get(quotas, i) else None
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=name,
                currency="CNY",
                price_monthly=pm,
                price_yearly_effective=pye,
                quota=quota,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_regex",
                status="ok" if pm is not None else "needs_review",
                notes=None if pm is not None else "未能可靠抽取价格，需人工复核。",
            )
        )
    return records


def _parse_kimi_code(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "Kimi"
    product = "Kimi Code Plan"
    text = _html_to_text(html_text)
    plans = ["日常使用", "效率升级", "专业优选", "全能尊享"]
    records = []
    for p in plans:
        segment_price = None
        segment_promo = None
        segment_year = None
        lower = text.lower()
        idx = lower.find(p.lower())
        if idx >= 0:
            seg = text[idx : idx + 260]
            m = re.search(r"(?:¥|￥)\s*([0-9]+)\s*/\s*月\s*(?:¥|￥)\s*([0-9]+)", seg)
            if m:
                segment_promo = float(m.group(1))
                segment_price = float(m.group(2))
            y = re.search(r"(?:¥|￥)\s*([0-9,]+)\s*/\s*年", seg)
            if y:
                segment_year = float(y.group(1).replace(",", ""))
        pye = round(segment_year / 12.0, 2) if segment_year else None
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=p,
                currency="CNY",
                price_monthly=segment_price,
                price_yearly_effective=pye,
                promo_price=segment_promo,
                promo_conditions="页面同时展示月度活动价与常规月费（以官方页面为准）" if segment_promo else None,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if segment_price is not None else "needs_review",
                notes=None if segment_price is not None else "未能可靠抽取价格，需人工复核。",
            )
        )
    return records


def _parse_zhipu_claw_plan_team(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "智谱"
    product = "智谱龙虾套餐（团队协作版）"
    text = _html_to_text(html_text)
    records = []
    if "龙虾体验卡" in text:
        price = _extract_price_near_keyword(text, "龙虾体验卡", currency="CNY", window=400)
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name="龙虾体验卡",
                currency="CNY",
                price_monthly=price,
                quota="3500 万 tokens（以官方页面为准）" if "3500" in text else None,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if price is not None else "needs_review",
                notes=None if price is not None else "识别到套餐但未能可靠抽取价格，需人工复核。",
            )
        )
    if "龙虾进阶卡" in text:
        price = _extract_price_near_keyword(text, "龙虾进阶卡", currency="CNY", window=400)
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name="龙虾进阶卡",
                currency="CNY",
                price_monthly=price,
                quota="1 亿 tokens（以官方页面为准）" if "1亿" in text or "1 亿" in text else None,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if price is not None else "needs_review",
                notes=None if price is not None else "识别到套餐但未能可靠抽取价格，需人工复核。",
            )
        )
    if not records:
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name="unknown",
                currency="CNY",
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_regex",
                status="parse_failed",
                notes="无法从页面文本中识别龙虾套餐字段，需人工复核。",
            )
        )
    return records


def _parse_zhipu_glm_coding(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "智谱"
    product = "智谱 GLM Coding Plan"
    text = _html_to_text(html_text)
    records = []

    def _extract_regular_and_discount_for_plan(plan_name):
        if not text:
            return None, None
        lower = text.lower()
        idx = lower.find(plan_name.lower())
        if idx < 0:
            return None, None

        seg = text[idx : idx + 3200]
        prices = _extract_cny_prices(seg)
        if not prices:
            return None, None

        uniq = []
        for p in prices:
            if p not in uniq:
                uniq.append(p)

        if len(uniq) == 1:
            return uniq[0], None
        discount = min(uniq)
        regular = max(uniq)
        return regular, discount

    for plan in ["Lite", "Pro", "Max"]:
        regular, discount = _extract_regular_and_discount_for_plan(plan)
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=plan,
                currency="CNY",
                price_monthly=regular,
                promo_price=discount,
                promo_conditions="连续包季 9 折 / 连续包年 8 折（以官方页面说明为准）" if discount else None,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_price_list",
                status="ok" if regular is not None else "needs_review",
                notes=None if regular is not None else "未能可靠抽取价格，需人工复核。",
            )
        )
    return records


def _parse_jetbrains_ai(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "JetBrains"
    product = "JetBrains AI"
    text = _html_to_text(html_text)
    candidates = ["AI Pro", "AI Ultimate", "AI Enterprise"]
    records = []
    for i, plan in enumerate(candidates):
        if plan.lower() not in text.lower():
            continue
        end_kws = candidates[i + 1 :]
        price = _extract_price_in_section(text, plan, end_kws, currency="USD")
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=plan,
                currency="USD",
                price_monthly=price,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if price is not None else "needs_review",
                notes=None if price is not None else "识别到套餐但未能可靠抽取价格，需人工复核。",
            )
        )
    if not records:
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name="unknown",
                currency="USD",
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_regex",
                status="parse_failed",
                notes="无法从页面文本中识别套餐，需人工复核。",
            )
        )
    return records


def _parse_tabnine(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "Tabnine"
    product = "Tabnine"
    text = _html_to_text(html_text)
    mapping = {"Dev": None, "Enterprise": None}
    mapping["Dev"] = _extract_price_near_keyword(text, "Dev", currency="USD", window=600)
    mapping["Enterprise"] = _extract_price_near_keyword(text, "Enterprise", currency="USD", window=600)
    records = []
    for plan in ["Dev", "Enterprise"]:
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=plan,
                currency="USD",
                price_monthly=mapping.get(plan),
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_heuristic",
                status="ok" if mapping.get(plan) is not None else "needs_review",
                notes=None if mapping.get(plan) is not None else "未能可靠抽取价格，需人工复核。",
            )
        )
    return records


def _parse_replit(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "Replit"
    product = "Replit"
    text = _html_to_text(html_text)
    records = []
    ordered = ["Core", "Pro"]
    for i, plan in enumerate(ordered):
        if plan.lower() not in text.lower():
            continue
        end_kws = ordered[i + 1 :]
        price = _extract_price_in_section(text, plan, end_kws, currency="USD")
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=plan,
                currency="USD",
                price_monthly=price,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if price is not None else "needs_review",
                notes=None if price is not None else "识别到套餐但未能可靠抽取价格，需人工复核。",
            )
        )
    if not records:
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name="unknown",
                currency="USD",
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_regex",
                status="parse_failed",
                notes="无法从页面文本中识别套餐，需人工复核。",
            )
        )
    return records


def _parse_aliyun_bailian(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "阿里云"
    product = "阿里云百炼 Coding Plan"
    text = _html_to_text(html_text)
    lite_price = _extract_price_in_section(text, "Lite", ["Pro"], currency="CNY")
    pro_price = _extract_price_in_section(text, "Pro", [], currency="CNY")
    promo_price = _extract_price_near_keyword(text, "首月", currency="CNY", window=800)
    records = []
    for plan in ["Lite", "Pro"]:
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=plan,
                currency="CNY",
                price_monthly=lite_price if plan == "Lite" else pro_price,
                promo_price=promo_price,
                promo_conditions="新客首月（以官方页面说明为准）" if promo_price else None,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if (lite_price if plan == "Lite" else pro_price) is not None else "needs_review",
                notes=None if (lite_price if plan == "Lite" else pro_price) is not None else "未能可靠抽取价格，需人工复核。",
            )
        )
    return records


def _parse_tencent_codingplan(html_text: str, snapshot_path: str, accessed_at: str, source_url: str) -> List[Dict[str, Any]]:
    vendor = "腾讯云"
    product = "腾讯云 Coding Plan"
    text = _html_to_text(html_text)
    lite_price = _extract_price_in_section(text, "Lite", ["Pro"], currency="CNY")
    pro_price = _extract_price_in_section(text, "Pro", [], currency="CNY")
    promo_price = _extract_price_near_keyword(text, "首月", currency="CNY", window=1200)
    records = []
    for plan in ["Lite", "Pro"]:
        price = lite_price if plan == "Lite" else pro_price
        records.append(
            _make_record(
                vendor=vendor,
                product=product,
                plan_name=plan,
                currency="CNY",
                price_monthly=price,
                promo_price=promo_price,
                promo_conditions="新客首月（以官方页面说明为准）" if promo_price else None,
                source_url=source_url,
                accessed_at=accessed_at,
                raw_snapshot_path=snapshot_path,
                extraction_method="html_text_window_price",
                status="ok" if price is not None else "needs_review",
                notes=None if price is not None else "未能可靠抽取价格，需人工复核。",
            )
        )
    return records


# =============================================================================
# 辅助合并与输出模块
# =============================================================================

def _load_manual_overrides(path: str) -> List[Dict[str, Any]]:
    """加载 manual_overrides.json 文件，该文件用于人工硬编码修正自动抓取的不稳定数据。"""
    if not path or (not os.path.exists(path)):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def _merge_overrides(records: List[Dict[str, Any]], overrides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将人工配置的 override 数据合并到自动抓取的 records 中。
    使用 (vendor, product, plan_name) 作为唯一键。
    """
    def key_of(r):
        return (r.get("vendor"), r.get("product"), r.get("plan_name"))

    merged = {key_of(r): r for r in records}
    for r in overrides:
        merged[key_of(r)] = r
    return list(merged.values())


def _write_markdown_table(path: str, records: List[Dict[str, Any]]):
    """将数据记录输出为规范的 Markdown 表格。"""
    cols = [
        "vendor",
        "product",
        "plan_name",
        "currency",
        "price_monthly",
        "price_yearly_effective",
        "promo_price",
        "quota",
        "restrictions",
        "source_url",
        "accessed_at",
        "status",
    ]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, (int, float)):
            return str(v)
        return str(v).replace("\n", " ").strip()

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in sorted(records, key=lambda x: (x.get("vendor") or "", x.get("product") or "", x.get("plan_name") or "")):
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    _write_text(path, "\n".join(lines) + "\n")


# =============================================================================
# 数据源配置与主程序
# =============================================================================

def get_data_sources() -> List[Dict[str, Any]]:
    """返回所有待抓取的厂商配置列表。"""
    return [
        {
            "vendor": "GitHub",
            "name": "copilot_plans",
            "url": "https://github.com/features/copilot/plans",
            "parser": _parse_github_copilot,
        },
        {
            "vendor": "Cursor",
            "name": "cursor_pricing",
            "url": "https://www.cursor.com/pricing",
            "parser": _parse_cursor,
        },
        {
            "vendor": "Windsurf",
            "name": "windsurf_pricing",
            "url": "https://windsurf.com/pricing",
            "parser": _parse_windsurf,
        },
        {
            "vendor": "AWS",
            "name": "amazon_q_pricing",
            "url": "https://aws.amazon.com/cn/q/developer/pricing/",
            "parser": _parse_amazon_q,
        },
        {
            "vendor": "Anthropic",
            "name": "claude_pricing",
            "url": "https://claude.com/pricing",
            "parser": _parse_claude_pricing,
        },
        {
            "vendor": "JetBrains",
            "name": "jetbrains_ai_licensing",
            "url": "https://www.jetbrains.com/help/ai-assistant/licensing-and-subscriptions.html",
            "parser": _parse_jetbrains_ai,
        },
        {
            "vendor": "Tabnine",
            "name": "tabnine_pricing",
            "url": "https://old-www.tabnine.com/pricing",
            "parser": _parse_tabnine,
        },
        {
            "vendor": "Replit",
            "name": "replit_pricing",
            "url": "https://replit.com/pricing",
            "parser": _parse_replit,
        },
        {
            "vendor": "阿里云",
            "name": "aliyun_bailian_coding_plan",
            "url": "https://help.aliyun.com/zh/model-studio/coding-plan",
            "parser": _parse_aliyun_bailian,
        },
        {
            "vendor": "腾讯云",
            "name": "tencent_coding_plan",
            "url": "https://cloud.tencent.com/act/pro/codingplan",
            "parser": _parse_tencent_codingplan,
        },
        {
            "vendor": "火山方舟",
            "name": "volcengine_codingplan_activity",
            "url": "https://www.volcengine.com/activity/codingplan",
            "parser": _parse_volcengine_codingplan_activity,
        },
        {
            "vendor": "MiniMax",
            "name": "minimax_token_plan_pricing",
            "url": "https://platform.minimaxi.com/docs/guides/pricing-token-plan",
            "parser": _parse_minimax_token_plan,
        },
        {
            "vendor": "Kimi",
            "name": "kimi_code",
            "url": "https://www.kimi.com/code",
            "parser": _parse_kimi_code,
        },
        {
            "vendor": "智谱",
            "name": "zhipu_claw_plan_team",
            "url": "https://www.bigmodel.cn/claw-plan-team",
            "parser": _parse_zhipu_claw_plan_team,
        },
        {
            "vendor": "智谱",
            "name": "zhipu_glm_coding",
            "url": "https://bigmodel.cn/glm-coding",
            "parser": _parse_zhipu_glm_coding,
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=_today_ymd())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--manual-overrides", default=None)
    parser.add_argument("--use-playwright", action="store_true", help="Use Playwright headless browser for rendering")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    accessed_at = args.date
    sources = get_data_sources()

    records = []
    
    playwright_context = None
    browser = None
    page = None
    
    if args.use_playwright:
        try:
            from playwright.sync_api import sync_playwright
            playwright_context = sync_playwright().start()
            browser = playwright_context.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=_get_random_user_agent(),
                viewport={"width": 1920, "height": 1080},
                extra_http_headers={
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Upgrade-Insecure-Requests": "1",
                }
            )
        except ImportError:
            print("WARNING: playwright is not installed. Please install it using 'pip install playwright' and 'playwright install chromium'. Falling back to urllib.")
            args.use_playwright = False
            
    try:
        for s in sources:
            snapshot_path, html = _fetch_snapshot_text(
                base_dir=base_dir,
                ymd=args.date,
                vendor=s["vendor"],
                name=s["name"],
                url=s["url"],
                refresh=args.refresh,
                page=page
            )
            if html is None:
                records.append(
                    _make_record(
                        vendor=s["vendor"],
                        product=s["vendor"],
                        plan_name="unknown",
                        currency="",
                        source_url=s["url"],
                        accessed_at=accessed_at,
                        raw_snapshot_path=snapshot_path,
                        extraction_method="download",
                        status="download_failed",
                        notes="抓取失败，详见对应 .error.txt 快照。",
                    )
                )
                continue
            try:
                parsed = s["parser"](html, snapshot_path, accessed_at, s["url"])
                records.extend(parsed)
            except Exception as e:
                records.append(
                    _make_record(
                        vendor=s["vendor"],
                        product=s["vendor"],
                        plan_name="unknown",
                        currency="",
                        source_url=s["url"],
                        accessed_at=accessed_at,
                        raw_snapshot_path=snapshot_path,
                        extraction_method="parser",
                        status="parse_error",
                        notes=f"解析异常: {repr(e)}",
                    )
                )
    finally:
        if browser:
            browser.close()
        if playwright_context:
            playwright_context.stop()

    manual_path = args.manual_overrides or os.path.join(base_dir, "data", "manual_overrides.json")
    overrides = _load_manual_overrides(manual_path)
    records = _merge_overrides(records, overrides)
    records = _drop_noisy_failures(records)

    out_json = args.out_json or os.path.join(base_dir, "data", "pricing_normalized.json")
    out_md = args.out_md or os.path.join(base_dir, "data", "pricing_table.md")
    _safe_mkdir(os.path.dirname(out_json))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")
    _write_markdown_table(out_md, records)

    print(f"OK: wrote {out_json}")
    print(f"OK: wrote {out_md}")


if __name__ == "__main__":
    main()
