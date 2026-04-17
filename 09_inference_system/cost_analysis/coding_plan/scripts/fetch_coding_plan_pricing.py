import argparse
import datetime as dt
import gzip
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks = []

    def handle_data(self, data):
        if data:
            self._chunks.append(data)

    def get_text(self):
        return " ".join(self._chunks)


def _html_to_text(html_str):
    parser = _TextExtractor()
    parser.feed(html_str)
    text = parser.get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def _today_ymd():
    return dt.date.today().strftime("%Y-%m-%d")


def _read_bytes_from_url(url, timeout_seconds=20):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip",
        },
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout_seconds, context=context) as resp:
        raw = resp.read()
        enc = (resp.headers.get("Content-Encoding") or "").lower()
        if "gzip" in enc:
            raw = gzip.decompress(raw)
        return raw


def _write_text(path, text):
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_bytes(path, data):
    _safe_mkdir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)


def _slugify(s):
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "snapshot"


def _snapshot_paths(base_dir, ymd, vendor, name, ext):
    vendor_dir = os.path.join(base_dir, "data", "pricing_raw", ymd, _slugify(vendor))
    file_name = f"{_slugify(name)}.{ext}"
    return vendor_dir, os.path.join(vendor_dir, file_name)


def _fetch_snapshot_text(base_dir, ymd, vendor, name, url, refresh=False):
    vendor_dir, snapshot_path = _snapshot_paths(base_dir, ymd, vendor, name, "html")
    if (not refresh) and os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            return snapshot_path, f.read()

    try:
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


def _drop_noisy_failures(records):
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


def _extract_usd_prices(text):
    items = []
    for m in re.finditer(r"\$\s*([0-9]+(?:\.[0-9]+)?)", text):
        items.append(float(m.group(1)))
    return items


def _extract_cny_prices(text):
    items = []
    for m in re.finditer(r"(?:¥|￥)\s*([0-9]+(?:\.[0-9]+)?)", text):
        items.append(float(m.group(1)))
    return items


def _extract_price_near_keyword(text, keyword, currency="USD", window=240):
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


def _extract_price_in_section(text, start_keyword, end_keywords, currency="USD"):
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


def _guess_plans_by_keywords(text, plan_keywords):
    hits = []
    lower = text.lower()
    for kw in plan_keywords:
        idx = lower.find(kw.lower())
        if idx >= 0:
            hits.append(kw)
    return hits


def _make_record(
    vendor,
    product,
    plan_name,
    currency,
    price_monthly=None,
    price_yearly_effective=None,
    quota=None,
    rate_limit_policy=None,
    restrictions=None,
    promo_price=None,
    promo_conditions=None,
    source_url=None,
    accessed_at=None,
    raw_snapshot_path=None,
    extraction_method=None,
    status="ok",
    notes=None,
):
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


def _parse_github_copilot(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_cursor(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_windsurf(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_amazon_q(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_claude_pricing(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_volcengine_codingplan_activity(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_minimax_token_plan(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_kimi_code(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_zhipu_claw_plan_team(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_zhipu_glm_coding(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_jetbrains_ai(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_tabnine(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_replit(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_aliyun_bailian(html_text, snapshot_path, accessed_at, source_url):
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


def _parse_tencent_codingplan(html_text, snapshot_path, accessed_at, source_url):
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


def _load_manual_overrides(path):
    if not path or (not os.path.exists(path)):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def _merge_overrides(records, overrides):
    def key_of(r):
        return (r.get("vendor"), r.get("product"), r.get("plan_name"))

    merged = {key_of(r): r for r in records}
    for r in overrides:
        merged[key_of(r)] = r
    return list(merged.values())


def _write_markdown_table(path, records):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=_today_ymd())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--manual-overrides", default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    accessed_at = args.date

    sources = [
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

    records = []
    for s in sources:
        snapshot_path, html = _fetch_snapshot_text(
            base_dir=base_dir,
            ymd=args.date,
            vendor=s["vendor"],
            name=s["name"],
            url=s["url"],
            refresh=args.refresh,
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
