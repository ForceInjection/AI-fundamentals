import urllib.request
import urllib.error
import json
import re
import os
import time
import argparse
import tempfile

# 默认参数定义
DEFAULT_EXCHANGE_RATE = 6.9
DEFAULT_HIT_RATE = 0.5
DEFAULT_INPUT_TOKENS = 21.6  # K
DEFAULT_OUTPUT_TOKENS = 0.8  # K
CACHE_FILE = os.path.join(tempfile.gettempdir(), "openrouter_models_cache.json")
CACHE_EXPIRY_SECONDS = 3600  # 1 小时

# 模型分类定义
CATEGORIES = [
    {
        "id": "gpt_flagship",
        "prefix": "openai/",
        "include": ["gpt"],
        "exclude": ["mini", "nano", "vision", "audio", "realtime", "search", "free", "oss", "-0"],
        "label": "GPT 旗舰 ({version})"
    },
    {
        "id": "gpt_mini",
        "prefix": "openai/",
        "include": ["gpt", "mini"],
        "exclude": ["vision", "audio", "realtime", "search", "free", "oss"],
        "label": "GPT 经济型 ({version})"
    },
    {
        "id": "o_series",
        "prefix": "openai/",
        "include": ["/o"],
        "exclude": ["mini", "audio", "realtime", "search", "free", "deep-research"],
        "label": "OpenAI O系列 高级推理 ({version})"
    },
    {
        "id": "o_mini",
        "prefix": "openai/",
        "include": ["/o", "mini"],
        "exclude": ["audio", "realtime", "search", "free", "deep-research"],
        "label": "OpenAI O系列 轻量推理 ({version})"
    },
    {
        "id": "claude_opus",
        "prefix": "anthropic/",
        "include": ["opus"],
        "exclude": ["free"],
        "label": "Claude Opus 顶级旗舰 ({version})"
    },
    {
        "id": "claude_sonnet",
        "prefix": "anthropic/",
        "include": ["sonnet"],
        "exclude": ["thinking", "free"],
        "label": "Claude Sonnet 均衡旗舰 ({version})"
    },
    {
        "id": "claude_haiku",
        "prefix": "anthropic/",
        "include": ["haiku"],
        "exclude": ["free"],
        "label": "Claude Haiku 经济型 ({version})"
    },
    {
        "id": "gemini_pro",
        "prefix": "google/",
        "include": ["gemini", "pro"],
        "exclude": ["vision", "experimental", "free", "image", "customtools", "05-06"],
        "label": "Gemini Pro ({version})"
    },
    {
        "id": "gemini_flash",
        "prefix": "google/",
        "include": ["gemini", "flash"],
        "exclude": ["lite", "vision", "experimental", "free", "image"],
        "label": "Gemini Flash ({version})"
    },
    {
        "id": "gemini_flash_lite",
        "prefix": "google/",
        "include": ["gemini", "flash", "lite"],
        "exclude": ["vision", "experimental", "free", "image", "09-2025"],
        "label": "Gemini Flash-Lite ({version})"
    },
    {
        "id": "deepseek_chat",
        "prefix": "deepseek/",
        "include": ["chat"],
        "exclude": ["free"],
        "label": "DeepSeek Chat (latest)" # 官方通常不改名，固定显示
    },
    {
        "id": "deepseek_reasoner",
        "prefix": "deepseek/",
        "include": ["r1"],
        "exclude": ["free", "qwen", "llama"],
        "label": "DeepSeek Reasoner (R1)"
    },
    {
        "id": "qwen_max",
        "prefix": "qwen/",
        "include": ["max"],
        "exclude": ["thinking", "free"],
        "label": "Qwen Max 通义旗舰 ({version})"
    },
    {
        "id": "qwen_plus",
        "prefix": "qwen/",
        "include": ["plus"],
        "exclude": ["thinking", "free", "0", "1", "2025"],
        "label": "Qwen Plus 通义均衡 ({version})"
    },
    {
        "id": "llama_70b",
        "prefix": "meta-llama/",
        "include": ["70b", "instruct"],
        "exclude": ["nemotron", "chatqa", "free"],
        "label": "Llama 70B 开源旗舰 ({version})"
    },
    {
        "id": "kimi_standard",
        "prefix": "moonshotai/",
        "include": ["kimi"],
        "exclude": ["thinking", "free"],
        "label": "Kimi 核心模型 ({version})"
    },
    {
        "id": "kimi_thinking",
        "prefix": "moonshotai/",
        "include": ["kimi", "thinking"],
        "exclude": ["free"],
        "label": "Kimi K2 Thinking"
    },
    {
        "id": "glm_flagship",
        "prefix": "z-ai/",
        "include": ["glm"],
        "exclude": ["turbo", "flash", "air", "v", "free"],
        "label": "GLM 旗舰 ({version})"
    },
    {
        "id": "glm_economic",
        "prefix": "z-ai/",
        "include": ["glm"],
        "exclude": ["v", "free", "glm-5", "glm-4.7", "glm-4.6", "glm-4.5"], # exclude flagship and vision
        "label": "GLM 经济型 ({version})"
    },
    {
        "id": "minimax_standard",
        "prefix": "minimax/",
        "include": ["minimax"],
        "exclude": ["her", "free", "01"],
        "label": "MiniMax 核心模型 ({version})"
    }
]

def extract_version(model_id):
    # 更全面的字母数字替换规则
    replacements = {
        "4o": "4.0",
        "o1": "1.0",
        "o3": "3.0",
        "o4": "4.0",
    }
    model_id_clean = model_id.lower()
    for pat, repl in replacements.items():
        model_id_clean = model_id_clean.replace(pat, repl)
    
    # 仅提取模型名称主体中的版本部分，避免匹配到日期等无关数字
    # 例如 openai/gpt-4.1-20240314 -> 关注 "4.1"
    nums = re.findall(r'(\d+\.\d+|\d+)', model_id_clean.split('/')[-1])
    if not nums:
        return (0.0,)
    # 返回前两个数字作为主版本和次版本，避免日期干扰
    return tuple(float(n) for n in nums[:2])

def load_data_with_cache(url, retries=3, use_cache=True):
    """带本地缓存和重试机制的数据获取函数"""
    # 检查缓存是否存在且未过期
    if use_cache and os.path.exists(CACHE_FILE):
        file_mod_time = os.path.getmtime(CACHE_FILE)
        if time.time() - file_mod_time < CACHE_EXPIRY_SECONDS:
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"读取缓存失败: {e}，将重新获取数据。")
    
    # 缓存不存在或已过期，请求网络数据
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                # 保存到缓存
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return data
        except urllib.error.URLError as e:
            print(f"第 {attempt + 1} 次请求失败: {e.reason}")
            time.sleep(2 ** attempt)  # 指数退避重试
        except Exception as e:
            print(f"第 {attempt + 1} 次请求发生未知错误: {e}")
            time.sleep(2 ** attempt)
            
    print("获取数据最终失败。")
    return None

def find_latest_models(args):
    """从 OpenRouter 动态获取各厂商最新模型并输出定价"""
    url = "https://openrouter.ai/api/v1/models"
    data = load_data_with_cache(url, use_cache=not args.no_cache)
    
    if not data:
        return

    models = data.get("data", [])
    
    # 记录每个分类的最新模型
    latest_models = {cat["id"]: {"version": (-1.0,), "info": None, "cat": cat} for cat in CATEGORIES}

    for model_info in models:
        model_id = model_info.get("id", "")
        model_id_lower = model_id.lower()
        
        # 排除明确标记为 :free 的模型
        if ":free" in model_id_lower:
            continue

        for cat in CATEGORIES:
            # 检查前缀
            if not model_id_lower.startswith(cat["prefix"]):
                continue
                
            # 检查包含词
            if not all(inc in model_id_lower for inc in cat["include"]):
                continue
                
            # 检查排除词
            if any(exc in model_id_lower for exc in cat["exclude"]):
                continue
                
            # 提取版本号
            version_tuple = extract_version(model_id)
            
            # 比较版本号 (如果版本号相同，偏好 ID 更短的，即不带后缀的稳定版)
            current_best_version = latest_models[cat["id"]]["version"]
            current_best_id = latest_models[cat["id"]]["info"]["id"] if latest_models[cat["id"]]["info"] else ""
            
            if version_tuple > current_best_version:
                latest_models[cat["id"]]["version"] = version_tuple
                latest_models[cat["id"]]["info"] = model_info
            elif version_tuple == current_best_version and model_info:
                # 版本相同时，取长度更短的 ID，例如 gpt-4.1 > gpt-4.1-0314
                if len(model_id) < len(current_best_id) or not current_best_id:
                    latest_models[cat["id"]]["info"] = model_info

    # 打印结果
    print("### 表 1：API 基础定价表")
    print("| 模型名称 (**自动匹配最新版**) | API ID | 输入单价 (¥/M) | 缓存读 (¥/M) | 缓存写 (¥/M) | 输出单价 (¥/M) | 联网搜索 (¥/次) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    results_for_scenarios = []

    for cat in CATEGORIES:
        match = latest_models[cat["id"]]["info"]
        if match:
            target_id = match["id"]
            pricing = match.get("pricing", {})
            try:
                prompt_price_usd = float(pricing.get("prompt", 0))
                completion_price_usd = float(pricing.get("completion", 0))
                cache_read_price_usd = float(pricing.get("input_cache_read", 0))
                cache_write_price_usd = float(pricing.get("input_cache_write", 0))
                web_search_price_usd = float(pricing.get("web_search", 0))

                prompt_price_cny = prompt_price_usd * 1_000_000 * args.exchange_rate
                completion_price_cny = completion_price_usd * 1_000_000 * args.exchange_rate
                cache_read_price_cny = cache_read_price_usd * 1_000_000 * args.exchange_rate
                cache_write_price_cny = cache_write_price_usd * 1_000_000 * args.exchange_rate
                web_search_price_cny = web_search_price_usd * args.exchange_rate  # 联网搜索按次收费，不乘一百万

                # 提取展示的版本号，取版本元组的前几位合并
                v_tuple = latest_models[cat["id"]]["version"]
                # 简单格式化版本号用于展示
                v_str = ".".join(str(int(x)) if x.is_integer() else str(x) for x in v_tuple[:1])
                if len(v_tuple) > 1 and v_tuple[1] != 0:
                    v_str = f"{v_str}.{int(v_tuple[1]) if v_tuple[1].is_integer() else v_tuple[1]}"
                    
                display_name = cat["label"].format(version=f"v{v_str}" if v_str and v_str != "0" else "最新")

                # 格式化可选字段
                cache_read_str = f"{cache_read_price_cny:.4f}" if cache_read_price_usd > 0 else "-"
                cache_write_str = f"{cache_write_price_cny:.4f}" if cache_write_price_usd > 0 else "-"
                web_search_str = f"{web_search_price_cny:.4f}" if web_search_price_usd > 0 else "-"

                print(f"| **{display_name}** | `{target_id}` | {prompt_price_cny:.4f} | {cache_read_str} | {cache_write_str} | {completion_price_cny:.4f} | {web_search_str} |")
                
                results_for_scenarios.append({
                    "name": display_name,
                    "p_in": prompt_price_cny,
                    "p_out": completion_price_cny,
                    "p_cr": cache_read_price_cny,
                    "p_cw": cache_write_price_cny,
                    "p_ws": web_search_price_cny
                })
            except (ValueError, TypeError):
                print(f"| **{cat['label'].format(version='未知')}** | `{target_id}` | 数据解析异常 | - | - | 数据解析异常 | - |")
        else:
            print(f"| **{cat['label'].format(version='未找到')}** | `-` | 未收录/未找到 | - | - | 未收录/未找到 | - |")

    # 打印场景对标结果
    print("\n### 表 2：典型业务场景单次调用成本估算 (¥)")
    print("| 模型名称 | 场景A: RAG/长文档 (20K入/1K出/20%命中) | 场景B: 沉浸式Agent (30K入/1K出/80%命中) | 场景C: 批量翻译 (5K入/5K出/0%命中) |")
    print("| :--- | :--- | :--- | :--- |")
    
    def calc_cost(p_in, p_out, p_cr, p_cw, p_ws, tokens_in, tokens_out, hit_rate, ws_count=0):
        # 缓存回退逻辑
        # 如果 p_cr > 0，说明模型支持缓存；否则不支持
        # 如果不支持缓存，p_hit 和 p_miss 都回退为 p_in
        if p_cr > 0:
            p_hit = p_cr
            # 如果支持缓存但 p_cw 未定义（为0），通常按普通输入价收费
            p_miss = p_cw if p_cw > 0 else p_in
        else:
            p_hit = p_in
            p_miss = p_in
            
        mixed_in = (p_hit * hit_rate) + (p_miss * (1 - hit_rate))
        
        # tokens_in 是以 K 为单位传入的，因此要除以 1000 转换为 M (Millions)
        # 例如：20K = 20 / 1000 = 0.02M
        return (mixed_in * (tokens_in / 1000.0)) + (p_out * (tokens_out / 1000.0)) + (p_ws * ws_count)

    for r in results_for_scenarios:
        cost_a = calc_cost(r["p_in"], r["p_out"], r["p_cr"], r["p_cw"], r["p_ws"], 20.0, 1.0, 0.2)
        cost_b = calc_cost(r["p_in"], r["p_out"], r["p_cr"], r["p_cw"], r["p_ws"], 30.0, 1.0, 0.8)
        cost_c = calc_cost(r["p_in"], r["p_out"], r["p_cr"], r["p_cw"], r["p_ws"], 5.0, 5.0, 0.0)
        print(f"| **{r['name']}** | {cost_a:.4f} | {cost_b:.4f} | {cost_c:.4f} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="获取并分析大模型 API 定价")
    parser.add_argument("--exchange-rate", type=float, default=DEFAULT_EXCHANGE_RATE,
                        help=f"USD to CNY 汇率 (默认: {DEFAULT_EXCHANGE_RATE})")
    parser.add_argument("--no-cache", action="store_true", 
                        help="强制忽略本地缓存，从服务器拉取最新数据")
    
    args = parser.parse_args()
    find_latest_models(args)