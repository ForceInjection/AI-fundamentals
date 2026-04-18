import json
import os
import matplotlib.pyplot as plt
import numpy as np

def escape_md(text):
    if not text:
        return "-"
    return str(text).replace('\n', ' ').replace('|', '&#124;').replace('<br>', ' ').replace('<br/>', ' ')

def get_tier(plan_name):
    """根据套餐名称自动分类为：入门、高级、团队、专家"""
    p = str(plan_name).lower()
    if any(kw in p for kw in ["free", "lite", "hobby", "starter", "日常", "core", "体验"]):
        return "入门"
    if any(kw in p for kw in ["business", "teams", "进阶", "team"]):
        return "团队"
    if any(kw in p for kw in ["pro+", "max", "ultra", "enterprise", "全能", "专业", "旗舰"]):
        return "专家"
    if any(kw in p for kw in ["pro", "plus", "效率", "ultimate"]):
        return "高级"
    return "高级"

def main():
    base_dir = "/Users/wangtianqing/Project/study/AI-fundermentals/09_inference_system/cost_analysis/coding_plan"
    json_path = os.path.join(base_dir, "data/pricing_normalized.json")
    output_path = os.path.join(base_dir, "objective_pricing_comparison.md")
    
    intl_img_path = "assets/intl_pricing_chart.png"
    dom_img_path = "assets/dom_pricing_chart.png"
    
    os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intl = []
    dom = []

    for item in data:
        if item['currency'] == 'USD':
            intl.append(item)
        elif item['currency'] == 'CNY':
            dom.append(item)

    # Tables
    def make_table(items, currency_symbol):
        header = "| 厂商 | 产品 | 套餐 | 分类 | 月费 | 用量限制 | 限制条款 |\n| --- | --- | --- | --- | --- | --- | --- |\n"
        rows = []
        for i in items:
            price = f"{currency_symbol}{i['price_monthly']}" if i['price_monthly'] is not None else "按需 / 未公开"
            tier = get_tier(i['plan_name'])
            row = f"| {escape_md(i['vendor'])} | {escape_md(i['product'])} | {escape_md(i['plan_name'])} | **{tier}** | {price} | {escape_md(i['quota'])} | {escape_md(i['restrictions'])} |"
            rows.append(row)
        return header + "\n".join(rows)

    intl_table = make_table(intl, "$")
    dom_table = make_table(dom, "¥")

    # Grouped Bar Charts with Matplotlib
    def generate_grouped_chart(items, title, ylabel, filename):
        # Configure fonts for macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        vendor_groups = {}
        for i in items:
            if i['price_monthly'] is None or float(i['price_monthly']) <= 0:
                continue
            vendor = i['vendor']
            vendor = vendor.replace("GitHub Copilot", "GitHub").replace("Windsurf (Codeium)", "Windsurf")
            vendor = vendor.replace("阿里云百炼 Coding Plan", "阿里百炼").replace("腾讯云 Coding Plan", "腾讯云")
            vendor = vendor.replace("火山方舟 Coding Plan", "火山方舟").replace("智谱 GLM Coding Plan", "智谱GLM")
            vendor = vendor.replace("智谱龙虾套餐（团队协作版）", "智谱龙虾")
            vendor = vendor.replace("MiniMax Token Plan", "MiniMax").replace("Kimi Code Plan", "Kimi")
            vendor = vendor.replace("阿里云", "阿里百炼")
            
            tier = get_tier(i['plan_name'])
            if vendor not in vendor_groups:
                vendor_groups[vendor] = {}
            if tier not in vendor_groups[vendor]:
                vendor_groups[vendor][tier] = []
            vendor_groups[vendor][tier].append(float(i['price_monthly']))

        tiers = ["入门", "高级", "团队", "专家"]
        # 专业进阶配色（绿 -> 蓝），完全避开红/紫的“高昂/警告”心理暗示
        colors = ["#34D399", "#0EA5E9", "#2563EB", "#1E40AF"] 
        
        # Sort vendors by their minimum price across all tiers
        sorted_vendors = sorted(vendor_groups.keys(), key=lambda v: min([min(prices) for prices in vendor_groups[v].values()]))
        
        x = np.arange(len(sorted_vendors))
        # Total width for all bars is 0.8. We have 4 tiers, so width per bar = 0.2
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, tier in enumerate(tiers):
            tier_prices = []
            for v in sorted_vendors:
                if tier in vendor_groups[v]:
                    tier_prices.append(min(vendor_groups[v][tier])) # Pick lowest if multiple
                else:
                    tier_prices.append(0)
            
            if any(p > 0 for p in tier_prices):
                offset = (idx - 1.5) * width
                bars = ax.bar(x + offset, tier_prices, width, label=tier, color=colors[idx], alpha=0.9, edgecolor='white', linewidth=0.5)
                
                for bar in bars:
                    yval = bar.get_height()
                    if yval > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:g}", ha='center', va='bottom', fontsize=9, color='#333333')

        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='#333333')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#1a202c')
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_vendors, fontsize=11, color='#4a5568')
        
        ax.legend(title="套餐级别", title_fontsize='10', fontsize='10', loc='upper left', frameon=True)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.grid(axis='y', linestyle='--', alpha=0.6, color='#cbd5e0')
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, filename), dpi=200, bbox_inches='tight')
        plt.close()

    # Generate images
    generate_grouped_chart(intl, "海外 Coding Plan 月费分类对比 (按厂商分组)", "价格 (USD)", intl_img_path)
    generate_grouped_chart(dom, "国内 Coding Plan 月费分类对比 (按厂商分组)", "价格 (CNY)", dom_img_path)

    md_content = f"""# 附录：2026 年国内外主流 Coding Plan 真实数据看板

作为《每月花多少才不冤？2026 年国内外 11 款 Coding Plan 深度对比与避坑指南》的配套数据附录，本文档直接提取归一化后的厂商定价源数据，通过结构化表格与分组柱状图，直观呈现各家工具的费用阶梯与核心用量限制，为您提供纯粹的数据查阅参考。

## 1 海外套餐数据看板

### 1.1 海外套餐详情表

基于美元（USD）计费体系，覆盖 GitHub、Cursor、Windsurf 等海外主流服务商。

{intl_table}

### 1.2 海外套餐月费阶梯图
  
海外工具的定价普遍以 $10–$20 作为个人开发者的标准入场门槛，而包含高阶推理模型与无限制并发特性的专家版（如 Cursor Ultra、Anthropic Max）则迅速拉升至 $100–$200 的高位区间，形成显著的消费分层。
  
![海外 Coding Plan 月费分类对比](./{intl_img_path})

## 2 国内套餐数据看板

### 2.1 国内套餐详情表

基于人民币（CNY）计费体系，重点披露国内厂商在流量控制（如 5 小时限流窗口）与 API 禁用协议方面的硬性约束条款。

{dom_table}

### 2.2 国内套餐月费阶梯图
  
国内市场的竞争使其入门定价高度收敛于 ¥30–¥50 区间，而面向高阶研发的专业版则跃升至 ¥100–¥300 的价格带，此阶梯差价本质上是开发者为突破“5 小时限流窗口”与获取大额度 API 配额所支付的溢价。
  
![国内 Coding Plan 月费分类对比](./{dom_img_path})
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Successfully wrote independent report to {output_path} and generated charts in assets/")

if __name__ == "__main__":
    main()
